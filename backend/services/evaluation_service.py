import math
import os
import pandas as pd

from services.retrieval_service import run_search
from services.personalization_service import personalize_results
from services.topic_detection_service import detect_topic_from_results



def evaluate_ranked_results(results, relevant_doc_ids, top_k=10):
    relevant_doc_ids = set(str(doc_id) for doc_id in relevant_doc_ids)

    if len(relevant_doc_ids) == 0:
        return {
            "Precision@10": 0,
            "Recall": 0,
            "Average_Precision": 0,
            "nDCG": 0,
            "relevant_retrieved_count": 0,
            "relevant_total_count": 0
        }

    retrieved_doc_ids = [
        str(item["doc_id"])
        for item in results[:top_k]
    ]

    relevant_retrieved = [
        doc_id
        for doc_id in retrieved_doc_ids
        if doc_id in relevant_doc_ids
    ]

    precision = len(relevant_retrieved) / top_k
    recall = len(relevant_retrieved) / len(relevant_doc_ids)

    average_precision = average_precision_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=top_k
    )

    ndcg = ndcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=top_k
    )

    return {
        "Precision@10": precision,
        "Recall": recall,
        "Average_Precision": average_precision,
        "nDCG": ndcg,
        "relevant_retrieved_count": len(relevant_retrieved),
        "relevant_total_count": len(relevant_doc_ids)
    }


def precision_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    retrieved_at_k = retrieved_doc_ids[:k]

    if k == 0:
        return 0

    relevant_count = sum(
        1
        for doc_id in retrieved_at_k
        if str(doc_id) in relevant_doc_ids
    )

    return relevant_count / k


def recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    if len(relevant_doc_ids) == 0:
        return 0

    retrieved_at_k = retrieved_doc_ids[:k]

    relevant_count = sum(
        1
        for doc_id in retrieved_at_k
        if str(doc_id) in relevant_doc_ids
    )

    return relevant_count / len(relevant_doc_ids)


def average_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    score = 0
    relevant_found = 0

    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if str(doc_id) in relevant_doc_ids:
            relevant_found += 1
            score += relevant_found / index

    if len(relevant_doc_ids) == 0:
        return 0

    return score / min(len(relevant_doc_ids), k)


def dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    dcg = 0

    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = 1 if str(doc_id) in relevant_doc_ids else 0

        if index == 1:
            dcg += relevance
        else:
            dcg += relevance / math.log2(index + 1)

    return dcg


def ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    dcg = dcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=k
    )

    ideal_relevant_count = min(len(relevant_doc_ids), k)
    ideal_dcg = 0

    for index in range(1, ideal_relevant_count + 1):
        relevance = 1

        if index == 1:
            ideal_dcg += relevance
        else:
            ideal_dcg += relevance / math.log2(index + 1)

    if ideal_dcg == 0:
        return 0

    return dcg / ideal_dcg


def load_evaluation_files(dataset, save_dir="saved_files"):
    if dataset == "dataset1":
        queries_file = "queries_dataset1.csv"
        qrels_file = "qrels_dataset1.csv"
    elif dataset == "dataset2":
        queries_file = "queries_dataset2.csv"
        qrels_file = "qrels_dataset2.csv"
    else:
        raise ValueError("dataset must be dataset1 or dataset2")

    queries_path = os.path.join(save_dir, queries_file)
    qrels_path = os.path.join(save_dir, qrels_file)

    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Missing evaluation queries file: {queries_path}")

    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Missing evaluation qrels file: {qrels_path}")

    queries_df = pd.read_csv(queries_path)
    qrels_df = pd.read_csv(qrels_path)

    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_df["text"] = queries_df["text"].astype(str)

    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    if "relevance" not in qrels_df.columns:
        qrels_df["relevance"] = 1

    qrels_df = qrels_df[qrels_df["relevance"] > 0]

    return queries_df, qrels_df


def load_indexed_documents(dataset, save_dir="saved_files"):
    if dataset == "dataset1":
        docs_file = "work_dataset1.csv"
    elif dataset == "dataset2":
        docs_file = "work_dataset2.csv"
    else:
        raise ValueError("dataset must be dataset1 or dataset2")

    docs_path = os.path.join(save_dir, docs_file)

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing indexed documents file: {docs_path}")

    docs_df = pd.read_csv(docs_path)
    docs_df["doc_id"] = docs_df["doc_id"].astype(str)

    return docs_df


def get_available_doc_ids(dataset, save_dir="saved_files"):
    docs_df = load_indexed_documents(
        dataset=dataset,
        save_dir=save_dir
    )

    return set(
        str(doc_id)
        for doc_id in docs_df["doc_id"].tolist()
    )


def filter_qrels_to_available_docs(qrels_df, available_doc_ids):
    return qrels_df[
        qrels_df["doc_id"].astype(str).isin(available_doc_ids)
    ].copy()


def get_valid_queries(queries_df, filtered_qrels_df, max_queries):
    valid_query_ids = filtered_qrels_df["query_id"].unique().tolist()

    valid_queries_df = queries_df[
        queries_df["query_id"].isin(valid_query_ids)
    ].copy()

    return valid_queries_df.head(max_queries)


def summarize_metric_difference(before_result, after_result):
    return {
        "MAP_difference": after_result["MAP"] - before_result["MAP"],
        "Recall_difference": after_result["Recall"] - before_result["Recall"],
        "Precision@10_difference": after_result["Precision@10"] - before_result["Precision@10"],
        "nDCG_difference": after_result["nDCG"] - before_result["nDCG"]
    }



def tokenize_text(text):
    import re

    return set(
        re.findall(
            r"[a-zA-Z]{2,}",
            str(text).lower()
        )
    )


def topic_rerank_results(results, topic_boost_factor=0.08):
    topic_info = detect_topic_from_results(
        results=results,
        max_terms=8
    )

    if not topic_info.get("topic_detection_applied", False):
        return {
            "topic_detection_applied": False,
            "topic_info": topic_info,
            "results": results
        }

    topic_terms = topic_info.get("top_topic_terms", [])
    reranked_results = []

    for item in results:
        new_item = item.copy()

        text_tokens = tokenize_text(
            new_item.get("text", "")
        )

        matched_topic_terms = []
        topic_score = 0

        for term_info in topic_terms:
            term = str(term_info.get("term", "")).lower()
            score = float(term_info.get("score", 0))

            if term in text_tokens:
                matched_topic_terms.append(term)
                topic_score += score

        original_score = float(new_item.get("score", 0))

        if topic_score > 0:
            adjusted_score = original_score * (1 + topic_boost_factor * topic_score)
        else:
            adjusted_score = original_score

        new_item["original_score"] = original_score
        new_item["score"] = adjusted_score
        new_item["topic_score"] = topic_score
        new_item["matched_topic_terms"] = matched_topic_terms

        reranked_results.append(new_item)

    reranked_results = sorted(
        reranked_results,
        key=lambda item: item["score"],
        reverse=True
    )

    for index, item in enumerate(reranked_results, start=1):
        item["rank"] = index

    return {
        "topic_detection_applied": True,
        "topic_info": topic_info,
        "results": reranked_results
    }


def evaluate_single_method(
    dataset,
    method,
    loaded_data,
    queries_df,
    filtered_qrels_df,
    top_k=10,
    k1=1.5,
    b=0.75,
    alpha=0.6,
    use_personalization=False,
    search_history=None,
    use_topic_detection=False,
    candidate_pool_size=None
):
    average_precisions = []
    recalls = []
    precisions_at_10 = []
    ndcgs = []

    evaluated_queries_count = 0

    if candidate_pool_size is None:
        candidate_pool_size = top_k

    if candidate_pool_size < top_k:
        candidate_pool_size = top_k

    for _, query_row in queries_df.iterrows():
        query_id = str(query_row["query_id"])
        query_text = str(query_row["text"])

        relevant_doc_ids = filtered_qrels_df[
            filtered_qrels_df["query_id"] == query_id
        ]["doc_id"].astype(str).tolist()

        if len(relevant_doc_ids) == 0:
            continue

        # First-stage retrieval:
        # Retrieve a larger candidate pool when personalization is enabled.
        # Then personalization re-ranks these candidates.
        results = run_search(
            query=query_text,
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            top_k=candidate_pool_size,
            k1=k1,
            b=b,
            alpha=alpha
        )

        if use_personalization:
            personalization_output = personalize_results(
                results=results,
                search_history=search_history
            )
            results = personalization_output["results"]

        if use_topic_detection:
            topic_output = topic_rerank_results(
                results=results,
                topic_boost_factor=0.08
            )
            results = topic_output["results"]

        # Final evaluation is always on the final Top K.
        final_results = results[:top_k]

        retrieved_doc_ids = [
            str(item["doc_id"])
            for item in final_results
        ]

        relevant_doc_ids_set = set(relevant_doc_ids)

        ap = average_precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids_set,
            k=top_k
        )

        recall = recall_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids_set,
            k=top_k
        )

        precision = precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids_set,
            k=10
        )

        ndcg = ndcg_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids_set,
            k=top_k
        )

        average_precisions.append(ap)
        recalls.append(recall)
        precisions_at_10.append(precision)
        ndcgs.append(ndcg)

        evaluated_queries_count += 1

    if evaluated_queries_count == 0:
        return {
            "dataset": dataset,
            "method": method,
            "use_personalization": use_personalization,
            "use_topic_detection": use_topic_detection,
            "evaluated_queries_count": 0,
            "candidate_pool_size": candidate_pool_size,
            "final_top_k": top_k,
            "MAP": 0,
            "Recall": 0,
            "Precision@10": 0,
            "nDCG": 0
        }

    result = {
        "dataset": dataset,
        "method": method,
        "use_personalization": use_personalization,
        "use_topic_detection": use_topic_detection,
        "evaluated_queries_count": evaluated_queries_count,
        "candidate_pool_size": candidate_pool_size,
        "final_top_k": top_k,
        "MAP": sum(average_precisions) / evaluated_queries_count,
        "Recall": sum(recalls) / evaluated_queries_count,
        "Precision@10": sum(precisions_at_10) / evaluated_queries_count,
        "nDCG": sum(ndcgs) / evaluated_queries_count
    }

    if use_personalization:
        result["personalization_history_used"] = search_history

    return result


def evaluate_all_methods(
    dataset,
    methods,
    loaded_data,
    top_k=10,
    max_queries=10,
    k1=1.5,
    b=0.75,
    alpha=0.6,
    save_dir="saved_files"
):
    queries_df, qrels_df = load_evaluation_files(
        dataset=dataset,
        save_dir=save_dir
    )

    available_doc_ids = get_available_doc_ids(
        dataset=dataset,
        save_dir=save_dir
    )

    filtered_qrels_df = filter_qrels_to_available_docs(
        qrels_df=qrels_df,
        available_doc_ids=available_doc_ids
    )

    valid_queries_df = get_valid_queries(
        queries_df=queries_df,
        filtered_qrels_df=filtered_qrels_df,
        max_queries=max_queries
    )

    summary = []

    for method in methods:
        method_result = evaluate_single_method(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            use_personalization=False,
            search_history=[],
            use_topic_detection=False,
            candidate_pool_size=top_k
        )

        method_result["available_documents_count"] = len(available_doc_ids)
        method_result["available_qrels_count"] = len(filtered_qrels_df)

        summary.append(method_result)

    return {
        "dataset": dataset,
        "top_k": top_k,
        "max_queries": max_queries,
        "evaluation_mode": "before_additional_features",
        "summary": summary
    }


def evaluate_all_methods_with_personalization(
    dataset,
    methods,
    loaded_data,
    top_k=10,
    max_queries=10,
    k1=1.5,
    b=0.75,
    alpha=0.6,
    search_history=None,
    save_dir="saved_files"
):
    if search_history is None:
        search_history = []

    queries_df, qrels_df = load_evaluation_files(
        dataset=dataset,
        save_dir=save_dir
    )

    available_doc_ids = get_available_doc_ids(
        dataset=dataset,
        save_dir=save_dir
    )

    filtered_qrels_df = filter_qrels_to_available_docs(
        qrels_df=qrels_df,
        available_doc_ids=available_doc_ids
    )

    valid_queries_df = get_valid_queries(
        queries_df=queries_df,
        filtered_qrels_df=filtered_qrels_df,
        max_queries=max_queries
    )

    before_additional_feature = []
    after_personalization = []
    metric_differences = []

    # Baseline evaluates the normal final Top K.
    baseline_candidate_pool_size = top_k

    # Personalization evaluates a practical two-stage retrieval pipeline:
    # retrieve a larger candidate pool, personalize/re-rank, then evaluate final Top K.
    personalization_candidate_pool_size = max(top_k * 5, 50)

    available_docs_count = len(available_doc_ids)

    if personalization_candidate_pool_size > available_docs_count:
        personalization_candidate_pool_size = available_docs_count

    for method in methods:
        baseline_result = evaluate_single_method(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            use_personalization=False,
            search_history=[],
            use_topic_detection=False,
            candidate_pool_size=baseline_candidate_pool_size
        )

        personalized_result = evaluate_single_method(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            use_personalization=True,
            search_history=search_history,
            use_topic_detection=False,
            candidate_pool_size=personalization_candidate_pool_size
        )

        baseline_result["available_documents_count"] = len(available_doc_ids)
        baseline_result["available_qrels_count"] = len(filtered_qrels_df)

        personalized_result["available_documents_count"] = len(available_doc_ids)
        personalized_result["available_qrels_count"] = len(filtered_qrels_df)

        difference = summarize_metric_difference(
            before_result=baseline_result,
            after_result=personalized_result
        )

        difference["dataset"] = dataset
        difference["method"] = method

        before_additional_feature.append(baseline_result)
        after_personalization.append(personalized_result)
        metric_differences.append(difference)

    return {
        "dataset": dataset,
        "top_k": top_k,
        "max_queries": max_queries,
        "evaluation_mode": "before_and_after_personalization",
        "evaluation_strategy": "Baseline uses Top K directly. Personalization retrieves a larger candidate pool, re-ranks it using user history, then evaluates the final Top K.",
        "baseline_candidate_pool_size": baseline_candidate_pool_size,
        "personalization_candidate_pool_size": personalization_candidate_pool_size,
        "personalization_history_used": search_history,
        "before_additional_feature": before_additional_feature,
        "after_personalization": after_personalization,
        "metric_differences": metric_differences
    }



def evaluate_all_methods_with_topic_detection(
    dataset,
    methods,
    loaded_data,
    top_k=10,
    max_queries=10,
    k1=1.5,
    b=0.75,
    alpha=0.6,
    save_dir="saved_files"
):
    queries_df, qrels_df = load_evaluation_files(
        dataset=dataset,
        save_dir=save_dir
    )

    available_doc_ids = get_available_doc_ids(
        dataset=dataset,
        save_dir=save_dir
    )

    filtered_qrels_df = filter_qrels_to_available_docs(
        qrels_df=qrels_df,
        available_doc_ids=available_doc_ids
    )

    valid_queries_df = get_valid_queries(
        queries_df=queries_df,
        filtered_qrels_df=filtered_qrels_df,
        max_queries=max_queries
    )

    before_additional_feature = []
    after_topic_detection = []
    metric_differences = []

    baseline_candidate_pool_size = top_k
    topic_candidate_pool_size = max(top_k * 5, 50)

    if topic_candidate_pool_size > len(available_doc_ids):
        topic_candidate_pool_size = len(available_doc_ids)

    for method in methods:
        baseline_result = evaluate_single_method(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            use_personalization=False,
            search_history=[],
            use_topic_detection=False,
            candidate_pool_size=baseline_candidate_pool_size
        )

        topic_result = evaluate_single_method(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            use_personalization=False,
            search_history=[],
            use_topic_detection=True,
            candidate_pool_size=topic_candidate_pool_size
        )

        baseline_result["available_documents_count"] = len(available_doc_ids)
        baseline_result["available_qrels_count"] = len(filtered_qrels_df)

        topic_result["available_documents_count"] = len(available_doc_ids)
        topic_result["available_qrels_count"] = len(filtered_qrels_df)

        difference = summarize_metric_difference(
            before_result=baseline_result,
            after_result=topic_result
        )

        difference["dataset"] = dataset
        difference["method"] = method

        before_additional_feature.append(baseline_result)
        after_topic_detection.append(topic_result)
        metric_differences.append(difference)

    return {
        "dataset": dataset,
        "top_k": top_k,
        "max_queries": max_queries,
        "evaluation_mode": "before_and_after_topic_detection",
        "evaluation_strategy": "Baseline uses Top K directly. Topic Detection retrieves a larger candidate pool, detects dominant topic terms using TF-IDF, re-ranks documents using topic-term matches, then evaluates the final Top K.",
        "baseline_candidate_pool_size": baseline_candidate_pool_size,
        "topic_candidate_pool_size": topic_candidate_pool_size,
        "before_additional_feature": before_additional_feature,
        "after_topic_detection": after_topic_detection,
        "metric_differences": metric_differences
    }
