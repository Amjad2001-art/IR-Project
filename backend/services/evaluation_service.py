import math
import os
import statistics
import time
import pandas as pd

from services.retrieval_service import run_search
from services.topic_detection_service import detect_topic_from_results
from services.query_refinement_service import refine_query



def evaluate_ranked_results(results, relevant_doc_ids, top_k=10):
    # Evaluation Metrics
    # هنا نحسب مقاييس جودة الاسترجاع المطلوبة رسمياً.
    # نحول وثائق الصلة إلى نصوص حتى تتطابق مع معرفات النتائج.
    relevant_doc_ids = set(str(doc_id) for doc_id in relevant_doc_ids)

    # إذا لم توجد وثائق صلة فلا يمكن حساب قيمة إيجابية للمقاييس.
    if len(relevant_doc_ids) == 0:
        return {
            "Precision@10": 0,
            "Recall": 0,
            "Average_Precision": 0,
            "nDCG": 0,
            "relevant_retrieved_count": 0,
            "relevant_total_count": 0
        }

    # نأخذ معرفات الوثائق المسترجعة ضمن أول النتائج.
    retrieved_doc_ids = [
        str(item["doc_id"])
        for item in results[:top_k]
    ]

    # Qrels
    # نحدد الوثائق المسترجعة التي تعتبر صحيحة حسب أحكام الصلة.
    relevant_retrieved = [
        doc_id
        for doc_id in retrieved_doc_ids
        if doc_id in relevant_doc_ids
    ]

    # نحسب نسبة الصحيح من أول النتائج.
    precision = len(relevant_retrieved) / top_k
    # نحسب نسبة ما تم استرجاعه من كل الوثائق الصحيحة.
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
    # K
    # نأخذ أول عدد محدد من النتائج فقط.
    retrieved_at_k = retrieved_doc_ids[:k]

    # K
    # إذا كان العدد يساوي صفراً نعيد صفراً لتجنب القسمة على صفر.
    if k == 0:
        return 0

    # K
    # نعد الوثائق الصحيحة الموجودة ضمن أول عدد محدد من النتائج.
    relevant_count = sum(
        1
        for doc_id in retrieved_at_k
        if str(doc_id) in relevant_doc_ids
    )

    return relevant_count / k


def recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    # Qrels
    # Recall
    # إذا لم توجد وثائق صحيحة في أحكام الصلة فلا يمكن حساب المقياس.
    if len(relevant_doc_ids) == 0:
        return 0

    # K
    # نأخذ أول عدد محدد من النتائج فقط.
    retrieved_at_k = retrieved_doc_ids[:k]

    # K
    # نعد الوثائق الصحيحة التي ظهرت ضمن أول عدد محدد من النتائج.
    relevant_count = sum(
        1
        for doc_id in retrieved_at_k
        if str(doc_id) in relevant_doc_ids
    )

    return relevant_count / len(relevant_doc_ids)


def average_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    # MAP
    # هنا نحسب الدقة المتوسطة لكل استعلام لاستخدامها لاحقاً في المتوسط العام.
    # هذا المتغير يجمع الدقة عند كل موضع ظهرت فيه وثيقة صحيحة.
    score = 0
    # هذا المتغير يعد الوثائق الصحيحة التي ظهرت حتى الموضع الحالي.
    relevant_found = 0

    # K
    # نمر على أول عدد محدد من النتائج ونحسب الدقة عند كل نتيجة صحيحة.
    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if str(doc_id) in relevant_doc_ids:
            relevant_found += 1
            score += relevant_found / index

    if len(relevant_doc_ids) == 0:
        return 0

    return score / min(len(relevant_doc_ids), k)


def dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    # DCG
    # نبدأ قيمة المقياس من الصفر.
    dcg = 0

    # نمر على النتائج ونخفض وزن الوثائق الصحيحة كلما تأخر ترتيبها.
    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = 1 if str(doc_id) in relevant_doc_ids else 0

        if index == 1:
            dcg += relevance
        else:
            dcg += relevance / math.log2(index + 1)

    return dcg


def ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k=10):
    # nDCG
    # هنا نقيس جودة ترتيب النتائج وليس فقط عدد الوثائق الصحيحة.
    # DCG
    # نحسب قيمة الترتيب الفعلي.
    dcg = dcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=k
    )

    # نحدد أفضل ترتيب ممكن نظرياً للوثائق الصحيحة.
    ideal_relevant_count = min(len(relevant_doc_ids), k)
    ideal_dcg = 0

    # Ideal DCG
    # نحسب القيمة المثالية لاستخدامها في التطبيع.
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
    # Test Queries And Qrels
    # هنا نقرأ استعلامات الاختبار وأحكام الصلة الرسمية الخاصة ب داتا سيت.
    # نختار أسماء ملفات التقييم حسب الداتا سيت المطلوبة.
    if dataset != "dataset1":
        raise ValueError("dataset must be dataset1 (WikIR)")

    queries_file = "queries_dataset1.csv"
    qrels_file = "qrels_dataset1.csv"

    # نبني المسار الكامل لملفات الاستعلامات وأحكام الصلة.
    queries_path = os.path.join(save_dir, queries_file)
    qrels_path = os.path.join(save_dir, qrels_file)

    # نتحقق أن ملف الاستعلامات موجود.
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Missing evaluation queries file: {queries_path}")

    # نتحقق أن ملف أحكام الصلة موجود.
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Missing evaluation qrels file: {qrels_path}")

    # نقرأ الملفات الرسمية من الكاش المحلي.
    queries_df = pd.read_csv(queries_path)
    qrels_df = pd.read_csv(qrels_path)

    # نوحد نوع المعرفات حتى تتم المطابقة بشكل صحيح.
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_df["text"] = queries_df["text"].astype(str)

    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    # Relevance
    # إذا لم يوجد عمود الصلة نعتبر كل الصفوف ذات صلة.
    if "relevance" not in qrels_df.columns:
        qrels_df["relevance"] = 1

    # نحتفظ فقط بالوثائق ذات الصلة الإيجابية.
    qrels_df = qrels_df[qrels_df["relevance"] > 0]

    return queries_df, qrels_df


def load_indexed_documents(dataset, save_dir="saved_files"):
    if dataset != "dataset1":
        raise ValueError("dataset must be dataset1 (WikIR)")

    docs_file = "work_dataset1.csv"

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

    if max_queries is None or int(max_queries) <= 0:
        return valid_queries_df

    return valid_queries_df.head(max_queries)


def summarize_metric_difference(before_result, after_result):
    # Before And After Comparison
    # هنا نحسب فرق المقاييس قبل تطبيق الميزة الإضافية وبعدها.
    return {
        "MAP_difference": after_result["MAP"] - before_result["MAP"],
        "Recall_difference": after_result["Recall"] - before_result["Recall"],
        "Precision@10_difference": after_result["Precision@10"] - before_result["Precision@10"],
        "nDCG_difference": after_result["nDCG"] - before_result["nDCG"],
        "average_retrieval_time_difference_seconds": (
            after_result["average_retrieval_time_seconds"]
            - before_result["average_retrieval_time_seconds"]
        ),
        "average_total_time_difference_seconds": (
            after_result["average_total_time_seconds"]
            - before_result["average_total_time_seconds"]
        )
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
    # Topic-Based Re-Ranking
    # هنا نعيد ترتيب النتائج بإضافة تعزيز بسيط للوثائق المطابقة لكلمات الموضوع.
    # نستخرج كلمات الموضوع من النتائج المرشحة.
    topic_info = detect_topic_from_results(
        results=results,
        max_terms=8
    )

    # إذا لم تنجح عملية كشف الموضوع نعيد النتائج كما هي.
    if not topic_info.get("topic_detection_applied", False):
        return {
            "topic_detection_applied": False,
            "topic_info": topic_info,
            "results": results
        }

    # نأخذ أهم كلمات الموضوع مع أوزانها.
    topic_terms = topic_info.get("top_topic_terms", [])
    # نخزن هنا النتائج بعد تعديل درجاتها.
    reranked_results = []

    # نمر على كل نتيجة مرشحة.
    for item in results:
        # ننسخ النتيجة حتى لا نعدل العنصر الأصلي مباشرة.
        new_item = item.copy()

        # نحول نص الوثيقة إلى كلمات للمطابقة مع كلمات الموضوع.
        text_tokens = tokenize_text(
            new_item.get("text", "")
        )

        # نخزن الكلمات الموضوعية التي ظهرت داخل الوثيقة.
        matched_topic_terms = []
        # نجمع وزن الكلمات الموضوعية المطابقة داخل الوثيقة.
        topic_score = 0

        # نمر على كل كلمة موضوعية مستخرجة.
        for term_info in topic_terms:
            term = str(term_info.get("term", "")).lower()
            score = float(term_info.get("score", 0))

            # إذا ظهرت الكلمة الموضوعية في الوثيقة نضيف وزنها.
            if term in text_tokens:
                matched_topic_terms.append(term)
                topic_score += score

        # الدرجة الأصلية هي درجة طريقة الاسترجاع قبل الميزة.
        original_score = float(new_item.get("score", 0))

        # إذا كانت الوثيقة مطابقة للموضوع نرفع درجتها بنسبة محسوبة.
        if topic_score > 0:
            adjusted_score = original_score * (1 + topic_boost_factor * topic_score)
        else:
            adjusted_score = original_score

        # نخزن الدرجة الأصلية والدرجة الجديدة ومعلومات الموضوع للشرح.
        new_item["original_score"] = original_score
        new_item["score"] = adjusted_score
        new_item["topic_score"] = topic_score
        new_item["matched_topic_terms"] = matched_topic_terms

        reranked_results.append(new_item)

    # نرتب النتائج من جديد حسب الدرجة المعدلة.
    reranked_results = sorted(
        reranked_results,
        key=lambda item: item["score"],
        reverse=True
    )

    # نعيد ترقيم النتائج بعد إعادة الترتيب.
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
    use_topic_detection=False,
    candidate_pool_size=None
):
# Single Method Evaluation
# هنا نقيم طريقة استرجاع واحدة على استعلامات الاختبار الرسمية.
    # قوائم لتجميع قيم المقاييس لكل الاستعلامات.
    average_precisions = []
    recalls = []
    precisions_at_10 = []
    ndcgs = []
    retrieval_times = []
    total_times = []

    # نعد الاستعلامات التي تم تقييمها فعلياً.
    evaluated_queries_count = 0

    # Top K
    # إذا لم يحدد حجم المرشحين نستخدم نفس عدد النتائج النهائي.
    if candidate_pool_size is None:
        candidate_pool_size = top_k

    # نضمن أن حجم المرشحين لا يقل عن عدد النتائج النهائي.
    if candidate_pool_size < top_k:
        candidate_pool_size = top_k

    # نمر على استعلامات الاختبار الرسمية.
    for _, query_row in queries_df.iterrows():
        # نبدأ قياس الزمن الكلي لهذا الاستعلام.
        query_start = time.perf_counter()
        query_id = str(query_row["query_id"])
        query_text = str(query_row["text"])
        # Query Refinement
        # هنا نفعل تحسين الاستعلام في التقييم لأنه طلب أساسي وليس ميزة إضافية.
        refined_query = refine_query(
            query=query_text,
            dataset=dataset,
        )["refined_query"]

        # Qrels
        # نستخرج وثائق الصلة الخاصة بهذا الاستعلام من أحكام الصلة.
        relevant_doc_ids = filtered_qrels_df[
            filtered_qrels_df["query_id"] == query_id
        ]["doc_id"].astype(str).tolist()

        # إذا لم يوجد لهذا الاستعلام وثائق صلة نتجاوزه.
        if len(relevant_doc_ids) == 0:
            continue

        # نقيس زمن الاسترجاع فقط.
        retrieval_start = time.perf_counter()
        results = run_search(
            query=refined_query,
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            top_k=candidate_pool_size,
            k1=k1,
            b=b,
            alpha=alpha
        )
        retrieval_times.append(time.perf_counter() - retrieval_start)

        # إذا طلبنا كشف الموضوع نعيد ترتيب النتائج قبل حساب المقاييس.
        if use_topic_detection:
            topic_output = topic_rerank_results(
                results=results,
                topic_boost_factor=0.08
            )
            results = topic_output["results"]

        # Final Top K
        # التقييم النهائي دائماً على أول النتائج المطلوبة فقط.
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
        total_times.append(time.perf_counter() - query_start)

        evaluated_queries_count += 1

    if evaluated_queries_count == 0:
        return {
            "dataset": dataset,
            "method": method,
            "use_topic_detection": use_topic_detection,
            "evaluated_queries_count": 0,
            "candidate_pool_size": candidate_pool_size,
            "final_top_k": top_k,
            "MAP": 0,
            "Recall": 0,
            "Precision@10": 0,
            "nDCG": 0,
            "average_retrieval_time_seconds": 0,
            "average_total_time_seconds": 0,
            "total_execution_time_seconds": 0
        }

    result = {
        "dataset": dataset,
        "method": method,
        "use_topic_detection": use_topic_detection,
        "evaluated_queries_count": evaluated_queries_count,
        "candidate_pool_size": candidate_pool_size,
        "final_top_k": top_k,
        "MAP": sum(average_precisions) / evaluated_queries_count,
        "Recall": sum(recalls) / evaluated_queries_count,
        "Precision@10": sum(precisions_at_10) / evaluated_queries_count,
        "nDCG": sum(ndcgs) / evaluated_queries_count,
        "average_retrieval_time_seconds": sum(retrieval_times) / evaluated_queries_count,
        "average_total_time_seconds": sum(total_times) / evaluated_queries_count,
        "total_execution_time_seconds": sum(total_times)
    }

    return result


def _metrics_for_results(results, relevant_doc_ids, top_k):
    retrieved_doc_ids = [
        str(item["doc_id"])
        for item in results[:top_k]
    ]
    relevant_doc_ids = set(str(doc_id) for doc_id in relevant_doc_ids)

    return {
        "Average_Precision": average_precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=top_k
        ),
        "Recall": recall_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=top_k
        ),
        "Precision@10": precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=10
        ),
        "nDCG": ndcg_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=top_k
        )
    }


def evaluate_single_method_with_topic_cost(
    dataset,
    method,
    loaded_data,
    queries_df,
    filtered_qrels_df,
    top_k=10,
    k1=1.5,
    b=0.75,
    alpha=0.6,
    candidate_pool_size=50,
    retrieval_repetitions=3,
    feature_repetitions=10
):
# Feature Cost
# هنا نقيس أثر كشف الموضوع على الجودة والزمن باستخدام نفس مجموعة المرشحين.
    # نخزن مقاييس الجودة قبل تشغيل الميزة.
    before_metrics = []
    # نخزن مقاييس الجودة بعد تشغيل الميزة.
    after_metrics = []
    # نخزن أزمنة الاسترجاع بعد التسخين.
    retrieval_times = []
    # نخزن كلفة ميزة كشف الموضوع بشكل منفصل.
    feature_costs = []

    # إذا لم توجد استعلامات فلا يوجد تقييم.
    if len(queries_df) == 0:
        return None

    # نستخدم أول استعلام لتسخين النموذج والكاش قبل قياس الزمن.
    first_query = str(queries_df.iloc[0]["text"])
    first_refined_query = refine_query(
        query=first_query,
        dataset=dataset,
    )["refined_query"]
    run_search(
        query=first_refined_query,
        dataset=dataset,
        method=method,
        loaded_data=loaded_data,
        top_k=candidate_pool_size,
        k1=k1,
        b=b,
        alpha=alpha
    )

    # نمر على استعلامات الاختبار الرسمية.
    for _, query_row in queries_df.iterrows():
        query_id = str(query_row["query_id"])
        query_text = str(query_row["text"])
        # Query Refinement
        # نفس الاستعلام المحسن يستخدم قبل وبعد كشف الموضوع لتبقى المقارنة عادلة.
        refined_query = refine_query(
            query=query_text,
            dataset=dataset,
        )["refined_query"]
        # نقرأ وثائق الصلة الخاصة بالاستعلام الحالي.
        relevant_doc_ids = filtered_qrels_df[
            filtered_qrels_df["query_id"] == query_id
        ]["doc_id"].astype(str).tolist()

        # إذا لا توجد وثائق صلة نتجاوز هذا الاستعلام.
        if not relevant_doc_ids:
            continue

        # نجمع عدة قياسات للاسترجاع ثم نأخذ الوسيط لتقليل ضجيج الزمن.
        retrieval_samples = []
        candidate_results = None
        for _ in range(retrieval_repetitions):
            retrieval_start = time.perf_counter()
            candidate_results = run_search(
                query=refined_query,
                dataset=dataset,
                method=method,
                loaded_data=loaded_data,
                top_k=candidate_pool_size,
                k1=k1,
                b=b,
                alpha=alpha
            )
            retrieval_samples.append(
                time.perf_counter() - retrieval_start
            )

        # نحسب المقاييس قبل تطبيق كشف الموضوع.
        before_metrics.append(
            _metrics_for_results(
                results=candidate_results,
                relevant_doc_ids=relevant_doc_ids,
                top_k=top_k
            )
        )

        # تشغيل أولي للميزة قبل القياس لتجنب أثر التحميل الأول.
        topic_rerank_results(
            results=candidate_results,
            topic_boost_factor=0.08
        )

        # نقيس كلفة الميزة وحدها عدة مرات على نفس النتائج المرشحة.
        feature_samples = []
        topic_results = candidate_results
        for _ in range(feature_repetitions):
            feature_start = time.perf_counter()
            topic_output = topic_rerank_results(
                results=candidate_results,
                topic_boost_factor=0.08
            )
            feature_samples.append(
                time.perf_counter() - feature_start
            )
            topic_results = topic_output["results"]

        # نحسب المقاييس بعد تطبيق كشف الموضوع.
        after_metrics.append(
            _metrics_for_results(
                results=topic_results,
                relevant_doc_ids=relevant_doc_ids,
                top_k=top_k
            )
        )
        # نحفظ وسيط زمن الاسترجاع ووسيط كلفة الميزة.
        retrieval_times.append(statistics.median(retrieval_samples))
        feature_costs.append(statistics.median(feature_samples))

    # عدد الاستعلامات المقيمة فعلياً.
    count = len(before_metrics)
    if count == 0:
        return None

    # نحسب متوسط زمن الاسترجاع.
    average_retrieval = sum(retrieval_times) / count
    # نحسب متوسط كلفة الميزة.
    average_feature_cost = sum(feature_costs) / count
    # الزمن الكلي المقدر بعد إضافة الميزة.
    estimated_total = average_retrieval + average_feature_cost
    # نسبة كلفة الميزة مقارنة بزمن الاسترجاع.
    overhead_percent = (
        average_feature_cost / average_retrieval * 100
        if average_retrieval > 0
        else 0
    )

    def summarize(metrics, use_topic_detection):
        # نبني ملخصاً موحداً لحالة قبل أو بعد الميزة.
        return {
            "dataset": dataset,
            "method": method,
            "use_topic_detection": use_topic_detection,
            "evaluated_queries_count": count,
            "candidate_pool_size": candidate_pool_size,
            "final_top_k": top_k,
            "MAP": sum(item["Average_Precision"] for item in metrics) / count,
            "Recall": sum(item["Recall"] for item in metrics) / count,
            "Precision@10": sum(item["Precision@10"] for item in metrics) / count,
            "nDCG": sum(item["nDCG"] for item in metrics) / count,
            "average_warmed_retrieval_time_seconds": average_retrieval,
            "average_topic_feature_cost_seconds": (
                average_feature_cost if use_topic_detection else 0
            ),
            "average_estimated_total_time_seconds": (
                estimated_total if use_topic_detection else average_retrieval
            ),
            "feature_overhead_percent": (
                overhead_percent if use_topic_detection else 0
            )
        }

    return {
        "before": summarize(before_metrics, False),
        "after": summarize(after_metrics, True),
        "timing": {
            "average_warmed_retrieval_time_seconds": average_retrieval,
            "average_topic_feature_cost_seconds": average_feature_cost,
            "average_estimated_total_time_seconds": estimated_total,
            "feature_overhead_percent": overhead_percent,
            "retrieval_repetitions_per_query": retrieval_repetitions,
            "feature_repetitions_per_query": feature_repetitions
        }
    }


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
# Baseline Evaluation
# هنا نقيم كل طرق الاسترجاع قبل تشغيل أي ميزة إضافية.
    # نقرأ استعلامات الاختبار وملفات الصلة الرسمية.
    queries_df, qrels_df = load_evaluation_files(
        dataset=dataset,
        save_dir=save_dir
    )

    # نقرأ معرفات الوثائق المتاحة فعلياً في الداتا المحملة.
    available_doc_ids = get_available_doc_ids(
        dataset=dataset,
        save_dir=save_dir
    )

    # نحتفظ فقط بأحكام الصلة التي تشير إلى وثائق موجودة عندنا.
    filtered_qrels_df = filter_qrels_to_available_docs(
        qrels_df=qrels_df,
        available_doc_ids=available_doc_ids
    )

    # نختار عدداً محدداً من الاستعلامات الرسمية الصالحة للتقييم.
    valid_queries_df = get_valid_queries(
        queries_df=queries_df,
        filtered_qrels_df=filtered_qrels_df,
        max_queries=max_queries
    )

    # نخزن نتائج كل طريقة.
    summary = []

    # نقيم كل طريقة استرجاع على نفس الاستعلامات.
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
# Topic Detection Evaluation
# هنا نقيم كل الطرق قبل وبعد ميزة كشف الموضوع للمقارنة العادلة.
    # نقرأ استعلامات الاختبار وأحكام الصلة الرسمية.
    queries_df, qrels_df = load_evaluation_files(
        dataset=dataset,
        save_dir=save_dir
    )

    # نقرأ الوثائق الموجودة فعلياً في الداتا.
    available_doc_ids = get_available_doc_ids(
        dataset=dataset,
        save_dir=save_dir
    )

    # Qrels
    # نفلتر أحكام الصلة حتى تبقى فقط الوثائق الموجودة في الداتا.
    filtered_qrels_df = filter_qrels_to_available_docs(
        qrels_df=qrels_df,
        available_doc_ids=available_doc_ids
    )

    # نختار الاستعلامات الرسمية التي لها أحكام صلة صالحة.
    valid_queries_df = get_valid_queries(
        queries_df=queries_df,
        filtered_qrels_df=filtered_qrels_df,
        max_queries=max_queries
    )

    # نتائج قبل الميزة.
    before_additional_feature = []
    # نتائج بعد كشف الموضوع.
    after_topic_detection = []
    # فروق المقاييس والزمن بين الحالتين.
    metric_differences = []

    # نستخدم مجموعة مرشحين أكبر حتى تستطيع الميزة إعادة الترتيب.
    topic_candidate_pool_size = max(top_k * 5, 50)
    # نستخدم نفس حجم المرشحين قبل وبعد لضمان عدالة المقارنة.
    baseline_candidate_pool_size = topic_candidate_pool_size

    # إذا كان عدد الوثائق أقل من حجم المرشحين نستخدم عدد الوثائق فقط.
    if topic_candidate_pool_size > len(available_doc_ids):
        topic_candidate_pool_size = len(available_doc_ids)

    # نقيم كل طريقة استرجاع قبل وبعد الميزة.
    for method in methods:
        paired_result = evaluate_single_method_with_topic_cost(
            dataset=dataset,
            method=method,
            loaded_data=loaded_data,
            queries_df=valid_queries_df,
            filtered_qrels_df=filtered_qrels_df,
            top_k=top_k,
            k1=k1,
            b=b,
            alpha=alpha,
            candidate_pool_size=baseline_candidate_pool_size,
            retrieval_repetitions=3,
            feature_repetitions=10
        )

        # إذا لم توجد نتائج صالحة لهذه الطريقة نتجاوزها.
        if paired_result is None:
            continue

        # نأخذ ملخص قبل الميزة وبعدها.
        baseline_result = paired_result["before"]
        topic_result = paired_result["after"]

        # نضيف معلومات حجم الداتا وملفات الصلة للشرح في الواجهة.
        baseline_result["available_documents_count"] = len(available_doc_ids)
        baseline_result["available_qrels_count"] = len(filtered_qrels_df)

        topic_result["available_documents_count"] = len(available_doc_ids)
        topic_result["available_qrels_count"] = len(filtered_qrels_df)

        # نحسب فروق المقاييس بين بعد وقبل.
        timing = paired_result["timing"]
        difference = {
            "dataset": dataset,
            "method": method,
            "MAP_difference": topic_result["MAP"] - baseline_result["MAP"],
            "Recall_difference": (
                topic_result["Recall"] - baseline_result["Recall"]
            ),
            "Precision@10_difference": (
                topic_result["Precision@10"]
                - baseline_result["Precision@10"]
            ),
            "nDCG_difference": (
                topic_result["nDCG"] - baseline_result["nDCG"]
            ),
            "topic_feature_cost_seconds": (
                timing["average_topic_feature_cost_seconds"]
            ),
            "topic_feature_cost_milliseconds": (
                timing["average_topic_feature_cost_seconds"] * 1000
            ),
            "estimated_total_with_feature_seconds": (
                timing["average_estimated_total_time_seconds"]
            ),
            "feature_overhead_percent": timing["feature_overhead_percent"]
        }

        # نجمع النتائج النهائية في القوائم المناسبة.
        before_additional_feature.append(baseline_result)
        after_topic_detection.append(topic_result)
        metric_differences.append(difference)

    return {
        "dataset": dataset,
        "top_k": top_k,
        "max_queries": max_queries,
        "evaluation_mode": "before_and_after_topic_detection",
        "evaluation_strategy": "Each method is warmed first. Retrieval is measured three times per query and the median is used. Topic detection is then measured separately ten times on the same candidate results. Quality before and after uses the identical candidate pool.",
        "timing_mode": "isolated_warmed_feature_cost",
        "retrieval_repetitions_per_query": 3,
        "feature_repetitions_per_query": 10,
        "baseline_candidate_pool_size": baseline_candidate_pool_size,
        "topic_candidate_pool_size": topic_candidate_pool_size,
        "before_additional_feature": before_additional_feature,
        "after_topic_detection": after_topic_detection,
        "metric_differences": metric_differences
    }
