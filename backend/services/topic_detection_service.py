import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from services.retrieval_service import run_search


def _safe_text(value):
    if value is None:
        return ""

    return str(value)


def _get_text_column(docs_df):
    possible_columns = [
        "text",
        "processed_text",
        "body",
        "title"
    ]

    for column in possible_columns:
        if column in docs_df.columns:
            return column

    raise ValueError(
        "No text column found. Expected one of: text, processed_text, body, title"
    )


def detect_topic_from_results(results, max_terms=8):
    texts = [
        _safe_text(item.get("text", ""))
        for item in results
        if len(_safe_text(item.get("text", "")).strip()) > 0
    ]

    if len(texts) == 0:
        return {
            "topic_detection_applied": False,
            "reason": "No result texts available for topic detection",
            "detected_topic": "No clear topic",
            "top_topic_terms": [],
            "documents_analyzed": 0
        }

    if len(texts) == 1:
        max_df = 1.0
    else:
        max_df = 0.95

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        min_df=1,
        max_df=max_df
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {
            "topic_detection_applied": False,
            "reason": "Not enough meaningful terms after preprocessing",
            "detected_topic": "No clear topic",
            "top_topic_terms": [],
            "documents_analyzed": len(texts)
        }

    feature_names = vectorizer.get_feature_names_out()
    term_scores = tfidf_matrix.sum(axis=0).A1

    ranked_terms = sorted(
        zip(feature_names, term_scores),
        key=lambda item: item[1],
        reverse=True
    )

    top_terms = [
        {
            "term": term,
            "score": float(score)
        }
        for term, score in ranked_terms[:max_terms]
    ]

    detected_topic_terms = [
        item["term"]
        for item in top_terms[:3]
    ]

    if len(detected_topic_terms) == 0:
        detected_topic = "No clear topic"
    else:
        detected_topic = " / ".join(detected_topic_terms)

    return {
        "topic_detection_applied": True,
        "reason": "Topic was detected from the top retrieved documents using TF-IDF term importance",
        "detected_topic": detected_topic,
        "top_topic_terms": top_terms,
        "documents_analyzed": len(texts)
    }


def detect_topic_for_query(
    query,
    dataset,
    method,
    loaded_data,
    top_k=10,
    k1=1.5,
    b=0.75,
    alpha=0.6
):
    results = run_search(
        query=query,
        dataset=dataset,
        method=method,
        loaded_data=loaded_data,
        top_k=top_k,
        k1=k1,
        b=b,
        alpha=alpha
    )

    topic_info = detect_topic_from_results(
        results=results,
        max_terms=8
    )

    return {
        "query": query,
        "dataset": dataset,
        "method": method,
        "top_k": top_k,
        "topic_info": topic_info,
        "results": results
    }


def load_documents_for_clustering(dataset, save_dir="saved_files", max_docs=1000):
    if dataset == "dataset1":
        docs_file = "work_dataset1.csv"
    elif dataset == "dataset2":
        docs_file = "work_dataset2.csv"
    else:
        raise ValueError("dataset must be dataset1 or dataset2")

    docs_path = os.path.join(save_dir, docs_file)

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing documents file: {docs_path}")

    docs_df = pd.read_csv(docs_path)
    docs_df["doc_id"] = docs_df["doc_id"].astype(str)

    text_column = _get_text_column(docs_df)

    docs_df[text_column] = docs_df[text_column].fillna("").astype(str)
    docs_df = docs_df[docs_df[text_column].str.strip().str.len() > 0].copy()

    if max_docs is not None and max_docs > 0:
        docs_df = docs_df.head(max_docs).copy()

    return docs_df, text_column


def cluster_documents(
    dataset,
    save_dir="saved_files",
    n_clusters=5,
    max_docs=1000,
    top_terms_per_cluster=8
):
    docs_df, text_column = load_documents_for_clustering(
        dataset=dataset,
        save_dir=save_dir,
        max_docs=max_docs
    )

    if len(docs_df) == 0:
        return {
            "clustering_applied": False,
            "reason": "No documents available for clustering",
            "dataset": dataset,
            "clusters": []
        }

    if n_clusters < 2:
        n_clusters = 2

    if n_clusters > len(docs_df):
        n_clusters = len(docs_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(docs_df[text_column].tolist())

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(tfidf_matrix)

    docs_df["cluster"] = labels

    feature_names = vectorizer.get_feature_names_out()
    clusters = []

    for cluster_id in range(n_clusters):
        cluster_indexes = docs_df.index[docs_df["cluster"] == cluster_id].tolist()
        cluster_docs = docs_df.loc[cluster_indexes]

        center = model.cluster_centers_[cluster_id]
        top_term_indexes = center.argsort()[::-1][:top_terms_per_cluster]

        top_terms = [
            {
                "term": feature_names[index],
                "score": float(center[index])
            }
            for index in top_term_indexes
            if center[index] > 0
        ]

        sample_documents = []

        for _, row in cluster_docs.head(5).iterrows():
            sample_documents.append({
                "doc_id": str(row["doc_id"]),
                "text_preview": _safe_text(row[text_column])[:250]
            })

        clusters.append({
            "cluster_id": int(cluster_id),
            "document_count": int(len(cluster_docs)),
            "topic_label": " / ".join([item["term"] for item in top_terms[:3]]),
            "top_terms": top_terms,
            "sample_documents": sample_documents
        })

    return {
        "clustering_applied": True,
        "reason": "Documents were clustered offline using TF-IDF vectors and K-Means",
        "dataset": dataset,
        "documents_clustered": int(len(docs_df)),
        "n_clusters": int(n_clusters),
        "clusters": clusters
    }
