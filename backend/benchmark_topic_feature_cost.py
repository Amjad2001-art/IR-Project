import json
import statistics
import time
from pathlib import Path

from services.evaluation_service import (
    filter_qrels_to_available_docs,
    get_available_doc_ids,
    get_valid_queries,
    load_evaluation_files,
    topic_rerank_results,
)
from services.model_loader_service import ModelLoaderService
from services.retrieval_service import run_search


METHODS = [
    "tfidf",
    "word2vec",
    "bm25",
    "inverted_index",
    "serial_hybrid",
    "parallel_hybrid",
]
DATASETS = ["dataset1", "dataset2"]
QUERY_COUNT = 10
CANDIDATE_POOL_SIZE = 50
RETRIEVAL_REPETITIONS = 3
FEATURE_REPETITIONS = 20
K1 = 1.5
B = 0.75
ALPHA = 0.6


def median_seconds(samples):
    return statistics.median(samples)


def percentile(values, percentile):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def benchmark_method(dataset, method, loaded_data, queries):
    query_measurements = []

    # Warm the retrieval implementation and method-specific caches once.
    run_search(
        query=str(queries.iloc[0]["text"]),
        dataset=dataset,
        method=method,
        loaded_data=loaded_data,
        top_k=CANDIDATE_POOL_SIZE,
        k1=K1,
        b=B,
        alpha=ALPHA,
    )

    for _, query_row in queries.iterrows():
        query = str(query_row["text"])
        retrieval_samples = []
        candidate_results = None

        for _ in range(RETRIEVAL_REPETITIONS):
            start = time.perf_counter()
            candidate_results = run_search(
                query=query,
                dataset=dataset,
                method=method,
                loaded_data=loaded_data,
                top_k=CANDIDATE_POOL_SIZE,
                k1=K1,
                b=B,
                alpha=ALPHA,
            )
            retrieval_samples.append(time.perf_counter() - start)

        # Warm the feature implementation for this result shape.
        topic_rerank_results(candidate_results, topic_boost_factor=0.08)

        feature_samples = []
        for _ in range(FEATURE_REPETITIONS):
            start = time.perf_counter()
            topic_rerank_results(candidate_results, topic_boost_factor=0.08)
            feature_samples.append(time.perf_counter() - start)

        retrieval_median = median_seconds(retrieval_samples)
        feature_median = median_seconds(feature_samples)

        query_measurements.append(
            {
                "query_id": str(query_row["query_id"]),
                "warmed_retrieval_median_seconds": retrieval_median,
                "topic_feature_median_seconds": feature_median,
                "estimated_total_with_feature_seconds": (
                    retrieval_median + feature_median
                ),
                "feature_overhead_percent": (
                    feature_median / retrieval_median * 100
                    if retrieval_median > 0
                    else 0
                ),
            }
        )

    retrieval_values = [
        item["warmed_retrieval_median_seconds"]
        for item in query_measurements
    ]
    feature_values = [
        item["topic_feature_median_seconds"]
        for item in query_measurements
    ]
    total_values = [
        item["estimated_total_with_feature_seconds"]
        for item in query_measurements
    ]

    average_retrieval = statistics.mean(retrieval_values)
    average_feature = statistics.mean(feature_values)

    return {
        "dataset": dataset,
        "method": method,
        "query_count": len(query_measurements),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "retrieval_repetitions_per_query": RETRIEVAL_REPETITIONS,
        "feature_repetitions_per_query": FEATURE_REPETITIONS,
        "average_warmed_retrieval_seconds": average_retrieval,
        "average_topic_feature_cost_seconds": average_feature,
        "average_estimated_total_with_feature_seconds": statistics.mean(
            total_values
        ),
        "feature_overhead_percent": (
            average_feature / average_retrieval * 100
            if average_retrieval > 0
            else 0
        ),
        "median_topic_feature_cost_seconds": statistics.median(feature_values),
        "p95_topic_feature_cost_seconds": percentile(feature_values, 0.95),
        "per_query": query_measurements,
    }


def main():
    save_dir = Path("saved_files")
    loaded_data = ModelLoaderService(save_dir=str(save_dir)).load_all()
    output = {
        "benchmark": "isolated_warmed_topic_feature_cost",
        "query_count": QUERY_COUNT,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "retrieval_repetitions_per_query": RETRIEVAL_REPETITIONS,
        "feature_repetitions_per_query": FEATURE_REPETITIONS,
        "results": [],
    }

    for dataset in DATASETS:
        queries_df, qrels_df = load_evaluation_files(
            dataset=dataset,
            save_dir=str(save_dir),
        )
        available_doc_ids = get_available_doc_ids(
            dataset=dataset,
            save_dir=str(save_dir),
        )
        filtered_qrels = filter_qrels_to_available_docs(
            qrels_df=qrels_df,
            available_doc_ids=available_doc_ids,
        )
        valid_queries = get_valid_queries(
            queries_df=queries_df,
            filtered_qrels_df=filtered_qrels,
            max_queries=QUERY_COUNT,
        )

        for method in METHODS:
            print(f"Benchmarking {dataset} / {method}", flush=True)
            output["results"].append(
                benchmark_method(
                    dataset=dataset,
                    method=method,
                    loaded_data=loaded_data,
                    queries=valid_queries,
                )
            )

    target = save_dir / "verified_topic_feature_cost.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(target, flush=True)


if __name__ == "__main__":
    main()
