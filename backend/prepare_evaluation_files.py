
import os
import pandas as pd
import ir_datasets

SAVE_DIR = "saved_files"

DATASETS = {
    "dataset1": "beir/webis-touche2020/v2",
    "dataset2": "beir/quora/test"
}


def export_queries_and_qrels(dataset_key, dataset_name):
    print(f"Loading dataset: {dataset_name}")

    dataset = ir_datasets.load(dataset_name)

    queries = []
    for query in dataset.queries_iter():
        query_id = str(query.query_id)

        if hasattr(query, "text"):
            text = query.text
        elif hasattr(query, "title"):
            text = query.title
        else:
            text = str(query)

        queries.append({
            "query_id": query_id,
            "text": text
        })

    qrels = []
    for qrel in dataset.qrels_iter():
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)

        if hasattr(qrel, "relevance"):
            relevance = int(qrel.relevance)
        else:
            relevance = 1

        qrels.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "relevance": relevance
        })

    queries_df = pd.DataFrame(queries)
    qrels_df = pd.DataFrame(qrels)

    os.makedirs(SAVE_DIR, exist_ok=True)

    queries_path = os.path.join(SAVE_DIR, f"queries_{dataset_key}.csv")
    qrels_path = os.path.join(SAVE_DIR, f"qrels_{dataset_key}.csv")

    queries_df.to_csv(queries_path, index=False, encoding="utf-8")
    qrels_df.to_csv(qrels_path, index=False, encoding="utf-8")

    print(f"Saved queries to: {queries_path}")
    print(f"Saved qrels to: {qrels_path}")
    print(f"Queries count: {len(queries_df)}")
    print(f"Qrels count: {len(qrels_df)}")
    print("-" * 60)


def main():
    for dataset_key, dataset_name in DATASETS.items():
        export_queries_and_qrels(dataset_key, dataset_name)

    print("Evaluation files are ready.")


if __name__ == "__main__":
    main()

