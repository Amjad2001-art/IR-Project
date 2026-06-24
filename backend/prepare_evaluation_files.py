
import os
import zipfile

import pandas as pd
import ir_datasets

SAVE_DIR = "saved_files"
WIKIR_ARCHIVE = os.path.expanduser(
    "~/.ir_datasets/downloads/554299bca984640cb283d6ba55753608"
)

DATASETS = {
    "dataset1": "wikir/en1k/test",
    "dataset2": "beir/quora/test"
}


def load_wikir_evaluation_files():
    # Test Queries And Qrels
    # هنا نقرأ ملفات الاختبار الرسمية للداتا الأولى.
    if not os.path.exists(WIKIR_ARCHIVE):
        raise FileNotFoundError(
            "Build the complete WikIR dataset first to cache its archive."
        )

    with zipfile.ZipFile(WIKIR_ARCHIVE) as archive:
        with archive.open("wikIR1k/test/queries.csv") as file:
            queries_df = pd.read_csv(
                file,
                dtype={"id_left": str, "text_left": str},
            ).rename(
                columns={"id_left": "query_id", "text_left": "text"}
            )
        with archive.open("wikIR1k/test/qrels") as file:
            qrels_df = pd.read_csv(
                file,
                sep="\t",
                header=None,
                names=["query_id", "iteration", "doc_id", "relevance"],
                dtype={"query_id": str, "doc_id": str, "relevance": int},
            )[["query_id", "doc_id", "relevance"]]

    return queries_df, qrels_df


def export_queries_and_qrels(dataset_key, dataset_name):
    # Evaluation Files
    # هنا نحفظ استعلامات الاختبار وأحكام الصلة لكل داتا سيت.
    print(f"Loading dataset: {dataset_name}")

    if dataset_key == "dataset1":
        queries_df, qrels_df = load_wikir_evaluation_files()
        save_evaluation_files(dataset_key, queries_df, qrels_df)
        return

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

    save_evaluation_files(
        dataset_key,
        pd.DataFrame(queries),
        pd.DataFrame(qrels),
    )


def save_evaluation_files(dataset_key, queries_df, qrels_df):
    # Evaluation Cache
    # هنا نخزن ملفات التقييم في الكاش لاستخدامها في حساب المقاييس.
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

