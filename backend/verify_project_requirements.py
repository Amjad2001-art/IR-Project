import gc
import json
import os
import pickle

import numpy as np
import pandas as pd


SAVE_DIR = "saved_files"

DATASETS = {
    "dataset1": {
        "name": "wikir/en1k/test",
        "suffix": "1",
        "expected_documents": 369721,
        "expected_queries": 100,
        "expected_qrels": 4435,
    },
}


def file_path(file_name):
    return os.path.join(SAVE_DIR, file_name)


def print_check(label, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def count_csv_rows(path):
    return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=50000))


def load_pickle(file_name):
    with open(file_path(file_name), "rb") as file:
        return pickle.load(file)


def verify_dataset(dataset_id, specification):
    suffix = specification["suffix"]
    required_files = [
        f"work_dataset{suffix}.csv",
        f"queries_dataset{suffix}.csv",
        f"qrels_dataset{suffix}.csv",
        f"tfidf_vectorizer_{suffix}.pkl",
        f"tfidf_matrix_{suffix}.pkl",
        f"word2vec_model_dataset{suffix}.model",
        f"word2vec_matrix_{suffix}.npy",
        f"tokenized_corpus_{suffix}.pkl",
        f"inverted_index_{suffix}.pkl",
        f"dataset{suffix}_build_metadata.json",
    ]

    print()
    print(f"{dataset_id}: {specification['name']}")
    print("-" * 72)

    checks = []
    missing = [
        name for name in required_files
        if not os.path.exists(file_path(name))
    ]
    checks.append(
        print_check(
            "Required artifacts",
            not missing,
            "all files found" if not missing else f"missing: {missing}",
        )
    )
    if missing:
        return checks

    with open(
        file_path(f"dataset{suffix}_build_metadata.json"),
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    document_count = count_csv_rows(
        file_path(f"work_dataset{suffix}.csv")
    )
    query_count = count_csv_rows(
        file_path(f"queries_dataset{suffix}.csv")
    )
    qrel_count = count_csv_rows(
        file_path(f"qrels_dataset{suffix}.csv")
    )

    checks.append(
        print_check(
            "Dataset identity",
            metadata.get("dataset") == specification["name"],
            metadata.get("dataset"),
        )
    )
    checks.append(
        print_check(
            "Complete source collection",
            (
                metadata.get("full_collection") is True
                and document_count
                == specification["expected_documents"]
                == int(metadata.get("source_document_count", -1))
            ),
            (
                f"stored={document_count}, "
                f"expected={specification['expected_documents']}"
            ),
        )
    )
    checks.append(
        print_check(
            "Evaluation files",
            (
                query_count == specification["expected_queries"]
                and qrel_count == specification["expected_qrels"]
            ),
            f"queries={query_count}, qrels={qrel_count}",
        )
    )
    checks.append(
        print_check(
            "No qrels-based selection",
            (
                metadata.get("selection_uses_qrels") is False
                and metadata.get("selection_strategy")
                == "all_documents_from_source"
                and metadata.get("qrels_usage") == "evaluation_only"
            ),
            metadata.get("selection_strategy"),
        )
    )
    checks.append(
        print_check(
            "Artifacts cached on disk",
            metadata.get("artifacts_cached_on_disk") is True,
            metadata.get("artifacts_cached_on_disk"),
        )
    )

    tokenized_corpus = load_pickle(f"tokenized_corpus_{suffix}.pkl")
    checks.append(
        print_check(
            "Tokenized corpus alignment",
            len(tokenized_corpus) == document_count,
            f"rows={len(tokenized_corpus)}, docs={document_count}",
        )
    )
    del tokenized_corpus
    gc.collect()

    tfidf_matrix = load_pickle(f"tfidf_matrix_{suffix}.pkl")
    checks.append(
        print_check(
            "TF-IDF matrix alignment",
            tfidf_matrix.shape[0] == document_count,
            f"rows={tfidf_matrix.shape[0]}, docs={document_count}",
        )
    )
    del tfidf_matrix
    gc.collect()

    word2vec_matrix = np.load(
        file_path(f"word2vec_matrix_{suffix}.npy"),
        mmap_mode="r",
    )
    checks.append(
        print_check(
            "Word2Vec matrix alignment",
            word2vec_matrix.shape[0] == document_count,
            f"rows={word2vec_matrix.shape[0]}, docs={document_count}",
        )
    )
    del word2vec_matrix
    gc.collect()

    inverted_index = load_pickle(f"inverted_index_{suffix}.pkl")
    checks.append(
        print_check(
            "Inverted index built",
            len(inverted_index) > 0,
            f"terms={len(inverted_index)}",
        )
    )
    del inverted_index
    gc.collect()
    return checks


def main():
    print("Complete IR Project Verification")
    print("=" * 72)

    checks = []
    for dataset_id, specification in DATASETS.items():
        checks.extend(verify_dataset(dataset_id, specification))

    print()
    print("=" * 72)
    if not checks or not all(checks):
        raise SystemExit("Verification failed.")
    print("The complete WikIR dataset and all cached artifacts passed.")


if __name__ == "__main__":
    main()
