import json
import os
import pickle
import zipfile

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from services.preprocessing_service import preprocess_stemming


SAVE_DIR = "saved_files"
DATASET_NAME = "wikir/en1k/test"
EXPECTED_DOCUMENT_COUNT = 369721
EXPECTED_QUERY_COUNT = 100
EXPECTED_QREL_COUNT = 4435
WIKIR_ARCHIVE = os.path.expanduser(
    "~/.ir_datasets/downloads/554299bca984640cb283d6ba55753608"
)


def load_source_files():
    # Full Dataset
    # هنا نقرأ الداتا الأولى كاملة من المصدر الرسمي بدون تقسيم حسب الصلة.
    if not os.path.exists(WIKIR_ARCHIVE):
        raise FileNotFoundError(
            "The official WikIR archive is missing from the ir_datasets cache."
        )

    print(f"Reading complete WikIR archive: {WIKIR_ARCHIVE}")
    with zipfile.ZipFile(WIKIR_ARCHIVE) as archive:
        with archive.open("wikIR1k/documents.csv") as file:
            docs_df = pd.read_csv(
                file,
                dtype={"id_right": str, "text_right": str},
            ).rename(
                columns={"id_right": "doc_id", "text_right": "text"}
            )
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

    actual_counts = (
        len(docs_df),
        len(queries_df),
        len(qrels_df),
    )
    expected_counts = (
        EXPECTED_DOCUMENT_COUNT,
        EXPECTED_QUERY_COUNT,
        EXPECTED_QREL_COUNT,
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected WikIR counts: {actual_counts}; "
            f"expected {expected_counts}."
        )
    return docs_df, queries_df, qrels_df


def build_inverted_index(docs_df):
    # Inverted Index
    # هنا نبني الفهرس المعكوس للداتا الأولى من النصوص المعالجة.
    inverted_index = {}
    for row in docs_df.itertuples(index=False):
        for token in str(row.processed_text).split():
            postings = inverted_index.setdefault(token, {})
            postings[str(row.doc_id)] = postings.get(str(row.doc_id), 0) + 1
    return inverted_index


def document_to_vector(tokens, model, vector_size):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)


def save_pickle(file_name, value):
    with open(os.path.join(SAVE_DIR, file_name), "wb") as file:
        pickle.dump(value, file)


def export_evaluation_files(queries_df, qrels_df):
    # Test Queries And Qrels
    # هنا نحفظ استعلامات الاختبار وأحكام الصلة للتقييم فقط.
    queries_df.to_csv(
        os.path.join(SAVE_DIR, "queries_dataset1.csv"),
        index=False,
        encoding="utf-8",
    )
    qrels_df.to_csv(
        os.path.join(SAVE_DIR, "qrels_dataset1.csv"),
        index=False,
        encoding="utf-8",
    )
    return len(queries_df), len(qrels_df)


def build_artifacts(docs_df, source_count, query_count, qrel_count):
    # Cached Artifacts
    # هنا نبني ونحفظ كل ملفات التمثيل المطلوبة حتى لا يعاد بناؤها عند التشغيل.
    print("Preprocessing WikIR documents...")
    docs_df["text"] = docs_df["text"].fillna("").astype(str)
    docs_df["processed_text"] = docs_df["text"].apply(preprocess_stemming)
    tokenized_corpus = [
        text.split() for text in docs_df["processed_text"].tolist()
    ]

    print("Building WikIR TF-IDF artifacts...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_df["processed_text"])

    print("Training WikIR Word2Vec model...")
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
    )

    print("Building WikIR Word2Vec matrix...")
    word2vec_matrix = np.array(
        [
            document_to_vector(tokens, word2vec_model, 100)
            for tokens in tokenized_corpus
        ],
        dtype=np.float32,
    )

    print("Building WikIR inverted index...")
    inverted_index = build_inverted_index(docs_df)

    print("Saving complete WikIR artifacts...")
    docs_df.to_csv(
        os.path.join(SAVE_DIR, "work_dataset1.csv"),
        index=False,
        encoding="utf-8",
    )
    save_pickle("tfidf_vectorizer_1.pkl", vectorizer)
    save_pickle("tfidf_matrix_1.pkl", tfidf_matrix)
    save_pickle("tokenized_corpus_1.pkl", tokenized_corpus)
    save_pickle("inverted_index_1.pkl", inverted_index)
    np.save(
        os.path.join(SAVE_DIR, "word2vec_matrix_1.npy"),
        word2vec_matrix,
    )
    word2vec_model.save(
        os.path.join(SAVE_DIR, "word2vec_model_dataset1.model")
    )

    metadata = {
        "dataset": DATASET_NAME,
        "document_count": int(len(docs_df)),
        "source_document_count": int(source_count),
        "query_count": int(query_count),
        "qrel_count": int(qrel_count),
        "full_collection": True,
        "selection_strategy": "all_documents_from_source",
        "selection_uses_qrels": False,
        "qrels_usage": "evaluation_only",
        "artifacts_cached_on_disk": True,
    }
    with open(
        os.path.join(SAVE_DIR, "dataset1_build_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Loading dataset: {DATASET_NAME}")
    docs_df, queries_df, qrels_df = load_source_files()
    query_count, qrel_count = export_evaluation_files(
        queries_df,
        qrels_df,
    )
    build_artifacts(
        docs_df,
        EXPECTED_DOCUMENT_COUNT,
        query_count,
        qrel_count,
    )
    print("The complete WikIR dataset and cached artifacts are ready.")


if __name__ == "__main__":
    main()
