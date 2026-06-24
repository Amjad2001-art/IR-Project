import os
import shutil
import pickle
import json
import numpy as np
import pandas as pd
import ir_datasets

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from services.preprocessing_service import preprocess_stemming


SAVE_DIR = "saved_files"
BACKUP_DIR = os.path.join(SAVE_DIR, "backup_quora_before_full_rebuild")

DATASET_NAME = "beir/quora/test"

FILES_TO_BACKUP = [
    "work_dataset2.csv",
    "tfidf_vectorizer_2.pkl",
    "tfidf_matrix_2.pkl",
    "word2vec_model_dataset2.model",
    "word2vec_matrix_2.npy",
    "tokenized_corpus_2.pkl",
    "inverted_index_2.pkl",
]


def report_current_collection_size():
    work_dataset_path = os.path.join(SAVE_DIR, "work_dataset2.csv")

    if os.path.exists(work_dataset_path):
        try:
            current_df = pd.read_csv(work_dataset_path)
            current_size = len(current_df)

            if current_size > 0:
                print(f"Current Quora collection size found: {current_size}")
                return
        except Exception:
            pass

    print("No valid existing Quora collection found.")


def backup_old_files():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    for file_name in FILES_TO_BACKUP:
        source_path = os.path.join(SAVE_DIR, file_name)

        if os.path.exists(source_path):
            destination_path = os.path.join(BACKUP_DIR, file_name)
            shutil.copy2(source_path, destination_path)
            print(f"Backed up: {file_name}")

    print("Backup completed.")
    print("-" * 60)


def get_text_from_document(doc):
    text_parts = []

    if hasattr(doc, "title") and doc.title:
        text_parts.append(str(doc.title))

    if hasattr(doc, "text") and doc.text:
        text_parts.append(str(doc.text))

    if hasattr(doc, "body") and doc.body:
        text_parts.append(str(doc.body))

    if len(text_parts) == 0:
        text_parts.append(str(doc))

    return " ".join(text_parts)


def build_full_document_collection(dataset):
    # Full Dataset
    # هنا نقرأ كل وثائق الداتا الثانية من المصدر الرسمي بدون أي اختيار جزئي.
    selected_docs = []
    expected_count = dataset.docs_count()

    print(f"Collecting the complete dataset: {expected_count} documents")

    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        text = get_text_from_document(doc)

        selected_docs.append({
            "doc_id": doc_id,
            "text": text
        })

        if len(selected_docs) % 25000 == 0:
            print(f"Collected documents: {len(selected_docs)} / {expected_count}")

    docs_df = pd.DataFrame(selected_docs)

    print(f"Final complete collection size: {len(docs_df)}")

    if len(docs_df) != expected_count:
        raise RuntimeError(
            f"Dataset source reports {expected_count} documents, "
            f"but only {len(docs_df)} were collected."
        )

    print("-" * 60)

    return docs_df


def build_inverted_index(docs_df):
    # Inverted Index
    # هنا نبني الفهرس المعكوس للداتا الثانية بعد المعالجة.
    inverted_index = {}

    for _, row in docs_df.iterrows():
        doc_id = str(row["doc_id"])
        tokens = str(row["processed_text"]).split()

        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}

            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0

            inverted_index[token][doc_id] += 1

    return inverted_index


def document_to_vector(tokens, model, vector_size):
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


def save_pickle(file_name, obj):
    path = os.path.join(SAVE_DIR, file_name)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def rebuild_dataset2_artifacts(docs_df, source_document_count):
    # Cached Artifacts
    # هنا نبني ملفات التمثيل والفهرسة للداتا الثانية ونحفظها للكاش.
    print("Preprocessing documents...")

    docs_df["text"] = docs_df["text"].astype(str)
    docs_df["processed_text"] = docs_df["text"].apply(preprocess_stemming)

    tokenized_corpus = [
        text.split()
        for text in docs_df["processed_text"].tolist()
    ]

    print("Building TF-IDF artifacts...")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs_df["processed_text"])

    print("Training Word2Vec model...")

    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    print("Building Word2Vec matrix...")

    word2vec_matrix = np.array(
        [
            document_to_vector(tokens, word2vec_model, 100)
            for tokens in tokenized_corpus
        ],
        dtype=np.float32
    )

    print("Building inverted index...")

    inverted_index = build_inverted_index(docs_df)

    print("Saving complete Quora dataset artifacts...")

    docs_df.to_csv(
        os.path.join(SAVE_DIR, "work_dataset2.csv"),
        index=False,
        encoding="utf-8"
    )

    save_pickle("tfidf_vectorizer_2.pkl", tfidf_vectorizer)
    save_pickle("tfidf_matrix_2.pkl", tfidf_matrix)
    save_pickle("tokenized_corpus_2.pkl", tokenized_corpus)
    save_pickle("inverted_index_2.pkl", inverted_index)

    np.save(
        os.path.join(SAVE_DIR, "word2vec_matrix_2.npy"),
        word2vec_matrix
    )

    word2vec_model.save(
        os.path.join(SAVE_DIR, "word2vec_model_dataset2.model")
    )

    metadata = {
        "dataset": DATASET_NAME,
        "document_count": int(len(docs_df)),
        "source_document_count": int(source_document_count),
        "full_collection": True,
        "selection_strategy": "all_documents_from_source",
        "selection_uses_qrels": False,
        "qrels_usage": "evaluation_only",
        "artifacts_cached_on_disk": True,
    }

    with open(
        os.path.join(SAVE_DIR, "dataset2_build_metadata.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("All Dataset 2 artifacts were rebuilt successfully.")
    print("-" * 60)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    report_current_collection_size()

    backup_old_files()

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = ir_datasets.load(DATASET_NAME)
    source_document_count = dataset.docs_count()

    docs_df = build_full_document_collection(dataset)

    rebuild_dataset2_artifacts(docs_df, source_document_count)

    print("The complete Quora dataset and cached artifacts are ready.")
    print("Now restart the FastAPI backend and run /evaluate-system again.")


if __name__ == "__main__":
    main()
