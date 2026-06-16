import os
import shutil
import pickle
import numpy as np
import pandas as pd
import ir_datasets

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from services.preprocessing_service import preprocess_stemming


SAVE_DIR = "saved_files"
BACKUP_DIR = os.path.join(SAVE_DIR, "backup_dataset1_before_smart_subset")

DATASET_NAME = "beir/webis-touche2020/v2"

SUBSET_SIZE = 3000
TARGET_EVALUATION_QUERIES = 40

FILES_TO_BACKUP = [
    "work_dataset1.csv",
    "tfidf_vectorizer_1.pkl",
    "tfidf_matrix_1.pkl",
    "word2vec_model_dataset1.model",
    "word2vec_matrix_1.npy",
    "tokenized_corpus_1.pkl",
    "inverted_index_1.pkl",
]


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


def collect_relevant_doc_ids_for_evaluation(dataset):
    qrels_by_query = {}

    for qrel in dataset.qrels_iter():
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)

        if hasattr(qrel, "relevance"):
            relevance = int(qrel.relevance)
        else:
            relevance = 1

        if relevance <= 0:
            continue

        if query_id not in qrels_by_query:
            qrels_by_query[query_id] = []

        qrels_by_query[query_id].append(doc_id)

    selected_query_ids = list(qrels_by_query.keys())[:TARGET_EVALUATION_QUERIES]

    required_doc_ids = set()

    for query_id in selected_query_ids:
        for doc_id in qrels_by_query[query_id]:
            required_doc_ids.add(str(doc_id))

    print(f"Selected evaluation queries: {len(selected_query_ids)}")
    print(f"Required relevant documents from qrels: {len(required_doc_ids)}")
    print("-" * 60)

    return required_doc_ids


def build_smart_documents_subset(dataset, required_doc_ids):
    selected_docs = []
    selected_doc_ids = set()

    print("Collecting documents...")

    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)

        if doc_id in required_doc_ids:
            text = get_text_from_document(doc)

            selected_docs.append({
                "doc_id": doc_id,
                "text": text
            })

            selected_doc_ids.add(doc_id)

    print(f"Relevant qrels documents found in dataset: {len(selected_docs)}")

    for doc in dataset.docs_iter():
        if len(selected_docs) >= SUBSET_SIZE:
            break

        doc_id = str(doc.doc_id)

        if doc_id in selected_doc_ids:
            continue

        text = get_text_from_document(doc)

        selected_docs.append({
            "doc_id": doc_id,
            "text": text
        })

        selected_doc_ids.add(doc_id)

    docs_df = pd.DataFrame(selected_docs)

    print(f"Final smart subset size: {len(docs_df)}")
    print("-" * 60)

    return docs_df


def build_inverted_index(docs_df):
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


def rebuild_dataset1_artifacts(docs_df):
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

    word2vec_matrix = np.array([
        document_to_vector(tokens, word2vec_model, 100)
        for tokens in tokenized_corpus
    ])

    print("Building inverted index...")

    inverted_index = build_inverted_index(docs_df)

    print("Saving Dataset 1 smart subset artifacts...")

    docs_df.to_csv(
        os.path.join(SAVE_DIR, "work_dataset1.csv"),
        index=False,
        encoding="utf-8"
    )

    save_pickle("tfidf_vectorizer_1.pkl", tfidf_vectorizer)
    save_pickle("tfidf_matrix_1.pkl", tfidf_matrix)
    save_pickle("tokenized_corpus_1.pkl", tokenized_corpus)
    save_pickle("inverted_index_1.pkl", inverted_index)

    np.save(
        os.path.join(SAVE_DIR, "word2vec_matrix_1.npy"),
        word2vec_matrix
    )

    word2vec_model.save(
        os.path.join(SAVE_DIR, "word2vec_model_dataset1.model")
    )

    print("All Dataset 1 artifacts were rebuilt successfully.")
    print("-" * 60)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    backup_old_files()

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = ir_datasets.load(DATASET_NAME)

    required_doc_ids = collect_relevant_doc_ids_for_evaluation(dataset)

    docs_df = build_smart_documents_subset(
        dataset=dataset,
        required_doc_ids=required_doc_ids
    )

    rebuild_dataset1_artifacts(docs_df)

    print("Smart subset for Dataset 1 is ready.")
    print("Now restart the FastAPI backend and run /evaluate-system again.")


if __name__ == "__main__":
    main()