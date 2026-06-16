import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing_service import preprocess_stemming
from services.ranking_service import rank_dataframe_by_score, dataframe_to_response
from services.indexing_service import match_and_rank_inverted_index


def document_to_vector(tokens, model, vector_size=100):
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


def min_max_normalize(scores):
    scores = np.array(scores)

    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score == 0:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def search_tfidf(query, docs_df, vectorizer, tfidf_matrix, top_k=10):
    processed_query = preprocess_stemming(query)
    query_vector = vectorizer.transform([processed_query])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = docs_df[["doc_id", "text"]].copy()
    results["tfidf_score"] = scores

    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="tfidf_score",
        top_k=top_k
    )

    return dataframe_to_response(ranked_results, "tfidf_score")


def search_word2vec(query, docs_df, word2vec_model, word2vec_matrix, top_k=10):
    processed_query = preprocess_stemming(query)
    query_tokens = processed_query.split()

    query_vector = document_to_vector(
        query_tokens,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    scores = cosine_similarity(query_vector, word2vec_matrix).flatten()

    results = docs_df[["doc_id", "text"]].copy()
    results["word2vec_score"] = scores

    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="word2vec_score",
        top_k=top_k
    )

    return dataframe_to_response(ranked_results, "word2vec_score")


def search_bm25(query, docs_df, tokenized_corpus, top_k=10, k1=1.5, b=0.75):
    processed_query = preprocess_stemming(query)
    tokenized_query = processed_query.split()

    bm25_model = BM25Okapi(
        tokenized_corpus,
        k1=k1,
        b=b
    )

    scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )

    results = docs_df[["doc_id", "text"]].copy()
    results["bm25_score"] = scores

    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="bm25_score",
        top_k=top_k
    )

    return dataframe_to_response(ranked_results, "bm25_score")


def serial_hybrid_search(
    query,
    docs_df,
    tokenized_corpus,
    word2vec_model,
    word2vec_matrix,
    top_k=10,
    candidate_k=100,
    k1=1.5,
    b=0.75
):
    processed_query = preprocess_stemming(query)
    tokenized_query = processed_query.split()

    bm25_model = BM25Okapi(
        tokenized_corpus,
        k1=k1,
        b=b
    )

    bm25_scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )

    candidate_k = min(candidate_k, len(docs_df))
    candidate_indices = bm25_scores.argsort()[::-1][:candidate_k]

    query_vector = document_to_vector(
        tokenized_query,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    candidate_embeddings = word2vec_matrix[candidate_indices]

    embedding_scores = cosine_similarity(
        query_vector,
        candidate_embeddings
    ).flatten()

    reranked_positions = embedding_scores.argsort()[::-1][:top_k]
    final_indices = candidate_indices[reranked_positions]

    results = docs_df.iloc[final_indices][["doc_id", "text"]].copy()
    results["hybrid_score"] = embedding_scores[reranked_positions]
    results["rank"] = range(1, len(results) + 1)

    response = []

    for _, row in results.iterrows():
        response.append({
            "rank": int(row["rank"]),
            "doc_id": str(row["doc_id"]),
            "score": float(row["hybrid_score"]),
            "text": str(row["text"])
        })

    return response


def parallel_hybrid_search(
    query,
    docs_df,
    tokenized_corpus,
    word2vec_model,
    word2vec_matrix,
    top_k=10,
    alpha=0.6,
    k1=1.5,
    b=0.75
):
    processed_query = preprocess_stemming(query)
    tokenized_query = processed_query.split()

    bm25_model = BM25Okapi(
        tokenized_corpus,
        k1=k1,
        b=b
    )

    bm25_scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )

    query_vector = document_to_vector(
        tokenized_query,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    word2vec_scores = cosine_similarity(
        query_vector,
        word2vec_matrix
    ).flatten()

    bm25_norm = min_max_normalize(bm25_scores)
    word2vec_norm = min_max_normalize(word2vec_scores)

    final_scores = alpha * bm25_norm + (1 - alpha) * word2vec_norm

    results = docs_df[["doc_id", "text"]].copy()
    results["final_hybrid_score"] = final_scores

    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="final_hybrid_score",
        top_k=top_k
    )

    return dataframe_to_response(ranked_results, "final_hybrid_score")


def run_search(
    query,
    dataset,
    method,
    loaded_data,
    top_k=10,
    k1=1.5,
    b=0.75,
    alpha=0.6
):
    if dataset == "dataset1":
        docs_df = loaded_data["work1_df"]
        vectorizer = loaded_data["tfidf_vectorizer_1"]
        tfidf_matrix = loaded_data["tfidf_matrix_1"]
        word2vec_model = loaded_data["word2vec_model_1"]
        word2vec_matrix = loaded_data["word2vec_matrix_1"]
        tokenized_corpus = loaded_data["tokenized_corpus_1"]
        inverted_index = loaded_data["inverted_index_1"]
    else:
        docs_df = loaded_data["work2_df"]
        vectorizer = loaded_data["tfidf_vectorizer_2"]
        tfidf_matrix = loaded_data["tfidf_matrix_2"]
        word2vec_model = loaded_data["word2vec_model_2"]
        word2vec_matrix = loaded_data["word2vec_matrix_2"]
        tokenized_corpus = loaded_data["tokenized_corpus_2"]
        inverted_index = loaded_data["inverted_index_2"]

    if method == "tfidf":
        return search_tfidf(query, docs_df, vectorizer, tfidf_matrix, top_k)

    if method == "word2vec":
        return search_word2vec(query, docs_df, word2vec_model, word2vec_matrix, top_k)

    if method == "bm25":
        return search_bm25(query, docs_df, tokenized_corpus, top_k, k1, b)

    if method == "inverted_index":
        return match_and_rank_inverted_index(query, docs_df, inverted_index, top_k)

    if method == "serial_hybrid":
        return serial_hybrid_search(
            query=query,
            docs_df=docs_df,
            tokenized_corpus=tokenized_corpus,
            word2vec_model=word2vec_model,
            word2vec_matrix=word2vec_matrix,
            top_k=top_k,
            candidate_k=100,
            k1=k1,
            b=b
        )

    if method == "parallel_hybrid":
        return parallel_hybrid_search(
            query=query,
            docs_df=docs_df,
            tokenized_corpus=tokenized_corpus,
            word2vec_model=word2vec_model,
            word2vec_matrix=word2vec_matrix,
            top_k=top_k,
            alpha=alpha,
            k1=k1,
            b=b
        )

    return []