from services.preprocessing_service import preprocess_stemming
from services.ranking_service import rank_dataframe_by_score, dataframe_to_response


# Inverted Index
# هنا نحقق البحث عبر الفهرس المعكوس وجمع تكرارات كلمات الاستعلام داخل الوثائق.
def match_and_rank_inverted_index(query, docs_df, inverted_index, top_k=10):
    processed_query = preprocess_stemming(query)
    query_tokens = processed_query.split()

    document_scores = {}

    for token in query_tokens:
        if token in inverted_index:
            posting_list = inverted_index[token]

            for doc_id, frequency in posting_list.items():
                doc_id = str(doc_id)

                if doc_id not in document_scores:
                    document_scores[doc_id] = 0

                document_scores[doc_id] += frequency

    if len(document_scores) == 0:
        return []

    temp_docs_df = docs_df.copy()
    temp_docs_df["doc_id"] = temp_docs_df["doc_id"].astype(str)

    results = temp_docs_df[
        temp_docs_df["doc_id"].isin(document_scores.keys())
    ][["doc_id", "text"]].copy()

    results["matching_score"] = results["doc_id"].map(document_scores)

    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="matching_score",
        top_k=top_k
    )

    return dataframe_to_response(ranked_results, "matching_score")
