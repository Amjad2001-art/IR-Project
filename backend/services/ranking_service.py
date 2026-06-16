def rank_dataframe_by_score(results_df, score_column, top_k=10):
    ranked_results = results_df.sort_values(
        by=score_column,
        ascending=False
    ).head(top_k).copy()

    ranked_results["rank"] = range(1, len(ranked_results) + 1)

    columns_order = ["rank", "doc_id", score_column, "text"]

    available_columns = [
        column
        for column in columns_order
        if column in ranked_results.columns
    ]

    return ranked_results[available_columns]


def dataframe_to_response(results_df, score_column):
    response = []

    for _, row in results_df.iterrows():
        response.append({
            "rank": int(row["rank"]) if row["rank"] != "-" else "-",
            "doc_id": str(row["doc_id"]),
            "score": float(row[score_column]) if row[score_column] != "-" else 0,
            "text": str(row["text"])
        })

    return response