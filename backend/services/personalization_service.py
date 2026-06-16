from services.preprocessing_service import preprocess_stemming


def build_user_profile_terms(search_history, max_terms=30):
    term_scores = {}

    if search_history is None:
        search_history = []

    for index, query in enumerate(search_history):
        processed_query = preprocess_stemming(query)
        tokens = processed_query.split()

        recency_weight = 1 / (index + 1)

        for token in tokens:
            if token not in term_scores:
                term_scores[token] = 0

            term_scores[token] += recency_weight

    sorted_terms = sorted(
        term_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return dict(sorted_terms[:max_terms])


def personalize_results(results, search_history, boost_factor=0.20):
    user_profile_terms = build_user_profile_terms(search_history)

    if len(user_profile_terms) == 0:
        return {
            "personalization_applied": False,
            "reason": "No search history available",
            "user_profile_terms": {},
            "results": results
        }

    personalized_results = []

    for item in results:
        new_item = item.copy()

        document_text = str(new_item.get("text", ""))
        processed_document = preprocess_stemming(document_text)
        document_tokens = set(processed_document.split())

        matched_terms = []
        personalization_score = 0

        for term, term_weight in user_profile_terms.items():
            if term in document_tokens:
                matched_terms.append(term)
                personalization_score += term_weight

        original_score = float(new_item.get("score", 0))

        if personalization_score > 0:
            normalized_boost = boost_factor * personalization_score

            if original_score > 0:
                personalized_score = original_score * (1 + normalized_boost)
            else:
                personalized_score = normalized_boost
        else:
            personalized_score = original_score

        new_item["original_score"] = original_score
        new_item["score"] = personalized_score
        new_item["personalization_score"] = personalization_score
        new_item["matched_profile_terms"] = matched_terms

        personalized_results.append(new_item)

    personalized_results = sorted(
        personalized_results,
        key=lambda item: item["score"],
        reverse=True
    )

    for index, item in enumerate(personalized_results, start=1):
        item["rank"] = index

    return {
        "personalization_applied": True,
        "reason": "Results were boosted using the user's search history",
        "user_profile_terms": user_profile_terms,
        "results": personalized_results
    }
