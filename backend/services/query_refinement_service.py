from difflib import get_close_matches
from services.preprocessing_service import process_query, preprocess_stemming


spelling_corrections = {
    "weigth": "weight",
    "wieght": "weight",
    "wiehgt": "weight",
    "wheight": "weight",
    "weigt": "weight",
    "weig": "weight",
    "waight": "weight",

    "loose": "lose",
    "losingg": "losing",
    "los": "lose",

    "helth": "health",
    "healt": "health",

    "excersise": "exercise",
    "excercise": "exercise",
    "exersize": "exercise",

    "programing": "programming",
    "progamming": "programming",

    "lern": "learn",
    "lerning": "learning",

    "tecnology": "technology",
    "technlogy": "technology",

    "enviroment": "environment",
    "environmnt": "environment"
}


synonym_dictionary = {
    "weight": ["loss", "diet", "fitness"],
    "lose": ["weight", "loss", "diet"],
    "health": ["medical", "fitness", "wellness"],
    "exercise": ["workout", "fitness", "training"],
    "programming": ["coding", "software", "development"],
    "learn": ["study", "education", "training"],
    "car": ["automobile", "vehicle"],
    "climate": ["environment", "weather"],
    "movie": ["film", "cinema"],
    "food": ["meal", "diet", "nutrition"]
}


user_search_history = [
    "weight loss diet",
    "best exercise for fitness",
    "healthy food and nutrition",
    "learn programming basics"
]


def correct_query_spelling(query):
    query_info = process_query(query)
    tokens = query_info["tokens"]

    corrected_tokens = []

    known_words = list(spelling_corrections.keys()) + list(synonym_dictionary.keys())

    for token in tokens:
        if token in spelling_corrections:
            corrected_tokens.append(spelling_corrections[token])
        else:
            close_match = get_close_matches(
                token,
                known_words,
                n=1,
                cutoff=0.78
            )

            if close_match and close_match[0] in spelling_corrections:
                corrected_tokens.append(spelling_corrections[close_match[0]])
            elif close_match and close_match[0] in synonym_dictionary:
                corrected_tokens.append(close_match[0])
            else:
                corrected_tokens.append(token)

    return " ".join(corrected_tokens)


def expand_query_with_synonyms(query, max_synonyms_per_term=2):
    query_info = process_query(query)
    tokens = query_info["tokens"]

    expanded_tokens = []

    for token in tokens:
        expanded_tokens.append(token)

        if token in synonym_dictionary:
            related_terms = synonym_dictionary[token][:max_synonyms_per_term]

            for related_term in related_terms:
                if related_term not in expanded_tokens:
                    expanded_tokens.append(related_term)

    return " ".join(expanded_tokens)


def suggest_query_from_history(query, search_history=None, top_k=3):
    if search_history is None or len(search_history) == 0:
        search_history = user_search_history

    processed_query = preprocess_stemming(query)
    query_tokens = set(processed_query.split())

    suggestions = []

    for old_query in search_history:
        processed_old_query = preprocess_stemming(old_query)
        old_query_tokens = set(processed_old_query.split())

        common_terms = query_tokens.intersection(old_query_tokens)
        score = len(common_terms)

        if score > 0:
            suggestions.append({
                "suggested_query": old_query,
                "similarity_score": score
            })

    suggestions = sorted(
        suggestions,
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    return suggestions[:top_k]


def refine_query(query, use_spelling=True, use_expansion=True, use_history=True, search_history=None):
    original_query = query
    processed_original = process_query(original_query)["processed_query"]

    if use_spelling:
        corrected_query = correct_query_spelling(original_query)
    else:
        corrected_query = processed_original

    if use_expansion:
        expanded_query = expand_query_with_synonyms(corrected_query)
    else:
        expanded_query = corrected_query

    if use_history:
        suggestions = suggest_query_from_history(
            expanded_query,
            search_history=search_history,
            top_k=3
        )
    else:
        suggestions = []

    final_tokens = expanded_query.split()

    if len(suggestions) > 0:
        best_history_query = suggestions[0]["suggested_query"]
        history_tokens = preprocess_stemming(best_history_query).split()

        for token in history_tokens:
            if token not in final_tokens:
                final_tokens.append(token)

    refined_query = " ".join(final_tokens)

    return {
        "original_query": original_query,
        "processed_original_query": processed_original,
        "corrected_query": corrected_query,
        "expanded_query": expanded_query,
        "history_suggestions": suggestions,
        "refined_query": refined_query
    }
def get_query_suggestions(query, search_history=None, top_k=6):
    if search_history is None:
        search_history = []

    query = str(query).strip()

    # إذا مربع البحث فارغ نعرض سجل البحث فقط
    if query == "":
        recent_history = list(dict.fromkeys(search_history))
        return recent_history[:top_k]

    query_info = process_query(query)
    tokens = query_info["tokens"]

    suggestions = []

    corrected_query = correct_query_spelling(query)

    # اقتراح التصحيح الإملائي فقط إذا اختلف عن query الحالية
    if corrected_query and corrected_query != query_info["processed_query"]:
        suggestions.append(corrected_query)

    expanded_query = expand_query_with_synonyms(corrected_query)

    # اقتراح التوسيع بالمرادفات
    if expanded_query and expanded_query not in suggestions:
        suggestions.append(expanded_query)

    # استخدام سجل البحث للتثقيل فقط، وليس إظهار جمل history كما هي
    history_suggestions = suggest_query_from_history(
        expanded_query,
        search_history=search_history,
        top_k=top_k
    )

    for item in history_suggestions:
        suggested_query = item["suggested_query"]
        history_tokens = preprocess_stemming(suggested_query).split()

        weighted_tokens = expanded_query.split()

        for token in history_tokens:
            if token not in weighted_tokens:
                weighted_tokens.append(token)

        weighted_query = " ".join(weighted_tokens)

        if weighted_query not in suggestions:
            suggestions.append(weighted_query)

    # اقتراحات إضافية مبنية على المرادفات فقط
    for token in tokens:
        if token in synonym_dictionary:
            related_terms = synonym_dictionary[token]

            for related_term in related_terms:
                candidate = corrected_query + " " + related_term

                if candidate not in suggestions:
                    suggestions.append(candidate)

    clean_suggestions = []

    for suggestion in suggestions:
        suggestion = " ".join(str(suggestion).split())

        if suggestion and suggestion not in clean_suggestions:
            clean_suggestions.append(suggestion)

    return clean_suggestions[:top_k]