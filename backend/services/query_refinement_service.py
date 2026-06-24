from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
import os

import pandas as pd

from services.preprocessing_service import normalize_text, preprocess_stemming


SPELLING_CORRECTIONS = {
# Query Refinement
# هنا نحقق تصحيح الأخطاء الشائعة في الاستعلام قبل البحث.
    "algoritm": "algorithm",
    "algorthm": "algorithm",
    "artifical": "artificial",
    "automibile": "automobile",
    "begginer": "beginner",
    "beginer": "beginner",
    "climat": "climate",
    "climite": "climate",
    "codig": "coding",
    "dabase": "database",
    "databse": "database",
    "develoment": "development",
    "developement": "development",
    "enviroment": "environment",
    "environmnt": "environment",
    "excercise": "exercise",
    "excersise": "exercise",
    "exersize": "exercise",
    "fitnes": "fitness",
    "helth": "health",
    "healt": "health",
    "infomation": "information",
    "informaton": "information",
    "inteligence": "intelligence",
    "intilligence": "intelligence",
    "lern": "learn",
    "lerning": "learning",
    "machin": "machine",
    "medecine": "medicine",
    "nutriton": "nutrition",
    "optimisation": "optimization",
    "progamming": "programming",
    "programing": "programming",
    "programmin": "programming",
    "querry": "query",
    "reccomendation": "recommendation",
    "recomendation": "recommendation",
    "retreival": "retrieval",
    "retrival": "retrieval",
    "serach": "search",
    "sofware": "software",
    "tecnology": "technology",
    "technlogy": "technology",
    "traning": "training",
    "wieght": "weight",
    "wiehgt": "weight",
    "wheight": "weight",
    "weigth": "weight",
    "weigt": "weight",
    "waight": "weight",
    "weght": "weight",
    "webiste": "website",
    "wordvec": "word2vec",
    "sothern": "southern",
    "soutern": "southern",
    "universty": "university",
    "unversity": "university",
    "methodis": "methodist",
    "methodst": "methodist",
    "justce": "justice",
    "langauge": "language",
    "languge": "language",
    "goidelc": "goidelic",
    "calclus": "calculus",
    "calcullus": "calculus",
    "cheif": "chief",
    "securty": "security",
    "histry": "history",
    "religon": "religion",
    "educaton": "education",
    "stats": "states",
}


SYNONYM_DICTIONARY = {
# Query Expansion
# هنا نحقق توسيع الاستعلام بإضافة مرادفات مناسبة.
    "ai": ["artificial intelligence", "machine learning"],
    "algorithm": ["method", "procedure", "technique"],
    "automobile": ["car", "vehicle"],
    "beginner": ["basics", "introduction", "tutorial"],
    "car": ["automobile", "vehicle"],
    "climate": ["environment", "weather"],
    "coding": ["programming", "software development"],
    "database": ["data storage", "information system"],
    "diet": ["nutrition", "healthy food"],
    "education": ["learning", "study", "training"],
    "environment": ["climate", "ecology"],
    "exercise": ["workout", "fitness", "training"],
    "fitness": ["exercise", "workout", "health"],
    "food": ["meal", "diet", "nutrition"],
    "health": ["medical", "fitness", "wellness"],
    "information": ["data", "knowledge"],
    "learn": ["study", "education", "training"],
    "learning": ["education", "study", "training"],
    "lose": ["weight loss", "diet", "fitness"],
    "machine": ["computer", "automated system"],
    "medical": ["health", "medicine", "clinical"],
    "movie": ["film", "cinema"],
    "programming": ["coding", "software", "development"],
    "query": ["search request", "search terms"],
    "recommendation": ["suggestion", "related results"],
    "retrieval": ["search", "information retrieval"],
    "search": ["retrieval", "lookup", "find"],
    "software": ["programming", "application", "development"],
    "technology": ["computing", "innovation", "software"],
    "training": ["learning", "education", "practice"],
    "weight": ["weight loss", "diet", "fitness"],
    "workout": ["exercise", "fitness", "training"],
    "university": ["college", "higher education", "academic institution"],
    "justice": ["judge", "court", "law"],
    "language": ["linguistics", "dialect", "speech"],
    "history": ["historical", "past", "chronology"],
    "religion": ["faith", "theology", "belief"],
    "music": ["song", "artist", "musician"],
    "county": ["region", "district", "area"],
}


QUERY_CATALOG = [
# Query Suggestion
# هنا نوفر اقتراحات جاهزة تساعد المستخدم أثناء صياغة الاستعلام.
    "artificial intelligence applications",
    "best diet for weight loss",
    "best exercise for fitness",
    "climate change and environment",
    "coding tutorials for beginners",
    "database design fundamentals",
    "exercise and healthy lifestyle",
    "healthy food and nutrition",
    "how to learn programming",
    "how to lose weight safely",
    "information retrieval algorithms",
    "learn machine learning basics",
    "learn python programming",
    "medical health information",
    "natural language processing",
    "programming and software development",
    "query expansion techniques",
    "query refinement and suggestion",
    "search engine optimization",
    "software development practices",
    "technology and artificial intelligence",
    "tf idf information retrieval",
    "word2vec semantic similarity",
]


DEFAULT_SEARCH_HISTORY = [
    "weight loss diet",
    "best exercise for fitness",
    "healthy food and nutrition",
    "learn programming basics",
]


CANONICAL_WORDS = sorted(
    set(SPELLING_CORRECTIONS.values())
    | set(SYNONYM_DICTIONARY.keys())
    | {
        word
        for phrase in QUERY_CATALOG
        for word in normalize_text(phrase).split()
    }
)

DATASET_QUERY_FILES = {
    "dataset1": "queries_dataset1.csv",
    "dataset2": "queries_dataset2.csv",
}


@lru_cache(maxsize=2)
def _load_dataset_query_resources(dataset):
    file_name = DATASET_QUERY_FILES.get(dataset)
    if not file_name:
        return (), ()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(backend_dir, "saved_files", file_name)
    if not os.path.exists(path):
        return (), ()

    queries_df = pd.read_csv(path, usecols=["text"])
    catalog = []
    words = set()

    for value in queries_df["text"].dropna().astype(str):
        clean_query = normalize_text(value)
        if not clean_query:
            continue
        catalog.append(clean_query)
        words.update(
            token
            for token in clean_query.split()
            if len(token) >= 3
        )

    return tuple(_unique(catalog)), tuple(sorted(words))


@lru_cache(maxsize=3)
def _get_query_catalog(dataset=None):
    dataset_catalog, _ = _load_dataset_query_resources(dataset)
    return tuple(dataset_catalog) + tuple(QUERY_CATALOG)


@lru_cache(maxsize=3)
def _get_canonical_words(dataset=None):
    _, dataset_words = _load_dataset_query_resources(dataset)
    return tuple(sorted(set(CANONICAL_WORDS).union(dataset_words)))


def _clean_query(query):
    return normalize_text(query)


def _unique(items):
    result = []

    for item in items:
        clean_item = " ".join(str(item).split())
        if clean_item and clean_item not in result:
            result.append(clean_item)

    return result


def _replace_last_token(query, replacement):
    clean_query = _clean_query(query)
    tokens = clean_query.split()

    if not tokens:
        return replacement

    tokens[-1] = replacement
    return " ".join(tokens)


def correct_query_spelling(query, dataset=None):
    # Spelling Correction
    # هنا نصحح كلمات الاستعلام اعتماداً على القاموس وكلمات الداتا سيت.
    tokens = _clean_query(query).split()
    corrected_tokens = []
    canonical_words = _get_canonical_words(dataset)
    canonical_word_set = set(canonical_words)

    for token in tokens:
        if token in SPELLING_CORRECTIONS:
            corrected_tokens.append(SPELLING_CORRECTIONS[token])
            continue

        if token in canonical_word_set or len(token) < 3:
            corrected_tokens.append(token)
            continue

        close_match = get_close_matches(
            token,
            canonical_words,
            n=1,
            cutoff=0.84,
        )
        corrected_tokens.append(close_match[0] if close_match else token)

    return " ".join(corrected_tokens)


def expand_query_with_synonyms(
    query,
    max_synonyms_per_term=2,
    dataset=None,
):
# Query Expansion
# هنا نضيف مرادفات للكلمات المصححة حتى يصبح الاستعلام أغنى.
    corrected_query = correct_query_spelling(query, dataset=dataset)
    expanded_terms = corrected_query.split()

    for token in corrected_query.split():
        for synonym in SYNONYM_DICTIONARY.get(token, [])[:max_synonyms_per_term]:
            for synonym_token in synonym.split():
                if synonym_token not in expanded_terms:
                    expanded_terms.append(synonym_token)

    return " ".join(expanded_terms)


def suggest_query_from_history(query, search_history=None, top_k=3):
    # Query Formulation Assistance
    # هنا نستفيد من سجل البحث فقط لاقتراح استعلامات مشابهة وليس لتغيير ترتيب النتائج.
    history = search_history or DEFAULT_SEARCH_HISTORY
    clean_query = _clean_query(query)
    query_tokens = set(preprocess_stemming(clean_query).split())
    suggestions = []

    for index, old_query in enumerate(history):
        clean_old_query = _clean_query(old_query)
        old_tokens = set(preprocess_stemming(clean_old_query).split())
        overlap = len(query_tokens.intersection(old_tokens))
        text_similarity = SequenceMatcher(None, clean_query, clean_old_query).ratio()
        prefix_bonus = 1.0 if clean_old_query.startswith(clean_query) else 0.0
        recency_bonus = 1 / (index + 1)
        score = overlap * 2 + text_similarity + prefix_bonus + recency_bonus

        if overlap > 0 or prefix_bonus > 0 or text_similarity >= 0.45:
            suggestions.append({
                "suggested_query": old_query,
                "similarity_score": round(score, 4),
            })

    suggestions.sort(
        key=lambda item: item["similarity_score"],
        reverse=True,
    )
    return suggestions[:top_k]


def get_prefix_completions(query, top_k=6, dataset=None):
    # Live Query Suggestion
    # هنا تظهر الاقتراحات مع كل حرف اعتماداً على بادئة الاستعلام.
    clean_query = _clean_query(query)
    query_catalog = list(_get_query_catalog(dataset))

    if not clean_query:
        return query_catalog[:top_k]

    tokens = clean_query.split()
    partial = tokens[-1]
    prefix = " ".join(tokens[:-1])
    candidates = []

    for phrase in query_catalog:
        clean_phrase = _clean_query(phrase)
        if clean_phrase.startswith(clean_query):
            candidates.append(clean_phrase)
        elif partial and any(word.startswith(partial) for word in clean_phrase.split()):
            candidates.append(clean_phrase)

    for word in _get_canonical_words(dataset):
        if word.startswith(partial) and word != partial:
            candidates.append(f"{prefix} {word}".strip())

    return _unique(candidates)[:top_k]


def refine_query(
    query,
    use_spelling=True,
    use_expansion=True,
    use_history=True,
    search_history=None,
    dataset=None,
):
# Query Refinement
# هنا نجمع التصحيح والتوسيع والاقتراحات لإنتاج استعلام محسّن.
    original_query = str(query)
    processed_original = preprocess_stemming(original_query)
    corrected_query = (
        correct_query_spelling(original_query, dataset=dataset)
        if use_spelling
        else _clean_query(original_query)
    )
    expanded_query = (
        expand_query_with_synonyms(corrected_query, dataset=dataset)
        if use_expansion
        else corrected_query
    )
    history_suggestions = (
        suggest_query_from_history(
            expanded_query,
            search_history=search_history,
            top_k=3,
        )
        if use_history
        else []
    )

    final_terms = expanded_query.split()

    if history_suggestions:
        history_terms = _clean_query(
            history_suggestions[0]["suggested_query"]
        ).split()
        for term in history_terms:
            if term not in final_terms:
                final_terms.append(term)

    return {
        "original_query": original_query,
        "processed_original_query": processed_original,
        "corrected_query": corrected_query,
        "expanded_query": expanded_query,
        "history_suggestions": history_suggestions,
        "refined_query": " ".join(final_terms),
    }


def get_query_suggestions(
    query,
    search_history=None,
    top_k=8,
    dataset=None,
):
# Query Suggestion
# هنا نعيد اقتراحات الواجهة المباشرة قبل تنفيذ البحث.
    history = search_history or []
    clean_query = _clean_query(query)
    query_catalog = list(_get_query_catalog(dataset))

    if not clean_query:
        return _unique(history + query_catalog)[:top_k]

    suggestions = []
    suggestions.extend(
        get_prefix_completions(
            clean_query,
            top_k=top_k,
            dataset=dataset,
        )
    )

    corrected_query = correct_query_spelling(
        clean_query,
        dataset=dataset,
    )
    if corrected_query != clean_query:
        suggestions.insert(0, corrected_query)

    expanded_query = expand_query_with_synonyms(
        corrected_query,
        dataset=dataset,
    )
    if expanded_query != corrected_query:
        suggestions.append(expanded_query)

    for item in suggest_query_from_history(
        clean_query,
        search_history=history,
        top_k=top_k,
    ):
        suggestions.append(item["suggested_query"])

    last_token = clean_query.split()[-1]
    if last_token in SPELLING_CORRECTIONS:
        suggestions.insert(
            0,
            _replace_last_token(
                clean_query,
                SPELLING_CORRECTIONS[last_token],
            ),
        )

    return _unique(suggestions)[:top_k]
