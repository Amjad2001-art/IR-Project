from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing_service import normalize_text, preprocess_stemming


# Spelling Correction
# قاموس التصحيح يحتوي أخطاء شائعة وكلماتها الصحيحة.
# يستخدم قبل البحث حتى لا تضيع النتائج بسبب خطأ كتابي بسيط.
SPELLING_CORRECTIONS = {
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


# Query Expansion
# قاموس المرادفات يضيف كلمات قريبة من معنى الاستعلام.
# الهدف زيادة فرصة مطابقة وثائق مفيدة لا تحتوي نفس كلمة المستخدم حرفياً.
SYNONYM_DICTIONARY = {
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


# Query Suggestion
# كتالوج اقتراحات عام يستخدم عندما لا تكون ملفات استعلامات الداتا متاحة.
# لا يستبدل الاستعلامات الرسمية، بل يساعد المستخدم أثناء الكتابة.
QUERY_CATALOG = [
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


# سجل افتراضي بسيط يستخدم إذا لم يرسل المستخدم سجل بحث من الواجهة.
DEFAULT_SEARCH_HISTORY = [
    "weight loss diet",
    "best exercise for fitness",
    "healthy food and nutrition",
    "learn programming basics",
]


# نجمع الكلمات الصحيحة من قاموس التصحيح وقاموس المرادفات وكتالوج الاستعلامات.
# هذا يساعد التصحيح الإملائي على اختيار كلمات معروفة بدل تخمين عشوائي.
CANONICAL_WORDS = sorted(
    set(SPELLING_CORRECTIONS.values())
    | set(SYNONYM_DICTIONARY.keys())
    | {
        word
        for phrase in QUERY_CATALOG
        for word in normalize_text(phrase).split()
    }
)


# أسماء ملفات الاستعلامات الرسمية لكل داتا سيت.
# هذه الملفات تستخدم للاقتراحات والتصحيح، وليس لتغيير ملفات التقييم المخزنة.
DATASET_QUERY_FILES = {
    "dataset1": "queries_dataset1.csv",
    "dataset2": "queries_dataset2.csv",
}


@lru_cache(maxsize=2)
def _load_dataset_query_resources(dataset):
    # نحمل استعلامات الداتا سيت الرسمية من ملفات الكاش.
    # استخدام الكاش هنا يمنع قراءة الملف من القرص مع كل حرف يكتبه المستخدم.
    file_name = DATASET_QUERY_FILES.get(dataset)
    if not file_name:
        return (), ()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(backend_dir, "saved_files", file_name)
    if not os.path.exists(path):
        return (), ()

    # نحتاج عمود النص فقط لتقليل الذاكرة والوقت.
    queries_df = pd.read_csv(path, usecols=["text"])
    catalog = []
    words = set()

    for value in queries_df["text"].dropna().astype(str):
        # نوحد شكل الاستعلام قبل استخدامه في الاقتراحات.
        clean_query = normalize_text(value)
        if not clean_query:
            continue

        # نخزن الاستعلام الكامل للاقتراحات.
        catalog.append(clean_query)
        # ونخزن الكلمات المفردة للتصحيح الإملائي.
        words.update(
            token
            for token in clean_query.split()
            if len(token) >= 3
        )

    return tuple(_unique(catalog)), tuple(sorted(words))


@lru_cache(maxsize=3)
def _get_query_catalog(dataset=None):
    # ندمج استعلامات الداتا سيت مع الكتالوج العام.
    # هذا يجعل الاقتراحات مرتبطة بسياق الداتا المختارة.
    dataset_catalog, _ = _load_dataset_query_resources(dataset)
    return tuple(dataset_catalog) + tuple(QUERY_CATALOG)


@lru_cache(maxsize=3)
def _get_canonical_words(dataset=None):
    # ندمج كلمات الداتا سيت مع الكلمات العامة المعروفة.
    # هذا يحسن التصحيح الإملائي للداتا الأولى والثانية.
    _, dataset_words = _load_dataset_query_resources(dataset)
    return tuple(sorted(set(CANONICAL_WORDS).union(dataset_words)))


def _clean_query(query):
    # نوحد شكل الاستعلام بنفس معالجة النص الأساسية.
    return normalize_text(query)


def _unique(items):
    # نحذف التكرارات مع الحفاظ على ترتيب أول ظهور.
    result = []

    for item in items:
        clean_item = " ".join(str(item).split())
        if clean_item and clean_item not in result:
            result.append(clean_item)

    return result


def _replace_last_token(query, replacement):
    # تستخدم عندما تكون آخر كلمة يكتبها المستخدم خطأ.
    # نستبدل آخر كلمة فقط ونترك باقي الاستعلام كما هو.
    clean_query = _clean_query(query)
    tokens = clean_query.split()

    if not tokens:
        return replacement

    tokens[-1] = replacement
    return " ".join(tokens)


def correct_query_spelling(query, dataset=None):
    # Spelling Correction
    # نصحح كلمات الاستعلام قبل البحث.
    # نعتمد أولاً على القاموس اليدوي، ثم على كلمات الداتا سيت المشابهة.
    tokens = _clean_query(query).split()
    corrected_tokens = []
    canonical_words = _get_canonical_words(dataset)
    canonical_word_set = set(canonical_words)

    for token in tokens:
        # إذا كان الخطأ معروفاً نستخدم التصحيح المباشر.
        if token in SPELLING_CORRECTIONS:
            corrected_tokens.append(SPELLING_CORRECTIONS[token])
            continue

        # إذا كانت الكلمة صحيحة أو قصيرة جداً نتركها كما هي.
        if token in canonical_word_set or len(token) < 3:
            corrected_tokens.append(token)
            continue

        # نحاول إيجاد أقرب كلمة معروفة تشبه الكلمة الحالية.
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
    # نضيف مرادفات مناسبة بعد التصحيح الإملائي.
    # هذا يجعل البحث أوسع ويزيد فرصة العثور على وثائق مرتبطة.
    corrected_query = correct_query_spelling(query, dataset=dataset)
    expanded_terms = corrected_query.split()

    for token in corrected_query.split():
        # لكل كلمة نأخذ عدداً محدوداً من المرادفات حتى لا يصبح الاستعلام ضجيجياً.
        for synonym in SYNONYM_DICTIONARY.get(token, [])[:max_synonyms_per_term]:
            for synonym_token in synonym.split():
                if synonym_token not in expanded_terms:
                    expanded_terms.append(synonym_token)

    return " ".join(expanded_terms)


def suggest_query_from_history(query, search_history=None, top_k=3):
    # Query Formulation Assistance
    # نقترح استعلامات مشابهة من سجل البحث.
    # هذا لا يغير ترتيب النتائج مباشرة، بل يساعد المستخدم على صياغة الكويري.
    history = search_history or DEFAULT_SEARCH_HISTORY
    clean_query = _clean_query(query)
    query_tokens = set(preprocess_stemming(clean_query).split())
    suggestions = []

    for index, old_query in enumerate(history):
        # ننظف الاستعلام القديم ونمثله بجذور الكلمات للمقارنة.
        clean_old_query = _clean_query(old_query)
        old_tokens = set(preprocess_stemming(clean_old_query).split())
        # تقاطع الكلمات يعطي إشارة تشابه دلالية بسيطة.
        overlap = len(query_tokens.intersection(old_tokens))
        # التشابه النصي يعطي إشارة على قرب صياغة الجملتين.
        text_similarity = SequenceMatcher(None, clean_query, clean_old_query).ratio()
        # إذا كان الاستعلام القديم يبدأ بما كتبه المستخدم نعطيه أولوية.
        prefix_bonus = 1.0 if clean_old_query.startswith(clean_query) else 0.0
        # الاستعلامات الأحدث في السجل تحصل على وزن أعلى.
        recency_bonus = 1 / (index + 1)
        score = overlap * 2 + text_similarity + prefix_bonus + recency_bonus

        if overlap > 0 or prefix_bonus > 0 or text_similarity >= 0.45:
            suggestions.append({
                "suggested_query": old_query,
                "similarity_score": round(score, 4),
            })

    # نرتب الاقتراحات من الأعلى تشابهاً إلى الأقل.
    suggestions.sort(
        key=lambda item: item["similarity_score"],
        reverse=True,
    )
    return suggestions[:top_k]


def suggest_query_by_ir_similarity(
    query,
    search_history=None,
    top_k=5,
    dataset=None,
):
    # IR Query Similarity
    # هذه ليست مطابقة نصية بسيطة فقط.
    # نمثل الاستعلام الحالي والاستعلامات المرشحة بتمثيل متجهي ثم نقيس التشابه.
    clean_query = _clean_query(query)

    if not clean_query:
        return []

    # المرشحون يأتون من سجل المستخدم ومن استعلامات الداتا سيت.
    candidates = _unique(
        list(search_history or [])
        + list(_get_query_catalog(dataset))
    )
    # لا نقترح نفس الاستعلام الذي كتبه المستخدم.
    candidates = [
        item
        for item in candidates
        if _clean_query(item) != clean_query
    ]

    if not candidates:
        return []

    # نعالج الاستعلام والمرشحين بنفس طريقة المعالجة حتى تكون المقارنة عادلة.
    processed_query = preprocess_stemming(clean_query)
    processed_candidates = [
        preprocess_stemming(item)
        for item in candidates
    ]
    valid_pairs = [
        (original, processed)
        for original, processed in zip(candidates, processed_candidates)
        if processed
    ]

    if not processed_query or not valid_pairs:
        return []

    originals, processed_texts = zip(*valid_pairs)

    try:
        # TF-IDF
        # نبني تمثيلاً متجهياً للاستعلام الحالي وكل الاستعلامات المرشحة.
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(
            [processed_query, *processed_texts]
        )
        # Cosine Similarity
        # نقيس قرب كل استعلام مرشح من الاستعلام الحالي.
        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:],
        ).flatten()
    except ValueError:
        return []

    # نرتب الاقتراحات حسب درجة التشابه.
    ranked = sorted(
        zip(originals, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "suggested_query": original,
            "similarity_score": round(float(score), 4),
        }
        for original, score in ranked[:top_k]
        if score > 0
    ]


def get_prefix_completions(query, top_k=6, dataset=None):
    # Live Query Suggestion
    # تعطي اقتراحات أثناء الكتابة مع كل حرف.
    # تعتمد على بادئة الاستعلام وعلى كلمات الداتا سيت المعروفة.
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
        # إذا كان الاستعلام الكامل بداية لعبارة محفوظة نعرضها مباشرة.
        if clean_phrase.startswith(clean_query):
            candidates.append(clean_phrase)
        # إذا كانت آخر كلمة جزئية تطابق بداية كلمة داخل اقتراح نعرضه أيضاً.
        elif partial and any(word.startswith(partial) for word in clean_phrase.split()):
            candidates.append(clean_phrase)

    for word in _get_canonical_words(dataset):
        # نقترح إكمال آخر كلمة بناءً على كلمات الداتا والقاموس.
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
    # هذه الدالة تجمع مراحل تحسين الاستعلام الأساسية.
    # التصحيح والتوسيع يعملان قبل البحث، لذلك الاستعلام المستخدم فعلياً يكون محسناً.
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

    # نبدأ بالكلمات الناتجة عن التصحيح والتوسيع.
    final_terms = expanded_query.split()

    if history_suggestions:
        # إذا وجد اقتراح قوي من السجل نضيف كلماته غير الموجودة.
        # هذا يثقل الاستعلام بمعلومة من سجل البحث دون حذف كلمات المستخدم.
        history_terms = _clean_query(
            history_suggestions[0]["suggested_query"]
        ).split()
        for term in history_terms:
            if term not in final_terms:
                final_terms.append(term)

    # نعيد كل المراحل حتى تظهر في الواجهة والتقرير.
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
    # هذه الدالة مسؤولة عن اقتراحات الواجهة المباشرة قبل تنفيذ البحث.
    # تجمع اقتراحات البادئة والتصحيح والتوسيع والسجل والتشابه المتجهي.
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

    # نقترح النسخة المصححة إذا كان هناك خطأ إملائي.
    corrected_query = correct_query_spelling(
        clean_query,
        dataset=dataset,
    )
    if corrected_query != clean_query:
        suggestions.insert(0, corrected_query)

    # نقترح النسخة الموسعة بالمرادفات إذا أضافت كلمات جديدة.
    expanded_query = expand_query_with_synonyms(
        corrected_query,
        dataset=dataset,
    )
    if expanded_query != corrected_query:
        suggestions.append(expanded_query)

    # نضيف اقتراحات من سجل البحث.
    for item in suggest_query_from_history(
        clean_query,
        search_history=history,
        top_k=top_k,
    ):
        suggestions.append(item["suggested_query"])

    # نضيف اقتراحات مبنية على قواعد استرجاع معلومات وتشابه متجهي.
    for item in suggest_query_by_ir_similarity(
        clean_query,
        search_history=history,
        top_k=top_k,
        dataset=dataset,
    ):
        suggestions.append(item["suggested_query"])

    # إذا كانت آخر كلمة خطأ معروفاً نضع تصحيحها في مقدمة الاقتراحات.
    last_token = clean_query.split()[-1]
    if last_token in SPELLING_CORRECTIONS:
        suggestions.insert(
            0,
            _replace_last_token(
                clean_query,
                SPELLING_CORRECTIONS[last_token],
            ),
        )

    # نحذف التكرارات ونرجع أعلى عدد مطلوب.
    return _unique(suggestions)[:top_k]
