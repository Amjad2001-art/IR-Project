import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing_service import preprocess_stemming
from services.ranking_service import rank_dataframe_by_score, dataframe_to_response
from services.indexing_service import match_and_rank_inverted_index


# BM25 Cache
# نخزن النماذج في الذاكرة حتى لا نعيد بناءها مع كل عملية بحث.
_bm25_cache = {}


# BM25
# k1
# b
# هنا نستخدم مكتبة جاهزة ونخزن النموذج حسب المعاملات لتسريع البحث.
def get_bm25_model(tokenized_corpus, k1=1.5, b=0.75):
    # المفتاح يميز بين نفس الكوربس ونفس معاملات النموذج.
    cache_key = (id(tokenized_corpus), float(k1), float(b))

    # إذا لم يكن النموذج مبنياً سابقاً، نبنيه ونحفظه في الكاش.
    if cache_key not in _bm25_cache:
        _bm25_cache[cache_key] = BM25Okapi(
            tokenized_corpus,
            k1=k1,
            b=b
        )

    # نعيد النموذج الجاهز للاستخدام في حساب درجات الوثائق.
    return _bm25_cache[cache_key]


# Embedding Representation
# هنا نحول الوثيقة أو الاستعلام إلى متجه دلالي متوسط من متجهات الكلمات.
def document_to_vector(tokens, model, vector_size=100):
    # Word2Vec
    # نخزن متجهات الكلمات الموجودة داخل النموذج.
    vectors = []

    # نمر على كل كلمة ونأخذ متجهها إذا كانت موجودة في النموذج.
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    # إذا لم نجد أي كلمة داخل النموذج، نعيد متجهاً صفرياً بنفس الحجم.
    if len(vectors) == 0:
        return np.zeros(vector_size)

    # نمثل الوثيقة أو الاستعلام بمتوسط متجهات كلماته.
    return np.mean(vectors, axis=0)


def min_max_normalize(scores):
    # نحول الدرجات إلى أرقام صالحة ونستبدل القيم غير الصالحة بالصفر.
    scores = np.nan_to_num(
        np.array(scores, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # نأخذ أصغر وأكبر قيمة لاستخدامهما في التطبيع.
    min_score = scores.min()
    max_score = scores.max()

    # إذا كانت كل الدرجات متساوية، لا يوجد فرق للترتيب فنرجع أصفاراً.
    if max_score - min_score == 0:
        return np.zeros_like(scores)

    # نطبع الدرجات إلى المجال من صفر إلى واحد.
    return (scores - min_score) / (max_score - min_score)


# TF-IDF
# VSM_TF-IDF
# هنا نحقق تمثيل الوثائق والاستعلام باستخدام المتجهات الإحصائية.
def search_tfidf(query, docs_df, vectorizer, tfidf_matrix, top_k=10):
    # نعالج الاستعلام بنفس طريقة معالجة الوثائق.
    processed_query = preprocess_stemming(query)
    # TF-IDF
    # Vectorizer
    # نحول الاستعلام إلى متجه باستخدام نفس الأداة الخاصة بالداتا.
    query_vector = vectorizer.transform([processed_query])

    # TF-IDF Matrix
    # نحسب التشابه بين متجه الاستعلام ومصفوفة الوثائق.
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # نجهز إطار نتائج يحتوي معرف الوثيقة والنص.
    results = docs_df[["doc_id", "text"]].copy()
    # TF-IDF Score
    # نضيف الدرجة لكل وثيقة.
    results["tfidf_score"] = scores

    # نرتب الوثائق تنازلياً حسب الدرجة ونأخذ أعلى النتائج.
    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="tfidf_score",
        top_k=top_k
    )

    # نحول النتائج إلى شكل موحد ترسله الواجهة.
    return dataframe_to_response(ranked_results, "tfidf_score")


# Word2Vec
# Embedding
# هنا نحقق التمثيل الدلالي للوثائق والاستعلام.
def search_word2vec(query, docs_df, word2vec_model, word2vec_matrix, top_k=10):
    # نعالج الاستعلام ثم نقسمه إلى كلمات.
    processed_query = preprocess_stemming(query)
    query_tokens = processed_query.split()

    # نحول الاستعلام إلى متجه دلالي بنفس حجم متجهات الوثائق.
    query_vector = document_to_vector(
        query_tokens,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    # نحسب التشابه الدلالي بين الاستعلام وكل وثيقة.
    scores = cosine_similarity(query_vector, word2vec_matrix).flatten()

    # Word2Vec Score
    # نجهز النتائج ونضيف الدرجة الدلالية.
    results = docs_df[["doc_id", "text"]].copy()
    results["word2vec_score"] = scores

    # نرتب الوثائق حسب التشابه الدلالي.
    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="word2vec_score",
        top_k=top_k
    )

    # نعيد النتائج بشكل موحد.
    return dataframe_to_response(ranked_results, "word2vec_score")


# BM25
# Probabilistic Model
# هنا نحقق النموذج الاحتمالي مع دعم تغيير المعاملات من الواجهة.
def search_bm25(query, docs_df, tokenized_corpus, top_k=10, k1=1.5, b=0.75):
    # نعالج الاستعلام بنفس طريقة الوثائق.
    processed_query = preprocess_stemming(query)
    # Tokens
    # نقسم الاستعلام إلى كلمات لأن النموذج يعمل على الكلمات المنفصلة.
    tokenized_query = processed_query.split()

    # BM25
    # نحصل على النموذج الجاهز أو نبنيه إذا لم يكن موجوداً في الكاش.
    bm25_model = get_bm25_model(
        tokenized_corpus=tokenized_corpus,
        k1=k1,
        b=b
    )

    # BM25 Score
    # نحسب الدرجة لكل وثيقة في الداتا.
    scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )
    # نعالج أي قيم غير صالحة حتى لا تؤثر على الترتيب.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # نضيف الدرجات إلى الوثائق.
    results = docs_df[["doc_id", "text"]].copy()
    results["bm25_score"] = scores

    # BM25 Score
    # نرتب الوثائق حسب الدرجة.
    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="bm25_score",
        top_k=top_k
    )

    # نعيد النتائج للواجهة بشكل موحد.
    return dataframe_to_response(ranked_results, "bm25_score")


# Serial Hybrid Representation
# هنا نطبق التمثيل الهجين التسلسلي باختيار مرشحين أولاً ثم إعادة ترتيبهم دلالياً.
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
    # نعالج الاستعلام ونقسمه إلى كلمات.
    processed_query = preprocess_stemming(query)
    tokenized_query = processed_query.split()

    # BM25
    # نبني أو نحصل على النموذج لاختيار المرشحين الأوائل.
    bm25_model = get_bm25_model(
        tokenized_corpus=tokenized_corpus,
        k1=k1,
        b=b
    )

    # BM25 Scores
    # نحسب درجات كل الوثائق.
    bm25_scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )
    # نعالج القيم غير الصالحة قبل اختيار المرشحين.
    bm25_scores = np.nan_to_num(
        bm25_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # لا نسمح أن يتجاوز عدد المرشحين عدد الوثائق.
    candidate_k = min(candidate_k, len(docs_df))
    # BM25 Candidates
    # نختار أعلى المرشحين حسب الدرجة النصية.
    candidate_indices = bm25_scores.argsort()[::-1][:candidate_k]

    # Word2Vec
    # نحول الاستعلام إلى متجه لإعادة ترتيب المرشحين دلالياً.
    query_vector = document_to_vector(
        tokenized_query,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    # نأخذ متجهات الوثائق المرشحة فقط.
    candidate_embeddings = word2vec_matrix[candidate_indices]

    # نحسب التشابه الدلالي بين الاستعلام والمرشحين.
    embedding_scores = cosine_similarity(
        query_vector,
        candidate_embeddings
    ).flatten()

    # نعيد ترتيب المرشحين حسب التشابه الدلالي ونأخذ أعلى النتائج.
    reranked_positions = embedding_scores.argsort()[::-1][:top_k]
    final_indices = candidate_indices[reranked_positions]

    # نبني جدول النتائج النهائي.
    results = docs_df.iloc[final_indices][["doc_id", "text"]].copy()
    results["hybrid_score"] = embedding_scores[reranked_positions]
    results["rank"] = range(1, len(results) + 1)

    # نحول النتائج إلى قائمة قواميس موحدة.
    response = []

    # نمر على النتائج ونجهزها للإرجاع.
    for _, row in results.iterrows():
        response.append({
            "rank": int(row["rank"]),
            "doc_id": str(row["doc_id"]),
            "score": float(row["hybrid_score"]),
            "text": str(row["text"])
        })

    # نعيد النتائج بعد الهجين التسلسلي.
    return response


# Parallel Hybrid Representation
# هنا نطبق التمثيل الهجين المتوازي بدمج درجة النموذج النصي مع الدرجة الدلالية.
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
    # نعالج الاستعلام ونقسمه إلى كلمات.
    processed_query = preprocess_stemming(query)
    tokenized_query = processed_query.split()

    # BM25
    # نحصل على النموذج لحساب الدرجة النصية.
    bm25_model = get_bm25_model(
        tokenized_corpus=tokenized_corpus,
        k1=k1,
        b=b
    )

    # BM25 Scores
    # نحسب درجات كل الوثائق.
    bm25_scores = np.array(
        bm25_model.get_scores(tokenized_query)
    )
    # نعالج القيم غير الصالحة.
    bm25_scores = np.nan_to_num(
        bm25_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # نحول الاستعلام إلى متجه دلالي.
    query_vector = document_to_vector(
        tokenized_query,
        word2vec_model,
        vector_size=word2vec_matrix.shape[1]
    ).reshape(1, -1)

    # نحسب درجات التشابه الدلالي مع كل الوثائق.
    word2vec_scores = cosine_similarity(
        query_vector,
        word2vec_matrix
    ).flatten()

    # BM25 Scores
    # Word2Vec Scores
    # نطبع الدرجات حتى يمكن دمجها بعدل.
    bm25_norm = min_max_normalize(bm25_scores)
    # Word2Vec Scores
    # نطبع الدرجات إلى نفس المجال.
    word2vec_norm = min_max_normalize(word2vec_scores)

    # Alpha
    # ندمج الدرجتين باستخدام معامل الدمج.
    final_scores = alpha * bm25_norm + (1 - alpha) * word2vec_norm

    # نضيف الدرجة النهائية إلى الوثائق.
    results = docs_df[["doc_id", "text"]].copy()
    results["final_hybrid_score"] = final_scores

    # نرتب الوثائق حسب الدرجة المدمجة.
    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="final_hybrid_score",
        top_k=top_k
    )

    # نعيد النتائج بشكل موحد.
    return dataframe_to_response(ranked_results, "final_hybrid_score")


# Retrieval Service
# هنا نوجه البحث إلى طريقة التمثيل المختارة مع نفس معاملات المستخدم.
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
    # نحدد لاحقة الملفات حسب الداتا سيت المختارة.
    if dataset != "dataset1":
        raise ValueError("dataset must be dataset1 (WikIR)")

    suffix = "1"

    # نقرأ وثائق الداتا المختارة من الذاكرة.
    docs_df = loaded_data[f"work{suffix}_df"]
    # TF-IDF
    # نقرأ المكونات المحفوظة للداتا المختارة.
    vectorizer = loaded_data[f"tfidf_vectorizer_{suffix}"]
    tfidf_matrix = loaded_data[f"tfidf_matrix_{suffix}"]
    # Word2Vec
    # نقرأ النموذج ومصفوفة متجهات الوثائق.
    word2vec_model = loaded_data[f"word2vec_model_{suffix}"]
    word2vec_matrix = loaded_data[f"word2vec_matrix_{suffix}"]
    # Tokenized Corpus
    # BM25
    # نقرأ الكوربس المقسم إلى كلمات لاستخدامه في النموذج.
    tokenized_corpus = loaded_data[f"tokenized_corpus_{suffix}"]
    # Inverted Index
    # نقرأ الفهرس المعكوس لاستخدام طريقة الفهرسة.
    inverted_index = loaded_data[f"inverted_index_{suffix}"]

    # TF-IDF
    # VSM_TF-IDF
    # إذا اختار المستخدم هذه الطريقة نستخدم التمثيل الإحصائي.
    if method == "tfidf":
        return search_tfidf(query, docs_df, vectorizer, tfidf_matrix, top_k)

    # Word2Vec
    # إذا اختار المستخدم هذه الطريقة نستخدم التمثيل الدلالي.
    if method == "word2vec":
        return search_word2vec(query, docs_df, word2vec_model, word2vec_matrix, top_k)

    # BM25
    # k1
    # b
    # إذا اختار المستخدم هذه الطريقة نستخدم النموذج الاحتمالي مع معاملاته.
    if method == "bm25":
        return search_bm25(query, docs_df, tokenized_corpus, top_k, k1, b)

    # إذا اختار المستخدم الفهرس المعكوس نستخدم خدمة الفهرسة.
    if method == "inverted_index":
        return match_and_rank_inverted_index(query, docs_df, inverted_index, top_k)

    # إذا اختار المستخدم الهجين التسلسلي نرشح أولاً ثم نعيد الترتيب دلالياً.
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

    # BM25
    # Word2Vec
    # إذا اختار المستخدم الهجين المتوازي ندمج درجات النموذجين.
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

    # إذا كانت الطريقة غير معروفة نعيد قائمة فارغة.
    return []
