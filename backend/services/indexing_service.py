from services.preprocessing_service import preprocess_stemming
from services.ranking_service import rank_dataframe_by_score, dataframe_to_response


# Inverted Index
# الفهرس المعكوس.
# هذا الملف يحقق طلب الفهرسة والبحث عبر قوائم الوثائق لكل كلمة.
def match_and_rank_inverted_index(query, docs_df, inverted_index, top_k=10):
    # Query Processing
    # نعالج الاستعلام بنفس معالجة الوثائق حتى تتطابق الكلمات بشكل صحيح.
    processed_query = preprocess_stemming(query)
    # نحول الاستعلام المعالج إلى كلمات منفصلة.
    query_tokens = processed_query.split()

    # نحفظ هنا درجة كل وثيقة بناء على تكرارات كلمات الاستعلام داخلها.
    document_scores = {}

    # نمر على كل كلمة في الاستعلام.
    for token in query_tokens:
        # إذا كانت الكلمة موجودة في الفهرس المعكوس نقرأ قائمة الوثائق الخاصة بها.
        if token in inverted_index:
            # Posting List
            # هذه القائمة تحتوي الوثائق التي ظهرت فيها الكلمة وعدد تكرارها.
            posting_list = inverted_index[token]

            # نمر على كل وثيقة وتكرار الكلمة داخلها.
            for doc_id, frequency in posting_list.items():
                # نحول معرف الوثيقة إلى نص لتوحيد الشكل مع بيانات الوثائق.
                doc_id = str(doc_id)

                # إذا لم تظهر الوثيقة سابقاً في النتائج نبدأ درجتها من الصفر.
                if doc_id not in document_scores:
                    document_scores[doc_id] = 0

                # نضيف تكرار الكلمة إلى درجة الوثيقة.
                document_scores[doc_id] += frequency

    # إذا لم نجد أي وثيقة مطابقة نعيد قائمة فارغة.
    if len(document_scores) == 0:
        return []

    # ننسخ جدول الوثائق حتى لا نعدل البيانات الأصلية.
    temp_docs_df = docs_df.copy()
    # نوحد نوع معرف الوثيقة حتى تتم المطابقة مع مفاتيح الدرجات.
    temp_docs_df["doc_id"] = temp_docs_df["doc_id"].astype(str)

    # نختار فقط الوثائق التي حصلت على درجة مطابقة.
    results = temp_docs_df[
        temp_docs_df["doc_id"].isin(document_scores.keys())
    ][["doc_id", "text"]].copy()

    # Matching Score
    # نضيف درجة المطابقة لكل وثيقة.
    results["matching_score"] = results["doc_id"].map(document_scores)

    # Query Matching And Ranking
    # نرتب الوثائق تنازلياً حسب درجة المطابقة ونأخذ أعلى النتائج.
    ranked_results = rank_dataframe_by_score(
        results_df=results,
        score_column="matching_score",
        top_k=top_k
    )

    # نحول النتائج إلى الشكل الموحد الذي تستخدمه الواجهة وباقي الخدمات.
    return dataframe_to_response(ranked_results, "matching_score")
