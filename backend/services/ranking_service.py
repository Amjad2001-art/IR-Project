def rank_dataframe_by_score(results_df, score_column, top_k=10):
    # Query Matching And Ranking
    # نرتب الوثائق حسب درجة الطريقة المختارة.
    ranked_results = results_df.sort_values(
        # نحدد عمود الدرجة الذي ستعتمد عليه عملية الترتيب.
        by=score_column,
        # نرتب تنازلياً لأن الدرجة الأكبر تعني نتيجة أفضل.
        ascending=False
    ).head(top_k).copy()

    # نضيف رقم ترتيب واضح لكل وثيقة بعد الفرز.
    ranked_results["rank"] = range(1, len(ranked_results) + 1)

    # نحدد ترتيب الأعمدة الذي نريد إرجاعه.
    columns_order = ["rank", "doc_id", score_column, "text"]

    # نأخذ فقط الأعمدة الموجودة فعلياً حتى لا يحدث خطأ إذا غاب عمود ما.
    available_columns = [
        column
        for column in columns_order
        if column in ranked_results.columns
    ]

    # نرجع جدول النتائج المرتبة بالأعمدة المطلوبة.
    return ranked_results[available_columns]


def dataframe_to_response(results_df, score_column):
    # Ranked Results
    # نحول جدول النتائج إلى قائمة قواميس موحدة ترسلها خدمات البحث للواجهة.
    response = []

    # نمر على كل صف في جدول النتائج.
    for _, row in results_df.iterrows():
        # نضيف نتيجة واحدة بالشكل الموحد المعتمد في المشروع.
        response.append({
            # رقم ترتيب الوثيقة في النتائج.
            "rank": int(row["rank"]) if row["rank"] != "-" else "-",
            # معرف الوثيقة.
            "doc_id": str(row["doc_id"]),
            # الدرجة النهائية التي حسبتها طريقة الاسترجاع.
            "score": float(row[score_column]) if row[score_column] != "-" else 0,
            # نص الوثيقة أو السؤال المعروض في الواجهة.
            "text": str(row["text"])
        })

    # نعيد القائمة النهائية للواجهة أو خدمة التقييم.
    return response
