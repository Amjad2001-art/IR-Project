import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from services.retrieval_service import run_search


def _safe_text(value):
    # نحمي الخدمة من القيم الفارغة.
    # إذا كانت القيمة غير موجودة نعيد نصاً فارغاً بدل إيقاف التنفيذ.
    if value is None:
        return ""

    # نحول أي قيمة إلى نص لأن بيانات  قد تعطي أرقاماً أو قيماً مختلفة.
    return str(value)


def _get_text_column(docs_df):
    # نحدد اسم العمود الذي يحتوي نص الوثيقة.
    # بعض الملفات تحتوي النص الخام، وبعضها يحتوي النص المعالج.
    possible_columns = [
        "text",
        "processed_text",
        "body",
        "title"
    ]

    # نبحث عن أول عمود متوفر من الأعمدة المقبولة.
    for column in possible_columns:
        if column in docs_df.columns:
            return column

    # إذا لم يوجد عمود نصي واضح نعيد خطأ مفهوم للمطور.
    raise ValueError(
        "No text column found. Expected one of: text, processed_text, body, title"
    )


def detect_topic_from_results(results, max_terms=8):
    # Topic Detection
    # هذه هي الميزة الإضافية في المشروع.
    # نحدد موضوع الاستعلام من الوثائق الراجعة لهذا الاستعلام فقط.
    # لا نستخدم كل وثائق الداتا سيت حتى لا يصبح الموضوع عاماً وغير مرتبط بالكويري.
    texts = [
        _safe_text(item.get("text", ""))
        for item in results
        if len(_safe_text(item.get("text", "")).strip()) > 0
    ]

    # إذا لم توجد نصوص في النتائج فلا يمكن استخراج موضوع.
    if len(texts) == 0:
        return {
            "topic_detection_applied": False,
            "reason": "No result texts available for topic detection",
            "detected_topic": "No clear topic",
            "top_topic_terms": [],
            "documents_analyzed": 0
        }

    # عندما تكون النتيجة وثيقة واحدة فقط نسمح بظهور الكلمات داخلها.
    if len(texts) == 1:
        max_df = 1.0
    else:
        # عندما توجد عدة وثائق نخفف أثر الكلمات العامة جداً.
        max_df = 0.95

    # TF-IDF
    # نمثل نصوص النتائج ونحسب أهمية الكلمات داخل الوثائق المرشحة.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        min_df=1,
        max_df=max_df
    )

    try:
        # نبني مصفوفة الكلمات المهمة من نصوص النتائج.
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # قد تفشل العملية إذا لم توجد كلمات مفيدة بعد إزالة الكلمات العامة.
        return {
            "topic_detection_applied": False,
            "reason": "Not enough meaningful terms after preprocessing",
            "detected_topic": "No clear topic",
            "top_topic_terms": [],
            "documents_analyzed": len(texts)
        }

    # أسماء الكلمات الموجودة في تمثيل النتائج.
    feature_names = vectorizer.get_feature_names_out()
    # نجمع درجة كل كلمة عبر جميع الوثائق المرشحة.
    # الدرجة الأعلى تعني أن الكلمة أهم في موضوع النتائج.
    term_scores = tfidf_matrix.sum(axis=0).A1

    # نرتب الكلمات تنازلياً حسب أهميتها.
    ranked_terms = sorted(
        zip(feature_names, term_scores),
        key=lambda item: item[1],
        reverse=True
    )

    # نحفظ أهم الكلمات ودرجاتها للعرض في الواجهة أو التقرير.
    top_terms = [
        {
            "term": term,
            "score": float(score)
        }
        for term, score in ranked_terms[:max_terms]
    ]

    # نستخدم أول ثلاث كلمات كعنوان مختصر للموضوع.
    detected_topic_terms = [
        item["term"]
        for item in top_terms[:3]
    ]

    if len(detected_topic_terms) == 0:
        detected_topic = "No clear topic"
    else:
        detected_topic = " / ".join(detected_topic_terms)

    # نعيد نتيجة كشف الموضوع بشكل منظم.
    return {
        "topic_detection_applied": True,
        "reason": "Topic was detected from the top retrieved documents using TF-IDF term importance",
        "detected_topic": detected_topic,
        "top_topic_terms": top_terms,
        "documents_analyzed": len(texts)
    }


def detect_topic_for_query(
    query,
    dataset,
    method,
    loaded_data,
    top_k=10,
    k1=1.5,
    b=0.75,
    alpha=0.6
):
    # Topic Detection For Query
    # هذه الدالة تنفذ البحث أولاً ثم تستخرج الموضوع من النتائج الراجعة.
    # الهدف هو عرض الموضوع للمستخدم أو تجربة الميزة على استعلام محدد.
    results = run_search(
        query=query,
        dataset=dataset,
        method=method,
        loaded_data=loaded_data,
        top_k=top_k,
        k1=k1,
        b=b,
        alpha=alpha
    )

    # نطبق كشف الموضوع على نتائج البحث الحقيقية.
    topic_info = detect_topic_from_results(
        results=results,
        max_terms=8
    )

    # نعيد الاستعلام والإعدادات والموضوع والنتائج معاً.
    return {
        "query": query,
        "dataset": dataset,
        "method": method,
        "top_k": top_k,
        "topic_info": topic_info,
        "results": results
    }


def load_documents_for_clustering(dataset, save_dir="saved_files", max_docs=1000):
    # Document Clustering
    # نجهز الوثائق التي ستستخدم في التجميع الموضوعي.
    # التجميع يساعد في شرح الموضوعات والمخططات، ولا يغير نتائج البحث المباشر.
    if dataset != "dataset1":
        raise ValueError("dataset must be dataset1 (WikIR)")

    docs_file = "work_dataset1.csv"

    # نبني مسار ملف الوثائق داخل مجلد الكاش.
    docs_path = os.path.join(save_dir, docs_file)

    # إذا لم توجد الملفات فهذا يعني أن بناء الداتا سيت لم ينفذ بعد.
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing documents file: {docs_path}")

    # نقرأ الوثائق ونثبت نوع معرف الوثيقة كنص.
    docs_df = pd.read_csv(docs_path)
    docs_df["doc_id"] = docs_df["doc_id"].astype(str)

    # نحدد عمود النص المناسب.
    text_column = _get_text_column(docs_df)

    # ننظف القيم الفارغة ونبقي فقط الوثائق التي تحتوي نصاً صالحاً.
    docs_df[text_column] = docs_df[text_column].fillna("").astype(str)
    docs_df = docs_df[docs_df[text_column].str.strip().str.len() > 0].copy()

    # نأخذ عينة محددة حتى تبقى عملية التجميع مناسبة للعرض وسريعة.
    if max_docs is not None and max_docs > 0:
        docs_df = docs_df.head(max_docs).copy()

    return docs_df, text_column


def cluster_documents(
    dataset,
    save_dir="saved_files",
    n_clusters=5,
    max_docs=1000,
    top_terms_per_cluster=8
):
    # Topic Modeling Charts Support
    # هذه الدالة تبني مجموعات موضوعية من الوثائق.
    # تستخدم لدعم شرح الموضوعات والمخططات في التقرير أو الواجهة.
    docs_df, text_column = load_documents_for_clustering(
        dataset=dataset,
        save_dir=save_dir,
        max_docs=max_docs
    )

    # إذا لم توجد وثائق صالحة نعيد نتيجة واضحة دون إيقاف الخدمة.
    if len(docs_df) == 0:
        return {
            "clustering_applied": False,
            "reason": "No documents available for clustering",
            "dataset": dataset,
            "clusters": []
        }

    # التجميع يحتاج على الأقل مجموعتين.
    if n_clusters < 2:
        n_clusters = 2

    # لا يمكن أن يكون عدد المجموعات أكبر من عدد الوثائق.
    if n_clusters > len(docs_df):
        n_clusters = len(docs_df)

    # TF-IDF
    # نمثل كل وثيقة كمتجه كلمات لاستخدامها في التجميع.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        min_df=1,
        max_df=0.95
    )

    # نبني مصفوفة تمثيل الوثائق.
    tfidf_matrix = vectorizer.fit_transform(docs_df[text_column].tolist())

    # K-Means
    # نقسم الوثائق إلى مجموعات حسب تشابه تمثيلها النصي.
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # نحصل على رقم المجموعة لكل وثيقة.
    labels = model.fit_predict(tfidf_matrix)

    # نضيف رقم المجموعة إلى جدول الوثائق لاستخدامه في استخراج الأمثلة.
    docs_df["cluster"] = labels

    # نستخرج أسماء الكلمات من المتجه.
    feature_names = vectorizer.get_feature_names_out()
    clusters = []

    for cluster_id in range(n_clusters):
        # نحدد الوثائق التي تنتمي إلى المجموعة الحالية.
        cluster_indexes = docs_df.index[docs_df["cluster"] == cluster_id].tolist()
        cluster_docs = docs_df.loc[cluster_indexes]

        # مركز المجموعة يوضح وزن كل كلمة في موضوع هذه المجموعة.
        center = model.cluster_centers_[cluster_id]
        # نأخذ أعلى الكلمات وزناً لتسمية المجموعة.
        top_term_indexes = center.argsort()[::-1][:top_terms_per_cluster]

        top_terms = [
            {
                "term": feature_names[index],
                "score": float(center[index])
            }
            for index in top_term_indexes
            if center[index] > 0
        ]

        sample_documents = []

        # نضيف أمثلة مختصرة من الوثائق داخل المجموعة لتسهيل فهمها.
        for _, row in cluster_docs.head(5).iterrows():
            sample_documents.append({
                "doc_id": str(row["doc_id"]),
                "text_preview": _safe_text(row[text_column])[:250]
            })

        # نخزن ملخص المجموعة.
        clusters.append({
            "cluster_id": int(cluster_id),
            "document_count": int(len(cluster_docs)),
            "topic_label": " / ".join([item["term"] for item in top_terms[:3]]),
            "top_terms": top_terms,
            "sample_documents": sample_documents
        })

    # النتيجة النهائية تحتوي كل المجموعات وكلماتها المهمة.
    return {
        "clustering_applied": True,
        "reason": "Documents were clustered offline using TF-IDF vectors and K-Means",
        "dataset": dataset,
        "documents_clustered": int(len(docs_df)),
        "n_clusters": int(n_clusters),
        "clusters": clusters
    }
