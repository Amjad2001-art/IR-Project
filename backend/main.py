import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from services.model_loader_service import ModelLoaderService
from services.preprocessing_service import process_query
from services.query_refinement_service import refine_query, get_query_suggestions
from services.retrieval_service import run_search
from services.evaluation_service import (
    evaluate_ranked_results,
    evaluate_all_methods,
    evaluate_all_methods_with_topic_detection,
    topic_rerank_results
)
from services.topic_detection_service import (
    detect_topic_for_query,
    cluster_documents
)


# ننشئ تطبيق الباك الأساسي الذي تستدعيه الواجهة وكل نقاط الوصول.
app = FastAPI(
    title="Information Retrieval Search Engine",
    description="Search engine backend designed using Service Oriented Architecture",
    version="1.0.0"
)


# نسمح للواجهة بالاتصال مع الباك حتى لو كانت تعمل من منفذ مختلف.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# نحمل كل ملفات الكاش والنماذج مرة واحدة عند تشغيل الباك.
# هذا يجعل البحث أسرع لأننا لا نعيد قراءة الملفات مع كل طلب.
loader = ModelLoaderService(save_dir="saved_files")
loaded_data = loader.load_all()


# نماذج الطلبات التالية تحدد شكل البيانات القادمة من الواجهة.
# وجودها يفصل الواجهة عن الخدمات الداخلية ويحقق تنظيم البنية الخدمية.
class QueryRequest(BaseModel):
    # نص الاستعلام الذي يرسله المستخدم.
    query: str

    dataset: str = "dataset1"


class SuggestRequest(BaseModel):
    # النص الحالي أثناء كتابة المستخدم للاستعلام.
    query: str = ""
    # سجل البحث يساعد في اقتراح استعلامات قريبة من سياق المستخدم.
    history: list[str] = []
    # نستخدم الداتا المختارة حتى تكون الاقتراحات مناسبة لمفرداتها.
    dataset: str = "dataset1"


class SearchRequest(BaseModel):
    # الاستعلام الأصلي القادم من الواجهة.
    query: str
    
    dataset: str = "dataset1"
    # طريقة الاسترجاع المطلوبة.
    method: str = "bm25"
    # عدد النتائج النهائية التي يريدها المستخدم.
    top_k: int = 5
    # معامل النموذج الاحتمالي الأول.
    k1: float = 1.5
    # معامل ضبط طول الوثيقة.
    b: float = 0.75
    # معامل دمج الدرجات في التمثيل الهجين المتوازي.
    alpha: float = 0.6
    # تحسين الاستعلام أساسي ومفعل دائماً حتى لو بقي الحقل للتوافق مع الطلبات.
    use_refinement: bool = True
    # هذا الخيار يشغل الميزة الإضافية الخاصة بكشف الموضوع.
    use_topic_detection: bool = False
    # سجل البحث يستخدم في تحسين الاستعلام والاقتراحات.
    history: list[str] = []


class EvaluateRequest(BaseModel):
    # استعلام واحد نريد تقييم نتائجه عند وجود وثائق صلة.
    query: str
    # الداتا سيت التي سيتم التقييم عليها.
    dataset: str = "dataset1"
    # طريقة الاسترجاع التي سيتم تقييمها.
    method: str = "bm25"
    # عدد النتائج الداخلة في حساب المقاييس.
    top_k: int = 5
    # معاملات البحث نفسها حتى يكون التقييم مطابقاً للتجربة من الواجهة.
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6
    # تحسين الاستعلام أساسي ومفعل دائماً.
    use_refinement: bool = True
    # سجل البحث اختياري وقد يدخل في تحسين الاستعلام.
    history: list[str] = []
    # قائمة الوثائق الصحيحة المستخدمة في تقييم استعلام واحد.
    relevant_doc_ids: list[str] = []


class SystemEvaluationRequest(BaseModel):
    # الداتا التي سيجرى عليها تقييم جميع الطرق.
    dataset: str = "dataset1"
    # الطرق الرسمية المطلوب تقييمها في المشروع.
    methods: list[str] = [
        "tfidf",
        "word2vec",
        "bm25",
        "inverted_index",
        "serial_hybrid",
        "parallel_hybrid"
    ]
    # عدد النتائج النهائي في كل استعلام.
    top_k: int = 10
    # عدد استعلامات الاختبار الرسمية المستخدمة في التقييم.
    max_queries: int = 10
    # معاملات البحث التي تطبق أثناء التقييم.
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class TopicEvaluationRequest(BaseModel):
    # الداتا التي سيتم عليها تقييم قبل وبعد الميزة الإضافية.
    dataset: str = "dataset1"
    # جميع طرق الاسترجاع التي نقارنها قبل وبعد كشف الموضوع.
    methods: list[str] = [
        "tfidf",
        "word2vec",
        "bm25",
        "inverted_index",
        "serial_hybrid",
        "parallel_hybrid"
    ]
    # عدد النتائج النهائية في حساب المقاييس.
    top_k: int = 10
    # عدد استعلامات الاختبار الرسمية التي تدخل في التقرير.
    max_queries: int = 10
    # معاملات البحث التي تستخدم قبل وبعد حتى تكون المقارنة عادلة.
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class TopicDetectionRequest(BaseModel):
    # استعلام واحد لاختبار كشف الموضوع على نتائجه.
    query: str
    # الداتا التي يتم البحث فيها.
    dataset: str = "dataset1"
    # طريقة الاسترجاع المستخدمة لجلب النتائج الأولية.
    method: str = "bm25"
    # عدد النتائج المراد تحليل موضوعها.
    top_k: int = 10
    # معاملات البحث.
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class DocumentClusteringRequest(BaseModel):
    # الداتا التي سيتم تجميع وثائقها موضوعياً.
    dataset: str = "dataset1"
    # عدد التجمعات الموضوعية المطلوبة.
    n_clusters: int = 5
    # حد أعلى للوثائق المستخدمة حتى تبقى العملية عملية وسريعة.
    max_docs: int = 1000


@app.get("/health")
def health():
    # نقطة فحص بسيطة للتأكد أن الباك يعمل.
    return {
        "status": "running",
        "architecture": "Service Oriented Architecture",
        "message": "Search engine backend is working"
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # نرجع أيقونة صغيرة حتى لا يظهر خطأ عند طلب المتصفح للرمز.
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="12" fill="#1f4e79"/>
<circle cx="28" cy="28" r="15" fill="none" stroke="#ffffff" stroke-width="6"/>
<line x1="40" y1="40" x2="52" y2="52" stroke="#ffffff" stroke-width="7" stroke-linecap="round"/>
</svg>"""
    # نحدد نوع الاستجابة حتى يفهم المتصفح أنها صورة.
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/datasets")
def datasets():
    # نعرض الداتا سيت المتاحة للواجهة مع عدد الوثائق المحملة فعلياً.
    return {
        "datasets": [
            {
                "id": "dataset1",
                "name": "wikir/en1k/test",
                "document_count": len(loaded_data["work1_df"])
            }
        ]
    }


@app.get("/methods")
def methods():
    # نرجع طرق الاسترجاع والتمثيل التي يختار بينها المستخدم من الواجهة.
    return {
        "methods": [
            "tfidf",
            "word2vec",
            "bm25",
            "inverted_index",
            "serial_hybrid",
            "parallel_hybrid"
        ]
    }


@app.post("/preprocess")
def preprocess_endpoint(request: QueryRequest):
    # هذه النقطة تعرض نتيجة معالجة الاستعلام قبل البحث للشرح والاختبار.
    return process_query(request.query)


@app.post("/refine-query")
def refine_query_endpoint(request: QueryRequest):
    # هذه النقطة تطبق تحسين الاستعلام كخدمة مستقلة.
    return refine_query(
        request.query,
        dataset=request.dataset,
    )


@app.post("/suggest-query")
def suggest_query_endpoint(request: SuggestRequest):
    # نولد اقتراحات مباشرة أثناء الكتابة اعتماداً على الداتا سيت وسجل البحث.
    suggestions = get_query_suggestions(
        query=request.query,
        search_history=request.history,
        dataset=request.dataset
    )

    # نرجع النص الأصلي مع قائمة الاقتراحات للواجهة.
    return {
        "query": request.query,
        "suggestions": suggestions
    }


@app.post("/search")
def search_endpoint(request: SearchRequest):
    # هذه هي نقطة البحث الرئيسية التي تستخدمها الواجهة عند الضغط على زر البحث.
    request_start = time.perf_counter()

    # نحتفظ بالاستعلام الأصلي للعرض في الواجهة.
    final_query = request.query

    # نطبق تحسين الاستعلام دائماً لأنه طلب أساسي وليس ميزة إضافية.
    refinement_info = refine_query(
        query=request.query,
        search_history=request.history,
        dataset=request.dataset
    )

    # الاستعلام المستخدم فعلياً في البحث هو النسخة المحسنة.
    final_query = refinement_info["refined_query"]

    # هذه القيمة تبقى فارغة إذا لم يشغل المستخدم ميزة كشف الموضوع.
    topic_info = None

    # نبدأ بعدد النتائج المطلوب من المستخدم.
    retrieval_top_k = request.top_k

    # عند تفعيل كشف الموضوع نجلب مرشحين أكثر حتى نستطيع إعادة ترتيبهم.
    if request.use_topic_detection:
        retrieval_top_k = max(request.top_k * 5, 50)

    # نقيس زمن الاسترجاع الأساسي فقط.
    retrieval_start = time.perf_counter()

    # نستدعي خدمة الاسترجاع الموحدة التي تختار الطريقة المطلوبة داخلياً.
    results = run_search(
        query=final_query,
        dataset=request.dataset,
        method=request.method,
        loaded_data=loaded_data,
        top_k=retrieval_top_k,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha
    )

    # نحسب زمن الاسترجاع قبل أي إعادة ترتيب موضوعية.
    retrieval_time_seconds = time.perf_counter() - retrieval_start

    # إذا فعل المستخدم الميزة الإضافية نعيد ترتيب النتائج حسب كلمات الموضوع.
    if request.use_topic_detection:
        topic_output = topic_rerank_results(
            results=results,
            topic_boost_factor=0.08
        )
        # نستبدل النتائج بالترتيب الجديد بعد تطبيق تعزيز الموضوع.
        results = topic_output["results"]
        # نحتفظ بمعلومات الموضوع لعرضها في الواجهة.
        topic_info = topic_output["topic_info"]

    # بعد أي إعادة ترتيب نعيد فقط عدد النتائج النهائي المطلوب.
    results = results[:request.top_k]

    # نضيف رقم ترتيب واضح لكل نتيجة قبل إرسالها للواجهة.
    for index, item in enumerate(results, start=1):
        item["rank"] = index

    # نحسب الزمن الكلي للطلب كاملاً.
    total_time_seconds = time.perf_counter() - request_start

    # نرجع كل المعلومات التي تحتاجها الواجهة والتقرير والشرح.
    return {
        "original_query": request.query,
        "used_query": final_query,
        "refinement_info": refinement_info,
        "topic_info": topic_info,
        "dataset": request.dataset,
        "method": request.method,
        "top_k": request.top_k,
        "candidate_pool_size": retrieval_top_k,
        "history_used": request.history,
        "use_topic_detection": request.use_topic_detection,
        "timing": {
            "retrieval_time_seconds": retrieval_time_seconds,
            "total_time_seconds": total_time_seconds
        },
        "results": results
    }


@app.post("/detect-topic")
def detect_topic_endpoint(request: TopicDetectionRequest):
    # هذه نقطة مستقلة لاختبار كشف الموضوع على استعلام واحد.
    return detect_topic_for_query(
        query=request.query,
        dataset=request.dataset,
        method=request.method,
        loaded_data=loaded_data,
        top_k=request.top_k,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha
    )


@app.post("/cluster-documents")
def cluster_documents_endpoint(request: DocumentClusteringRequest):
    # هذه نقطة مساعدة لتجميع الوثائق موضوعياً عند شرح الميزة.
    return cluster_documents(
        dataset=request.dataset,
        save_dir="saved_files",
        n_clusters=request.n_clusters,
        max_docs=request.max_docs
    )


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    # هذه نقطة تقييم لاستعلام واحد عند إدخال وثائق الصلة يدوياً.
    refinement_info = refine_query(
        query=request.query,
        search_history=request.history,
        dataset=request.dataset
    )

    # نستخدم الاستعلام المحسن في التقييم أيضاً لأنه جزء أساسي من النظام.
    final_query = refinement_info["refined_query"]

    # نجلب النتائج بالطريقة المطلوبة.
    results = run_search(
        query=final_query,
        dataset=request.dataset,
        method=request.method,
        loaded_data=loaded_data,
        top_k=request.top_k,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha
    )

    # نحسب مقاييس التقييم للاستعلام الواحد اعتماداً على وثائق الصلة.
    evaluation = evaluate_ranked_results(
        results=results,
        relevant_doc_ids=request.relevant_doc_ids,
        top_k=request.top_k
    )

    # نرجع النتائج مع المقاييس لتسهيل الفحص من الواجهة البرمجية.
    return {
        "original_query": request.query,
        "used_query": final_query,
        "refinement_info": refinement_info,
        "dataset": request.dataset,
        "method": request.method,
        "top_k": request.top_k,
        "history_used": request.history,
        "relevant_doc_ids": request.relevant_doc_ids,
        "evaluation": evaluation,
        "results": results
    }


@app.post("/evaluate-system")
def evaluate_system_endpoint(request: SystemEvaluationRequest):
    # هذه النقطة تقيم الطرق الأساسية بدون تشغيل الميزة الإضافية.
    evaluation_result = evaluate_all_methods(
        dataset=request.dataset,
        methods=request.methods,
        loaded_data=loaded_data,
        top_k=request.top_k,
        max_queries=request.max_queries,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha,
        save_dir="saved_files"
    )

    # نرجع ملخص التقييم للطرق المختارة.
    return evaluation_result


@app.post("/evaluate-topic-detection")
def evaluate_topic_detection_endpoint(request: TopicEvaluationRequest):
    # هذه النقطة هي الأهم للتقرير لأنها تقارن قبل وبعد الميزة الإضافية.
    # تحسين الاستعلام مطبق في الحالتين، لذلك الفرق هو أثر كشف الموضوع فقط.
    evaluation_result = evaluate_all_methods_with_topic_detection(
        dataset=request.dataset,
        methods=request.methods,
        loaded_data=loaded_data,
        top_k=request.top_k,
        max_queries=request.max_queries,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha,
        save_dir="saved_files"
    )

    # نرجع جداول قبل وبعد وفروق المقاييس وكلفة الميزة.
    return evaluation_result
