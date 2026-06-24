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

app = FastAPI(
    title="Information Retrieval Search Engine",
    description="Search engine backend designed using Service Oriented Architecture",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = ModelLoaderService(save_dir="saved_files")
loaded_data = loader.load_all()


# SOA Gateway
# هنا نعرّف نماذج الطلبات التي تفصل الواجهة عن خدمات التنفيذ الداخلية.
class QueryRequest(BaseModel):
    query: str
    dataset: str = "dataset2"


class SuggestRequest(BaseModel):
    query: str = ""
    history: list[str] = []
    dataset: str = "dataset2"


class SearchRequest(BaseModel):
    query: str
    dataset: str = "dataset2"
    method: str = "bm25"
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6
    use_refinement: bool = False
    use_topic_detection: bool = False
    history: list[str] = []


class EvaluateRequest(BaseModel):
    query: str
    dataset: str = "dataset2"
    method: str = "bm25"
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6
    use_refinement: bool = False
    history: list[str] = []
    relevant_doc_ids: list[str] = []


class SystemEvaluationRequest(BaseModel):
    dataset: str = "dataset2"
    methods: list[str] = [
        "tfidf",
        "word2vec",
        "bm25",
        "inverted_index",
        "serial_hybrid",
        "parallel_hybrid"
    ]
    top_k: int = 10
    max_queries: int = 10
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class TopicEvaluationRequest(BaseModel):
    dataset: str = "dataset2"
    methods: list[str] = [
        "tfidf",
        "word2vec",
        "bm25",
        "inverted_index",
        "serial_hybrid",
        "parallel_hybrid"
    ]
    top_k: int = 10
    max_queries: int = 10
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class TopicDetectionRequest(BaseModel):
    query: str
    dataset: str = "dataset2"
    method: str = "bm25"
    top_k: int = 10
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class DocumentClusteringRequest(BaseModel):
    dataset: str = "dataset2"
    n_clusters: int = 5
    max_docs: int = 1000


@app.get("/health")
def health():
    return {
        "status": "running",
        "architecture": "Service Oriented Architecture",
        "message": "Search engine backend is working"
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="12" fill="#1f4e79"/>
<circle cx="28" cy="28" r="15" fill="none" stroke="#ffffff" stroke-width="6"/>
<line x1="40" y1="40" x2="52" y2="52" stroke="#ffffff" stroke-width="7" stroke-linecap="round"/>
</svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/datasets")
def datasets():
    # Dataset Selection
    # هنا نعرض الداتا سيت المتاحة ليختار المستخدم منها من الواجهة.
    return {
        "datasets": [
            {
                "id": "dataset1",
                "name": "wikir/en1k/test",
                "document_count": len(loaded_data["work1_df"])
            },
            {
                "id": "dataset2",
                "name": "beir/quora/test",
                "document_count": len(loaded_data["work2_df"])
            }
        ]
    }


@app.get("/methods")
def methods():
    # Retrieval Methods
    # هنا نعرض كل طرق التمثيل والاسترجاع المطلوبة في التكليف.
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
    # Preprocessing Service
    # هنا نوفر نقطة مستقلة لمعالجة الاستعلام قبل البحث.
    return process_query(request.query)


@app.post("/refine-query")
def refine_query_endpoint(request: QueryRequest):
    # Query Refinement
    # هنا نوفر تحسين الاستعلام كميزة أساسية قبل تنفيذ البحث.
    return refine_query(
        request.query,
        dataset=request.dataset,
    )


@app.post("/suggest-query")
def suggest_query_endpoint(request: SuggestRequest):
    # Query Suggestion
    # هنا نوفر اقتراحات مباشرة تظهر أثناء كتابة الاستعلام في الواجهة.
    suggestions = get_query_suggestions(
        query=request.query,
        search_history=request.history,
        dataset=request.dataset
    )

    return {
        "query": request.query,
        "suggestions": suggestions
    }


@app.post("/search")
def search_endpoint(request: SearchRequest):
    # Query Matching And Ranking
    # هنا ننفذ البحث ونرتب النتائج حسب الطريقة والمعاملات المختارة.
    request_start = time.perf_counter()
    final_query = request.query
    refinement_info = None
    topic_info = None

    if request.use_refinement:
        refinement_info = refine_query(
            query=request.query,
            search_history=request.history,
            dataset=request.dataset
        )
        final_query = refinement_info["refined_query"]

    retrieval_top_k = request.top_k

    if request.use_topic_detection:
        retrieval_top_k = max(request.top_k * 5, 50)

    retrieval_start = time.perf_counter()
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
    retrieval_time_seconds = time.perf_counter() - retrieval_start

    if request.use_topic_detection:
        # Topic Detection
        # هنا نطبق الميزة الإضافية الوحيدة بإعادة ترتيب النتائج حسب كلمات الموضوع.
        topic_output = topic_rerank_results(
            results=results,
            topic_boost_factor=0.08
        )
        results = topic_output["results"]
        topic_info = topic_output["topic_info"]

    results = results[:request.top_k]

    for index, item in enumerate(results, start=1):
        item["rank"] = index

    total_time_seconds = time.perf_counter() - request_start

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
    # Topic Modeling
    # هنا نكشف موضوع النتائج لاستعلام واحد ونرجعه للواجهة.
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
    # Document Clustering
    # هنا نوفر تجميعاً موضوعياً للوثائق لدعم شرح الموضوعات.
    return cluster_documents(
        dataset=request.dataset,
        save_dir="saved_files",
        n_clusters=request.n_clusters,
        max_docs=request.max_docs
    )


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    # Evaluation
    # هنا نقيم نتائج استعلام واحد عند توفر وثائق الصلة.
    final_query = request.query
    refinement_info = None

    if request.use_refinement:
        refinement_info = refine_query(
            query=request.query,
            search_history=request.history,
            dataset=request.dataset
        )
        final_query = refinement_info["refined_query"]

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

    evaluation = evaluate_ranked_results(
        results=results,
        relevant_doc_ids=request.relevant_doc_ids,
        top_k=request.top_k
    )

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
    # Baseline Evaluation
    # هنا نقيم الطرق الأساسية قبل تفعيل الميزة الإضافية.
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

    return evaluation_result


@app.post("/evaluate-topic-detection")
def evaluate_topic_detection_endpoint(request: TopicEvaluationRequest):
    # Before And After Topic Detection
    # هنا نقيم كل الطرق قبل وبعد ميزة كشف الموضوع كما طلبت المعيدة.
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

    return evaluation_result
