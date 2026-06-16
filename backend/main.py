from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.model_loader_service import ModelLoaderService
from services.preprocessing_service import process_query
from services.query_refinement_service import refine_query, get_query_suggestions
from services.retrieval_service import run_search
from services.evaluation_service import (
    evaluate_ranked_results,
    evaluate_all_methods,
    evaluate_all_methods_with_personalization,
    evaluate_all_methods_with_topic_detection
)
from services.personalization_service import personalize_results
from services.topic_detection_service import (
    detect_topic_from_results,
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


class QueryRequest(BaseModel):
    query: str


class SuggestRequest(BaseModel):
    query: str = ""
    history: list[str] = []


class SearchRequest(BaseModel):
    query: str
    dataset: str = "dataset1"
    method: str = "bm25"
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6
    use_refinement: bool = False
    use_personalization: bool = False
    use_topic_detection: bool = False
    history: list[str] = []


class EvaluateRequest(BaseModel):
    query: str
    dataset: str = "dataset1"
    method: str = "bm25"
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6
    use_refinement: bool = False
    use_personalization: bool = False
    history: list[str] = []
    relevant_doc_ids: list[str] = []


class SystemEvaluationRequest(BaseModel):
    dataset: str = "dataset1"
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


class PersonalizationEvaluationRequest(BaseModel):
    dataset: str = "dataset1"
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
    history: list[str] = [
        "weight loss fitness",
        "healthy diet food",
        "exercise workout body"
    ]



class TopicEvaluationRequest(BaseModel):
    dataset: str = "dataset1"
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
    dataset: str = "dataset1"
    method: str = "bm25"
    top_k: int = 10
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.6


class DocumentClusteringRequest(BaseModel):
    dataset: str = "dataset1"
    n_clusters: int = 5
    max_docs: int = 1000


@app.get("/health")
def health():
    return {
        "status": "running",
        "architecture": "Service Oriented Architecture",
        "message": "Search engine backend is working"
    }


@app.get("/datasets")
def datasets():
    return {
        "datasets": [
            {"id": "dataset1", "name": "beir/webis-touche2020/v2"},
            {"id": "dataset2", "name": "beir/quora/test"}
        ]
    }


@app.get("/methods")
def methods():
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
    return process_query(request.query)


@app.post("/refine-query")
def refine_query_endpoint(request: QueryRequest):
    return refine_query(request.query)


@app.post("/suggest-query")
def suggest_query_endpoint(request: SuggestRequest):
    suggestions = get_query_suggestions(
        query=request.query,
        search_history=request.history
    )

    return {
        "query": request.query,
        "suggestions": suggestions
    }


@app.post("/search")
def search_endpoint(request: SearchRequest):
    final_query = request.query
    refinement_info = None
    personalization_info = None
    topic_info = None

    if request.use_refinement:
        refinement_info = refine_query(
            query=request.query,
            search_history=request.history
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

    if request.use_personalization:
        personalization_output = personalize_results(
            results=results,
            search_history=request.history
        )

        results = personalization_output["results"]

        personalization_info = {
            "personalization_applied": personalization_output["personalization_applied"],
            "reason": personalization_output["reason"],
            "user_profile_terms": personalization_output["user_profile_terms"]
        }

    if request.use_topic_detection:
        topic_info = detect_topic_from_results(
            results=results,
            max_terms=8
        )

    return {
        "original_query": request.query,
        "used_query": final_query,
        "refinement_info": refinement_info,
        "personalization_info": personalization_info,
        "topic_info": topic_info,
        "dataset": request.dataset,
        "method": request.method,
        "top_k": request.top_k,
        "history_used": request.history,
        "use_personalization": request.use_personalization,
        "use_topic_detection": request.use_topic_detection,
        "results": results
    }


@app.post("/detect-topic")
def detect_topic_endpoint(request: TopicDetectionRequest):
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
    return cluster_documents(
        dataset=request.dataset,
        save_dir="saved_files",
        n_clusters=request.n_clusters,
        max_docs=request.max_docs
    )


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    final_query = request.query
    refinement_info = None
    personalization_info = None

    if request.use_refinement:
        refinement_info = refine_query(
            query=request.query,
            search_history=request.history
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

    if request.use_personalization:
        personalization_output = personalize_results(
            results=results,
            search_history=request.history
        )

        results = personalization_output["results"]

        personalization_info = {
            "personalization_applied": personalization_output["personalization_applied"],
            "reason": personalization_output["reason"],
            "user_profile_terms": personalization_output["user_profile_terms"]
        }

    evaluation = evaluate_ranked_results(
        results=results,
        relevant_doc_ids=request.relevant_doc_ids,
        top_k=request.top_k
    )

    return {
        "original_query": request.query,
        "used_query": final_query,
        "refinement_info": refinement_info,
        "personalization_info": personalization_info,
        "dataset": request.dataset,
        "method": request.method,
        "top_k": request.top_k,
        "history_used": request.history,
        "use_personalization": request.use_personalization,
        "relevant_doc_ids": request.relevant_doc_ids,
        "evaluation": evaluation,
        "results": results
    }


@app.post("/evaluate-system")
def evaluate_system_endpoint(request: SystemEvaluationRequest):
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


@app.post("/evaluate-personalization")
def evaluate_personalization_endpoint(request: PersonalizationEvaluationRequest):
    evaluation_result = evaluate_all_methods_with_personalization(
        dataset=request.dataset,
        methods=request.methods,
        loaded_data=loaded_data,
        top_k=request.top_k,
        max_queries=request.max_queries,
        k1=request.k1,
        b=request.b,
        alpha=request.alpha,
        search_history=request.history,
        save_dir="saved_files"
    )

    return evaluation_result



@app.post("/evaluate-topic-detection")
def evaluate_topic_detection_endpoint(request: TopicEvaluationRequest):
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
