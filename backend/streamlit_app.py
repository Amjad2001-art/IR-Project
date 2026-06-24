import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

try:
    from st_keyup import st_keyup
except ImportError:
    st_keyup = None


API_URL = os.getenv("IR_API_URL", "http://127.0.0.1:8000")
HISTORY_PATH = Path(__file__).resolve().parent / "saved_files" / "persistent_search_history.json"

DATASET_LABELS = {
# Dataset Selection
# هنا تظهر أسماء الداتا سيت التي يختارها المستخدم من الواجهة.
    "dataset1": "WikIR - Complete Test Dataset",
    "dataset2": "Quora - Complete Dataset",
}


METHODS = {
# Retrieval Methods
# هنا تظهر طرق التمثيل والاسترجاع المطلوبة في واجهة المستخدم.
    "tfidf": "TF-IDF",
    "word2vec": "Word2Vec",
    "bm25": "BM25",
    "inverted_index": "Inverted Index",
    "serial_hybrid": "Serial Hybrid",
    "parallel_hybrid": "Parallel Hybrid",
}

st.set_page_config(
    page_title="Information Retrieval Search Engine",
    page_icon="IR",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1320px;
    }

    .app-hero {
        padding: 1.35rem 1.5rem;
        border-radius: 10px;
        color: #f8fafc;
        background:
            linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(15, 76, 92, 0.92)),
            url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        margin-bottom: 1rem;
    }

    .app-hero h1 {
        font-size: 2.15rem;
        line-height: 1.15;
        margin: 0 0 .5rem 0;
        color: #ffffff;
    }

    .app-hero p {
        margin: 0;
        max-width: 850px;
        color: #dbeafe;
        font-size: 1rem;
    }

    .metric-card {
        padding: .85rem 1rem;
        border: 1px solid #d8e3ea;
        border-radius: 8px;
        background: #ffffff;
    }

    .result-card {
        border: 1px solid #d8e3ea;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: .8rem;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    }

    .result-meta {
        color: #475569;
        font-size: .88rem;
        margin-bottom: .45rem;
    }

    .small-note {
        color: #64748b;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def get_api_metadata():
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()

    datasets_response = requests.get(f"{API_URL}/datasets", timeout=5)
    datasets_response.raise_for_status()

    return {
        "health": health_response.json(),
        "datasets": datasets_response.json().get("datasets", []),
    }


def update_history(query):
    clean_query = query.strip()

    if not clean_query:
        return

    history = st.session_state.setdefault("search_history", [])
    st.session_state.search_history = [
        clean_query,
        *[item for item in history if item != clean_query],
    ][:10]
    save_persistent_history(st.session_state.search_history)


def load_persistent_history():
    if not HISTORY_PATH.exists():
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(history, list):
        return []

    return [
        str(item).strip()
        for item in history
        if str(item).strip()
    ][:10]


def save_persistent_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_history = [
        str(item).strip()
        for item in history
        if str(item).strip()
    ][:10]

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(clean_history, file, indent=2)


def get_live_suggestions(query, history, dataset):
    # Query Suggestion
    # هنا نطلب اقتراحات مباشرة مع كل حرف يكتبه المستخدم.
    response = requests.post(
        f"{API_URL}/suggest-query",
        json={
            "query": query,
            "history": history,
            "dataset": dataset,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("suggestions", [])


def run_ui_search(payload):
    response = requests.post(
        f"{API_URL}/search",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_topic_evaluation(payload):
    response = requests.post(
        f"{API_URL}/evaluate-topic-detection",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


if "search_history" not in st.session_state:
    st.session_state.search_history = load_persistent_history()
st.session_state.setdefault("query_input", "")
st.session_state.setdefault("query_widget_version", 0)

try:
    api_metadata = get_api_metadata()
except (requests.RequestException, StopIteration, ValueError) as error:
    st.error(
        "FastAPI backend is not available. Start it with: "
        "python -m uvicorn main:app --reload"
    )
    st.code(str(error))
    st.stop()

st.markdown(
    """
    <div class="app-hero">
      <h1>Information Retrieval Search Engine</h1>
      <p>
        Python interface for the complete WikIR and Quora collections with TF-IDF, Word2Vec,
        BM25, inverted index, serial hybrid, parallel hybrid, query refinement,
        and topic detection.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Search Controls")

    available_datasets = {
        item["id"]: item
        for item in api_metadata["datasets"]
        if item.get("id") in DATASET_LABELS
    }

    dataset = st.selectbox(
        # Dataset Selection
        # هنا يختار المستخدم الداتا سيت قبل تنفيذ الاستعلام.
        "Dataset",
        options=list(available_datasets.keys()),
        format_func=lambda item: DATASET_LABELS[item],
        index=1 if "dataset2" in available_datasets else 0,
    )

    method = st.selectbox(
        # Retrieval Method Selection
        # هنا يختار المستخدم طريقة التمثيل أو الاسترجاع المطلوبة.
        "Retrieval Method",
        options=list(METHODS.keys()),
        format_func=lambda item: METHODS[item],
        index=2,
    )

    top_k = st.slider("Top K Results", min_value=1, max_value=50, value=10)

    st.divider()
    st.subheader("BM25 and Hybrid")

    k1 = st.slider("BM25 k1", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    alpha = st.slider("Parallel Hybrid alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    st.divider()
    st.subheader("Query Processing")
    # Query Refinement
    # هنا تحسين الاستعلام ميزة أساسية وليست ضمن المزايا الإضافية.
    use_refinement = st.checkbox("Use Query Refinement")

    st.divider()
    st.subheader("Additional Feature")
    # Topic Detection
    # هنا تظهر الميزة الإضافية الوحيدة المطلوبة للمجموعة.
    use_topic_detection = st.checkbox("Use Topic Detection")

    st.subheader("Search History")
    if st.session_state.search_history:
        for index, history_query in enumerate(
            st.session_state.search_history,
            start=1,
        ):
            if st.button(
                history_query,
                key=f"history-query-{index}-{history_query}",
                width="stretch",
            ):
                st.session_state.query_input = history_query
                st.session_state.query_widget_version += 1
                st.rerun()
    else:
        st.caption("No searches yet.")

    if st.button("Clear Search History", width="stretch"):
        st.session_state.search_history = []
        save_persistent_history([])
        st.rerun()

    st.divider()
    st.subheader("Evaluation")

    evaluation_methods = st.multiselect(
        "Methods to Evaluate",
        options=list(METHODS.keys()),
        default=list(METHODS.keys()),
        format_func=lambda item: METHODS[item],
    )
    evaluation_queries = st.slider(
        "Evaluation Queries",
        min_value=1,
        max_value=40,
        value=10,
    )


dataset_info = available_datasets[dataset]
doc_count = int(dataset_info.get("document_count", 0))

metric_cols = st.columns(4)
metric_cols[0].metric("Selected Dataset", DATASET_LABELS[dataset])
metric_cols[1].metric("Loaded Documents", f"{doc_count:,}")
metric_cols[2].metric("Method", METHODS[method])
metric_cols[3].metric("History Items", len(st.session_state.search_history))

default_query = (
    "information retrieval"
    if dataset == "dataset1"
    else "learn programming"
)

if not st.session_state.query_input:
    st.session_state.query_input = default_query

if st_keyup:
    query = st_keyup(
        "Search Query",
        value=st.session_state.query_input,
        placeholder="Enter your query...",
        debounce=120,
        key=f"live_query_input_{st.session_state.query_widget_version}",
    )
    st.session_state.query_input = query
else:
    query = st.text_input(
        "Search Query",
        key="query_input",
        placeholder="Enter your query...",
    )

suggestion_query = query.strip()
if suggestion_query:
    try:
        suggestions = get_live_suggestions(
            suggestion_query,
            st.session_state.search_history,
            dataset,
        )
    except requests.RequestException:
        suggestions = []

    if suggestions:
        st.caption("Query suggestions")
        suggestion_cols = st.columns(min(len(suggestions), 3))
        for index, suggestion in enumerate(suggestions):
            if suggestion_cols[index % len(suggestion_cols)].button(
                suggestion,
                key=f"suggestion-{index}-{suggestion}",
                width="stretch",
            ):
                st.session_state.query_input = suggestion
                st.session_state.query_widget_version += 1
                st.rerun()

search_clicked = st.button("Search", type="primary", width="stretch")
evaluation_clicked = st.button(
    # Before And After Evaluation
    # هنا نشغل تقييم أثر كشف الموضوع قبل وبعد من الواجهة.
    "Run Before/After Topic Evaluation",
    width="stretch",
)

if search_clicked:
    # Query Matching And Ranking
    # هنا نرسل الاستعلام والمعاملات للباك ليعيد النتائج المرتبة.
    payload = {
        "query": query,
        "dataset": dataset,
        "method": method,
        "top_k": int(top_k),
        "k1": float(k1),
        "b": float(b),
        "alpha": float(alpha),
        "use_refinement": use_refinement,
        "use_topic_detection": use_topic_detection,
        "history": st.session_state.search_history,
    }

    try:
        with st.spinner("Searching through FastAPI services..."):
            response = run_ui_search(payload)
    except requests.RequestException as error:
        st.error(f"Search request failed: {error}")
    else:
        update_history(query)
        st.session_state.last_response = response

if evaluation_clicked:
    # Evaluation
    # هنا نرسل طلب التقييم الرسمي باستخدام الاستعلامات وأحكام الصلة.
    if not evaluation_methods:
        st.warning("Select at least one retrieval method for evaluation.")
    else:
        evaluation_payload = {
            "dataset": dataset,
            "methods": evaluation_methods,
            "top_k": 10,
            "max_queries": int(evaluation_queries),
            "k1": float(k1),
            "b": float(b),
            "alpha": float(alpha),
        }

        try:
            with st.spinner("Running fair before/after evaluation..."):
                evaluation_response = run_topic_evaluation(evaluation_payload)
        except requests.RequestException as error:
            st.error(f"Evaluation request failed: {error}")
        else:
            st.session_state.last_evaluation = evaluation_response


response = st.session_state.get("last_response")

if not response:
    st.info("Enter a query and press Search to display ranked results.")
else:
    st.subheader("Search Summary")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Original Query", response["original_query"])
    summary_cols[1].metric("Used Query", response["used_query"])
    summary_cols[2].metric("Returned Results", len(response["results"]))
    summary_cols[3].metric(
        "Retrieval Time",
        f"{response.get('timing', {}).get('retrieval_time_seconds', 0):.4f} s"
    )

    if response["refinement_info"]:
        with st.expander("Query Refinement Details", expanded=True):
            st.json(response["refinement_info"])

    if response["topic_info"]:
        with st.expander("Topic Detection Details", expanded=True):
            st.json(response["topic_info"])

    st.subheader("Ranked Results")

    if not response["results"]:
        st.warning("No matching records were found for this query and selected settings.")
    else:
        table_rows = [
            {
                "rank": item.get("rank"),
                "doc_id": item.get("doc_id"),
                "score": round(float(item.get("score", 0)), 6),
                "text_preview": str(item.get("text", ""))[:240],
            }
            for item in response["results"]
        ]
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        for item in response["results"]:
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-meta">
                    Rank #{item.get("rank")} | Document ID: {item.get("doc_id")}
                    | Score: {float(item.get("score", 0)):.6f}
                  </div>
                  <div>{str(item.get("text", ""))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


evaluation_response = st.session_state.get("last_evaluation")

if evaluation_response:
    st.divider()
    st.subheader("Topic Detection Evaluation")
    # Evaluation Report
    # هنا نعرض جداول قبل وبعد وفروق المقاييس والزمن في الواجهة.

    pool_cols = st.columns(4)
    pool_cols[0].metric(
        "Evaluated Queries",
        evaluation_response.get("max_queries", 0),
    )
    pool_cols[1].metric(
        "Final Top K",
        evaluation_response.get("top_k", 0),
    )
    pool_cols[2].metric(
        "Candidate Pool Before",
        evaluation_response.get("baseline_candidate_pool_size", 0),
    )
    pool_cols[3].metric(
        "Candidate Pool After",
        evaluation_response.get("topic_candidate_pool_size", 0),
    )

    before_frame = pd.DataFrame(
        evaluation_response.get("before_additional_feature", [])
    )
    after_frame = pd.DataFrame(
        evaluation_response.get("after_topic_detection", [])
    )
    difference_frame = pd.DataFrame(
        evaluation_response.get("metric_differences", [])
    )

    st.caption(
        "Timing is measured after warm-up. Retrieval uses the median of "
        "3 runs per query, and Topic Detection cost uses the median of "
        "10 isolated runs on the same candidate results."
    )

    st.markdown("#### Before Topic Detection")
    if not before_frame.empty:
        before_frame["Retrieval Time (ms)"] = (
            before_frame["average_warmed_retrieval_time_seconds"] * 1000
        ).round(3)
        st.dataframe(
            before_frame[
                [
                    "method",
                    "MAP",
                    "Recall",
                    "Precision@10",
                    "nDCG",
                    "Retrieval Time (ms)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### After Topic Detection")
    if not after_frame.empty:
        after_frame["Feature Cost (ms)"] = (
            after_frame["average_topic_feature_cost_seconds"] * 1000
        ).round(3)
        after_frame["Estimated Total (ms)"] = (
            after_frame["average_estimated_total_time_seconds"] * 1000
        ).round(3)
        after_frame["Overhead (%)"] = (
            after_frame["feature_overhead_percent"]
        ).round(2)
        st.dataframe(
            after_frame[
                [
                    "method",
                    "MAP",
                    "Recall",
                    "Precision@10",
                    "nDCG",
                    "Feature Cost (ms)",
                    "Estimated Total (ms)",
                    "Overhead (%)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Metric Differences")
    if not difference_frame.empty:
        difference_columns = [
            "method",
            "MAP_difference",
            "Recall_difference",
            "Precision@10_difference",
            "nDCG_difference",
            "topic_feature_cost_milliseconds",
            "feature_overhead_percent",
        ]
        difference_frame = difference_frame.rename(
            columns={
                "topic_feature_cost_milliseconds": "Feature Cost (ms)",
                "feature_overhead_percent": "Overhead (%)",
            }
        )
        difference_columns = [
            "method",
            "MAP_difference",
            "Recall_difference",
            "Precision@10_difference",
            "nDCG_difference",
            "Feature Cost (ms)",
            "Overhead (%)",
        ]
        difference_frame["Feature Cost (ms)"] = (
            difference_frame["Feature Cost (ms)"].round(3)
        )
        difference_frame["Overhead (%)"] = (
            difference_frame["Overhead (%)"].round(2)
        )
        st.dataframe(
            difference_frame[
                [
                    column
                    for column in difference_columns
                    if column in difference_frame.columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )
