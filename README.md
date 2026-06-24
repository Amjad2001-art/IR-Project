# Information Retrieval Search Engine

A Python Information Retrieval system built with a service-oriented backend and a Streamlit interface.

The system works on two complete test collections:

- `wikir/en1k/test`
- `beir/quora/test`

Generated datasets, indexes, matrices, models, and report assets are not committed to GitHub because they are large and can be rebuilt with the commands below.

## Project Structure

```text
IR_Search_Engine_Project/
|-- backend/
|   |-- main.py
|   |-- streamlit_app.py
|   |-- requirements.txt
|   |-- build_full_wikir_dataset.py
|   |-- build_full_quora_dataset.py
|   |-- prepare_evaluation_files.py
|   |-- run_topic_evaluation_for_report.py
|   |-- benchmark_topic_feature_cost.py
|   |-- verify_project_requirements.py
|   |-- saved_files/
|   |   |-- .gitkeep
|   |   |-- generated cache files are created here
|   |-- services/
|   |   |-- preprocessing_service.py
|   |   |-- query_refinement_service.py
|   |   |-- retrieval_service.py
|   |   |-- indexing_service.py
|   |   |-- ranking_service.py
|   |   |-- topic_detection_service.py
|   |   |-- evaluation_service.py
|   |   |-- model_loader_service.py
|   |-- tests/
|       |-- test_core_services.py
|       |-- test_query_refinement.py
|-- notebook/
|-- .gitignore
|-- README.md
```

## Implemented Requirements

| Requirement | Implementation |
| --- | --- |
| Complete datasets | Full `wikir/en1k/test` and full `beir/quora/test` are read from the official sources. |
| Data preprocessing | Normalization, stop-word removal, and stemming. |
| TF-IDF | Implemented with `TfidfVectorizer` and cosine similarity. |
| Word2Vec | Trained per dataset and used for semantic retrieval. |
| BM25 | Implemented with the `rank-bm25` library and configurable `k1`, `b`. |
| Inverted Index | Built and cached for term matching. |
| Serial Hybrid | BM25 candidate retrieval followed by Word2Vec reranking. |
| Parallel Hybrid | Normalized BM25 and Word2Vec score fusion with `alpha`. |
| Query Refinement | Spelling correction, synonym expansion, history-based suggestion, and live prefix suggestions. |
| Topic Detection | Additional feature that detects topic terms from retrieved results and reranks candidates. |
| Evaluation | MAP, Recall, Precision@10, and nDCG before and after Topic Detection. |
| SOA | Code is split into independent services under `backend/services`. |

## Datasets

| ID | Dataset | Documents | Queries | Qrels |
| --- | --- | ---: | ---: | ---: |
| `dataset1` | `wikir/en1k/test` | 369,721 | 100 | 4,435 |
| `dataset2` | `beir/quora/test` | 522,931 | 10,000 | 15,675 |

Qrels are used for evaluation only. They are not used to select or reduce documents.

## Environment

Use Python 3.10 or newer. An Anaconda environment is also fine.

Recommended setup:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use Anaconda:

```bash
conda create -n ir-search python=3.11 -y
conda activate ir-search
cd backend
pip install -r requirements.txt
```

## Build The Complete Datasets

Run these commands once after cloning the repository.

```bash
cd backend
python build_full_wikir_dataset.py
python build_full_quora_dataset.py
python prepare_evaluation_files.py
python verify_project_requirements.py
```

These commands generate the local cache inside:

```text
backend/saved_files/
```

Generated files include:

- processed full document files
- TF-IDF vectorizers and matrices
- Word2Vec models and matrices
- tokenized corpora
- inverted indexes
- official test queries and qrels
- dataset build metadata

These files are intentionally ignored by Git.

## Run The Backend

Open a terminal:

```bash
cd backend
python -m uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run The Python Interface

Open a second terminal:

```bash
cd backend
python -m streamlit run streamlit_app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## Run Tests

```bash
cd backend
python -m unittest discover -s tests -v
```

Expected result:

```text
OK
```

## Run Topic Evaluation

This evaluates all methods before and after Topic Detection.

```bash
cd backend
python run_topic_evaluation_for_report.py
```

The output is written to:

```text
report_assets/topic_evaluation_results.json
```

`report_assets/` is ignored by Git because it contains generated report data and charts.

## Main API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Check backend status. |
| GET | `/datasets` | List loaded datasets. |
| GET | `/methods` | List retrieval methods. |
| POST | `/preprocess` | Process a query. |
| POST | `/refine-query` | Refine a query. |
| POST | `/suggest-query` | Return live query suggestions. |
| POST | `/search` | Search and rank documents. |
| POST | `/detect-topic` | Detect a topic from retrieved results. |
| POST | `/cluster-documents` | Cluster documents for topic analysis. |
| POST | `/evaluate` | Evaluate one ranked result list. |
| POST | `/evaluate-system` | Evaluate all retrieval methods before additional features. |
| POST | `/evaluate-topic-detection` | Evaluate before and after Topic Detection. |

## Example Search Request

```json
{
  "query": "learn programming",
  "dataset": "dataset2",
  "method": "bm25",
  "top_k": 10,
  "k1": 1.5,
  "b": 0.75,
  "alpha": 0.6,
  "use_refinement": false,
  "use_topic_detection": false,
  "history": []
}
```

## Notes

- The React frontend was removed. The project now uses the Python Streamlit interface.
- The additional feature kept in the project is Topic Detection only.
- Query Refinement is treated as a core requirement, not as an additional feature.
- Large generated files are excluded from GitHub and can be rebuilt using the dataset build commands.

## Author

Amjad2001-art
