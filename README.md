# Information Retrieval Search Engine

A service-oriented Information Retrieval project that provides a searchable web interface over two datasets using multiple retrieval and ranking methods.

## Project Overview

This project implements a complete Information Retrieval search engine with:

- FastAPI backend following a Service Oriented Architecture.
- React frontend for interactive search and result exploration.
- Multiple retrieval methods: TF-IDF, Word2Vec, BM25, Inverted Index, Serial Hybrid, and Parallel Hybrid.
- Query preprocessing, query refinement, personalization, topic detection, and evaluation endpoints.
- Saved processed datasets, evaluation files, and generated ranking results.

## Repository

GitHub repository:

```text
https://github.com/Amjad2001-art/IR-Project.git
```

## Project Structure

```text
IR_Search_Engine_Project/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── prepare_evaluation_files.py
│   ├── build_smart_subset_dataset1.py
│   ├── build_smart_subset_dataset2.py
│   ├── services/
│   │   ├── indexing_service.py
│   │   ├── model_loader_service.py
│   │   ├── personalization_service.py
│   │   ├── preprocessing_service.py
│   │   ├── query_refinement_service.py
│   │   ├── ranking_service.py
│   │   ├── retrieval_service.py
│   │   └── topic_detection_service.py
│   └── saved_files/
│       ├── processed datasets
│       ├── evaluation files
│       ├── ranking result CSV files
│       └── generated model artifacts
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       └── main.jsx
├── notebook/
│   └── IR_Req4.ipynb
├── .gitignore
└── README.md
```

## Datasets

The backend supports two datasets:

| ID | Dataset |
| --- | --- |
| `dataset1` | `beir/webis-touche2020/v2` |
| `dataset2` | `beir/quora/test` |

## Retrieval Methods

The search engine supports the following methods:

| Method | Description |
| --- | --- |
| `tfidf` | Ranks documents using TF-IDF vectors and cosine similarity. |
| `word2vec` | Represents documents and queries using Word2Vec embeddings. |
| `bm25` | Uses BM25 ranking with configurable `k1` and `b`. |
| `inverted_index` | Uses an inverted index for term matching and ranking. |
| `serial_hybrid` | Retrieves candidates with BM25, then reranks with Word2Vec similarity. |
| `parallel_hybrid` | Combines normalized BM25 and Word2Vec scores using `alpha`. |

## Main Features

- Query preprocessing with token normalization and stemming.
- Query refinement with correction, expansion, and search-history suggestions.
- Search personalization based on previous user queries.
- Topic detection from ranked search results.
- Document clustering endpoint.
- Search evaluation for precision-oriented IR experiments.
- Frontend controls for dataset, ranking method, top-k, BM25 parameters, hybrid alpha, refinement, personalization, and topic detection.

## Backend Setup

From the project root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

From the project root:

```bash
cd frontend
npm install
npm run dev
```

The frontend usually runs at:

```text
http://127.0.0.1:5173
```

The frontend expects the backend to be running at:

```text
http://127.0.0.1:8000
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check backend status. |
| `GET` | `/datasets` | List available datasets. |
| `GET` | `/methods` | List retrieval methods. |
| `POST` | `/preprocess` | Preprocess a query. |
| `POST` | `/refine-query` | Refine a query. |
| `POST` | `/suggest-query` | Generate query suggestions. |
| `POST` | `/search` | Run search over the selected dataset and method. |
| `POST` | `/detect-topic` | Detect topic for a query. |
| `POST` | `/cluster-documents` | Cluster documents. |
| `POST` | `/evaluate` | Evaluate a ranked result list. |
| `POST` | `/evaluate-system` | Evaluate multiple methods. |
| `POST` | `/evaluate-personalization` | Evaluate personalization. |
| `POST` | `/evaluate-topic-detection` | Evaluate topic detection. |

## Example Search Request

```json
{
  "query": "weight loss fitness",
  "dataset": "dataset1",
  "method": "bm25",
  "top_k": 5,
  "k1": 1.5,
  "b": 0.75,
  "alpha": 0.6,
  "use_refinement": false,
  "use_personalization": false,
  "use_topic_detection": false,
  "history": []
}
```

## Saved Files Note

The `backend/saved_files/` directory contains generated datasets, ranking outputs, and model artifacts used by the backend.

Some binary/cache artifacts such as `.pkl`, `.npy`, and similar generated files are ignored by Git to keep the repository clean and avoid committing large generated files. If these files are missing after cloning, regenerate them using the dataset build scripts:

```bash
cd backend
python build_smart_subset_dataset1.py
python build_smart_subset_dataset2.py
python prepare_evaluation_files.py
```

## Technologies

- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- NLTK
- spaCy
- Gensim
- rank-bm25
- React
- Vite

## Architecture

The project separates responsibilities into independent backend services:

- `preprocessing_service`: query and document preprocessing.
- `indexing_service`: inverted index matching.
- `retrieval_service`: TF-IDF, Word2Vec, BM25, and hybrid retrieval.
- `ranking_service`: ranking utilities and response formatting.
- `query_refinement_service`: query correction, expansion, and suggestions.
- `personalization_service`: user-history-based reranking.
- `topic_detection_service`: topic analysis and clustering.
- `model_loader_service`: loading processed data and trained model artifacts.

## Author

Amjad2001-art
