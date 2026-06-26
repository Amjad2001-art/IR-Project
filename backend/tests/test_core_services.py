import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from services.evaluation_service import (
    average_precision_at_k,
    evaluate_single_method_with_topic_cost,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    topic_rerank_results,
)
from services.indexing_service import match_and_rank_inverted_index
from services.preprocessing_service import normalize_text, preprocess_stemming
from services.ranking_service import dataframe_to_response, rank_dataframe_by_score
from services.retrieval_service import (
    parallel_hybrid_search,
    search_bm25,
    search_tfidf,
    search_word2vec,
    serial_hybrid_search,
)
from services.topic_detection_service import detect_topic_from_results


class FakeWordVectors:
    def __init__(self, vectors):
        self.vectors = vectors

    def __contains__(self, token):
        return token in self.vectors

    def __getitem__(self, token):
        return self.vectors[token]


class FakeWord2Vec:
    def __init__(self, vectors):
        self.wv = FakeWordVectors(vectors)


class CoreServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = pd.DataFrame(
            [
                {"doc_id": "1", "text": "learn python programming"},
                {"doc_id": "2", "text": "healthy diet and fitness"},
                {"doc_id": "3", "text": "software coding tutorial"},
            ]
        )
        cls.processed = [
            preprocess_stemming(text)
            for text in cls.docs["text"].tolist()
        ]
        cls.tokens = [text.split() for text in cls.processed]

        cls.vectorizer = TfidfVectorizer()
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.processed)

        cls.word2vec_model = FakeWord2Vec(
            {
                "learn": np.array([1.0, 0.0], dtype=np.float32),
                "python": np.array([1.0, 0.0], dtype=np.float32),
                "program": np.array([1.0, 0.0], dtype=np.float32),
                "softwar": np.array([0.9, 0.1], dtype=np.float32),
                "code": np.array([0.9, 0.1], dtype=np.float32),
                "tutori": np.array([0.8, 0.2], dtype=np.float32),
                "healthi": np.array([0.0, 1.0], dtype=np.float32),
                "diet": np.array([0.0, 1.0], dtype=np.float32),
                "fit": np.array([0.0, 1.0], dtype=np.float32),
            }
        )

        def vector_for(tokens):
            vectors = [
                cls.word2vec_model.wv[token]
                for token in tokens
                if token in cls.word2vec_model.wv
            ]
            return np.mean(vectors, axis=0) if vectors else np.zeros(2)

        cls.word2vec_matrix = np.array(
            [vector_for(tokens) for tokens in cls.tokens],
            dtype=np.float32,
        )

    def test_preprocessing_service(self):
        self.assertEqual(
            normalize_text("Python!!! https://example.com"),
            "python",
        )
        self.assertEqual(preprocess_stemming("Learning Python"), "learn python")

    def test_ranking_service(self):
        frame = self.docs.copy()
        frame["score"] = [0.2, 0.9, 0.5]
        ranked = rank_dataframe_by_score(frame, "score", top_k=2)
        response = dataframe_to_response(ranked, "score")
        self.assertEqual([item["doc_id"] for item in response], ["2", "3"])
        self.assertEqual([item["rank"] for item in response], [1, 2])

    def test_inverted_index_service(self):
        index = {
            "python": {"1": 2},
            "program": {"1": 1, "3": 1},
        }
        results = match_and_rank_inverted_index(
            "python programming",
            self.docs,
            index,
            top_k=3,
        )
        self.assertEqual(results[0]["doc_id"], "1")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_tfidf_retrieval(self):
        results = search_tfidf(
            "python programming",
            self.docs,
            self.vectorizer,
            self.tfidf_matrix,
            top_k=2,
        )
        self.assertEqual(results[0]["doc_id"], "1")

    def test_word2vec_retrieval(self):
        results = search_word2vec(
            "python programming",
            self.docs,
            self.word2vec_model,
            self.word2vec_matrix,
            top_k=2,
        )
        self.assertEqual(results[0]["doc_id"], "1")

    def test_bm25_retrieval(self):
        results = search_bm25(
            "python programming",
            self.docs,
            self.tokens,
            top_k=2,
        )
        self.assertEqual(results[0]["doc_id"], "1")

    def test_serial_hybrid_retrieval(self):
        results = serial_hybrid_search(
            "python programming",
            self.docs,
            self.tokens,
            self.word2vec_model,
            self.word2vec_matrix,
            top_k=2,
            candidate_k=3,
        )
        self.assertEqual(results[0]["doc_id"], "1")

    def test_parallel_hybrid_retrieval(self):
        results = parallel_hybrid_search(
            "python programming",
            self.docs,
            self.tokens,
            self.word2vec_model,
            self.word2vec_matrix,
            top_k=2,
            alpha=0.6,
        )
        self.assertEqual(results[0]["doc_id"], "1")

    def test_topic_detection_service(self):
        results = [
            {"text": "python programming software development"},
            {"text": "python coding and software engineering"},
        ]
        topic = detect_topic_from_results(results, max_terms=5)
        self.assertTrue(topic["topic_detection_applied"])
        self.assertEqual(topic["documents_analyzed"], 2)
        self.assertTrue(topic["top_topic_terms"])

    def test_topic_reranking_changes_result_metadata(self):
        results = [
            {
                "rank": 1,
                "doc_id": "1",
                "score": 1.0,
                "text": "python programming software development",
            },
            {
                "rank": 2,
                "doc_id": "2",
                "score": 0.95,
                "text": "healthy diet and fitness",
            },
            {
                "rank": 3,
                "doc_id": "3",
                "score": 0.9,
                "text": "python coding software tutorial",
            },
        ]
        output = topic_rerank_results(results, topic_boost_factor=0.08)
        self.assertTrue(output["topic_detection_applied"])
        self.assertIn("original_score", output["results"][0])
        self.assertIn("topic_score", output["results"][0])
        self.assertIn("matched_topic_terms", output["results"][0])

    def test_evaluation_metrics(self):
        retrieved = ["1", "2", "3"]
        relevant = {"1", "3"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 1.0)
        self.assertGreater(average_precision_at_k(retrieved, relevant, k=3), 0)
        self.assertGreater(ndcg_at_k(retrieved, relevant, k=3), 0)

    @patch("services.evaluation_service.run_search")
    def test_isolated_topic_feature_timing(self, mocked_search):
        mocked_search.return_value = [
            {
                "rank": 1,
                "doc_id": "1",
                "score": 1.0,
                "text": "python programming software",
            },
            {
                "rank": 2,
                "doc_id": "2",
                "score": 0.8,
                "text": "python coding tutorial",
            },
        ]
        queries = pd.DataFrame(
            [{"query_id": "q1", "text": "python programming"}]
        )
        qrels = pd.DataFrame(
            [{"query_id": "q1", "doc_id": "1", "relevance": 1}]
        )

        result = evaluate_single_method_with_topic_cost(
            dataset="dataset1",
            method="tfidf",
            loaded_data={},
            queries_df=queries,
            filtered_qrels_df=qrels,
            top_k=2,
            candidate_pool_size=2,
            retrieval_repetitions=2,
            feature_repetitions=2,
        )

        self.assertIsNotNone(result)
        self.assertGreater(
            result["timing"]["average_topic_feature_cost_seconds"],
            0,
        )
        self.assertGreaterEqual(
            result["timing"]["average_estimated_total_time_seconds"],
            result["timing"]["average_warmed_retrieval_time_seconds"],
        )
        self.assertNotIn(
            "average_total_time_difference_seconds",
            result["timing"],
        )


if __name__ == "__main__":
    unittest.main()
