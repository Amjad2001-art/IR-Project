import unittest

from services.query_refinement_service import (
    correct_query_spelling,
    expand_query_with_synonyms,
    get_query_suggestions,
    refine_query,
    suggest_query_from_history,
)


class QueryRefinementTests(unittest.TestCase):
    def test_corrects_multiple_misspellings(self):
        corrected = correct_query_spelling("lern programing and retreival")
        self.assertEqual(corrected, "learn programming and retrieval")

    def test_expands_query_with_synonyms(self):
        expanded = expand_query_with_synonyms("programming")
        self.assertIn("coding", expanded.split())
        self.assertIn("software", expanded.split())

    def test_suggests_from_first_character(self):
        suggestions = get_query_suggestions("p", top_k=12)
        self.assertTrue(
            any("programming" in suggestion for suggestion in suggestions)
        )

    def test_suggestions_continue_for_each_prefix(self):
        for prefix in ["p", "pr", "pro", "prog", "progr"]:
            suggestions = get_query_suggestions(prefix, top_k=12)
            self.assertTrue(
                any("programming" in suggestion for suggestion in suggestions),
                msg=f"No programming suggestion for prefix: {prefix}",
            )

    def test_history_affects_suggestion_ranking(self):
        history = [
            "python programming tutorial",
            "healthy food recipes",
        ]
        suggestions = suggest_query_from_history(
            "python prog",
            search_history=history,
            top_k=2,
        )
        self.assertEqual(
            suggestions[0]["suggested_query"],
            "python programming tutorial",
        )

    def test_refinement_combines_features(self):
        result = refine_query(
            "lern programing",
            search_history=["learn programming basics"],
        )
        self.assertEqual(result["corrected_query"], "learn programming")
        self.assertIn("study", result["expanded_query"].split())
        self.assertTrue(result["history_suggestions"])
        self.assertIn("basics", result["refined_query"].split())

    def test_wikir_suggestions_use_dataset_queries(self):
        suggestions = get_query_suggestions(
            "sou",
            dataset="dataset1",
            top_k=12,
        )
        self.assertIn(
            "southern methodist university",
            suggestions,
        )

    def test_wikir_spelling_uses_dataset_vocabulary(self):
        corrected = correct_query_spelling(
            "sothern methodis universty",
            dataset="dataset1",
        )
        self.assertEqual(
            corrected,
            "southern methodist university",
        )

    def test_wikir_corrects_short_transposition_errors(self):
        corrected = correct_query_spelling(
            "cheif justce united stats",
            dataset="dataset1",
        )
        self.assertEqual(
            corrected,
            "chief justice united states",
        )

    def test_wikir_catalog_is_used_for_suggestions(self):
        wikir_suggestions = get_query_suggestions(
            "hal",
            dataset="dataset1",
            top_k=12,
        )
        self.assertIn("halakha", wikir_suggestions)


if __name__ == "__main__":
    unittest.main()
