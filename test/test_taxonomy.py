"""Unit tests for the error taxonomy (webpage/taxonomy.py).

Run from the project root:
    .venv/Scripts/python.exe test/test_taxonomy.py -v
"""
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webpage"))

from taxonomy import (  # noqa: E402
    PATTERNS, TAXONOMY_VERSION, THRESHOLDS, aggregate_diagnosis,
    classify_query_verdict, diagnose, is_refusal, metric_mean, needs_reclassification,
)


def report(corr=0.1, prec=0.9, recall=0.9, faith=0.9, rel=0.9, answer="x", invalid=None):
    return {
        "correctness_voto": corr, "retrieval_voto": prec, "recall_voto": recall,
        "faithfulness_voto": faith, "relevancy_voto": rel, "risposta_rag": answer,
        "metriche_non_valide": invalid or [],
    }

def pattern(**kw):
    return classify_query_verdict(report(**kw))["error_pattern"]


class TestVerdictThresholds(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(pattern(corr=THRESHOLDS["correct"]), "CORRECT")
        self.assertEqual(pattern(corr=THRESHOLDS["correct"] - 0.01), "PARTIAL")
        self.assertEqual(pattern(corr=THRESHOLDS["partial"]), "PARTIAL")
        self.assertNotIn(pattern(corr=THRESHOLDS["partial"] - 0.01), ("CORRECT", "PARTIAL"))

    def test_every_pattern_has_label(self):
        out = classify_query_verdict(report())
        self.assertEqual(out["error_pattern_label"], PATTERNS[out["error_pattern"]][0])
        self.assertEqual(out["taxonomy_version"], TAXONOMY_VERSION)


class TestDecisionTable(unittest.TestCase):
    def test_wrong_answers(self):
        self.assertEqual(pattern(prec=0.1, recall=0.1, faith=0.1), "TOTAL_FAIL")
        self.assertEqual(pattern(prec=0.1, recall=0.1, faith=0.9), "RET_FAIL")
        self.assertEqual(pattern(prec=0.9, recall=0.9, faith=0.1), "HALLUC")
        self.assertEqual(pattern(prec=0.9, recall=0.1, faith=0.9), "RET_INCOMPLETE")
        self.assertEqual(pattern(prec=0.1, recall=0.9, faith=0.9), "RET_NOISE")
        self.assertEqual(pattern(prec=0.9, recall=0.9, faith=0.9, rel=0.9), "NOT_EXTRACTED")
        self.assertEqual(pattern(prec=0.9, recall=0.9, faith=0.9, rel=0.1), "OFF_TOPIC")

    def test_total_fail_uses_recall(self):
        # retrieval found the passages (high recall): it is not a system collapse
        self.assertEqual(pattern(prec=0.1, recall=0.9, faith=0.1), "HALLUC")

    def test_no_unknown_when_all_metrics_present(self):
        grid = [0.0, 0.29, 0.3, 0.49, 0.5, 0.69, 0.7, 1.0]
        for prec, recall, faith, rel in itertools.product(grid, repeat=4):
            self.assertNotEqual(diagnose(prec, recall, faith, rel), "UNKNOWN", (prec, recall, faith, rel))


class TestMissingMetrics(unittest.TestCase):
    def test_nan_metric_is_not_a_zero(self):
        # a real Faithfulness of 0 is a hallucination, a missing one is not
        self.assertEqual(pattern(faith=0.0), "HALLUC")
        self.assertEqual(pattern(faith=0.0, invalid=["faithfulness_voto"]), "UNKNOWN")

    def test_missing_correctness(self):
        out = classify_query_verdict(report(invalid=["correctness_voto"]))
        self.assertEqual(out["verdict"], "not_evaluated")
        self.assertEqual(out["error_pattern"], "EVAL_FAIL")

    def test_deselected_recall(self):
        # without recall, retrieval incomplete and context not extracted can't be told apart
        self.assertEqual(pattern(recall=0.0, invalid=["recall_voto"]), "UNKNOWN")
        # but retrieval failure on precision alone is still not claimed
        self.assertEqual(pattern(prec=0.1, recall=0.0, faith=0.9, invalid=["recall_voto"]), "UNKNOWN")

    def test_legacy_reports(self):
        legacy = report(faith=0.0)
        del legacy["metriche_non_valide"]
        self.assertEqual(classify_query_verdict({**legacy, "ragas_fallita": False})["error_pattern"], "HALLUC")
        # failed query: the 0.0 Faithfulness was a NaN, the other metrics are still usable
        self.assertEqual(classify_query_verdict({**legacy, "ragas_fallita": True})["error_pattern"], "UNKNOWN")
        self.assertEqual(classify_query_verdict({**legacy, "correctness_voto": 0.0, "ragas_fallita": True})["error_pattern"], "EVAL_FAIL")

    def test_metric_mean_skips_invalid(self):
        reps = [report(faith=1.0), report(faith=0.0, invalid=["faithfulness_voto"])]
        self.assertEqual(metric_mean(reps, "faithfulness_voto"), 1.0)
        self.assertIsNone(metric_mean([report(invalid=["recall_voto"])], "recall_voto"))


class TestPartialAndRefusal(unittest.TestCase):
    def test_partial_has_cause(self):
        out = classify_query_verdict(report(corr=0.5, prec=0.1, recall=0.1, faith=0.9))
        self.assertEqual(out["error_pattern"], "PARTIAL")
        self.assertEqual(out["error_cause"], "RET_FAIL")

    def test_refusal_only_at_start(self):
        self.assertTrue(is_refusal("Non ho informazioni sufficienti per rispondere."))
        self.assertTrue(is_refusal("I don’t know the answer."))
        self.assertTrue(is_refusal("Non ho trovato informazioni abbastanza pertinenti nella base documentale."))
        long_answer = "The contract was signed in 2019 by both parties. " * 5 + "There is no information about the fee."
        self.assertFalse(is_refusal(long_answer))

    def test_refusal_cause_from_retrieval(self):
        out = classify_query_verdict(report(prec=0.1, recall=0.1, faith=0.0, answer="I cannot answer."))
        self.assertEqual((out["error_pattern"], out["error_cause"]), ("REFUSAL", "RET_FAIL"))
        out = classify_query_verdict(report(answer="I cannot answer."))
        self.assertEqual(out["error_cause"], "NOT_EXTRACTED")


class TestAggregation(unittest.TestCase):
    def test_diagnosis_counts_causes_and_skips_not_evaluated(self):
        reps = [classify_query_verdict(r) for r in (
            report(corr=0.9),
            report(prec=0.1, recall=0.1, faith=0.9),             # wrong, RET_FAIL
            report(corr=0.5, prec=0.1, recall=0.1, faith=0.9),   # partial, cause RET_FAIL
            report(invalid=["correctness_voto"]),                # not evaluated
        )]
        diag = aggregate_diagnosis(reps)
        self.assertEqual(diag["evaluated"], 3)
        self.assertEqual(diag["causes"][0][:2], ("RET_FAIL", 2))

    def test_reclassification_on_version_change(self):
        rep = {**report(), **classify_query_verdict(report())}
        self.assertFalse(needs_reclassification(rep))
        self.assertTrue(needs_reclassification({**rep, "taxonomy_version": TAXONOMY_VERSION - 1}))
        self.assertTrue(needs_reclassification(report()))


if __name__ == "__main__":
    unittest.main()
