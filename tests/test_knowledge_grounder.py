import json
import unittest

from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.features.rhythm import RhythmFeatures
from src.knowledge.grounder import ECGKnowledgeGrounder
from src.knowledge.retriever import (
    BM25Retriever,
    KnowledgeChunk,
)


class TestKnowledgeGrounder(unittest.TestCase):

    def make_result(self):
        return ECGAnalysisResult(
            input=ECGInputInfo(
                num_samples=4800,
                num_leads=12,
                sampling_rate=500,
            ),
            model=ECGModelOutput(
                anomaly_score=-0.9,
                reconstruction_error=-0.95,
                shape_error=0.05,
            ),
            evidence=ECGEvidenceSummary(),
            provenance={"crop_start_sample": 100},
        )

    def make_grounder(self):
        # 仅用于验证匹配和去重，不是医学依据。
        chunk = KnowledgeChunk(
            id="TEST_ONLY",
            title="项目",
            text="虚构软件测试条目。",
            source="test-fixture://not-medical-evidence",
            locator="fixture-1",
        )

        return ECGKnowledgeGrounder(
            BM25Retriever([chunk])
        )

    def test_missing_rhythm_skips_rhythm_queries(self):
        queries = ECGKnowledgeGrounder.build_queries(
            self.make_result()
        )

        ids = {query["id"] for query in queries}

        self.assertIn("model_score_limits", ids)
        self.assertIn("segment_coordinates", ids)
        self.assertNotIn("heart_rate_calculation", ids)
        self.assertNotIn("rr_measurement_scope", ids)

    def test_rhythm_queries(self):
        result = self.make_result()
        result.rhythm = RhythmFeatures(
            heart_rate=75.0,
            mean_rr=0.8,
            median_rr=0.8,
            rr_std=0.0,
            rr_cv=0.0,
            num_beats=3,
            r_peaks=[400, 800, 1200],
            rhythm_regularity=1.0,
            sampling_rate=500,
            lead_index=1,
        )

        ids = {
            query["id"]
            for query in ECGKnowledgeGrounder.build_queries(result)
        }

        self.assertIn("peak_measurement_limits", ids)
        self.assertIn("heart_rate_calculation", ids)
        self.assertIn("rr_measurement_scope", ids)

    def test_deduplication_preserves_matches(self):
        context = self.make_grounder().ground(self.make_result())
        documents = context["retrieved_knowledge"]["documents"]

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["id"], "TEST_ONLY")
        self.assertEqual(len(documents[0]["matches"]), 2)
        self.assertEqual(
            documents[0]["evidence_status"],
            "candidate_reference",
        )

    def test_empty_corpus_keeps_interpretation(self):
        grounder = ECGKnowledgeGrounder(BM25Retriever([]))
        context = grounder.ground(self.make_result())

        self.assertEqual(
            context["retrieved_knowledge"]["status"],
            "no_candidates",
        )
        self.assertEqual(
            context["retrieved_knowledge"]["documents"],
            [],
        )
        self.assertIn("interpretation", context)

    def test_result_is_not_modified(self):
        result = self.make_result()
        before = result.to_llm_context()

        context = self.make_grounder().ground(result)
        context["provenance"]["crop_start_sample"] = 999

        self.assertEqual(result.to_llm_context(), before)

    def test_serialization_and_source(self):
        context = self.make_grounder().ground(self.make_result())

        json.dumps(context, allow_nan=False)

        document = context["retrieved_knowledge"]["documents"][0]
        self.assertEqual(
            document["source"],
            "test-fixture://not-medical-evidence",
        )
        self.assertEqual(document["locator"], "fixture-1")

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            ECGKnowledgeGrounder(
                BM25Retriever([]),
                top_k_per_query=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)