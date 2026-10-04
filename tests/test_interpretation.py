import json
import unittest

from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.analysis.interpretation import (
    build_interpretation,
    build_grounded_context,
)
from src.features.rhythm import RhythmFeatures


class TestInterpretation(unittest.TestCase):

    def make_result(self):
        # 仅为测试占位值，不代表真实推理。
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
            provenance={"source_id": "synthetic_test"},
        )

    def make_rhythm(self, available=True):
        return RhythmFeatures(
            heart_rate=75.0 if available else None,
            mean_rr=0.8 if available else None,
            median_rr=0.8 if available else None,
            rr_std=0.0 if available else None,
            rr_cv=0.0 if available else None,
            num_beats=3 if available else 0,
            r_peaks=[400, 800, 1200] if available else [],
            rhythm_regularity=1.0 if available else None,
            sampling_rate=500,
            lead_index=1,
        )

    def test_rhythm_not_provided(self):
        policy = build_interpretation(self.make_result())

        self.assertEqual(
            policy["status"]["rhythm"],
            "not_provided",
        )

    def test_rhythm_unavailable(self):
        result = self.make_result()
        result.rhythm = self.make_rhythm(available=False)

        policy = build_interpretation(result)

        self.assertEqual(
            policy["status"]["rhythm"],
            "unavailable",
        )

    def test_rhythm_remains_unvalidated(self):
        result = self.make_result()
        result.rhythm = self.make_rhythm()

        policy = build_interpretation(result)

        self.assertEqual(
            policy["status"]["rhythm"],
            "unvalidated",
        )

        codes = {
            item["code"]
            for item in policy["limitations"]
        }

        self.assertIn("RHYTHM_ACCURACY_UNVALIDATED", codes)
        self.assertIn("RR_STATISTICS_USE_FILTERED_INTERVALS", codes)

    def test_uncertainty_semantics(self):
        result = self.make_result()

        self.assertEqual(
            build_interpretation(result)["status"]["uncertainty"],
            "unavailable",
        )

        result.model.uncertainty_mean = -0.99

        self.assertEqual(
            build_interpretation(result)["status"]["uncertainty"],
            "semantics_unverified",
        )

    def test_confidence_is_not_automatically_trusted(self):
        result = self.make_result()

        self.assertEqual(
            build_interpretation(result)["status"]["confidence"],
            "unavailable",
        )

        result.confidence = 0.95

        self.assertEqual(
            build_interpretation(result)["status"]["confidence"],
            "unverified",
        )

    def test_core_boundaries(self):
        policy = build_interpretation(self.make_result())

        self.assertEqual(
            policy["status"]["model_scores"],
            "uncalibrated",
        )
        self.assertEqual(
            policy["status"]["evidence_fusion"],
            "not_validated",
        )

        self.assertTrue(policy["allowed_claims"])
        self.assertTrue(policy["prohibited_claims"])

        codes = {
            item["code"]
            for item in policy["limitations"]
        }

        self.assertIn("MODEL_SCORES_UNCALIBRATED", codes)
        self.assertIn("NO_DIAGNOSTIC_VALIDATION", codes)
        self.assertIn("NO_VALIDATED_FUSION_RULE", codes)

    def test_context_preserves_original_fields(self):
        result = self.make_result()
        result.rhythm = self.make_rhythm()

        original = result.to_llm_context()
        context = build_grounded_context(result)

        self.assertIn("interpretation", context)

        for key, value in original.items():
            self.assertEqual(context[key], value)

        # 确保最终结构可以转换成标准 JSON。
        json.dumps(context, allow_nan=False)

    def test_context_does_not_modify_result(self):
        result = self.make_result()
        before = result.to_llm_context()

        context = build_grounded_context(result)
        context["provenance"]["source_id"] = "changed"
        context["model"]["anomaly_score"] = 999

        self.assertEqual(result.to_llm_context(), before)
        self.assertNotIn(
            "interpretation",
            result.to_llm_context(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)