import json
import unittest

from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.features.rhythm import RhythmFeatureExtractor
from tests.test_rhythm import create_synthetic_ecg


class TestAnalysisRhythm(unittest.TestCase):

    def make_analysis(self):
        # 模型数值只是测试占位，不代表实际推理结果。
        return ECGAnalysisResult(
            input=ECGInputInfo(
                num_samples=5000,
                num_leads=12,
                sampling_rate=500,
            ),
            model=ECGModelOutput(
                anomaly_score=0.0,
                reconstruction_error=0.0,
                shape_error=0.0,
            ),
            evidence=ECGEvidenceSummary(),
        )

    def test_old_call_without_rhythm(self):
        analysis = self.make_analysis()
        context = analysis.to_llm_context()

        self.assertIsNone(
            context["signal_features"]["rhythm"]
        )
        self.assertIn("model", context)
        self.assertIn("evidence", context)

    def test_synthetic_rhythm_in_context(self):
        ecg = create_synthetic_ecg(
            duration=10.0,
            sampling_rate=500,
            heart_rate=60,
        )

        analysis = self.make_analysis()
        analysis.rhythm = RhythmFeatureExtractor(
            sampling_rate=500,
            lead_index=1,
        ).extract(ecg)

        context = analysis.to_llm_context()
        rhythm = context["signal_features"]["rhythm"]

        self.assertAlmostEqual(
            rhythm["heart_rate_bpm"],
            60.0,
            delta=5.0,
        )
        self.assertEqual(
            rhythm["candidate_beat_count"],
            len(analysis.rhythm.r_peaks),
        )
        self.assertEqual(
            rhythm["measurement_status"],
            "unvalidated",
        )
        self.assertNotIn("r_peaks", rhythm)
        self.assertNotIn("rhythm_regularity", rhythm)
        self.assertIsNone(context["confidence"])

        # 确认可以作为标准 JSON 传递。
        json.dumps(context, allow_nan=False)

    def test_unavailable_measurement(self):
        from src.features.rhythm import RhythmFeatures

        analysis = self.make_analysis()
        analysis.rhythm = RhythmFeatures(
            heart_rate=None,
            mean_rr=None,
            median_rr=None,
            rr_std=None,
            rr_cv=None,
            num_beats=0,
            r_peaks=[],
            rhythm_regularity=None,
            sampling_rate=500,
            lead_index=1,
        )

        rhythm = analysis.to_llm_context()[
            "signal_features"
        ]["rhythm"]

        self.assertIsNone(rhythm["heart_rate_bpm"])
        self.assertEqual(
            rhythm["measurement_status"],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)