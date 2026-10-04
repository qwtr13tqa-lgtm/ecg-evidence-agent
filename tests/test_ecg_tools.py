import unittest
from unittest.mock import patch
import numpy as np
from src.analysis.store import AnalysisStore
from src.features.rhythm import RhythmFeatureExtractor
from src.tools.executor import ECGToolExecutor
from tests.test_analysis_isolation import sample_result, save


class TestECGTools(unittest.TestCase):
    def setUp(self):
        self.store = AnalysisStore()
        self.result = sample_result()
        extractor = RhythmFeatureExtractor()
        with patch.object(extractor, "_detect_r_peaks", return_value=np.array([0, 100, 500, 1700])):
            self.result.rhythm = extractor.extract(np.zeros((4800, 12)))
        self.a = save(self.store, self.result)
        self.b = save(self.store, sample_result(2))
        self.executor = ECGToolExecutor(self.store, self.a)

    def test_window_matches_direct_calculation(self):
        response = self.executor.execute("inspect_error_window", {
            "start_sample": 20, "end_sample": 40, "lead": "V1"})
        self.assertTrue(response["ok"])
        row = response["data"]["leads"][0]
        self.assertEqual(row["mean"], float(self.result.model.error_map[20:40, 6].mean()))
        self.assertEqual(response["data"]["original_start_sample"], 120)
        self.assertTrue(response["evidence_id"].startswith(self.a))

    def test_reject_cross_analysis_argument(self):
        response = self.executor.execute("get_analysis_summary", {"analysis_id": self.b})
        self.assertFalse(response["ok"])
        self.assertEqual(response["analysis_id"], self.a)

    def test_invalid_windows(self):
        for start, end, lead in [(-1, 2, None), (1, 101, None), (4, 4, None), (True, 3, None), (1, 4, "X")]:
            with self.subTest(start=start, end=end, lead=lead):
                self.assertFalse(self.executor.execute("inspect_error_window", {
                    "start_sample": start, "end_sample": end, "lead": lead})["ok"])

    def test_rr_filter_trace_and_pagination(self):
        # RR peaks were extracted from 4800 samples; match that input metadata.
        self.result.input.num_samples = 4800
        aid = save(self.store, self.result)
        self.executor = ECGToolExecutor(self.store, aid)
        response = self.executor.execute("inspect_rr_intervals", {"limit": 2})
        self.assertTrue(response["ok"])
        data = response["data"]
        self.assertEqual((data["total_intervals"], data["retained_count"], data["excluded_count"]), (3, 1, 2))
        self.assertEqual(data["next_offset"], 2)
        self.assertEqual(data["intervals"][0]["exclusion_reason"], "below_configured_min_rr")
        next_page = self.executor.execute("inspect_rr_intervals", {"offset": 2})["data"]
        self.assertEqual(next_page["intervals"][0]["exclusion_reason"], "above_configured_max_rr")
        self.assertAlmostEqual(self.result.rhythm.heart_rate, 75, places=4)

    def test_rr_metadata_stays_out_of_legacy_context(self):
        self.assertIsNotNone(self.result.rhythm.rr_details)
        self.assertNotIn("rr_details", self.result.rhythm.to_dict())

    def test_deleted_binding_fails(self):
        self.store.discard(self.a)
        self.assertFalse(self.executor.execute("get_analysis_summary")["ok"])

    def test_unknown_tool_and_history_copy(self):
        self.assertFalse(self.executor.execute("execute_python")["ok"])
        history = self.executor.history()
        history.clear()
        self.assertEqual(len(self.executor.history()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
