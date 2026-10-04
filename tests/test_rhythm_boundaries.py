import unittest

import numpy as np

from src.features.rhythm import RhythmFeatureExtractor


class TestRhythmBoundaries(unittest.TestCase):

    def setUp(self):
        self.extractor = RhythmFeatureExtractor(
            sampling_rate=500,
            lead_index=1,
        )

    def test_flat_signal(self):
        """平直输入不应产生心率；不代表临床诊断。"""
        ecg = np.zeros((4800, 12), dtype=np.float32)

        features = self.extractor.extract(ecg)

        self.assertEqual(features.num_beats, 0)
        self.assertEqual(features.r_peaks, [])
        self.assertIsNone(features.heart_rate)
        self.assertIsNone(features.mean_rr)
        self.assertIsNone(features.rr_std)
        self.assertIsNone(features.rr_cv)
        self.assertIsNone(features.rhythm_regularity)

    def test_reject_one_dimensional_input(self):
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros(4800))

    def test_reject_nan(self):
        ecg = np.zeros((4800, 12))
        ecg[100, 1] = np.nan

        with self.assertRaises(ValueError):
            self.extractor.extract(ecg)

    def test_reject_infinity(self):
        ecg = np.zeros((4800, 12))
        ecg[100, 1] = np.inf

        with self.assertRaises(ValueError):
            self.extractor.extract(ecg)

    def test_reject_invalid_lead(self):
        extractor = RhythmFeatureExtractor(lead_index=12)

        with self.assertRaises(ValueError):
            extractor.extract(np.zeros((4800, 12)))

    def test_rr_conversion(self):
        """500 Hz 下相隔 400 个采样点，应为 0.8 秒。"""
        peaks = np.array([400, 800, 1200])

        rr = self.extractor._compute_rr_intervals(peaks)

        np.testing.assert_allclose(
            rr,
            [0.8, 0.8],
            rtol=1e-6,
        )

    def test_single_peak_has_no_rr(self):
        peaks = np.array([400])

        rr = self.extractor._compute_rr_intervals(peaks)
        features = self.extractor._build_features(peaks, rr)

        self.assertEqual(len(rr), 0)
        self.assertEqual(features.num_beats, 1)
        self.assertEqual(features.r_peaks, [400])
        self.assertIsNone(features.heart_rate)
        self.assertIsNone(features.mean_rr)


if __name__ == "__main__":
    unittest.main(verbosity=2)