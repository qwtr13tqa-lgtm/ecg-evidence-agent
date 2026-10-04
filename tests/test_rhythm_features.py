import json
import unittest

from src.features.rhythm import RhythmFeatures


class TestRhythmFeatures(unittest.TestCase):

    def make_features(self):
        return RhythmFeatures(
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

    def test_fields(self):
        features = self.make_features()
        result = features.to_dict()

        self.assertEqual(result, {
            "heart_rate": 75.0,
            "mean_rr": 0.8,
            "median_rr": 0.8,
            "rr_std": 0.0,
            "rr_cv": 0.0,
            "num_beats": 3,
            "r_peaks": [400, 800, 1200],
            "rhythm_regularity": 1.0,
            "sampling_rate": 500,
            "lead_index": 1,
        })

        self.assertEqual(
            result["num_beats"],
            len(result["r_peaks"]),
        )

    def test_json_serialization(self):
        result = self.make_features().to_dict()

        encoded = json.dumps(result, allow_nan=False)
        decoded = json.loads(encoded)

        self.assertEqual(decoded, result)

    def test_missing_measurements(self):
        features = RhythmFeatures(
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

        result = json.loads(
            json.dumps(features.to_dict(), allow_nan=False)
        )

        self.assertIsNone(result["heart_rate"])
        self.assertIsNone(result["mean_rr"])
        self.assertEqual(result["r_peaks"], [])

    def test_dictionary_is_independent(self):
        features = self.make_features()
        result = features.to_dict()

        result["r_peaks"].append(1600)

        self.assertEqual(
            features.r_peaks,
            [400, 800, 1200],
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRhythmFeatures
    )

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)

    print("\nRHYTHM FEATURE SCHEMA: PASS")