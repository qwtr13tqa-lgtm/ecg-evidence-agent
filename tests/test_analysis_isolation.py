import unittest
from threading import RLock
from unittest.mock import Mock
import numpy as np
from src.analysis.result import ECGAnalysisResult, ECGInputInfo, ECGModelOutput, ECGEvidenceSummary
from src.analysis.store import AnalysisStore
from src.analysis.pipeline import ECGAnalysisPipeline


def sample_result(value=1.0):
    return ECGAnalysisResult(
        input=ECGInputInfo(100, 12, 500),
        model=ECGModelOutput(value, value, 0.1,
            error_map=np.arange(1200, dtype=np.float32).reshape(100, 12) * value),
        evidence=ECGEvidenceSummary(),
        provenance={"crop_start_sample": 100},
    )


def save(store, result):
    analysis_id = store.begin()
    result.analysis_id = analysis_id
    result.provenance["analysis_id"] = analysis_id
    store.complete(analysis_id, result)
    return analysis_id


class TestAnalysisIsolation(unittest.TestCase):
    def test_distinct_snapshots(self):
        store = AnalysisStore()
        first = sample_result()
        a = save(store, first)
        b = save(store, sample_result(2))
        self.assertNotEqual(a, b)
        first.model.error_map[:] = 999
        copy = store.get_result(a)
        copy.model.error_map[:] = -1
        self.assertEqual(store.get_result(a).model.error_map[1, 0], 12)
        self.assertEqual(store.get_result(b).model.error_map[1, 0], 24)

    def test_terminal_states(self):
        store = AnalysisStore()
        a = store.begin()
        store.fail(a, "ValueError")
        self.assertEqual(store.get_record(a).status, "failed")
        with self.assertRaises(ValueError):
            store.get_result(a)
        with self.assertRaises(ValueError):
            store.fail(a, "Again")

    def test_capacity_and_discard(self):
        store = AnalysisStore(capacity=1)
        a = store.begin()
        with self.assertRaises(RuntimeError):
            store.begin()
        with self.assertRaises(ValueError):
            store.discard(a)
        store.fail(a, "Test")
        store.discard(a)
        self.assertNotEqual(a, store.begin())

    def test_wrong_result_id(self):
        store = AnalysisStore()
        a = store.begin()
        with self.assertRaises(ValueError):
            store.complete(a, sample_result())

    def make_pipeline(self):
        # Exercise lifecycle wrapper without importing Torch or loading weights.
        pipeline = ECGAnalysisPipeline.__new__(ECGAnalysisPipeline)
        pipeline.store = AnalysisStore()
        pipeline._analysis_lock = RLock()
        pipeline.checkpoint_sha256 = "test-only"
        pipeline.sampling_rate = 500
        pipeline.evidence_extractor = Mock()
        pipeline.evidence_extractor.configuration.return_value = {"version": "test"}
        return pipeline

    def test_pipeline_failure_recorded(self):
        pipeline = self.make_pipeline()
        pipeline._analyze = Mock(side_effect=ValueError("invalid input"))
        with self.assertRaises(ValueError):
            pipeline.analyze(np.zeros((1, 12)))
        record = pipeline.store.list_records()[0]
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error_type, "ValueError")

    def test_pipeline_success_and_hash(self):
        from types import SimpleNamespace
        pipeline = self.make_pipeline()
        result = sample_result()
        result.rhythm = SimpleNamespace(rr_details={"parameters": {"min_hr": 30}})
        pipeline._analyze = Mock(return_value=result)
        output = pipeline.analyze(np.zeros((4800, 12)))
        record = pipeline.store.get_record(output.analysis_id)
        self.assertEqual(record.status, "succeeded")
        self.assertEqual(len(record.metadata["input_sha256"]), 64)
        self.assertEqual(record.metadata["shape"], [4800, 12])


if __name__ == "__main__":
    unittest.main(verbosity=2)
