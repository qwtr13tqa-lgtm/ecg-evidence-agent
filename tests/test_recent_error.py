import json
import unittest
import numpy as np
from src.agent.demo_analysis import synthetic_analysis
from src.tools.ecg_tools import inspect_recent_error
from src.tools.executor import ECGToolExecutor
from src.agent.window_regression import check_window_regression
from src.agent.evidence_agent import ECGEvidenceAgent
from src.knowledge.retriever import BM25Retriever


class TestRecentError(unittest.TestCase):
    def setUp(self):
        self.store, self.result = synthetic_analysis()

    def test_last_point_six_seconds(self):
        data = inspect_recent_error(self.result, 0.6, "V1")
        self.assertEqual((data["start_sample"], data["end_sample"]), (4500, 4800))
        self.assertEqual((data["start_seconds"], data["end_seconds"]), (9.0, 9.6))
        self.assertEqual(data["actual_duration_seconds"], 0.6)
        self.assertEqual(data["original_start_sample"], 4600)
        self.assertEqual(data["original_end_sample"], 4900)

    def test_statistics_use_correct_slice(self):
        self.result.model.error_map = np.arange(4800 * 12, dtype=np.float32).reshape(4800, 12)
        data = inspect_recent_error(self.result, 0.6, "V1")
        values = self.result.model.error_map[4500:4800, 6]
        self.assertEqual(data["leads"][0]["mean"], float(values.mean()))
        self.assertEqual(data["leads"][0]["maximum"], float(values.max()))
        self.assertEqual(data["leads"][0]["peak_sample"], 4799)

    def test_non_500_hz(self):
        self.result.input.sampling_rate = 250
        data = inspect_recent_error(self.result, 0.6, "II")
        self.assertEqual(data["start_sample"], 4650)

    def test_half_sample_rounding(self):
        data = inspect_recent_error(self.result, 0.003, "V1")
        self.assertEqual(data["window_num_samples"], 2)
        self.assertEqual(data["actual_duration_seconds"], 0.004)

    def test_whole_segment(self):
        self.assertEqual(inspect_recent_error(self.result, 9.6)["start_sample"], 0)

    def test_invalid_duration(self):
        for value in (True, False, "0.6", None, 0, -1, float("nan"), float("inf"), 9.601, 0.0001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_recent_error(self.result, value)

    def test_invalid_timebase(self):
        for fs in (0, -1, True, float("nan")):
            self.result.input.sampling_rate = fs
            with self.assertRaises(ValueError):
                inspect_recent_error(self.result, 0.6)

    def test_executor_binding_and_args(self):
        executor = ECGToolExecutor(self.store, self.result.analysis_id)
        ok = executor.execute("inspect_recent_error", {"duration_seconds": 0.6, "lead": "V1"})
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["analysis_id"], self.result.analysis_id)
        self.assertFalse(executor.execute("inspect_recent_error", {"duration_seconds": 0.6, "analysis_id": "other"})["ok"])
        self.assertFalse(executor.execute("inspect_recent_error", {"duration_seconds": 0.6, "lead": "unknown"})["ok"])

    def test_constant_window_argmax(self):
        row = inspect_recent_error(self.result, 0.6, "V1")["leads"][0]
        self.assertEqual(row["peak_sample"], 4500)

    def test_agent_recent_tool_and_synthetic_policy(self):
        class Gateway:
            count = 0
            def complete(inner, messages, tools):
                inner.count += 1
                if inner.count == 1:
                    self.assertIn("未运行 SGRF-Net", messages[0]["content"])
                    self.assertIn("inspect_recent_error", [t["function"]["name"] for t in tools])
                    return {"finish_reason": "tool_calls", "tool_calls": [{"id": "recent1", "type": "function",
                        "function": {"name": "inspect_recent_error", "arguments": '{"duration_seconds":0.6,"lead":"V1"}'}}]}
                response = json.loads(messages[-1]["content"])
                eid = response["evidence_id"]
                return {"finish_reason": "stop", "tool_calls": [], "content": json.dumps({
                    "answer": "虚构数组的最后 0.6 秒窗口是 [4500,4800)，均值为 0。",
                    "evidence_ids": [eid], "knowledge_ids": [], "observations": [
                        {"evidence_id": eid, "path": "/leads/0/mean", "value": 0.0}]})}
        out = ECGEvidenceAgent(self.store, BM25Retriever([]), Gateway()).run(
            self.result.analysis_id, "V1 最后0.6秒", allow_external=True, data_kind="synthetic_software_test")
        self.assertTrue(check_window_regression(out)["passed"])
        self.assertIn("SYNTHETIC_DATA", [x["code"] for x in out["limitations"]])
        # Reproduce the observed regression: a valid but incorrect 1.2-second window.
        eid = out["draft"]["evidence_ids"][0]
        out["evidence"][eid]["data"]["start_sample"] = 4200
        self.assertFalse(check_window_regression(out)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
