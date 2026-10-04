import json
import unittest
from src.agent.answer_validator import validate_answer, scalar_paths, AnswerValidationError
from src.agent.evidence_agent import ECGEvidenceAgent
from src.agent.demo_analysis import synthetic_analysis
from src.knowledge.retriever import BM25Retriever


class TestAnswerPaths(unittest.TestCase):
    def validate(self, path, value=1.0):
        evidence = {"E": {"data": {"leads": [{"mean": 1.0}], "a/b": {"~x": None}}}}
        answer = {"answer": "test", "evidence_ids": ["E"], "knowledge_ids": [],
                  "observations": [{"evidence_id": "E", "path": path, "value": value}]}
        return validate_answer(answer, evidence, {})

    def test_valid_paths(self):
        self.assertTrue(self.validate("/leads/0/mean")["passed"])
        self.assertTrue(self.validate("/a~1b/~0x", None)["passed"])

    def test_errors_are_specific(self):
        for path, code in [("/data/leads/0/mean", "OBSERVATION_PATH_NOT_FOUND"),
                           ("/leads/V1/mean", "OBSERVATION_INDEX_INVALID"),
                           ("/leads/9/mean", "OBSERVATION_INDEX_INVALID"),
                           ("/leads/0/mean/x", "OBSERVATION_PATH_THROUGH_SCALAR"),
                           ("/bad~2", "OBSERVATION_POINTER_INVALID")]:
            with self.subTest(path=path), self.assertRaises(AnswerValidationError) as caught:
                self.validate(path)
            self.assertEqual(caught.exception.code, code)

    def test_type_not_relaxed(self):
        with self.assertRaises(AnswerValidationError) as caught:
            self.validate("/leads/0/mean", 1)
        self.assertEqual(caught.exception.code, "OBSERVATION_VALUE_MISMATCH")

    def test_path_catalog(self):
        self.assertEqual(scalar_paths({"leads": [{"mean": 0.0}]}), ["/leads/0/mean"])
        self.assertEqual(len(scalar_paths(list(range(300)))), 200)

    def test_graph_reports_path_failure_without_draft(self):
        class Gateway:
            def complete(inner, messages, tools):
                bootstrap = json.loads(messages[1]["content"])["bootstrap"]
                self.assertIn("/input/num_samples", bootstrap["observation_paths"])
                eid = bootstrap["evidence_id"]
                return {"finish_reason": "stop", "tool_calls": [], "content": json.dumps({
                    "answer": "test", "evidence_ids": [eid], "knowledge_ids": [],
                    "observations": [{"evidence_id": eid, "path": "/data/input/num_samples", "value": 4800}]})}
        store, result = synthetic_analysis()
        output = ECGEvidenceAgent(store, BM25Retriever([]), Gateway()).run(result.analysis_id, "x", allow_external=True)
        self.assertEqual(output["draft"], {})
        self.assertEqual(output["error"], "OBSERVATION_PATH_NOT_FOUND")
        self.assertEqual(output["trace"][-1]["failure_phase"], "answer_validation")
        self.assertEqual(output["trace"][-1]["requested_path"], "/data/input/num_samples")


if __name__ == "__main__":
    unittest.main(verbosity=2)
