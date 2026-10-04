import json
import unittest
from src.agent.demo_analysis import synthetic_analysis
from src.agent.evidence_agent import ECGEvidenceAgent
from src.knowledge.retriever import BM25Retriever


class TestSubmitAnswer(unittest.TestCase):
    def run_case(self, mode):
        class Gateway:
            calls = 0
            def complete(inner, messages, tools):
                inner.calls += 1
                self.assertIn("submit_answer", [t["function"]["name"] for t in tools])
                eid = json.loads(messages[1]["content"])["bootstrap"]["evidence_id"]
                obj = {"answer": "虚构测试输入共4800点。", "evidence_ids": [eid], "knowledge_ids": [],
                       "observations": [{"evidence_id": eid, "path": "/input/num_samples", "value": 4800}]}
                if mode == "wrong_value":
                    obj["observations"][0]["value"] = 42
                if mode == "wrong_id":
                    obj["evidence_ids"] = ["another-analysis"]
                if mode == "missing_field":
                    del obj["observations"]
                arguments = json.dumps(obj)
                if mode == "broken_json":
                    arguments = '{"answer":}'
                if mode == "duplicate":
                    arguments = '{"answer":"one","answer":"two"}'
                calls = [{"id": "submit1", "type": "function", "function": {
                    "name": "submit_answer", "arguments": arguments}}]
                if mode == "mixed":
                    calls.append({"id": "query1", "type": "function", "function": {
                        "name": "get_analysis_summary", "arguments": "{}"}})
                return {"finish_reason": "length" if mode == "truncated" else "tool_calls",
                        "content": "unstructured_text_should_not_be_saved", "tool_calls": calls}
        store, result = synthetic_analysis()
        gateway = Gateway()
        output = ECGEvidenceAgent(store, BM25Retriever([]), gateway).run(result.analysis_id, "x", allow_external=True)
        self.assertEqual(gateway.calls, 1)
        self.assertNotIn("unstructured_text_should_not_be_saved", json.dumps(output))
        return output

    def test_valid_submission_ignores_content(self):
        out = self.run_case("valid")
        self.assertEqual(out["status"], "completed_draft")
        self.assertTrue(out["validation"]["passed"])
        self.assertEqual(out["trace"][-1]["stage"], "submission")

    def test_wrong_value_rejected(self):
        self.assertEqual(self.run_case("wrong_value")["error"], "OBSERVATION_VALUE_MISMATCH")

    def test_wrong_id_rejected(self):
        self.assertEqual(self.run_case("wrong_id")["status"], "failed")

    def test_missing_field_rejected(self):
        self.assertEqual(self.run_case("missing_field")["status"], "failed")

    def test_broken_arguments_rejected(self):
        self.assertEqual(self.run_case("broken_json")["error"], "ANSWER_JSON_INVALID")

    def test_duplicate_rejected(self):
        self.assertEqual(self.run_case("duplicate")["status"], "failed")

    def test_mixed_submit_query_rejected(self):
        self.assertEqual(self.run_case("mixed")["error"], "SUBMIT_MUST_BE_ALONE")

    def test_truncated_submission_rejected(self):
        self.assertEqual(self.run_case("truncated")["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
