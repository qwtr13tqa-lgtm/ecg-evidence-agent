import json
import unittest
from src.agent.demo_analysis import synthetic_analysis
from src.agent.evidence_agent import ECGEvidenceAgent
from src.agent.tool_protocol import strict_json
from src.knowledge.retriever import BM25Retriever, KnowledgeChunk


def call(name="inspect_error_window", args=None, cid="call_1"):
    return {"finish_reason": "tool_calls", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name,
         "arguments": json.dumps(args if args is not None else {"start_sample": 4500, "end_sample": 4800, "lead": "V1"})}}]}


def answer(messages, tools):
    bootstrap = json.loads(messages[1]["content"])["bootstrap"]
    eid = bootstrap["evidence_id"]
    return {"finish_reason": "stop", "tool_calls": [], "content": json.dumps({
        "answer": "虚构数据输入共 4800 个采样点。", "evidence_ids": [eid],
        "knowledge_ids": [], "observations": [
            {"evidence_id": eid, "path": "/input/num_samples", "value": 4800}]})}


class FakeGateway:
    def __init__(self, steps):
        self.steps = list(steps)
        self.requests = []
    def complete(self, messages, tools):
        self.requests.append(messages)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages, tools) if callable(step) else step


class TestStage2Agent(unittest.TestCase):
    def setUp(self):
        self.store, self.result = synthetic_analysis()
        self.retriever = BM25Retriever([KnowledgeChunk("K1", "RR", "RR interval", "test", "fixture")])

    def run_agent(self, steps, **kwargs):
        gateway = FakeGateway(steps)
        agent = ECGEvidenceAgent(self.store, self.retriever, gateway, **kwargs)
        output = agent.run(self.result.analysis_id, "查询证据", allow_external=True, data_kind="synthetic_software_test")
        return output, gateway

    def test_native_tool_roundtrip(self):
        out, gateway = self.run_agent([call(), answer])
        self.assertEqual(out["status"], "completed_draft")
        self.assertEqual(out["tool_calls"], 1)
        response = json.loads(gateway.requests[1][-1]["content"])
        self.assertEqual(response["data"]["start_sample"], 4500)
        self.assertEqual(response["analysis_id"], self.result.analysis_id)

    def test_missing_consent_blocks_request(self):
        gateway = FakeGateway([])
        out = ECGEvidenceAgent(self.store, self.retriever, gateway).run(self.result.analysis_id, "x")
        self.assertEqual(out["error"], "EXTERNAL_TRANSFER_NOT_ENABLED")
        self.assertEqual(gateway.requests, [])

    def test_window_answer_cites_returned_value(self):
        def window_answer(messages, tools):
            response = json.loads(messages[-1]["content"])
            eid = response["evidence_id"]
            return {"finish_reason": "stop", "tool_calls": [], "content": json.dumps({
                "answer": "虚构误差窗口均值为零。", "evidence_ids": [eid], "knowledge_ids": [],
                "observations": [{"evidence_id": eid, "path": "/leads/0/mean",
                                  "value": response["data"]["leads"][0]["mean"]}]})}
        out, _ = self.run_agent([call(), window_answer])
        self.assertEqual(out["status"], "completed_draft")
        self.assertEqual(out["draft"]["observations"][0]["value"], 0.0)

    def test_multiple_lookup_rounds(self):
        out, _ = self.run_agent([call(), call("search_knowledge", {"query": "RR"}, "call_2"), answer])
        self.assertEqual(out["model_calls"], 3)
        self.assertEqual(out["tool_calls"], 2)
        self.assertEqual(out["status"], "completed_draft")

    def test_missing_analysis_blocks_request(self):
        gateway = FakeGateway([])
        out = ECGEvidenceAgent(self.store, self.retriever, gateway).run("missing", "x", allow_external=True)
        self.assertEqual(out["error"], "ANALYSIS_UNAVAILABLE")
        self.assertEqual(gateway.requests, [])

    def test_invalid_tool_json_returns_error(self):
        reply = call()
        reply["tool_calls"][0]["function"]["arguments"] = '{"x":1,"x":2}'
        out, gateway = self.run_agent([reply, answer])
        self.assertFalse(json.loads(gateway.requests[1][-1]["content"])["ok"])

    def test_reused_native_call_id_rejected(self):
        out, _ = self.run_agent([call(), call("get_analysis_summary", {})])
        self.assertEqual(out["status"], "failed")

    def test_repeated_call_stops(self):
        out, _ = self.run_agent([call(), call(cid="call_2")])
        self.assertEqual(out["error"], "REPEATED_TOOL_CALL")
        self.assertEqual(out["draft"], {})

    def test_model_budget(self):
        out, gateway = self.run_agent([call()], max_model_calls=1)
        self.assertEqual(out["error"], "MODEL_BUDGET_EXHAUSTED")
        self.assertEqual(len(gateway.requests), 1)

    def test_tool_budget(self):
        reply = call()
        reply["tool_calls"] += call(cid="call_2")["tool_calls"]
        out, _ = self.run_agent([reply], max_tool_calls=1)
        self.assertEqual(out["error"], "TOOL_BUDGET_EXHAUSTED")

    def test_cross_analysis_argument_rejected(self):
        out, gateway = self.run_agent([call("get_analysis_summary", {"analysis_id": "other"}), answer])
        self.assertFalse(json.loads(gateway.requests[1][-1]["content"])["ok"])

    def test_unknown_tool_is_recoverable(self):
        out, _ = self.run_agent([call("unknown", {}), answer])
        self.assertEqual(out["status"], "completed_draft")
        self.assertFalse(out["trace"][1]["ok"])

    def test_retrieval_records_source(self):
        out, _ = self.run_agent([call("search_knowledge", {"query": "RR"}), answer])
        self.assertEqual(out["knowledge"]["K1"]["source"], "test")

    def test_empty_retrieval(self):
        out, _ = self.run_agent([call("search_knowledge", {"query": "zzzz"}), answer])
        self.assertEqual(out["knowledge"], {})

    def test_timeout_no_retry(self):
        out, gateway = self.run_agent([TimeoutError("secret")])
        self.assertEqual(out["draft"], {})
        self.assertEqual(len(gateway.requests), 1)
        self.assertNotIn("secret", json.dumps(out))

    def test_truncated_answer(self):
        out, _ = self.run_agent([{"finish_reason": "length", "tool_calls": [], "content": "{}"}])
        self.assertEqual(out["status"], "failed")

    def test_wrong_value(self):
        def changed(messages, tools):
            reply = answer(messages, tools)
            reply["content"] = reply["content"].replace('"value": 4800', '"value": 999')
            return reply
        out, _ = self.run_agent([changed])
        self.assertEqual(out["status"], "failed")

    def test_bad_knowledge_reference(self):
        def changed(messages, tools):
            reply = answer(messages, tools)
            obj = json.loads(reply["content"])
            obj["knowledge_ids"] = ["invented"]
            reply["content"] = json.dumps(obj)
            return reply
        out, _ = self.run_agent([changed])
        self.assertEqual(out["status"], "failed")

    def test_duplicate_json_and_nonfinite(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                strict_json(value)

    def test_fresh_runs_do_not_reuse_draft(self):
        gateway = FakeGateway([answer, TimeoutError()])
        agent = ECGEvidenceAgent(self.store, self.retriever, gateway)
        first = agent.run(self.result.analysis_id, "x", allow_external=True)
        second = agent.run(self.result.analysis_id, "x", allow_external=True)
        self.assertEqual(first["status"], "completed_draft")
        self.assertEqual(second["draft"], {})

    def test_provenance_not_sent(self):
        out, gateway = self.run_agent([call("get_analysis_summary", {}), answer])
        self.assertNotIn("provenance", gateway.requests[1][1]["content"])
        self.assertNotIn("provenance", gateway.requests[1][-1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
