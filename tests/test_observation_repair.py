import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from copy import deepcopy
import time

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/agent" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
v = load("answer_validator")
h = load("observation_repair")

class RepairTests(unittest.TestCase):
    def setUp(self):
        data = {"evidence": {"temporal_regions": [{"start_sample": 4200, "end_sample": 4800}]}}
        self.state = {"messages": [], "evidence": {"E": {"data": data, "observation_paths": v.scalar_paths(data)}},
            "knowledge": {}, "trace": [], "call_ids": [], "tool_calls": 0, "model_calls": 0}
        self.answer = {"answer": "区域", "evidence_ids": ["E"], "knowledge_ids": [],
            "observations": [{"evidence_id": "E", "path": "/evidence/temporal_regions", "value": []}]}
        self.calls = [{"id": "c1", "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps(self.answer)}}]
        self.details = {"observation_index": 0, "requested_path": "/evidence/temporal_regions"}
    def repair(self, **kw):
        return h.repair_non_scalar(self.state, self.calls, self.answer, self.details,
            kw.get("count", 1), kw.get("max_models", 4), kw.get("max_tools", 6))
    def test_container_error(self):
        with self.assertRaises(v.AnswerValidationError) as ctx:
            v.validate_answer(self.answer, self.state["evidence"], {})
        self.assertEqual(ctx.exception.code, "OBSERVATION_NON_SCALAR")
    def test_wrong_scalar_still_rejected(self):
        self.answer["observations"][0].update(path="/evidence/temporal_regions/0/start_sample", value=4199)
        with self.assertRaises(v.AnswerValidationError) as ctx:
            v.validate_answer(self.answer, self.state["evidence"], {})
        self.assertEqual(ctx.exception.code, "OBSERVATION_VALUE_MISMATCH")
    def test_type_still_rejected(self):
        self.answer["observations"][0].update(path="/evidence/temporal_regions/0/start_sample", value=4200.0)
        with self.assertRaises(v.AnswerValidationError):
            v.validate_answer(self.answer, self.state["evidence"], {})
    def test_native_protocol_and_paths(self):
        original = deepcopy(self.state)
        r = self.repair()
        self.assertEqual(r["messages"][-1]["tool_call_id"], "c1")
        self.assertEqual(r["tool_calls"], 1)
        feedback = json.loads(r["messages"][-1]["content"])
        self.assertIn("/evidence/temporal_regions/0/start_sample", feedback["candidate_scalar_paths"])
        self.assertEqual(original, self.state)
    def test_plain_reply_protocol(self):
        self.calls = []
        r = self.repair()
        self.assertEqual(r["messages"][-1]["role"], "user")
        self.assertEqual(r["tool_calls"], 0)
    def test_no_second_correction(self):
        self.state["observation_repairs"] = 1
        self.assertIsNone(self.repair())
    def test_model_budget(self):
        self.assertIsNone(self.repair(count=4))
    def test_tool_budget(self):
        self.assertIsNone(self.repair(max_tools=1))
    def test_actual_decide_correction_and_success(self):
        # Execute the actual installed decide node, with a scripted gateway.
        tree = ast.parse((ROOT / "src/agent/evidence_agent.py").read_text(encoding="utf-8-sig"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide")
        fixed = deepcopy(self.answer)
        fixed["observations"][0].update(path="/evidence/temporal_regions/0/start_sample", value=4200)
        replies = iter([
            {"finish_reason": "tool_calls", "tool_calls": self.calls},
            {"finish_reason": "tool_calls", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps(fixed)}}]},
        ])
        env = {"self": SimpleNamespace(max_model_calls=4, max_tool_calls=6,
            gateway=SimpleNamespace(complete=lambda *a: next(replies))),
            "failed": lambda code: {"status":"failed", "error":code, "route":"stop", "draft":{}, "validation":{}},
            "json":json, "time":time, "deepcopy":deepcopy, "TOOLS":[],
            "strict_json":json.loads, "parse_answer":json.loads,
            "validate_answer":v.validate_answer, "AnswerValidationError":v.AnswerValidationError,
            "AnswerParseError":type("AnswerParseError", (Exception,), {}),
            "repair_non_scalar":h.repair_non_scalar,
            "reference":{"status":"not_requested"}, "check_submission":lambda *a: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "decide", "exec"), env)
        first = env["decide"](self.state)
        self.assertEqual(first["route"], "decide")
        self.state.update(first)
        second = env["decide"](self.state)
        self.assertEqual(second["status"], "completed_draft")
        self.assertEqual(second["model_calls"], 2)
        self.assertEqual(second["tool_calls"], 2)
        self.assertTrue(second["validation"]["passed"])

if __name__ == "__main__":
    unittest.main()
