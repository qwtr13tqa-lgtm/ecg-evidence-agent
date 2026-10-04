import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from src.agent.gateway import ToolGateway
from src.agent.gateway_diagnostics import GatewayDiagnostics


class TestGatewayDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.g = ToolGateway.__new__(ToolGateway)
        self.g.model, self.g.timeout, self.g.max_tokens = "fixture", 180, 1800
        self.g.base_url = "http://user:secret_password@example.test/v1?token=secret_token"
        self.g.diagnostics = GatewayDiagnostics(self.root)
        self.create = Mock()
        self.g.client = NS(chat=NS(completions=NS(create=self.create)))
    def tearDown(self):
        self.temp.cleanup()
    def response(self):
        return NS(_request_id="req-test", choices=[NS(finish_reason="stop", message=NS(content="private_answer", tool_calls=[]))],
                  usage=NS(prompt_tokens=20, completion_tokens=10, total_tokens=30))
    def records(self):
        return [json.loads(p.read_text()) for p in self.root.glob("*.json")]
    def test_success_and_no_text(self):
        self.create.return_value = self.response()
        out = self.g.complete([{"role": "user", "content": "private_question"}], [])
        self.assertEqual(out["content"], "private_answer")
        text = json.dumps(self.records())
        for secret in ("private_question", "private_answer", "secret_password", "secret_token"):
            self.assertNotIn(secret, text)
        r = self.records()[0]
        self.assertEqual(r["status"], "response_received")
        self.assertEqual(r["usage"]["total_tokens"], 30)
        self.assertEqual(r["provider_request_id"], "req-test")
    def test_timeout_preserved_no_retry(self):
        exc = TimeoutError("private_exception")
        self.create.side_effect = exc
        with self.assertRaises(TimeoutError):
            self.g.complete([], [])
        self.assertEqual(self.create.call_count, 1)
        self.assertEqual(self.records()[0]["status"], "request_failed")
        self.assertNotIn("private_exception", json.dumps(self.records()))
    def test_adapter_error_separate(self):
        self.create.return_value = NS(choices=[])
        with self.assertRaises(ValueError):
            self.g.complete([], [])
        self.assertEqual(self.records()[0]["status"], "response_adapter_failed")
    def test_arguments_unchanged(self):
        self.create.return_value = self.response()
        messages = [{"role": "user", "content": "x"}]
        self.g.complete(messages, [])
        self.assertEqual(self.create.call_args.kwargs,
            dict(model="fixture", messages=messages, tools=[], tool_choice="auto", temperature=0, max_tokens=1800))
    def test_multiple_requests_separate(self):
        self.create.return_value = self.response()
        self.g.complete([], [])
        self.g.complete([], [])
        rows = self.records()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["request_id_local"], rows[1]["request_id_local"])
        self.assertEqual(rows[0]["session_id"], rows[1]["session_id"])
    def test_analysis_binding(self):
        aid = "b877d6bd-c99b-4a3c-aff5-a5963f400047"
        self.create.return_value = self.response()
        self.g.complete([{"role": "user", "content": json.dumps({"analysis_id": aid, "bootstrap": {}})}], [])
        self.assertEqual(self.records()[0]["analysis_id"], aid)
    def test_start_saved_before_request(self):
        def create(**kwargs):
            self.assertEqual(self.records()[0]["status"], "started")
            return self.response()
        self.create.side_effect = create
        self.g.complete([], [])
    def test_tool_response_and_no_arguments(self):
        response = self.response()
        response.choices[0].message.tool_calls = [NS(id="call1", type="function", function=NS(name="submit_answer", arguments='{"answer":"private_text"}'))]
        self.create.return_value = response
        self.assertEqual(len(self.g.complete([], [])["tool_calls"]), 1)
        self.assertNotIn("private_text", json.dumps(self.records()))
        self.assertEqual(self.records()[0]["response_tool_count"], 1)
    def test_http_error_metadata(self):
        exc = RuntimeError("private_body")
        exc.status_code, exc.request_id = 503, "req-503"
        self.create.side_effect = exc
        with self.assertRaises(RuntimeError):
            self.g.complete([], [])
        self.assertEqual(self.records()[0]["http_status"], 503)
    def test_log_failure_does_not_repeat_request(self):
        file = self.root / "not_a_directory"
        file.write_text("x")
        self.g.diagnostics = GatewayDiagnostics(file)
        self.create.return_value = self.response()
        self.assertEqual(self.g.complete([], [])["finish_reason"], "stop")
        self.assertEqual(self.create.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
