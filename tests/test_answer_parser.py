import json
import unittest
from src.agent.answer_parser import parse_answer, AnswerParseError
from src.agent.tool_protocol import strict_json
from src.agent.demo_analysis import synthetic_analysis
from src.knowledge.retriever import BM25Retriever


class TestAnswerParser(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(parse_answer(' {"answer":"中文"} '), {"answer": "中文"})

    def test_complete_fences(self):
        for marker in ("json", "JSON", ""):
            self.assertEqual(parse_answer('```' + marker + '\n{"x":1}\n```'), {"x": 1})
        self.assertEqual(parse_answer('```json\r\n{"x":1}\r\n```'), {"x": 1})

    def test_explanation_rejected(self):
        for text in ('说明\n```json\n{}\n```', '```json\n{}\n```\n说明', '{}\n说明'):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)

    def test_broken_fences_rejected(self):
        for text in ('```json\n{}', '```python\n{}\n```', '```json\n{}\n```\n```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)

    def test_json_not_repaired(self):
        for text in ('{"x":1,}', '{"x":1 "y":2}', "{'x':1}", '{"x":"unterminated}'):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)

    def test_duplicate_nonfinite_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer('```json\n' + text + '\n```')

    def test_missing_and_non_object(self):
        for text in (None, 123, '', '  ', '[]', 'null', 'true', '"text"'):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)

    def test_diagnostics_no_content(self):
        with self.assertRaises(AnswerParseError) as caught:
            parse_answer('```json\n{\n"secret_marker":\n}\n```')
        details = caught.exception.details
        self.assertEqual(details["json_line"], 3)
        self.assertEqual(details["json_column"], 1)
        self.assertTrue(details["outer_fence_removed"])
        self.assertNotIn("secret_marker", json.dumps(details))

    def test_tool_arguments_still_strict(self):
        with self.assertRaises(ValueError):
            strict_json('```json\n{}\n```')

    def test_literal_backticks_preserved(self):
        text = json.dumps({"answer": "使用 ``` 字符"})
        self.assertEqual(parse_answer(text)["answer"], "使用 ``` 字符")

    def test_graph_fence_and_value_validation(self):
        from src.agent.evidence_agent import ECGEvidenceAgent
        class Gateway:
            wrong = False
            def complete(inner, messages, tools):
                eid = json.loads(messages[1]["content"])["bootstrap"]["evidence_id"]
                obj = {"answer": "测试", "evidence_ids": [eid], "knowledge_ids": [],
                       "observations": [{"evidence_id": eid, "path": "/input/num_samples",
                                         "value": 1 if inner.wrong else 4800}]}
                return {"finish_reason": "stop", "tool_calls": [],
                        "content": '```json\n' + json.dumps(obj) + '\n```'}
        store, result = synthetic_analysis()
        gateway = Gateway()
        agent = ECGEvidenceAgent(store, BM25Retriever([]), gateway)
        self.assertEqual(agent.run(result.analysis_id, 'x', allow_external=True)["status"], "completed_draft")
        gateway.wrong = True
        out = agent.run(result.analysis_id, 'x', allow_external=True)
        self.assertEqual(out["error"], "OBSERVATION_VALUE_MISMATCH")
        self.assertEqual(out["draft"], {})

    def test_graph_parse_diagnostics(self):
        from src.agent.evidence_agent import ECGEvidenceAgent
        class Gateway:
            def complete(inner, messages, tools):
                return {"finish_reason": "stop", "tool_calls": [], "content": '{"x":}'}
        store, result = synthetic_analysis()
        out = ECGEvidenceAgent(store, BM25Retriever([]), Gateway()).run(result.analysis_id, 'x', allow_external=True)
        self.assertEqual(out["error"], "ANSWER_JSON_INVALID")
        self.assertEqual(out["trace"][-1]["parse_reason"], "JSON_SYNTAX_ERROR")
        self.assertIn("json_column", out["trace"][-1])
        self.assertEqual(out["draft"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
