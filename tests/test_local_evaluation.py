import json
import tempfile
import unittest
from pathlib import Path
from tests.test_development_evaluation import example
from src.evaluation.checks import score_output
from src.evaluation.cases import load_cases


class TestLocalEvaluation(unittest.TestCase):
    def test_suites_remain_separate(self):
        root = Path(__file__).resolve().parents[1] / "evaluation"
        original, a = load_cases(root / "agent_development_cases.jsonl")
        local, b = load_cases(root / "agent_local_cases.jsonl")
        self.assertEqual(len(original), 6)
        self.assertEqual(local[0]["id"], "DEV_RR_LOCAL")
        self.assertNotEqual(a, b)

    def test_local_path_passes(self):
        case, out, ref = example()
        case["expectation"]["knowledge_policy"] = "local_only"
        metrics = score_output(case, out, ref)["metrics"]
        self.assertTrue(metrics["no_knowledge_query"])
        self.assertTrue(metrics["task_observations"])

    def test_even_failed_search_violates_local_policy(self):
        case, out, ref = example()
        case["expectation"]["knowledge_policy"] = "local_only"
        for success in (True, False):
            out["trace"].append({"stage": "tool", "tool": "search_knowledge", "ok": success})
            self.assertFalse(score_output(case, out, ref)["metrics"]["no_knowledge_query"])

    def test_original_policy_unchanged(self):
        case, out, ref = example()
        self.assertNotIn("no_knowledge_query", score_output(case, out, ref)["metrics"])

    def test_unknown_policy_rejected(self):
        case, _, _ = example()
        case["expectation"]["knowledge_policy"] = "typo"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cases.jsonl"
            path.write_text(json.dumps(case))
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
