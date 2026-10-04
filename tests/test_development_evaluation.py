import json
import tempfile
import unittest
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from src.evaluation.cases import load_cases
from src.evaluation.records import RunRecord, atomic_json, read_review
from src.evaluation.checks import score_output
from evaluation.summarize_development import summarize, render_markdown

ROOT = Path(__file__).resolve().parents[1]


def example(kind="rr"):
    case = {"id": "TEST", "version": "1", "category": kind, "sample_index": 0,
            "question": "test", "rubric": ["test"], "expectation": {"kind": kind}}
    rr = {"total_intervals": 9, "retained_count": 9, "excluded_count": 0,
          "candidate_beat_count": 10, "heart_rate_bpm": 60.85, "mean_rr_seconds": 0.986,
          "parameters": {"min_rr_seconds": 0.27, "max_rr_seconds": 2.0}}
    evidence = {"A:rr": {"analysis_id": "A", "data": {k: rr[k] for k in ("total_intervals", "retained_count", "excluded_count", "parameters")}},
                "A:summary": {"analysis_id": "A", "data": {"input": {}, "signal_features": {"rhythm": {
                    k: rr[k] for k in ("candidate_beat_count", "heart_rate_bpm", "mean_rr_seconds")}}}}}
    obs = [{"evidence_id": "A:rr", "path": "/"+k, "value": rr[k]}
           for k in ("total_intervals", "retained_count", "excluded_count")]
    obs += [{"evidence_id": "A:summary", "path": "/signal_features/rhythm/"+k, "value": rr[k]}
            for k in ("candidate_beat_count", "heart_rate_bpm", "mean_rr_seconds")]
    output = {"analysis_id": "A", "status": "completed_draft", "error": "", "evidence": evidence,
              "knowledge": {}, "draft": {"answer": "test", "evidence_ids": list(evidence), "knowledge_ids": [], "observations": obs},
              "trace": [{"stage": "tool", "tool": "inspect_rr_intervals", "ok": True, "evidence_id": "A:rr"}],
              "model_calls": 2, "tool_calls": 2}
    return case, output, {"analysis_id": "A", "rr": rr}


class TestDevelopmentEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def record(self, **config):
        case, output, reference = example()
        run = RunRecord(self.root, case, {"model": "fixture", **config}, data_kind="synthetic_software_test")
        run.finish(output, reference, score_output(case, output, reference), wall_seconds=1.0)
        return run

    def test_six_cases_and_hash(self):
        cases, digest = load_cases(ROOT / "evaluation/agent_development_cases.jsonl")
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(digest), 64)

    def test_cli_list_without_model(self):
        result = subprocess.run([sys.executable, "-m", "evaluation.run_development", "--list"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().splitlines()), 6)

    def test_cli_without_external_flag_stops(self):
        result = subprocess.run([sys.executable, "-m", "evaluation.run_development", "--case", "DEV_RR",
                                 "--runs-dir", str(self.root / "runs")], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allow-external", result.stderr)
        self.assertFalse((self.root / "runs").exists())

    def test_duplicate_case_rejected(self):
        case, _, _ = example()
        path = self.root / "cases.jsonl"
        path.write_text(json.dumps(case)+'\n'+json.dumps(case))
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_unknown_case_kind_rejected(self):
        case, _, _ = example("unknown")
        path = self.root / "cases.jsonl"
        path.write_text(json.dumps(case))
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_runs_do_not_overwrite(self):
        a, b = self.record(), self.record()
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertTrue((a.path / "result.json").exists())
        with self.assertRaises(ValueError):
            a.finish({}, {}, {})

    def test_progress_exists_before_result(self):
        run = RunRecord(self.root, example()[0], {})
        self.assertTrue((run.path / "progress.json").exists())
        self.assertEqual(summarize(self.root)["incomplete_runs"], [run.run_id])

    def test_failed_result_saved(self):
        run = RunRecord(self.root, example()[0], {})
        run.finish({"status": "failed", "draft": {}}, {}, {}, failure={"stage": "agent", "error_type": "TimeoutError"})
        stored = json.loads((run.path / "result.json").read_text())
        self.assertEqual(stored["runner_failure"]["error_type"], "TimeoutError")

    def test_nonfinite_write_preserves_previous(self):
        path = self.root / "x.json"
        atomic_json(path, {"x": 1})
        with self.assertRaises(ValueError):
            atomic_json(path, {"x": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_manual_defaults_pending(self):
        run = self.record()
        self.assertEqual(read_review(run.path)["status"], "pending")
        group = next(iter(summarize(self.root)["groups"].values()))
        self.assertIsNone(group["manual"]["task_correct"]["pass_rate_on_scored"])
        self.assertEqual(group["manual"]["task_correct"]["scored_denominator"], 0)

    def test_manual_scoring_and_denominator(self):
        run = self.record()
        self.record()
        path = run.path / "review.json"
        review = json.loads(path.read_text())
        review.update(reviewer="tester", task_correct="pass", evidence_support="pass", text_complete="fail")
        atomic_json(path, review)
        group = next(iter(summarize(self.root)["groups"].values()))
        self.assertEqual(group["manual"]["task_correct"]["scored_denominator"], 1)
        self.assertEqual(group["manual"]["task_correct"]["unscored_or_not_applicable"], 1)
        self.assertEqual(group["manual"]["text_complete"]["fail"], 1)

    def test_review_cannot_attach_to_other_run(self):
        a, b = self.record(), self.record()
        (b.path / "review.json").write_bytes((a.path / "review.json").read_bytes())
        with self.assertRaises(ValueError):
            read_review(b.path)
        self.assertEqual(len(summarize(self.root)["invalid_records"]), 1)

    def test_result_edit_invalidates_review(self):
        run = self.record()
        with (run.path / "result.json").open("a") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            read_review(run.path)

    def test_reviewer_required(self):
        run = self.record()
        path = run.path / "review.json"
        review = json.loads(path.read_text())
        review["task_correct"] = "pass"
        atomic_json(path, review)
        with self.assertRaises(ValueError):
            read_review(run.path)

    def test_configuration_groups_separated(self):
        self.record(source_sha256="a")
        self.record(source_sha256="b")
        summary = summarize(self.root)
        self.assertEqual(len(summary["groups"]), 2)
        self.assertIn("不是医学准确率", render_markdown(summary))

    def test_rr_checks(self):
        case, output, ref = example()
        metrics = score_output(case, output, ref)["metrics"]
        self.assertTrue(metrics["task_observations"])
        self.assertTrue(metrics["structure_values_references"])
        self.assertTrue(metrics["task_evidence_cited"])

    def test_self_reported_validation_not_trusted(self):
        case, output, ref = example()
        output["validation"] = {"passed": True}
        output["draft"]["observations"][0]["value"] = 999
        metrics = score_output(case, output, ref)["metrics"]
        self.assertFalse(metrics["structure_values_references"])
        self.assertFalse(metrics["task_observations"])

    def test_analysis_mismatch(self):
        case, output, ref = example()
        output["evidence"]["A:rr"]["analysis_id"] = "B"
        self.assertFalse(score_output(case, output, ref)["metrics"]["analysis_binding"])

    def test_missing_observation_is_not_false_prose_claim(self):
        case, output, ref = example()
        output["draft"]["observations"].pop()
        scored = score_output(case, output, ref)
        self.assertFalse(scored["metrics"]["task_observations"])
        self.assertIn("task_correct", scored["manual_required"])

    def test_submission_not_counted_as_query(self):
        case, output, ref = example()
        output["trace"].append({"stage": "submission", "tool": "submit_answer", "ok": True})
        self.assertEqual(score_output(case, output, ref)["tool_execution"]["attempts"], 1)

    def test_oversize_direct_refusal_does_not_require_tool(self):
        case, output, ref = example("oversize")
        output["trace"] = []
        metrics = score_output(case, output, ref)["metrics"]
        self.assertTrue(metrics["no_successful_substitute_window"])
        self.assertIsNone(metrics["required_tool_selected"])
        self.assertIsNone(metrics["oversize_tool_rejected"])

    def test_oversize_silent_substitution_flagged(self):
        case, output, ref = example("oversize")
        output["trace"] = [{"stage": "tool", "tool": "inspect_recent_error", "ok": True}]
        self.assertFalse(score_output(case, output, ref)["metrics"]["no_successful_substitute_window"])

    def test_missing_measurement_requires_human(self):
        case, output, ref = example("missing")
        metrics = score_output(case, output, ref)["metrics"]
        self.assertIsNone(metrics["task_observations"])

    def test_no_knowledge_does_not_imply_success(self):
        case, output, ref = example("score")
        metrics = score_output(case, output, ref)["metrics"]
        self.assertFalse(metrics["knowledge_candidates_available"])
        self.assertFalse(metrics["task_evidence_cited"])

    def test_window_regression(self):
        case, output, ref = example("window")
        ref["window"] = {"start_sample": 4500, "end_sample": 4800, "start_seconds": 9.0,
                         "end_seconds": 9.6, "mean": -0.8, "maximum": 2.0, "minimum": -1.0}
        data = {k: ref["window"][k] for k in ("start_sample", "end_sample", "start_seconds", "end_seconds")}
        data["leads"] = [{"lead": "V1", "mean": -0.8, "maximum": 2.0, "minimum": -1.0}]
        output["evidence"]["A:window"] = {"analysis_id": "A", "data": data}
        output["trace"] = [{"stage": "tool", "tool": "inspect_recent_error", "ok": True, "evidence_id": "A:window"}]
        output["draft"]["evidence_ids"].append("A:window")
        output["draft"]["observations"].append({"evidence_id": "A:window", "path": "/leads/0/mean", "value": -0.8})
        self.assertTrue(score_output(case, output, ref)["metrics"]["window_selection"])
        data["start_sample"] = 4200
        self.assertFalse(score_output(case, output, ref)["metrics"]["window_selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
