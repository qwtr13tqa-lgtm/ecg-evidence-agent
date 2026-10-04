import unittest
from copy import deepcopy
from src.agent.window_memory import (
    WindowReferenceError, check_query, check_response, check_submission,
    prepare_reference, resolve_reference,
)


def make_turn(aid="A", windows=(("V2", 4200, 4800),)):
    evidence, trace = {}, []
    for index, (lead, start, end) in enumerate(windows):
        eid = f"{aid}:window:{index}"
        evidence[eid] = {
            "analysis_id": aid, "evidence_id": eid, "ok": True,
            "data": {"window": {"start_sample": start, "end_sample": end}},
        }
        trace.append({
            "stage": "tool", "tool": "inspect_recent_error",
            "arguments": {"lead": lead, "duration_seconds": 1.2},
            "ok": True, "evidence_id": eid,
        })
    return {
        "analysis_id": aid,
        "output": {
            "analysis_id": aid, "evidence": evidence, "trace": trace,
            "status": "completed_draft",
        },
    }


class MemoryTests(unittest.TestCase):
    question = "回到第一轮提到的那个窗口，它的最大模型分数在哪个采样点？"

    def session(self):
        # At turn six, the first turn is outside a three-turn prose context.
        return {
            "analysis_id": "A",
            "turns": [make_turn()] + [make_turn(windows=()) for _ in range(4)],
        }

    def test_sixth_turn_recovers_first_window(self):
        result = resolve_reference(self.session(), "A", self.question)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(
            {k: result["window"][k]
             for k in ("lead", "start_sample", "end_sample")},
            {"lead": "V2", "start_sample": 4200, "end_sample": 4800},
        )

    def test_multiple_windows_require_clarification(self):
        session = self.session()
        session["turns"][0] = make_turn(
            windows=(("V2", 4200, 4800), ("V1", 4730, 4800)))
        result = resolve_reference(session, "A", self.question)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(len(result["candidates"]), 2)

    def test_cross_analysis_session_rejected(self):
        with self.assertRaises(WindowReferenceError):
            resolve_reference(self.session(), "B", self.question)

    def test_cross_analysis_historical_evidence_rejected(self):
        session = self.session()
        session["turns"][0]["output"]["evidence"][
            "A:window:0"]["analysis_id"] = "B"
        with self.assertRaises(WindowReferenceError):
            resolve_reference(session, "A", self.question)

    def test_wrong_window_rejected(self):
        ref = resolve_reference(self.session(), "A", self.question)
        with self.assertRaises(WindowReferenceError):
            check_query(ref, "inspect_error_window", {
                "lead": "V1", "start_sample": 4730, "end_sample": 4800})

    def test_exact_window_accepted(self):
        ref = resolve_reference(self.session(), "A", self.question)
        check_query(ref, "inspect_error_window", {
            "lead": "V2", "start_sample": 4200, "end_sample": 4800})

    def test_response_window_must_match(self):
        ref = resolve_reference(self.session(), "A", self.question)
        response = {
            "analysis_id": "A", "ok": True,
            "data": {"lead": "V1", "start_sample": 4730, "end_sample": 4800},
        }
        with self.assertRaises(WindowReferenceError):
            check_response(ref, response)

    def test_submission_requires_current_structured_citation(self):
        ref = resolve_reference(self.session(), "A", self.question)
        ref["current_evidence_id"] = "A:current"
        with self.assertRaises(WindowReferenceError):
            check_submission(ref, {
                "answer": "A:current",
                "observations": [{"evidence_id": "A:old"}],
            })
        check_submission(ref, {
            "answer": "窗口最大值位置",
            "observations": [{"evidence_id": "A:current"}],
        })

    def test_missing_history_clarifies(self):
        result = prepare_reference(object(), "A", self.question)
        self.assertEqual(result["status"], "needs_clarification")

    def test_resolver_does_not_mutate_history(self):
        session = self.session()
        before = deepcopy(session)
        resolve_reference(session, "A", self.question)
        self.assertEqual(session, before)


if __name__ == "__main__":
    unittest.main()
