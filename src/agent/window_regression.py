"""One explicit regression case, not a general natural-language intent validator."""
DEFAULT_WINDOW_QUESTION = "请查询 V1 最后 0.6 秒的误差，再解释结果。"


def check_window_regression(output):
    """For the 4800-sample/500-Hz fixture only. Check selection and citation."""
    traces = [t for t in output.get("trace", []) if t.get("stage") == "tool"
              and t.get("tool") in ("inspect_recent_error", "inspect_error_window")]
    evidence = output.get("evidence", {})
    cited = output.get("draft", {}).get("evidence_ids", [])
    correct = []
    all_correct = bool(traces)
    for trace in traces:
        eid = trace.get("evidence_id")
        data = evidence.get(eid, {}).get("data", {})
        rows = data.get("leads", [])
        ok = (trace.get("ok") is True and data.get("start_sample") == 4500
              and data.get("end_sample") == 4800 and data.get("start_seconds") == 9.0
              and data.get("end_seconds") == 9.6 and len(rows) == 1
              and rows[0].get("lead") == "V1")
        all_correct = all_correct and ok
        if ok:
            correct.append(eid)
    checks = {
        "completed_draft": output.get("status") == "completed_draft",
        "window_selection_correct": all_correct,
        "correct_window_cited": any(eid in cited for eid in correct),
        "copied_window_value_present": any(
            obs.get("evidence_id") in correct and obs.get("path") == "/leads/0/mean"
            for obs in output.get("draft", {}).get("observations", [])),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "fixed_case_window_selection_and_citation",
            "free_text_semantics_checked": False}
