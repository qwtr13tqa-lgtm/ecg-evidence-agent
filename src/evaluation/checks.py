"""Automatic engineering checks, never a medical or free-text correctness score."""
import math
from src.agent.answer_validator import validate_answer


def build_reference(result):
    """Local reference snapshot. Never sent to the LLM as an answer key."""
    import numpy as np
    reference = {"analysis_id": result.analysis_id, "context": result.to_llm_context(),
                 "rr": None, "window": None}
    rhythm = result.rhythm
    if rhythm is not None and rhythm.rr_details is not None:
        details = rhythm.rr_details
        reference["rr"] = {"total_intervals": len(details["raw_rr_seconds"]),
            "retained_count": sum(details["valid_mask"]),
            "excluded_count": len(details["valid_mask"]) - sum(details["valid_mask"]),
            "raw_rr_seconds": details["raw_rr_seconds"], "valid_mask": details["valid_mask"],
            "candidate_beat_count": rhythm.num_beats, "heart_rate_bpm": rhythm.heart_rate,
            "mean_rr_seconds": rhythm.mean_rr, "parameters": details["parameters"]}
    # Fixed development case: V1, last 0.6 seconds. Independent slice calculation.
    fs, total = result.input.sampling_rate, result.input.num_samples
    count = round(0.6 * fs)
    if 0 < count <= total and result.model.error_map is not None:
        values = np.asarray(result.model.error_map)[total-count:total, 6]
        reference["window"] = {"start_sample": total-count, "end_sample": total,
            "start_seconds": (total-count)/fs, "end_seconds": total/fs,
            "mean": float(values.mean()), "maximum": float(values.max()),
            "minimum": float(values.min()), "lead": "V1"}
    return reference


def score_output(case, output, reference):
    metrics = {"completed_draft": output.get("status") == "completed_draft",
               "structure_values_references": False, "analysis_binding": False,
               "required_tool_selected": None, "task_observations": None,
               "task_evidence_cited": None}
    evidence, knowledge = output.get("evidence", {}), output.get("knowledge", {})
    aid = reference.get("analysis_id")
    metrics["analysis_binding"] = bool(aid) and output.get("analysis_id") == aid and all(
        item.get("analysis_id") == aid for item in evidence.values())
    if metrics["completed_draft"]:
        try:
            validate_answer(output.get("draft"), evidence, knowledge)
            metrics["structure_values_references"] = True
        except Exception:
            pass
    draft = output.get("draft", {})
    cited = draft.get("evidence_ids", [])
    observations = draft.get("observations", [])
    trace = [t for t in output.get("trace", []) if t.get("stage") == "tool"]
    def successes(name):
        return [t for t in trace if t.get("tool") == name and t.get("ok") is True
                and t.get("evidence_id") in evidence]
    def copied(path, expected, eligible=None):
        return any(o.get("path") == path and type(o.get("value")) is type(expected)
                   and o.get("value") == expected and o.get("evidence_id") in cited
                   and (eligible is None or o.get("evidence_id") in eligible) for o in observations)
    kind = case["expectation"]["kind"]
    if case["expectation"].get("knowledge_policy") == "local_only":
        metrics["no_knowledge_query"] = not any(t.get("tool") == "search_knowledge" for t in trace)
    if kind == "window":
        hits = successes("inspect_recent_error") + successes("inspect_error_window")
        metrics["required_tool_selected"] = bool(hits)
        expected = reference.get("window")
        correct = []
        if expected:
            for hit in hits:
                data = evidence[hit["evidence_id"]]["data"]
                rows = data.get("leads", [])
                if (all(data.get(k) == expected[k] for k in ("start_sample", "end_sample", "start_seconds", "end_seconds"))
                        and len(rows) == 1 and rows[0].get("lead") == "V1"
                        and all(type(rows[0].get(k)) in (int, float) and math.isclose(rows[0][k], expected[k], rel_tol=1e-6, abs_tol=1e-8)
                                for k in ("mean", "maximum", "minimum"))):
                    correct.append(hit["evidence_id"])
            metrics["window_selection"] = bool(correct) and len(correct) == len(hits)
            metrics["task_observations"] = any(copied("/leads/0/mean", evidence[eid]["data"]["leads"][0]["mean"], [eid]) for eid in correct)
            metrics["task_evidence_cited"] = any(eid in cited for eid in correct)
    elif kind in ("rr", "filter"):
        hits = successes("inspect_rr_intervals")
        ids = [t["evidence_id"] for t in hits]
        metrics["required_tool_selected"] = bool(ids)
        rr = reference.get("rr")
        if rr is not None:
            paths = [("/total_intervals", rr["total_intervals"]), ("/retained_count", rr["retained_count"]),
                     ("/excluded_count", rr["excluded_count"])]
            if kind == "filter":
                paths += [("/parameters/min_rr_seconds", rr["parameters"]["min_rr_seconds"]),
                          ("/parameters/max_rr_seconds", rr["parameters"]["max_rr_seconds"])]
            metrics["task_observations"] = all(copied(p, v, ids) for p, v in paths)
            if kind == "rr":
                summaries = [eid for eid, item in evidence.items() if "input" in item.get("data", {})]
                metrics["task_observations"] &= all(copied("/signal_features/rhythm/"+k, rr[k], summaries)
                    for k in ("candidate_beat_count", "heart_rate_bpm", "mean_rr_seconds"))
            metrics["task_evidence_cited"] = any(eid in cited for eid in ids)
    elif kind == "oversize":
        # Direct refusal from summary is valid too; no mandatory tool path.
        attempted = [t for t in trace if t.get("tool") in ("inspect_recent_error", "inspect_error_window")]
        metrics["no_successful_substitute_window"] = not any(t.get("ok") is True for t in attempted)
        metrics["oversize_tool_rejected"] = (all(t.get("ok") is False for t in attempted) if attempted else None)
    elif kind == "score":
        hits = successes("search_knowledge")
        metrics["required_tool_selected"] = bool(hits)
        metrics["knowledge_candidates_available"] = bool(knowledge)
        metrics["task_evidence_cited"] = bool(draft.get("knowledge_ids")) and metrics["structure_values_references"]
    # Missing QTc, refusal meaning, and prose semantics remain human judgments.
    return {"version": "development-checks-1.0", "metrics": metrics,
            "tool_execution": {"attempts": len(trace), "successes": sum(t.get("ok") is True for t in trace),
                               "errors": sum(t.get("ok") is not True for t in trace)},
            "manual_required": ["task_correct", "evidence_support", "text_complete"],
            "scope": "engineering_checks_not_prose_or_medical_validation"}
