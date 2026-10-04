import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

VERSION = "ecg-unified-score-1.0"
REVIEW_FIELDS = [
    "task_correct", "evidence_support", "text_complete", "object_match"
]
VALID = {"pass", "fail", "pending", "na"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(n, d):
    return round(100 * n / d, 2) if d else None


def average(values):
    return round(statistics.mean(values), 4) if values else None


def numeric(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def citation_check(output):
    """Audit structured observation values only; not prose or task semantics."""
    evidence = output.get("evidence") or {}
    draft = output.get("draft") or {}
    observations = draft.get("observations") or []
    declared = draft.get("evidence_ids") or []
    aid = output.get("analysis_id")
    supported = 0

    for observation in observations:
        try:
            eid = observation["evidence_id"]
            response = evidence[eid]
            if (
                not aid
                or eid not in declared
                or response.get("analysis_id") != aid
                or response.get("evidence_id") != eid
                or response.get("ok") is not True
            ):
                continue
            path = observation["path"]
            if not isinstance(path, str) or not path.startswith("/"):
                continue
            value = response["data"]
            for token in path[1:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                value = value[int(token)] if isinstance(value, list) else value[token]

            expected = observation["value"]
            # Exact stored value matching; rounding in prose is reviewed separately.
            if isinstance(value, bool) or isinstance(expected, bool):
                match = type(value) is type(expected) and value == expected
            elif isinstance(value, (int, float)) and isinstance(expected, (int, float)):
                match = value == expected
            else:
                match = type(value) is type(expected) and value == expected
            supported += int(match)
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return supported, len(observations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evaluation/capability_runs")
    parser.add_argument("--reviews", help="Previously exported and edited review.csv")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Run directory not found: {root}")

    previous = {}
    if args.reviews:
        with Path(args.reviews).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = row["record_path"]
                if key in previous:
                    raise SystemExit(f"Duplicate review record: {key}")
                for field in REVIEW_FIELDS:
                    if row.get(field) not in VALID:
                        raise SystemExit(f"Invalid {field} in {key}")
                if any(row[field] == "na" for field in REVIEW_FIELDS[:3]):
                    raise SystemExit(f"Core review fields cannot be na: {key}")
                previous[key] = row

    records, reviews, warnings = [], [], []

    for path in sorted(root.rglob("result.json")):
        record_path = path.relative_to(root).as_posix()
        result = load_json(path)
        output = result.get("output")
        case = result.get("case")
        if not isinstance(output, dict) or not isinstance(case, dict):
            warnings.append(f"Skipped incompatible record: {record_path}")
            continue

        # Unknown metadata stays explicit, never inferred from file ordering.
        batch = result.get("batch_id") or result.get("batch")
        if not isinstance(batch, str) or not batch:
            batch = f"UNKNOWN:{record_path}"
            warnings.append(f"Missing batch metadata: {record_path}")

        scheme = result.get("scheme") or case.get("scheme")
        if not isinstance(scheme, str) or not scheme:
            scheme = "UNKNOWN"
            warnings.append(f"Missing scheme metadata: {record_path}")

        status = output.get("status", "unknown")
        completed = status == "completed_draft"
        answer = (output.get("draft") or {}).get("answer")
        automatic = result.get("automatic") or {}
        resolution = output.get("reference_resolution") or {}
        tool_events = [
            event for event in output.get("trace", [])
            if event.get("stage") == "tool"
        ]
        checks = automatic.get("field_checks") or {}

        old = previous.get(record_path, {})
        review = {
            "record_path": record_path,
            "batch": batch,
            "scheme": scheme,
            "case_id": case.get("case_id") or case.get("id") or "",
            "turn": case.get("turn") or result.get("turn") or "",
            "status": status,
            "question": case.get("question", ""),
            "answer": answer or "",
            "field_checks": json.dumps(checks, ensure_ascii=False),
            "resolved_window": json.dumps(resolution, ensure_ascii=False),
            "tool_queries": json.dumps(tool_events, ensure_ascii=False),
            "task_correct": old.get("task_correct", "pending"),
            "evidence_support": old.get("evidence_support", "pending"),
            "text_complete": old.get("text_complete", "pending"),
            "object_match": old.get("object_match", "pending"),
            "notes": old.get("notes", ""),
        }
        reviews.append(review)

        # Clarification can solve an ambiguous request if manually accepted.
        deliverable = status in {"completed_draft", "needs_clarification"}
        if not deliverable:
            verdict = "fail"
        else:
            votes = [review[field] for field in REVIEW_FIELDS[:3]]
            verdict = (
                "fail" if "fail" in votes
                else "pass" if all(v == "pass" for v in votes)
                else "pending"
            )
            if not isinstance(answer, str) or not answer.strip():
                verdict = "fail"

        supported, observation_count = citation_check(output)
        elapsed = output.get("elapsed_seconds")
        model_calls = output.get("model_calls")
        records.append({
            "batch": batch,
            "scheme": scheme,
            "completed": completed,
            "verdict": verdict,
            "object_match": review["object_match"],
            "supported": supported,
            "observations": observation_count,
            "elapsed": elapsed if numeric(elapsed) else None,
            "model_calls": model_calls if numeric(model_calls) else None,
            # submit_answer isn't a query; automatic memory requery is included.
            "query_calls": len(tool_events) if isinstance(output.get("trace"), list) else None,
        })

    if not records:
        raise SystemExit("No compatible records found; no report generated.")

    summaries = []
    groups = defaultdict(list)
    for record in records:
        groups[(record["batch"], record["scheme"])].append(record)

    for (batch, scheme), items in sorted(groups.items()):
        total = len(items)
        passed = sum(x["verdict"] == "pass" for x in items)
        failed = sum(x["verdict"] == "fail" for x in items)
        pending = total - passed - failed
        object_pending = sum(x["object_match"] == "pending" for x in items)
        object_applicable = [
            x for x in items if x["object_match"] in {"pass", "fail"}
        ]
        object_pass = sum(x["object_match"] == "pass" for x in object_applicable)
        observations = sum(x["observations"] for x in items)
        elapsed = [x["elapsed"] for x in items if x["elapsed"] is not None]
        models = [x["model_calls"] for x in items if x["model_calls"] is not None]
        queries = [x["query_calls"] for x in items if x["query_calls"] is not None]

        summaries.append({
            "batch": batch,
            "scheme": scheme,
            "runs": total,
            "completed_draft_pct": rate(sum(x["completed"] for x in items), total),
            "task_pass": passed,
            "task_fail": failed,
            "task_pending": pending,
            "task_pass_pct": rate(passed, total) if pending == 0 else None,
            "task_pass_lower_bound_pct": rate(passed, total),
            "task_pass_upper_bound_pct": rate(passed + pending, total),
            "object_applicable_reviewed": len(object_applicable),
            "object_pending": object_pending,
            "object_match_pct": (
                rate(object_pass, len(object_applicable))
                if object_pending == 0 else None
            ),
            "observation_count": observations,
            "exact_observation_support_pct": rate(
                sum(x["supported"] for x in items), observations
            ),
            "elapsed_records": len(elapsed),
            "median_elapsed_seconds": (
                round(statistics.median(elapsed), 4) if elapsed else None
            ),
            "model_count_records": len(models),
            "mean_model_calls": average(models),
            "query_count_records": len(queries),
            "mean_query_calls": average(queries),
        })

    destination = Path("evaluation/unified_reports") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "review.csv", reviews, list(reviews[0]))
    write_csv(destination / "summary.csv", summaries, list(summaries[0]))
    (destination / "summary.json").write_text(
        json.dumps({
            "scoring_version": VERSION,
            "unit": "individual_turn",
            "observation_check": "exact_structured_values_not_prose",
            "groups": summaries,
            "warnings": warnings,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Report:", destination)
    for row in summaries:
        print(
            row["batch"], row["scheme"],
            f'runs={row["runs"]}',
            f'pass={row["task_pass"]}',
            f'fail={row["task_fail"]}',
            f'pending={row["task_pending"]}',
        )
    if warnings:
        print("Metadata warnings: see summary.json")
    print("Original evaluation records were not modified.")


if __name__ == "__main__":
    main()