"""Summarize saved runs only. Does not invoke models or overwrite manual reviews."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from src.evaluation.records import atomic_json, read_review


def summarize(root):
    root = Path(root)
    groups, incomplete, errors = {}, [], []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (directory / "result.json").exists():
            if (directory / "progress.json").exists():
                incomplete.append(directory.name)
            continue
        try:
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            review = read_review(directory)
            signature = {"config": {k: v for k, v in result["config"].items() if k != "repetition"},
                         "data_kind": result["data_kind"], "dataset_role": result["dataset_role"]}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
            group = groups.setdefault(key, {"signature": signature, "runs": [], "automatic": {}, "manual": {}})
            row = {"run_id": result["run_id"], "case_id": result["case"]["id"],
                   "status": result["output"].get("status", "failed"),
                   "error": result["output"].get("error"), "runner_failure": result["runner_failure"],
                   "wall_seconds": result.get("wall_seconds"),
                   "model_calls": result["output"].get("model_calls"),
                   "tool_calls_including_submission": result["output"].get("tool_calls"),
                   "tool_execution": result["automatic"].get("tool_execution", {}),
                   "review_status": review["status"],
                   "automatic": result["automatic"].get("metrics", {}), "manual": review["scores"]}
            group["runs"].append(row)
        except Exception as exc:
            errors.append({"directory": directory.name, "error_type": type(exc).__name__})
    for group in groups.values():
        runs = group["runs"]
        group["run_count"] = len(runs)
        group["distinct_cases"] = sorted({r["case_id"] for r in runs})
        for field in ("automatic", "manual"):
            keys = sorted({k for r in runs for k in r[field]})
            for key in keys:
                vals = [r[field].get(key) for r in runs]
                yes = sum(v is True if field == "automatic" else v == "pass" for v in vals)
                no = sum(v is False if field == "automatic" else v == "fail" for v in vals)
                group[field][key] = {"pass": yes, "fail": no, "unscored_or_not_applicable": len(vals)-yes-no,
                                    "scored_denominator": yes+no,
                                    "pass_rate_on_scored": yes/(yes+no) if yes+no else None}
        times = [r["wall_seconds"] for r in runs if isinstance(r["wall_seconds"], (int, float))]
        group["wall_time"] = {"count": len(times), "mean_seconds": sum(times)/len(times) if times else None}
        attempts = sum(r["tool_execution"].get("attempts", 0) for r in runs)
        successes = sum(r["tool_execution"].get("successes", 0) for r in runs)
        group["tool_execution"] = {"attempts": attempts, "successes": successes,
            "success_rate": successes/attempts if attempts else None,
            "scope": "query_tools_only_excludes_submission_not_tool_selection_accuracy"}
        failures = defaultdict(int)
        for row in runs:
            if row["status"] != "completed_draft":
                label = row["error"] or "UNKNOWN_FAILURE"
                if row["runner_failure"]:
                    label = row["runner_failure"]["stage"] + ":" + row["runner_failure"]["error_type"]
                failures[label] += 1
        group["failure_types"] = dict(failures)
    return {"dataset_role": "development", "groups": groups, "incomplete_runs": incomplete,
            "invalid_records": errors,
            "scope": "small_development_suite_no_generalization_or_medical_accuracy_claim"}


def render_markdown(summary):
    lines = ["# Agent 开发集结果", "", "这是开发集，不是医学准确率或独立泛化评测。未评分不计入已评分分母。", ""]
    for key, group in summary["groups"].items():
        lines += [f"## 配置组 {key}", "", f"运行 {group['run_count']} 次，覆盖 {len(group['distinct_cases'])} 类任务。", "",
                  "| 运行 ID | 任务 | 状态 | 耗时/秒 | 人工评审 |", "|---|---|---|---:|---|"]
        for row in group["runs"]:
            lines.append(f"| {row['run_id']} | {row['case_id']} | {row['status']} | {row['wall_seconds']} | {row['review_status']} |")
        for field, label in (("automatic", "自动工程检查"), ("manual", "人工评分")):
            lines += ["", f"### {label}", "", "| 项目 | 通过 | 失败 | 未评分/不适用 | 已评分分母 |", "|---|---:|---:|---:|---:|"]
            for metric, counts in group[field].items():
                lines.append(f"| {metric} | {counts['pass']} | {counts['fail']} | {counts['unscored_or_not_applicable']} | {counts['scored_denominator']} |")
        lines += ["", f"查询工具执行成功：{group['tool_execution']['successes']}/{group['tool_execution']['attempts']}；不包含提交，也不代表选择正确。",
                  "", "失败类型计数：" + json.dumps(group["failure_types"], ensure_ascii=False)]
    lines += ["", f"未完成运行数：{len(summary['incomplete_runs'])}；无效记录数：{len(summary['invalid_records'])}。", "",
              "任务事实与全文语义需结合人工评分。不同模型、代码、语料或配置分组显示；重复样本不代表独立样本。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(Path(__file__).resolve().parent / "runs"))
    args = parser.parse_args()
    root = Path(args.runs_dir)
    if not root.is_dir():
        parser.error("No runs directory yet")
    summary = summarize(root)
    atomic_json(root / "summary.json", summary)
    (root / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print("Saved:", root / "summary.json")
    print("Saved:", root / "summary.md")
    print("Incomplete:", len(summary["incomplete_runs"]), "Invalid:", len(summary["invalid_records"]))


if __name__ == "__main__":
    main()
