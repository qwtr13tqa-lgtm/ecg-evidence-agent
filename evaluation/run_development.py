"""python -m evaluation.run_development --list; explicit --allow-external for runs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from src.evaluation.cases import load_cases
from src.evaluation.records import RunRecord
from src.evaluation.checks import build_reference, score_output

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", help="Repeat to select multiple cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--runs-dir", default=str(ROOT / "evaluation/runs"))
    parser.add_argument("--cases-file", default=str(ROOT / "evaluation/agent_development_cases.jsonl"))
    args = parser.parse_args()
    cases, suite_hash = load_cases(args.cases_file)
    if args.list or (not args.case and not args.all):
        for case in cases:
            print(case["id"], "|", case["category"], "|", case["question"])
        return
    if args.case and args.all:
        parser.error("Choose --case or --all")
    if args.repeat < 1 or args.repeat > 10:
        parser.error("--repeat must be 1..10")
    if not args.allow_external:
        parser.error("Running sends real structured analysis to configured gateway; add --allow-external")
    requested = set(args.case or [])
    if requested - {c["id"] for c in cases}:
        parser.error("Unknown case id")
    selected = cases if args.all else [c for c in cases if c["id"] in requested]
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    corpus = ROOT / "data/knowledge/ecg_knowledge.jsonl"
    versions = {}
    for package in ("numpy", "torch", "langgraph", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    config = {"model": os.getenv("ECG_MODEL", ""),
        "suite_sha256": suite_hash, "source_sha256": source_fingerprint(),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest() if corpus.exists() else None,
        "max_model_calls": 4, "max_tool_calls": 6, "timeout_seconds": 180,
        "max_tokens": 1800, "temperature": 0, "sdk_retries": 0,
        "endpoint_sha256": hashlib.sha256(os.getenv("ECG_BASE_URL", "").encode()).hexdigest(),
        "python": platform.python_version(), "packages": versions}
    # No API key, arbitrary environment dictionary, or credential-bearing URL is stored.
    print(f"Development runs: {len(selected)*args.repeat}; up to {len(selected)*args.repeat*4} model requests. No automatic retries.", flush=True)
    pipeline, data, retriever = None, None, None
    for repetition in range(args.repeat):
        for case in selected:
            record = RunRecord(args.runs_dir, case, {**config, "repetition": repetition+1})
            started = time.perf_counter()
            output, reference, failure = {}, {}, None
            gateway = None
            stage = "analysis"
            print(f"Running {case['id']} | run_id={record.run_id}", flush=True)
            try:
                record.checkpoint(stage)
                import numpy as np
                from src.analysis.pipeline import ECGAnalysisPipeline
                if data is None:
                    data = np.load(ROOT / "data/Processed_PTBXL/test.npy", mmap_mode="r")
                if pipeline is None:
                    pipeline = ECGAnalysisPipeline()
                result = pipeline.analyze(data[case["sample_index"], 100:4900, :],
                    source_id="data/Processed_PTBXL/test.npy", sample_index=case["sample_index"], crop_start_sample=100)
                reference = build_reference(result)
                stage = "gateway_setup"
                record.checkpoint(stage)
                from src.knowledge.retriever import BM25Retriever
                from src.agent.gateway import ToolGateway
                from src.agent.evidence_agent import ECGEvidenceAgent
                if retriever is None:
                    retriever = BM25Retriever.from_jsonl(corpus)
                gateway = ToolGateway(timeout=180, max_tokens=1800)
                stage = "agent"
                record.checkpoint(stage)
                output = ECGEvidenceAgent(pipeline.store, retriever, gateway).run(
                    result.analysis_id, case["question"], allow_external=True, data_kind="real_ecg")
            except (Exception, KeyboardInterrupt) as exc:
                failure = {"stage": stage, "error_type": type(exc).__name__}
                output = {"analysis_id": reference.get("analysis_id"), "status": "failed",
                          "draft": {}, "error": "RUNNER_FAILURE", "trace": []}
            finally:
                if gateway is not None:
                    try:
                        gateway.close()
                    except Exception:
                        pass  # Preserve the completed run even if transport cleanup fails.
            try:
                scores = score_output(case, output, reference)
            except Exception as exc:
                scores = {"metrics": {}, "scoring_error_type": type(exc).__name__}
            record.finish(output, reference, scores, failure=failure, wall_seconds=time.perf_counter()-started)
            print(f"Saved {record.path} | status={output.get('status')}", flush=True)
            # Release completed analysis arrays; on-disk structured record remains.
            if pipeline is not None and reference.get("analysis_id"):
                pipeline.store.discard(reference["analysis_id"])
            if failure:
                print("Stopped on runner failure:", json.dumps(failure), flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
