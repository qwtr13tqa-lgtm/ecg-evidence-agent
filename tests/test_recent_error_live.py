"""Real data local check by default; --with-agent --allow-external enables gateway."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--with-agent", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    args = parser.parse_args()
    if args.with_agent and not args.allow_external:
        parser.error("--with-agent requires --allow-external")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    import numpy as np
    from src.analysis.pipeline import ECGAnalysisPipeline
    from src.tools.executor import ECGToolExecutor
    root = Path(__file__).resolve().parents[1]
    data = np.load(root / "data/Processed_PTBXL/test.npy", mmap_mode="r")
    if not 0 <= args.sample < len(data):
        parser.error("sample index out of range")
    pipeline = ECGAnalysisPipeline()
    result = pipeline.analyze(data[args.sample, 100:4900, :], source_id="test.npy",
                              sample_index=args.sample, crop_start_sample=100)
    response = ECGToolExecutor(pipeline.store, result.analysis_id).execute(
        "inspect_recent_error", {"duration_seconds": 0.6, "lead": "V1"})
    assert response["ok"], response
    actual = response["data"]
    assert (actual["start_sample"], actual["end_sample"]) == (4500, 4800)
    assert actual["window_num_samples"] == 300
    assert actual["actual_duration_seconds"] == 0.6
    expected = result.model.error_map[4500:4800, 6]
    row = actual["leads"][0]
    np.testing.assert_allclose([row["mean"], row["maximum"], row["minimum"]],
                              [expected.mean(), expected.max(), expected.min()])
    assert row["peak_sample"] == 4500 + int(np.argmax(expected))
    print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
    print("REAL LOCAL WINDOW: PASS (no gateway request)", flush=True)
    if not args.with_agent:
        return
    from src.agent.gateway import ToolGateway
    from src.agent.evidence_agent import ECGEvidenceAgent
    from src.agent.window_regression import DEFAULT_WINDOW_QUESTION, check_window_regression
    from src.knowledge.retriever import BM25Retriever
    gateway = ToolGateway()
    try:
        print("Sending real structured context; up to 4 model requests, timeout 180s each.", flush=True)
        output = ECGEvidenceAgent(pipeline.store,
            BM25Retriever.from_jsonl(root / "data/knowledge/ecg_knowledge.jsonl"), gateway).run(
                result.analysis_id, DEFAULT_WINDOW_QUESTION, allow_external=True, data_kind="real_ecg")
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        checked = check_window_regression(output)
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        if not checked["passed"]:
            raise SystemExit("REAL AGENT WINDOW REGRESSION: FAIL")
        print("REAL AGENT WINDOW REGRESSION: PASS (selection/citation only; prose needs review)")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
