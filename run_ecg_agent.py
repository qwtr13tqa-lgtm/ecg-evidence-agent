"""Default: synthetic native-tool protocol test. Real context requires explicit flag."""
import argparse
import json
import os
from pathlib import Path
from src.agent.window_regression import DEFAULT_WINDOW_QUESTION, check_window_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--allow-external", action="store_true",
        help="Allow sending this analysis summary, requested local statistics and question to configured gateway")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--question", default=DEFAULT_WINDOW_QUESTION)
    args = parser.parse_args()
    if args.real and not args.allow_external:
        parser.error("真实分析外发需 --real --allow-external；请先用默认虚构模式验证协议。")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from src.agent.gateway import ToolGateway
    from src.agent.evidence_agent import ECGEvidenceAgent
    from src.knowledge.retriever import BM25Retriever
    root = Path(__file__).resolve().parent
    if args.real:
        import numpy as np
        from src.analysis.pipeline import ECGAnalysisPipeline
        data = np.load(root / "data/Processed_PTBXL/test.npy", mmap_mode="r")
        if not 0 <= args.sample < len(data):
            parser.error("sample index out of range")
        pipeline = ECGAnalysisPipeline()
        result = pipeline.analyze(data[args.sample, 100:4900, :],
            source_id="data/Processed_PTBXL/test.npy", sample_index=args.sample, crop_start_sample=100)
        store = pipeline.store
    else:
        from src.agent.demo_analysis import synthetic_analysis
        store, result = synthetic_analysis()
    gateway = ToolGateway()
    try:
        agent = ECGEvidenceAgent(store, BM25Retriever.from_jsonl(root / "data/knowledge/ecg_knowledge.jsonl"), gateway)
        kind = "real_ecg" if args.real else "synthetic_software_test"
        print("Input:", kind, "Model:", gateway.model, flush=True)
        print("最多 4 次模型请求、6 次工具执行；每次请求超时 180 秒，无自动重试。", flush=True)
        output = agent.run(result.analysis_id, args.question, allow_external=True, data_kind=kind)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        if output["status"] != "completed_draft":
            raise SystemExit(1)
        if not args.real and output["tool_calls"] == 0:
            raise SystemExit("生成成功但没有原生工具调用：协议验证未通过。")
        print("PROTOCOL AND STRUCTURE: PASS (不代表任务回答正确)")
        if args.question == DEFAULT_WINDOW_QUESTION:
            checked = check_window_regression(output)
            print(json.dumps(checked, ensure_ascii=False, indent=2))
            if not checked["passed"]:
                raise SystemExit("WINDOW REGRESSION: FAIL")
            print("WINDOW REGRESSION: PASS (窗口选择与引用；非全文语义验证)")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
