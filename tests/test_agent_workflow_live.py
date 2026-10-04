"""虚构数据的完整工作流联调。

使用真实 Grounder、LangGraph 和网关生成器。
不读取真实 ECG，不加载 SGRF-Net。
只有显式运行本模块才会调用网关。
"""

import json
import os
import time
from pathlib import Path

# 在导入工作流前关闭远程追踪。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.agent.workflow import ECGReportWorkflow
from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.features.rhythm import RhythmFeatures
from src.knowledge.grounder import ECGKnowledgeGrounder
from src.reporting.generator import (
    ECGReportGenerator,
    ReportGenerationError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = (
    PROJECT_ROOT / "data/knowledge/ecg_knowledge.jsonl"
)


def make_synthetic_analysis():
    """所有数值均为测试占位，不来自真实患者或模型推理。"""
    result = ECGAnalysisResult(
        input=ECGInputInfo(
            num_samples=4800,
            num_leads=12,
            sampling_rate=500,
        ),
        model=ECGModelOutput(
            anomaly_score=-0.9,
            reconstruction_error=-0.95,
            shape_error=0.05,
        ),
        evidence=ECGEvidenceSummary(),
        provenance={
            "source_id": "SYNTHETIC_WORKFLOW_TEST",
            "data_kind": "synthetic_software_test",
            "crop_start_sample": 100,
            "crop_end_sample_exclusive": 4900,
            "peak_index_reference": "cropped_input",
        },
    )

    # 500 Hz 下，峰间距 400 点对应 0.8 秒。
    peaks = list(range(200, 4800, 400))

    result.rhythm = RhythmFeatures(
        heart_rate=75.0,
        mean_rr=0.8,
        median_rr=0.8,
        rr_std=0.0,
        rr_cv=0.0,
        num_beats=len(peaks),
        r_peaks=peaks,
        rhythm_regularity=1.0,
        sampling_rate=500,
        lead_index=1,
    )

    return result


def main():
    if not KNOWLEDGE_PATH.is_file():
        raise SystemExit(
            f"知识文件不存在：{KNOWLEDGE_PATH}"
        )

    if not os.environ.get("ECG_API_KEY", "").strip():
        raise SystemExit(
            "缺少 ECG_API_KEY，请在当前终端设置并 export。"
        )

    analysis = make_synthetic_analysis()
    before = analysis.to_llm_context()

    grounder = ECGKnowledgeGrounder.from_jsonl(
        KNOWLEDGE_PATH,
        top_k_per_query=2,
    )

    print("Input: synthetic analysis only", flush=True)
    print(f"Knowledge file: {KNOWLEDGE_PATH.name}", flush=True)
    print(
        "Running retrieval -> generation -> validation...",
        flush=True,
    )
    print(
        "One model request if input/retrieval succeeds; "
        "automatic retries disabled.",
        flush=True,
    )

    start = time.perf_counter()

    try:
        with ECGReportGenerator() as generator:
            workflow = ECGReportWorkflow(
                grounder=grounder,
                generator=generator,
            )

            result = workflow.run(
                analysis,
                synthetic_only=True,
            )

    except ReportGenerationError as exc:
        print(f"Initialization failed: {exc}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - start

    print(f"\nElapsed: {elapsed:.2f} seconds")
    print(f"Workflow status: {result['status']}")

    if result["status"] != "completed_draft":
        print("Workflow error:")
        print(json.dumps(
            result["error"],
            ensure_ascii=False,
            indent=2,
        ))
        raise SystemExit(1)

    output = result["output"]
    report = output["report"]
    validation = output["validation"]

    assert analysis.to_llm_context() == before
    assert output["data_kind"] == "synthetic_software_test"
    assert output["requires_review"] is True
    assert validation["passed"] is True
    assert validation["semantic_support_checked"] is False
    assert validation["medical_correctness_checked"] is False

    print("\nGenerated report:")
    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))

    print("\nValidation:")
    print(json.dumps(
        validation,
        ensure_ascii=False,
        indent=2,
    ))

    print(f"\nObservation count: {len(report['observations'])}")
    print(f"Explanation count: {len(report['explanations'])}")

    cited_ids = sorted({
        knowledge_id
        for explanation in report["explanations"]
        for knowledge_id in explanation["knowledge_ids"]
    })
    print(f"Cited knowledge IDs: {cited_ids}")

    if not report["explanations"]:
        print(
            "注意：解释为空，流程可以通过，"
            "但本次没有展示引用解释能力。"
        )

    print("\nAGENT WORKFLOW LIVE SMOKE TEST: PASS")
    print(
        "仅验证虚构数据的完整工作流；"
        "真实 ECG 外发和医学正确性尚未验证。"
    )


if __name__ == "__main__":
    main()