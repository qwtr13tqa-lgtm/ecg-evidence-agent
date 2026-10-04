"""虚构数据的报告生成联调。

运行本模块会调用一次网关，可能产生费用。
不读取任何真实 ECG 数据。
"""

import json
from copy import deepcopy

from src.reporting.generator import (
    ECGReportGenerator,
    ReportGenerationError,
)


def make_synthetic_context():
    return {
        "input": {
            "num_samples": 4800,
            "num_leads": 12,
            "sampling_rate": 500,
            "duration_seconds": 9.6,
        },
        "model": {
            "anomaly_score": -0.9,
            "reconstruction_error": -0.95,
            "shape_error": 0.05,
            "uncertainty_mean": None,
            "uncertainty_max": None,
        },
        "evidence": {
            "lead_evidence": [],
            "temporal_regions": [],
            "relative_evidence_score": None,
        },
        "signal_features": {
            "rhythm": {
                "heart_rate_bpm": 75.0,
                "mean_rr_seconds": 0.8,
                "median_rr_seconds": 0.8,
                "rr_std_seconds": 0.0,
                "rr_cv": 0.0,
                "candidate_beat_count": 3,
                "measurement_status": "unvalidated",
            },
        },
        "confidence": None,
        "interpretation": {
            "scope": "synthetic_software_test",
            "limitations": [
                {
                    "code": "SYNTHETIC_DATA",
                    "message": (
                        "全部数值为软件测试占位值，"
                        "不代表真实患者或实际模型推理。"
                    ),
                },
                {
                    "code": "UNCALIBRATED",
                    "message": (
                        "模型分数未校准，不能解释为异常概率。"
                    ),
                },
                {
                    "code": "UNVALIDATED_RHYTHM",
                    "message": (
                        "节律测量准确性未验证，不能据此诊断。"
                    ),
                },
            ],
            "allowed_claims": [
                "描述虚构观察字段及其未验证状态。",
            ],
            "prohibited_claims": [
                "诊断或排除疾病。",
                "给出治疗建议。",
                "将测试占位值称为真实测量。",
            ],
        },
        "retrieved_knowledge": {
            "status": "candidates_returned",
            "documents": [
                {
                    "id": "TEST_REFERENCE",
                    "title": "软件测试解释规则",
                    "text": (
                        "本测试中的 heart_rate_bpm 是虚构占位值。"
                        "measurement_status 为 unvalidated，"
                        "表示不能声称测量已经验证。"
                        "本条只用于软件接口测试，不是医学依据。"
                    ),
                    "source": "test-fixture://report-generation",
                    "locator": "fixture-1",
                    "evidence_status": "candidate_reference",
                },
            ],
            "usage_constraints": [
                "测试条目不是医学文献。",
                "只有内容支持对应陈述时才能引用。",
            ],
        },
    }


def main():
    context = make_synthetic_context()
    original = deepcopy(context)

    print("Sending synthetic Context only...")
    print("One gateway request; automatic retries disabled.")

    try:
        with ECGReportGenerator() as generator:
            output = generator.generate(
                context,
                synthetic_only=True,
            )
    except ReportGenerationError as exc:
        print(f"\nREPORT GENERATION FAILED: {exc}")
        raise SystemExit(1)

    assert context == original, "生成过程修改了输入 Context"

    assert output["status"] == "structurally_valid_draft"
    assert output["validation"]["passed"] is True
    assert output["requires_review"] is True

    assert (
        output["validation"]["semantic_support_checked"]
        is False
    )
    assert (
        output["validation"]["medical_correctness_checked"]
        is False
    )

    report = output["report"]

    assert report["limitations"] == (
        context["interpretation"]["limitations"]
    )

    print("\nGenerated draft:")
    print(json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))

    if not report["explanations"]:
        print(
            "\n注意：本次 explanations 为空。"
            "这是允许的保守输出，但尚未展示引用生成能力。"
        )

    print("\nREPORT GENERATION SMOKE TEST: PASS")
    print("仅表示生成与结构校验流程成功，文字内容仍需人工检查。")


if __name__ == "__main__":
    main()