"""ECG 证据解释边界。

集中声明：
    - 可以如何描述当前证据
    - 当前分析有哪些限制
    - 不允许从当前结果推出哪些结论

这些约束是下游生成与校验的输入，不保证 LLM 自动遵守。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.result import ECGAnalysisResult


POLICY_VERSION = "0.1.0"


def build_interpretation(
    result: "ECGAnalysisResult",
) -> Dict[str, Any]:
    """根据当前结果生成解释边界，不修改 result。

    当前版本面向未校准模型和未完成真实 ECG 准确性验证的
    节律提取流程。以后完成验证时，应显式更新策略版本，
    不能仅因为数值存在就自动升级可信程度。
    """
    limitations: List[Dict[str, str]] = [
        {
            "code": "MODEL_SCORES_UNCALIBRATED",
            "message": (
                "模型分数尚未校准；anomaly_score、"
                "reconstruction_error、shape_error 和 "
                "relative_evidence_score 不能解释为异常概率。"
                "未提供已验证阈值，不能据此判定正常或异常。"
            ),
        },
        {
            "code": "EVIDENCE_IS_MODEL_LOCALIZATION",
            "message": (
                "导联排名与时间区域表示模型证据的相对分布，"
                "不是已确认的病变位置或疾病事件。"
            ),
        },
        {
            "code": "NO_DIAGNOSTIC_VALIDATION",
            "message": (
                "当前系统用于医疗时间序列辅助分析；"
                "当前输出不足以确诊或排除疾病。"
            ),
        },
        {
            "code": "NO_VALIDATED_FUSION_RULE",
            "message": (
                "模型证据与节律特征尚未建立经过验证的融合规则。"
                "节律间隔较一致与模型误差较高不必然构成冲突，"
                "也不能据此推断两者相互验证。"
            ),
        },
    ]

    allowed_claims: List[str] = [
        "可以原样报告模型数值，但必须标明其为未校准分数。",
        "可以描述返回的导联排名，但不能称其为患病导联。",
        (
            "可以描述模型关注的时间区域；必须注明采样点或秒的单位"
            "以及相对于输入片段的坐标，不能称为已确认疾病事件。"
        ),
    ]

    prohibited_claims: List[str] = [
        "将任意模型分数或相对证据分数乘以 100 后称为异常概率。",
        "仅根据当前分数判断 ECG 正常或异常。",
        "将导联证据排名解释为病变定位。",
        "根据当前输出确诊、排除疾病或给出治疗建议。",
        "将节律规则性或 RR 变异系数解释为检测置信度。",
        "声称模型证据和节律特征已经完成临床交叉验证。",
    ]

    if result.provenance.get('model_decision',{}).get('status')=='configured':
        limitations[0]['message']='模型分数不是异常概率。已绑定阈值，可报告模型二分类结果并说明阈值数据用途；不能据此确诊或排除疾病，独立测试性能另行评估。'
        if result.provenance['model_decision'].get('selection',{}).get('split')=='development_reused':
            limitations.append({'code':'DEVELOPMENT_THRESHOLD','message':'阈值数据已参与checkpoint选择，仅用于开发演示，不能声称独立测试性能。'})
        prohibited_claims.remove('仅根据当前分数判断 ECG 正常或异常。')
        prohibited_claims.append('脱离已绑定阈值或把模型二分类称为临床结论。')
        allowed_claims.append('可以引用 get_model_decision 返回的阈值和模型预测类别。')
    model = result.model

    has_uncertainty = (
        model.uncertainty_mean is not None
        or model.uncertainty_max is not None
    )

    if has_uncertainty:
        uncertainty_status = "semantics_unverified"

        limitations.append({
            "code": "UNCERTAINTY_SEMANTICS_UNVERIFIED",
            "message": (
                "uncertainty 字段的参数化与数值语义尚未核实；"
                "不能将其直接解释为标准差、方差或置信度，"
                "也不能仅凭正负或大小判断模型是否可靠。"
            ),
        })

        prohibited_claims.append(
            "将 uncertainty 数值直接解释为标准差、概率或可靠性等级。"
        )
    else:
        uncertainty_status = "unavailable"

    # 数值存在不等于置信度已经校准。
    if result.confidence is None:
        confidence_status = "unavailable"
    else:
        confidence_status = "unverified"
        limitations.append({
            "code": "CONFIDENCE_NOT_VERIFIED",
            "message": (
                "confidence 字段虽有数值，但本策略未获得其校准与"
                "验证依据，不能将其表述为可信概率。"
            ),
        })

    rhythm = result.rhythm

    if rhythm is None:
        rhythm_status = "not_provided"

        limitations.append({
            "code": "RHYTHM_NOT_PROVIDED",
            "message": (
                "本次结果未附带节律特征；"
                "不能将缺失解释为未发现节律问题。"
            ),
        })

    elif rhythm.heart_rate is None:
        rhythm_status = "unavailable"

        limitations.append({
            "code": "RHYTHM_MEASUREMENT_UNAVAILABLE",
            "message": (
                "本次未得到可用心率估计；"
                "不能据此判断没有心拍或信号正常。"
            ),
        })

        allowed_claims.append(
            "可以报告候选峰数量，并说明本次心率估计不可用。"
        )

    else:
        rhythm_status = "unvalidated"

        limitations.append({
            "code": "RHYTHM_ACCURACY_UNVALIDATED",
            "message": (
                "节律提取器仅完成基础工程测试，"
                "真实 ECG 的峰检测和测量准确性尚未验证。"
                "心率与 RR 统计应描述为估计值。"
            ),
        })

        allowed_claims.append(
            "可以报告估计心率和 RR 统计，并注明单位及未验证状态。"
        )

    if rhythm is not None:
        limitations.append({
            "code": "RR_STATISTICS_USE_FILTERED_INTERVALS",
            "message": (
                "当前提取器会按配置的心率范围筛选 RR 间隔；"
                "候选峰数不等于参与统计的 RR 数量。"
                "较低 RR 变异不能证明全部候选间隔都一致或有效。"
            ),
        })

    return {
        "policy_version": POLICY_VERSION,
        "scope": "research_assistance_not_diagnosis",
        "status": {
            "model_scores": "uncalibrated",
            "uncertainty": uncertainty_status,
            "rhythm": rhythm_status,
            "confidence": confidence_status,
            "evidence_fusion": "not_validated",
        },
        "limitations": limitations,
        "allowed_claims": allowed_claims,
        "prohibited_claims": prohibited_claims,
    }


def build_grounded_context(
    result: "ECGAnalysisResult",
) -> Dict[str, Any]:
    """返回附带解释边界的独立 Context。

    不修改 result 或它的 provenance，也不调用 LLM。
    """
    context = deepcopy(result.to_llm_context())
    context["interpretation"] = build_interpretation(result)
    return context