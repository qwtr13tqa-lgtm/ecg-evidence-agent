import unittest

from src.reporting.schema import (
    ECGReport,
    Explanation,
    Observation,
    ReportLimitation,
)
from src.reporting.validator import validate_report


class TestReportValidator(unittest.TestCase):

    def setUp(self):
        # 仅测试接口，不代表真实病例或正式医学资料。
        self.context = {
            "signal_features": {
                "rhythm": {
                    "heart_rate_bpm": 60.85,
                    "measurement_status": "unvalidated",
                }
            },
            "interpretation": {
                "limitations": [
                    {
                        "code": "TEST_LIMIT",
                        "message": "测试限制：测量尚未验证。",
                    }
                ]
            },
            "retrieved_knowledge": {
                "documents": [{"id": "TEST_ONLY"}]
            },
        }

        self.report = ECGReport(
            summary="测试摘要：记录估计值，不作诊断。",
            observations=[
                Observation(
                    id="OBS_1",
                    evidence_path=(
                        "/signal_features/rhythm/heart_rate_bpm"
                    ),
                    value=60.85,
                )
            ],
            explanations=[
                Explanation(
                    id="EXP_1",
                    text="测试解释，仅用于验证引用接口。",
                    observation_ids=["OBS_1"],
                    knowledge_ids=["TEST_ONLY"],
                )
            ],
            limitations=[
                ReportLimitation(
                    code="TEST_LIMIT",
                    message="测试限制：测量尚未验证。",
                )
            ],
        ).to_dict()

    def test_valid_report(self):
        result = validate_report(self.report, self.context)
        self.assertTrue(result.passed, result.errors)

    def test_changed_value(self):
        self.report["observations"][0]["value"] = 90.0
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_missing_path(self):
        self.report["observations"][0]["evidence_path"] = "/model/missing"
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_unknown_knowledge_id(self):
        self.report["explanations"][0]["knowledge_ids"] = ["FAKE"]
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_unknown_observation_id(self):
        self.report["explanations"][0]["observation_ids"] = ["FAKE"]
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_missing_limitation(self):
        self.report["limitations"] = []
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_changed_limitation(self):
        self.report["limitations"][0]["message"] = "已完成验证"
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_nan_rejected(self):
        self.report["observations"][0]["value"] = float("nan")
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_duplicate_observation(self):
        self.report["observations"].append(
            dict(self.report["observations"][0])
        )
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_missing_policy_rejected(self):
        del self.context["interpretation"]
        self.assertFalse(
            validate_report(self.report, self.context).passed
        )

    def test_validation_scope_is_explicit(self):
        result = validate_report(self.report, self.context).to_dict()
        self.assertFalse(result["semantic_support_checked"])
        self.assertFalse(result["medical_correctness_checked"])


if __name__ == "__main__":
    unittest.main(verbosity=2)