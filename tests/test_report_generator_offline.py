"""报告生成器离线测试。

替换 OpenAI 客户端，所有响应均为本地模拟。
不调用网关，不读取真实 ECG，不需要真实 API Key。
"""

import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.reporting.generator import (
    ECGReportGenerator,
    ReportGenerationError,
)
from tests.test_report_generation import make_synthetic_context


def make_response(text, finish_reason="stop"):
    """模拟生成器实际读取的响应结构。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )
        ]
    )


class TestReportGeneratorOffline(unittest.TestCase):

    def setUp(self):
        # 仅在测试期间替换环境变量，结束后自动恢复。
        env_patch = patch.dict(
            os.environ,
            {"ECG_API_KEY": "offline-test-not-a-real-key", "ECG_BASE_URL": "https://example.invalid/v1", "ECG_MODEL": "test-model"},
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

        # 必须替换 generator 模块中引用的 OpenAI。
        # 显式构造生成器实际使用的接口。
        self.create = Mock(name="chat_completions_create")

        self.client = Mock(spec_set=["chat", "close"])
        self.client.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create,
            )
        )

        client_patch = patch(
            "src.reporting.generator.OpenAI",
            return_value=self.client,
        )
        self.openai_class = client_patch.start()
        self.addCleanup(client_patch.stop)

        self.context = make_synthetic_context()

        self.valid_output = {
            "summary": "虚构软件测试草稿，不代表真实测量。",
            "explanations": [
                {
                    "id": "EXP_1",
                    "text": "观察值仅用于虚构软件测试。",
                    "observation_ids": ["OBS_11"],
                    "knowledge_ids": ["TEST_REFERENCE"],
                }
            ],
        }

    def set_model_output(self, output):
        self.create.return_value = make_response(
            json.dumps(output, ensure_ascii=False)
        )

    def generate(self):
        with ECGReportGenerator() as generator:
            return generator.generate(
                self.context,
                synthetic_only=True,
            )

    def assert_rejected(self, message_fragment):
        with self.assertRaises(ReportGenerationError) as caught:
            self.generate()

        self.assertIn(message_fragment, str(caught.exception))
        self.create.assert_called_once()
        self.client.close.assert_called_once()

    def test_valid_output(self):
        self.set_model_output(self.valid_output)

        before = deepcopy(self.context)
        result = self.generate()

        self.assertEqual(
            result["status"],
            "structurally_valid_draft",
        )
        self.assertTrue(result["validation"]["passed"])
        self.assertTrue(result["requires_review"])
        self.assertFalse(
            result["validation"]["semantic_support_checked"]
        )
        self.assertFalse(
            result["validation"]["medical_correctness_checked"]
        )

        self.assertEqual(self.context, before)
        self.assertEqual(
            result["report"]["limitations"],
            before["interpretation"]["limitations"],
        )

        observations = {
            item["evidence_path"]: item["value"]
            for item in result["report"]["observations"]
        }
        self.assertEqual(
            observations[
                "/signal_features/rhythm/heart_rate_bpm"
            ],
            75.0,
        )

        self.create.assert_called_once()
        self.client.close.assert_called_once()

        # 确认发送的是虚构测试载荷，而非完整 Context。
        kwargs = self.create.call_args.kwargs
        payload = json.loads(
            kwargs["messages"][1]["content"]
        )

        self.assertEqual(
            payload["data_kind"],
            "synthetic_software_test",
        )
        self.assertNotIn("provenance", payload)

    def test_non_json_rejected(self):
        self.create.return_value = make_response(
            "这是一段普通文字，不是 JSON。"
        )
        self.assert_rejected("严格 JSON")

    def test_duplicate_json_key_rejected(self):
        self.create.return_value = make_response(
            '{"summary":"first","summary":"second",'
            '"explanations":[]}'
        )
        self.assert_rejected("严格 JSON")

    def test_nan_rejected(self):
        self.create.return_value = make_response(
            '{"summary":NaN,"explanations":[]}'
        )
        self.assert_rejected("严格 JSON")

    def test_truncated_output_rejected(self):
        self.create.return_value = make_response(
            '{"summary":"unfinished',
            finish_reason="length",
        )
        self.assert_rejected("未正常完成")

    def test_empty_content_rejected(self):
        self.create.return_value = make_response("   ")
        self.assert_rejected("内容为空")

    def test_missing_choices_rejected(self):
        self.create.return_value = SimpleNamespace(choices=[])
        self.assert_rejected("未返回 choices")

    def test_unknown_knowledge_id_rejected(self):
        output = deepcopy(self.valid_output)
        output["explanations"][0]["knowledge_ids"] = [
            "NONEXISTENT_SOURCE"
        ]

        self.set_model_output(output)
        self.assert_rejected("invalid knowledge_ids")

    def test_unknown_observation_id_rejected(self):
        output = deepcopy(self.valid_output)
        output["explanations"][0]["observation_ids"] = [
            "OBS_DOES_NOT_EXIST"
        ]

        self.set_model_output(output)
        self.assert_rejected("invalid observation_ids")

    def test_extra_top_level_field_rejected(self):
        output = deepcopy(self.valid_output)

        # 模型无权自行提供或覆盖 observations。
        output["observations"] = []

        self.set_model_output(output)
        self.assert_rejected(
            "只能包含 summary 和 explanations"
        )

    def test_empty_summary_rejected(self):
        output = deepcopy(self.valid_output)
        output["summary"] = ""

        self.set_model_output(output)
        self.assert_rejected(
            "summary must be a non-empty string"
        )

    def test_empty_explanations_allowed(self):
        # 没有适当解释时，允许保守地不生成解释。
        self.set_model_output({
            "summary": "虚构测试草稿，没有生成解释。",
            "explanations": [],
        })

        result = self.generate()

        self.assertTrue(result["validation"]["passed"])
        self.assertEqual(
            result["report"]["explanations"],
            [],
        )

    def test_timeout_has_no_application_retry(self):
        # 本地模拟超时，绝不发出真实请求。
        self.create.side_effect = TimeoutError(
            "simulated timeout with private diagnostic detail"
        )

        with self.assertRaises(ReportGenerationError) as caught:
            self.generate()

        message = str(caught.exception)

        self.assertIn("模型请求失败", message)
        self.assertIn("TimeoutError", message)
        self.assertNotIn("private diagnostic detail", message)

        self.create.assert_called_once()

        # SDK 自动重试配置也必须关闭。
        self.assertEqual(
            self.openai_class.call_args.kwargs["max_retries"],
            0,
        )
        self.client.close.assert_called_once()

    def test_missing_synthetic_confirmation_blocks_request(self):
        with ECGReportGenerator() as generator:
            with self.assertRaises(ReportGenerationError):
                generator.generate(self.context)

        self.create.assert_not_called()

    def test_missing_policy_blocks_request(self):
        del self.context["interpretation"]

        with self.assertRaises(ReportGenerationError):
            self.generate()

        self.create.assert_not_called()

    def test_invalid_context_blocks_request(self):
        self.context["model"]["anomaly_score"] = float("nan")

        with self.assertRaises(ReportGenerationError):
            self.generate()

        self.create.assert_not_called()

    def test_missing_key_blocks_client_creation(self):
        with patch.dict(os.environ, {"ECG_API_KEY": ""}):
            with self.assertRaises(ReportGenerationError):
                ECGReportGenerator()

        self.openai_class.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)