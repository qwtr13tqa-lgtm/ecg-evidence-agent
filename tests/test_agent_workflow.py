"""LangGraph 离线流程测试：不加载模型、不调用网关。"""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from src.agent.workflow import ECGReportWorkflow
from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.knowledge.grounder import ECGKnowledgeGrounder
from src.knowledge.retriever import BM25Retriever


def make_analysis():
    # 仅为软件测试占位值。
    return ECGAnalysisResult(
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
    )


def fake_generate(context, *, synthetic_only):
    if synthetic_only is not True:
        raise ValueError("Synthetic confirmation required")

    return {
        "status": "structurally_valid_draft",
        "data_kind": "synthetic_software_test",
        "requires_review": True,
        "report": {
            "schema_version": "0.1.0",
            "report_type": "ecg_auxiliary_analysis",
            "summary": "虚构软件测试草稿，没有生成医学解释。",
            "observations": [
                {
                    "id": "OBS_1",
                    "evidence_path": "/model/anomaly_score",
                    "value": context["model"]["anomaly_score"],
                }
            ],
            "explanations": [],
            "limitations": deepcopy(
                context["interpretation"]["limitations"]
            ),
        },
    }


class TestAgentWorkflow(unittest.TestCase):

    def setUp(self):
        self.analysis = make_analysis()

        real_grounder = ECGKnowledgeGrounder(
            BM25Retriever([])
        )

        self.grounder = Mock(spec=["ground"])
        self.grounder.ground.side_effect = real_grounder.ground

        self.generator = Mock(spec=["generate"])
        self.generator.generate.side_effect = fake_generate

        self.workflow = ECGReportWorkflow(
            grounder=self.grounder,
            generator=self.generator,
        )

    def run_workflow(self):
        return self.workflow.run(
            self.analysis,
            synthetic_only=True,
        )

    def test_success_with_empty_knowledge(self):
        result = self.run_workflow()

        self.assertEqual(result["status"], "completed_draft")
        self.assertEqual(result["error"], {})
        self.assertTrue(
            result["output"]["validation"]["passed"]
        )
        self.assertFalse(
            result["output"]["validation"][
                "medical_correctness_checked"
            ]
        )

        self.grounder.ground.assert_called_once()
        self.generator.generate.assert_called_once()

    def test_missing_confirmation_stops_before_retrieval(self):
        result = self.workflow.run(self.analysis)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["stage"], "input")
        self.grounder.ground.assert_not_called()
        self.generator.generate.assert_not_called()

    def test_invalid_input_stops_before_retrieval(self):
        result = self.workflow.run(
            {"not": "ECGAnalysisResult"},
            synthetic_only=True,
        )

        self.assertEqual(result["error"]["stage"], "input")
        self.grounder.ground.assert_not_called()
        self.generator.generate.assert_not_called()

    def test_retrieval_failure_stops_generation(self):
        self.grounder.ground.side_effect = RuntimeError(
            "private diagnostic detail"
        )

        result = self.run_workflow()

        self.assertEqual(result["error"]["stage"], "retrieval")
        self.assertNotIn("private diagnostic detail", str(result))
        self.generator.generate.assert_not_called()

    def test_generation_failure_returns_no_report(self):
        self.generator.generate.side_effect = TimeoutError()

        result = self.run_workflow()

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["stage"], "generation")
        self.assertEqual(result["output"], {})
        self.generator.generate.assert_called_once()

    def test_modified_value_rejected_independently(self):
        def bad_generate(context, *, synthetic_only):
            output = fake_generate(
                context,
                synthetic_only=synthetic_only,
            )
            output["report"]["observations"][0]["value"] = 999
            return output

        self.generator.generate.side_effect = bad_generate

        result = self.run_workflow()

        self.assertEqual(result["error"]["stage"], "validation")
        self.assertEqual(result["output"], {})

    def test_input_not_modified(self):
        before = deepcopy(self.analysis.to_llm_context())

        self.run_workflow()

        self.assertEqual(
            self.analysis.to_llm_context(),
            before,
        )

    def test_previous_success_not_reused_after_failure(self):
        first = self.run_workflow()
        self.assertEqual(first["status"], "completed_draft")

        self.generator.generate.side_effect = TimeoutError()
        second = self.run_workflow()

        self.assertEqual(second["status"], "failed")
        self.assertEqual(second["output"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)