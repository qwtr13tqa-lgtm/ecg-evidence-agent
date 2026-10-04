"""确定性的 ECG 报告工作流。

当前仅支持虚构数据。
不创建模型客户端、不自动重试、不配置远程追踪或持久化。
依赖由调用方传入并负责关闭。
"""

from copy import deepcopy
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, START, END

from src.analysis.result import ECGAnalysisResult
from src.reporting.validator import validate_report


class WorkflowState(TypedDict, total=False):
    analysis: Any
    synthetic_only: bool
    context: Dict[str, Any]
    output: Dict[str, Any]
    status: str
    error: Dict[str, str]


class ECGReportWorkflow:

    def __init__(self, grounder, generator):
        self.grounder = grounder
        self.generator = generator

        graph = StateGraph(WorkflowState)

        graph.add_node("check_input", self._check_input)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)

        graph.add_edge(START, "check_input")

        graph.add_conditional_edges(
            "check_input",
            self._route,
            {"continue": "retrieve", "stop": END},
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route,
            {"continue": "generate", "stop": END},
        )
        graph.add_conditional_edges(
            "generate",
            self._route,
            {"continue": "validate", "stop": END},
        )

        graph.add_edge("validate", END)

        self.graph = graph.compile()

    @staticmethod
    def _route(state):
        return "stop" if state.get("status") == "failed" else "continue"

    @staticmethod
    def _failure(stage, code):
        # 不返回原始异常，避免包含密钥或请求内容。
        return {
            "status": "failed",
            "output": {},
            "error": {
                "stage": stage,
                "code": code,
            },
        }

    def _check_input(self, state):
        if state.get("synthetic_only") is not True:
            return self._failure(
                "input",
                "SYNTHETIC_CONFIRMATION_REQUIRED",
            )

        if not isinstance(state.get("analysis"), ECGAnalysisResult):
            return self._failure(
                "input",
                "INVALID_ANALYSIS_TYPE",
            )

        return {"status": "input_checked"}

    def _retrieve(self, state):
        try:
            # 该流程只传结构化结果，不包含原始 ECG 数组。
            context = self.grounder.ground(state["analysis"])

            if (
                not isinstance(context, dict)
                or not isinstance(context.get("interpretation"), dict)
                or not isinstance(context.get("retrieved_knowledge"), dict)
            ):
                return self._failure(
                    "retrieval",
                    "INVALID_GROUNDED_CONTEXT",
                )

            return {
                "context": context,
                "status": "knowledge_retrieved",
            }

        except Exception:
            return self._failure(
                "retrieval",
                "RETRIEVAL_FAILED",
            )

    def _generate(self, state):
        try:
            # 避免被注入的生成器修改校验所用 Context。
            output = self.generator.generate(
                deepcopy(state["context"]),
                synthetic_only=True,
            )

            return {
                "output": output,
                "status": "report_generated",
            }

        except Exception:
            return self._failure(
                "generation",
                "GENERATION_FAILED",
            )

    def _validate(self, state):
        output = state.get("output")

        if (
            not isinstance(output, dict)
            or output.get("status") != "structurally_valid_draft"
            or output.get("data_kind") != "synthetic_software_test"
            or output.get("requires_review") is not True
        ):
            return self._failure(
                "validation",
                "INVALID_OUTPUT_ENVELOPE",
            )

        try:
            validation = validate_report(
                output.get("report"),
                state["context"],
            )
        except Exception:
            return self._failure(
                "validation",
                "VALIDATION_ERROR",
            )

        if not validation.passed:
            return self._failure(
                "validation",
                "REPORT_VALIDATION_FAILED",
            )

        checked_output = deepcopy(output)
        checked_output["validation"] = validation.to_dict()

        return {
            "output": checked_output,
            "status": "completed_draft",
        }

    def run(self, analysis, *, synthetic_only=False):
        """每次创建独立状态，防止上次结果污染本次运行。

        synthetic_only 是调用方声明，不能自动验证数据是否虚构。
        """
        state = self.graph.invoke({
            "analysis": analysis,
            "synthetic_only": synthetic_only,
            "context": {},
            "output": {},
            "status": "pending",
            "error": {},
        })

        # 不把完整分析对象和 Context 暴露在返回封装中。
        return {
            "status": state["status"],
            "output": state.get("output", {}),
            "error": state.get("error", {}),
        }