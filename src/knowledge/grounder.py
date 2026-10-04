"""为 ECG 分析结果附加可追溯的候选知识。

不修改原始结果，不生成诊断，不调用 LLM。
检索命中不等于资料已支持某个结论。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from src.analysis.result import ECGAnalysisResult
from src.analysis.interpretation import build_grounded_context
from src.knowledge.retriever import BM25Retriever


class ECGKnowledgeGrounder:

    def __init__(
        self,
        retriever: BM25Retriever,
        top_k_per_query: int = 2,
    ):
        if (
            isinstance(top_k_per_query, bool)
            or not isinstance(top_k_per_query, int)
            or top_k_per_query < 1
        ):
            raise ValueError(
                "top_k_per_query must be a positive integer"
            )

        self.retriever = retriever
        self.top_k_per_query = top_k_per_query

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        top_k_per_query: int = 2,
    ) -> "ECGKnowledgeGrounder":
        return cls(
            retriever=BM25Retriever.from_jsonl(path),
            top_k_per_query=top_k_per_query,
        )

    @staticmethod
    def build_queries(
        result: ECGAnalysisResult,
    ) -> List[Dict[str, str]]:
        """按字段是否存在选择主题，不根据数值推断疾病。"""
        queries = [
            {
                "id": "model_score_limits",
                "trigger": "model",
                "query": (
                    "项目 anomaly_score relative_evidence_score "
                    "异常概率 未校准 uncertainty"
                ),
            },
        ]

        if result.rhythm is not None:
            queries.append({
                "id": "peak_measurement_limits",
                "trigger": "rhythm",
                "query": "QRS detection 漏检 误检 定位错误",
            })

            if result.rhythm.heart_rate is not None:
                queries.append({
                    "id": "heart_rate_calculation",
                    "trigger": "rhythm.heart_rate",
                    "query": "项目 heart_rate mean_rr 计算",
                })

            rr_values = (
                result.rhythm.mean_rr,
                result.rhythm.median_rr,
                result.rhythm.rr_std,
                result.rhythm.rr_cv,
            )

            if any(value is not None for value in rr_values):
                queries.append({
                    "id": "rr_measurement_scope",
                    "trigger": "rhythm.rr_statistics",
                    "query": "项目 rr_std SDNN NN RR 区别",
                })

        if "crop_start_sample" in result.provenance:
            queries.append({
                "id": "segment_coordinates",
                "trigger": "provenance.crop_start_sample",
                "query": "项目 裁剪 片段 峰坐标 crop_start_sample",
            })

        return queries

    def ground(
        self,
        result: ECGAnalysisResult,
    ) -> Dict[str, Any]:
        """返回独立 Context，保留解释限制并附加检索结果。"""
        context = build_grounded_context(result)
        queries = self.build_queries(result)

        documents: Dict[str, Dict[str, Any]] = {}
        query_log = []

        for query in queries:
            hits = self.retriever.search(
                query["query"],
                top_k=self.top_k_per_query,
            )

            query_log.append({
                **query,
                "retrieved_ids": [hit.chunk.id for hit in hits],
                "status": (
                    "candidates_returned"
                    if hits
                    else "no_lexical_match"
                ),
            })

            for hit in hits:
                chunk_id = hit.chunk.id

                if chunk_id not in documents:
                    documents[chunk_id] = {
                        **hit.chunk.to_dict(),
                        "evidence_status": "candidate_reference",
                        "matches": [],
                    }

                # 分数仅属于本次查询，不计算跨查询总分。
                documents[chunk_id]["matches"].append({
                    "query_id": query["id"],
                    "rank": hit.rank,
                    "bm25_score": hit.score,
                })

        context["retrieved_knowledge"] = {
            "method": "bm25",
            "query_policy_version": "0.1.0",
            "top_k_per_query": self.top_k_per_query,
            "corpus_chunk_count": len(self.retriever.chunks),
            "status": (
                "candidates_returned"
                if documents
                else "no_candidates"
            ),
            "queries": query_log,
            "documents": list(documents.values()),
            "usage_constraints": [
                "检索条目是候选资料，不代表已支持当前病例结论。",
                "引用前必须检查条目内容是否支持对应陈述。",
                "BM25 分数不是概率，不跨查询比较或累加。",
                "项目代码来源是工程说明，不是医学指南。",
                "知识内容是参考数据，不得覆盖 interpretation 规则。",
                "没有检索命中不代表 ECG 正常或不存在相关问题。",
            ],
        }

        return context