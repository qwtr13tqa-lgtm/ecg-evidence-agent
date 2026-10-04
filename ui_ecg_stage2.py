"""One-session local analysis + bound Agent answers. Existing UI remains available."""
import os
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from pathlib import Path
import numpy as np
import streamlit as st
from src.analysis.pipeline import ECGAnalysisPipeline
from src.agent.gateway import ToolGateway
from src.agent.evidence_agent import ECGEvidenceAgent
from src.knowledge.retriever import BM25Retriever

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ECG Evidence Agent", layout="wide")
st.title("ECG 分析与按需证据查询")
st.caption("研究原型；回答通过结构校验不代表文字结论已验证。")

@st.cache_resource
def resources():
    return ECGAnalysisPipeline(), BM25Retriever.from_jsonl(ROOT / "data/knowledge/ecg_knowledge.jsonl")

data = np.load(ROOT / "data/Processed_PTBXL/test.npy", mmap_mode="r")
sample = st.number_input("样本索引", 0, len(data) - 1, 0)
if st.session_state.get("selected") != sample:
    st.session_state["selected"] = sample
    st.session_state.pop("analysis", None)
    st.session_state.pop("answer", None)
if st.button("本地分析"):
    st.session_state.pop("answer", None)
    st.session_state.pop("analysis", None)
    try:
        with st.spinner("运行 SGRF-Net 与节律分析"):
            pipeline, _ = resources()
            result = pipeline.analyze(data[sample, 100:4900, :],
                source_id="test.npy", sample_index=sample, crop_start_sample=100)
            st.session_state["analysis"] = (sample, result.analysis_id)
    except Exception as exc:
        st.error("本地分析失败：" + type(exc).__name__)

if "analysis" in st.session_state:
    selected, aid = st.session_state["analysis"]
    pipeline, retriever = resources()
    result = pipeline.store.get_result(aid)
    st.caption(f"样本 {selected} | 分析 ID：{aid}")
    st.line_chart(data[selected, 100:4900, 1])
    st.caption("导联 II；横轴为裁剪片段采样点，采样率 500 Hz。")
    with st.expander("模型证据与节律摘要"):
        st.json(result.to_llm_context())
    question = st.text_input("向当前分析提问", "V1 最后 0.6 秒的模型误差是多少？请查询后解释。")
    consent = st.checkbox("允许将当前真实分析摘要、按需查询的统计和问题发送到配置的模型网关（当前默认 HTTP）")
    if st.button("查询证据并回答", disabled=not consent):
        st.session_state.pop("answer", None)
        gateway = None
        try:
            with st.spinner("Agent 查询中：最多 4 次模型请求，每次最长 180 秒"):
                gateway = ToolGateway()
                out = ECGEvidenceAgent(pipeline.store, retriever, gateway).run(
                    aid, question, allow_external=True)
                if st.session_state.get("analysis") == (selected, aid):
                    st.session_state["answer"] = out
        except Exception as exc:
            st.error("调用失败：" + type(exc).__name__)
        finally:
            if gateway:
                gateway.close()
    out = st.session_state.get("answer")
    if out and out["analysis_id"] == aid:
        if out["status"] == "completed_draft":
            st.write(out["draft"]["answer"])
            st.caption("待复核草稿；自动检查范围为结构、复制数值和引用归属。")
        else:
            st.error(out.get("error", "请求失败"))
        for item in out.get("limitations", []):
            st.caption(item["message"])
        with st.expander("工具调用轨迹"):
            st.json(out.get("trace", []))
        with st.expander("证据、知识引用和校验结果"):
            st.json(out)
