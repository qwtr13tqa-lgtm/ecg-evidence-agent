"""本地 Streamlit UI。

真实 ECG：本地推理和检索，不发送网关。
虚构报告：固定占位数据，通过网关生成报告。
"""

import os
import threading
import time
from pathlib import Path

# 在导入工作流前关闭远程追踪。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.pipeline import ECGAnalysisPipeline
from src.analysis.result import (
    ECGAnalysisResult,
    ECGInputInfo,
    ECGModelOutput,
    ECGEvidenceSummary,
)
from src.features.rhythm import RhythmFeatures
from src.knowledge.grounder import ECGKnowledgeGrounder
from src.reporting.generator import ECGReportGenerator
from src.agent.workflow import ECGReportWorkflow


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/Processed_PTBXL/test.npy"
KNOWLEDGE_PATH = ROOT / "data/knowledge/ecg_knowledge.jsonl"

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

st.set_page_config(
    page_title="ECG Agent",
    page_icon="🫀",
    layout="wide",
)


@st.cache_resource
def get_task_lock():
    return threading.Lock()


@st.cache_resource
def get_pipeline():
    # 仅在取得任务锁后调用。
    return ECGAnalysisPipeline()


def make_synthetic_analysis():
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
            "source_id": "SYNTHETIC_UI_DEMO",
            "data_kind": "synthetic_software_test",
        },
    )

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


def display_number(value, digits=3):
    return "不可用" if value is None else f"{value:.{digits}f}"


def show_sources(documents):
    if not documents:
        st.info("没有候选参考资料。")
        return

    for document in documents:
        title = f"{document['id']}｜{document['title']}"
        with st.expander(title):
            st.write(document["text"])
            st.caption(f"来源：{document['source']}")
            st.caption(f"定位：{document['locator']}")
            st.caption("候选参考资料，不代表已支持具体结论。")


def run_local(index):
    data = np.load(DATA_PATH, mmap_mode="r")

    if (
        data.ndim != 3
        or index >= len(data)
        or data.shape[1] < 4900
        or data.shape[2] != 12
    ):
        raise ValueError("样本索引或数据形状不正确")

    ecg = np.array(
        data[index, 100:4900, :],
        dtype=np.float32,
        copy=True,
    )

    pipeline = get_pipeline()
    result = pipeline.analyze(
        ecg,
        source_id="data/Processed_PTBXL/test.npy",
        sample_index=index,
        crop_start_sample=100,
    )

    grounder = ECGKnowledgeGrounder.from_jsonl(KNOWLEDGE_PATH)
    context = grounder.ground(result)

    return {
        "index": index,
        "ecg": ecg,
        "peaks": result.rhythm.r_peaks,
        "context": context,
    }


def run_demo():
    grounder = ECGKnowledgeGrounder.from_jsonl(KNOWLEDGE_PATH)
    analysis = make_synthetic_analysis()

    # 留下实际用于本次演示的来源，供报告展示。
    documents = grounder.ground(analysis)[
        "retrieved_knowledge"
    ]["documents"]

    with ECGReportGenerator() as generator:
        workflow = ECGReportWorkflow(grounder, generator)
        output = workflow.run(
            analysis,
            synthetic_only=True,
        )

    return {
        "workflow": output,
        "documents": documents,
    }


def execute_task(kind, function):
    """串行运行；按钮之外的页面重绘不会自动调用模型。"""
    lock = get_task_lock()

    if not lock.acquire(blocking=False):
        st.warning("已有任务运行中，请等待完成后再试。")
        return

    # 失败后不展示上一次结果冒充本次成功。
    st.session_state.pop(kind, None)
    started = time.perf_counter()

    try:
        label = (
            "正在加载模型、分析 ECG 并检索资料……"
            if kind == "local_result"
            else "正在运行虚构报告工作流；网关响应可能较慢……"
        )

        with st.spinner(label):
            value = function()

        value["elapsed"] = time.perf_counter() - started
        st.session_state[kind] = value

    except Exception as exc:
        # 不展示原始请求或异常正文，以免泄漏密钥。
        st.error(
            f"运行失败：{type(exc).__name__}。"
            "请检查文件、依赖或网关状态。"
        )
    finally:
        lock.release()


st.title("ECG 信号分析与证据解释")
st.caption("SGRF-Net · Signal Features · Knowledge Retrieval · LangGraph")
st.warning(
    "研究原型，不用于诊断。报告结构校验不等于医学正确性验证。"
)

mode = st.sidebar.radio(
    "选择模式",
    ["真实 ECG｜仅本地", "虚构报告｜网关演示"],
)

st.sidebar.caption(
    "两种模式互相独立。虚构报告不是所选真实 ECG 的报告。"
)
st.sidebar.caption(
    "请只打开一个演示页面。刷新页面可能丢失当前结果，"
    "也不能保证已发出的网关请求停止。"
)

if mode == "真实 ECG｜仅本地":
    st.subheader("真实 ECG 本地分析")
    st.info("只运行本地模型与知识检索，不调用报告网关。")

    index = st.number_input(
        "test.npy 样本索引（从 0 开始）",
        min_value=0,
        value=0,
        step=1,
    )

    if st.button("开始本地分析", type="primary"):
        execute_task(
            "local_result",
            lambda: run_local(int(index)),
        )

    saved = st.session_state.get("local_result")

    if saved is not None:
        context = saved["context"]
        rhythm = context["signal_features"]["rhythm"]
        model = context["model"]

        st.success(
            f"样本 {saved['index']} 分析完成｜"
            f"耗时 {saved['elapsed']:.2f} 秒"
        )

        if int(index) != saved["index"]:
            st.warning("下面仍是上次运行的样本结果；修改索引后需重新分析。")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "估计心率 / bpm",
            display_number(rhythm["heart_rate_bpm"], 2),
        )
        c2.metric(
            "平均 RR / 秒",
            display_number(rhythm["mean_rr_seconds"]),
        )
        c3.metric(
            "候选峰数",
            rhythm["candidate_beat_count"],
        )
        c4.metric(
            "模型原始分数",
            display_number(model["anomaly_score"], 4),
        )
        st.caption("心率未验证；模型原始分数不是异常概率。")

        lead_index = st.selectbox(
            "显示导联（按项目预期列顺序）",
            options=list(range(12)),
            index=1,
            format_func=lambda value: LEADS[value],
        )

        ecg = saved["ecg"]
        time_axis = np.arange(len(ecg)) / 500.0
        figure = go.Figure()

        figure.add_trace(go.Scatter(
            x=time_axis,
            y=ecg[:, lead_index],
            mode="lines",
            name=LEADS[lead_index],
            line={"color": "#2563eb", "width": 1},
        ))

        # 当前节律提取使用第 2 列；不将峰位置冒充其他导联检测结果。
        if lead_index == 1:
            peaks = np.asarray(saved["peaks"], dtype=int)
            if len(peaks):
                figure.add_trace(go.Scatter(
                    x=peaks / 500.0,
                    y=ecg[peaks, lead_index],
                    mode="markers",
                    name="候选峰（未验证）",
                    marker={"color": "#dc2626", "size": 7},
                ))

        figure.update_layout(
            height=350,
            xaxis_title="相对于裁剪输入片段的时间 / 秒",
            yaxis_title="输入信号幅值（单位未核实）",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        st.plotly_chart(figure, use_container_width=True)

        st.subheader("模型证据")
        left, right = st.columns(2)
        with left:
            st.write("导联相对证据")
            st.dataframe(
                context["evidence"]["lead_evidence"],
                use_container_width=True,
            )
        with right:
            st.write("模型关注时间区域")
            st.dataframe(
                context["evidence"]["temporal_regions"],
                use_container_width=True,
            )
            st.caption("start/end/duration 为采样点，不是秒。")

        st.subheader("候选参考资料")
        show_sources(context["retrieved_knowledge"]["documents"])

        with st.expander("完整 Context 与解释限制"):
            st.json(context)

else:
    st.subheader("虚构数据报告演示")
    st.warning(
        "所有数值为程序预设占位值，不是真实测量。"
        "本模式不读取 test.npy，也不使用本地分析页的结果。"
    )

    consent = st.checkbox(
        "我确认发送的仅为虚构数据，并接受 HTTP 明文传输 Token "
        "以及可能产生的模型调用费用。"
    )
    has_key = bool(os.environ.get("ECG_API_KEY", "").strip())

    if not has_key:
        st.info(
            "当前进程没有 ECG_API_KEY。"
            "请在终端设置并 export 后，重新启动 Streamlit。"
        )

    if st.button(
        "生成虚构报告",
        type="primary",
        disabled=not (consent and has_key),
    ):
        execute_task("demo_result", run_demo)

    saved = st.session_state.get("demo_result")

    if saved is not None:
        workflow = saved["workflow"]
        st.caption(f"本次工作流耗时：{saved['elapsed']:.2f} 秒")

        if workflow["status"] != "completed_draft":
            st.error("工作流失败，未返回可展示报告。")
            st.json(workflow["error"])
        else:
            output = workflow["output"]
            report = output["report"]

            st.success("结构校验通过，仍需人工复核。")
            st.subheader("摘要")
            st.write(report["summary"])

            st.subheader("解释与引用")
            if not report["explanations"]:
                st.info("本次未生成解释。")

            source_map = {
                item["id"]: item
                for item in saved["documents"]
            }

            for explanation in report["explanations"]:
                st.markdown(f"**{explanation['id']}**")
                st.write(explanation["text"])
                st.caption(
                    "观察 ID："
                    + ", ".join(explanation["observation_ids"])
                )

                for knowledge_id in explanation["knowledge_ids"]:
                    document = source_map.get(knowledge_id)
                    if document:
                        with st.expander(
                            f"引用 {knowledge_id}｜{document['title']}"
                        ):
                            st.write(document["text"])
                            st.caption(f"来源：{document['source']}")
                            st.caption(f"定位：{document['locator']}")
                    else:
                        st.warning(f"没有找到来源详情：{knowledge_id}")

            st.subheader("固定限制")
            for limitation in report["limitations"]:
                st.write(f"• {limitation['message']}")

            with st.expander("观察值（代码复制）"):
                st.json(report["observations"])

            with st.expander("校验范围与完整结果"):
                st.json(output)

            st.caption(
                "自然语言可能仍有表述或引用支持问题。"
                "不要将结构校验结果展示为临床验证通过。"
            )