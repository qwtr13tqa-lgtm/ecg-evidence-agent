"""ECG Agent 最小本地 UI。

真实 ECG：本地分析与检索，不发送网关。
虚构报告：固定虚构数据，通过网关生成报告。
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# 必须在导入业务模块前关闭远程追踪。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import numpy as np

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


def make_synthetic_analysis():
    """固定占位值，不读取任何真实 ECG。"""
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
            "data_kind": "synthetic_software_test",
            "source_id": "SYNTHETIC_UI_DEMO",
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


class ECGApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ECG Agent — 辅助分析研究原型")
        self.root.geometry("1150x850")

        self.events = queue.Queue()
        self.busy = False
        self.started = 0.0
        self.phase = "就绪"
        self.pipeline = None

        ttk.Label(
            root,
            text="ECG 信号分析与证据解释",
            font=("", 18, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 4))

        ttk.Label(
            root,
            text=(
                "研究演示，不用于诊断。"
                "结构校验通过不代表医学正确或引用支持关系已验证。"
            ),
        ).pack(anchor="w", padx=16, pady=(0, 10))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=12)

        local_tab = ttk.Frame(notebook)
        demo_tab = ttk.Frame(notebook)
        notebook.add(local_tab, text="真实 ECG｜仅本地")
        notebook.add(demo_tab, text="虚构报告｜网关演示")

        controls = ttk.Frame(local_tab)
        controls.pack(fill="x", padx=10, pady=10)

        ttk.Label(controls, text="测试集样本索引：").pack(side="left")
        self.sample_index = tk.StringVar(value="0")
        self.index_entry = ttk.Entry(
            controls, textvariable=self.sample_index, width=10
        )
        self.index_entry.pack(side="left", padx=8)

        self.local_button = ttk.Button(
            controls, text="开始本地分析", command=self.start_local
        )
        self.local_button.pack(side="left")

        ttk.Label(
            local_tab,
            text=(
                "读取 test.npy，裁剪 [100:4900]。"
                "模型证据、节律及检索均在本地处理。"
            ),
        ).pack(anchor="w", padx=10)

        self.canvas = tk.Canvas(
            local_tab, height=210, background="#f8fafc",
            highlightthickness=0,
        )
        self.canvas.pack(fill="x", padx=10, pady=8)

        self.local_text = ScrolledText(
            local_tab, wrap="word", font=("Monospace", 10)
        )
        self.local_text.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Label(
            demo_tab,
            text=(
                "仅发送程序内置的虚构占位数据和检索资料。"
                "不是所选真实 ECG 的报告。"
            ),
            foreground="#a04b00",
        ).pack(anchor="w", padx=10, pady=10)

        self.demo_button = ttk.Button(
            demo_tab,
            text="生成虚构演示报告（会调用网关）",
            command=self.start_demo,
        )
        self.demo_button.pack(anchor="w", padx=10, pady=5)

        self.demo_text = ScrolledText(
            demo_tab, wrap="word", font=("Monospace", 10)
        )
        self.demo_text.pack(fill="both", expand=True, padx=10, pady=8)

        self.status = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status).pack(
            anchor="w", padx=16, pady=8
        )

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=12, pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self.poll)

    @staticmethod
    def write(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def phase_update(self, message):
        # 后台线程不能直接操作 Tk 控件。
        self.events.put(("phase", message))

    def launch(self, mode, job):
        if self.busy:
            return

        self.busy = True
        self.started = time.monotonic()
        self.phase = "准备运行"
        self.local_button.configure(state="disabled")
        self.demo_button.configure(state="disabled")
        self.index_entry.configure(state="disabled")
        self.progress.start(12)

        target = self.local_text if mode == "local" else self.demo_text
        self.write(target, "运行中，请等待；不要重复提交。")
        if mode == "local":
            self.canvas.delete("all")

        def worker():
            try:
                value = job()
                self.events.put(("success", mode, value))
            except Exception as exc:
                # 不显示完整异常响应或密钥。
                self.events.put(("error", mode, type(exc).__name__))

        threading.Thread(target=worker, daemon=True).start()

    def start_local(self):
        if self.busy:
            return

        try:
            index = int(self.sample_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "样本索引必须是非负整数。")
            return

        def job():
            self.phase_update("读取本地 ECG")
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

            if self.pipeline is None:
                self.phase_update("首次加载 SGRF-Net 权重")
                self.pipeline = ECGAnalysisPipeline()

            self.phase_update("本地模型推理与节律分析")
            result = self.pipeline.analyze(
                ecg,
                source_id="data/Processed_PTBXL/test.npy",
                sample_index=index,
                crop_start_sample=100,
            )

            self.phase_update("本地知识检索")
            grounder = ECGKnowledgeGrounder.from_jsonl(KNOWLEDGE_PATH)
            context = grounder.ground(result)

            return {
                "lead": ecg[:, 1].tolist(),
                "peaks": result.rhythm.r_peaks,
                "context": context,
            }

        self.launch("local", job)

    def start_demo(self):
        if self.busy:
            return

        if not os.environ.get("ECG_API_KEY", "").strip():
            messagebox.showerror(
                "缺少密钥",
                "请在启动 UI 的终端设置并 export ECG_API_KEY，"
                "然后重新启动 UI。",
            )
            return

        confirmed = messagebox.askyesno(
            "确认虚构数据调用",
            "将调用学校 HTTP 网关，可能产生费用。\n"
            "Token 会通过明文 HTTP 传输。\n"
            "仅发送固定虚构数据，不发送真实 ECG。\n\n"
            "是否继续？",
        )
        if not confirmed:
            return

        def job():
            self.phase_update("检索与报告生成，等待网关响应")
            grounder = ECGKnowledgeGrounder.from_jsonl(KNOWLEDGE_PATH)

            with ECGReportGenerator() as generator:
                workflow = ECGReportWorkflow(grounder, generator)
                return workflow.run(
                    make_synthetic_analysis(),
                    synthetic_only=True,
                )

        self.launch("demo", job)

    def draw_wave(self, lead, peaks):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 400)
        height = 210
        values = np.asarray(lead)

        low, high = float(values.min()), float(values.max())
        span = high - low if high > low else 1.0

        def point(index):
            x = 30 + index / (len(values) - 1) * (width - 60)
            y = height - 30 - (values[index] - low) / span * (height - 65)
            return float(x), float(y)

        coordinates = []
        for index in range(len(values)):
            coordinates.extend(point(index))

        self.canvas.create_line(
            *coordinates, fill="#2563eb", width=1
        )

        for peak in peaks:
            peak = int(peak)
            if 0 <= peak < len(values):
                x, y = point(peak)
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3,
                    fill="#dc2626", outline="",
                )

        self.canvas.create_text(
            12, 12, anchor="nw",
            text=(
                "输入第 2 列（预期 Lead II）｜红点：候选峰，未验证"
                "｜纵轴：输入数据单位，非已确认 mV"
            ),
        )
        self.canvas.create_text(30, 198, text="0 s", anchor="w")
        self.canvas.create_text(
            width - 30, 198, text="约 9.6 s", anchor="e"
        )

    def finish(self):
        self.busy = False
        self.progress.stop()
        self.local_button.configure(state="normal")
        self.demo_button.configure(state="normal")
        self.index_entry.configure(state="normal")

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "phase":
                    self.phase = event[1]
                    continue

                _, mode, value = event
                elapsed = time.monotonic() - self.started
                target = (
                    self.local_text if mode == "local" else self.demo_text
                )

                if kind == "error":
                    self.write(
                        target,
                        f"运行失败：{value}\n"
                        "请检查本地文件、依赖或网关状态。"
                        "界面未展示原始异常，以免泄漏请求信息。",
                    )
                    self.phase = "失败"
                elif mode == "local":
                    self.draw_wave(value["lead"], value["peaks"])
                    self.write(
                        target,
                        "真实 ECG 本地分析结果；未调用网关。\n\n"
                        + json.dumps(
                            value["context"],
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        ),
                    )
                    self.phase = "本地分析完成"
                else:
                    prefix = (
                        "虚构报告草稿，需人工复核；"
                        "所有数值均为预设占位值。\n"
                        "不是当前真实 ECG 样本的报告。\n\n"
                    )
                    self.write(
                        target,
                        prefix + json.dumps(
                            value, ensure_ascii=False, indent=2
                        ),
                    )
                    self.phase = (
                        "虚构报告完成"
                        if value["status"] == "completed_draft"
                        else "工作流失败"
                    )

                self.finish()
                self.status.set(f"{self.phase}｜耗时 {elapsed:.1f} 秒")

        except queue.Empty:
            pass

        if self.busy:
            elapsed = time.monotonic() - self.started
            self.status.set(f"{self.phase}｜已运行 {elapsed:.1f} 秒")

        self.root.after(150, self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo(
                "任务仍在运行",
                "请等待当前任务完成。\n"
                "终止本地程序不保证网关停止生成或计费。",
            )
            return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ECGApp(root)
    root.mainloop()