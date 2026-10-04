#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /usr/bin/python3 ui_sgrfnet_three_entry.py
import os
import sys
import traceback
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("Agg")


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

REAL_MODEL_AVAILABLE = False
IMPORT_ERROR = None
try:
    import torch
    from scipy.signal import stft
    REAL_MODEL_AVAILABLE = True
except Exception as e:
    IMPORT_ERROR = e
    REAL_MODEL_AVAILABLE = False

WFDB_AVAILABLE = False
WFDB_IMPORT_ERROR = None
try:
    import wfdb
    import pandas as pd
    import ast
    WFDB_AVAILABLE = True
except Exception as e:
    WFDB_IMPORT_ERROR = e
    WFDB_AVAILABLE = False

HEARTPY_AVAILABLE = False
HEARTPY_IMPORT_ERROR = None
try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except Exception as e:
    HEARTPY_IMPORT_ERROR = e
    HEARTPY_AVAILABLE = False

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def normalize_instance(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    out = data.copy()
    if out.ndim != 2:
        raise ValueError("normalize_instance 仅支持二维数组。")
    for ch in range(out.shape[1]):
        seq = out[:, ch]
        dmin = float(seq.min())
        dmax = float(seq.max())
        if dmax - dmin > 1e-6:
            out[:, ch] = 2.0 * (seq - dmin) / (dmax - dmin) - 1.0
        else:
            out[:, ch] = 0.0
    return out


def hp_preprocess_single(ecg: np.ndarray) -> np.ndarray:
    if not HEARTPY_AVAILABLE:
        return ecg.astype(np.float32)
    ecg = np.asarray(ecg, dtype=np.float32)
    out = []
    for lead in range(ecg.shape[1]):
        x = ecg[:, lead]
        try:
            filtered = hp.filter_signal(x, sample_rate=500, filtertype="highpass", cutoff=1)
            filt = hp.filter_signal(filtered, sample_rate=500, cutoff=35, filtertype="notch")
            y = hp.filter_signal(filt, sample_rate=500, filtertype="lowpass", cutoff=25)
            out.append(y.astype(np.float32))
        except Exception:
            out.append(x.astype(np.float32))
    return np.asarray(out, dtype=np.float32).T


def compute_spectrogram(time_data: np.ndarray) -> np.ndarray:
    if not REAL_MODEL_AVAILABLE:
        raise RuntimeError("缺少 scipy.signal.stft，无法计算频谱图。")
    _, _, zxx = stft(time_data.transpose(1, 0), fs=500, window="hann", nperseg=125)
    spec = np.abs(zxx)
    return spec.transpose(1, 2, 0).astype(np.float32)


def ensure_2d_12lead(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError("检测入口需要单条样本；当前数组为三维，请使用 test.npy + 索引入口。")
    if data.ndim != 2:
        raise ValueError("输入数据必须为二维数组，形状类似 (5000,12) 或 (4800,12)。")
    if data.shape[1] != 12 and data.shape[0] == 12:
        data = data.T
    if data.shape[1] != 12:
        raise ValueError("当前输入不是 12 导联数据，收到形状 {}。".format(data.shape))
    return data.astype(np.float32)


def crop_or_pad_to_5000(data: np.ndarray) -> np.ndarray:
    n, c = data.shape
    if c != 12:
        raise ValueError("仅支持 12 导联输入。")
    if n == 5000:
        return data
    if n > 5000:
        start = (n - 5000) // 2
        return data[start:start + 5000, :]
    padded = np.zeros((5000, 12), dtype=np.float32)
    padded[:n, :] = data
    if n > 0:
        padded[n:, :] = data[-1, :]
    return padded


class DemoDetector:
    def infer(self, ecg: np.ndarray):
        ecg = normalize_instance(ecg)
        recon = ecg.copy()
        kernel = np.ones(9, dtype=np.float32) / 9.0
        for i in range(recon.shape[1]):
            recon[:, i] = np.convolve(recon[:, i], kernel, mode="same")
        err_map = (ecg - recon) ** 2
        lead_scores = err_map.mean(axis=0)
        score = float(lead_scores.mean())
        pred = "异常" if score >= 0.020 else "正常"
        return {
            "input": ecg,
            "recon": recon,
            "score": score,
            "lead_scores": lead_scores,
            "pred": pred,
            "mode": "演示模式",
        }


class RealSGRFDetector:
    def __init__(self, project_dir: str, weight_path: str, device: str = "cpu"):
        self.project_dir = Path(project_dir)
        self.weight_path = Path(weight_path)
        self.device = torch.device(device)
        self.model = None
        self._load_model()

    def _load_model(self):
        sys.path.insert(0, str(self.project_dir))
        sys.path.insert(0, str(self.project_dir / "lib"))
        from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX

        self.model = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
            enc_in=12,
            channel=12,
            d_model=96,
            shapex_num_shapelets=16,
            shapex_shapelet_len=96,
            shapex_use_encoder=True,
        ).to(self.device)

        ckpt = torch.load(self.weight_path, map_location=self.device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def infer(self, ecg: np.ndarray):
        ecg = normalize_instance(ecg)
        time_instance = ecg[100:4900, :]
        if time_instance.shape != (4800, 12):
            raise ValueError("真实模型要求裁剪后为 (4800,12)，当前为 {}。".format(time_instance.shape))

        spec = compute_spectrogram(time_instance)
        with torch.no_grad():
            t = torch.tensor(time_instance, dtype=torch.float32).unsqueeze(0).to(self.device)
            s = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(self.device)
            recon, sigma = self.model(t, s, return_aux=False)
            recon = recon[0].detach().cpu().numpy()
            sigma = sigma[0].detach().cpu().numpy()

        err_map = (recon - time_instance) ** 2
        sigma = np.clip(sigma, -5.0, 5.0)
        sigma = np.repeat(sigma, 12, axis=1) if sigma.shape[1] == 1 else sigma
        score_map = np.exp(-sigma) * err_map + sigma
        lead_scores = score_map.mean(axis=0)
        score = float(score_map.mean())
        pred = "异常" if score >= 0.12 else "正常"
        return {
            "input": time_instance,
            "recon": recon,
            "score": score,
            "lead_scores": lead_scores,
            "pred": pred,
            "mode": "SGRFNet真实模型",
        }


def load_single_sample_from_npy(path: str) -> np.ndarray:
    data = np.load(path)
    return ensure_2d_12lead(data)


def load_sample_from_test_npy(path: str, index: int) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError("test.npy 应为三维数组 (N,5000,12)，当前为 {}。".format(data.shape))
    if index < 0 or index >= data.shape[0]:
        raise IndexError("样本索引越界：index={}，有效范围为 0 到 {}。".format(index, data.shape[0] - 1))
    return ensure_2d_12lead(data[index])


def load_single_ptbxl_record(dataset_dir: str, ecg_id: int, sampling_rate: int = 500):
    if not WFDB_AVAILABLE:
        raise RuntimeError("缺少 wfdb/pandas，无法读取原始 PTB-XL 记录：{}".format(WFDB_IMPORT_ERROR))

    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError("未找到 {}".format(csv_path))

    df = pd.read_csv(csv_path, index_col="ecg_id")
    if ecg_id not in df.index:
        raise KeyError("ecg_id={} 不存在于 ptbxl_database.csv 中。".format(ecg_id))
    row = df.loc[ecg_id]

    rel_path = row["filename_hr"] if sampling_rate == 500 else row["filename_lr"]
    record_path = str(dataset_dir / rel_path)
    signal, meta = wfdb.rdsamp(record_path)
    signal = ensure_2d_12lead(signal)

    diag_text = ""
    try:
        scp = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
        diag_text = str(scp)
    except Exception:
        diag_text = str(row.get("scp_codes", ""))

    info = {
        "ecg_id": int(ecg_id),
        "record_path": record_path,
        "sampling_rate": sampling_rate,
        "patient_id": row.get("patient_id", ""),
        "age": row.get("age", ""),
        "sex": row.get("sex", ""),
        "report": row.get("report", ""),
        "scp_codes": diag_text,
    }
    return signal.astype(np.float32), info


class SGRFNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SGRFNet 心电异常检测软件（三入口版）")
        self.root.geometry("1500x920")
        self.root.minsize(1300, 820)

        self.detector = DemoDetector()
        self.current_result = None
        self.current_source_desc = "未加载样本"

        self.entry_mode = tk.StringVar(value="single_npy")
        self.single_npy_path = tk.StringVar(value="")
        self.test_npy_path = tk.StringVar(value="")
        self.test_index_var = tk.StringVar(value="0")
        self.ptbxl_dir = tk.StringVar(value="")
        self.ptbxl_ecg_id = tk.StringVar(value="")
        self.ptbxl_sr = tk.StringVar(value="500")

        self._build_style()
        self._build_ui()
        self.log("软件已启动，当前默认使用演示模式。")
        if IMPORT_ERROR is not None:
            self.log("提示：未检测到完整深度学习环境，自动进入演示模式。{}".format(IMPORT_ERROR))
        if not HEARTPY_AVAILABLE:
            self.log("提示：未检测到 heartpy，原始记录预处理将跳过滤波。{}".format(HEARTPY_IMPORT_ERROR))
        if not WFDB_AVAILABLE:
            self.log("提示：未检测到 wfdb/pandas，暂时无法读取 PTB-XL 原始记录。{}".format(WFDB_IMPORT_ERROR))

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel")
        style.configure("Head.TLabel")
        style.configure("Info.TLabel")
        style.configure("Primary.TButton")

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        top.pack(fill=tk.X)
        ttk.Label(top, text="SGRFNet：基于形状引导残差融合网络的十二导联心电异常检测软件", style="Title.TLabel").pack(anchor="center")
        ttk.Label(
            top,
            text="支持三种数据入口：单条预处理样本、test.npy 指定索引样本、PTB-XL 原始记录 ecg_id",
            style="Info.TLabel",
        ).pack(anchor="center", pady=(6, 0))

        ctrl = ttk.LabelFrame(self.root, text="控制面板", padding=12)
        ctrl.pack(fill=tk.X, padx=18, pady=8)

        ttk.Button(ctrl, text="加载 SGRFNet 权重", style="Primary.TButton", command=self.load_real_model).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(ctrl, text="开始检测", style="Primary.TButton", command=self.run_detection).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(ctrl, text="导出结果", style="Primary.TButton", command=self.export_result).grid(row=0, column=2, padx=6, pady=6)

        self.model_var = tk.StringVar(value="当前模型：演示模式")
        self.source_var = tk.StringVar(value="当前样本：未选择")
        ttk.Label(ctrl, textvariable=self.model_var, style="Info.TLabel").grid(row=0, column=3, columnspan=3, sticky="e", padx=6, pady=4)
        ttk.Label(ctrl, textvariable=self.source_var, style="Info.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=4)

        entry_box = ttk.LabelFrame(self.root, text="数据入口", padding=12)
        entry_box.pack(fill=tk.X, padx=18, pady=(0, 8))

        ttk.Radiobutton(entry_box, text="入口1：单条预处理样本 .npy", variable=self.entry_mode, value="single_npy").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(entry_box, textvariable=self.single_npy_path, width=80).grid(row=0, column=1, padx=4, pady=4, sticky="we")
        ttk.Button(entry_box, text="选择文件", command=self.pick_single_npy).grid(row=0, column=2, padx=4, pady=4)

        ttk.Radiobutton(entry_box, text="入口2：test.npy + 样本索引", variable=self.entry_mode, value="test_npy").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(entry_box, textvariable=self.test_npy_path, width=60).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(entry_box, text="选择 test.npy", command=self.pick_test_npy).grid(row=1, column=2, padx=4, pady=4)
        ttk.Label(entry_box, text="索引").grid(row=1, column=3, padx=(12, 4), pady=4)
        ttk.Entry(entry_box, textvariable=self.test_index_var, width=10).grid(row=1, column=4, padx=4, pady=4)

        ttk.Radiobutton(entry_box, text="入口3：PTB-XL 原始记录目录 + ecg_id", variable=self.entry_mode, value="ptbxl_raw").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(entry_box, textvariable=self.ptbxl_dir, width=60).grid(row=2, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(entry_box, text="选择数据集目录", command=self.pick_ptbxl_dir).grid(row=2, column=2, padx=4, pady=4)
        ttk.Label(entry_box, text="ecg_id").grid(row=2, column=3, padx=(12, 4), pady=4)
        ttk.Entry(entry_box, textvariable=self.ptbxl_ecg_id, width=10).grid(row=2, column=4, padx=4, pady=4)
        ttk.Label(entry_box, text="采样率").grid(row=2, column=5, padx=(12, 4), pady=4)
        ttk.Combobox(entry_box, textvariable=self.ptbxl_sr, values=["500", "100"], width=8, state="readonly").grid(row=2, column=6, padx=4, pady=4)

        entry_box.columnconfigure(1, weight=1)

        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=18, pady=(4, 18))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(content, text="信号与重构可视化", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ttk.LabelFrame(content, text="检测结果与运行日志", padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        self.fig = Figure(figsize=(8.5, 6.8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder()

        result_box = ttk.Frame(right)
        result_box.pack(fill=tk.X, pady=(0, 10))
        self.pred_var = tk.StringVar(value="检测结论：--")
        self.score_var = tk.StringVar(value="异常分数：--")
        self.lead_var = tk.StringVar(value="最高风险导联：--")
        self.meta_var = tk.StringVar(value="样本说明：--")
        ttk.Label(result_box, textvariable=self.pred_var, style="Head.TLabel").pack(anchor="w", pady=3)
        ttk.Label(result_box, textvariable=self.score_var, style="Head.TLabel").pack(anchor="w", pady=3)
        ttk.Label(result_box, textvariable=self.lead_var, style="Head.TLabel").pack(anchor="w", pady=3)
        ttk.Label(result_box, textvariable=self.meta_var, style="Info.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=3)

        self.tree = ttk.Treeview(right, columns=("lead", "score"), show="headings", height=10)
        self.tree.heading("lead", text="导联")
        self.tree.heading("score", text="异常得分")
        self.tree.column("lead", width=100, anchor="center")
        self.tree.column("score", width=120, anchor="center")
        self.tree.pack(fill=tk.X, pady=(4, 10))

        ttk.Label(right, text="日志输出", style="Head.TLabel").pack(anchor="w")
        self.log_text = tk.Text(right, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _draw_placeholder(self):
        self.ax1.clear()
        self.ax2.clear()
        x = np.linspace(0, 10, 1000)
        y = np.sin(2 * np.pi * 1.2 * x) + 0.2 * np.sin(2 * np.pi * 4 * x)
        self.ax1.plot(x, y)

        self.ax2.bar(LEAD_NAMES, np.linspace(0.01, 0.12, 12))

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def pick_single_npy(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy ECG File", "*.npy"), ("All Files", "*.*")])
        if path:
            self.single_npy_path.set(path)
            self.entry_mode.set("single_npy")
            self.source_var.set("当前样本：{}".format(path))
            self.log("入口1 已选择文件：{}".format(path))

    def pick_test_npy(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy ECG File", "*.npy"), ("All Files", "*.*")])
        if path:
            self.test_npy_path.set(path)
            self.entry_mode.set("test_npy")
            self.source_var.set("当前样本：{} | 索引 {}".format(path, self.test_index_var.get()))
            self.log("入口2 已选择 test.npy：{}".format(path))

    def pick_ptbxl_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.ptbxl_dir.set(path)
            self.entry_mode.set("ptbxl_raw")
            self.source_var.set("当前样本：{} | ecg_id {}".format(path, self.ptbxl_ecg_id.get()))
            self.log("入口3 已选择 PTB-XL 目录：{}".format(path))

    def load_real_model(self):
        if not REAL_MODEL_AVAILABLE:
            messagebox.showwarning("环境不完整", "当前环境缺少 torch 或 scipy，无法加载真实模型，将继续使用演示模式。\n{}".format(IMPORT_ERROR))
            return
        project_dir = filedialog.askdirectory(title="选择项目根目录（需包含 lib/SGRFNet.py）")
        if not project_dir:
            return
        weight_path = filedialog.askopenfilename(title="选择模型权重", filetypes=[("PyTorch Model", "*.pt *.pth"), ("All Files", "*.*")])
        if not weight_path:
            return
        try:
            device = "cpu"
            self.detector = RealSGRFDetector(project_dir, weight_path, device=device)
            self.model_var.set("当前模型：SGRFNet真实模型（{}）".format(device))
            self.log("真实模型加载成功：{}".format(weight_path))
        except Exception as e:
            self.detector = DemoDetector()
            self.model_var.set("当前模型：演示模式")
            self.log("真实模型加载失败，已退回演示模式。")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("模型加载失败", str(e))

    def _resolve_current_sample(self):
        mode = self.entry_mode.get()

        if mode == "single_npy":
            path = self.single_npy_path.get().strip()
            if not path:
                raise ValueError("请先选择单条预处理样本 .npy 文件。")
            sample = load_single_sample_from_npy(path)
            source_desc = "入口1：单条 .npy | {}".format(path)
            meta_text = "来源：单条预处理样本；原始形状：{}".format(sample.shape)
            return sample, source_desc, meta_text

        if mode == "test_npy":
            path = self.test_npy_path.get().strip()
            if not path:
                raise ValueError("请先选择 test.npy 文件。")
            index = int(self.test_index_var.get().strip())
            sample = load_sample_from_test_npy(path, index)
            source_desc = "入口2：test.npy | {} | index={}".format(path, index)
            meta_text = "来源：test.npy 第 {} 条样本；样本形状：{}".format(index, sample.shape)
            return sample, source_desc, meta_text

        if mode == "ptbxl_raw":
            dataset_dir = self.ptbxl_dir.get().strip()
            ecg_id_text = self.ptbxl_ecg_id.get().strip()
            if not dataset_dir:
                raise ValueError("请先选择 PTB-XL 数据集目录。")
            if not ecg_id_text:
                raise ValueError("请先输入 ecg_id。")
            ecg_id = int(ecg_id_text)
            sr = int(self.ptbxl_sr.get().strip())
            raw_signal, info = load_single_ptbxl_record(dataset_dir, ecg_id, sampling_rate=sr)
            raw_signal = crop_or_pad_to_5000(raw_signal)
            preprocessed = hp_preprocess_single(raw_signal)
            source_desc = "入口3：PTB-XL 原始记录 | ecg_id={}".format(ecg_id)
            meta_text = "来源：PTB-XL 原始记录 ecg_id={}；采样率={}；patient_id={}；age={}；sex={}".format(
                info["ecg_id"], info["sampling_rate"], info["patient_id"], info["age"], info["sex"]
            )
            extra = "scp_codes={}".format(info["scp_codes"])
            self.log(meta_text)
            if extra:
                self.log(extra)
            return preprocessed, source_desc, meta_text

        raise ValueError("未知入口模式：{}".format(mode))

    def run_detection(self):
        threading.Thread(target=self._run_detection_worker, daemon=True).start()

    def _run_detection_worker(self):
        try:
            self.log("开始检测...")
            sample, source_desc, meta_text = self._resolve_current_sample()
            self.current_source_desc = source_desc
            self.source_var.set("当前样本：{}".format(source_desc))
            result = self.detector.infer(sample)
            result["meta_text"] = meta_text
            self.current_result = result
            self.root.after(0, self.update_result_view)
            self.log("检测完成。模式：{}，结论：{}，异常分数：{:.6f}".format(result["mode"], result["pred"], result["score"]))
        except Exception as e:
            err_msg = str(e)
            self.log("检测失败：{}".format(err_msg))
            self.log(traceback.format_exc())
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("检测失败", msg))


    def update_result_view(self):
        if not self.current_result:
            return
        res = self.current_result
        best_idx = int(np.argmax(res["lead_scores"]))

        self.pred_var.set("检测结论：{}（{}）".format(res["pred"], res["mode"]))
        self.score_var.set("异常分数：{:.6f}".format(res["score"]))
        self.lead_var.set("最高风险导联：{}（{:.6f}）".format(LEAD_NAMES[best_idx], res["lead_scores"][best_idx]))
        self.meta_var.set(res.get("meta_text", ""))

        for i in self.tree.get_children():
            self.tree.delete(i)
        for ld, sc in zip(LEAD_NAMES, res["lead_scores"]):
            self.tree.insert("", tk.END, values=(ld, "{:.6f}".format(float(sc))))

        sig = res["input"]
        recon = res["recon"]
        x = np.arange(sig.shape[0])

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(x, sig[:, best_idx], label="Input signal", linewidth=1.0)
        self.ax1.plot(x, recon[:, best_idx], label="Reconstructed signal", linewidth=1.0)

        self.ax1.legend()

        self.ax2.bar(LEAD_NAMES, res["lead_scores"])

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def export_result(self):
        if not self.current_result:
            messagebox.showinfo("提示", "当前没有可导出的结果。")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
        if not save_path:
            return
        res = self.current_result
        lines = [
            "SGRFNet ECG Anomaly Detection Result",
            "source: {}".format(self.current_source_desc),
            "mode: {}".format(res["mode"]),
            "prediction: {}".format(res["pred"]),
            "score: {:.6f}".format(res["score"]),
            "meta: {}".format(res.get("meta_text", "")),
            "lead_scores:",
        ]
        for ld, sc in zip(LEAD_NAMES, res["lead_scores"]):
            lines.append("  {}: {:.6f}".format(ld, float(sc)))
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        self.log("结果已导出：{}".format(save_path))
        messagebox.showinfo("导出成功", "结果已保存到\n{}".format(save_path))


def main():
    root = tk.Tk()
    app = SGRFNetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
