"""ECG 分析统一入口。

输入 ECG
    ├── SGRF-Net → 模型证据
    └── RhythmFeatureExtractor → 节律特征
                    ↓
             ECGAnalysisResult

当前接口要求输入已裁剪的 (4800, 12) ECG。
不自动裁剪、不执行临床诊断、不生成校准置信度。
"""

from __future__ import annotations

from pathlib import Path
import hashlib
from threading import RLock
from typing import Optional, Union

import numpy as np

from src.analysis.result import ECGAnalysisResult
from src.analysis.store import AnalysisStore
from src.features.rhythm import RhythmFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "ckpt_shape_guided_shapex_gate/"
    "best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt"
)


class ECGAnalysisPipeline:
    """复用已加载的模型，分析单条 ECG。

    Args:
        checkpoint: 模型权重路径。相对路径按项目根目录解析。
        device: 例如 "cuda"、"cpu"；None 由模型适配器选择。
        sampling_rate: 当前流程固定使用 500 Hz。
        lead_index: 节律分析使用的导联列索引，默认 1。
            输入导联顺序须与模型训练时一致。
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        sampling_rate: int = 500,
        lead_index: int = 1,
        store: Optional[AnalysisStore] = None,
    ) -> None:
        if sampling_rate != 500:
            raise ValueError(
                "当前 Pipeline 仅支持 500 Hz 输入，"
                "其他采样率请先进行经过验证的重采样。"
            )

        if (
            isinstance(lead_index, bool)
            or not isinstance(lead_index, (int, np.integer))
            or not 0 <= lead_index < 12
        ):
            raise ValueError("lead_index 必须是 0 到 11 的整数。")

        checkpoint_path = (
            DEFAULT_CHECKPOINT
            if checkpoint is None
            else Path(checkpoint).expanduser()
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        self.checkpoint = checkpoint_path
        self.sampling_rate = int(sampling_rate)
        self.lead_index = int(lead_index)
        self.store = store if store is not None else AnalysisStore()
        self._analysis_lock = RLock()
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()

        # 只在初始化时加载权重。
        from src.inference.sgrf_adapter import SGRFDetector
        from src.evidence.extractor import EvidenceExtractor

        self.detector = SGRFDetector(
            checkpoint=self.checkpoint,
            device=device,
        )

        self.evidence_extractor = EvidenceExtractor()

        self.rhythm_extractor = RhythmFeatureExtractor(
            sampling_rate=self.sampling_rate,
            lead_index=self.lead_index,
        )

    @property
    def device(self):
        return self.detector.device

    def analyze(
        self, ecg, *, source_id=None, sample_index=None, crop_start_sample=0,
    ) -> ECGAnalysisResult:
        """Record every submitted analysis; keep the existing return interface.

        Initialization/checkpoint failures happen before a record can be made.
        Store capacity failures do not allocate a new ID. Records are in-memory.
        """
        analysis_id = self.store.begin({"pipeline_version": "stage1-1.0"})
        try:
            with self._analysis_lock:
                signal = self._validate_ecg(ecg)
                result = self._analyze(
                    signal, source_id=source_id, sample_index=sample_index,
                    crop_start_sample=crop_start_sample,
                )
                result.analysis_id = analysis_id
                result.provenance.update({
                    "analysis_id": analysis_id,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "input_sha256": hashlib.sha256(
                        np.ascontiguousarray(signal).tobytes()
                    ).hexdigest(),
                    "pipeline_version": "stage1-1.0",
                    "evidence_configuration": self.evidence_extractor.configuration(),
                })
                from src.analysis.model_decision import freeze_decision
                freeze_decision(result, getattr(self, "detector", None))
                metadata = dict(result.provenance)
                metadata["sampling_rate"] = self.sampling_rate
                metadata["shape"] = list(signal.shape)
                metadata["rhythm_parameters"] = result.rhythm.rr_details["parameters"]
                self.store.complete(analysis_id, result, metadata)
                return result
        except Exception as exc:
            self.store.fail(analysis_id, type(exc).__name__)
            raise

    def _analyze(
        self,
        ecg: np.ndarray,
        *,
        source_id: Optional[str] = None,
        sample_index: Optional[int] = None,
        crop_start_sample: int = 0,
    ) -> ECGAnalysisResult:
        """分析一条 ECG，返回统一结果。

        Args:
            ecg:
                已准备好的 (4800, 12)、500 Hz 输入信号。
                节律分支使用此信号，不使用模型重建信号。
            source_id:
                可选来源标识。避免放入患者身份信息。
            sample_index:
                可选数据集样本索引。
            crop_start_sample:
                本片段在原始记录中的起始采样点。
                仅记录来源，不执行裁剪。
                例如传入 original[100:4900] 时填写 100。

        Notes:
            返回的 r_peaks 相对于本次输入片段。
            原始记录中的峰索引 = r_peaks + crop_start_sample。
            计算错误会向上传播，不伪装成成功或空结果。
        """
        signal = self._validate_ecg(ecg)

        if (
            isinstance(crop_start_sample, bool)
            or not isinstance(crop_start_sample, (int, np.integer))
            or crop_start_sample < 0
        ):
            raise ValueError("crop_start_sample 必须是非负整数。")

        if sample_index is not None:
            if (
                isinstance(sample_index, bool)
                or not isinstance(sample_index, (int, np.integer))
                or sample_index < 0
            ):
                raise ValueError("sample_index 必须是非负整数或 None。")

        if source_id is not None and not isinstance(source_id, str):
            raise TypeError("source_id 必须是字符串或 None。")

        # 两个分支各用独立副本，避免潜在的原地修改。
        sgrf_result = self.detector.predict(signal.copy())

        evidence = self.evidence_extractor.extract(sgrf_result)

        rhythm = self.rhythm_extractor.extract(signal.copy())

        if np.asarray(sgrf_result.error_map).shape != signal.shape:
            raise ValueError(
                "模型 error_map 的形状与输入 ECG 不一致。"
            )

        result = ECGAnalysisResult.from_sgrf_result(
            sgrf_result=sgrf_result,
            evidence=evidence,
            sampling_rate=self.sampling_rate,
        )

        result.rhythm = rhythm

        try:
            checkpoint_label = str(
                self.checkpoint.relative_to(PROJECT_ROOT)
            )
        except ValueError:
            checkpoint_label = self.checkpoint.name

        result.provenance.update({
            "pipeline": "ECGAnalysisPipeline",
            "checkpoint": checkpoint_label,
            "rhythm_extractor": "RhythmFeatureExtractor",
            "rhythm_method": "scipy_pan_tompkins_style",
            "rhythm_validation": "unvalidated_on_real_ecg",
            "crop_start_sample": int(crop_start_sample),
            "crop_end_sample_exclusive": (
                int(crop_start_sample) + signal.shape[0]
            ),
            "peak_index_reference": "cropped_input",
        })

        if source_id is not None:
            result.provenance["source_id"] = source_id

        if sample_index is not None:
            result.provenance["sample_index"] = int(sample_index)

        return result

    @staticmethod
    def _validate_ecg(ecg: np.ndarray) -> np.ndarray:
        array = np.asarray(ecg)

        if array.shape != (4800, 12):
            raise ValueError(
                f"Expected ECG shape (4800, 12), got {array.shape}. "
                "5000 点记录请显式裁剪为 sample[100:4900, :]。"
            )

        if (
            not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
        ):
            raise TypeError("ECG 必须是实数数值数组。")

        if not np.isfinite(array).all():
            raise ValueError("ECG contains NaN or infinite values.")

        with np.errstate(over="ignore", invalid="ignore"):
            signal = np.array(
                array,
                dtype=np.float32,
                copy=True,
            )

        if not np.isfinite(signal).all():
            raise ValueError("ECG 数值超出 float32 可表示范围。")

        return signal
