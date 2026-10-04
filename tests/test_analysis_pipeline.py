"""真实 ECG 双分支联调。

同一段 ECG：
    1. SGRF-Net → Model Evidence
    2. RhythmFeatureExtractor → Rhythm Features

汇总到 ECGAnalysisResult → LLM Context。

PASS 只表示流程与数据接口通过检查，不代表临床准确性。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analysis.result import ECGAnalysisResult
from src.evidence.extractor import EvidenceExtractor
from src.features.rhythm import RhythmFeatureExtractor
from src.inference.sgrf_adapter import SGRFDetector


CHECKPOINT = PROJECT_ROOT / (
    "ckpt_shape_guided_shapex_gate/"
    "best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt"
)

DATA_PATH = PROJECT_ROOT / "data/Processed_PTBXL/test.npy"

SAMPLE_INDEX = 0
SAMPLING_RATE = 500
LEAD_INDEX = 1

CROP_START = 100
CROP_END = 4900


def main() -> None:
    # ----------------------------------------------------------
    # 1. 读取同一段真实 ECG，供两个分支使用
    # ----------------------------------------------------------
    print("[1] Loading ECG...")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    data = np.load(DATA_PATH, mmap_mode="r")

    print(f"Dataset shape: {data.shape}")

    if data.ndim != 3:
        raise ValueError(
            f"Expected dataset shape (N, T, C), got {data.shape}"
        )

    if not 0 <= SAMPLE_INDEX < data.shape[0]:
        raise ValueError(f"Invalid sample index: {SAMPLE_INDEX}")

    if data.shape[1] < CROP_END or data.shape[2] != 12:
        raise ValueError(
            f"Expected at least {CROP_END} samples and 12 leads, "
            f"got {data.shape}"
        )

    ecg = np.array(
        data[SAMPLE_INDEX, CROP_START:CROP_END, :],
        dtype=np.float32,
        copy=True,
    )

    assert ecg.shape == (4800, 12)
    assert np.isfinite(ecg).all()

    print(f"ECG shape: {ecg.shape}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Duration: {len(ecg) / SAMPLING_RATE:.2f} seconds")

    # ----------------------------------------------------------
    # 2. SGRF-Net 推理
    # ----------------------------------------------------------
    print("\n[2] Running SGRF-Net...")

    detector = SGRFDetector(
        checkpoint=CHECKPOINT,
        device=None,
    )

    print(f"Device: {detector.device}")

    # 使用副本，避免模型内部处理意外影响另一个分支的输入。
    sgrf_result = detector.predict(ecg.copy())

    print(
        f"anomaly_score        = "
        f"{sgrf_result.anomaly_score:.8f}"
    )
    print(
        f"reconstruction_error = "
        f"{sgrf_result.reconstruction_error:.8f}"
    )
    print(
        f"shape_error          = "
        f"{sgrf_result.shape_error:.8f}"
    )
    print(f"error_map shape      = {sgrf_result.error_map.shape}")

    # ----------------------------------------------------------
    # 3. 模型证据提取
    # ----------------------------------------------------------
    print("\n[3] Extracting model evidence...")

    evidence_extractor = EvidenceExtractor()
    evidence = evidence_extractor.extract(sgrf_result)

    print(f"Lead evidence count: {len(evidence.lead_evidence)}")
    print(f"Temporal region count: {len(evidence.temporal_regions)}")
    print(
        "Relative evidence score: "
        f"{evidence.relative_evidence_score}"
    )

    # ----------------------------------------------------------
    # 4. 独立节律特征提取
    # ----------------------------------------------------------
    print("\n[4] Extracting rhythm features...")

    rhythm_extractor = RhythmFeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        lead_index=LEAD_INDEX,
    )

    rhythm = rhythm_extractor.extract(ecg.copy())

    print("\nReal ECG rhythm features:")
    print(json.dumps(
        rhythm.to_dict(),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))

    # ----------------------------------------------------------
    # 5. 汇总结果，记录来源
    # ----------------------------------------------------------
    print("\n[5] Building ECGAnalysisResult...")

    result = ECGAnalysisResult.from_sgrf_result(
        sgrf_result=sgrf_result,
        evidence=evidence,
        sampling_rate=SAMPLING_RATE,
    )

    result.rhythm = rhythm

    result.provenance.update({
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "sample_index": SAMPLE_INDEX,
        "crop_start_sample": CROP_START,
        "crop_end_sample_exclusive": CROP_END,
        "peak_index_reference": "cropped_input",
        "rhythm_extractor": "RhythmFeatureExtractor",
        "rhythm_method": "scipy_pan_tompkins_style",
        "rhythm_validation": "unvalidated_on_real_ecg",
    })

    print("\nSummary:")
    print(result.summary())

    # ----------------------------------------------------------
    # 6. 生成 LLM Context
    # ----------------------------------------------------------
    print("\n[6] Building LLM context...")

    llm_context = result.to_llm_context()

    # 同时验证整个 Context 能转换为标准 JSON。
    context_json = json.dumps(
        llm_context,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    print("\nLLM Context:")
    print(context_json)

    # ----------------------------------------------------------
    # 7. 验证模型输出、节律接口和输入对齐
    # ----------------------------------------------------------
    print("\n[7] Validating pipeline outputs...")

    assert np.isfinite(sgrf_result.anomaly_score)
    assert np.isfinite(sgrf_result.reconstruction_error)
    assert np.isfinite(sgrf_result.shape_error)
    assert sgrf_result.error_map.shape == ecg.shape

    assert len(evidence.lead_evidence) > 0

    assert result.input.num_samples == ecg.shape[0]
    assert result.input.num_leads == ecg.shape[1]
    assert result.input.sampling_rate == SAMPLING_RATE

    assert result.rhythm is not None
    assert result.rhythm.sampling_rate == result.input.sampling_rate
    assert result.rhythm.lead_index == LEAD_INDEX

    peaks = np.asarray(result.rhythm.r_peaks)

    assert result.rhythm.num_beats == len(peaks)

    if len(peaks) > 0:
        assert np.issubdtype(peaks.dtype, np.integer)
        assert np.all(peaks >= 0)
        assert np.all(peaks < ecg.shape[0])
        assert np.all(np.diff(peaks) > 0)

    rhythm_context = llm_context["signal_features"]["rhythm"]

    assert rhythm_context is not None
    assert (
        rhythm_context["candidate_beat_count"]
        == result.rhythm.num_beats
    )
    assert (
        rhythm_context["heart_rate_bpm"]
        == result.rhythm.heart_rate
    )

    expected_status = (
        "unavailable"
        if result.rhythm.heart_rate is None
        else "unvalidated"
    )

    assert rhythm_context["measurement_status"] == expected_status

    # 尚未校准，不生成置信度。
    assert llm_context["confidence"] is None

    # 递归检查，避免大数组或峰位置藏在嵌套结构中。
    excluded_fields = {
        "error_map",
        "reconstruction",
        "normalized_ecg",
        "r_peaks",
        "rhythm_regularity",
    }

    def check_context(value):
        assert not isinstance(value, np.ndarray), (
            "LLM context must not contain numpy arrays"
        )

        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in excluded_fields, (
                    f"Unexpected field in LLM context: {key}"
                )
                check_context(child)

        elif isinstance(value, (list, tuple)):
            for child in value:
                check_context(child)

    check_context(llm_context)

    print("All assertions passed.")

    print("\n========================================")
    print("PIPELINE SMOKE TEST: PASS")
    print("========================================")
    print(
        "接口联调通过；真实 ECG 的节律测量准确性尚未验证。"
    )


if __name__ == "__main__":
    main()