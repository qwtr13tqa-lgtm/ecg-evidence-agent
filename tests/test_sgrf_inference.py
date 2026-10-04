"""Smoke test for the single-sample SGRF inference adapter.

Run from the project root:
    python tests/test_sgrf_inference.py

Optional environment variables:
    SGRF_CHECKPOINT=/path/to/checkpoint.pt
    SGRF_DATA_PATH=/path/to/Processed_PTBXL
    SGRF_DEVICE=cpu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.sgrf_adapter import SGRFDetector


DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt"
)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/Processed_PTBXL"


def load_one_ecg(data_path: Path) -> np.ndarray:
    """Load the first test ECG using the project's existing TestSet."""
    from dataloader import TestSet

    dataset = TestSet(folder=str(data_path))
    sample = dataset.test_data[0]

    # test1.py uses dataset.test_data[idx][100:4900, :]
    ecg = np.asarray(sample[100:4900, :], dtype=np.float32)
    return ecg


def run_smoke_test() -> None:
    checkpoint = Path(os.getenv("SGRF_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    data_path = Path(os.getenv("SGRF_DATA_PATH", str(DEFAULT_DATA_PATH)))
    device = os.getenv("SGRF_DEVICE")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not data_path.exists():
        raise FileNotFoundError(f"PTB-XL processed data not found: {data_path}")

    print(f"[1/4] Loading ECG from: {data_path}")
    ecg = load_one_ecg(data_path)
    print(f"      ECG shape: {ecg.shape}")

    print(f"[2/4] Loading SGRF-Net checkpoint: {checkpoint}")
    detector = SGRFDetector(checkpoint=checkpoint, device=device)
    print(f"      Device: {detector.device}")

    print("[3/4] Running single-sample inference...")
    result = detector.predict(ecg)

    print("[4/4] Checking outputs...")
    assert np.isfinite(result.anomaly_score)
    assert np.isfinite(result.reconstruction_error)
    assert np.isfinite(result.shape_error)
    assert result.reconstruction.shape == (4800, 12)
    assert result.sigma.shape == (4800, 1)
    assert result.error_map.shape == (4800, 12)
    assert result.normalized_ecg.shape == (4800, 12)

    print("\nPASS")
    print(f"  anomaly_score       = {result.anomaly_score:.8f}")
    print(f"  reconstruction_error= {result.reconstruction_error:.8f}")
    print(f"  shape_error         = {result.shape_error:.8f}")
    print(f"  reconstruction      = {result.reconstruction.shape}")
    print(f"  sigma               = {result.sigma.shape}")
    print(f"  error_map           = {result.error_map.shape}")


if __name__ == "__main__":
    run_smoke_test()
