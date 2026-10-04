"""Single-sample inference adapter for the existing SGRF-Net model.

This module intentionally mirrors the inference path in ``test1.py`` instead
of changing the trained model or its anomaly-score definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import copy
import numpy as np
import torch
from scipy.signal import stft

from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class SGRFResult:
    """Result returned by :class:`SGRFDetector`.

    ``anomaly_score`` is the raw SGRF anomaly score, not a calibrated
    probability. Array fields are kept so the next evidence-extraction layer
    can use the reconstruction/error maps without rerunning the model.
    """

    anomaly_score: float
    reconstruction_error: float
    shape_error: float
    reconstruction: np.ndarray
    sigma: np.ndarray
    error_map: np.ndarray
    normalized_ecg: np.ndarray

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "anomaly_score": float(self.anomaly_score),
            "reconstruction_error": float(self.reconstruction_error),
            "shape_error": float(self.shape_error),
        }
        if include_arrays:
            result.update(
                {
                    "reconstruction": self.reconstruction,
                    "sigma": self.sigma,
                    "error_map": self.error_map,
                    "normalized_ecg": self.normalized_ecg,
                }
            )
        return result


class SGRFDetector:
    """Inference wrapper around the existing trained SGRF-Net checkpoint.

    Expected ECG input:
        NumPy array / Torch tensor with shape ``(4800, 12)`` or ``(1, 4800, 12)``.

    The preprocessing and anomaly-score calculation intentionally follow the
    current ``test1.py`` implementation:
      1. per-lead min-max normalization to [-1, 1]
      2. magnitude STFT with fs=500 and nperseg=125
      3. the same time/spectrogram masking strategies
      4. uncertainty-aware reconstruction error + 0.15 * cosine shape error
      5. average over masking strategies
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Union[str, torch.device, None] = None,
        dims: int = 12,
        d_model: int = 96,
        shapex_num_shapelets: int = 16,
        shapex_shapelet_len: int = 96,
        shapex_use_encoder: bool = True,
        shapex_delta: float = 0.5,
        gate_mlp_dropout: float = 0.0,
        attention_heads: int = 2,
        d_ff: int = 128,
        num_transformer_blocks: int = 2,
        dropout: float = 0.2,
        mask_ratio_time: int = 30,
        mask_ratio_spec: int = 20,
        expected_time_length: int = 4800,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        self.dims = dims
        self.mask_ratio_time = mask_ratio_time
        self.mask_ratio_spec = mask_ratio_spec
        self.expected_time_length = expected_time_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
            enc_in=dims,
            channel=dims,
            h=attention_heads,
            d_model=d_model,
            dropout=dropout,
            time_seq_len=4800,
            spec_h_dim=63,
            spec_w_dim=78,
            d_ff=d_ff,
            num_transformer_blocks=num_transformer_blocks,
            shapex_num_shapelets=shapex_num_shapelets,
            shapex_shapelet_len=shapex_shapelet_len,
            shapex_use_encoder=shapex_use_encoder,
            shapex_delta=shapex_delta,
            gate_mlp_dropout=gate_mlp_dropout,
        ).to(self.device)

        checkpoint_obj = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = (
            checkpoint_obj["model_state_dict"]
            if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj
            else checkpoint_obj
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def normalize_instance(data: np.ndarray) -> np.ndarray:
        """Match ``test1.py`` per-lead normalization exactly."""
        data = np.asarray(data, dtype=np.float32)
        data_norm = copy.deepcopy(data)
        for ch in range(data.shape[1]):
            seq = data[:, ch]
            d_min = seq.min()
            d_max = seq.max()
            if d_max - d_min > 1e-6:
                data_norm[:, ch] = 2 * (seq - d_min) / (d_max - d_min) - 1
            else:
                data_norm[:, ch] = 0
        return data_norm

    @staticmethod
    def compute_spectrogram(time_data: np.ndarray) -> np.ndarray:
        """Match ``test1.py`` STFT preprocessing exactly."""
        _, _, zxx = stft(
            time_data.transpose(1, 0),
            fs=500,
            window="hann",
            nperseg=125,
        )
        return np.abs(zxx).transpose(1, 2, 0).astype(np.float32)

    def _validate_input(self, ecg: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.detach().cpu().numpy()
        ecg = np.asarray(ecg, dtype=np.float32)

        if ecg.ndim == 3:
            if ecg.shape[0] != 1:
                raise ValueError(
                    f"Only one ECG is supported per predict() call; got shape {ecg.shape}."
                )
            ecg = ecg[0]

        if ecg.ndim != 2:
            raise ValueError(f"Expected ECG shape (4800, 12), got {ecg.shape}.")

        if ecg.shape != (self.expected_time_length, self.dims):
            raise ValueError(
                f"Expected ECG shape ({self.expected_time_length}, {self.dims}), "
                f"got {ecg.shape}."
            )

        if not np.isfinite(ecg).all():
            raise ValueError("ECG contains NaN or infinite values.")

        return ecg

    def _build_masks(self, time_shape, spec_shape, strategy_index: int):
        _, time_length, _ = time_shape
        bs_s, freq, t_dim, _ = spec_shape

        mask_t = torch.zeros(
            (1, time_length, 1), dtype=torch.bool, device=self.device
        )
        patch_interval_time = 4800 // self.mask_ratio_time
        for k in range(self.mask_ratio_time):
            cut_idx = 48 * strategy_index + patch_interval_time * k
            if cut_idx + 48 <= time_length:
                mask_t[:, cut_idx : cut_idx + 48] = True

        mask_s = torch.zeros(
            (bs_s, freq, t_dim, 1), dtype=torch.bool, device=self.device
        )
        patch_interval_spec = 66 // self.mask_ratio_spec
        for k in range(self.mask_ratio_spec):
            cut_idx = 1 * strategy_index + patch_interval_spec * k
            if cut_idx < t_dim:
                mask_s[:, :, cut_idx : cut_idx + 1] = True

        return mask_t, mask_s

    @torch.no_grad()
    def predict(self, ecg: Union[np.ndarray, torch.Tensor]) -> SGRFResult:
        """Run single-ECG anomaly detection."""
        raw_ecg = self._validate_input(ecg)
        norm_ecg = self.normalize_instance(raw_ecg)
        spec = self.compute_spectrogram(norm_ecg)

        time_tensor = torch.from_numpy(norm_ecg).unsqueeze(0).to(self.device)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(self.device)

        num_strategies = max(1, 100 // self.mask_ratio_time)
        strategy_scores = []
        strategy_recon_errors = []
        strategy_shape_errors = []
        strategy_error_maps = []
        strategy_recons = []
        strategy_sigmas = []

        target = time_tensor[:, :, : self.model.channel]

        for j in range(num_strategies):
            mask_t, mask_s = self._build_masks(
                time_tensor.shape, spec_tensor.shape, strategy_index=j
            )
            mask_time = torch.mul(time_tensor, ~mask_t)
            mask_spec = torch.mul(spec_tensor, ~mask_s)

            recon, sigma = self.model(mask_time, mask_spec, return_aux=False)
            sigma = torch.clamp(sigma, min=-5.0, max=5.0)

            recon_err_sq = torch.square(recon - target)
            sigma_bd = sigma.expand_as(recon_err_sq)
            mse_score = torch.exp(-sigma_bd) * recon_err_sq + sigma_bd

            cosine_sim = torch.nn.functional.cosine_similarity(
                recon, target, dim=1
            ).unsqueeze(1)
            shape_error = 1 - cosine_sim
            shape_error = shape_error.expand(1, time_tensor.shape[1], self.model.channel)

            combined_map = mse_score + (0.15 * shape_error)

            strategy_scores.append(float(torch.mean(combined_map).item()))
            strategy_recon_errors.append(float(torch.mean(mse_score).item()))
            strategy_shape_errors.append(float(torch.mean(shape_error).item()))
            strategy_error_maps.append(combined_map[0].detach().cpu().numpy())
            strategy_recons.append(recon[0].detach().cpu().numpy())
            strategy_sigmas.append(sigma[0].detach().cpu().numpy())

        error_map = np.mean(np.stack(strategy_error_maps, axis=0), axis=0)
        reconstruction = np.mean(np.stack(strategy_recons, axis=0), axis=0)
        sigma_np = np.mean(np.stack(strategy_sigmas, axis=0), axis=0)

        return SGRFResult(
            anomaly_score=float(np.mean(strategy_scores)),
            reconstruction_error=float(np.mean(strategy_recon_errors)),
            shape_error=float(np.mean(strategy_shape_errors)),
            reconstruction=reconstruction.astype(np.float32),
            sigma=sigma_np.astype(np.float32),
            error_map=error_map.astype(np.float32),
            normalized_ecg=norm_ecg.astype(np.float32),
        )
