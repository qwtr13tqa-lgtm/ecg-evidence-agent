"""
ECG rhythm feature extraction.

This module extracts beat-level rhythm features directly from the raw ECG
signal. It does not depend on SGRF-Net.

Current version:
    - Lead II based R-peak detection
    - RR interval extraction
    - Heart rate estimation
    - RR variability
    - Basic rhythm regularity

This is a research/prototype implementation, not a clinical ECG measurement
algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal


@dataclass
class RhythmFeatures:
    """Structured rhythm features extracted directly from ECG."""

    heart_rate: Optional[float]
    mean_rr: Optional[float]
    median_rr: Optional[float]
    rr_std: Optional[float]
    rr_cv: Optional[float]

    num_beats: int
    r_peaks: List[int]

    rhythm_regularity: Optional[float]

    sampling_rate: int
    lead_index: int = 1
    rr_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Preserve the existing public serialization contract for legacy callers.
        result.pop("rr_details")
        return result


class RhythmFeatureExtractor:
    """
    Extract rhythm features from a multi-lead ECG.

    Expected input:
        ECG shape = (T, C)

    Default:
        T = 4800
        C = 12
        sampling_rate = 500 Hz

    Lead II is used as the default rhythm lead.
    """

    def __init__(
        self,
        sampling_rate: int = 500,
        lead_index: int = 1,
        min_hr: float = 30.0,
        max_hr: float = 220.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.lead_index = lead_index

        self.min_hr = min_hr
        self.max_hr = max_hr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, ecg: np.ndarray) -> RhythmFeatures:
        """
        Extract rhythm features from a multi-lead ECG.

        Args:
            ecg:
                ECG array with shape (T, C).

        Returns:
            RhythmFeatures
        """

        ecg = self._validate_input(ecg)

        lead = ecg[:, self.lead_index]

        processed = self._preprocess(lead)

        r_peaks = self._detect_r_peaks(processed, lead)

        rr_intervals = self._compute_rr_intervals(r_peaks)

        features = self._build_features(
            r_peaks=r_peaks,
            rr_intervals=rr_intervals,
        )
        raw_rr = np.diff(r_peaks) / self.sampling_rate
        minimum, maximum = 60.0 / self.max_hr, 60.0 / self.min_hr
        mask = (raw_rr >= minimum) & (raw_rr <= maximum)
        features.rr_details = {
            "raw_rr_seconds": raw_rr.tolist(),
            "valid_mask": mask.tolist(),
            "exclusion_reasons": [
                None if keep else (
                    "below_configured_min_rr" if rr < minimum
                    else "above_configured_max_rr"
                ) for rr, keep in zip(raw_rr, mask)
            ],
            "retained_rr_seconds": rr_intervals.tolist(),
            "parameters": {
                "sampling_rate": self.sampling_rate,
                "lead_index": self.lead_index,
                "min_hr": self.min_hr, "max_hr": self.max_hr,
                "min_rr_seconds": minimum, "max_rr_seconds": maximum,
                "std_ddof": 0, "method_version": "rhythm-trace-1.0",
            },
        }
        return features

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_input(self, ecg: np.ndarray) -> np.ndarray:
        ecg = np.asarray(ecg, dtype=np.float32)

        if ecg.ndim != 2:
            raise ValueError(
                f"Expected ECG shape (T, C), got {ecg.shape}"
            )

        if ecg.shape[1] < 2:
            raise ValueError(
                "Expected at least 2 ECG leads because Lead II "
                "is used by default."
            )

        if not 0 <= self.lead_index < ecg.shape[1]:
            raise ValueError(
                f"lead_index={self.lead_index} is out of range "
                f"for ECG with {ecg.shape[1]} leads."
            )

        if not np.isfinite(ecg).all():
            raise ValueError("ECG contains NaN or infinite values.")

        return ecg

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, lead: np.ndarray) -> np.ndarray:
        """
        Pan-Tompkins-style preprocessing.

        1. Band-pass filtering
        2. Differentiation
        3. Squaring
        4. Moving-window integration
        """

        fs = self.sampling_rate

        # Band-pass approximately 5-20 Hz.
        low = 5.0
        high = 20.0

        sos = signal.butter(
            3,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        filtered = signal.sosfiltfilt(sos, lead)

        # Differentiation.
        derivative = np.diff(
            filtered,
            prepend=filtered[0],
        )

        # Squaring.
        squared = derivative ** 2

        # Moving-window integration.
        integration_window = max(
            1,
            int(0.15 * fs),
        )

        kernel = np.ones(integration_window) / integration_window

        integrated = np.convolve(
            squared,
            kernel,
            mode="same",
        )

        return integrated

    # ------------------------------------------------------------------
    # R peak detection
    # ------------------------------------------------------------------

    def _detect_r_peaks(
        self,
        processed: np.ndarray,
        original_lead: np.ndarray,
    ) -> np.ndarray:
        """
        Detect candidate QRS regions and map them back to R peaks
        in the original ECG signal.
        """

        fs = self.sampling_rate

        # Minimum physiological distance between detected beats.
        min_distance = int(
            fs * 60.0 / self.max_hr
        )

        # Robust threshold.
        median = np.median(processed)
        mad = np.median(
            np.abs(processed - median)
        )

        threshold = median + 3.0 * mad

        if threshold <= 0:
            threshold = np.mean(processed)

        candidates, _ = signal.find_peaks(
            processed,
            height=threshold,
            distance=min_distance,
        )

        if len(candidates) == 0:
            return np.asarray([], dtype=int)

        # Refine each candidate to the strongest absolute ECG peak
        # in a local neighborhood.
        search_radius = int(0.08 * fs)

        refined_peaks = []

        for candidate in candidates:
            start = max(
                0,
                candidate - search_radius,
            )

            end = min(
                len(original_lead),
                candidate + search_radius + 1,
            )

            local = original_lead[start:end]

            if len(local) == 0:
                continue

            # ECG polarity can vary between records.
            local_index = int(
                np.argmax(np.abs(local))
            )

            refined_peak = start + local_index

            refined_peaks.append(refined_peak)

        if not refined_peaks:
            return np.asarray([], dtype=int)

        refined_peaks = np.asarray(
            refined_peaks,
            dtype=int,
        )

        # Remove duplicated / overly close peaks.
        refined_peaks = self._remove_close_peaks(
            refined_peaks,
            original_lead,
        )

        return refined_peaks

    # ------------------------------------------------------------------
    # Peak cleanup
    # ------------------------------------------------------------------

    def _remove_close_peaks(
        self,
        peaks: np.ndarray,
        signal_data: np.ndarray,
    ) -> np.ndarray:

        if len(peaks) <= 1:
            return peaks

        min_distance = int(
            self.sampling_rate * 60.0 / self.max_hr
        )

        selected = [int(peaks[0])]

        for peak in peaks[1:]:

            peak = int(peak)

            if peak - selected[-1] >= min_distance:
                selected.append(peak)

            else:
                # Keep the stronger peak.
                previous = selected[-1]

                if abs(signal_data[peak]) > abs(
                    signal_data[previous]
                ):
                    selected[-1] = peak

        return np.asarray(
            selected,
            dtype=int,
        )

    # ------------------------------------------------------------------
    # RR intervals
    # ------------------------------------------------------------------

    def _compute_rr_intervals(
        self,
        r_peaks: np.ndarray,
    ) -> np.ndarray:

        if len(r_peaks) < 2:
            return np.asarray([], dtype=np.float32)

        rr = np.diff(r_peaks) / self.sampling_rate

        # Physiological sanity filtering.
        min_rr = 60.0 / self.max_hr
        max_rr = 60.0 / self.min_hr

        valid = (
            (rr >= min_rr)
            & (rr <= max_rr)
        )

        return rr[valid].astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(
        self,
        r_peaks: np.ndarray,
        rr_intervals: np.ndarray,
    ) -> RhythmFeatures:

        if len(rr_intervals) == 0:

            return RhythmFeatures(
                heart_rate=None,
                mean_rr=None,
                median_rr=None,
                rr_std=None,
                rr_cv=None,
                num_beats=len(r_peaks),
                r_peaks=r_peaks.tolist(),
                rhythm_regularity=None,
                sampling_rate=self.sampling_rate,
                lead_index=self.lead_index,
            )

        mean_rr = float(
            np.mean(rr_intervals)
        )

        median_rr = float(
            np.median(rr_intervals)
        )

        rr_std = float(
            np.std(rr_intervals)
        )

        rr_cv = float(
            rr_std / mean_rr
        ) if mean_rr > 0 else None

        heart_rate = (
            60.0 / mean_rr
            if mean_rr > 0
            else None
        )

        # A simple regularity score:
        #
        # CV = 0   → perfectly regular
        # larger CV → less regular
        #
        # This is a relative engineering feature,
        # NOT a clinical rhythm diagnosis.
        if rr_cv is not None:
            rhythm_regularity = float(
                np.exp(-rr_cv)
            )
        else:
            rhythm_regularity = None

        return RhythmFeatures(
            heart_rate=heart_rate,
            mean_rr=mean_rr,
            median_rr=median_rr,
            rr_std=rr_std,
            rr_cv=rr_cv,
            num_beats=len(r_peaks),
            r_peaks=r_peaks.tolist(),
            rhythm_regularity=rhythm_regularity,
            sampling_rate=self.sampling_rate,
            lead_index=self.lead_index,
        )
