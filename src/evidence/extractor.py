import numpy as np
from typing import List

from src.inference.sgrf_adapter import LEAD_NAMES, SGRFResult
from .schema import ECGEvidence, LeadEvidence, TemporalRegion


class EvidenceExtractor:
    """Convert SGRF error_map into structured, within-sample evidence.

    Each lead is scored using its raw upper tail and peak, then the twelve
    scores are normalized together within this sample (not per time series).
    Temporal evidence is then obtained by aggregating the raw error map
    across leads. These scores are for evidence localization/ranking only;
    they are not calibrated probabilities or clinical diagnoses.
    """

    def __init__(
        self,
        top_k_leads: int = 3,
        top_k_regions: int = 3,
        region_threshold: float = 0.80,
        min_region_length: int = 48,
        merge_gap: int = 24,
        lead_high_quantile: float = 0.95,
    ):
        self.top_k_leads = top_k_leads
        self.top_k_regions = top_k_regions
        self.region_threshold = region_threshold
        self.min_region_length = min_region_length
        self.merge_gap = merge_gap
        self.lead_high_quantile = lead_high_quantile

    def configuration(self):
        return {
            "version": "evidence-1.0",
            "top_k_leads": self.top_k_leads,
            "top_k_regions": self.top_k_regions,
            "region_threshold": self.region_threshold,
            "min_region_length": self.min_region_length,
            "merge_gap": self.merge_gap,
            "lead_high_quantile": self.lead_high_quantile,
            "tail_weight": 0.7, "peak_weight": 0.3,
            "normalization_percentiles": [5, 95],
            "coordinate_reference": "input_segment",
            "interval_convention": "[start,end)",
        }

    @staticmethod
    def _robust_normalize(values: np.ndarray) -> np.ndarray:
        """Robustly map an array to [0, 1]."""
        x = np.asarray(values, dtype=np.float32)
        lo, hi = np.percentile(x, [5, 95])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _lead_scores(self, error_map: np.ndarray) -> np.ndarray:
        """Score each lead by its upper-tail anomaly magnitude.

        We combine the mean of the top 5% time points with the maximum.
        This emphasizes localized anomalies without letting a single spike
        completely determine the ranking.
        """
        scores = np.zeros(error_map.shape[1], dtype=np.float32)
        for lead_idx in range(error_map.shape[1]):
            x = error_map[:, lead_idx]
            q = float(np.quantile(x, self.lead_high_quantile))
            tail = x[x >= q]
            tail_mean = float(np.mean(tail)) if len(tail) else float(np.mean(x))
            peak = float(np.max(x))
            scores[lead_idx] = 0.7 * tail_mean + 0.3 * peak
        return self._robust_normalize(scores)

    def _find_regions(self, temporal_score: np.ndarray) -> List[TemporalRegion]:
        active = temporal_score >= self.region_threshold
        raw = []
        i = 0

        while i < len(active):
            if not active[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(active) and active[i + 1]:
                i += 1
            raw.append([start, i + 1])
            i += 1

        merged = []
        for start, end in raw:
            if merged and start - merged[-1][1] <= self.merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        regions = []
        for start, end in merged:
            if end - start >= self.min_region_length:
                regions.append(
                    TemporalRegion(
                        start=int(start),
                        end=int(end),
                        score=float(np.mean(temporal_score[start:end])),
                        duration=int(end - start),
                    )
                )

        regions.sort(key=lambda x: x.score, reverse=True)
        return regions[: self.top_k_regions]

    def extract(self, result: SGRFResult) -> ECGEvidence:
        error_map = np.asarray(result.error_map, dtype=np.float32)

        if error_map.ndim != 2 or error_map.shape[1] != len(LEAD_NAMES):
            raise ValueError(
                f"Expected error_map shape (T, {len(LEAD_NAMES)}), got {error_map.shape}"
            )
        if not np.all(np.isfinite(error_map)):
            raise ValueError("error_map contains NaN or Inf")

        lead_scores = self._lead_scores(error_map)
        order = np.argsort(lead_scores)[::-1][: self.top_k_leads]

        lead_evidence = [
            LeadEvidence(
                lead=LEAD_NAMES[int(idx)],
                score=float(lead_scores[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(order)
        ]

        # Temporal evidence uses the raw error magnitude aggregated across leads.
        temporal_raw = np.mean(error_map, axis=1)
        temporal_score = self._robust_normalize(temporal_raw)
        temporal_regions = self._find_regions(temporal_score)

        k = min(self.top_k_leads, len(lead_scores))
        relative_score = float(np.mean(np.sort(lead_scores)[-k:]))

        return ECGEvidence(
            anomaly_score=float(result.anomaly_score),
            reconstruction_error=float(result.reconstruction_error),
            shape_error=float(result.shape_error),
            relative_evidence_score=relative_score,
            lead_evidence=lead_evidence,
            temporal_regions=temporal_regions,
        )
