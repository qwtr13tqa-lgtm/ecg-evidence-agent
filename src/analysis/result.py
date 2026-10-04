from __future__ import annotations
from src.features.rhythm import RhythmFeatures
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ECGInputInfo:
    """Information about the input ECG."""

    num_samples: int
    num_leads: int
    sampling_rate: int = 500

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sampling_rate


@dataclass
class ECGModelOutput:
    """Core output produced by SGRF-Net."""

    anomaly_score: float
    reconstruction_error: float
    shape_error: float

    uncertainty_mean: Optional[float] = None
    uncertainty_max: Optional[float] = None

    reconstruction: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    error_map: Optional[np.ndarray] = None


@dataclass
class ECGEvidenceSummary:
    """Structured evidence extracted from the model output."""

    lead_evidence: List[Any] = field(default_factory=list)
    temporal_regions: List[Any] = field(default_factory=list)

    relative_evidence_score: Optional[float] = None


@dataclass
class ECGAnalysisResult:
    """
    Unified result passed from the signal-analysis layer
    to the future Agent / reasoning layer.
    """

    input: ECGInputInfo
    model: ECGModelOutput
    evidence: ECGEvidenceSummary

    confidence: Optional[float] = None

    provenance: Dict[str, Any] = field(default_factory=dict)
    rhythm: Optional[RhythmFeatures] = None
    analysis_id: Optional[str] = None

    @classmethod
    def from_sgrf_result(
        cls,
        sgrf_result: Any,
        evidence: Any,
        sampling_rate: int = 500,
    ) -> "ECGAnalysisResult":
        """
        Build a unified ECGAnalysisResult from SGRF-Net output
        and extracted evidence.

        Parameters
        ----------
        sgrf_result:
            SGRFResult returned by SGRFDetector.predict().

        evidence:
            ECGEvidence returned by EvidenceExtractor.extract().

        sampling_rate:
            ECG sampling rate in Hz.
        """

        # --------------------------------------------------------
        # 1. Determine ECG input shape
        # --------------------------------------------------------
        error_map = np.asarray(sgrf_result.error_map)

        if error_map.ndim != 2:
            raise ValueError(
                f"Expected error_map with shape (T, C), "
                f"got {error_map.shape}"
            )

        num_samples, num_leads = error_map.shape

        # --------------------------------------------------------
        # 2. Convert SGRFResult → ECGModelOutput
        # --------------------------------------------------------
        sigma = getattr(sgrf_result, "sigma", None)

        uncertainty_mean = None
        uncertainty_max = None

        if sigma is not None:
            sigma_array = np.asarray(sigma)

            if sigma_array.size > 0:
                uncertainty_mean = float(np.mean(sigma_array))
                uncertainty_max = float(np.max(sigma_array))

        model_output = ECGModelOutput(
            anomaly_score=float(sgrf_result.anomaly_score),
            reconstruction_error=float(
                sgrf_result.reconstruction_error
            ),
            shape_error=float(sgrf_result.shape_error),
            uncertainty_mean=uncertainty_mean,
            uncertainty_max=uncertainty_max,
            reconstruction=getattr(
                sgrf_result,
                "reconstruction",
                None,
            ),
            sigma=sigma,
            error_map=getattr(
                sgrf_result,
                "error_map",
                None,
            ),
        )

        # --------------------------------------------------------
        # 3. Convert ECGEvidence → ECGEvidenceSummary
        # --------------------------------------------------------
        evidence_summary = ECGEvidenceSummary(
            lead_evidence=list(
                getattr(evidence, "lead_evidence", [])
            ),
            temporal_regions=list(
                getattr(evidence, "temporal_regions", [])
            ),
            relative_evidence_score=(
                float(evidence.relative_evidence_score)
                if evidence.relative_evidence_score is not None
                else None
            ),
        )

        # --------------------------------------------------------
        # 4. Build unified result
        # --------------------------------------------------------
        return cls(
            input=ECGInputInfo(
                num_samples=num_samples,
                num_leads=num_leads,
                sampling_rate=sampling_rate,
            ),
            model=model_output,
            evidence=evidence_summary,
            confidence=None,
            provenance={
                "model": "SGRF-Net",
                "evidence_extractor": "EvidenceExtractor",
            },
        )

    def to_llm_context(self) -> Dict[str, Any]:
        """
        Convert the analysis result into a compact,
        LLM-friendly structured representation.

        Raw ECG arrays, reconstruction arrays and error maps
        are intentionally excluded.
        """

        lead_evidence = []

        for item in self.evidence.lead_evidence:
            if hasattr(item, "__dataclass_fields__"):
                lead_evidence.append(asdict(item))
            elif isinstance(item, dict):
                lead_evidence.append(item)
            else:
                lead_evidence.append(item)

        temporal_regions = []

        for item in self.evidence.temporal_regions:
            if hasattr(item, "__dataclass_fields__"):
                temporal_regions.append(asdict(item))
            elif isinstance(item, dict):
                temporal_regions.append(item)
            else:
                temporal_regions.append(item)
        rhythm_context = None

        if self.rhythm is not None:
            rhythm_context = {
                "heart_rate_bpm": self.rhythm.heart_rate,
                "mean_rr_seconds": self.rhythm.mean_rr,
                "median_rr_seconds": self.rhythm.median_rr,
                "rr_std_seconds": self.rhythm.rr_std,
                "rr_cv": self.rhythm.rr_cv,
                "candidate_beat_count": self.rhythm.num_beats,
                "sampling_rate_hz": self.rhythm.sampling_rate,
                "lead_index": self.rhythm.lead_index,
                "measurement_status": (
                    "unavailable"
                    if self.rhythm.heart_rate is None
                    else "unvalidated"
                ),
            }
        return {
            "input": {
                "num_samples": self.input.num_samples,
                "num_leads": self.input.num_leads,
                "sampling_rate": self.input.sampling_rate,
                "duration_seconds": self.input.duration_seconds,
            },

            "model": {
                "anomaly_score": self.model.anomaly_score,
                "reconstruction_error": self.model.reconstruction_error,
                "shape_error": self.model.shape_error,
                "uncertainty_mean": self.model.uncertainty_mean,
                "uncertainty_max": self.model.uncertainty_max,
            },

            "evidence": {
                "lead_evidence": lead_evidence,
                "temporal_regions": temporal_regions,
                "relative_evidence_score": (
                    self.evidence.relative_evidence_score
                ),
            },
            "signal_features": {
                "rhythm": rhythm_context,
            },
            "confidence": self.confidence,

            "provenance": self.provenance,
        }

    def summary(self) -> Dict[str, Any]:
        """Return a compact machine-readable summary."""

        return {
            "anomaly_score": self.model.anomaly_score,
            "reconstruction_error": self.model.reconstruction_error,
            "shape_error": self.model.shape_error,
            "confidence": self.confidence,
            "num_lead_evidence": len(
                self.evidence.lead_evidence
            ),
            "num_temporal_regions": len(
                self.evidence.temporal_regions
            ),
        }
