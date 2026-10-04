from dataclasses import asdict, dataclass
from typing import Any, Dict, List

@dataclass
class LeadEvidence:
    lead: str
    score: float
    rank: int

@dataclass
class TemporalRegion:
    start: int
    end: int
    score: float
    duration: int

@dataclass
class ECGEvidence:
    anomaly_score: float
    reconstruction_error: float
    shape_error: float
    relative_evidence_score: float
    lead_evidence: List[LeadEvidence]
    temporal_regions: List[TemporalRegion]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "reconstruction_error": self.reconstruction_error,
            "shape_error": self.shape_error,
            "relative_evidence_score": self.relative_evidence_score,
            "lead_evidence": [asdict(x) for x in self.lead_evidence],
            "temporal_regions": [asdict(x) for x in self.temporal_regions],
        }
