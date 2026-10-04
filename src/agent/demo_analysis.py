"""Explicit synthetic fixture, never presented as a measured ECG."""
import numpy as np
from src.analysis.result import ECGAnalysisResult, ECGInputInfo, ECGModelOutput, ECGEvidenceSummary
from src.analysis.store import AnalysisStore


def synthetic_analysis():
    store = AnalysisStore()
    result = ECGAnalysisResult(
        input=ECGInputInfo(4800, 12, 500),
        model=ECGModelOutput(-0.9, -0.95, 0.05,
            error_map=np.zeros((4800, 12), dtype=np.float32)),
        evidence=ECGEvidenceSummary(),
        provenance={"crop_start_sample": 100, "data_kind": "synthetic_software_test"})
    result.analysis_id = store.begin()
    result.provenance["analysis_id"] = result.analysis_id
    store.complete(result.analysis_id, result)
    return store, result
