from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence.extractor import EvidenceExtractor
from src.inference.sgrf_adapter import SGRFResult


def main():
    # Construct a deliberately unambiguous synthetic case:
    # lead II has the strongest repeated local anomaly.
    error_map = np.zeros((4800, 12), dtype=np.float32)
    error_map += 0.01

    error_map[1000:1150, 1] = 2.0       # II: strongest
    error_map[1000:1150, 2] = 1.2       # III
    error_map[2500:2600, 3] = 1.0       # aVR

    result = SGRFResult(
        anomaly_score=-0.9,
        reconstruction_error=-0.95,
        shape_error=0.05,
        reconstruction=np.zeros((4800, 12), dtype=np.float32),
        sigma=np.zeros((4800, 1), dtype=np.float32),
        error_map=error_map,
        normalized_ecg=np.zeros((4800, 12), dtype=np.float32),
    )

    evidence = EvidenceExtractor(
        top_k_leads=3,
        top_k_regions=3,
        region_threshold=0.80,
        min_region_length=48,
    ).extract(result)

    assert evidence.lead_evidence, "No lead evidence produced"
    assert evidence.lead_evidence[0].lead == "II", (
        f"Expected II to rank first, got "
        f"{[(x.lead, round(x.score, 4)) for x in evidence.lead_evidence]}"
    )
    assert evidence.temporal_regions, "No temporal region produced"
    assert any(
        r.start <= 1000 and r.end >= 1150
        for r in evidence.temporal_regions
    ), f"Expected region around [1000,1150], got {evidence.temporal_regions}"
    assert 0.0 <= evidence.relative_evidence_score <= 1.0

    print("PASS")
    print("Top leads:", [(x.lead, round(x.score, 4)) for x in evidence.lead_evidence])
    print(
        "Regions:",
        [(x.start, x.end, round(x.score, 4)) for x in evidence.temporal_regions],
    )
    print("Relative evidence:", round(evidence.relative_evidence_score, 4))


if __name__ == "__main__":
    main()
