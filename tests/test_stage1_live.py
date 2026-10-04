"""Local real-ECG smoke test. Requires existing weights/data/Torch; no API calls."""
import json
from pathlib import Path
import numpy as np
from src.analysis.pipeline import ECGAnalysisPipeline
from src.tools.executor import ECGToolExecutor


def main():
    root = Path(__file__).resolve().parents[1]
    data = np.load(root / "data/Processed_PTBXL/test.npy", mmap_mode="r")
    pipeline = ECGAnalysisPipeline()
    a = pipeline.analyze(data[0, 100:4900], sample_index=0,
                         source_id="PTBXL/test.npy", crop_start_sample=100)
    b = pipeline.analyze(data[1, 100:4900], sample_index=1,
                         source_id="PTBXL/test.npy", crop_start_sample=100)
    assert a.analysis_id != b.analysis_id
    executor = ECGToolExecutor(pipeline.store, a.analysis_id)
    window = executor.execute("inspect_error_window", {
        "start_sample": 4500, "end_sample": 4800, "lead": "V1"})
    assert window["ok"], window
    np.testing.assert_allclose(
        window["data"]["leads"][0]["mean"], a.model.error_map[4500:4800, 6].mean())
    rr = executor.execute("inspect_rr_intervals")
    assert rr["ok"], rr
    assert not executor.execute("get_analysis_summary", {"analysis_id": b.analysis_id})["ok"]
    print(json.dumps({"analysis_a": a.analysis_id, "analysis_b": b.analysis_id,
                      "window": window, "rr": rr}, indent=2, ensure_ascii=False, allow_nan=False))
    print("STAGE 1 REAL PIPELINE: PASS")


if __name__ == "__main__":
    main()
