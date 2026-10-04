"""Pure queries over an analysis snapshot; no inference and no network."""
from src.analysis.rr_facts import build_rr_facts
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6")


def get_analysis_summary(result):
    return result.to_llm_context()


def inspect_recent_error(result, duration_seconds, lead=None):
    """End-aligned window; nearest sample, half samples rounded up, no clipping."""
    fs, total = result.input.sampling_rate, result.input.num_samples
    if (type(duration_seconds) not in (int, float)
            or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive and finite")
    if (type(fs) not in (int, float) or not math.isfinite(fs) or fs <= 0
            or type(total) is not int or total <= 0):
        raise ValueError("Invalid input time base")
    requested_samples = Decimal(str(duration_seconds)) * Decimal(str(fs))
    if requested_samples > total:
        raise ValueError("Requested duration exceeds input; no silent clipping")
    count = int(requested_samples.to_integral_value(rounding=ROUND_HALF_UP))
    if count < 1:
        raise ValueError("Requested duration rounds to zero samples")
    data = inspect_error_window(result, total - count, total, lead)
    data.update({
        "requested_duration_seconds": float(duration_seconds),
        "actual_duration_seconds": count / fs,
        "window_num_samples": count,
        "sampling_rate_hz": fs,
        "window_anchor": "input_end",
        "rounding_policy": "nearest_sample_half_up",
        "peak_semantics": "first_argmax_in_window_not_confirmed_event",
    })
    return data


def inspect_error_window(result, start_sample, end_sample, lead=None):
    error_map = np.asarray(result.model.error_map)
    if error_map.shape != (result.input.num_samples, 12):
        raise ValueError("Missing or invalid error map")
    if (type(start_sample) is not int or type(end_sample) is not int
            or not 0 <= start_sample < end_sample <= len(error_map)):
        raise ValueError("Invalid half-open sample window")
    if lead is not None and (not isinstance(lead, str) or lead not in LEADS):
        raise ValueError("Unknown lead")
    selected = [LEADS.index(lead)] if lead is not None else range(12)
    rows = []
    for index in selected:
        values = error_map[start_sample:end_sample, index]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite error map")
        peak = start_sample + int(np.argmax(values))
        rows.append({
            "lead": LEADS[index], "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "minimum": float(np.min(values)),
            "peak_sample": peak,
        })
    offset = result.provenance.get("crop_start_sample", 0)
    fs = result.input.sampling_rate
    return {
        "start_sample": start_sample, "end_sample": end_sample,
        "start_seconds": start_sample / fs, "end_seconds": end_sample / fs,
        "original_start_sample": offset + start_sample,
        "original_end_sample": offset + end_sample,
        "touches_input_end": end_sample == len(error_map),
        "interval_convention": "[start,end)",
        "coordinate_reference": "input_segment",
        "method": "raw_error_map_statistics_v1",
        "score_semantics": "model_error_values_not_probabilities",
        "leads": rows,
    }


def inspect_rr_intervals(result, offset=0, limit=50):
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    rhythm = result.rhythm
    if rhythm is None or rhythm.rr_details is None:
        raise ValueError("RR trace unavailable; rerun the updated pipeline")
    facts = build_rr_facts(result)
    if facts["status"] != "verified":
        raise ValueError("RR_FACT_CHECK_FAILED: " + ",".join(facts["errors"]))
    details = rhythm.rr_details
    raw = details["raw_rr_seconds"]
    if offset > len(raw):
        raise ValueError("offset exceeds interval count")
    stop = min(offset + limit, len(raw))
    mask = details["valid_mask"]
    crop = result.provenance.get("crop_start_sample", 0)
    rows = []
    for index in range(offset, stop):
        left, right = rhythm.r_peaks[index:index + 2]
        rows.append({
            "rr_index": index, "left_peak_sample": left,
            "right_peak_sample": right,
            "original_left_peak_sample": crop + left,
            "original_right_peak_sample": crop + right,
            "rr_seconds": raw[index], "retained": mask[index],
            "exclusion_reason": details["exclusion_reasons"][index],
        })
    return {
        "intervals": rows, "total_intervals": len(raw),
        "retained_count": sum(mask), "excluded_count": len(mask) - sum(mask),
        "next_offset": stop if stop < len(raw) else None,
        "parameters": details["parameters"],
        "measurement_status": "unvalidated",
        "coordinate_reference": "input_segment",
        "source": "stored_rhythm_trace",
        "calculation_facts": {k: v for k, v in facts.items() if k != "rr_intervals"},
    }
