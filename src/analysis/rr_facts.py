"""Deterministic checks of stored candidate RR calculations, not clinical validation."""
import math


def build_rr_facts(result):
    base = {"version": "rr-facts-1.0", "analysis_id": result.analysis_id,
            "scope": "stored_rr_arithmetic_only", "measurement_status": "unvalidated",
            "prose_semantics_checked": False}
    rhythm = result.rhythm
    if rhythm is None or rhythm.rr_details is None:
        return dict(base, status="unavailable", errors=["RR_TRACE_UNAVAILABLE"])
    try:
        facts = _check(result, rhythm)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return dict(base, status="invalid", errors=[str(exc) if isinstance(exc, ValueError)
                                                   else "RR_TRACE_SCHEMA_INVALID"])
    return dict(base, status="verified", errors=[], **facts)


def _check(result, rhythm):
    def require(condition, code):
        if not condition:
            raise ValueError(code)

    def finite(x):
        return type(x) in (int, float) and math.isfinite(x)

    def near(a, b):
        return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)

    peaks = rhythm.r_peaks
    d = rhythm.rr_details
    p = d["parameters"]
    fs = result.input.sampling_rate
    require(finite(fs) and fs > 0 and p["sampling_rate"] == fs, "SAMPLING_RATE_MISMATCH")
    require(isinstance(peaks, list) and all(type(x) is int for x in peaks), "INVALID_PEAKS")
    require(all(0 <= x < result.input.num_samples for x in peaks)
            and all(b > a for a, b in zip(peaks, peaks[1:])), "INVALID_PEAK_ORDER_OR_RANGE")
    require(type(rhythm.num_beats) is int and rhythm.num_beats == len(peaks), "PEAK_COUNT_MISMATCH")
    raw = d["raw_rr_seconds"]
    mask = d["valid_mask"]
    reasons = d["exclusion_reasons"]
    n = max(len(peaks) - 1, 0)
    require(all(isinstance(x, list) and len(x) == n for x in (raw, mask, reasons)), "TRACE_LENGTH_MISMATCH")
    require(all(type(x) is bool for x in mask), "INVALID_RETAINED_MASK")
    expected = [(b-a)/fs for a,b in zip(peaks, peaks[1:])]
    require(all(near(a,b) for a,b in zip(raw, expected)), "RR_COORDINATE_MISMATCH")
    low, high = p["min_rr_seconds"], p["max_rr_seconds"]
    require(finite(low) and finite(high) and 0 < low <= high, "INVALID_RR_LIMITS")
    require(finite(p["min_hr"]) and finite(p["max_hr"]) and 0 < p["min_hr"] <= p["max_hr"], "INVALID_HR_LIMITS")
    require(near(low, 60/p["max_hr"]) and near(high, 60/p["min_hr"]), "RR_LIMITS_MISMATCH")
    expected_mask = [low <= x <= high for x in expected]
    require(mask == expected_mask, "FILTER_MASK_MISMATCH")
    expected_reasons = [None if keep else "below_configured_min_rr" if x < low
                        else "above_configured_max_rr" for x,keep in zip(expected, mask)]
    require(reasons == expected_reasons, "EXCLUSION_REASON_MISMATCH")
    retained = [x for x,keep in zip(expected, mask) if keep]
    stored = d["retained_rr_seconds"]
    require(isinstance(stored, list) and len(stored) == len(retained)
            and all(near(a,b) for a,b in zip(stored, retained)), "RETAINED_VALUES_MISMATCH")
    mean = math.fsum(retained)/len(retained) if retained else None
    hr = 60/mean if mean is not None else None
    require((mean is None and rhythm.mean_rr is None and rhythm.heart_rate is None)
            or (mean is not None and near(mean, rhythm.mean_rr) and near(hr, rhythm.heart_rate)),
            "SUMMARY_ARITHMETIC_MISMATCH")
    rows = [{"rr_index": i, "left_peak_sample": peaks[i], "right_peak_sample": peaks[i+1],
             "rr_seconds": x, "retained": mask[i], "exclusion_reason": reasons[i]}
            for i,x in enumerate(expected)]
    return {"candidate_peak_count": len(peaks), "total_intervals": n,
            "retained_count": len(retained), "excluded_count": n-len(retained),
            "mean_rr_seconds": mean, "heart_rate_bpm": hr,
            "count_formula": "raw_rr_count = max(candidate_peak_count - 1, 0)",
            "heart_rate_formula": "60 / mean(retained_rr_seconds)",
            "count_difference_cause": "adjacent_peak_differences",
            "filter_effect": "none_excluded" if len(retained) == n else "intervals_excluded",
            "rr_intervals": rows,
            "numeric_policy": "float64_recompute; stored_summary_rtol_1e-6_atol_1e-7"}


def render_rr_facts(facts):
    """Render only verified calculations. Never rewrite a model's original answer."""
    if facts["status"] != "verified":
        return "RR 计算事实不可展示：" + "、".join(facts["errors"])
    n, total = facts["candidate_peak_count"], facts["total_intervals"]
    lines = [f"候选峰 {n} 个，相邻作差形成 {total} 个原始 RR 间隔。",
             f"保留 {facts['retained_count']} 个，排除 {facts['excluded_count']} 个。",
             "峰数与原始间隔数的差别来自相邻作差，不来自筛选。"]
    if total == 0:
        lines.append("候选峰不足两个，无法计算 RR 均值和心率。")
    elif facts["retained_count"] == 0:
        lines.append("全部间隔被配置范围筛除，无法计算保留 RR 均值和心率。")
    else:
        mean, hr = facts["mean_rr_seconds"], facts["heart_rate_bpm"]
        lines.append(f"保留 RR 的平均值约为 {mean:.6f} 秒；估计心率 = 60 / 平均 RR，约 {hr:.2f} bpm。")
        if facts["excluded_count"] == 0:
            lines.append("本次没有任何 RR 间隔被筛除。")
    lines.append("以上为程序核算的候选峰统计，峰检测准确性尚未验证；不代表临床结论。")
    return "\n\n".join(lines)
