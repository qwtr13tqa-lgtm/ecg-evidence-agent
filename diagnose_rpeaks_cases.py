import numpy as np
import heartpy as hp


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/Processed_PTBXL/test.npy"

FS = 500.0
LEAD_INDEX = 1
LEAD_NAME = "II"

# 已知典型样本
FIXED_INDICES = [2159, 1679]

# 随机抽取多少个异常样本
RANDOM_CASES = 5

# 为了结果可复现
RANDOM_SEED = 42

# ------------------------------------------------------------
# Peak validation parameters
# ------------------------------------------------------------

# 两个峰之间小于这个距离，认为可能是 duplicate-like
DUPLICATE_RR_SEC = 0.08

# 生理上非常短的 RR
MIN_RR_SEC = 0.25

# 生理上非常长的 RR
MAX_RR_SEC = 2.50

# 合理 HR 范围，仅用于诊断
MIN_HR = 30.0
MAX_HR = 220.0


# ============================================================
# Utility functions
# ============================================================

def format_array(arr, precision=3):
    """Pretty-print numpy array."""
    if arr is None:
        return "None"

    arr = np.asarray(arr)

    if arr.size == 0:
        return "[]"

    if np.issubdtype(arr.dtype, np.floating):
        return np.array2string(
            arr,
            precision=precision,
            separator=", ",
        )

    return np.array2string(
        arr,
        separator=", ",
    )


def calculate_rr(peaks, fs=FS):
    """Calculate RR intervals in seconds."""
    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) < 2:
        return np.array([], dtype=float)

    return np.diff(peaks) / fs


def calculate_hr_from_rr(rr):
    """
    Calculate HR using median RR.

    This is intentionally NOT HeartPy's bpm.
    It is used to compare raw and validated peaks.
    """
    rr = np.asarray(rr, dtype=float)

    if len(rr) == 0:
        return np.nan

    median_rr = np.median(rr)

    if median_rr <= 0:
        return np.nan

    return 60.0 / median_rr


def rr_stats(rr):
    """Return basic RR statistics."""
    rr = np.asarray(rr, dtype=float)

    if len(rr) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    mean_rr = np.mean(rr)
    std_rr = np.std(rr)

    return {
        "mean": mean_rr,
        "median": np.median(rr),
        "std": std_rr,
        "cv": std_rr / mean_rr if mean_rr > 0 else np.nan,
        "min": np.min(rr),
        "max": np.max(rr),
    }


# ============================================================
# HeartPy detection
# ============================================================

def detect_peaks(signal):
    """
    Run HeartPy and return raw candidate peaks.

    This function does NOT modify the signal.
    """
    wd, m = hp.process(signal, sample_rate=FS)

    peaks = np.asarray(
        wd.get("peaklist", []),
        dtype=int,
    )

    return peaks, wd, m


# ============================================================
# Peak validation
# ============================================================

def validate_peaks(peaks, fs=FS):
    """
    Simple refractory-style peak validation.

    Strategy:

    - Keep the first peak.
    - If the next peak is too close:
        keep the one with larger local prominence/amplitude.
    - Otherwise keep the next peak.

    NOTE:
    This is a diagnostic prototype.
    It is NOT claimed to be a clinical-grade R-peak detector.
    """

    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) <= 1:
        return peaks.copy(), []

    min_distance = int(DUPLICATE_RR_SEC * fs)

    validated = [int(peaks[0])]
    removed = []

    for current in peaks[1:]:

        current = int(current)
        previous = validated[-1]

        distance = current - previous

        if distance >= min_distance:
            validated.append(current)
            continue

        # ----------------------------------------------------
        # Duplicate-like peak detected.
        #
        # We cannot decide which peak is better here because
        # this function only receives peak positions.
        #
        # Therefore, for this diagnostic prototype:
        # keep the earlier peak and remove the later one.
        # ----------------------------------------------------

        removed.append(current)

    return np.asarray(validated, dtype=int), removed


def validate_peaks_by_amplitude(signal, peaks, fs=FS):
    """
    Better diagnostic version.

    If two peaks are closer than DUPLICATE_RR_SEC,
    compare local absolute amplitudes and keep the stronger one.

    This is still only a prototype.
    """

    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) <= 1:
        return peaks.copy(), []

    min_distance = int(DUPLICATE_RR_SEC * fs)

    validated = [int(peaks[0])]
    removed = []

    for current in peaks[1:]:

        current = int(current)
        previous = validated[-1]

        distance = current - previous

        if distance >= min_distance:
            validated.append(current)
            continue

        # ----------------------------------------------------
        # Duplicate-like pair.
        # Compare local absolute amplitudes.
        # ----------------------------------------------------

        previous_amp = abs(signal[previous])
        current_amp = abs(signal[current])

        if current_amp > previous_amp:
            removed.append(previous)
            validated[-1] = current
        else:
            removed.append(current)

    return np.asarray(validated, dtype=int), removed


# ============================================================
# Diagnostic classification
# ============================================================

def classify_sample(
    raw_peaks,
    raw_rr,
    validated_peaks,
    validated_rr,
):
    """
    Produce simple diagnostic flags.
    """

    flags = []

    # Raw duplicate-like intervals
    duplicate_count = int(
        np.sum(raw_rr < DUPLICATE_RR_SEC)
    )

    short_count = int(
        np.sum(raw_rr < MIN_RR_SEC)
    )

    long_count = int(
        np.sum(raw_rr > MAX_RR_SEC)
    )

    raw_hr = calculate_hr_from_rr(raw_rr)
    validated_hr = calculate_hr_from_rr(validated_rr)

    if duplicate_count > 0:
        flags.append(
            f"duplicate-like RR: {duplicate_count}"
        )

    if short_count > 0:
        flags.append(
            f"short RR: {short_count}"
        )

    if long_count > 0:
        flags.append(
            f"long RR: {long_count}"
        )

    if np.isfinite(raw_hr):
        if raw_hr < MIN_HR or raw_hr > MAX_HR:
            flags.append(
                f"raw HR suspicious: {raw_hr:.1f}"
            )

    if np.isfinite(validated_hr):
        if validated_hr < MIN_HR or validated_hr > MAX_HR:
            flags.append(
                f"validated HR suspicious: {validated_hr:.1f}"
            )

    if len(validated_peaks) < 2:
        flags.append("too few validated peaks")

    return flags


# ============================================================
# Print one case
# ============================================================

def diagnose_case(index, signal):
    print()
    print("=" * 80)
    print(f"SAMPLE {index}")
    print("=" * 80)

    print(f"Lead                 : {LEAD_NAME}")
    print(f"Signal length        : {len(signal)}")
    print(f"Sampling rate        : {FS} Hz")

    # --------------------------------------------------------
    # HeartPy
    # --------------------------------------------------------

    try:
        raw_peaks, wd, metrics = detect_peaks(signal)
    except Exception as e:
        print()
        print("[HeartPy ERROR]")
        print(str(e))
        return

    raw_rr = calculate_rr(raw_peaks)

    # HeartPy bpm
    heartpy_bpm = metrics.get("bpm", np.nan)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validated_peaks, removed = validate_peaks_by_amplitude(
        signal,
        raw_peaks,
    )

    validated_rr = calculate_rr(validated_peaks)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    raw_stats = rr_stats(raw_rr)
    validated_stats = rr_stats(validated_rr)

    raw_hr = calculate_hr_from_rr(raw_rr)
    validated_hr = calculate_hr_from_rr(validated_rr)

    flags = classify_sample(
        raw_peaks,
        raw_rr,
        validated_peaks,
        validated_rr,
    )

    # --------------------------------------------------------
    # Print detection
    # --------------------------------------------------------

    print()
    print("[1] HeartPy")
    print("-" * 80)

    print(f"HeartPy bpm         : {heartpy_bpm:.2f}")
    print(f"Raw peak count      : {len(raw_peaks)}")
    print(f"Raw peaks           : {format_array(raw_peaks, 0)}")

    # --------------------------------------------------------
    # Raw RR
    # --------------------------------------------------------

    print()
    print("[2] Raw RR intervals")
    print("-" * 80)

    print(f"Mean RR             : {raw_stats['mean']:.4f} sec")
    print(f"Median RR           : {raw_stats['median']:.4f} sec")
    print(f"Std RR              : {raw_stats['std']:.4f} sec")
    print(f"RR CV               : {raw_stats['cv']:.4f}")
    print(f"Min RR              : {raw_stats['min']:.4f} sec")
    print(f"Max RR              : {raw_stats['max']:.4f} sec")

    print(f"HR from median RR   : {raw_hr:.2f} bpm")

    print()
    print(
        f"RR < {DUPLICATE_RR_SEC:.2f}s      : "
        f"{np.sum(raw_rr < DUPLICATE_RR_SEC)}"
    )

    print(
        f"RR < {MIN_RR_SEC:.2f}s      : "
        f"{np.sum(raw_rr < MIN_RR_SEC)}"
    )

    print(
        f"RR > {MAX_RR_SEC:.2f}s      : "
        f"{np.sum(raw_rr > MAX_RR_SEC)}"
    )

    print()
    print(
        "Raw RR              : "
        + format_array(raw_rr, 4)
    )

    # --------------------------------------------------------
    # Duplicate-like pairs
    # --------------------------------------------------------

    print()
    print("[3] Duplicate-like peak pairs")
    print("-" * 80)

    duplicate_pairs = []

    for i in range(1, len(raw_peaks)):

        rr = (raw_peaks[i] - raw_peaks[i - 1]) / FS

        if rr < DUPLICATE_RR_SEC:

            duplicate_pairs.append(
                (
                    int(raw_peaks[i - 1]),
                    int(raw_peaks[i]),
                    rr,
                )
            )

    if duplicate_pairs:

        for p1, p2, rr in duplicate_pairs:

            amp1 = abs(signal[p1])
            amp2 = abs(signal[p2])

            print(
                f"{p1:5d} -> {p2:5d} | "
                f"RR={rr:.4f}s | "
                f"|amp1|={amp1:.5f} | "
                f"|amp2|={amp2:.5f}"
            )

    else:
        print("None")

    # --------------------------------------------------------
    # Validated peaks
    # --------------------------------------------------------

    print()
    print("[4] Validated peaks")
    print("-" * 80)

    print(f"Validated count     : {len(validated_peaks)}")
    print(
        f"Removed count       : "
        f"{len(removed)}"
    )

    print(
        "Validated peaks     : "
        + format_array(validated_peaks, 0)
    )

    print(
        "Removed peaks       : "
        + format_array(np.asarray(removed), 0)
    )

    # --------------------------------------------------------
    # Validated RR
    # --------------------------------------------------------

    print()
    print("[5] Validated RR intervals")
    print("-" * 80)

    print(f"Mean RR             : {validated_stats['mean']:.4f} sec")
    print(f"Median RR           : {validated_stats['median']:.4f} sec")
    print(f"Std RR              : {validated_stats['std']:.4f} sec")
    print(f"RR CV               : {validated_stats['cv']:.4f}")
    print(f"Min RR              : {validated_stats['min']:.4f} sec")
    print(f"Max RR              : {validated_stats['max']:.4f} sec")

    print(
        f"HR from median RR   : "
        f"{validated_hr:.2f} bpm"
    )

    print()
    print(
        "Validated RR        : "
        + format_array(validated_rr, 4)
    )

    # --------------------------------------------------------
    # Before / after comparison
    # --------------------------------------------------------

    print()
    print("[6] BEFORE vs AFTER")
    print("-" * 80)

    print(
        f"Peak count          : "
        f"{len(raw_peaks)} -> {len(validated_peaks)}"
    )

    print(
        f"Median RR           : "
        f"{raw_stats['median']:.4f} -> "
        f"{validated_stats['median']:.4f} sec"
    )

    print(
        f"RR CV               : "
        f"{raw_stats['cv']:.4f} -> "
        f"{validated_stats['cv']:.4f}"
    )

    print(
        f"HR                  : "
        f"{raw_hr:.2f} -> "
        f"{validated_hr:.2f} bpm"
    )

    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    print()
    print("[7] Diagnostic flags")
    print("-" * 80)

    if flags:
        for flag in flags:
            print(f"[!] {flag}")
    else:
        print("[OK] No obvious rule-based problem detected.")

    print()
    print("NOTE:")
    print(
        "This validation is an engineering diagnostic prototype. "
        "It is NOT a clinical-grade R-peak detector."
    )


# ============================================================
# Find random abnormal samples
# ============================================================

def find_random_abnormal_samples(data):
    """
    Quickly scan all samples and identify samples with
    duplicate-like or short RR intervals.

    Only used to select diagnostic cases.
    """

    abnormal = []

    print()
    print("=" * 80)
    print("SCANNING DATASET FOR RANDOM ABNORMAL CASES")
    print("=" * 80)

    print(
        f"Searching for samples with RR < "
        f"{MIN_RR_SEC}s ..."
    )

    for i in range(len(data)):

        signal = data[i, :, LEAD_INDEX]

        try:
            peaks, _, _ = detect_peaks(signal)

            rr = calculate_rr(peaks)

            if len(rr) == 0:
                continue

            has_short = np.any(
                rr < MIN_RR_SEC
            )

            has_duplicate = np.any(
                rr < DUPLICATE_RR_SEC
            )

            if has_short or has_duplicate:

                abnormal.append(i)

        except Exception:
            continue

    print(
        f"Abnormal candidates found : "
        f"{len(abnormal)}"
    )

    if len(abnormal) == 0:
        return []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n = min(
        RANDOM_CASES,
        len(abnormal)
    )

    selected = rng.choice(
        abnormal,
        size=n,
        replace=False,
    )

    return sorted(
        int(x)
        for x in selected
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("R-PEAK CASE-BY-CASE DIAGNOSTIC")
    print("=" * 80)

    print(f"Data path       : {DATA_PATH}")
    print(f"Sampling rate   : {FS} Hz")
    print(f"Lead            : {LEAD_NAME} (index {LEAD_INDEX})")
    print()

    # --------------------------------------------------------
    # Read-only mmap
    # --------------------------------------------------------

    data = np.load(
        DATA_PATH,
        mmap_mode="r",
    )

    print("Dataset")
    print("-" * 80)

    print(f"Shape           : {data.shape}")
    print(f"Dtype           : {data.dtype}")

    # --------------------------------------------------------
    # Validate fixed indices
    # --------------------------------------------------------

    fixed = []

    for idx in FIXED_INDICES:

        if 0 <= idx < len(data):
            fixed.append(idx)

    # --------------------------------------------------------
    # Find random abnormal cases
    # --------------------------------------------------------

    random_cases = find_random_abnormal_samples(
        data
    )

    # Avoid duplicates
    selected_cases = []

    for idx in fixed + random_cases:

        if idx not in selected_cases:
            selected_cases.append(idx)

    print()
    print("=" * 80)
    print("SELECTED CASES")
    print("=" * 80)

    print(
        "Fixed cases          : "
        + str(fixed)
    )

    print(
        "Random abnormal cases: "
        + str(random_cases)
    )

    print(
        "Final diagnostic set : "
        + str(selected_cases)
    )

    # --------------------------------------------------------
    # Diagnose each case
    # --------------------------------------------------------

    for idx in selected_cases:

        signal = np.asarray(
            data[idx, :, LEAD_INDEX],
            dtype=np.float64,
        )

        diagnose_case(
            idx,
            signal,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DIAGNOSTIC FINISHED")
    print("=" * 80)

    print()
    print("READ-ONLY MODE:")
    print("test.npy was only read and was not modified.")
    print("No output data files were created.")


if __name__ == "__main__":
    main()