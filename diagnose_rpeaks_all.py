import os
import numpy as np
import heartpy as hp


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/Processed_PTBXL/test.npy"

FS = 500.0
LEAD_INDEX = 1
LEAD_NAME = "II"

# Physiological / algorithmic sanity thresholds
MIN_RR_SEC = 0.25       # 240 bpm
MAX_RR_SEC = 2.50       # 24 bpm

MIN_HR = 30.0
MAX_HR = 220.0

# Duplicate / suspicious peak threshold
# At 500 Hz:
# 10 samples = 20 ms
DUPLICATE_RR_SEC = 0.08

# Large RR gap threshold.
# 2.0 sec means a potential missed beat may exist.
LARGE_GAP_SEC = 2.0


# ============================================================
# Analyze one ECG
# ============================================================

def analyze_one(ecg):

    result = {
        "success": False,
        "num_peaks": 0,
        "hr_heartpy": np.nan,
        "hr_from_rr": np.nan,

        "rr_mean": np.nan,
        "rr_median": np.nan,
        "rr_std": np.nan,
        "rr_cv": np.nan,

        "min_rr": np.nan,
        "max_rr": np.nan,

        "short_rr_count": 0,
        "duplicate_rr_count": 0,
        "long_rr_count": 0,

        "suspicious_hr": False,
        "error": None,
    }

    if ecg.ndim != 2 or ecg.shape[1] != 12:
        result["error"] = f"invalid shape: {ecg.shape}"
        return result

    # Lead II
    signal = ecg[:, LEAD_INDEX]

    try:

        working_data, measures = hp.process(
            signal,
            sample_rate=FS
        )

        peaks = np.asarray(
            working_data["peaklist"],
            dtype=np.int64
        )

        result["success"] = True
        result["num_peaks"] = len(peaks)

        # HeartPy's own HR
        if "bpm" in measures:
            result["hr_heartpy"] = float(measures["bpm"])

        # Need at least two peaks for RR
        if len(peaks) < 2:
            return result

        rr = np.diff(peaks) / FS

        result["rr_mean"] = float(np.mean(rr))
        result["rr_median"] = float(np.median(rr))
        result["rr_std"] = float(np.std(rr))

        if result["rr_mean"] > 0:
            result["hr_from_rr"] = 60.0 / result["rr_mean"]

        result["min_rr"] = float(np.min(rr))
        result["max_rr"] = float(np.max(rr))

        # RR coefficient of variation
        if result["rr_mean"] > 0:
            result["rr_cv"] = (
                result["rr_std"] /
                result["rr_mean"]
            )

        # Very short RR intervals
        short_mask = rr < MIN_RR_SEC

        result["short_rr_count"] = int(
            np.sum(short_mask)
        )

        # Extremely short intervals = likely duplicate peak
        duplicate_mask = rr < DUPLICATE_RR_SEC

        result["duplicate_rr_count"] = int(
            np.sum(duplicate_mask)
        )

        # Very long RR intervals
        long_mask = rr > MAX_RR_SEC

        result["long_rr_count"] = int(
            np.sum(long_mask)
        )

        # Suspicious HR
        hr = result["hr_heartpy"]

        if np.isfinite(hr):
            if hr < MIN_HR or hr > MAX_HR:
                result["suspicious_hr"] = True

        return result

    except Exception as e:

        result["error"] = (
            f"{type(e).__name__}: {e}"
        )

        return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("PTB-XL FULL R-PEAK DIAGNOSTIC")
    print("=" * 80)

    print(f"Data path       : {DATA_PATH}")
    print(f"Sampling rate   : {FS} Hz")
    print(f"Lead            : {LEAD_NAME} (index {LEAD_INDEX})")
    print()

    if not os.path.exists(DATA_PATH):
        print("[ERROR] Data file not found.")
        return

    # --------------------------------------------------------
    # READ ONLY
    # --------------------------------------------------------

    data = np.load(
        DATA_PATH,
        mmap_mode="r"
    )

    print("Dataset")
    print("-" * 80)
    print(f"Shape           : {data.shape}")
    print(f"Dtype           : {data.dtype}")
    print()

    n_samples = data.shape[0]

    print(f"Total samples   : {n_samples}")
    print()

    # --------------------------------------------------------
    # Global counters
    # --------------------------------------------------------

    success_count = 0
    failed_count = 0

    short_rr_samples = 0
    duplicate_peak_samples = 0
    long_rr_samples = 0
    suspicious_hr_samples = 0

    all_hr = []
    all_rr_cv = []
    all_peak_counts = []
    all_min_rr = []
    all_max_rr = []

    failed_indices = []
    duplicate_indices = []
    short_rr_indices = []
    long_rr_indices = []
    suspicious_hr_indices = []

    # --------------------------------------------------------
    # Process all 2160 samples
    # --------------------------------------------------------

    for i in range(n_samples):

        result = analyze_one(data[i])

        if result["success"]:
            success_count += 1

        else:
            failed_count += 1
            failed_indices.append(i)

        # HR
        if np.isfinite(result["hr_heartpy"]):
            all_hr.append(
                result["hr_heartpy"]
            )

        # RR CV
        if np.isfinite(result["rr_cv"]):
            all_rr_cv.append(
                result["rr_cv"]
            )

        # Number of peaks
        all_peak_counts.append(
            result["num_peaks"]
        )

        # Min / max RR
        if np.isfinite(result["min_rr"]):
            all_min_rr.append(
                result["min_rr"]
            )

        if np.isfinite(result["max_rr"]):
            all_max_rr.append(
                result["max_rr"]
            )

        # Short RR
        if result["short_rr_count"] > 0:
            short_rr_samples += 1
            short_rr_indices.append(i)

        # Duplicate peaks
        if result["duplicate_rr_count"] > 0:
            duplicate_peak_samples += 1
            duplicate_indices.append(i)

        # Long RR
        if result["long_rr_count"] > 0:
            long_rr_samples += 1
            long_rr_indices.append(i)

        # Suspicious HR
        if result["suspicious_hr"]:
            suspicious_hr_samples += 1
            suspicious_hr_indices.append(i)

        # Progress
        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1:4d} / {n_samples}"
            )

    # ========================================================
    # Convert statistics
    # ========================================================

    all_hr = np.asarray(all_hr)
    all_rr_cv = np.asarray(all_rr_cv)
    all_peak_counts = np.asarray(all_peak_counts)
    all_min_rr = np.asarray(all_min_rr)
    all_max_rr = np.asarray(all_max_rr)

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)

    print()
    print("[1] Detection")
    print("-" * 80)

    print(f"Total samples              : {n_samples}")
    print(f"Successful HeartPy         : {success_count}")
    print(f"Failed HeartPy             : {failed_count}")

    if n_samples > 0:
        print(
            f"Success rate               : "
            f"{success_count / n_samples * 100:.2f}%"
        )

    # --------------------------------------------------------

    print()
    print("[2] R-peak count")
    print("-" * 80)

    if len(all_peak_counts) > 0:

        print(
            f"Mean R peaks / 10 sec      : "
            f"{np.mean(all_peak_counts):.2f}"
        )

        print(
            f"Median R peaks / 10 sec    : "
            f"{np.median(all_peak_counts):.2f}"
        )

        print(
            f"Min R peaks                : "
            f"{np.min(all_peak_counts)}"
        )

        print(
            f"Max R peaks                : "
            f"{np.max(all_peak_counts)}"
        )

    # --------------------------------------------------------

    print()
    print("[3] Heart rate")
    print("-" * 80)

    if len(all_hr) > 0:

        print(
            f"Mean HR                    : "
            f"{np.mean(all_hr):.2f} bpm"
        )

        print(
            f"Median HR                  : "
            f"{np.median(all_hr):.2f} bpm"
        )

        print(
            f"Std HR                     : "
            f"{np.std(all_hr):.2f} bpm"
        )

        print(
            f"Min HR                     : "
            f"{np.min(all_hr):.2f} bpm"
        )

        print(
            f"Max HR                     : "
            f"{np.max(all_hr):.2f} bpm"
        )

    # --------------------------------------------------------

    print()
    print("[4] RR interval")
    print("-" * 80)

    if len(all_min_rr) > 0:

        print(
            f"Median minimum RR          : "
            f"{np.median(all_min_rr):.4f} sec"
        )

        print(
            f"Minimum RR observed       : "
            f"{np.min(all_min_rr):.4f} sec"
        )

    if len(all_max_rr) > 0:

        print(
            f"Median maximum RR          : "
            f"{np.median(all_max_rr):.4f} sec"
        )

        print(
            f"Maximum RR observed       : "
            f"{np.max(all_max_rr):.4f} sec"
        )

    # --------------------------------------------------------

    print()
    print("[5] RR variability")
    print("-" * 80)

    if len(all_rr_cv) > 0:

        print(
            f"Mean RR CV                 : "
            f"{np.mean(all_rr_cv):.4f}"
        )

        print(
            f"Median RR CV               : "
            f"{np.median(all_rr_cv):.4f}"
        )

        print(
            f"Max RR CV                  : "
            f"{np.max(all_rr_cv):.4f}"
        )

    # --------------------------------------------------------

    print()
    print("[6] Potential problems")
    print("-" * 80)

    print(
        f"Samples with short RR "
        f"(< {MIN_RR_SEC:.2f}s)       : "
        f"{short_rr_samples}"
    )

    print(
        f"Samples with duplicate-like "
        f"RR (< {DUPLICATE_RR_SEC:.2f}s) : "
        f"{duplicate_peak_samples}"
    )

    print(
        f"Samples with long RR "
        f"(> {MAX_RR_SEC:.2f}s)       : "
        f"{long_rr_samples}"
    )

    print(
        f"Samples with suspicious HR "
        f"({MIN_HR:.0f}-{MAX_HR:.0f} bpm) : "
        f"{suspicious_hr_samples}"
    )

    # ========================================================
    # Show suspicious sample indices
    # ========================================================

    print()
    print("=" * 80)
    print("SUSPICIOUS SAMPLE INDICES")
    print("=" * 80)

    print()
    print("Failed:")
    print(failed_indices[:100])

    print()
    print("Duplicate-like peaks:")
    print(duplicate_indices[:100])

    print()
    print("Short RR:")
    print(short_rr_indices[:100])

    print()
    print("Long RR:")
    print(long_rr_indices[:100])

    print()
    print("Suspicious HR:")
    print(suspicious_hr_indices[:100])

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 80)
    print("Diagnostic finished.")
    print("=" * 80)

    print()
    print(
        "READ-ONLY MODE: "
        "test.npy was only read and was not modified."
    )

    print(
        "No output data files were created."
    )


if __name__ == "__main__":
    main()