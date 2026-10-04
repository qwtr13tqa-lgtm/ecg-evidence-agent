import os
import numpy as np
import heartpy as hp


# =========================
# Configuration
# =========================

DATA_PATH = "data/Processed_PTBXL/test.npy"

FS = 500.0
LEAD_INDEX = 1          # Lead II
LEAD_NAME = "II"

NUM_SAMPLES = 10


# =========================
# Helper functions
# =========================

def calc_rr_and_hr(peaks, fs):
    """Calculate RR intervals and heart rate from R peaks."""
    peaks = np.asarray(peaks, dtype=np.int64)

    if len(peaks) < 2:
        return np.array([]), np.nan

    rr = np.diff(peaks) / fs

    # Mean HR
    mean_rr = np.mean(rr)

    if mean_rr <= 0:
        hr = np.nan
    else:
        hr = 60.0 / mean_rr

    return rr, hr


def diagnose_one(ecg, index):
    """Run HeartPy R-peak detection on one ECG."""

    # Original ECG shape should be (5000, 12)
    if ecg.ndim != 2:
        print(f"[ERROR] sample {index}: ndim={ecg.ndim}, expected 2")
        return

    if ecg.shape[1] != 12:
        print(
            f"[ERROR] sample {index}: shape={ecg.shape}, "
            f"expected second dimension = 12"
        )
        return

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

        rr, hr = calc_rr_and_hr(peaks, FS)

        print("=" * 70)
        print(f"Sample index : {index}")
        print(f"ECG shape    : {ecg.shape}")
        print(f"Lead         : {LEAD_NAME}")
        print(f"R peaks      : {len(peaks)}")

        if len(peaks) > 0:
            print(f"R positions  : {peaks.tolist()}")

        if len(rr) > 0:
            print(
                f"RR mean      : {np.mean(rr):.4f} s"
            )
            print(
                f"RR median    : {np.median(rr):.4f} s"
            )
            print(
                f"RR std       : {np.std(rr):.4f} s"
            )
            print(
                f"HR           : {hr:.2f} bpm"
            )

        # HeartPy may provide bpm directly
        if "bpm" in measures:
            print(
                f"HeartPy bpm  : {measures['bpm']:.2f}"
            )

        # Show the signal range of Lead II
        print(
            f"Lead II min  : {signal.min():.6f}"
        )
        print(
            f"Lead II max  : {signal.max():.6f}"
        )
        print(
            f"Lead II mean : {signal.mean():.6f}"
        )
        print(
            f"Lead II std  : {signal.std():.6f}"
        )

    except Exception as e:
        print("=" * 70)
        print(f"[FAILED] Sample index: {index}")
        print(f"Error: {type(e).__name__}: {e}")


# =========================
# Main
# =========================

def main():

    print("=" * 70)
    print("PTB-XL R-peak diagnostic")
    print("=" * 70)

    # Check file existence
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] File not found: {DATA_PATH}")
        return

    print(f"Data path    : {DATA_PATH}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Lead         : {LEAD_NAME} (index {LEAD_INDEX})")
    print()

    # IMPORTANT:
    # mmap_mode='r' means read-only memory mapping.
    # No data modification is performed.
    data = np.load(DATA_PATH, mmap_mode="r")

    print("Dataset information")
    print("-" * 70)
    print(f"Shape        : {data.shape}")
    print(f"Dtype        : {data.dtype}")
    print(f"Global min   : {data.min():.6f}")
    print(f"Global max   : {data.max():.6f}")
    print(f"Global mean  : {data.mean():.6f}")
    print(f"Global std   : {data.std():.6f}")
    print()

    # Basic shape check
    if data.ndim != 3 or data.shape[2] != 12:
        print(
            "[ERROR] Unexpected dataset shape. "
            "Expected (N, 5000, 12)."
        )
        return

    num_available = data.shape[0]
    num_test = min(NUM_SAMPLES, num_available)

    # Deterministic selection:
    # first 10 samples, no random modification.
    indices = np.linspace(
        0,
        num_available - 1,
        num=num_test,
        dtype=int
    )

    print(
        f"Testing {len(indices)} samples: "
        f"{indices.tolist()}"
    )

    print()

    success = 0
    failed = 0

    for index in indices:

        try:
            diagnose_one(data[index], int(index))
            success += 1

        except Exception as e:
            failed += 1
            print(
                f"[ERROR] Unexpected failure on "
                f"sample {index}: {e}"
            )

    print()
    print("=" * 70)
    print("Diagnostic finished")
    print("=" * 70)
    print(f"Successful : {success}")
    print(f"Failed     : {failed}")
    print()
    print("No .npy/.csv/.txt output file was created or modified.")


if __name__ == "__main__":
    main()
