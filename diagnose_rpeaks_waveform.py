import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/Processed_PTBXL/test.npy"

FS = 500.0
LEAD_INDEX = 1
LEAD_NAME = "II"

# 我们重点观察的样本
CASE_INDICES = [
    2159,
    1679,
    1549,
    1779,
]

# RR thresholds
DUPLICATE_RR_SEC = 0.08
SHORT_RR_SEC = 0.25
LONG_RR_SEC = 2.50

# ============================================================
# Detection
# ============================================================

def detect_peaks(signal):
    """Run HeartPy and return candidate peaks."""
    wd, metrics = hp.process(
        signal,
        sample_rate=FS,
    )

    peaks = np.asarray(
        wd.get("peaklist", []),
        dtype=int,
    )

    return peaks, metrics


# ============================================================
# RR
# ============================================================

def calculate_rr(peaks):
    """RR intervals in seconds."""
    if len(peaks) < 2:
        return np.array([], dtype=float)

    return np.diff(peaks) / FS


# ============================================================
# Print diagnostic information
# ============================================================

def print_diagnostics(index, signal, peaks, rr):
    print()
    print("=" * 90)
    print(f"SAMPLE {index}")
    print("=" * 90)

    print(f"Lead          : {LEAD_NAME}")
    print(f"Length        : {len(signal)} samples")
    print(f"Duration      : {len(signal) / FS:.2f} sec")
    print(f"Peak count    : {len(peaks)}")

    if len(peaks) == 0:
        print("No peaks detected.")
        return

    print()
    print("Peak positions:")
    print(peaks.tolist())

    print()
    print("RR intervals:")
    print(
        np.array2string(
            rr,
            precision=4,
            separator=", ",
        )
    )

    print()
    print("-" * 90)
    print("SHORT / DUPLICATE-LIKE PEAKS")
    print("-" * 90)

    found = False

    for i, interval in enumerate(rr):

        if interval < SHORT_RR_SEC:

            found = True

            p1 = peaks[i]
            p2 = peaks[i + 1]

            amp1 = abs(signal[p1])
            amp2 = abs(signal[p2])

            if interval < DUPLICATE_RR_SEC:
                category = "DUPLICATE"
            else:
                category = "SHORT"

            print(
                f"{category:9s} | "
                f"{p1:5d} -> {p2:5d} | "
                f"RR={interval:.4f}s | "
                f"t={p1 / FS:.3f}s -> {p2 / FS:.3f}s | "
                f"|amp|={amp1:.5f}, {amp2:.5f}"
            )

    if not found:
        print("None")

    print()
    print("-" * 90)
    print("LONG RR GAPS")
    print("-" * 90)

    found_long = False

    for i, interval in enumerate(rr):

        if interval > LONG_RR_SEC:

            found_long = True

            p1 = peaks[i]
            p2 = peaks[i + 1]

            print(
                f"{p1:5d} -> {p2:5d} | "
                f"RR={interval:.4f}s | "
                f"t={p1 / FS:.3f}s -> {p2 / FS:.3f}s"
            )

    if not found_long:
        print("None")


# ============================================================
# Plot
# ============================================================

def plot_case(index, signal, peaks, rr):
    """
    Plot the complete 10-second Lead II waveform.

    Marker semantics:
        green circle  = all HeartPy candidate peaks
        red x          = RR < 80 ms
        orange triangle = 80 ms <= RR < 250 ms
        purple lines   = long RR gap
    """

    time = np.arange(len(signal)) / FS

    fig, ax = plt.subplots(
        figsize=(18, 6)
    )

    # --------------------------------------------------------
    # ECG waveform
    # --------------------------------------------------------

    ax.plot(
        time,
        signal,
        linewidth=1.0,
        label="Lead II ECG",
    )

    # --------------------------------------------------------
    # All HeartPy peaks
    # --------------------------------------------------------

    valid_peaks = peaks[
        (peaks >= 0)
        & (peaks < len(signal))
    ]

    ax.scatter(
        valid_peaks / FS,
        signal[valid_peaks],
        s=35,
        marker="o",
        label="HeartPy candidate peaks",
        zorder=5,
    )

    # --------------------------------------------------------
    # Classify peaks according to the FOLLOWING RR
    # --------------------------------------------------------

    duplicate_peaks = []
    short_peaks = []

    for i, interval in enumerate(rr):

        p1 = peaks[i]
        p2 = peaks[i + 1]

        if interval < DUPLICATE_RR_SEC:

            duplicate_peaks.extend(
                [p1, p2]
            )

        elif interval < SHORT_RR_SEC:

            short_peaks.extend(
                [p1, p2]
            )

    # Remove duplicates while preserving order
    duplicate_peaks = np.array(
        list(dict.fromkeys(duplicate_peaks)),
        dtype=int,
    )

    short_peaks = np.array(
        list(dict.fromkeys(short_peaks)),
        dtype=int,
    )

    # --------------------------------------------------------
    # Duplicate-like peaks
    # --------------------------------------------------------

    if len(duplicate_peaks) > 0:

        ax.scatter(
            duplicate_peaks / FS,
            signal[duplicate_peaks],
            s=100,
            marker="x",
            linewidths=2.5,
            label="RR < 0.08s",
            zorder=7,
        )

    # --------------------------------------------------------
    # Short RR peaks
    # --------------------------------------------------------

    if len(short_peaks) > 0:

        ax.scatter(
            short_peaks / FS,
            signal[short_peaks],
            s=80,
            marker="^",
            label="0.08s <= RR < 0.25s",
            zorder=6,
        )

    # --------------------------------------------------------
    # Long RR gaps
    # --------------------------------------------------------

    for i, interval in enumerate(rr):

        if interval > LONG_RR_SEC:

            p1 = peaks[i]
            p2 = peaks[i + 1]

            middle = (
                (p1 + p2) / 2
            ) / FS

            ax.axvline(
                middle,
                linestyle="--",
                linewidth=1.5,
                label="RR > 2.5s"
                if i == 0
                else None,
            )

            ax.annotate(
                f"long RR\n{interval:.2f}s",
                xy=(middle, np.max(signal)),
                xytext=(middle, np.max(signal)),
                ha="center",
                va="top",
            )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(
        f"PTB-XL Test Sample {index} — Lead {LEAD_NAME}"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="upper right"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT save files.
    # Just display the figure.
    # --------------------------------------------------------

    plt.show()

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("ECG R-PEAK WAVEFORM DIAGNOSTIC")
    print("=" * 90)

    print(f"Data path : {DATA_PATH}")
    print(f"Sampling  : {FS} Hz")
    print(f"Lead      : {LEAD_NAME}")
    print(f"Cases     : {CASE_INDICES}")

    # --------------------------------------------------------
    # Read-only memory map
    # --------------------------------------------------------

    data = np.load(
        DATA_PATH,
        mmap_mode="r",
    )

    print()
    print(
        f"Dataset shape: {data.shape}"
    )

    # --------------------------------------------------------
    # Process selected cases
    # --------------------------------------------------------

    for index in CASE_INDICES:

        if index < 0 or index >= len(data):

            print(
                f"\n[WARNING] "
                f"Sample {index} is out of range."
            )

            continue

        # Read Lead II only
        signal = np.asarray(
            data[
                index,
                :,
                LEAD_INDEX
            ],
            dtype=np.float64,
        )

        try:

            peaks, metrics = detect_peaks(
                signal
            )

        except Exception as e:

            print()
            print(
                f"[ERROR] Sample {index}"
            )
            print(str(e))
            continue

        rr = calculate_rr(peaks)

        print_diagnostics(
            index,
            signal,
            peaks,
            rr,
        )

        plot_case(
            index,
            signal,
            peaks,
            rr,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("WAVEFORM DIAGNOSTIC FINISHED")
    print("=" * 90)

    print()
    print("READ-ONLY MODE:")
    print("test.npy was only read.")
    print("No ECG data was modified.")
    print("No output data files were created.")


if __name__ == "__main__":
    main()