import numpy as np

from src.features.rhythm import RhythmFeatureExtractor


def create_synthetic_ecg(
    duration=10.0,
    sampling_rate=500,
    heart_rate=60,
):
    """
    Create a simple ECG-like synthetic signal.

    This is only for testing the rhythm extractor,
    not for medical simulation.
    """

    n = int(duration * sampling_rate)

    t = np.arange(n) / sampling_rate

    ecg = np.zeros(n, dtype=np.float32)

    rr = 60.0 / heart_rate

    r_times = np.arange(
        0.5,
        duration - 0.2,
        rr,
    )

    for r_time in r_times:

        center = int(
            r_time * sampling_rate
        )

        width = int(
            0.025 * sampling_rate
        )

        start = max(
            0,
            center - width,
        )

        end = min(
            n,
            center + width + 1,
        )

        x = np.arange(
            start,
            end,
        )

        ecg[x] += np.exp(
            -0.5
            * (
                (x - center)
                / (0.01 * sampling_rate)
            )
            ** 2
        )

    # Add very small noise.
    rng = np.random.default_rng(42)

    ecg += (
        0.005
        * rng.standard_normal(n)
    )

    return np.column_stack(
        [ecg] * 12
    )


def test_rhythm_60_bpm():

    ecg = create_synthetic_ecg(
        duration=10,
        sampling_rate=500,
        heart_rate=60,
    )

    extractor = RhythmFeatureExtractor(
        sampling_rate=500,
        lead_index=1,
    )

    features = extractor.extract(ecg)

    print("\nRhythm Features:")
    print(features.to_dict())

    assert features.num_beats >= 8

    assert features.heart_rate is not None

    assert abs(
        features.heart_rate - 60
    ) < 5


def test_rhythm_100_bpm():

    ecg = create_synthetic_ecg(
        duration=10,
        sampling_rate=500,
        heart_rate=100,
    )

    extractor = RhythmFeatureExtractor(
        sampling_rate=500,
        lead_index=1,
    )

    features = extractor.extract(ecg)

    print("\nRhythm Features:")
    print(features.to_dict())

    assert features.heart_rate is not None

    assert abs(
        features.heart_rate - 100
    ) < 8