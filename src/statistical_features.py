import numpy as np
from scipy.stats import skew

from src.features import compute_acc_magnitude


def compute_slope(signal_1d: np.ndarray, sampling_frequency: float) -> float:
    """Least-squares linear fit of the signal against time in seconds; returns the fitted slope."""
    time_seconds = np.arange(len(signal_1d)) / sampling_frequency
    slope, _ = np.polyfit(time_seconds, signal_1d, 1)
    return float(slope)


def extract_statistical_features(signal_1d: np.ndarray, sampling_frequency: float, prefix: str) -> dict:
    """Mean, standard deviation, skewness, and slope of a windowed signal, as `{prefix}_*` features."""
    standard_deviation = float(np.std(signal_1d))
    # skew() divides by std**3; a near-constant window (std ~ 0) has no meaningful
    # asymmetry and would otherwise return NaN (0/0), same guard as normalize_record.
    skewness = float(skew(signal_1d)) if standard_deviation > 1e-8 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(signal_1d)),
        f"{prefix}_std": standard_deviation,
        f"{prefix}_skewness": skewness,
        f"{prefix}_slope": compute_slope(signal_1d, sampling_frequency),
    }


def extract_window_statistical_features(window: dict, sampling_frequency: float, signal_names: list[str]) -> dict:
    """
    Dispatch each channel in a window to `extract_statistical_features` and merge the results.

    Same dispatch shape as `extract_window_features` in `src/features.py`, but
    computing plain time-domain statistics (mean/std/skewness/slope) instead
    of spectral features.
    """
    signal = window["signal"]
    features = {"label": window["label"], "start_time": window["start_time"]}

    if {"ax", "ay", "az"}.issubset(signal_names):
        magnitude = compute_acc_magnitude(signal, signal_names)
        features.update(extract_statistical_features(magnitude, sampling_frequency, "acc"))
    if "EDA" in signal_names:
        features.update(extract_statistical_features(signal[:, signal_names.index("EDA")], sampling_frequency, "eda"))
    if "temp" in signal_names:
        features.update(extract_statistical_features(signal[:, signal_names.index("temp")], sampling_frequency, "temp"))
    if "hr" in signal_names:
        features.update(extract_statistical_features(signal[:, signal_names.index("hr")], sampling_frequency, "hr"))
    if "SpO2" in signal_names:
        features.update(extract_statistical_features(signal[:, signal_names.index("SpO2")], sampling_frequency, "spo2"))

    return features
