import numpy as np
from scipy.signal import welch, spectrogram


HR_BANDS = {"lf": (0.04, 0.15), "hf": (0.15, 0.4)}
EDA_BANDS = {"tonic": (0.01, 0.045), "sympathetic_low": (0.045, 0.15), "sympathetic_high": (0.15, 0.25)}
ACC_BANDS = {"postural": (0.0, 0.5), "movement": (0.5, 4.0)}

_ENTROPY_EPSILON = 1e-12


def compute_power_spectrum(
    signal_1d: np.ndarray, sampling_frequency: float, nperseg: int, noverlap: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate: Hann-tapered, mean-removed per sub-segment. Returns (freqs, psd)."""
    nperseg = min(nperseg, len(signal_1d))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    return welch(signal_1d, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap,
                 detrend="constant", window="hann")


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """
    PSD power within [low, high) Hz: a Riemann sum (sum of PSD bins times bin width).

    Deliberately not trapezoidal integration -- a band containing exactly one
    frequency bin (common for narrow physiological bands at coarse
    resolution) has zero width under the trapezoidal rule and would silently
    integrate to 0 regardless of that bin's power.
    """
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask) or len(freqs) < 2:
        return 0.0
    bin_width = freqs[1] - freqs[0]
    return float(np.sum(psd[mask]) * bin_width)


def total_power(freqs: np.ndarray, psd: np.ndarray) -> float:
    """PSD power as a Riemann sum, excluding the DC (0 Hz) bin."""
    if len(freqs) <= 1:
        return 0.0
    bin_width = freqs[1] - freqs[0]
    return float(np.sum(psd[1:]) * bin_width)


def dominant_frequency(freqs: np.ndarray, psd: np.ndarray, low: float = 0.0, high: float | None = None) -> float:
    """Frequency at the PSD maximum within [low, high] Hz."""
    high = freqs[-1] if high is None else high
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    band_freqs, band_psd = freqs[mask], psd[mask]
    return float(band_freqs[np.argmax(band_psd)])


def spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the PSD, normalized to a probability distribution (DC bin excluded)."""
    psd_ac = psd[1:] if len(psd) > 1 else psd
    total = psd_ac.sum()
    if total <= 0:
        return 0.0
    probabilities = psd_ac / total
    return float(-np.sum(probabilities * np.log(probabilities + _ENTROPY_EPSILON)))


def compute_spectrogram(
    signal_1d: np.ndarray, sampling_frequency: float, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-varying PSD estimate (STFT-based spectrogram), for visualization only. Returns (freqs, times, Sxx)."""
    nperseg = min(nperseg, len(signal_1d))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    return spectrogram(signal_1d, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap,
                        detrend="constant", window="hann")


def compute_acc_magnitude(window_signal: np.ndarray, signal_names: list[str]) -> np.ndarray:
    """Detrend each accelerometer axis (removes the ~1g gravity offset) and return sqrt(ax^2+ay^2+az^2)."""
    axes = [window_signal[:, signal_names.index(name)] for name in ("ax", "ay", "az")]
    axes = [axis - axis.mean() for axis in axes]
    return np.sqrt(sum(axis ** 2 for axis in axes))


def _generic_spectral_summary(freqs: np.ndarray, psd: np.ndarray, prefix: str) -> dict:
    nonzero_low = freqs[1] if len(freqs) > 1 else 0.0
    return {
        f"{prefix}_total_power": total_power(freqs, psd),
        f"{prefix}_dominant_frequency": dominant_frequency(freqs, psd, low=nonzero_low),
        f"{prefix}_spectral_entropy": spectral_entropy(psd),
    }


def extract_generic_spectral_features(signal_1d: np.ndarray, sampling_frequency: float, nperseg: int, prefix: str) -> dict:
    """Universal total_power/dominant_frequency/spectral_entropy features, averaged-Welch. Used for temp."""
    freqs, psd = compute_power_spectrum(signal_1d, sampling_frequency, nperseg=nperseg, noverlap=nperseg // 2)
    return _generic_spectral_summary(freqs, psd, prefix)


def extract_generic_low_rate_features(signal_1d: np.ndarray, sampling_frequency: float, prefix: str) -> dict:
    """Universal spectral summary for 1 Hz signals (SpO2): single-segment Welch, no sub-averaging (see extract_hr_features)."""
    freqs, psd = compute_power_spectrum(signal_1d, sampling_frequency, nperseg=len(signal_1d), noverlap=0)
    return _generic_spectral_summary(freqs, psd, prefix)


def extract_hr_features(hr_window_1d: np.ndarray, sampling_frequency: float) -> dict:
    """
    HR spectral features: universal summary + LF/HF band power + LF/HF ratio.

    Computed on the 1 Hz HR-in-bpm series (not an RR-interval tachogram) — an
    approximation of canonical HRV analysis, appropriate given this dataset.
    Uses the whole window as one Welch segment (no sub-averaging): at only 60
    samples, sub-segmenting would worsen the already-coarse frequency
    resolution needed to resolve the LF band.
    """
    freqs, psd = compute_power_spectrum(hr_window_1d, sampling_frequency, nperseg=len(hr_window_1d), noverlap=0)
    features = _generic_spectral_summary(freqs, psd, "hr")
    lf = band_power(freqs, psd, *HR_BANDS["lf"])
    hf = band_power(freqs, psd, *HR_BANDS["hf"])
    features["hr_lf_power"] = lf
    features["hr_hf_power"] = hf
    features["hr_lf_hf_ratio"] = lf / hf if hf > 0 else 0.0
    return features


def extract_eda_features(eda_window_1d: np.ndarray, sampling_frequency: float) -> dict:
    """
    EDA spectral features: universal summary + tonic/sympathetic band powers + sympathetic ratio.

    Bands per Posada-Quintero et al. (2016) sympathetic-index-from-EDA
    spectral analysis, each only 0.03-0.1 Hz wide. Uses the whole window as
    one Welch segment (no sub-averaging), like extract_hr_features: averaged
    16s sub-segments (the variance-reduction choice originally intended here)
    give only Delta-f=0.0625 Hz resolution, which resolves to zero bins for
    the tonic band and exactly one bin for sympathetic_high -- too coarse to
    represent these bands at all. The full 60s window (Delta-f~0.0167 Hz)
    gives each band several bins, at the cost of a noisier per-bin estimate.
    """
    freqs, psd = compute_power_spectrum(eda_window_1d, sampling_frequency, nperseg=len(eda_window_1d), noverlap=0)
    features = _generic_spectral_summary(freqs, psd, "eda")
    tonic = band_power(freqs, psd, *EDA_BANDS["tonic"])
    sympathetic_low = band_power(freqs, psd, *EDA_BANDS["sympathetic_low"])
    sympathetic_high = band_power(freqs, psd, *EDA_BANDS["sympathetic_high"])
    features["eda_tonic_power"] = tonic
    features["eda_sympathetic_low_power"] = sympathetic_low
    features["eda_sympathetic_high_power"] = sympathetic_high
    features["eda_sympathetic_ratio"] = sympathetic_high / sympathetic_low if sympathetic_low > 0 else 0.0
    return features


def extract_acc_features(acc_window_3axis: np.ndarray, signal_names: list[str], sampling_frequency: float) -> dict:
    """
    Accelerometer spectral features from the detrended acceleration magnitude.

    Bands: postural sway (0-0.5 Hz) vs. movement/gait (0.5-4 Hz, well within
    the 4 Hz Nyquist at 8 Hz sampling). Uses the same averaged-Welch approach
    as EDA to denoise the gait-frequency peak during PhysicalStress.
    """
    magnitude = compute_acc_magnitude(acc_window_3axis, signal_names)
    nperseg = min(128, len(magnitude))
    freqs, psd = compute_power_spectrum(magnitude, sampling_frequency, nperseg=nperseg, noverlap=nperseg // 2)
    features = _generic_spectral_summary(freqs, psd, "acc")
    postural = band_power(freqs, psd, *ACC_BANDS["postural"])
    movement = band_power(freqs, psd, *ACC_BANDS["movement"])
    features["acc_postural_power"] = postural
    features["acc_movement_power"] = movement
    features["acc_movement_ratio"] = movement / postural if postural > 0 else 0.0
    features["acc_dominant_frequency"] = dominant_frequency(freqs, psd, *ACC_BANDS["movement"])
    return features


def extract_window_features(window: dict, sampling_frequency: float, signal_names: list[str]) -> dict:
    """
    Dispatch each channel in a window to its matching extractor and merge the results.

    Args:
        window:              dict with 'label', 'signal' (n_samples, n_channels), 'start_time' — e.g. from make_windows
        sampling_frequency:  sampling frequency of this window's record, in Hz
        signal_names:        channel names for window['signal'], in column order

    Returns a flat dict of prefixed features plus 'label' and 'start_time'.
    """
    signal = window["signal"]
    features = {"label": window["label"], "start_time": window["start_time"]}

    if {"ax", "ay", "az"}.issubset(signal_names):
        features.update(extract_acc_features(signal, signal_names, sampling_frequency))
    if "EDA" in signal_names:
        features.update(extract_eda_features(signal[:, signal_names.index("EDA")], sampling_frequency))
    if "temp" in signal_names:
        features.update(extract_generic_spectral_features(
            signal[:, signal_names.index("temp")], sampling_frequency, nperseg=128, prefix="temp"
        ))
    if "hr" in signal_names:
        features.update(extract_hr_features(signal[:, signal_names.index("hr")], sampling_frequency))
    if "SpO2" in signal_names:
        features.update(extract_generic_low_rate_features(
            signal[:, signal_names.index("SpO2")], sampling_frequency, prefix="spo2"
        ))

    return features
