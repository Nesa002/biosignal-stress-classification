from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt


DATA_DIR = str(
    Path(__file__).parent.parent
    / "data/noneeg/non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
)


def load_record(record_path: str, ann_ext: str | None = "atr") -> dict:
    """
    Load a WFDB record and optionally its annotations.

    Args:
        record_path: full path to record without extension
        ann_ext:     annotation extension ('atr'), or None to skip

    Returns a dict with:
      - signal:      np.ndarray (n_samples, n_channels)
      - sig_name:    list of channel names
      - fs:          sampling frequency (Hz)
      - ann_samples: sample indices of phase boundaries (empty if no annotation)
      - ann_labels:  phase label at each boundary (empty if no annotation)
    """
    record = wfdb.rdrecord(record_path)

    ann_samples, ann_labels = [], []
    if ann_ext is not None:
        ann = wfdb.rdann(record_path, ann_ext)
        ann_samples = ann.sample.tolist()
        ann_labels = ann.aux_note

    return {
        "signal": record.p_signal,
        "sig_name": record.sig_name,
        "fs": record.fs,
        "ann_samples": ann_samples,
        "ann_labels": ann_labels,
    }


def get_phase_segments(record: dict) -> list[dict]:
    """
    Split a record into per-phase segments using annotation boundaries.

    Returns a list of dicts, each with:
      - label:  phase name (e.g. 'Relax', 'PhysicalStress')
      - signal: np.ndarray slice (n_samples, n_channels)
    """
    segments = []
    samples = record["ann_samples"]
    labels = record["ann_labels"]

    for i, (start, label) in enumerate(zip(samples, labels)):
        end = samples[i + 1] if i + 1 < len(samples) else len(record["signal"])
        segments.append({
            "label": label.strip(),
            "signal": record["signal"][start:end],
        })

    return segments


def filter_signal(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase low-pass Butterworth filter to every channel.

    Args:
        signal: np.ndarray (n_samples, n_channels)
        fs:     sampling frequency in Hz
        cutoff: low-pass cutoff frequency in Hz
        order:  filter order

    Returns filtered signal of same shape.
    """
    nyq = fs / 2.0
    if cutoff >= nyq:
        raise ValueError(f"cutoff ({cutoff} Hz) must be below Nyquist ({nyq} Hz)")
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


def normalize_record(record: dict) -> dict:
    """
    Z-score normalize a record's signal per channel across the full recording.

    Normalization stats are computed before segmentation to avoid leakage
    between phases. Near-constant channels (std < 1e-8) are left at zero.

    Returns a new record dict with the same structure; original is unchanged.
    """
    sig = record["signal"]
    mean = sig.mean(axis=0)
    std = sig.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {**record, "signal": (sig - mean) / std}


def subject_paths(subject_id: int) -> tuple[str, str]:
    """Return (AccTempEDA path, SpO2HR path) for a given subject number."""
    eda = f"{DATA_DIR}/Subject{subject_id}_AccTempEDA"
    hr  = f"{DATA_DIR}/Subject{subject_id}_SpO2HR"
    return eda, hr
