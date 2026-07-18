from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt


DATA_DIR = str(
    Path(__file__).parent.parent
    / "data/noneeg/non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
)


def load_record(record_path: str, annotation_extension: str | None = "atr") -> dict:
    """
    Load a WFDB record and optionally its annotations.

    Args:
        record_path:          full path to record without extension
        annotation_extension: annotation extension ('atr'), or None to skip

    Returns a dict with:
      - signal:               np.ndarray (n_samples, n_channels)
      - signal_names:         list of channel names
      - sampling_frequency:   sampling frequency (Hz)
      - annotation_samples:   sample indices of phase boundaries (empty if no annotation)
      - annotation_labels:    phase label at each boundary (empty if no annotation)
    """
    record = wfdb.rdrecord(record_path)

    annotation_samples, annotation_labels = [], []
    if annotation_extension is not None:
        annotation = wfdb.rdann(record_path, annotation_extension)
        annotation_samples = annotation.sample.tolist()
        annotation_labels = annotation.aux_note

    return {
        "signal": record.p_signal,
        "signal_names": record.sig_name,
        "sampling_frequency": record.fs,
        "annotation_samples": annotation_samples,
        "annotation_labels": annotation_labels,
    }


def get_phase_segments(record: dict) -> list[dict]:
    """
    Split a record into per-phase segments using annotation boundaries.

    Returns a list of dicts, each with:
      - label:  phase name (e.g. 'Relax', 'PhysicalStress')
      - signal: np.ndarray slice (n_samples, n_channels)
    """
    segments = []
    samples = record["annotation_samples"]
    labels = record["annotation_labels"]

    for i, (start, label) in enumerate(zip(samples, labels)):
        end = samples[i + 1] if i + 1 < len(samples) else len(record["signal"])
        segments.append({
            "label": label.strip(),
            "signal": record["signal"][start:end],
        })

    return segments


def filter_signal(signal: np.ndarray, sampling_frequency: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase low-pass Butterworth filter to every channel.

    Args:
        signal:              np.ndarray (n_samples, n_channels)
        sampling_frequency:  sampling frequency in Hz
        cutoff:              low-pass cutoff frequency in Hz
        order:               filter order

    Returns filtered signal of same shape.
    """
    nyquist = sampling_frequency / 2.0
    if cutoff >= nyquist:
        raise ValueError(f"cutoff ({cutoff} Hz) must be below Nyquist ({nyquist} Hz)")
    numerator_coeffs, denominator_coeffs = butter(order, cutoff / nyquist, btype="low")
    return filtfilt(numerator_coeffs, denominator_coeffs, signal, axis=0)


def normalize_record(record: dict) -> dict:
    """
    Z-score normalize a record's signal per channel across the full recording.

    Normalization stats are computed before segmentation to avoid leakage
    between phases. Near-constant channels (standard_deviation < 1e-8) are left at zero.

    Returns a new record dict with the same structure; original is unchanged.
    """
    signal = record["signal"]
    mean = signal.mean(axis=0)
    standard_deviation = signal.std(axis=0)
    standard_deviation = np.where(standard_deviation < 1e-8, 1.0, standard_deviation)
    return {**record, "signal": (signal - mean) / standard_deviation}


def subject_paths(subject_id: int) -> tuple[str, str]:
    """Return (AccTempEDA path, SpO2HR path) for a given subject number."""
    acc_temp_eda_path = f"{DATA_DIR}/Subject{subject_id}_AccTempEDA"
    spo2_hr_path = f"{DATA_DIR}/Subject{subject_id}_SpO2HR"
    return acc_temp_eda_path, spo2_hr_path


def make_windows(segment: dict, sampling_frequency: float, window_seconds: float, step_seconds: float) -> list[dict]:
    """
    Slice a phase segment into fixed-length, possibly overlapping windows.

    Segments shorter than window_seconds yield no windows — this is the
    mechanism that drops phases too short for a meaningful spectrum (e.g.
    the ~40s mini EmotionalStress phase) without any label-based special-casing.

    Args:
        segment:            dict with 'label' and 'signal' (n_samples, n_channels), e.g. from get_phase_segments
        sampling_frequency:  sampling frequency in Hz
        window_seconds:      window length in seconds
        step_seconds:        step between window starts, in seconds

    Returns a list of dicts, each with:
      - label:       phase name, copied from the segment
      - signal:      np.ndarray slice (n_window_samples, n_channels)
      - start_time:  offset of the window's start within the segment, in seconds
    """
    window_samples = round(window_seconds * sampling_frequency)
    step_samples = round(step_seconds * sampling_frequency)
    signal = segment["signal"]

    windows = []
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        windows.append({
            "label": segment["label"],
            "signal": signal[start:start + window_samples],
            "start_time": start / sampling_frequency,
        })

    return windows


def get_phase_windows(record: dict, window_seconds: float, step_seconds: float) -> list[dict]:
    """
    Split a record into per-phase segments, then slice each into fixed-length windows.

    Windows never cross a phase boundary — phase segments come from
    get_phase_segments, so the annotation boundaries still define window labels
    and limits even though windows are shorter than a phase.

    Returns a flattened list of window dicts (see make_windows).
    """
    windows = []
    for segment in get_phase_segments(record):
        windows.extend(make_windows(segment, record["sampling_frequency"], window_seconds, step_seconds))
    return windows


def load_and_preprocess_subject(
    subject_id: int,
    accel_cutoff: float | None = 1.0,
    eda_temp_cutoff: float | None = 1.0,
    spo2_hr_cutoff: float | None = 0.1,
) -> tuple[dict, dict]:
    """
    Load, optionally low-pass filter, and normalize both records for one subject.

    By default reproduces notebook 02's time-domain pipeline: load both
    records, rescale the SpO2HR phase boundaries from AccTempEDA's
    annotations (SpO2HR has no annotation file of its own), low-pass filter
    (1.0 Hz cutoff for AccTempEDA's channels, 0.1 Hz for SpO2HR), then
    z-score normalize.

    AccTempEDA's accelerometer (ax, ay, az) and EDA/temp channels are
    filtered as two independent groups, each with its own cutoff, rather
    than one shared cutoff for the whole record: their frequency content of
    interest lives in very different ranges (movement: 0.5-4 Hz, right up to
    this record's 4 Hz Nyquist; EDA/temp: well under 1 Hz), so a single
    shared cutoff can't serve both without either doing nothing or cutting
    into one of them.

    Pass a cutoff of None to skip filtering that group/record. This matters
    for frequency-domain analysis: notebook 02's cutoffs were tuned to
    denoise for time-domain trend visualization, and applying them before
    FFT/Welch would remove the exact bands spectral features analyze —
    HR/SpO2 LF/HF needs content up to 0.4 Hz but 0.1 Hz cutoff removes it,
    and Acc's movement/gait band (0.5-4 Hz) already reaches this record's
    Nyquist, leaving no headroom for any cutoff below it. EDA/temp's bands
    top out at 0.25 Hz, leaving real headroom (0.25-4 Hz) for their own
    cutoff to still help, independent of the accelerometer.

    Returns (acc_temp_eda_record, spo2_hr_record), both normalized (and
    filtered wherever a cutoff was given).
    """
    acc_temp_eda_path, spo2_hr_path = subject_paths(subject_id)

    record_acc_temp_eda = load_record(acc_temp_eda_path, annotation_extension="atr")
    record_spo2_hr = load_record(spo2_hr_path, annotation_extension=None)

    boundary_seconds = [
        s / record_acc_temp_eda["sampling_frequency"] for s in record_acc_temp_eda["annotation_samples"]
    ]
    record_spo2_hr["annotation_samples"] = [
        round(t * record_spo2_hr["sampling_frequency"]) for t in boundary_seconds
    ]
    record_spo2_hr["annotation_labels"] = record_acc_temp_eda["annotation_labels"]

    if accel_cutoff is not None or eda_temp_cutoff is not None:
        signal = record_acc_temp_eda["signal"].copy()
        signal_names = record_acc_temp_eda["signal_names"]
        sampling_frequency = record_acc_temp_eda["sampling_frequency"]

        if accel_cutoff is not None:
            accel_indices = [signal_names.index(name) for name in ("ax", "ay", "az")]
            signal[:, accel_indices] = filter_signal(
                signal[:, accel_indices], sampling_frequency=sampling_frequency, cutoff=accel_cutoff
            )

        if eda_temp_cutoff is not None:
            eda_temp_indices = [signal_names.index(name) for name in ("temp", "EDA")]
            signal[:, eda_temp_indices] = filter_signal(
                signal[:, eda_temp_indices], sampling_frequency=sampling_frequency, cutoff=eda_temp_cutoff
            )

        record_acc_temp_eda = {**record_acc_temp_eda, "signal": signal}

    if spo2_hr_cutoff is not None:
        spo2_hr_filtered = filter_signal(
            record_spo2_hr["signal"], sampling_frequency=record_spo2_hr["sampling_frequency"], cutoff=spo2_hr_cutoff
        )
        record_spo2_hr = {**record_spo2_hr, "signal": spo2_hr_filtered}

    return normalize_record(record_acc_temp_eda), normalize_record(record_spo2_hr)


def group_by_phase_occurrence(windows: list[dict]) -> list[list[dict]]:
    """Group consecutive windows sharing the same phase label into ordered occurrence groups."""
    return [list(group) for _, group in groupby(windows, key=lambda w: w["label"])]


def assemble_subject_rows(
    record_acc: dict,
    record_hr: dict,
    window_seconds: float,
    step_seconds: float,
    extractor: Callable[[dict, float, list[str]], dict],
) -> list[dict]:
    """
    Build one feature row per aligned (AccTempEDA, SpO2HR) window pair for a subject.

    Windows are grouped by phase occurrence (not just label, since e.g. 'Relax'
    recurs 4 times) and paired index-wise within each occurrence; any per-phase
    count mismatch between the two records (possible after 1 Hz boundary
    rounding) is resolved by trimming to the shorter of the two.

    `extractor(window, sampling_frequency, signal_names) -> dict` is called once
    per window (once for the AccTempEDA window, once for the matching SpO2HR
    window); the two resulting dicts are merged into one row, with the second
    call's 'label'/'start_time' dropped in favor of the first's. Each row also
    gets a 'window_index' (0-based position in this subject's window
    sequence) -- 'start_time' resets to 0 within every phase occurrence (e.g.
    'Relax' recurs 4 times), so it isn't unique on its own; 'window_index' is,
    and is meant to be combined with 'subject_id' to join two independently-
    computed feature tables (e.g. frequency-domain and statistical) that were
    built by calling this function with the same window/step parameters.
    """
    windows_acc = get_phase_windows(record_acc, window_seconds, step_seconds)
    windows_hr = get_phase_windows(record_hr, window_seconds, step_seconds)

    acc_groups = group_by_phase_occurrence(windows_acc)
    hr_groups = group_by_phase_occurrence(windows_hr)
    assert len(acc_groups) == len(hr_groups), "Phase-occurrence count mismatch between AccTempEDA and SpO2HR windows"

    rows = []
    for acc_group, hr_group in zip(acc_groups, hr_groups):
        n = min(len(acc_group), len(hr_group))
        for acc_window, hr_window in zip(acc_group[:n], hr_group[:n]):
            acc_features = extractor(acc_window, record_acc["sampling_frequency"], record_acc["signal_names"])
            hr_features = extractor(hr_window, record_hr["sampling_frequency"], record_hr["signal_names"])
            hr_features.pop("start_time")
            hr_features.pop("label")
            rows.append({"window_index": len(rows), **acc_features, **hr_features})

    return rows
