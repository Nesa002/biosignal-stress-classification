"""Data-quality checks for the frequency-domain feature table (see notebooks/03_feature_extraction.ipynb)."""

import numpy as np
import pandas as pd

# Nyquist frequency (Hz) of the underlying signal group, used to bound
# *_dominant_frequency columns -- acc/eda/temp are 8 Hz signals, hr/spo2 are 1 Hz
# (see src/preprocessing.py's per-record sampling rates).
NYQUIST_BY_PREFIX = {
    'acc': 4.0, 'eda': 4.0, 'temp': 4.0,
    'hr': 0.5, 'spo2': 0.5,
}


def _prefix(column: str) -> str:
    return column.split('_', 1)[0]


def _sanity_bound(column: str) -> tuple[float | None, float | None]:
    """Construction-derived (lower, upper) bound for a column, or (None, None) if none applies."""
    name = column.lower()
    if name.endswith('dominant_frequency'):
        nyquist = NYQUIST_BY_PREFIX.get(_prefix(name))
        return (0.0, nyquist) if nyquist is not None else (None, None)
    if 'power' in name or 'entropy' in name or 'ratio' in name:
        return (0.0, None)
    return (None, None)


def missing_value_report(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """NaN and Inf counts per feature column."""
    values = df[feature_columns].to_numpy(dtype=float)
    return pd.DataFrame({
        'n_nan': np.isnan(values).sum(axis=0),
        'n_inf': np.isinf(values).sum(axis=0),
    }, index=feature_columns)


def sanity_violation_report(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Construction-derived bound violations: power/entropy/ratio columns must be
    non-negative (see src/features.py), and *_dominant_frequency columns must
    fall within [0, Nyquist] for their channel group. A violation here is
    numerical noise on an otherwise-legitimate window (e.g. Shannon entropy
    landing at -1e-12 for a fully concentrated spectrum), not a sign the
    window itself is bad -- see `clean_features`, which corrects rather than
    drops these.
    """
    rows = []
    for column in feature_columns:
        lower, upper = _sanity_bound(column)
        series = df[column]
        if lower is None and upper is None:
            rule, n_violations = 'n/a', 0
        else:
            below = series < lower if lower is not None else False
            above = series > upper if upper is not None else False
            n_violations = int((below | above).sum())
            rule = f'>= {lower}' if upper is None else f'in [{lower}, {upper}]'
        rows.append({'column': column, 'rule': rule, 'n_violations': n_violations})
    return pd.DataFrame(rows).set_index('column')


def clean_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Clip sanity-bound violations to their valid bound in place -- corrects floating-point noise without discarding the row."""
    cleaned = df.copy()
    for column in feature_columns:
        lower, upper = _sanity_bound(column)
        if lower is not None or upper is not None:
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
    return cleaned


def audit_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Per-column report combining missing-value and sanity-bound checks."""
    missing = missing_value_report(df, feature_columns)
    sanity = sanity_violation_report(df, feature_columns)
    return missing.join(sanity)
