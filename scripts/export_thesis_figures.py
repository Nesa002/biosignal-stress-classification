"""
Exports the figures selected for the thesis (see thesis_overview.ipynb for the
narrative versions these are lifted from) as standalone PNG files, so they can
be copied into the Typst thesis repo's slike/ directory.

Produces, under outputs/figures/:
  - subject1_raw_{ax,ay,az,temp,eda,spo2,hr}.png   -- 7 raw single-channel plots
  - subject1_filter_{eda,temp}.png                 -- filtered-vs-raw + residual, zoomed
  - subject1_spectrogram_{acc,eda,hr}.png          -- Relax vs. a stress phase, per signal
  - confusion_matrices.png                         -- 3-panel, tuned models, all 20 subjects
  - model_comparison.png                           -- tuned macro-F1 bar chart, 3 models
  - permutation_importance.png                     -- top-15 features, untuned RF

And, under outputs/models/:
  - best_model_<name>.joblib                       -- best tuned pipeline, refit on all data

Run from anywhere: `python scripts/export_thesis_figures.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_quality import clean_features
from src.features import compute_acc_magnitude, compute_spectrogram, extract_window_features
from src.modeling import PARAM_GRIDS, RANDOM_STATE, build_models, run_cv, tune_models
from src.preprocessing import (
    assemble_subject_rows,
    filter_signal,
    get_phase_segments,
    load_and_preprocess_subject,
    load_record,
    subject_paths,
)
from src.statistical_features import extract_window_statistical_features

CLASS_ORDER = ["Relax", "PhysicalStress", "CognitiveStress", "EmotionalStress"]
WINDOW_SECONDS, STEP_SECONDS = 60, 30
METADATA_COLUMNS = ["window_index", "label", "start_time", "subject_id"]
# <=0 permutation importance in the full run (see thesis_overview.ipynb section 8)
LOW_IMPORTANCE_COLUMNS = [
    "acc_total_power", "eda_total_power", "spo2_dominant_frequency", "acc_postural_power",
    "eda_skewness", "hr_lf_power", "hr_dominant_frequency", "eda_dominant_frequency", "temp_dominant_frequency",
]

FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "outputs" / "models"
DPI = 200


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


# --- Section A: 7 raw single-channel plots, Subject 1 -----------------------

def plot_single_channel(time, values, phase_times, phase_labels, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.plot(time, values, lw=0.6, color="steelblue")
    ymin, ymax = ax.get_ylim()
    headroom = ymax + (ymax - ymin) * 0.35
    ax.set_ylim(ymin, headroom)
    for t, label in zip(phase_times, phase_labels):
        ax.axvline(t, color="red", lw=0.7, alpha=0.5)
        ax.text(t + 1, ymax + (ymax - ymin) * 0.03, label, fontsize=7, color="red", rotation=45, va="bottom")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def export_raw_channels():
    print("== raw channel plots (Subject 1) ==")
    acc_path, hr_path = subject_paths(1)
    record_acc = load_record(acc_path, annotation_extension="atr")
    record_hr = load_record(hr_path, annotation_extension=None)

    # SpO2HR carries no annotations of its own; phase boundaries are shared real
    # time, so AccTempEDA's boundaries (converted to seconds) apply to both.
    phase_times = [s / record_acc["sampling_frequency"] for s in record_acc["annotation_samples"]]
    phase_labels = record_acc["annotation_labels"]

    fs_acc = record_acc["sampling_frequency"]
    time_acc = np.arange(record_acc["signal"].shape[0]) / fs_acc
    for channel in ("ax", "ay", "az", "temp", "EDA"):
        idx = record_acc["signal_names"].index(channel)
        fig = plot_single_channel(
            time_acc, record_acc["signal"][:, idx], phase_times, phase_labels,
            ylabel=channel, title=f"Subject 1 -- {channel} (raw, phase boundaries in red)",
        )
        save(fig, f"subject1_raw_{channel.lower()}.png")

    fs_hr = record_hr["sampling_frequency"]
    time_hr = np.arange(record_hr["signal"].shape[0]) / fs_hr
    for channel in ("SpO2", "hr"):
        idx = record_hr["signal_names"].index(channel)
        fig = plot_single_channel(
            time_hr, record_hr["signal"][:, idx], phase_times, phase_labels,
            ylabel=channel, title=f"Subject 1 -- {channel} (raw, phase boundaries in red)",
        )
        save(fig, f"subject1_raw_{channel.lower()}.png")


# --- Section B: filtered vs. raw + residual, zoomed, EDA and temp -----------

def plot_filter_effect(raw_full, filtered_full, fs, channel_label, cutoff, window_start=630, window_end=720):
    residual_full = raw_full - filtered_full
    start_sample, end_sample = int(window_start * fs), int(window_end * fs)
    time_window = np.arange(start_sample, end_sample) / fs
    raw_window = raw_full[start_sample:end_sample]
    filtered_window = filtered_full[start_sample:end_sample]
    residual_window = residual_full[start_sample:end_sample]

    fig, (ax_signal, ax_residual) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax_signal.plot(time_window, raw_window, lw=1.0, alpha=0.6, label="raw", color="gray")
    ax_signal.plot(time_window, filtered_window, lw=1.0, label=f"filtered ({cutoff:.1f} Hz low-pass)", color="steelblue")
    ax_signal.set_ylabel(channel_label)
    ax_signal.set_title(f"Subject 1 -- {channel_label}, zoomed to {window_end - window_start}s: raw vs. filtered")
    ax_signal.legend(loc="upper right", fontsize=8)
    ax_signal.grid(alpha=0.3)

    ax_residual.plot(time_window, residual_window, lw=0.8, color="firebrick")
    ax_residual.axhline(0, color="black", lw=0.5)
    ax_residual.set_xlabel("Time (s)")
    ax_residual.set_ylabel("Removed")
    ax_residual.set_title(f"What the filter actually removed: content above {cutoff:.1f} Hz")
    ax_residual.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def export_filter_effect():
    print("== filter before/after (Subject 1: EDA, temp) ==")
    acc_path, _ = subject_paths(1)
    record_acc = load_record(acc_path, annotation_extension="atr")
    fs = record_acc["sampling_frequency"]
    names = record_acc["signal_names"]
    cutoff = 1.0

    for channel in ("EDA", "temp"):
        idx = names.index(channel)
        raw_full = record_acc["signal"][:, idx]
        filtered_full = filter_signal(record_acc["signal"], sampling_frequency=fs, cutoff=cutoff)[:, idx]
        fig = plot_filter_effect(raw_full, filtered_full, fs, channel, cutoff)
        save(fig, f"subject1_filter_{channel.lower()}.png")


# --- Section C: spectrograms, Relax vs. a stress phase, per signal ----------

def plot_spectrogram_pair(signal_a, signal_b, label_a, label_b, fs, title, nperseg, noverlap, dynamic_range_db=40):
    freqs_a, times_a, sxx_a = compute_spectrogram(signal_a, fs, nperseg, noverlap)
    freqs_b, times_b, sxx_b = compute_spectrogram(signal_b, fs, nperseg, noverlap)

    epsilon = 1e-12
    sxx_a_db, sxx_b_db = 10 * np.log10(sxx_a + epsilon), 10 * np.log10(sxx_b + epsilon)
    vmax = max(sxx_a_db.max(), sxx_b_db.max())
    vmin = vmax - dynamic_range_db

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharey=True, constrained_layout=True)
    mesh = None
    for ax, freqs, times, sxx_db, label in [
        (axes[0], freqs_a, times_a, sxx_a_db, label_a), (axes[1], freqs_b, times_b, sxx_b_db, label_b),
    ]:
        mesh = ax.pcolormesh(times, freqs, sxx_db, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes, label="Power (dB)", fraction=0.046, pad=0.02)
    fig.suptitle(title)
    return fig


def export_spectrograms():
    print("== spectrograms (Subject 1) ==")
    acc_path, hr_path = subject_paths(1)
    record_acc = load_record(acc_path, annotation_extension="atr")
    record_hr_raw = load_record(hr_path, annotation_extension=None)

    boundary_seconds = [s / record_acc["sampling_frequency"] for s in record_acc["annotation_samples"]]
    record_hr = {
        **record_hr_raw,
        "annotation_samples": [round(t * record_hr_raw["sampling_frequency"]) for t in boundary_seconds],
        "annotation_labels": record_acc["annotation_labels"],
    }

    acc_segments = get_phase_segments(record_acc)
    hr_segments = get_phase_segments(record_hr)
    names = record_acc["signal_names"]
    # segment order for this protocol: 0=Relax, 1=PhysicalStress, 2=Relax,
    # 3=EmotionalStress (mini, ~40s), 4=CognitiveStress, 5=Relax,
    # 6=EmotionalStress (main), 7=Relax
    RELAX, PHYSICAL, COGNITIVE, EMOTIONAL = 0, 1, 4, 6

    relax_acc = compute_acc_magnitude(acc_segments[RELAX]["signal"], names)
    physical_acc = compute_acc_magnitude(acc_segments[PHYSICAL]["signal"], names)
    fig = plot_spectrogram_pair(
        relax_acc, physical_acc, "Relax", "PhysicalStress", record_acc["sampling_frequency"],
        "Subject 1 -- Acc magnitude spectrogram: Relax vs. PhysicalStress", nperseg=64, noverlap=48,
    )
    save(fig, "subject1_spectrogram_acc.png")

    eda_idx = names.index("EDA")
    relax_eda = acc_segments[RELAX]["signal"][:, eda_idx]
    cognitive_eda = acc_segments[COGNITIVE]["signal"][:, eda_idx]
    fig = plot_spectrogram_pair(
        relax_eda, cognitive_eda, "Relax", "CognitiveStress", record_acc["sampling_frequency"],
        "Subject 1 -- EDA spectrogram: Relax vs. CognitiveStress", nperseg=128, noverlap=96,
    )
    save(fig, "subject1_spectrogram_eda.png")

    hr_idx = record_hr["signal_names"].index("hr")
    relax_hr = hr_segments[RELAX]["signal"][:, hr_idx]
    emotional_hr = hr_segments[EMOTIONAL]["signal"][:, hr_idx]
    fig = plot_spectrogram_pair(
        relax_hr, emotional_hr, "Relax", "EmotionalStress", record_hr["sampling_frequency"],
        "Subject 1 -- HR spectrogram: Relax vs. EmotionalStress", nperseg=32, noverlap=24,
    )
    save(fig, "subject1_spectrogram_hr.png")


# --- Section D: confusion matrices, tuned models, all 20 subjects -----------

def build_combined_features():
    frequency_rows, stat_rows = [], []
    for subject_id in range(1, 21):
        r_acc, r_hr = load_and_preprocess_subject(subject_id, accel_cutoff=None, eda_temp_cutoff=1.0, spo2_hr_cutoff=None)
        freq_rows = assemble_subject_rows(r_acc, r_hr, WINDOW_SECONDS, STEP_SECONDS, extract_window_features)
        stat_rows_subject = assemble_subject_rows(r_acc, r_hr, WINDOW_SECONDS, STEP_SECONDS, extract_window_statistical_features)
        for row in freq_rows:
            row["subject_id"] = subject_id
        for row in stat_rows_subject:
            row["subject_id"] = subject_id
        frequency_rows.extend(freq_rows)
        stat_rows.extend(stat_rows_subject)

    frequency_features_df = pd.DataFrame(frequency_rows)
    statistical_features_df = pd.DataFrame(stat_rows)

    feature_columns = [c for c in frequency_features_df.columns if c not in METADATA_COLUMNS]
    frequency_features_df = clean_features(frequency_features_df, feature_columns)

    combined_features_df = frequency_features_df.merge(
        statistical_features_df.drop(columns=["label", "start_time"]),
        on=["subject_id", "window_index"], how="inner", validate="one_to_one",
    )
    return combined_features_df


def run_full_pipeline():
    """
    Builds the combined feature table (all 20 subjects), grid-searches and
    cross-validates all three models. Shared by every Section D figure so the
    expensive GridSearchCV only runs once.

    Returns (X, y, results, tuned):
      - results is run_cv's per-model dict (oof_predictions, oof_macro_f1,
        fold_macro_f1_mean/std), keyed in build_models() order (Logistic
        Regression, Random Forest, Gradient Boosting).
      - tuned is tune_models' per-model dict (pipeline, best_params). Its
        "pipeline" ends up refit on just the last CV fold (run_cv mutates
        the same objects afterwards for evaluation), so export_best_model
        rebuilds a fresh pipeline from "best_params" and refits it on the
        full dataset rather than reusing "pipeline" directly.
    """
    print("  building combined feature table for all 20 subjects...")
    combined_features_df = build_combined_features()

    feature_columns = [c for c in combined_features_df.columns if c not in METADATA_COLUMNS + LOW_IMPORTANCE_COLUMNS]
    X = combined_features_df[feature_columns]
    y = combined_features_df["label"]
    groups = combined_features_df["subject_id"]
    print(f"  {X.shape[0]} windows x {X.shape[1]} features, {groups.nunique()} subjects")

    print("  grid-searching hyperparameters (StratifiedGroupKFold, 5 folds)...")
    models = build_models()
    tuned = tune_models(X, y, groups, models, PARAM_GRIDS)
    tuned_models = {name: info["pipeline"] for name, info in tuned.items()}
    results = run_cv(X, y, groups, tuned_models)
    return X, y, results, tuned


def export_confusion_matrices(y, results):
    print("== confusion matrices (all 20 subjects, tuned models) ==")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (model_name, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y, r["oof_predictions"], labels=CLASS_ORDER, normalize="true", cmap="Blues",
            ax=ax, colorbar=False, xticks_rotation=45, values_format=".2%",
        )
        ax.set_title(f"{model_name}\n(macro-F1={r['oof_macro_f1']:.3f})")
    fig.tight_layout()
    save(fig, "confusion_matrices.png")


def export_model_comparison(results):
    print("== model comparison bar chart (tuned models) ==")
    summary_df = pd.DataFrame([
        {"model": name, "oof_macro_f1": r["oof_macro_f1"], "fold_macro_f1_std": r["fold_macro_f1_std"]}
        for name, r in results.items()
    ]).sort_values("oof_macro_f1")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(summary_df["model"], summary_df["oof_macro_f1"], xerr=summary_df["fold_macro_f1_std"], color="steelblue", capsize=4)
    ax.set_xlabel("Macro-averaged F1 (out-of-fold)")
    ax.set_xlim(0, 1)
    ax.set_title("Tuned model comparison -- 5-fold subject-grouped cross-validation")
    ax.grid(axis="x", alpha=0.3)
    for bar, value, std in zip(bars, summary_df["oof_macro_f1"], summary_df["fold_macro_f1_std"]):
        label_x = min(value + std + 0.02, 0.94)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "model_comparison.png")


def export_permutation_importance(X, y, top_n=15):
    print("== permutation importance bar chart ==")
    # Matches thesis_overview.ipynb section 8: an untuned RF (n_estimators=300,
    # library defaults otherwise), fit on the full feature table -- deliberately
    # not the tuned pipeline, to match the numbers already reported in the thesis.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X, y)
    result = permutation_importance(
        pipeline, X, y, n_repeats=15, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X.columns, "importance": result.importances_mean,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1.5))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Permutation importance (Δ macro-F1)")
    ax.set_title(f"Top {top_n} features by permutation importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "permutation_importance.png")


def export_best_model(X, y, results, tuned):
    print("== exporting best model (refit on full dataset) ==")
    best_name = max(results, key=lambda name: results[name]["oof_macro_f1"])
    best_params = tuned[best_name]["best_params"]
    print(f"  best model: {best_name} (macro-F1={results[best_name]['oof_macro_f1']:.3f}), params={best_params}")

    pipeline = build_models()[best_name]
    pipeline.set_params(**best_params)
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = best_name.lower().replace(" ", "_")
    path = MODELS_DIR / f"best_model_{slug}.joblib"
    joblib.dump(pipeline, path)
    print(f"  saved {path.relative_to(ROOT)}")


def main():
    export_raw_channels()
    export_filter_effect()
    export_spectrograms()
    print("== full pipeline (all 20 subjects, tuned models) ==")
    X, y, results, tuned = run_full_pipeline()
    export_confusion_matrices(y, results)
    export_model_comparison(results)
    export_permutation_importance(X, y)
    export_best_model(X, y, results, tuned)
    print(f"\ndone -- figures in {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
