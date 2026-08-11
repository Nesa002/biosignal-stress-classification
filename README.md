# Biosignal Stress Classification

This project investigates whether the type of stress a person is experiencing — physical exertion, cognitive load, or emotional stress — can be identified from wrist-worn biosignals alone. Using only accelerometer, electrodermal activity (EDA), skin temperature, heart rate, and blood oxygen saturation (SpO2), the goal is to classify which phase of a structured laboratory stress protocol a subject is currently in: Relax, Physical Stress, Cognitive Stress, or Emotional Stress.

## Dataset

The analysis is built on the *Non-EEG Dataset for Assessment of Neurological Status* (PhysioNet, v1.0.0): 20 healthy subjects wearing an Empatica E4 wrist sensor while completing a fixed eight-phase lab protocol (baseline relaxation, treadmill exercise, mental arithmetic and Stroop tasks, and a startle/horror-clip induction), each phase separated by a return to a relaxed baseline.

## Approach

Raw signals are filtered and normalized per subject, then split into overlapping 60-second windows (30-second step) within each phase. Two complementary feature sets are extracted per window: frequency-domain descriptors (via Welch's method, including physiologically defined band powers for EDA and heart-rate variability) and time-domain statistical descriptors (mean, standard deviation, skewness, slope). Three classifiers — Logistic Regression, Random Forest, and Gradient Boosting — are trained and compared using subject-grouped 5-fold cross-validation, with macro-averaged F1 as the primary metric to account for class imbalance.

## Result

The tuned Random Forest model performs best, reaching a macro-F1 of roughly 0.77 across all features. Investigating feature importance surfaced a notable confound: skin temperature partly acts as a proxy for elapsed session time rather than genuine physiological response, since every subject follows the same fixed phase order. Excluding temperature and SpO2 features drops performance to roughly 0.74 macro-F1 — a more trustworthy estimate of real-world performance, and a result treated in the project as more meaningful than the raw headline score.

## Where to look

[`notebooks/thesis_overview.ipynb`](notebooks/thesis_overview.ipynb) is the canonical entry point: a single, fully executed notebook that runs the real pipeline end to end (raw signals → preprocessing → features → modeling → conclusions), reproducing every number and plot live. The detailed step-by-step notebooks, with the full derivations and experiments behind each decision, live alongside it as `01_exploration.ipynb` through `05_modeling.ipynb`.