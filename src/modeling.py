import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "Gradient Boosting": {
        "clf__max_iter": [100, 200, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__learning_rate": [0.05, 0.1, 0.2],
    },
}


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """StandardScaler + classifier pipelines for the 3 models kept after dropping k-NN/SVM."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state)),
        ]),
    }


def run_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, models: dict[str, Pipeline], random_state: int = RANDOM_STATE
) -> dict:
    """
    5-fold StratifiedGroupKFold evaluation, grouped by subject so a subject's windows
    never span train and test. Returns per-model oof_predictions and macro-F1/accuracy summaries.
    """
    class_order = sorted(y.unique())
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    for name, pipeline in models.items():
        oof_predictions = pd.Series(index=X.index, dtype=object)
        fold_macro_f1 = []
        for train_idx, test_idx in cv.split(X, y, groups):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = pipeline.predict(X.iloc[test_idx])
            oof_predictions.iloc[test_idx] = predictions
            fold_macro_f1.append(f1_score(y.iloc[test_idx], predictions, labels=class_order, average="macro"))
        results[name] = {
            "oof_predictions": oof_predictions,
            "fold_macro_f1_mean": float(np.mean(fold_macro_f1)),
            "fold_macro_f1_std": float(np.std(fold_macro_f1)),
            "oof_macro_f1": f1_score(y, oof_predictions, labels=class_order, average="macro"),
            "oof_accuracy": accuracy_score(y, oof_predictions),
        }
    return results


def tune_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    models: dict[str, Pipeline],
    param_grids: dict[str, dict] = PARAM_GRIDS,
    scoring: str = "f1_macro",
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Grid search each model's hyperparameters, scored on the same grouped 5-fold split
    used for evaluation (StratifiedGroupKFold by subject) so the search itself can't
    leak a subject's windows across its own train/validation split.

    This is a single tuning pass over the whole dataset, not fully nested per outer
    fold — the resulting score is "best CV score found," not an estimate independent
    of the search that chose it.
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    tuned = {}
    for name, pipeline in models.items():
        grid = param_grids.get(name)
        if not grid:
            tuned[name] = {"pipeline": pipeline, "best_params": {}}
            continue
        search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X, y, groups=groups)
        tuned[name] = {"pipeline": search.best_estimator_, "best_params": search.best_params_}
    return tuned
