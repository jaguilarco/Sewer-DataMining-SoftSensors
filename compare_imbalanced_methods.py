from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# Optionals (skip if not installed)
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    LGBMClassifier = None
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    CatBoostClassifier = None
    _HAS_CAT = False

# =========================
# GLOBAL CONFIGURATION
# =========================

CSV_PATH = "dataset_sadeco.csv"   # ← Path to your CSV file
SEED = 0
N_SPLITS = 10
N_REPEATS = 10

OUT_PNG = "results_activity_table_pr.png"
OUT_SVG = "results_activity_table_pr.svg"
OUT_DIR = "algorithms_results"

VERBOSE_EVERY = 10

# =========================
# DATASET-SPECIFIC CONFIG
# =========================
FEATURE_COLS = ["Temperature", "Humidity", "CarbonDioxide"]
TARGET_COL = "Activity"

LABEL_MAP = {0: "None", 1: "Low", 2: "High"}
MINORITY_CLASS = 2  # High


# -------------------------
# Utilities
# -------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means)

    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(means, alpha)
    hi = np.quantile(means, 1 - alpha)
    return (float(np.mean(values)), float(lo), float(hi))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)


def get_proba_matrix(est, X_test: np.ndarray, n_classes: int = 3) -> Optional[np.ndarray]:
    """
    Returns a matrix (n_samples, n_classes) with probabilities if possible.
    - predict_proba is preferred.
    - Otherwise, decision_function + softmax (if multiclass).
    - If not possible, returns None.
    """
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_test)
        # Align columns to [0,1,2] if the estimator has classes_ attribute
        if hasattr(est, "classes_"):
            classes = list(est.classes_)
            out = np.zeros((proba.shape[0], n_classes), dtype=float)
            for k in range(n_classes):
                if k in classes:
                    j = classes.index(k)
                    out[:, k] = proba[:, j]
            # If a class was missing, it remains 0 (unlikely but safe)
            return out
        return proba

    if hasattr(est, "decision_function"):
        scores = est.decision_function(X_test)
        if scores.ndim == 1:
            # binary: not applicable here (3 classes problem)
            return None
        return _softmax(scores)

    return None


def get_minority_scores(est, X_test: np.ndarray, minority_class: int) -> np.ndarray:
    """
    Returns score/probability for the minority_class (one-vs-rest).
    Prefer predict_proba. Otherwise, decision_function + softmax.
    """
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_test)
        if hasattr(est, "classes_"):
            classes = list(est.classes_)
            if minority_class in classes:
                j = classes.index(minority_class)
                return proba[:, j]
        return proba[:, int(minority_class)]

    if hasattr(est, "decision_function"):
        scores = est.decision_function(X_test)
        if scores.ndim == 1:
            return scores
        proba = _softmax(scores)
        return proba[:, int(minority_class)]

    # fallback: null binary score
    return np.zeros(len(X_test), dtype=float)


def compute_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """
    Computes 'balanced' class weights (inverse frequency) as sample_weight.
    Useful for multiclass XGBoost (which lacks a standard class_weight parameter).
    """
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_map = {c: w for c, w in zip(classes, cw)}
    return np.asarray([cw_map[int(v)] for v in y], dtype=float)


class BalancedSampleWeightWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper: calculates balanced sample_weight during fit and passes it to the base estimator.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator)
        sw = compute_balanced_sample_weight(np.asarray(y))
        try:
            self.model_.fit(X, y, sample_weight=sw)
        except TypeError:
            self.model_.fit(X, y)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)


# -------------------------
# Numerical Preprocessing
# -------------------------
def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_models(seed: int) -> Dict[str, BaseEstimator]:
    lr_std = LogisticRegression(max_iter=8000, solver="lbfgs", random_state=seed)
    lr_bal = LogisticRegression(max_iter=8000, solver="lbfgs", class_weight="balanced", random_state=seed)

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )

    dt = DecisionTreeClassifier(
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
    )

    knn = KNeighborsClassifier(n_neighbors=7)

    brf = BalancedRandomForestClassifier(
        n_estimators=200,
        bootstrap=False,
        sampling_strategy="all",
        replacement=True,
        random_state=seed,
    )

    models: Dict[str, BaseEstimator] = {
        "Softmax_LR": lr_std,
        "Softmax_LR_balanced": lr_bal,
        "SMOTE+Softmax_LR": "SMOTE_LR",
        "BalancedRF": brf,
        "SVM_balanced": svm,
        "DecisionTree_balanced": dt,
        "KNN": knn,
    }

    # LightGBM balanced
    if _HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
            verbose=-1,
            min_split_gain=0.0,
        )
        models["LightGBM_balanced"] = lgbm

    # XGBoost balanced (via sample_weight)
    if _HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        models["XGBoost_balanced"] = BalancedSampleWeightWrapper(xgb)

    # CatBoost balanced
    if _HAS_CAT:
        cat = CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            depth=6,
            learning_rate=0.05,
            iterations=800,
            random_seed=seed,
            verbose=False,
        )
        models["CatBoost_balanced"] = cat

    # Ensemble soft voting (only if >=3 models are available)
    pre = build_preprocessor()
    ensemble_estimators = []
    candidate_names = [
        "Softmax_LR_balanced",
        "SVM_balanced",
        "KNN",
        "BalancedRF",
        "LightGBM_balanced",
        "XGBoost_balanced",
        "CatBoost_balanced",
    ]
    for name in candidate_names:
        if name not in models:
            continue
        m = models[name]
        if isinstance(m, str):
            continue
        ensemble_estimators.append((name, Pipeline([("pre", pre), ("clf", m)])))

    if len(ensemble_estimators) >= 3:
        ens = VotingClassifier(
            estimators=ensemble_estimators,
            voting="soft",
            n_jobs=None,
        )
        models["Ensemble_soft"] = ens

    return models


# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score_high: np.ndarray) -> Dict[str, float]:
    prec_u = precision_score(y_true, y_pred, labels=[MINORITY_CLASS], average=None, zero_division=0)[0]
    rec_u = recall_score(y_true, y_pred, labels=[MINORITY_CLASS], average=None, zero_division=0)[0]
    f1_u = f1_score(y_true, y_pred, labels=[MINORITY_CLASS], average=None, zero_division=0)[0]
    y_true_bin = (y_true == MINORITY_CLASS).astype(int)
    auprc = float(average_precision_score(y_true_bin, y_score_high))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_HIGH": float(prec_u),
        "recall_HIGH": float(rec_u),
        "f1_HIGH": float(f1_u),
        "auprc_HIGH": auprc,
    }


@dataclass
class ResultRow:
    method: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float


def evaluate_with_mljar_outputs(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    out_dir: Path,
    verbose_every: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    - Executes repeated stratified k-fold.
    - Saves:
        * oof_predictions_raw.csv (instance x fold x repeat)
        * fold_metrics.csv
        * oof_predictions_aggregated.csv (per instance, mean proba + final pred)
    - Returns summary_df (mean + CI) and oof_scores (for PR curves) using aggregated OOF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = build_preprocessor()
    models = make_models(seed=seed)

    metric_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "precision_HIGH", "recall_HIGH", "f1_HIGH", "auprc_HIGH"
    ]

    # Storage for CI (by fold)
    metrics_store: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in metric_keys} for m in models.keys()}

    # raw oof (by fold)
    raw_preds_by_method: Dict[str, List[pd.DataFrame]] = {m: [] for m in models.keys()}
    fold_metrics_rows: Dict[str, List[dict]] = {m: [] for m in models.keys()}

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    total_folds = n_splits * n_repeats

    # Reconstruct repeat/fold labels since RepeatedStratifiedKFold doesn't provide them explicitly:
    # fold_i from 0..total_folds-1 => repeat = fold_i // n_splits, fold = fold_i % n_splits
    for fold_i, (tr, te) in enumerate(cv.split(X, y), start=0):
        repeat_id = fold_i // n_splits
        fold_id = fold_i % n_splits

        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        test_idx = te  # instance_id = original index from CSV (after dropna)

        for model_name, model in models.items():
            if model_name == "SMOTE+Softmax_LR":
                pipe = ImbPipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=seed)),
                    ("clf", LogisticRegression(
                        max_iter=8000,
                        solver="lbfgs",
                        random_state=seed
                    )),
                ])
            else:
                if isinstance(model, Pipeline) or isinstance(model, ImbPipeline):
                    pipe = model
                else:
                    pipe = Pipeline([("pre", pre), ("clf", model)])

            est = clone(pipe)
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)

            # scores/probabilities
            score_u = get_minority_scores(est, X_test, MINORITY_CLASS)
            proba_mat = get_proba_matrix(est, X_test, n_classes=3)

            # metrics per fold
            m = compute_metrics(y_test, y_pred, score_u)
            for k, v in m.items():
                metrics_store[model_name][k].append(v)

            fold_metrics_rows[model_name].append({
                "repeat": repeat_id,
                "fold": fold_id,
                **m,
            })

            # raw predictions (instance × fold × repeat)
            n = len(y_test)

            df_raw = pd.DataFrame({
                "instance_id": np.asarray(test_idx).ravel(),
                "repeat": np.full(n, repeat_id),
                "fold": np.full(n, fold_id),
                "y_true": np.asarray(y_test).ravel(),
                "y_pred": np.asarray(y_pred).ravel(),
                "score_HIGH": np.asarray(score_u).ravel(),
            })

            if proba_mat is not None and proba_mat.shape[1] >= 3:
                df_raw["proba_0"] = proba_mat[:, 0]
                df_raw["proba_1"] = proba_mat[:, 1]
                df_raw["proba_2"] = proba_mat[:, 2]
            else:
                df_raw["proba_0"] = np.nan
                df_raw["proba_1"] = np.nan
                df_raw["proba_2"] = np.nan

            raw_preds_by_method[model_name].append(df_raw)

        if (fold_i + 1) % max(1, verbose_every) == 0:
            print(f"Fold {fold_i + 1}/{total_folds} ...")

    # Save CSVs by method
    oof_scores_aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for method, parts in raw_preds_by_method.items():
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        raw_df = pd.concat(parts, ignore_index=True).sort_values(["instance_id", "repeat", "fold"])
        raw_path = method_dir / "oof_predictions_raw.csv"
        raw_df.to_csv(raw_path, index=False)

        # fold metrics
        fm_df = pd.DataFrame(fold_metrics_rows[method])
        fm_path = method_dir / "fold_metrics.csv"
        fm_df.to_csv(fm_path, index=False)

        # Aggregated OOF per instance (average probabilities across repeats)
        # - if probabilities are missing, aggregate by score_HIGH
        agg = raw_df.groupby("instance_id", as_index=False).agg(
            y_true=("y_true", "first"),
            proba_0=("proba_0", "mean"),
            proba_1=("proba_1", "mean"),
            proba_2=("proba_2", "mean"),
            score_HIGH=("score_HIGH", "mean"),
        )
        # Final prediction = argmax of probabilities if available; otherwise, threshold score_HIGH
        if agg[["proba_0", "proba_1", "proba_2"]].notna().all(axis=None):
            probs = agg[["proba_0", "proba_1", "proba_2"]].to_numpy()
            agg["y_pred"] = np.argmax(probs, axis=1)
        else:
            agg["y_pred"] = (agg["score_HIGH"] >= 0.5).astype(int)  # fallback binary (unlikely)

        agg_path = method_dir / "oof_predictions_aggregated.csv"
        agg.to_csv(agg_path, index=False)

        # For PR(HIGH): use aggregated score_HIGH
        y_true_bin = (agg["y_true"].to_numpy() == MINORITY_CLASS).astype(int)
        y_score = agg["score_HIGH"].to_numpy(dtype=float)
        oof_scores_aggregated[method] = {"y_true_bin": y_true_bin, "y_score": y_score}

    # Global summary mean + CI
    rows: List[ResultRow] = []
    for method, md in metrics_store.items():
        for metric, values in md.items():
            mean, lo, hi = bootstrap_ci(np.array(values), n_boot=2000, ci=0.95, seed=seed)
            rows.append(ResultRow(method, metric, mean, lo, hi))
    summary_df = pd.DataFrame([r.__dict__ for r in rows])

    # Save global summary
    summary_df.to_csv(out_dir / "summary_mean_ci95.csv", index=False)

    return summary_df, oof_scores_aggregated


# -------------------------
# Table + PR + AutoML scatter
# -------------------------
def _family_of_method(method: str) -> str:
    m = method.lower()
    if "ensemble" in m:
        return "Ensemble"
    if "catboost" in m:
        return "CatBoost"
    if "xgboost" in m:
        return "Xgboost"
    if "lightgbm" in m or "lgbm" in m:
        return "LightGBM"
    if "balancedrf" in m or "randomforest" in m:
        return "Random Forest"
    if "decisiontree" in m:
        return "Decision Tree"
    if "softmax" in m or "linear" in m or "lr" in m:
        return "Linear"
    return "Other"


def plot_table_and_pr_and_automl(
    summary_df: pd.DataFrame,
    oof_scores: Dict[str, Dict[str, np.ndarray]],
    out_png: Path,
    out_svg: Path,
    title: str,
    methods_order: List[str],
):
    show_metrics = [
        "macro_f1",
        "balanced_accuracy",
        "precision_HIGH",
        "recall_HIGH",
        "f1_HIGH",
        "auprc_HIGH",
        "weighted_f1",
    ]
    metric_names = {
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Balanced Acc.",
        "precision_HIGH": "Prec(HIGH=2)",
        "recall_HIGH": "Rec(HIGH=2)",
        "f1_HIGH": "F1(HIGH=2)",
        "auprc_HIGH": "AUPRC(HIGH=2)",
        "weighted_f1": "Weighted-F1",
    }

    col_labels = ["Method"] + [metric_names[m] for m in show_metrics]
    cell_text = []
    for method in methods_order:
        row = [method]
        sub = summary_df[(summary_df["method"] == method)]
        for met in show_metrics:
            r = sub[sub["metric"] == met].iloc[0]
            row.append(f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        cell_text.append(row)

    fig = plt.figure(figsize=(18, 12), layout='constrained')
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.20, 1.00, 0.90], hspace=0.30)

    # --- Panel 1: Table ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title(title, fontsize=12, pad=10)
    table = ax0.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for c in range(len(col_labels)):
        table[(0, c)].set_text_props(weight="bold")

    # --- Panel 2: PR curves (Top 5 by macro_f1) ---
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    top5_methods = (
        summary_df[summary_df["metric"] == "macro_f1"]
        .sort_values("mean", ascending=False)
        .head(5)["method"]
        .tolist()
    )

    for method in top5_methods:
        d = oof_scores.get(method)
        if d is None:
            continue
        y_true_bin = d["y_true_bin"]
        y_score = d["y_score"]
        if len(np.unique(y_true_bin)) < 2:
            continue

        prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
        auprc = average_precision_score(y_true_bin, y_score)
        ax1.plot(rec, prec, label=f"{method} (AUPRC={auprc:.3f})", linewidth=2)

    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_title("Precision-Recall (High) — Top 5 Models (OOF aggregated)")

    # --- Panel 3: “AutoML Performance” (ranking vs Macro-F1) ---
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_title("Performance Comparison", fontsize=12, pad=10)
    ax2.set_xlabel("Ranking")
    ax2.set_ylabel("Macro-F1 Score")

    acc_rows = summary_df[summary_df["metric"] == "macro_f1"].copy()
    acc_rows = acc_rows.set_index("method")

    xs, ys, fams = [], [], []
    for i, method in enumerate(methods_order, start=1):
        if method not in acc_rows.index:
            continue
        xs.append(i)
        ys.append(float(acc_rows.loc[method, "mean"]))
        fams.append(_family_of_method(method))

    marker_map = {
        "Decision Tree": "^",
        "Linear": "s",
        "LightGBM": "P",
        "Xgboost": "*",
        "CatBoost": "D",
        "Random Forest": "o",
        "Ensemble": "p",
        "Other": "x",
    }

    families = ["Decision Tree", "Linear", "LightGBM", "Xgboost", "CatBoost", "Random Forest", "Ensemble", "Other"]
    for fam in families:
        idx = [k for k, f in enumerate(fams) if f == fam]
        if not idx:
            continue
        ax2.scatter(
            [xs[k] for k in idx],
            [ys[k] for k in idx],
            marker=marker_map.get(fam, "o"),
            s=90,
            label=fam,
        )

    ax2.set_xlim(0, max(xs) + 1 if xs else 1)
    if ys:
        ymin = min(ys)
        ymax = max(ys)
        pad = max(0.002, (ymax - ymin) * 0.2)
        ax2.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    ax2.legend(loc="upper right", fontsize=9)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# MAIN
# -------------------------
def main():

    df = pd.read_csv(CSV_PATH)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    X = df[FEATURE_COLS].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    mask = y.notna()
    X = X.loc[mask].to_numpy()
    y = y.loc[mask].astype(int).to_numpy()

    vc = pd.Series(y).value_counts().sort_index()
    total = int(vc.sum())
    print("\nTarget Activity distribution:")
    for k, cnt in vc.items():
        name = LABEL_MAP.get(int(k), str(k))
        pct = 100.0 * cnt / total
        print(f"  {k} ({name}): {cnt} ({pct:.1f}%)")

    out_dir = Path(OUT_DIR)
    summary_df, oof_scores = evaluate_with_mljar_outputs(
        X=X,
        y=y,
        seed=SEED,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        out_dir=Path(OUT_DIR),
        verbose_every=VERBOSE_EVERY,
    )

    # Order by macro_f1
    order = (
        summary_df[summary_df["metric"] == "macro_f1"]
        .sort_values("mean", ascending=False)["method"]
        .tolist()
    )
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=order, ordered=True)
    summary_df = summary_df.sort_values(["method", "metric"]).reset_index(drop=True)

    print("\nSummary (mean and 95% CI):")
    print(summary_df.to_string(index=False))

    out_png = Path(OUT_PNG)
    out_svg = Path(OUT_SVG)
    plot_table_and_pr_and_automl(
        summary_df=summary_df,
        oof_scores=oof_scores,
        out_png=out_png,
        out_svg=out_svg,
        title=f"Repeated {N_REPEATS}x{N_SPLITS} CV — Activity (0/1/2), minority=2 (High)",
        methods_order=order,
    )
    print(f"\n✅ Figure saved at: {out_png.resolve()}")
    print(f"✅ mljar-style outputs (CSVs) at: {out_dir.resolve()}")

    if not _HAS_LGBM:
        print("ℹ️ LightGBM is not installed: `pip install lightgbm` to include it.")
    if not _HAS_XGB:
        print("ℹ️ XGBoost is not installed: `pip install xgboost` to include it.")
    if not _HAS_CAT:
        print("ℹ️ CatBoost is not installed: `pip install catboost` to include it.")


if __name__ == "__main__":
    main()