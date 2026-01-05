#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP global feature importance (multi-class) for Sadeco cockroach activity prediction.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
import shap


# -----------------------------
# User configuration
# -----------------------------
DATA_CSV = Path("dataset_sadeco.csv")   # in current working directory
TARGET_COL = "Activity"  # e.g. "Actividad"

TEST_SIZE = 0.25
RANDOM_STATE = 42

XGB_PARAMS = dict(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    min_child_weight=1.0,
    gamma=0.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
)

MAX_SHAP_ROWS = 2500     
BACKGROUND_ROWS = 300    

def infer_target_column(df: pd.DataFrame) -> str:
    label_set = {"NONE", "LOW", "HIGH"}
    for col in df.columns:
        vals = set(df[col].dropna().astype(str).unique().tolist())
        if label_set.issubset(vals) or vals.issubset(label_set):
            return col
    raise ValueError("I could not infer the target column.")

def prepare_xy(df: pd.DataFrame, target_col: str):
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])
    drop_like = [c for c in X.columns if c.lower() in {"id", "index"}]
    if drop_like:
        X = X.drop(columns=drop_like)
    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.datetime64):
            X[c] = X[c].view("int64") // 10**9
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    return X, y

def train_xgb_balanced(X_train, y_train):
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    classes = sorted(y_train.unique().tolist())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_train_int = y_train.map(class_to_int).astype(int)
    model = XGBClassifier(**XGB_PARAMS, num_class=len(classes))
    model.fit(X_train, y_train_int, sample_weight=sw)
    return model, classes

def shap_global_importance_multiclass(model, X_bg, X_explain, n_classes: int):
    feature_names = list(X_explain.columns)
    try:
        explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="tree_path_dependent", model_output="raw")
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            sv = np.stack(shap_values, axis=0)
        else:
            sv = np.array(shap_values)
            if sv.ndim == 3:
                if sv.shape[-1] > 1:
                    sv = np.transpose(sv, (2, 0, 1))
                else:
                    sv = sv[None, :, :]
            elif sv.ndim == 2:
                sv = sv[None, :, :]
        mean_abs = np.mean(np.abs(sv), axis=(0, 1))
        return mean_abs, explainer, shap_values
    except Exception as e:
        import xgboost as xgb
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_explain, feature_names=feature_names)
        contribs = np.array(booster.predict(dmat, pred_contribs=True))
        n_samples = contribs.shape[0]
        n_features = len(feature_names)
        if contribs.ndim == 3:
            sv = contribs[:, :, :n_features]
        elif contribs.ndim == 2:
            sv = contribs.reshape(n_samples, n_classes, n_features + 1)[:, :, :n_features]
        sv = np.transpose(sv, (1, 0, 2))
        mean_abs = np.mean(np.abs(sv), axis=(0, 1))
        shap_values_list = [sv[c] for c in range(min(n_classes, sv.shape[0]))]
        return mean_abs, None, shap_values_list

def plot_bar_importance(feature_names, importances, outpath: Path, top_n=25):
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    s = s.head(top_n)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(s.index, s.values)
    plt.xlabel("Mean(|SHAP value|) across samples & classes")
    plt.title("SHAP global feature importance (XGBoost_balanced, multi-class)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

def plot_bar_importance_per_class(feature_names, shap_values_list, classes, outdir: Path, top_n=25):
    outdir.mkdir(parents=True, exist_ok=True)
    for ci, cls in enumerate(classes):
        sv = np.asarray(shap_values_list[ci])
        mean_abs = np.mean(np.abs(sv), axis=0)
        s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(top_n)[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(s.index, s.values)
        plt.xlabel("Mean(|SHAP value|) for this class")
        plt.title(f"SHAP feature importance for class: {cls}")
        plt.tight_layout()
        plt.savefig(outdir / f"shap_importance_class_{cls}.png", dpi=200)
        plt.close()

def main():
    if not DATA_CSV.exists(): raise FileNotFoundError(f"Missing {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    target_col = TARGET_COL or infer_target_column(df)
    X, y = prepare_xy(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    model, classes = train_xgb_balanced(X_train, y_train)

    rng = np.random.default_rng(RANDOM_STATE)
    X_bg = X_train.iloc[rng.choice(len(X_train), size=min(BACKGROUND_ROWS, len(X_train)), replace=False)]
    X_explain = X_test.iloc[rng.choice(len(X_test), size=MAX_SHAP_ROWS, replace=False)] if len(X_test) > MAX_SHAP_ROWS else X_test

    importances, explainer, shap_values_for_plots = shap_global_importance_multiclass(model, X_bg, X_explain, n_classes=len(classes))

    # --- Global Plots ---
    plot_bar_importance(X.columns.tolist(), importances, Path("shap_global_importance_bar.png"), top_n=30)
    plot_bar_importance_per_class(X.columns.tolist(), shap_values_for_plots, classes, Path('shap_per_class'), top_n=30)

    try:
        # 1. Definimos el orden lógico que deseas (esto controla la leyenda y los colores)
        # El orden de esta lista será: Color 1 (None), Color 2 (Low), Color 3 (High)
        desired_labels = ["None", "Low", "High"]
        
        # 2. Creamos un mapa de "Nombre del CSV" -> "Nombre Elegante"
        # XGBoost suele codificar como '0', '1', '2' o mantener los nombres si son strings
        # Ajustamos este mapeo según lo que imprimió tu consola previamente
        label_to_idx = {
            "0": 0, "None": 0,
            "1": 1, "Low": 1,
            "2": 2, "High": 2
        }

        # 3. Reordenamos los valores SHAP para que coincidan con [None, Low, High]
        # Creamos una lista vacía de 3 posiciones y la rellenamos en el orden correcto
        sv_reordered = [None] * len(desired_labels)
        for i, cls_name in enumerate(classes):
            idx_destino = label_to_idx.get(str(cls_name))
            if idx_destino is not None:
                sv_reordered[idx_destino] = shap_values_for_plots[i]

        # Filtramos por si alguna clase no se encontró
        sv_reordered = [v for v in sv_reordered if v is not None]

        print(f"Graficando en orden: {desired_labels}")

        # 4. Generamos el gráfico con la lista reordenada y los nombres fijos
        plt.figure()
        shap.summary_plot(
            sv_reordered, 
            X_explain, 
            show=False, 
            class_names=desired_labels  # Ahora coinciden posición por posición
        )
        
        outdir_b = Path("shap_beeswarm_per_class_and_summary")
        outdir_b.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outdir_b / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
        plt.savefig(outdir_b / "shap_summary_beeswarm.svg", bbox_inches="tight")
        plt.close()

        # 5. Guardar beeswarms individuales con los nombres nuevos
        for i, name in enumerate(desired_labels):
            plt.figure()
            shap.summary_plot(sv_reordered[i], X_explain, show=False)
            plt.title(f"Impact for class: {name}")
            plt.tight_layout()
            plt.savefig(outdir_b / f"shap_beeswarm_class_{name}.png", dpi=200, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"WARNING: Error al reorganizar clases -> {e}")

    print(f"Target column: {target_col}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()