# xgboost_shap_config.py
"""
Configuración específica para los gráficos SHAP con XGBoost
Mensaje del compañero: "Chavales, solo una cosilla, el modelo utilizado para generar los gráficos SHAP per class están hechos sobre el XGBoost"
"""

from pathlib import Path

# Configuración exacta del mensaje
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

# Nota para el equipo:
"""
Este archivo contiene la configuración exacta usada para generar
los gráficos SHAP por clase con XGBoost.

Uso:
from xgboost_config import DATA_CSV, TARGET_COL, XGB_PARAMS
"""