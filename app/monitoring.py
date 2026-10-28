"""Étape 6 (temps réel) : indicateurs calculés à la volée sur les prédictions journalisées.

La dérive est mesurée comme dans Evidently pour des petites données numériques :
test de Kolmogorov-Smirnov par variable (dérive si p-value < 0.05), et dérive du jeu
de données si au moins la moitié des variables dérivent.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES = ["setosa", "versicolor", "virginica"]
MIN_ROWS = 30
P_THRESHOLD = 0.05

REFERENCE = pd.DataFrame(load_iris().data, columns=FEATURES)


def load_logs(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp", *FEATURES, "prediction"])
    return pd.read_csv(log_path)


def drift_table(current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        p_value = ks_2samp(REFERENCE[f], current[f]).pvalue if len(current) >= MIN_ROWS else None
        rows.append({
            "variable": f,
            "moyenne_reference": round(REFERENCE[f].mean(), 2),
            "moyenne_production": round(current[f].mean(), 2) if len(current) else None,
            "p_value": round(p_value, 4) if p_value is not None else None,
            "derive": None if p_value is None else bool(p_value < P_THRESHOLD),
        })
    return pd.DataFrame(rows)


def monitoring_summary(log_path: Path) -> dict:
    logs = load_logs(log_path)
    drift = drift_table(logs)
    counts = logs["prediction"].value_counts().to_dict() if len(logs) else {}
    drifted = int(drift["derive"].eq(True).sum())
    enough = len(logs) >= MIN_ROWS
    return {
        "total_predictions": len(logs),
        "last_prediction": logs["timestamp"].iloc[-1] if len(logs) else None,
        "predictions_par_classe": {c: int(counts.get(i, 0)) for i, c in enumerate(CLASSES)},
        "assez_de_donnees": enough,
        "variables_en_derive": drifted,
        "derive_detectee": enough and drifted >= len(FEATURES) / 2,
        "derive_par_variable": drift.to_dict(orient="records"),
    }
