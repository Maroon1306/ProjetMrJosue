"""Étape 6 : monitoring de la dérive des données avec Evidently.

Compare les données d'entraînement (référence) aux données reçues en production
(logs/predictions.csv, écrit par l'API). S'il n'y a pas encore de trafic, on simule
des données de production légèrement décalées pour illustrer la détection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_current(log_path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    if log_path.exists() and len(pd.read_csv(log_path)) >= 30:
        print(f"Données de production : {log_path}")
        return pd.read_csv(log_path)[FEATURES]
    print("Pas assez de trafic en production : simulation de données décalées")
    rng = np.random.default_rng(0)
    current = reference.sample(100, replace=True, random_state=0).reset_index(drop=True)
    current["petal_length"] += rng.normal(1.0, 0.3, len(current))  # dérive simulée
    return current


def main():
    params = yaml.safe_load(open("params.yaml"))
    reference = pd.read_csv(params["data"]["path"])[FEATURES]
    current = load_current(Path(params["monitor"]["production_log"]), reference)

    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)

    out = Path(params["monitor"]["report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    json.dump(snapshot.dict(), open(out.with_suffix(".json"), "w"), indent=2, default=str)
    print(f"Rapport de dérive : {out}")


if __name__ == "__main__":
    main()
