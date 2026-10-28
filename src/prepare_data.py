"""Étape 0 : récupère le jeu de données Iris et l'enregistre en CSV (versionné par DVC)."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    params = yaml.safe_load(open("params.yaml"))["data"]
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURES)
    df["target"] = iris.target

    out = Path(params["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Données enregistrées : {out} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
