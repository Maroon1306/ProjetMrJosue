"""Étape 1 : entraînement scikit-learn + suivi des expériences avec MLflow -> models/model.pkl."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    params = yaml.safe_load(open("params.yaml"))
    data_p, train_p = params["data"], params["train"]

    df = pd.read_csv(data_p["path"])
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_p["test_size"], random_state=data_p["random_state"], stratify=y
    )

    mlflow.set_experiment("iris-classification")
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=train_p["n_estimators"],
            max_depth=train_p["max_depth"],
            random_state=data_p["random_state"],
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

        mlflow.log_params({"n_estimators": train_p["n_estimators"], "max_depth": train_p["max_depth"]})
        mlflow.log_metrics(metrics)
        # Modèle créé par nous : on déclare explicitement le type d'arbre comme sûr pour skops
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

    Path(train_p["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, train_p["model_path"])
    Path(train_p["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(train_p["metrics_path"], "w"), indent=2)
    print(f"Modèle enregistré : {train_p['model_path']} | {metrics}")


if __name__ == "__main__":
    main()
