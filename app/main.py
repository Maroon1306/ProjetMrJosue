"""Étape 2 : API FastAPI qui sert model.pkl et journalise les prédictions pour le monitoring."""
import csv
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.monitoring import monitoring_summary

MODEL_PATH = Path("models/model.pkl")
LOG_PATH = Path("logs/predictions.csv")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris MLOps API", version="1.0.0")
model = joblib.load(MODEL_PATH)


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


def log_prediction(row: dict, prediction: int):
    """Enregistre chaque requête : ces données servent à Evidently pour détecter la dérive."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", *FEATURES, "prediction"])
        if new_file:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), **row, "prediction": prediction})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: IrisInput):
    row = data.model_dump()
    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    log_prediction(row, pred)
    return {
        "prediction": pred,
        "class_name": CLASSES[pred],
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
    }


@app.get("/monitoring")
def monitoring():
    """Indicateurs de monitoring en temps réel (trafic, prédictions, dérive des données)."""
    return monitoring_summary(LOG_PATH)


@app.get("/", response_class=HTMLResponse)
def home():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
