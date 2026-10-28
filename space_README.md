---
title: Iris MLOps API
emoji: 🌸
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: space_app.py
pinned: false
---

# Iris MLOps API

API FastAPI servant un modèle scikit-learn (RandomForest) entraîné sur le jeu de données Iris.
Déployée automatiquement depuis GitHub Actions.

- `GET /` : interface Gradio
- `POST /predict` : prédiction (API FastAPI)
- `GET /health` : état du service
- `GET /monitoring` : monitoring en temps réel (trafic, prédictions, dérive des données)
- `GET /docs` : documentation Swagger
