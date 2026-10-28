"""Point d'entrée du Space Hugging Face (SDK Gradio, matériel ZeroGPU).

C'est Gradio qui lance le serveur (demo.launch) : ZeroGPU ne démarre correctement que
de cette façon. On y ajoute ensuite les routes de l'API FastAPI (app/main.py) :
/predict, /health et /monitoring, documentées sur /docs.
"""
import os

try:
    # ZeroGPU exige au moins une fonction marquée @spaces.GPU (paquet présent sur Hugging Face)
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

import gradio as gr
import numpy as np
import pandas as pd
from fastapi.routing import APIRoute

from app.main import CLASSES, LOG_PATH, IrisInput, app as api, predict
from app.monitoring import FEATURES, MIN_ROWS, REFERENCE, drift_table, load_logs, monitoring_summary

API_PATHS = {"/predict", "/health", "/monitoring"}
REFRESH_SECONDS = 5


@gpu
def gradio_predict(sepal_length, sepal_width, petal_length, petal_width):
    result = predict(IrisInput(
        sepal_length=sepal_length, sepal_width=sepal_width,
        petal_length=petal_length, petal_width=petal_width,
    ))
    return result["probabilities"]


def simulate_traffic(n: int, drift: bool):
    """Envoie n requêtes au modèle, tirées des données d'entraînement (décalées si drift=True)."""
    rng = np.random.default_rng()
    sample = REFERENCE.sample(n, replace=True).reset_index(drop=True)
    sample += rng.normal(0, 0.1, sample.shape)
    if drift:
        sample["petal_length"] += 1.5
        sample["sepal_width"] -= 0.8
        sample["petal_width"] += 0.6
    for row in sample.clip(lower=0.1).round(2).to_dict(orient="records"):
        predict(IrisInput(**row))
    return refresh()


def reset_logs():
    LOG_PATH.unlink(missing_ok=True)
    return refresh()


def refresh():
    summary = monitoring_summary(LOG_PATH)
    logs = load_logs(LOG_PATH)

    if not summary["assez_de_donnees"]:
        status = (f"### ⏳ En attente de données\n{summary['total_predictions']} / {MIN_ROWS} prédictions "
                  "nécessaires pour calculer la dérive.")
    elif summary["derive_detectee"]:
        status = (f"### 🚨 Dérive des données détectée\n{summary['variables_en_derive']} variable(s) sur "
                  f"{len(FEATURES)} ont changé de distribution : il faut ré-entraîner le modèle (`dvc repro`).")
    else:
        status = (f"### ✅ Pas de dérive\n{summary['variables_en_derive']} variable(s) sur {len(FEATURES)} "
                  "en dérive : les données de production ressemblent aux données d'entraînement.")

    kpis = (f"**Prédictions reçues :** {summary['total_predictions']}  \n"
            f"**Dernière prédiction :** {summary['last_prediction'] or '-'}  \n"
            f"**Actualisation automatique :** toutes les {REFRESH_SECONDS} s")

    classes = pd.DataFrame({"espece": CLASSES, "nombre": list(summary["predictions_par_classe"].values())})

    recent = logs.tail(200).reset_index(drop=True)
    recent["requete"] = recent.index + 1
    timeline = recent.melt(id_vars="requete", value_vars=FEATURES, var_name="variable", value_name="valeur")

    return status, kpis, classes, drift_table(logs), timeline


with gr.Blocks(title="Iris MLOps") as demo:
    gr.Markdown("# Classification d'Iris (MLOps)\nModèle scikit-learn servi par FastAPI · "
                "API : `POST /predict` · `GET /monitoring` · Swagger : `/docs`")

    with gr.Tab("Prédiction"):
        gr.Interface(
            fn=gradio_predict,
            inputs=[
                gr.Number(5.1, label="Longueur sépale (cm)"),
                gr.Number(3.5, label="Largeur sépale (cm)"),
                gr.Number(1.4, label="Longueur pétale (cm)"),
                gr.Number(0.2, label="Largeur pétale (cm)"),
            ],
            outputs=gr.Label(num_top_classes=len(CLASSES), label="Espèce prédite"),
            flagging_mode="never",
        )

    with gr.Tab("Monitoring temps réel"):
        with gr.Row():
            status = gr.Markdown()
            kpis = gr.Markdown()
        with gr.Row():
            btn_normal = gr.Button("Simuler 50 requêtes normales")
            btn_drift = gr.Button("Simuler 50 requêtes avec dérive", variant="stop")
            btn_reset = gr.Button("Réinitialiser")
        with gr.Row():
            classes_plot = gr.BarPlot(x="espece", y="nombre", title="Prédictions par espèce")
            timeline_plot = gr.LinePlot(x="requete", y="valeur", color="variable",
                                        title="Valeurs reçues (200 dernières requêtes)")
        drift_df = gr.Dataframe(label="Dérive par variable (test de Kolmogorov-Smirnov, seuil p < 0.05)")

        outputs = [status, kpis, classes_plot, drift_df, timeline_plot]
        btn_normal.click(lambda: simulate_traffic(50, drift=False), outputs=outputs)
        btn_drift.click(lambda: simulate_traffic(50, drift=True), outputs=outputs)
        btn_reset.click(reset_logs, outputs=outputs)
        gr.Timer(REFRESH_SECONDS).tick(refresh, outputs=outputs)
        demo.load(refresh, outputs=outputs)

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        ssr_mode=False,  # sinon Gradio lance un serveur Node.js en plus sur Hugging Face
        app_kwargs={"docs_url": "/docs"},
        prevent_thread_lock=True,
    )
    # Routes FastAPI placées en tête pour qu'elles passent avant celles de Gradio
    demo.app.router.routes[0:0] = [r for r in api.routes if isinstance(r, APIRoute) and r.path in API_PATHS]
    demo.block_thread()
