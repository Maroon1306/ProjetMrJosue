"""Point d'entrée du Space Hugging Face (SDK Gradio, matériel ZeroGPU).

C'est Gradio qui lance le serveur (demo.launch) : ZeroGPU ne démarre correctement que
de cette façon. On y ajoute ensuite les routes de l'API FastAPI (app/main.py) :
/predict et /health, documentées sur /docs.
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
from fastapi.routing import APIRoute

from app.main import CLASSES, IrisInput, app as api, predict

API_PATHS = {"/predict", "/health"}


@gpu
def gradio_predict(sepal_length, sepal_width, petal_length, petal_width):
    result = predict(IrisInput(
        sepal_length=sepal_length, sepal_width=sepal_width,
        petal_length=petal_length, petal_width=petal_width,
    ))
    return result["probabilities"]


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(5.1, label="Longueur sépale (cm)"),
        gr.Number(3.5, label="Largeur sépale (cm)"),
        gr.Number(1.4, label="Longueur pétale (cm)"),
        gr.Number(0.2, label="Largeur pétale (cm)"),
    ],
    outputs=gr.Label(num_top_classes=len(CLASSES), label="Espèce prédite"),
    title="Classification d'Iris (MLOps)",
    description="Modèle scikit-learn servi par FastAPI. API : POST /predict · Swagger : /docs",
)

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
