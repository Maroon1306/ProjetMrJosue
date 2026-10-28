"""Point d'entrée du Space Hugging Face (SDK Gradio).

L'API FastAPI (app/main.py) reste le cœur du service : on y monte une interface Gradio
sur /gradio, et on sert le tout avec uvicorn sur le port attendu par Hugging Face.
"""
import os

import gradio as gr
import uvicorn

from app.main import CLASSES, IrisInput, app, predict

try:
    # Matériel ZeroGPU de Hugging Face : il exige au moins une fonction marquée @spaces.GPU
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn


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
    description="Modèle scikit-learn servi par FastAPI. API : /predict · Swagger : /docs",
)

# ssr_mode=False : sinon, sur Hugging Face, Gradio lance un serveur Node.js qui occupe le port 7860
app = gr.mount_gradio_app(app, demo, path="/gradio", ssr_mode=False)

if __name__ == "__main__":
    # Sur Hugging Face, le port et l'adresse sont fournis par les variables de Gradio
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    print(f"Démarrage sur {host}:{port}")
    uvicorn.run(app, host=host, port=port)
