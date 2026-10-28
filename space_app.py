"""Point d'entrée du Space Hugging Face (SDK Gradio).

L'API FastAPI (app/main.py) reste le cœur du service : on y monte une interface Gradio
sur /gradio, et on sert le tout avec uvicorn sur le port 7860 attendu par Hugging Face.
"""
import gradio as gr
import uvicorn

from app.main import CLASSES, IrisInput, app, predict


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

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
