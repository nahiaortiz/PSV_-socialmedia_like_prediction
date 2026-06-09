from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from predict import predict


app = FastAPI(title="Image Like Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict_endpoint(
    image:     UploadFile = File(...),
    caption:   str        = Form(default=""),
    timestamp: str        = Form(default=None),   # "2024-03-15 14:30:00"
):
    # Parse timestamp — use current time if not provided
    if timestamp:
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    else:
        ts = datetime.now()

    image_bytes = await image.read()
    result = predict(image_bytes, caption, ts)

    return {
        "status":          "ok",
        "labels":          result["labels"],
        "label_probs":     result["label_probs"],
        "predicted_likes": result["predicted_likes"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}