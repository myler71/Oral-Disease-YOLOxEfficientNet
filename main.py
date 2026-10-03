import io
import os
import time
import base64
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw, UnidentifiedImageError
from ultralytics import YOLO

# Fallback for CPU environments where torchvision::nms C++ operator is unavailable
try:
    import torch
    import torchvision.ops
    torchvision.ops.nms(torch.zeros((1, 4)), torch.ones((1,)), 0.5)
except Exception:
    try:
        from ultralytics.utils.nms import TorchNMS
        import torchvision.ops
        torchvision.ops.nms = TorchNMS.nms
    except Exception:
        pass
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from pathlib import Path

CLASS_NAMES = [
    "calculus",
    "caries",
    "gingivitis",
    "hypodontia",
    "toothDiscoloration",
    "ulcers",
]

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

EFFICIENTNET_MODEL_PATH = os.getenv(
    "EFFICIENTNET_MODEL_PATH",
    str(MODELS_DIR / "efficientnet_oral.keras")
)

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(MODELS_DIR / "my_yolov8n_model.pt")
)

def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")


def pil_to_base64(img: Image.Image, fmt="JPEG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_efficientnet(image_bytes: bytes):
    img = open_image_from_bytes(image_bytes)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)


def run_classification(model, image_bytes: bytes):
    x = preprocess_efficientnet(image_bytes)
    preds = model.predict(x, verbose=0)[0]

    top_idx = np.argsort(preds)[::-1]
    best_idx = int(top_idx[0])

    top_k = []
    for idx in top_idx[:3]:
        top_k.append({
            "label": CLASS_NAMES[int(idx)],
            "confidence": float(preds[int(idx)])
        })

    return {
        "label": CLASS_NAMES[best_idx],
        "confidence": float(preds[best_idx]),
        "top_k": top_k
    }


def create_classification_image(image_bytes: bytes, classification_result: dict) -> Image.Image:
    img = open_image_from_bytes(image_bytes).copy()
    draw = ImageDraw.Draw(img)

    text_1 = f"EfficientNet: {classification_result['label']}"
    text_2 = f"Confidence: {classification_result['confidence']:.2%}"

    draw.rectangle([(0, 0), (img.width, 70)], fill=(0, 0, 0))
    draw.text((15, 10), text_1, fill=(255, 255, 255))
    draw.text((15, 40), text_2, fill=(255, 255, 255))

    return img


def run_detection(model, image_bytes: bytes):
    image = open_image_from_bytes(image_bytes)
    results = model.predict(image, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(float(x)) for x in box.xyxy[0].tolist()]

        detections.append({
            "label": result.names[class_id],
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })

    return detections, result


def create_yolo_image(result) -> Image.Image:
    annotated = result.plot()
    return Image.fromarray(annotated[:, :, ::-1])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.classifier_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
        print("✅ EfficientNet loaded")
    except Exception as e:
        print(f"❌ Failed to load EfficientNet: {e}")
        app.state.classifier_model = None

    try:
        app.state.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        app.state.yolo_model = None

    yield


app = FastAPI(title="Oral AI API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body style="font-family: Arial; text-align: center; padding: 40px;">
            <h1>Oral AI Analyzer</h1>
            <form action="/predict/visual" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required><br><br>
                <button type="submit">Analyze</button>
            </form>
        </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": app.state.classifier_model is not None,
        "detector_loaded": app.state.yolo_model is not None
    }


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    if app.state.classifier_model is None:
        raise HTTPException(status_code=500, detail="EfficientNet model not loaded")
    if app.state.yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    start = time.time()

    classification_result = run_classification(app.state.classifier_model, image_bytes)
    detections, _ = run_detection(app.state.yolo_model, image_bytes)

    return {
        "task": "combined",
        "filename": file.filename,
        "processing_time_ms": round((time.time() - start) * 1000, 2),
        "efficientnet": classification_result,
        "yolo": {
            "detections": detections
        }
    }


@app.post("/predict/visual", response_class=HTMLResponse)
async def predict_visual(file: UploadFile = File(...)):
    if app.state.classifier_model is None:
        raise HTTPException(status_code=500, detail="EfficientNet model not loaded")
    if app.state.yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    start = time.time()

    classification_result = run_classification(app.state.classifier_model, image_bytes)
    classification_image = create_classification_image(image_bytes, classification_result)

    detections, yolo_result = run_detection(app.state.yolo_model, image_bytes)
    yolo_image = create_yolo_image(yolo_result)

    classification_b64 = pil_to_base64(classification_image)
    yolo_b64 = pil_to_base64(yolo_image)

    processing_time_ms = round((time.time() - start) * 1000, 2)

    detection_html = "".join(
        f"<li>{d['label']} - {d['confidence']:.2%} - bbox: {d['bbox']}</li>"
        for d in detections
    ) or "<li>No detections found</li>"

    top_k_html = "".join(
        f"<li>{item['label']} - {item['confidence']:.2%}</li>"
        for item in classification_result["top_k"]
    )

    return f"""
    <html>
        <body style="font-family: Arial; padding: 30px;">
            <h1 style="text-align:center;">Analysis Result</h1>
            <p style="text-align:center;"><b>File:</b> {file.filename}</p>
            <p style="text-align:center;"><b>Processing time:</b> {processing_time_ms} ms</p>

            <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center;">
                <div style="width:45%; min-width:350px; text-align:center;">
                    <h2>YOLO Detection Result</h2>
                    <img src="data:image/jpeg;base64,{yolo_b64}" style="max-width:100%;">
                    <ul style="text-align:left; display:inline-block;">{detection_html}</ul>
                </div>

                <div style="width:45%; min-width:350px; text-align:center;">
                    <h2>EfficientNet Classification Result</h2>
                    <img src="data:image/jpeg;base64,{classification_b64}" style="max-width:100%;">
                    <h3>{classification_result['label']}</h3>
                    <p>{classification_result['confidence']:.2%}</p>
                    <ul style="text-align:left; display:inline-block;">{top_k_html}</ul>
                </div>
            </div>

            <div style="text-align:center; margin-top:20px;">
                <a href="/">Upload another image</a>
            </div>
        </body>
    </html>
    """