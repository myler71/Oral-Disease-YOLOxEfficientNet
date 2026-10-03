import json
import time
from pathlib import Path

import streamlit as st
import tensorflow as tf
from fastapi import HTTPException
from ultralytics import YOLO

from main import (
    EFFICIENTNET_MODEL_PATH,
    YOLO_MODEL_PATH,
    create_classification_image,
    create_yolo_image,
    open_image_from_bytes,
    run_classification,
    run_detection,
)

st.set_page_config(page_title="Oral Disease Analyzer", page_icon="🦷", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def load_models() -> dict:
    errors = {"classifier": None, "detector": None}

    try:
        classifier = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
    except Exception as e:
        classifier = None
        errors["classifier"] = str(e)

    try:
        detector = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        detector = None
        errors["detector"] = str(e)

    return {"classifier": classifier, "detector": detector, "errors": errors}


models = load_models()

st.sidebar.header("Models")
for key, name in (
    ("classifier", "EfficientNetB0 (6-class classifier)"),
    ("detector", "YOLOv8n (4-class detector)"),
):
    if models[key] is not None:
        st.sidebar.write(f"✅ {name}")
    else:
        st.sidebar.write(f"❌ {name} — {models['errors'][key]}")
st.sidebar.caption(f"Weights: `{EFFICIENTNET_MODEL_PATH}`, `{YOLO_MODEL_PATH}`")
st.sidebar.caption("Research demo — not a medical diagnosis.")

st.title("🦷 Oral Disease Analyzer")
st.caption(
    "EfficientNetB0 classification (calculus, caries, gingivitis, hypodontia, "
    "tooth discoloration, ulcers) + YOLOv8n detection (caries, gingivitis, "
    "tooth discoloration, ulcers)"
)

failed = [
    f"{key}: {err}" for key, err in models["errors"].items() if models[key] is None
]
if failed:
    st.error("Model failed to load: " + "; ".join(failed))
    st.stop()

uploaded = st.file_uploader(
    "Upload an oral photo", type=["jpg", "jpeg", "png", "bmp", "webp"]
)
if uploaded is None:
    st.info("Upload an image to run both models.")
    st.stop()

image_bytes = uploaded.getvalue()
try:
    open_image_from_bytes(image_bytes)
except HTTPException:
    st.error("Invalid image file")
    st.stop()

with st.spinner("Analyzing…"):
    start = time.perf_counter()
    cls = run_classification(models["classifier"], image_bytes)
    detections, yolo_result = run_detection(models["detector"], image_bytes)
    yolo_img = create_yolo_image(yolo_result)
    cls_img = create_classification_image(image_bytes, cls)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

result = {
    "task": "combined",
    "filename": uploaded.name,
    "processing_time_ms": elapsed_ms,
    "efficientnet": cls,
    "yolo": {"detections": detections},
}

m1, m2, m3 = st.columns(3)
m1.metric("Classification", cls["label"], f"{cls['confidence']:.2%}", delta_color="off")
m2.metric("Detections", len(detections))
m3.metric("Processing time", f"{elapsed_ms} ms")

left, right = st.columns(2)

with left:
    st.subheader("YOLOv8n detection")
    st.image(yolo_img, width="stretch")
    if detections:
        st.dataframe(
            [
                {
                    "Label": d["label"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "BBox (x1,y1,x2,y2)": str(d["bbox"]),
                }
                for d in detections
            ],
            hide_index=True,
        )
    else:
        st.info("No detections found")

with right:
    st.subheader("EfficientNetB0 classification")
    st.image(cls_img, width="stretch")
    for item in cls["top_k"]:
        st.progress(
            float(item["confidence"]),
            text=f"{item['label']} — {item['confidence']:.2%}",
        )

with st.expander("Raw JSON"):
    st.json(result)

st.download_button(
    "Download JSON",
    json.dumps(result, indent=2),
    file_name=f"{Path(uploaded.name).stem}_analysis.json",
    mime="application/json",
)
