# Oral Disease Detection — YOLOv8n × EfficientNetB0

**FastAPI service** that runs two models in parallel on a single uploaded dental image:

- **EfficientNetB0** — 6-class image classification (calculus, caries, gingivitis, hypodontia, tooth discoloration, ulcers)
- **YOLOv8n** — 4-class object detection (caries, gingivitis, tooth discoloration, ulcers)

Returns both JSON predictions and a visual HTML page with annotated images.

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/myler71/Oral-Disease-YOLOxEfficientNet.git
cd Oral-Disease-YOLOxEfficientNet

# 2) Install deps (Python 3.10+)
pip install -r requirements.txt

# 3) Run API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser → upload an oral photo → get combined results.

---

## Project Structure

```
Oral-Disease-YOLOxEfficientNet/
├── main.py                    # FastAPI app (entry point)
├── requirements.txt
├── .gitignore
├── models/
│   ├── efficientnet_oral.keras    # EfficientNetB0 classifier (29 MB)
│   └── my_yolov8n_model.pt        # YOLOv8n detector (6 MB)
├── notebooks/
│   └── final_yolo.ipynb           # Training & experimentation notebook
└── docs/
    └── Efficientnet presentation.pdf
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Simple HTML upload form |
| `GET` | `/health` | Model load status |
| `POST` | `/predict/all` | JSON: combined classification + detection |
| `POST` | `/predict/visual` | HTML page with annotated images + results |

### Example JSON Response (`/predict/all`)

```json
{
  "task": "combined",
  "filename": "photo.jpg",
  "processing_time_ms": 342.15,
  "efficientnet": {
    "label": "caries",
    "confidence": 0.87,
    "top_k": [
      {"label": "caries", "confidence": 0.87},
      {"label": "calculus", "confidence": 0.08},
      {"label": "gingivitis", "confidence": 0.03}
    ]
  },
  "yolo": {
    "detections": [
      {"label": "Caries", "confidence": 0.91, "bbox": [120, 85, 210, 180]},
      {"label": "Gingivitis", "confidence": 0.78, "bbox": [300, 150, 380, 240]}
    ]
  }
}
```

---

## Model Details

| Model | Task | Classes | Input Size | Weights |
|-------|------|---------|------------|---------|
| EfficientNetB0 (TF/Keras) | Classification | 6 | 224×224 | `efficientnet_oral.keras` |
| YOLOv8n (Ultralytics) | Detection | 4 | 640×640 | `my_yolov8n_model.pt` |

**EfficientNet classes:** `calculus`, `caries`, `gingivitis`, `hypodontia`, `toothDiscoloration`, `ulcers`  
**YOLO classes:** `Caries`, `Gingivitus`, `ToothDiscoloration`, `Ulcer`

---

## Training Data

- **Dataset:** [salmansajid05/oral-diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) (Kaggle)
- **Pipeline (see notebook):**
  1. Ingestion via `kagglehub`
  2. Cleaning & deduplication
  3. YOLO-format conversion (images + `.txt` labels)
  4. Train/val split → class balancing via augmentation
  5. EfficientNet training (10 + 8 epochs fine-tune)
  6. YOLOv8n training on balanced set
  7. Export: `.keras` + `.pt` + `class_names.json`

---

## Configuration

Environment variables (optional):

```bash
export EFFICIENTNET_MODEL_PATH=./models/efficientnet_oral.keras
export YOLO_MODEL_PATH=./models/my_yolov8n_model.pt
```

Defaults point to `models/` directory.

---

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload

# Type checking (optional)
mypy main.py

# Lint
ruff check .
```

---

## Notes

- Models are **not** tracked by Git (see `.gitignore`). Place your trained weights in `models/`.
- For production, run behind a reverse proxy (nginx) with gunicorn workers:
  ```bash
  gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
  ```
- GPU acceleration: install `tensorflow[and-cuda]` / `torch` with CUDA for faster inference.

---

## License

MIT — free for research and commercial use. Attribution appreciated.

---

## Author

**Marwan Ammar** — Data Science & AI Trainee  
AI / ML / Computer Vision — TensorFlow, PyTorch, YOLO, AWS SageMaker  
[GitHub](https://github.com/myler71) · [LinkedIn](https://linkedin.com/in/marwanammar)