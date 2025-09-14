# Part 3 — Local Object Detection Service (Cars & People)

A lightweight **Python + FastAPI** service that runs **Ultralytics YOLO11** locally to detect objects in images and returns **JSON** results. It’s **containerized** and can be called from **Postman**, `curl`, or Python.

---

## What we built (high level)

- **Pretrained model**: Ultralytics **YOLO11n** (can swap weights via env var).
- **REST API**: `/health` (GET) and `/detect` (POST multipart `file`).
- **Output**: JSON with all detections (class name, confidence, bbox).  
- **Containerized**: Dockerfiles for **GPU (CUDA)**


Sample images:
data/car_people_samples/...


## Endpoints

### `GET /health`
Health probe.
```json
{"status":"ok","model":"yolo11n.pt","device":"cuda"}
````

### `POST /detect` 

Key: `file` (the image).

Response (example shape):

```json
{
  "model": "yolo11n.pt",
  "device": "cuda",
  "image_size": [1280, 720],
  "time_ms": 45.2,
  "detections": [
    {"id": 0, "class": 0, "name": "person", "conf": 0.92, "box": {"x": 123, "y": 87, "w": 210, "h": 402}},
    {"id": 1, "class": 2, "name": "car",    "conf": 0.88, "box": {"x": 420, "y": 300, "w": 260, "h": 140}}
  ]
}
```

---

## Run locally (no Docker)

python detector/app.py
# → http://localhost:8001

Test:

curl http://localhost:8001/health
curl -X POST "http://localhost:8001/detect" -H "Content-Type: multipart/form-data" -F "file=@data/car_people_samples/119502.jpg"

---

## Run in Docker

### GPU (CUDA)

`detector/Dockerfile.gpu`:

docker build -f detector/Dockerfile.gpu -t local-detector:gpu .
docker run --rm -p 8001:8001 --gpus all -e DEVICE=cuda -e DETECTOR_WEIGHTS=yolo11n.pt local-detector:gpu

---

## Postman quick guide

1. **Method**: `POST`
   **URL**: `http://localhost:8001/detect`
2. **Body**: `form-data`
   Key = `file` (Type = File) → choose an image.
3. **Send** and inspect JSON with `detections`.

For health: `GET http://localhost:8001/health`.

---

### Steps

- Define classes and labeling rules
- Ensure diverse images, lighting, backgrounds, scales, etc
- Label the images using the models format (YOLO)
- Split data ensuring balancing
- Train pretrained model and evaluate with per class metrics

### Risks, problems and mitigation

- Imbalanced classes - Oversample, undersample, weighting, data augmentation
- Overfitting - Early stopping, more diverse data, regularization, reduce epochs
- Latency and VRAM limits - Smaller models, quatization, limit image size

### Data size needs and expected metrics

- >500 images per class
- Mean average precision (mAP@0.5 > 0.5, mAP@0.5:0.95 >0.35), precision (>0.75), recall (>0.75), f1-macro (>0.75)

### Performance improvements techniques

- Progressive fine tuning (header -> backbone)
- Hyperparameter optimization
- Cuantization, half precision
- TensorRT