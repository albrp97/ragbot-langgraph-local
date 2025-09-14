from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from ultralytics import YOLO
from contextlib import asynccontextmanager


DEVICE_ENV = os.getenv("DEVICE", "auto").lower()  
WEIGHTS = os.getenv("DETECTOR_WEIGHTS", "yolo11n.pt")  

_model: YOLO | None = None
_model_device: str | None = None
_class_names: Dict[int, str] = {}


def _resolve_device() -> str:
    if DEVICE_ENV == "cuda" or (DEVICE_ENV == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def _load_model() -> None:
    global _model, _model_device, _class_names
    if _model is not None:
        return
    dev = _resolve_device()
    _model = YOLO(WEIGHTS)          
    _model.to(dev)                  
    _model_device = dev
    try:
        _class_names = dict(_model.model.names)
    except Exception:
        _class_names = {i: str(i) for i in range(1000)}  # safe fallback


def _to_json_dets(result) -> List[Dict[str, Any]]:
    """
    Convert a single YOLO result to JSON for *all* detected classes.
    """
    dets: List[Dict[str, Any]] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return dets

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        label = _class_names.get(k, str(k))
        dets.append({
            "class_id": int(k),
            "label": label,
            "confidence": float(c),
            "box": {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            },
        })
    return dets


# -----------------------------
# FastAPI
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield

app = FastAPI(title="Local Object Detector (YOLO11)", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    _load_model()
    return {
        "ok": True,
        "weights": WEIGHTS,
        "device": _model_device,
        "num_classes": len(_class_names),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Multipart image -> JSON detections for ALL classes.
    Defaults (conf/IoU/imgsz) left as model defaults.
    """
    _load_model()
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    t0 = time.perf_counter()
    results = _model(image, verbose=False)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    result = results[0]
    dets = _to_json_dets(result)
    h, w = result.orig_shape[:2]

    return {
        "ok": True,
        "image": {"width": int(w), "height": int(h)},
        "count": len(dets),
        "detections": dets,
        "runtime_ms": round(dt_ms, 2),
    }


if __name__ == "__main__":
    # Run directly: python services/detector/app.py
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)