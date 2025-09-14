from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
from ultralytics import YOLO
from PIL import Image

# Config (reuses your .env if present)
DEVICE = os.getenv("DEVICE", "auto").lower()
WEIGHTS = os.getenv("DETECTOR_WEIGHTS", "yolo11n.pt")
SAMPLES_DIR = Path("data/car_people_samples")
OUT_DIR = Path("data/detector_out")

def resolve_device() -> str:
    if DEVICE == "cuda" or (DEVICE == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"

def load_model() -> YOLO:
    dev = resolve_device()
    model = YOLO(WEIGHTS)
    model.to(dev)
    return model

def to_json_dets(result, class_names: dict[int, str]):
    dets = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        label = class_names.get(k, str(k))
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

def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"⚠️ No images found under {SAMPLES_DIR}. Put a few photos there.")
        return

    model = load_model()
    try:
        class_names = dict(model.model.names)
    except Exception:
        class_names = {}

    all_results = []

    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        results = model(im, verbose=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        dets = to_json_dets(r, class_names)
        h, w = r.orig_shape[:2]

        # Save per-image JSON
        out_json = {
            "image": img_path.name,
            "size": {"width": int(w), "height": int(h)},
            "count": len(dets),
            "detections": dets,
            "runtime_ms": round(dt_ms, 2),
        }
        (OUT_DIR / (img_path.stem + ".json")).write_text(json.dumps(out_json, indent=2), encoding="utf-8")

        # Optional: save annotated preview
        try:
            arr = r.plot()  # numpy array with boxes
            Image.fromarray(arr[..., ::-1]).save(OUT_DIR / (img_path.stem + "_annotated.jpg"))
        except Exception:
            pass

        all_results.append(out_json)
        print(f"✅ {img_path.name}: {len(dets)} detections in {dt_ms:.1f} ms")

    # Combined summary
    (OUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n📁 Results saved to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
