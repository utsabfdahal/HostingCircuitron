"""
YOLO detection wrapper  (ultralytics-based, 15-class model).

Loads the model once, then exposes:
  - ``detect(image_bytes, ...)``  -> YOLO-format label text string
  - ``detect_parsed(image_bytes, ...)``  -> list of structured dicts

The label text format:  ``class xc yc w h confidence``  (normalised to [0, 1]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ultralytics import YOLO

# ── Component class names (must match training labels) ──────────────────────

COMPONENT_NAMES = [
    "capacitor",              # 0
    "crossover",              # 1
    "diode",                  # 2
    "gnd",                    # 3
    "inductor",               # 4
    "integrated_circuit",     # 5
    "junction",               # 6
    "operational_amplifier",  # 7
    "resistor",               # 8
    "switch",                 # 9
    "terminal",               # 10
    "text",                   # 11
    "transistor",             # 12
    "voltage",                # 13
    "vss",                    # 14
]

# ── globals (lazy-loaded) ────────────────────────────────────────────────────
_model: YOLO | None = None

_DEFAULT_WEIGHTS = str(
    Path(__file__).resolve().parent.parent / "yolov7new" / "best.pt"
)


def _load_model(weights: str = _DEFAULT_WEIGHTS):
    """Load the ultralytics YOLO model (called once on first detect())."""
    global _model
    _model = YOLO(weights)
    print(
        f"[detector] Loaded {weights}  "
        f"({len(COMPONENT_NAMES)} classes)"
    )


# ── Public API ───────────────────────────────────────────────────────────────

def detect(
    image_bytes: bytes,
    *,
    weights: str | None = None,
    img_size: int = 640,
    conf_thres: float = 0.2,
    iou_thres: float = 0.7,
    _decoded_bgr: np.ndarray | None = None,
) -> str:
    """
    Run YOLO inference on raw image bytes.
~
    Parameters
    ----------
    _decoded_bgr : optional pre-decoded BGR image (np.ndarray).
                   If supplied, *image_bytes* is ignored for decoding,
                   avoiding a redundant cv2.imdecode.

    Returns
    -------
    str : YOLO-format label text (one detection per line):
          ``class xc yc w h confidence``
          where xc/yc/w/h are normalised to [0, 1].
    """
    global _model

    # lazy-load
    if _model is None:
        _load_model(weights or _DEFAULT_WEIGHTS)

    # decode image (skip if already provided)
    if _decoded_bgr is not None:
        im0 = _decoded_bgr
    else:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        im0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR, HWC
    if im0 is None:
        raise ValueError("Could not decode image")
    orig_h, orig_w = im0.shape[:2]

    # inference via ultralytics
    results = _model.predict(
        source=im0,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        line_width=1,
        save=False,
        verbose=False,
    )

    # format results as YOLO txt (normalised xywh + conf)
    lines: list[str] = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_int = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            xc = ((x1 + x2) / 2) / orig_w
            yc = ((y1 + y2) / 2) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            lines.append(
                f"{cls_int} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf_val:.6f}"
            )

    return "\n".join(lines)


def detect_parsed(
    image_bytes: bytes,
    *,
    weights: str | None = None,
    img_size: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    _decoded_bgr: np.ndarray | None = None,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    High-level wrapper: run detection and return structured dicts.

    Returns
    -------
    (detections, img_w, img_h)
    detections : list of dicts with keys
        cls, name, confidence, bbox [x1, y1, x2, y2] (pixel coords)
    """
    # decode image to get dimensions (reuse if already provided)
    if _decoded_bgr is not None:
        im0 = _decoded_bgr
    else:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        im0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im0 is None:
        raise ValueError("Could not decode image")
    orig_h, orig_w = im0.shape[:2]

    label_text = detect(
        image_bytes,
        weights=weights,
        img_size=img_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        _decoded_bgr=im0,
    )

    return parse_label_text(label_text, orig_w, orig_h), orig_w, orig_h


def parse_label_text(
    label_text: str, img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """Parse YOLO-format label text into a list of detection dicts."""
    results: list[dict[str, Any]] = []
    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_int = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        conf_val = float(parts[5]) if len(parts) > 5 else 1.0

        # denormalize to pixel coords
        cx = xc * img_w
        cy = yc * img_h
        w = bw * img_w
        h = bh * img_h
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))

        name = (
            COMPONENT_NAMES[cls_int] if cls_int < len(COMPONENT_NAMES)
            else f"class_{cls_int}"
        )
        results.append({
            "cls": cls_int,
            "name": name,
            "confidence": round(conf_val, 4),
            "bbox": [x1, y1, x2, y2],
        })
    return results
