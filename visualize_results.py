#!/usr/bin/env python3
"""Visualize YOLO + TrOCR results overlaid on the original image."""

import cv2
import numpy as np
from test.yolo_detector import detect_parsed
from test.ocr_service import OCRService

img_path = "/Users/mac/cktonutssab/finalfordemedited.JPG"
img_bgr = cv2.imread(img_path)
with open(img_path, "rb") as f:
    image_bytes = f.read()

detections, img_w, img_h = detect_parsed(image_bytes, _decoded_bgr=img_bgr)
text_dets = [d for d in detections if d["name"] == "text"]

ocr = OCRService(
    model_id="/Users/mac/cktonutssab/OCRmodel/trocrfinetuned/checkpoint-epoch-2"
)
ocr_results = ocr.extract_texts(img_bgr, text_dets)

# Build OCR lookup
ocr_map = {}
for det, res in zip(text_dets, ocr_results):
    ocr_map[tuple(det["bbox"])] = res.get("ocr_text", "")

# Color map per class
COLORS = {
    "resistor": (0, 255, 0),
    "capacitor": (255, 165, 0),
    "inductor": (255, 0, 255),
    "diode": (0, 255, 255),
    "voltage": (0, 200, 255),
    "gnd": (128, 128, 128),
    "junction": (255, 255, 0),
    "text": (0, 140, 255),
    "transistor": (255, 0, 128),
    "switch": (200, 200, 0),
    "terminal": (100, 255, 100),
}

vis = img_bgr.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 3

for d in detections:
    name = d["name"]
    bbox = d["bbox"]
    x1, y1, x2, y2 = bbox
    color = COLORS.get(name, (255, 255, 255))

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

    if name == "text":
        ocr_text = ocr_map.get(tuple(bbox), "")
        label = ocr_text if ocr_text else "text"
    else:
        label = name

    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(vis, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 8), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(vis, label, (x1 + 3, y1 - 8), font, font_scale, (255, 255, 255), thickness - 1)

out_path = "/Users/mac/cktonutssab/detection_result.jpg"
cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved visualization to: {out_path}")
