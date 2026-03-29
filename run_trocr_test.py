#!/usr/bin/env python3
"""Quick test: YOLO detection + TrOCR (checkpoint-epoch-2) on a single image."""

import cv2
import numpy as np
from PIL import Image
from test.yolo_detector import detect_parsed, COMPONENT_NAMES
from test.ocr_service import OCRService

# 1. Load image
img_path = "/Users/mac/cktonutssab/finalfordemedited.JPG"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise ValueError(f"Could not read image: {img_path}")
print(f"Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

# Read as bytes for YOLO
with open(img_path, "rb") as f:
    image_bytes = f.read()

# 2. Run YOLO detection
detections, img_w, img_h = detect_parsed(image_bytes, _decoded_bgr=img_bgr)
print(f"\nYOLO detections: {len(detections)} total")
print("-" * 60)
for i, d in enumerate(detections):
    print(f"  [{i}] {d['name']:25s}  conf={d['confidence']:.3f}  bbox={d['bbox']}")

# 3. Filter text detections
text_dets = [d for d in detections if d["name"] == "text"]
print(f"\nText detections (for OCR): {len(text_dets)}")

# 4. Run TrOCR checkpoint-epoch-2
ocr = OCRService(
    model_id="/Users/mac/cktonutssab/OCRmodel/trocrfinetuned/checkpoint-epoch-2"
)
ocr_results = ocr.extract_texts(img_bgr, text_dets)
print("\n" + "=" * 60)
print("TrOCR (checkpoint-epoch-2) OCR Results:")
print("=" * 60)
for i, (det, res) in enumerate(zip(text_dets, ocr_results)):
    text = res.get("ocr_text", "?")
    conf = res.get("ocr_confidence", 0)
    print(f'  Text box {i}: "{text}"  (confidence: {conf:.4f})  bbox={det["bbox"]}')

# 5. Full summary
print("\n" + "=" * 60)
print("FULL DETECTION SUMMARY:")
print("=" * 60)

# Build a lookup for OCR results by bbox
ocr_lookup = {}
for res in ocr_results:
    bbox_key = tuple(res.get("bbox", []))
    ocr_lookup[bbox_key] = res

for d in detections:
    label = d["name"]
    conf = d["confidence"]
    bbox = d["bbox"]
    extra = ""
    if label == "text":
        bbox_key = tuple(bbox)
        if bbox_key in ocr_lookup:
            r = ocr_lookup[bbox_key]
            extra = f'  ->  OCR: "{r.get("ocr_text", "?")}"'
    print(f"  {label:25s}  conf={conf:.3f}  bbox={bbox}{extra}")
