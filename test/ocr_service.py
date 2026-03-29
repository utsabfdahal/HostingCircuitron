"""
OCR service — TrOCR fine-tuned model.

Uses a HuggingFace TrOCR model for text recognition on cropped
component-value regions extracted by the YOLO detector.

Singleton access via :func:`get_ocr_service`.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ── Default model identifier ────────────────────────────────────────────────
# Use the locally fine-tuned TrOCR checkpoint.
_DEFAULT_MODEL_ID = str(
    Path(__file__).resolve().parent.parent
    / "OCRmodel" / "trocrfinetuned" / "checkpoint-epoch-2"
)

# Component values are short strings (e.g. "10k", "4.7uF") — no need for 32.
_MAX_NEW_TOKENS = 16


class OCRService:
    """
    Encapsulates TrOCR model loading and inference.

    Usage::

        svc = OCRService()                 # lazy-loads on first call
        results = svc.extract_texts(bgr_image, text_boxes)
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: Optional[str] = None):
        self._model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Optional[TrOCRProcessor] = None
        self._model: Optional[VisionEncoderDecoderModel] = None

    # ── lazy loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not _HAS_TRANSFORMERS:
            raise RuntimeError(
                "TrOCR requires the 'transformers' package. "
                "Install it or use ocr_mode='fast' (custom CRNN)."
            )
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print(f"[OCRService] Loading TrOCR model '{self._model_id}' on {self._device} …")
        self._processor = TrOCRProcessor.from_pretrained(self._model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_id)
        self._model.to(self._device)
        # Use float16 on CUDA for ~2× faster inference
        if self._device != "cpu":
            self._model.half()
        self._model.eval()
        print("[OCRService] Model ready.")

    # ── public API ───────────────────────────────────────────────────────

    def recognise(self, pil_image: Image.Image) -> tuple[str, float]:
        """
        Run TrOCR on a single PIL image crop.

        Returns (text, confidence).
        """
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        pixel_values = self._processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self._device)
        if self._device != "cpu":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=_MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode token ids → string
        text = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Approximate confidence from mean token log-probs
        if outputs.scores:
            log_probs = [
                torch.nn.functional.log_softmax(s, dim=-1) for s in outputs.scores
            ]
            token_ids = outputs.sequences[0, 1:]  # skip <bos>
            selected = [
                lp[0, tid].item()
                for lp, tid in zip(log_probs, token_ids)
                if tid != self._model.config.eos_token_id
            ]
            confidence = float(np.exp(np.mean(selected))) if selected else 0.0
        else:
            confidence = 0.0

        return text, round(confidence, 4)

    def _recognise_batch(
        self, pil_images: List[Image.Image]
    ) -> List[tuple[str, float]]:
        """
        Run TrOCR on a batch of PIL image crops in a single forward pass.

        Returns list of (text, confidence) tuples.
        """
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        pixel_values = self._processor(
            images=pil_images, return_tensors="pt", padding=True
        ).pixel_values.to(self._device)
        if self._device != "cpu":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=_MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode all sequences at once
        texts = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # Per-sample confidence from mean token log-probs
        batch_size = len(pil_images)
        confidences: List[float] = []
        if outputs.scores:
            # scores is a tuple of (num_tokens,) tensors, each (batch, vocab)
            log_probs = [
                torch.nn.functional.log_softmax(s, dim=-1) for s in outputs.scores
            ]
            eos_id = self._model.config.eos_token_id
            for b in range(batch_size):
                token_ids = outputs.sequences[b, 1:]  # skip <bos>
                selected = [
                    lp[b, tid].item()
                    for lp, tid in zip(log_probs, token_ids)
                    if tid != eos_id
                ]
                conf = float(np.exp(np.mean(selected))) if selected else 0.0
                confidences.append(round(conf, 4))
        else:
            confidences = [0.0] * batch_size

        return [(t.strip(), c) for t, c in zip(texts, confidences)]

    def extract_texts(
        self,
        image_bgr: np.ndarray,
        text_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Crop each text bounding box from the BGR image, run TrOCR, and
        return enriched dicts with ``ocr_text`` and ``ocr_confidence``.

        All valid crops are batched into a single model forward pass
        for dramatically faster inference compared to one-at-a-time.

        Parameters
        ----------
        image_bgr   : full circuit image (BGR, as decoded by cv2).
        text_boxes  : list of dicts, each containing at least ``bbox``
                      as ``[x1, y1, x2, y2]`` in pixel coords.

        Returns
        -------
        list of dicts — the input dicts augmented with ``ocr_text``
        and ``ocr_confidence`` keys.
        """
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]

        # Collect valid crops and track which indices they correspond to
        pil_crops: List[Image.Image] = []
        crop_indices: List[int] = []  # index into text_boxes
        results: List[Dict[str, Any]] = [{} for _ in text_boxes]

        for i, tb in enumerate(text_boxes):
            bbox = tb["bbox"]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                results[i] = {**tb, "ocr_text": "", "ocr_confidence": 0.0}
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crops.append(Image.fromarray(crop_rgb))
            crop_indices.append(i)

        # Batch inference on all valid crops at once
        if pil_crops:
            batch_results = self._recognise_batch(pil_crops)
            for idx, (text, conf) in zip(crop_indices, batch_results):
                results[idx] = {
                    **text_boxes[idx],
                    "ocr_text": text,
                    "ocr_confidence": conf,
                }

        # Fill any remaining empty slots (shouldn't happen, but be safe)
        for i, r in enumerate(results):
            if not r:
                results[i] = {**text_boxes[i], "ocr_text": "", "ocr_confidence": 0.0}

        return results


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[OCRService] = None


def get_ocr_service(model_id: str = _DEFAULT_MODEL_ID) -> OCRService:
    """Return (or create) the global OCRService singleton."""
    global _instance
    if _instance is None:
        _instance = OCRService(model_id=model_id)
    return _instance
