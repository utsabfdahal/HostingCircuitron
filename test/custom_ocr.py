"""
Custom CRNN OCR service — fast, lightweight text recognition.

Uses a CRNN (VGG + BiLSTM + CTC) model trained on circuit-diagram
text labels.  Much faster than TrOCR as it avoids the transformer
encoder-decoder overhead.

Singleton access via :func:`get_custom_ocr_service`.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torchvision import transforms

# ── Character set ────────────────────────────────────────────────────────────

CHARS = list(
    '!"#$%&'
    "()*+,-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ^_`abcdefghijklmnopqrstuvwxyz~§µ×ßäöüΩ"
)

char2idx = {c: i for i, c in enumerate(CHARS)}
idx2char = {i: c for c, i in char2idx.items()}

# ── Model architecture ──────────────────────────────────────────────────────

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            self.rnn.flatten_parameters()
        except Exception:
            pass
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel: int, output_channel: int = 256):
        super().__init__()
        oc = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, oc[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[0], oc[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[1], oc[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(oc[2], oc[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[2], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.Conv2d(oc[3], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[3], oc[3], 2, 1, 0), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class CRNN(nn.Module):
    def __init__(
        self,
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 256,
        num_class: int = len(CHARS) + 1,
    ):
        super().__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        visual_feature = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction.permute(1, 0, 2)


# ── Greedy CTC decoder ──────────────────────────────────────────────────────

def _greedy_decode(output: torch.Tensor) -> List[str]:
    """CTC greedy decoding. output shape: (T, B, C)."""
    probs = output.softmax(2)
    max_indices = probs.argmax(2).permute(1, 0)  # (B, T)
    decoded = []
    for indices in max_indices:
        s = ""
        prev = 0
        for i in indices:
            idx = i.item()
            if idx != prev and idx != 0:
                s += idx2char[idx]
            prev = idx
        decoded.append(s)
    return decoded


def _greedy_decode_with_confidence(output: torch.Tensor) -> List[tuple]:
    """CTC greedy decoding returning (text, confidence) tuples."""
    probs = output.softmax(2)
    max_probs, max_indices = probs.max(2)  # both (T, B)
    max_indices = max_indices.permute(1, 0)  # (B, T)
    max_probs = max_probs.permute(1, 0)  # (B, T)
    results = []
    for indices, confs in zip(max_indices, max_probs):
        s = ""
        char_confs = []
        prev = 0
        for idx_t, conf_t in zip(indices, confs):
            idx = idx_t.item()
            if idx != prev and idx != 0:
                s += idx2char[idx]
                char_confs.append(conf_t.item())
            prev = idx
        confidence = float(np.mean(char_confs)) if char_confs else 0.0
        results.append((s, round(confidence, 4)))
    return results


# ── Preprocessing ────────────────────────────────────────────────────────────

_IMG_HEIGHT = 32


class _ResizeKeepAspectRatio:
    def __init__(self, height: int):
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(1, int(w * (self.height / h)))
        return img.resize((new_w, self.height), Image.LANCZOS)


_transform = transforms.Compose([
    _ResizeKeepAspectRatio(_IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR numpy crop to a preprocessed (1, 1, H, W) tensor."""
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
    tensor = _transform(pil_img)
    return tensor.unsqueeze(0)


# ── Default model path ──────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "customOCR" / "crnn_last (1).pth"
)


# ── Service class ────────────────────────────────────────────────────────────

class CustomOCRService:
    """
    Encapsulates CRNN model loading and inference.

    Drop-in replacement for OCRService with the same ``extract_texts`` API.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH, device: Optional[str] = None):
        self._model_path = model_path
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[CRNN] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        print(f"[CustomOCR] Loading CRNN model from '{self._model_path}' on {self._device} …")
        self._model = CRNN().to(self._device)
        self._model.load_state_dict(
            torch.load(self._model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()
        print("[CustomOCR] Model ready.")

    def recognise(self, pil_image: Image.Image) -> tuple:
        """Run CRNN on a single PIL image crop. Returns (text, confidence)."""
        self._ensure_loaded()
        assert self._model is not None

        gray = pil_image.convert("L")
        tensor = _transform(gray).unsqueeze(0).to(self._device)

        with torch.no_grad():
            pred = self._model(tensor)
            results = _greedy_decode_with_confidence(pred)
        return results[0]

    def extract_texts(
        self,
        image_bgr: np.ndarray,
        text_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Crop each text box from the BGR image, run CRNN OCR, and return
        enriched dicts (same interface as OCRService.extract_texts).
        """
        self._ensure_loaded()
        assert self._model is not None

        img_h, img_w = image_bgr.shape[:2]
        results: List[Dict[str, Any]] = []

        for box in text_boxes:
            bb = box["bbox"]
            x1 = max(0, int(bb[0]))
            y1 = max(0, int(bb[1]))
            x2 = min(img_w, int(bb[2]))
            y2 = min(img_h, int(bb[3]))

            if x2 <= x1 or y2 <= y1:
                enriched = dict(box)
                enriched["ocr_text"] = ""
                enriched["ocr_confidence"] = 0.0
                results.append(enriched)
                continue

            crop = image_bgr[y1:y2, x1:x2]
            tensor = _preprocess_crop(crop).to(self._device)

            with torch.no_grad():
                pred = self._model(tensor)
                decoded = _greedy_decode_with_confidence(pred)

            text, confidence = decoded[0]
            enriched = dict(box)
            enriched["ocr_text"] = text
            enriched["ocr_confidence"] = confidence
            results.append(enriched)

        return results


# ── Singleton ────────────────────────────────────────────────────────────────

_custom_ocr_instance: Optional[CustomOCRService] = None


def get_custom_ocr_service(model_path: str = _DEFAULT_MODEL_PATH) -> CustomOCRService:
    """Return (or create) the global CustomOCRService singleton."""
    global _custom_ocr_instance
    if _custom_ocr_instance is None:
        _custom_ocr_instance = CustomOCRService(model_path=model_path)
    return _custom_ocr_instance
