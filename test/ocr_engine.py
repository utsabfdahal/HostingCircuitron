"""
OCR engine – thin compatibility shim.

Delegates all text recognition to the TrOCR fine-tuned model
via the local ``ocr_service`` module.

Any legacy code that does ``from .ocr_engine import OCREngine``
will get the real ``OCRService`` instance transparently.
"""

from __future__ import annotations

from .ocr_service import OCRService as OCREngine, get_ocr_service  # noqa: F401

__all__ = ["OCREngine", "get_ocr_service"]
