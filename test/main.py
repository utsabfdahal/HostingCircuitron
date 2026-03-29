"""
CIRCUITRON – FastAPI application entry-point.

Run with:
    uvicorn test.main:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
POST /upload    One-shot: image → Circuit JSON.
POST /analyze   Step 1: image → raw ML preview (for user review/editing).
POST /re-analyze  Re-run line detection with new threshold (no YOLO/OCR).
POST /finalize  Step 2: user-edited detections → Circuit JSON.
"""

from __future__ import annotations

import json
import copy
import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .schemas import Circuit, AnalysisPreview, EditedAnalysis
from .unified_pipeline import process_circuit_image, preview_analysis, finalize_from_edits
from .chat_service import get_chat_response

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _debug_dump(label: str, data: dict) -> None:
    """Pretty-print response JSON to terminal (base64 images truncated)."""
    def _truncate(obj):
        if isinstance(obj, dict):
            return {k: _truncate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_truncate(v) for v in obj]
        if isinstance(obj, str) and len(obj) > 200:
            return obj[:80] + f"...({len(obj)} chars total)"
        return obj

    cleaned = _truncate(copy.deepcopy(data))
    print(f"\n{'='*60}")
    print(f"DEBUG [{label}] response:")
    print(f"{'='*60}")
    print(json.dumps(cleaned, indent=2, default=str))
    print(f"{'='*60}\n")

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CIRCUITRON",
    description="Convert hand-drawn circuit diagrams into editable digital schematics.",
    version="0.1.0",
)

# Allow the React frontend to call the API.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=Circuit)
async def upload_circuit_image(
    file: UploadFile = File(...),
    ocr_mode: str = Form("fast"),
):
    """
    Accept an image upload, run the full ML pipeline, and return a
    :class:`Circuit` JSON payload.

    The image is read entirely into memory, handed to the orchestrator in
    ``unified_pipeline.process_circuit_image``, and the resulting dict is
    validated against the ``Circuit`` Pydantic model before being returned.
    """
    # ── validate content type ────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    # ── read bytes ───────────────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── run pipeline ─────────────────────────────────────────────────────
    try:
        result: dict = process_circuit_image(image_bytes, ocr_mode=ocr_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}",
        )

    # ── validate & return ────────────────────────────────────────────────
    circuit = Circuit(**result)
    _debug_dump("/upload", circuit.model_dump())
    return circuit


@app.post("/analyze", response_model=AnalysisPreview)
async def analyze_circuit_image(
    file: UploadFile = File(...),
    binary_thresh: int = Form(110),
    ocr_mode: str = Form("fast"),
):
    """
    **Step 1** of the two-step workflow.

    Run the full ML pipeline on an uploaded image and return the raw
    detection results — components, text, junctions, graph — plus a
    base64-encoded annotated image for visual review.

    The frontend should display these results and allow the user to
    add, remove, or edit any detection before calling ``POST /finalize``.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result: dict = preview_analysis(
            image_bytes,
            line_detection_params={"binary_thresh": binary_thresh},
            ocr_mode=ocr_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {exc}",
        )

    # Cache image + label text for fast re-analysis with different thresholds
    label_text = result.pop("_label_text", "")
    session_id = result.get("session_id", "")
    if session_id and label_text:
        _session_cache[session_id] = (image_bytes, label_text)

    preview = AnalysisPreview(**result)
    _debug_dump("/analyze", preview.model_dump())
    return preview


# ── Session cache for re-analysis ────────────────────────────────────────────
# Stores the image bytes and YOLO label text from the last /analyze call
# so /re-analyze can re-run only the cheap line-detection pipeline.
_session_cache: dict[str, tuple[bytes, str]] = {}


class ReAnalyzeRequest(BaseModel):
    session_id: str
    binary_thresh: int = 110


@app.post("/re-analyze")
async def re_analyze(req: ReAnalyzeRequest):
    """
    Re-run only the line-detection (skeleton + graph) pipeline with
    a new binary threshold.  Skips YOLO and OCR entirely — fast.
    """
    cached = _session_cache.get(req.session_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload.")

    image_bytes, label_text = cached

    from .pipeline import analyze as line_detection_analyze

    try:
        result = line_detection_analyze(
            image_bytes,
            label_text,
            params={"binary_thresh": req.binary_thresh},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Re-analysis error: {exc}")

    resp = {
        "graph": result.get("graph", {}),
        "images": result.get("images", {}),
        "detections": result.get("detections", []),
        "results": result.get("results", []),
    }
    _debug_dump("/re-analyze", resp)
    return resp


@app.post("/finalize", response_model=Circuit)
async def finalize_circuit(edited: EditedAnalysis):
    """
    **Step 2** of the two-step workflow.

    Accept the (possibly user-edited) detection data from the review
    step and produce the final :class:`Circuit` schematic JSON.
    """
    try:
        result: dict = finalize_from_edits(edited.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Finalize error: {exc}",
        )

    circuit = Circuit(**result)
    _debug_dump("/finalize", circuit.model_dump())
    return circuit


@app.post("/export-cjs")
async def export_cjs(circuit: Circuit):
    """
    Convert a :class:`Circuit` JSON to CircuitJS1 text format.

    Returns ``{"cjs_text": "..."}`` ready to be URL-encoded and
    passed to ``circuitjs.html?startCircuitText=``.
    """
    from .unified_pipeline import circuit_to_cjs

    try:
        text = circuit_to_cjs(circuit.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CJS export error: {exc}",
        )
    return {"cjs_text": text}


# ── AI Chat ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    circuit: Circuit
    message: str
    history: Optional[List[ChatMessage]] = None
    cjs_text: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a user message (with circuit context) to Lightning AI and
    return the assistant's reply.
    """
    history_dicts = (
        [m.model_dump() for m in req.history] if req.history else None
    )
    try:
        reply = await get_chat_response(
            circuit=req.circuit,
            user_message=req.message,
            history=history_dicts,
            cjs_text=req.cjs_text,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI service error: {exc}",
        )
    return {"reply": reply}
