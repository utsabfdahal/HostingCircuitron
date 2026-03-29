"""
Pydantic data models for CIRCUITRON.

These schemas define the canonical JSON structure exchanged between the
FastAPI backend and the React frontend.  Every response from the
``POST /upload`` endpoint is validated against the :class:`Circuit` model
before it leaves the server.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Reusable sub-models ─────────────────────────────────────────────────────

class Position(BaseModel):
    """2-D canvas coordinate."""
    x: float
    y: float


# ── Core domain models ──────────────────────────────────────────────────────

class Component(BaseModel):
    """
    A single circuit element detected in the hand-drawn image.

    Attributes
    ----------
    id          : SPICE-style identifier, e.g. "R1", "C2", "V1".
    type        : KiCad-compatible component type, e.g. "resistor",
                  "voltage.dc", "capacitor.unpolarized".
    label       : Human-readable label (often same as *id*).
    value       : Recognised value string from OCR, e.g. "10k", "1uF".
    unit        : Engineering unit extracted from *value*, e.g. "Ω", "F".
    position    : Centre of the component bounding box on the canvas.
    rotation    : Rotation angle in degrees (0, 90, 180, 270).
    terminals   : List of terminal ids that belong to this component
                  (e.g. ["t1", "t2"]).  These ids appear in
                  :class:`Connection.from_terminal`.
    """
    id: str
    type: str
    label: str = ""
    value: str = ""
    unit: str = ""
    position: Position
    rotation: float = 0.0
    terminals: List[str] = Field(default_factory=list)


class Node(BaseModel):
    """
    An electrical node — a point where two or more wires meet.

    Derived from YOLO junction detections and/or skeleton-graph vertices.
    """
    id: str
    position: Position


class Connection(BaseModel):
    """
    A single wire segment linking a component terminal to a node.
    """
    from_component: str
    from_terminal: str
    to_node: str


class Edge(BaseModel):
    """A wire segment connecting two nodes (from skeleton graph edges)."""
    source: str    # node id, e.g. "n0"
    target: str    # node id, e.g. "n3"


class Circuit(BaseModel):
    """
    Master response model returned by ``POST /upload``.

    Contains every piece of information the React frontend needs to
    render an editable digital schematic.
    """
    circuit_id: str = Field(default_factory=lambda: str(uuid4()))
    components: List[Component] = Field(default_factory=list)
    nodes: List[Node] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)


# ── Preview / Analysis models ───────────────────────────────────────────────
# These are returned by ``POST /analyze`` so the user can review and edit
# raw ML outputs before generating the final schematic.

class BBox(BaseModel):
    """Pixel-coordinate bounding box [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedComponent(BaseModel):
    """A component detected by YOLO (before schematic conversion)."""
    id: str
    cls: int = 0
    type: str
    name: str = ""
    confidence: float = 0.0
    bbox: BBox
    position: Position
    value: str = ""
    matched_text: str = ""


class DetectedText(BaseModel):
    """A text region detected by YOLO + recognised by TrOCR."""
    id: int
    bbox: BBox
    ocr_text: str = ""
    ocr_confidence: float = 0.0


class DetectedJunction(BaseModel):
    """A junction / crossover / terminal detected by YOLO."""
    id: int
    type: str = "junction"
    bbox: BBox
    confidence: float = 0.0
    position: Position


class GraphNode(BaseModel):
    """A vertex in the skeleton adjacency graph."""
    id: int
    x: float
    y: float


class LinkedComponentRef(BaseModel):
    """Reference to a component linked to a graph edge endpoint."""
    cls: int
    name: str
    bbox: List[float] = Field(default_factory=list)


class EdgeLinkage(BaseModel):
    """Component linkage info for a graph edge's endpoints."""
    source_components: List[LinkedComponentRef] = Field(default_factory=list)
    target_components: List[LinkedComponentRef] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """An edge in the skeleton adjacency graph."""
    source: int
    target: int
    # optional list of intermediate pixel coordinates along the wire
    path: List[Position] = Field(default_factory=list)
    linked_components: Optional[EdgeLinkage] = None


class DiagnosticImages(BaseModel):
    """Visualisation images produced by the skeleton / line-detection pipeline."""
    skeleton_png: str = ""           # base64 PNG – skeletonised binary image
    overlay_png: str = ""            # base64 PNG – skeleton + detected endpoints
    bbox_png: str = ""               # base64 PNG – bounding boxes on skeleton
    adjacency_graph_png: str = ""    # base64 PNG – adjacency graph render


class AnalysisPreview(BaseModel):
    """
    Returned by ``POST /analyze``.

    Contains all raw ML outputs plus a base64-encoded annotated image
    so the user can inspect and *edit* before generating the schematic.
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    image_width: int
    image_height: int
    annotated_image: str = ""        # base64-encoded PNG with overlays
    original_image: str = ""         # base64-encoded original (for re-rendering)
    components: List[DetectedComponent] = Field(default_factory=list)
    texts: List[DetectedText] = Field(default_factory=list)
    junctions: List[DetectedJunction] = Field(default_factory=list)
    graph_nodes: List[GraphNode] = Field(default_factory=list)
    graph_edges: List[GraphEdge] = Field(default_factory=list)
    diagnostic_images: DiagnosticImages = Field(default_factory=DiagnosticImages)


class EditedAnalysis(BaseModel):
    """
    Submitted by the frontend to ``POST /finalize``.

    The user may have deleted, added, or modified any of the detected
    items.  The backend converts this into a :class:`Circuit`.
    """
    image_width: int
    image_height: int
    components: List[DetectedComponent] = Field(default_factory=list)
    texts: List[DetectedText] = Field(default_factory=list)
    junctions: List[DetectedJunction] = Field(default_factory=list)
    graph_nodes: List[GraphNode] = Field(default_factory=list)
    graph_edges: List[GraphEdge] = Field(default_factory=list)
