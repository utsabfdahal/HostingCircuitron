"""
Unified circuit-analysis pipeline  –  CIRCUITRON orchestrator.

Exposes :func:`process_circuit_image` which is the single entry-point
called by the FastAPI ``POST /upload`` endpoint.

    Image bytes
      │
      ├─► YOLOv7 component detection  (yolo_detector.py)
      │       ├── text boxes  ──► Custom CRNN OCR (ocr_engine.py)
      │       └── component / junction boxes
      │w
      ├─► Proximity mapping  (text → nearest component)  (proximity_mapper.py)
      │
      └─► Line detection  (skeleton + Multi-Head BFS)  (pipeline.py)
              builds adjacency graph among junction / terminal endpoints
              then maps wire endpoints → component bounding boxes

All stages feed into the **Circuit** Pydantic schema
(``schemas.Circuit``) for the React frontend.
"""

from __future__ import annotations

import base64
import io
import math
import re
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .yolo_detector import detect, detect_parsed, parse_label_text, COMPONENT_NAMES
from .proximity_mapper import map_text_to_components

# ── OCR service imports ─────────────────────────────────────────────────────
# TrOCR fine-tuned model (slow/accurate) and custom CRNN (fast/lightweight).
from .ocr_service import get_ocr_service
from .custom_ocr import get_custom_ocr_service

# ── Line-detection pipeline (skeleton + adjacency graph) ────────────────────
# pipeline.py sits in the same package (test/).
from .pipeline import analyze as line_detection_analyze


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


_PREFIX_MAP = {
    "resistor": "R", "capacitor": "C", "inductor": "L",
    "diode": "D", "transistor": "Q", "voltage": "V",
    "switch": "SW", "gnd": "GND", "vss": "GND",
}


def _generate_component_id(class_name: str, index: int) -> str:
    """SPICE-style prefix + sequential number, e.g. R1, C2, V3."""
    n = class_name.lower()
    for key, prefix in _PREFIX_MAP.items():
        if key in n:
            return f"{prefix}{index + 1}"
    return f"U{index + 1}"


def _extract_unit(value: str) -> str:
    """
    Best-effort extraction of engineering unit from an OCR value string.

    Examples
    --------
    "10k"   → "Ω"
    "4.7uF" → "F"
    "100mH" → "H"
    "5V"    → "V"
    "2.2A"  → "A"
    """
    if not value:
        return ""
    v = value.strip()
    # match an optional SI prefix followed by a unit letter at the end
    m = re.search(r"[0-9.]+\s*[kKMmuμnp]?\s*([ΩΩVAFHW])\s*$", v, re.IGNORECASE)
    if m:
        u = m.group(1).upper()
        unit_map = {"V": "V", "A": "A", "F": "F", "H": "H", "W": "W"}
        return unit_map.get(u, u)
    # If value contains only digits + SI prefix (e.g. "10k", "4.7M") assume Ohms
    if re.match(r"^[0-9.]+\s*[kKMmuμnp]?$", v):
        return "Ω"
    return ""


def _estimate_rotation(bbox: List[int]) -> float:
    """
    Crude rotation estimate from bounding-box aspect ratio.
    Tall → 90°, wide → 0°.
    """
    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if h == 0:
        return 0.0
    return 90.0 if (h / max(w, 1)) > 1.5 else 0.0


def _point_inside_bbox(px: float, py: float, bbox: List[int],
                       margin: int = 10) -> bool:
    """Check if point (px, py) falls inside bbox ± margin."""
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= px <= (x2 + margin) and \
           (y1 - margin) <= py <= (y2 + margin)


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ── singleton holders (lazy init) ───────────────────────────────────────────

def _get_ocr(ocr_mode: str = "fast"):
    """Return the appropriate OCR service based on mode.
    'fast' = custom CRNN, 'slow' = TrOCR transformer."""
    if ocr_mode == "slow":
        return get_ocr_service()
    return get_custom_ocr_service()


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy helper (kept for backward compatibility with other callers)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_bytes: bytes,
    *,
    yolo_model_path: Optional[str] = None,
    proximity_max_dist: float = 250.0,
    line_detection_params: Optional[Dict[str, Any]] = None,
    ocr_mode: str = "fast",
) -> Dict[str, Any]:
    """
    Raw pipeline output — returns the *internal* dict with all
    intermediate artefacts.  Prefer :func:`process_circuit_image` for
    the frontend-compatible Circuit schema.
    """
    return _run_pipeline_raw(
        image_bytes,
        yolo_model_path=yolo_model_path,
        proximity_max_dist=proximity_max_dist,
        line_detection_params=line_detection_params,
        ocr_mode=ocr_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal raw pipeline (unchanged ML calls)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_raw(
    image_bytes: bytes,
    *,
    yolo_model_path: Optional[str] = None,
    proximity_max_dist: float = 250.0,
    line_detection_params: Optional[Dict[str, Any]] = None,
    ocr_mode: str = "fast",
) -> Dict[str, Any]:
    """
    Execute every ML stage and return an *internal* dict with all
    intermediate data (components, text_regions, junctions, graph, images).
    """

    # ── 1. Decode image ──────────────────────────────────────────────────
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image")
    img_h, img_w = int(image_bgr.shape[0]), int(image_bgr.shape[1])

    # ── 2. YOLOv7 detection ──────────────────────────────────────────────
    # TODO: Call your existing YOLO function here.
    # detect() returns normalised YOLO label text (cls xc yc w h conf).
    label_text = detect(
        image_bytes,
        weights=yolo_model_path,
        _decoded_bgr=image_bgr,
    ) if yolo_model_path else detect(image_bytes, _decoded_bgr=image_bgr)

    all_detections = parse_label_text(label_text, img_w, img_h)

    # Split detections into categories
    text_boxes = [d for d in all_detections if d["name"] == "text"]
    junction_dets = [
        d for d in all_detections
        if d["name"] in ("junction", "crossover", "terminal")
    ]
    component_dets = [
        d for d in all_detections
        if d["name"] not in ("text", "junction", "crossover", "terminal")
    ]

    # ── 3. Custom CRNN OCR on text boxes ─────────────────────────────────
    # TODO: Call your existing OCR function here.
    ocr = _get_ocr(ocr_mode)
    text_results = ocr.extract_texts(image_bgr, text_boxes) if text_boxes else []

    # ── 4. Proximity mapping (text → nearest component) ─────────────────
    # TODO: Call your existing proximity mapper here.
    components_raw: List[Dict[str, Any]] = []
    for idx, det in enumerate(component_dets):
        cid = _generate_component_id(det["name"], idx)
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        components_raw.append({
            "id": cid,
            "cls": det["cls"],
            "type": det["name"],
            "name": det["name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "position": [round(cx, 1), round(cy, 1)],
        })

    components_raw = map_text_to_components(
        components_raw, text_results, max_distance=proximity_max_dist,
    )

    # ── 5. Line detection (skeleton + Multi-Head BFS) ────────────────────
    # TODO: Call your existing line-detection / skeleton analysis here.
    # pipeline.analyze returns {graph: {nodes, edges}, results, detections, images}
    line_result = line_detection_analyze(
        image_bytes, label_text, params=line_detection_params,
    )

    # ── 6. Assemble raw internal response ────────────────────────────────
    junctions_raw = [
        {
            "id": i,
            "type": j["name"],
            "bbox": j["bbox"],
            "confidence": j["confidence"],
            "position": [
                round((j["bbox"][0] + j["bbox"][2]) / 2, 1),
                round((j["bbox"][1] + j["bbox"][3]) / 2, 1),
            ],
        }
        for i, j in enumerate(junction_dets)
    ]

    text_regions = [
        {
            "id": i,
            "bbox": tr["bbox"],
            "ocr_text": tr.get("ocr_text", ""),
            "ocr_confidence": tr.get("ocr_confidence", 0.0),
        }
        for i, tr in enumerate(text_results)
    ]

    return _sanitize({
        "image_size": {"width": img_w, "height": img_h},
        "components": components_raw,
        "text_regions": text_regions,
        "junctions": junctions_raw,
        "graph": line_result.get("graph", {}),
        "line_detection": {
            "detections": line_result.get("detections", []),
            "results": line_result.get("results", []),
        },
        "images": line_result.get("images", {}),
        "_label_text": label_text,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — Circuit-schema-compatible output
# ─────────────────────────────────────────────────────────────────────────────

def process_circuit_image(
    image_bytes: bytes,
    *,
    yolo_model_path: Optional[str] = None,
    proximity_max_dist: float = 250.0,
    line_detection_params: Optional[Dict[str, Any]] = None,
    ocr_mode: str = "fast",
) -> Dict[str, Any]:
    """
    End-to-end analysis of a hand-drawn circuit image.

    This is the single function called by ``main.py``'s ``POST /upload``
    handler.  It orchestrates every ML stage and reshapes the output into
    the dictionary expected by the **Circuit** Pydantic model::

        {
            "circuit_id": "...",
            "components": [ Component, ... ],
            "nodes": [ Node, ... ],
            "connections": [ Connection, ... ],
        }

    Steps
    -----
    1. Run the internal raw pipeline (YOLO → OCR → proximity → skeleton).
    2. Map raw components into ``Component`` dicts with terminals.
    3. Build ``Node`` list from graph vertices (junctions / wire endpoints).
    4. Map wire-endpoint graph edges to component bounding boxes to
       create ``Connection`` records linking terminals to nodes.
    """

    # ── 1. Run all ML stages ─────────────────────────────────────────────
    raw = _run_pipeline_raw(
        image_bytes,
        yolo_model_path=yolo_model_path,
        proximity_max_dist=proximity_max_dist,
        line_detection_params=line_detection_params,
        ocr_mode=ocr_mode,
    )

    raw_components: List[Dict[str, Any]] = raw.get("components", [])
    raw_junctions: List[Dict[str, Any]] = raw.get("junctions", [])
    graph: Dict[str, Any] = raw.get("graph", {})
    graph_nodes: List[Dict[str, Any]] = graph.get("nodes", [])
    graph_edges: List[Dict[str, Any]] = graph.get("edges", [])

    # ── 2. Build Node list ───────────────────────────────────────────────
    #       Each graph vertex becomes a Node.  Graph nodes have
    #       {id, x, y} from pipeline.analyze.
    nodes: List[Dict[str, Any]] = []
    for gn in graph_nodes:
        nodes.append({
            "id": f"n{gn['id']}",
            "position": {"x": float(gn["x"]), "y": float(gn["y"])},
        })

    # Also include YOLO-detected junctions that may not appear in
    # the skeleton graph (e.g. isolated junctions with no wires).
    existing_positions = {(n["position"]["x"], n["position"]["y"]) for n in nodes}
    for jd in raw_junctions:
        jx, jy = float(jd["position"][0]), float(jd["position"][1])
        # skip if a graph node is already very close
        if any(_euclidean((jx, jy), p) < 15 for p in existing_positions):
            continue
        nid = f"n{len(nodes)}"
        nodes.append({"id": nid, "position": {"x": jx, "y": jy}})
        existing_positions.add((jx, jy))

    # ── 3. Build Component list + assign terminals ───────────────────────
    #       For each component, find which graph nodes touch its bbox.
    #       Each such node becomes a terminal on that component.
    components: List[Dict[str, Any]] = []
    connections: List[Dict[str, Any]] = []

    for comp in raw_components:
        cid = comp["id"]
        comp_type = comp.get("type", comp.get("name", "unknown"))
        value_str = comp.get("value", "")
        bbox = comp["bbox"]

        # Determine which nodes fall inside (or very near) the bbox
        terminal_ids: List[str] = []
        for node in nodes:
            nx, ny = node["position"]["x"], node["position"]["y"]
            if _point_inside_bbox(nx, ny, bbox, margin=15):
                tid = f"{cid}_t{len(terminal_ids) + 1}"
                terminal_ids.append(tid)

                # Create a Connection for this terminal → node
                connections.append({
                    "from_component": cid,
                    "from_terminal": tid,
                    "to_node": node["id"],
                })

        # If no graph nodes matched, create default placeholder terminals
        # (most two-terminal components should have 2 terminals)
        if not terminal_ids:
            terminal_ids = [f"{cid}_t1", f"{cid}_t2"]

        components.append({
            "id": cid,
            "type": comp_type,
            "label": comp.get("name", cid),
            "value": value_str,
            "unit": _extract_unit(value_str),
            "position": {
                "x": float(comp["position"][0]),
                "y": float(comp["position"][1]),
            },
            "rotation": _estimate_rotation(bbox),
            "terminals": terminal_ids,
        })

    # ── 4. Build edges (node-to-node wire segments from skeleton graph) ──
    edges: List[Dict[str, Any]] = []
    node_ids = {n["id"] for n in nodes}
    for ge in graph_edges:
        src_id = f"n{ge['source']}"
        tgt_id = f"n{ge['target']}"
        # only include edges whose nodes exist in our node list
        if src_id in node_ids and tgt_id in node_ids:
            edge_entry: Dict[str, Any] = {
                "source": src_id,
                "target": tgt_id,
                "path": ge.get("path", []),
            }
            edges.append(edge_entry)

    # ── 5. Assemble final Circuit dict ───────────────────────────────────
    return _sanitize({
        "circuit_id": str(uuid4()),
        "components": components,
        "nodes": nodes,
        "connections": connections,
        "edges": edges,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  PREVIEW / ANALYSIS  (Step 1 of the two-step workflow)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_preview(image_bgr: np.ndarray, raw: Dict[str, Any]) -> np.ndarray:
    """
    Draw bounding boxes, OCR labels, junctions, and graph edges on the
    image so the user can visually verify the ML outputs.
    """
    vis = image_bgr.copy()

    # --- component boxes (green) ---
    for comp in raw.get("components", []):
        x1, y1, x2, y2 = [int(v) for v in comp["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{comp['id']} {comp.get('type', '')}"
        if comp.get("value"):
            label += f" ({comp['value']})"
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # --- text boxes (cyan) ---
    for tr in raw.get("text_regions", []):
        x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        txt = tr.get("ocr_text", "")
        if txt:
            cv2.putText(vis, txt, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- junctions (red circles) ---
    for junc in raw.get("junctions", []):
        cx, cy = int(junc["position"][0]), int(junc["position"][1])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

    # --- graph edges (magenta lines) ---
    graph = raw.get("graph", {})
    nodes_map: Dict[int, Tuple[int, int]] = {}
    for gn in graph.get("nodes", []):
        nodes_map[gn["id"]] = (int(gn["x"]), int(gn["y"]))
    for edge in graph.get("edges", []):
        src = edge.get("source", edge.get("from"))
        tgt = edge.get("target", edge.get("to"))
        if src in nodes_map and tgt in nodes_map:
            cv2.line(vis, nodes_map[src], nodes_map[tgt],
                     (255, 0, 255), 1, cv2.LINE_AA)

    # --- graph nodes (yellow dots) ---
    for nid, (nx, ny) in nodes_map.items():
        cv2.circle(vis, (nx, ny), 4, (0, 255, 255), -1)

    return vis


def _encode_image_b64(image_bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64-encoded PNG string."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def preview_analysis(
    image_bytes: bytes,
    *,
    yolo_model_path: Optional[str] = None,
    proximity_max_dist: float = 250.0,
    line_detection_params: Optional[Dict[str, Any]] = None,
    ocr_mode: str = "fast",
) -> Dict[str, Any]:
    """
    Run the ML pipeline and return *raw* detection data plus an annotated
    image so the frontend can display a reviewable preview.

    Returns a dict matching the ``AnalysisPreview`` Pydantic model.
    """
    # Decode image for annotation
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image")
    img_h, img_w = int(image_bgr.shape[0]), int(image_bgr.shape[1])

    # Run the full ML pipeline
    raw = _run_pipeline_raw(
        image_bytes,
        yolo_model_path=yolo_model_path,
        proximity_max_dist=proximity_max_dist,
        line_detection_params=line_detection_params,
        ocr_mode=ocr_mode,
    )

    # Draw annotated preview
    vis = _draw_preview(image_bgr, raw)

    # Convert raw components → DetectedComponent dicts
    det_components = []
    for comp in raw.get("components", []):
        bb = comp["bbox"]
        det_components.append({
            "id": comp["id"],
            "cls": comp.get("cls", 0),
            "type": comp.get("type", comp.get("name", "unknown")),
            "name": comp.get("name", ""),
            "confidence": round(comp.get("confidence", 0.0), 3),
            "bbox": {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]},
            "position": {"x": comp["position"][0], "y": comp["position"][1]},
            "value": comp.get("value", ""),
            "matched_text": comp.get("matched_text", comp.get("value", "")),
        })

    # Convert raw text regions → DetectedText dicts
    det_texts = []
    for tr in raw.get("text_regions", []):
        bb = tr["bbox"]
        det_texts.append({
            "id": tr["id"],
            "bbox": {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]},
            "ocr_text": tr.get("ocr_text", ""),
            "ocr_confidence": round(tr.get("ocr_confidence", 0.0), 3),
        })

    # Convert raw junctions → DetectedJunction dicts
    det_junctions = []
    for junc in raw.get("junctions", []):
        bb = junc["bbox"]
        det_junctions.append({
            "id": junc["id"],
            "type": junc.get("type", "junction"),
            "bbox": {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]},
            "confidence": round(junc.get("confidence", 0.0), 3),
            "position": {"x": junc["position"][0], "y": junc["position"][1]},
        })

    # Graph nodes & edges
    graph = raw.get("graph", {})
    graph_nodes = [
        {"id": gn["id"], "x": float(gn["x"]), "y": float(gn["y"])}
        for gn in graph.get("nodes", [])
    ]
    graph_edges = []
    for edge in graph.get("edges", []):
        edge_entry: Dict[str, Any] = {
            "source": edge.get("source", edge.get("from", 0)),
            "target": edge.get("target", edge.get("to", 0)),
            "path": edge.get("path", []),
        }
        if "linked_components" in edge:
            edge_entry["linked_components"] = edge["linked_components"]
        graph_edges.append(edge_entry)

    # Collect diagnostic images produced by the skeleton pipeline
    raw_images = raw.get("images", {})
    diagnostic_images = {
        "skeleton_png": raw_images.get("skeleton_png", "") or "",
        "overlay_png": raw_images.get("overlay_png", "") or "",
        "bbox_png": raw_images.get("bbox_png", "") or "",
        "adjacency_graph_png": raw_images.get("adjacency_graph_png", "") or "",
    }

    return _sanitize({
        "session_id": str(uuid4()),
        "image_width": img_w,
        "image_height": img_h,
        "annotated_image": _encode_image_b64(vis),
        "original_image": _encode_image_b64(image_bgr),
        "components": det_components,
        "texts": det_texts,
        "junctions": det_junctions,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "diagnostic_images": diagnostic_images,
        "_label_text": raw.get("_label_text", ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  FINALIZE  (Step 2 – convert user-edited detections → Circuit)
# ─────────────────────────────────────────────────────────────────────────────

def finalize_from_edits(edited: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take user-edited detection data (from the frontend review step) and
    produce a :class:`Circuit` dict.

    This mirrors the second half of :func:`process_circuit_image` but
    operates on the *user-corrected* data rather than raw ML output.
    """
    img_w = edited.get("image_width", 800)
    img_h = edited.get("image_height", 600)

    # ── 1. Rebuild graph nodes as Node dicts ─────────────────────────────
    nodes: List[Dict[str, Any]] = []
    for gn in edited.get("graph_nodes", []):
        nodes.append({
            "id": f"n{gn['id']}",
            "position": {"x": float(gn["x"]), "y": float(gn["y"])},
        })

    # Also include YOLO-detected junctions
    existing_positions = {(n["position"]["x"], n["position"]["y"]) for n in nodes}
    for jd in edited.get("junctions", []):
        jx = float(jd["position"]["x"])
        jy = float(jd["position"]["y"])
        if any(_euclidean((jx, jy), p) < 15 for p in existing_positions):
            continue
        nid = f"n{len(nodes)}"
        nodes.append({"id": nid, "position": {"x": jx, "y": jy}})
        existing_positions.add((jx, jy))

    # ── 2. Build Component list + assign terminals ───────────────────────
    components: List[Dict[str, Any]] = []
    connections: List[Dict[str, Any]] = []

    for comp in edited.get("components", []):
        cid = comp["id"]
        comp_type = comp.get("type", "unknown")
        value_str = comp.get("value", comp.get("matched_text", ""))
        bb = comp["bbox"]
        bbox = [bb["x1"], bb["y1"], bb["x2"], bb["y2"]]

        terminal_ids: List[str] = []
        for node in nodes:
            nx = node["position"]["x"]
            ny = node["position"]["y"]
            if _point_inside_bbox(nx, ny, bbox, margin=15):
                tid = f"{cid}_t{len(terminal_ids) + 1}"
                terminal_ids.append(tid)
                connections.append({
                    "from_component": cid,
                    "from_terminal": tid,
                    "to_node": node["id"],
                })

        if not terminal_ids:
            terminal_ids = [f"{cid}_t1", f"{cid}_t2"]

        components.append({
            "id": cid,
            "type": comp_type,
            "label": comp.get("name", cid),
            "value": value_str,
            "unit": _extract_unit(value_str),
            "position": comp["position"],
            "rotation": _estimate_rotation(bbox),
            "terminals": terminal_ids,
        })

    # ── 3. Build edges (node-to-node wire segments from skeleton graph) ──
    edges: List[Dict[str, Any]] = []
    node_ids = {n["id"] for n in nodes}
    for ge in edited.get("graph_edges", []):
        src_id = f"n{ge['source']}"
        tgt_id = f"n{ge['target']}"
        if src_id in node_ids and tgt_id in node_ids:
            edge_entry: Dict[str, Any] = {
                "source": src_id,
                "target": tgt_id,
                "path": ge.get("path", []),
            }
            edges.append(edge_entry)

    return _sanitize({
        "circuit_id": str(uuid4()),
        "components": components,
        "nodes": nodes,
        "connections": connections,
        "edges": edges,
    })


# ═══════════════════════════════════════════════════════════════════════════
# CircuitJS1 export
# ═══════════════════════════════════════════════════════════════════════════

_SI_PREFIXES = {
    "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9,
}

_CJS_GRID = 48          # CircuitJS1 default grid spacing in pixels
_CJS_COMP_LEN = 96      # default component length (2 × grid)


def _parse_value(raw: str) -> float:
    """Parse a value like '10k', '100µF', '4.7M', '5V' → float."""
    if not raw:
        return 0.0
    raw = raw.strip()
    cleaned = re.sub(r'[ΩΩoh FHVAWfhvaw]+$', '', raw, flags=re.IGNORECASE).strip()
    if not cleaned:
        return 0.0
    if cleaned[-1] in _SI_PREFIXES:
        prefix = cleaned[-1]
        num_str = cleaned[:-1]
        try:
            return float(num_str) * _SI_PREFIXES[prefix]
        except ValueError:
            return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _snap(v: float, grid: int = _CJS_GRID) -> int:
    """Snap a coordinate to the CJS grid."""
    return round(v / grid) * grid


def _type_to_cjs(comp_type: str) -> str:
    """Map our component type string to a CJS element code."""
    mapping = {
        "resistor": "r",
        "resistor.adjustable": "172",
        "capacitor": "c",
        "capacitor.unpolarized": "c",
        "capacitor.polarized": "209",
        "inductor": "l",
        "voltage": "v",
        "voltage.dc": "v",
        "voltage.ac": "v",
        "voltage.battery": "v",
        "diode": "d",
        "diode.zener": "z",
        "diode.light_emitting": "162",
        "transistor": "t",
        "transistor.bjt": "t",
        "transistor.fet": "j",
        "operational_amplifier": "a",
        "integrated_circuit": "x",
        "switch": "s",
        "gnd": "g",
        "vss": "V",
    }
    return mapping.get(comp_type, "r")


def _cjs_params(comp: dict, code: str) -> str:
    """Return the parameter tail for a CJS element line."""
    val = _parse_value(comp.get("value", "") or "")
    if code == "r":
        return f"0 {val if val else 1000}"
    if code in ("c", "209"):
        return f"0 {val if val else 1e-6} 0"
    if code == "l":
        return f"0 {val if val else 0.001} 0"
    if code == "v":
        ct = comp.get("type", "")
        if "ac" in ct:
            return f"16 1 40 {val if val else 5} 60 0 0.5"
        return f"16 0 40 {val if val else 5} 0 0 0.5"
    if code in ("d", "z", "162"):
        return "2 default"
    if code == "g":
        return "0"
    if code == "s":
        return "0 0 false"
    if code == "t":
        return "0 1 -5.434446395532359 0.6 100 default"
    if code == "j":
        return "0 1 -4 0 0.004 default"
    if code == "172":
        return f"0 {val if val else 1000} 0.5 0"
    if code == "a":
        return "0 15 -15 1000000 0.001 0 1000000"
    if code == "x":
        return "0"
    return f"0 {val if val else 1000}"


def _cjs_label_line(comp: dict, x1: int, y1: int) -> str | None:
    """Generate a CircuitJS1 label ('x') line for a component if it has a label or value."""
    label = comp.get("label", "")
    value = comp.get("value", "")
    # Build display text: prefer "label value" or just value
    parts = []
    if label:
        parts.append(label)
    if value:
        parts.append(value)
    if not parts:
        return None
    text = " ".join(parts)
    return f"x {x1} {y1 - 20} 0 12 {text}"


def circuit_to_cjs(circuit: dict) -> str:
    """
    Convert a Circuit dict to CircuitJS1 text format.

    The key insight: in CJS two elements are electrically connected when
    they share an *exact* endpoint coordinate.  So we:

    1. Assign every node (junction) a grid-snapped position.
    2. For each component, look up which nodes its terminals connect to
       (via the connections list) and place the component endpoints
       directly at those node positions.
    3. Render node-to-node edges as CJS wire ('w') elements.
    4. Components with only 1 connected node get a synthetic second
       endpoint offset along the component's axis.

    All image-pixel coordinates are rescaled to a compact schematic range
    so that component symbols remain readable relative to wire lengths.
    """
    lines: list[str] = []
    lines.append("$ 1 0.000005 10.20027730826997 63 10 62 5e-11")

    components = circuit.get("components", [])
    nodes_list = circuit.get("nodes", [])
    connections = circuit.get("connections", [])
    edges = circuit.get("edges", [])

    # ── 0. Collect raw positions & rescale to compact CJS space ─────────
    # Target: fit entire circuit within ~600px, offset from origin by ~100px
    _CJS_TARGET_SIZE = 500  # max span in x or y after rescaling
    _CJS_OFFSET = 100       # margin from top-left corner

    raw_positions: list[tuple[float, float]] = []
    for n in nodes_list:
        raw_positions.append((n["position"]["x"], n["position"]["y"]))
    for comp in components:
        p = comp.get("position", {"x": 0, "y": 0})
        raw_positions.append((p.get("x", 0), p.get("y", 0)))

    if raw_positions:
        xs = [p[0] for p in raw_positions]
        ys = [p[1] for p in raw_positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x if max_x > min_x else 1.0
        span_y = max_y - min_y if max_y > min_y else 1.0
        scale = _CJS_TARGET_SIZE / max(span_x, span_y)
        # Clamp scale so components never get smaller than 1 grid cell
        scale = min(scale, 1.0)

        def _rescale_snap(x: float, y: float) -> tuple[int, int]:
            nx = (x - min_x) * scale + _CJS_OFFSET
            ny = (y - min_y) * scale + _CJS_OFFSET
            return _snap(nx), _snap(ny)
    else:
        def _rescale_snap(x: float, y: float) -> tuple[int, int]:
            return _snap(x), _snap(y)

    # ── 1. Build node position map (rescaled + snapped) ─────────────────
    node_pos: dict[str, tuple[int, int]] = {}
    for n in nodes_list:
        nid = n["id"]
        node_pos[nid] = _rescale_snap(n["position"]["x"], n["position"]["y"])

    # ── 2. Index connections: component+terminal → node ─────────────────
    #   Also: node → list of (component_id, terminal_id)
    terminal_to_node: dict[str, str] = {}  # key = "comp_id::terminal_id"
    for conn in connections:
        key = f"{conn['from_component']}::{conn['from_terminal']}"
        terminal_to_node[key] = conn["to_node"]

    # ── 3. Build CJS lines for each component ──────────────────────────
    # Track the actual endpoint coords for each node so wires can share them
    node_used_coords: dict[str, tuple[int, int]] = dict(node_pos)

    for comp in components:
        cid = comp["id"]
        comp_type = comp.get("type", "resistor")
        code = _type_to_cjs(comp_type)
        pos = comp.get("position", {"x": 0, "y": 0})
        rotation = comp.get("rotation", 0) % 360
        terminals = comp.get("terminals", [])

        # Find the nodes each terminal connects to
        t_nodes: list[str | None] = []
        for tid in terminals:
            key = f"{cid}::{tid}"
            t_nodes.append(terminal_to_node.get(key))

        # Determine initial endpoint coordinates exactly from nodes
        if len(t_nodes) >= 2 and t_nodes[0] and t_nodes[1]:
            x1, y1 = node_pos[t_nodes[0]]
            x2, y2 = node_pos[t_nodes[1]]
        elif len(t_nodes) >= 1 and t_nodes[0]:
            x1, y1 = node_pos[t_nodes[0]]
            if rotation in (90, 270):
                x2, y2 = x1, y1 + _CJS_GRID
            else:
                x2, y2 = x1 + _CJS_GRID, y1
        elif len(t_nodes) >= 2 and t_nodes[1]:
            x2, y2 = node_pos[t_nodes[1]]
            if rotation in (90, 270):
                x1, y1 = x2, y2 - _CJS_GRID
            else:
                x1, y1 = x2 - _CJS_GRID, y2
        else:
            cx, cy = _rescale_snap(pos.get("x", 0), pos.get("y", 0))
            half = _CJS_GRID // 2
            if rotation in (90, 270):
                x1, y1, x2, y2 = cx, cy - half, cx, cy + half
            else:
                x1, y1, x2, y2 = cx - half, cy, cx + half, cy

        # --- TRUE "NO FORCE" MAPPING ---

        # 1. Snap the raw endpoints directly to the CJS grid
        x1, y1, x2, y2 = _snap(x1), _snap(y1), _snap(x2), _snap(y2)

        # 2. Prevent zero-length components (nodes snapped to same point)
        if x1 == x2 and y1 == y2 and code != "w":
            if pos.get("width", 1) >= pos.get("height", 1):
                x2 += _CJS_GRID
            else:
                y2 += _CJS_GRID

        # 3. Synchronize graph nodes so wires meet these exact pins,
        #    preserving any natural diagonals
        if len(t_nodes) >= 1 and t_nodes[0]:
            node_used_coords[t_nodes[0]] = (x1, y1)
        if len(t_nodes) >= 2 and t_nodes[1]:
            node_used_coords[t_nodes[1]] = (x2, y2)

        params = _cjs_params(comp, code)
        lines.append(f"{code} {x1} {y1} {x2} {y2} {params}")

        # Add label text element
        label_line = _cjs_label_line(comp, x1, y1)
        if label_line:
            lines.append(label_line)

    # ── 4. Add wires for node-to-node edges ─────────────────────────────
    for edge in edges:
        src_id = edge.get("source", "")
        tgt_id = edge.get("target", "")
        src_pos = node_used_coords.get(src_id)
        tgt_pos = node_used_coords.get(tgt_id)

        if src_pos and tgt_pos and src_pos != tgt_pos:
            sx, sy = src_pos
            tx, ty = tgt_pos
            lines.append(f"w {sx} {sy} {tx} {ty} 0")

    return "\n".join(lines) + "\n"
