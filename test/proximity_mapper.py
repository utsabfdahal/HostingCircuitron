"""
Proximity-based mapping of OCR text boxes to circuit components.

For each text detection, find the nearest non-text component by the
minimum edge-to-edge distance between their bounding boxes.  This is far
more accurate than centre-to-centre distance when components have very
different sizes.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any


def _bbox_center(bbox: List[int]) -> np.ndarray:
    """Return (cx, cy) of an [x1, y1, x2, y2] box."""
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


def _bbox_edge_distance(a: List[int], b: List[int]) -> float:
    """
    Minimum edge-to-edge distance between two [x1, y1, x2, y2] boxes.

    Returns 0 when the boxes overlap.
    """
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return float(np.sqrt(dx * dx + dy * dy))


def map_text_to_components(
    components: List[Dict[str, Any]],
    text_detections: List[Dict[str, Any]],
    max_distance: float = 250.0,
) -> List[Dict[str, Any]]:
    """
    Assign each OCR text result to its nearest component.

    The distance metric is the minimum gap between the edges of the text
    bounding box and each component bounding box — not the centre-to-centre
    distance.  This ensures that a small text label sitting right next to a
    large component still gets matched correctly.

    Parameters
    ----------
    components       : list of component dicts (each must have ``bbox``).
    text_detections  : list of text dicts (must have ``bbox``, ``ocr_text``, ``ocr_confidence``).
    max_distance     : maximum pixel distance for a valid assignment.

    Returns
    -------
    The *components* list, enriched with ``value`` and ``value_confidence``
    fields from the nearest text. Also adds ``mapped_text_bbox`` so the
    frontend can draw the association.
    """
    if not components or not text_detections:
        # Ensure every component has at least empty value fields
        for c in components:
            c.setdefault("value", "")
            c.setdefault("value_confidence", 0.0)
            c.setdefault("mapped_text_bbox", None)
        return components

    # Build an edge-distance matrix: (C, T)
    n_comp = len(components)
    n_text = len(text_detections)
    dists = np.zeros((n_comp, n_text), dtype=np.float64)

    for ci, comp in enumerate(components):
        for ti, td in enumerate(text_detections):
            dists[ci, ti] = _bbox_edge_distance(comp["bbox"], td["bbox"])

    # Greedy one-to-one assignment — sort all pairs by distance
    pairs = []
    for ci in range(n_comp):
        for ti in range(n_text):
            pairs.append((dists[ci, ti], ci, ti))
    pairs.sort(key=lambda x: x[0])

    used_texts: set[int] = set()
    assigned_comps: set[int] = set()

    for dist, ci, ti in pairs:
        if dist > max_distance:
            break
        if ci in assigned_comps or ti in used_texts:
            continue
        td = text_detections[ti]
        if not td.get("ocr_text"):
            continue
        components[ci]["value"] = td["ocr_text"]
        components[ci]["value_confidence"] = td.get("ocr_confidence", 0.0)
        components[ci]["mapped_text_bbox"] = td["bbox"]
        assigned_comps.add(ci)
        used_texts.add(ti)

    # Ensure every component has at least empty value fields
    for c in components:
        c.setdefault("value", "")
        c.setdefault("value_confidence", 0.0)
        c.setdefault("mapped_text_bbox", None)

    return components
