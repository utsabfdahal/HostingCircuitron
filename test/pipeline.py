"""
Circuit-image analysis pipeline.

Single entry-point:  analyze(image_bytes, label_text, params)
Returns a JSON-serialisable dict suitable for a REST API response.
"""

from __future__ import annotations

import base64
import io
import math
from collections import Counter, deque
from itertools import combinations
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")                       # no GUI backend``
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte


# ── constants ────────────────────────────────────────────────────────────────

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

DEFAULT_PARAMS = {
    "node_radius": 25,
    "cls1_margin": 8,
    "border_tol": 1,
    "global_merge_dist": 5,
    "find_node_dist_thresh": 5,
    "binary_thresh": 110,
    "skel_thresh": 128,
}


# ── image helpers ────────────────────────────────────────────────────────────

def _numpy_to_b64png(arr: np.ndarray, *, is_bgr: bool = False) -> str:
    """Encode a numpy image (gray or BGR) as a base64 data-URL PNG."""
    if is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    success, buf = cv2.imencode(".png", arr)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fig_to_b64png(fig: plt.Figure) -> str:
    """Render a matplotlib Figure to a base64 data-URL PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── YOLO parsing ─────────────────────────────────────────────────────────────

def _parse_yolo_text(label_text: str) -> list[dict]:
    dets: list[dict] = []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else None
        dets.append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf})
    return dets


def _yolo_to_xyxy(xc: float, yc: float, w: float, h: float,
                   W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int(round((xc - w / 2) * W))
    y1 = int(round((yc - h / 2) * H))
    x2 = int(round((xc + w / 2) * W))
    y2 = int(round((yc + h / 2) * H))
    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
    x2, y2 = max(0, min(W - 1, x2)), max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


# ── skeleton / endpoint helpers ──────────────────────────────────────────────

def _get_all_border_intersections(skel, inter_xy, merge_dist: float = 8.0):
    if len(inter_xy) == 0:
        return [], "no_border_hits"
    raw_pts = sorted(list(set(tuple(map(int, p)) for p in inter_xy)))
    merged: list[tuple[int, int]] = []
    while raw_pts:
        curr = raw_pts.pop(0)
        found_group = False
        for i, m in enumerate(merged):
            dist = np.sqrt((curr[0] - m[0]) ** 2 + (curr[1] - m[1]) ** 2)
            if dist <= merge_dist:
                merged[i] = (
                    int(round((curr[0] + m[0]) / 2)),
                    int(round((curr[1] + m[1]) / 2)),
                )
                found_group = True
                break
        if not found_group:
            merged.append(curr)
    return merged, "all_border_hits_cleaned"


def _skeleton_intersections_with_bbox_border(skel, x1, y1, x2, y2, tol: int = 1):
    H, W = skel.shape
    xx1, yy1 = max(0, x1 - tol), max(0, y1 - tol)
    xx2, yy2 = min(W - 1, x2 + tol), min(H - 1, y2 + tol)
    band = np.zeros_like(skel, dtype=bool)
    band[yy1:yy2 + 1, xx1:xx2 + 1] = True
    if (x2 - x1) > 2 * tol and (y2 - y1) > 2 * tol:
        band[y1 + tol:y2 - tol + 1, x1 + tol:x2 - tol + 1] = False
    ys, xs = np.where((skel > 0) & band)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)
    return np.stack([xs, ys], axis=1)


def _nearest_skeleton_point_with_dist(skel, anchor_xy, search_xyxy):
    x1, y1, x2, y2 = search_xyxy
    roi = skel[y1:y2 + 1, x1:x2 + 1]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return None, None
    pts = np.stack([x1 + xs, y1 + ys], axis=1)
    ax, ay = anchor_xy
    d2 = (pts[:, 0] - ax) ** 2 + (pts[:, 1] - ay) ** 2
    i = int(np.argmin(d2))
    return tuple(map(int, pts[i])), float(d2[i])


def _endpoint_for_class1_single(skel, x1, y1, x2, y2, margin: int = 8):
    H, W = skel.shape
    xc = int(round((x1 + x2) / 2))
    yc = int(round((y1 + y2) / 2))
    sx1, sy1 = max(0, x1 - margin), max(0, y1 - margin)
    sx2, sy2 = min(W - 1, x2 + margin), min(H - 1, y2 + margin)
    p, _ = _nearest_skeleton_point_with_dist(skel, (xc, yc), (sx1, sy1, sx2, sy2))
    return p, ("cls1_center_snap" if p else "cls1_not_found")


def _extract_all_endpoints(results, global_merge_dist: float = 5.0):
    all_pts: list[tuple[int, int]] = []
    for r in results:
        if "endpoint_xy" in r and r["endpoint_xy"] is not None:
            all_pts.append(tuple(map(int, r["endpoint_xy"])))
        if "endpoints" in r and r["endpoints"]:
            for ep in r["endpoints"]:
                if ep is not None:
                    all_pts.append(tuple(map(int, ep)))
    if not all_pts:
        return []
    unique: list[tuple[int, int]] = []
    for pt in all_pts:
        if not unique:
            unique.append(pt)
            continue
        pt_arr = np.array(pt)
        unique_arr = np.array(unique)
        distances = np.linalg.norm(unique_arr - pt_arr, axis=1)
        if np.min(distances) > global_merge_dist:
            unique.append(pt)
    return unique


# ── adjacency graph helpers ──────────────────────────────────────────────────

def _get_neighbors8(y, x, H, W):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W:
                neighbors.append((ny, nx_))
    return neighbors


def _disk_mask(radius: int):
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                offsets.append((dy, dx))
    return offsets


def _build_node_id_map(skel, nodes, radius: int):
    H, W = skel.shape
    node_id_map = np.full((H, W), -1, dtype=np.int32)
    disk = _disk_mask(radius)
    for i, (nx_, ny) in enumerate(nodes):
        for dy, dx in disk:
            py, px = ny + dy, nx_ + dx
            if 0 <= py < H and 0 <= px < W:
                node_id_map[py, px] = i
    return node_id_map


def _bfs_neighbors_for_node(node_idx, nodes, skel, node_id_map, radius: int):
    H, W = skel.shape
    nx_, ny = nodes[node_idx]
    neighbors_found: set[int] = set()
    visited: set[tuple[int, int]] = set()          # sparse set instead of full image array
    disk = _disk_mask(radius)
    for dy, dx in disk:
        py, px = ny + dy, nx_ + dx
        if 0 <= py < H and 0 <= px < W:
            visited.add((py, px))
    queue: deque[tuple[int, int]] = deque()
    for dy, dx in disk:
        py, px = ny + dy, nx_ + dx
        if 0 <= py < H and 0 <= px < W:
            for npy, npx in _get_neighbors8(py, px, H, W):
                if (npy, npx) not in visited and skel[npy, npx] > 0:
                    visited.add((npy, npx))
                    queue.append((npy, npx))
    while queue:
        cy, cx = queue.popleft()
        other_node = node_id_map[cy, cx]
        if other_node != -1 and other_node != node_idx:
            neighbors_found.add(other_node)
            continue
        for npy, npx in _get_neighbors8(cy, cx, H, W):
            if (npy, npx) not in visited and skel[npy, npx] > 0:
                visited.add((npy, npx))
                queue.append((npy, npx))
    return neighbors_found


def _build_adjacency(skel, nodes, radius: int):
    node_id_map = _build_node_id_map(skel, nodes, radius)
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for i in range(len(nodes)):
        neighbors = _bfs_neighbors_for_node(i, nodes, skel, node_id_map, radius)
        for j in neighbors:
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency


def _find_node_index(target_coord, all_nodes, dist_threshold: float):
    target = np.array(target_coord)
    nodes_arr = np.array(all_nodes)
    distances = np.linalg.norm(nodes_arr - target, axis=1)
    min_idx = int(np.argmin(distances))
    return min_idx if distances[min_idx] <= dist_threshold else -1


def _count_components(adj: dict[int, set[int]]) -> int:
    visited: set[int] = set()
    count = 0
    for node in adj:
        if node not in visited:
            count += 1
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(adj[curr] - visited)
    return count


# ── crossover dissolution helpers ────────────────────────────────────────────

def _normalized_vector(ax, ay, bx, by):
    """Return the unit vector from (ax,ay) to (bx,by), or None if zero-length."""
    dx = bx - ax
    dy = by - ay
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return None
    return (dx / mag, dy / mag)


def _dot(v1, v2):
    """Dot product of two 2D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def _dissolve_crossovers(results, all_nodes, adjacency, find_dist):
    """Dissolve crossover nodes (cls==1) using dot-product pairing."""
    crossover_node_indices = []
    for r in results:
        if r["id"] == 1 and "endpoint_xy" in r and r["endpoint_xy"] is not None:
            idx = _find_node_index(r["endpoint_xy"], all_nodes, find_dist)
            if idx != -1:
                crossover_node_indices.append(idx)

    for a_idx in crossover_node_indices:
        neighbors = list(adjacency.get(a_idx, set()))
        if len(neighbors) != 4:
            continue
        ax, ay = all_nodes[a_idx]
        vecs = {}
        skip = False
        for n_idx in neighbors:
            nx, ny = all_nodes[n_idx]
            v = _normalized_vector(ax, ay, nx, ny)
            if v is None:
                skip = True
                break
            vecs[n_idx] = v
        if skip:
            continue
        b, c, d, e = neighbors
        pairings = [((b, c), (d, e)), ((b, d), (c, e)), ((b, e), (c, d))]
        best_score = -float("inf")
        best_pairing = None
        for (p1, p2), (p3, p4) in pairings:
            score = _dot(vecs[p1], vecs[p2]) + _dot(vecs[p3], vecs[p4])
            if score < best_score or best_pairing is None:
                best_score = score
                best_pairing = ((p1, p2), (p3, p4))
        (pair1_a, pair1_b), (pair2_a, pair2_b) = best_pairing
        dot1 = _dot(vecs[pair1_a], vecs[pair1_b])
        dot2 = _dot(vecs[pair2_a], vecs[pair2_b])
        DOT_THRESHOLD = -0.7
        if dot1 > DOT_THRESHOLD or dot2 > DOT_THRESHOLD:
            continue
        for n_idx in neighbors:
            adjacency[n_idx].discard(a_idx)
        del adjacency[a_idx]
        adjacency[pair1_a].add(pair1_b)
        adjacency[pair1_b].add(pair1_a)
        adjacency[pair2_a].add(pair2_b)
        adjacency[pair2_b].add(pair2_a)


def _build_node_component_map(results, all_nodes, adjacency, merge_dist):
    """Map each node index to the component result indices whose endpoints touch it."""
    link_dist = merge_dist + 3
    node_to_components = {i: [] for i in range(len(all_nodes))}
    for r_idx, r in enumerate(results):
        comp_endpoints = []
        if "endpoint_xy" in r and r["endpoint_xy"] is not None:
            comp_endpoints.append(r["endpoint_xy"])
        if "endpoints" in r:
            for ep in r["endpoints"]:
                if ep is not None:
                    comp_endpoints.append(ep)
        for ep in comp_endpoints:
            ex, ey = ep
            best_dist = float("inf")
            best_node = -1
            for n_idx, (nx, ny) in enumerate(all_nodes):
                d = math.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_node = n_idx
            if best_node != -1 and best_dist <= link_dist:
                if r_idx not in node_to_components[best_node]:
                    node_to_components[best_node].append(r_idx)
    return node_to_components


# ── image renderers (to bytes, not disk) ─────────────────────────────────────

def _render_overlay(gray_img: np.ndarray, results: list[dict],
                    skel: np.ndarray | None = None) -> str:
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for r in results:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 139), 1)
        if "endpoint_xy" in r and r["endpoint_xy"]:
            cv2.circle(vis, r["endpoint_xy"], 4, (0, 0, 255), -1)
        if "endpoints" in r:
            for p in r["endpoints"]:
                cv2.circle(vis, p, 4, (0, 0, 255), -1)
    if skel is not None:
        vis[skel > 0] = (255, 0, 0)
    return _numpy_to_b64png(vis, is_bgr=True)


def _render_adjacency_graph(nodes, adjacency) -> str:
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    drawn_edges: set[tuple[int, int]] = set()
    for i, neighbors in adjacency.items():
        x1, y1 = nodes[i]
        for j in neighbors:
            edge = tuple(sorted([i, j]))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            x2, y2 = nodes[j]
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.7)
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    ax.scatter(xs, ys, s=150, c="red", zorder=5, edgecolors="black", linewidths=1.5)
    for i, (x, y) in enumerate(nodes):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Adjacency Graph: {len(nodes)} nodes, {len(drawn_edges)} edges")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64png(fig)


def _render_bbox_overlay(skeleton_bool: np.ndarray,
                         detections: list[dict]) -> str:
    vis = cv2.cvtColor(skeleton_bool.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    class_names_full = {i: name for i, name in enumerate(COMPONENT_NAMES)}
    class_colors = {
        0: (0, 165, 255),    # capacitor – orange
        1: (255, 0, 255),    # crossover – magenta
        2: (0, 255, 0),      # diode – green
        3: (0, 255, 255),    # gnd – yellow
        4: (255, 0, 0),      # inductor – blue
        5: (128, 128, 255),  # integrated_circuit – light red
        6: (0, 0, 255),      # junction – red
        7: (255, 128, 0),    # operational_amplifier – cyan-ish
        8: (0, 200, 200),    # resistor – dark yellow
        9: (200, 200, 0),    # switch – light cyan
        10: (0, 0, 255),     # terminal – red
        11: (180, 180, 180), # text – gray
        12: (255, 100, 100), # transistor – light blue
        13: (0, 255, 128),   # voltage – spring green
        14: (128, 0, 255),   # vss – purple
    }
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        color = class_colors.get(d["cls"], (128, 128, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        lbl = class_names_full.get(d["cls"], f"Class {d['cls']}")
        if d["conf"] is not None:
            lbl = f"{lbl} {d['conf']:.2f}"
        fs, ft = 0.5, 1
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(vis, lbl, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)
    return _numpy_to_b64png(vis, is_bgr=True)


# ── PUBLIC API ───────────────────────────────────────────────────────────────

def analyze(
    image_bytes: bytes,
    label_text: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the full circuit-image analysis pipeline.

    Parameters
    ----------
    image_bytes : raw bytes of a grayscale (or colour) PNG/JPG circuit image.
    label_text  : YOLO-format detections (one line per detection).
    params      : optional overrides for processing parameters.

    Returns
    -------
    dict with keys:
        detections  – list of detection dicts (cls, name, bbox, conf)
        results     – per-detection endpoint info
        graph       – {nodes, edges, num_components}
        images      – {skeleton_png, overlay_png, bbox_png, adjacency_graph_png}
                      (base64 data-URL strings)
    """
    # ── merge params ──
    p = {**DEFAULT_PARAMS, **(params or {})}
    NODE_RADIUS       = p["node_radius"]
    CLS1_MARGIN       = p["cls1_margin"]
    BORDER_TOL        = p["border_tol"]
    GLOBAL_MERGE_DIST = p["global_merge_dist"]
    FIND_NODE_DIST    = p["find_node_dist_thresh"]
    BINARY_THRESH     = p["binary_thresh"]
    SKEL_THRESH       = p["skel_thresh"]

    # ── decode image ──
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_original = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError("Could not decode the uploaded image")
    IMG_H, IMG_W = img_original.shape

    # ── parse detections ──
    yolo_dets = _parse_yolo_text(label_text)
    detections_with_bbox: list[dict] = []
    for d in yolo_dets:
        xyxy = _yolo_to_xyxy(d["xc"], d["yc"], d["w"], d["h"], IMG_W, IMG_H)
        detections_with_bbox.append({**d, "bbox_xyxy": xyxy})

    # ── pre-process ──
    denoised = cv2.medianBlur(img_original, 1)
    _, simple_bin = cv2.threshold(denoised, BINARY_THRESH, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(simple_bin)
    skeleton = skeletonize(inverted > 0)

    # erase text regions (cls 11) from skeleton
    for d in detections_with_bbox:
        if d["cls"] == 11:  # text
            x1, y1, x2, y2 = d["bbox_xyxy"]
            skeleton[y1:y2, x1:x2] = False

    # dilate + re-skeletonize
    img_u8 = skeleton.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.bitwise_not(cv2.dilate(img_u8, kernel, iterations=1))
    binary_fg = (dilated < 128).astype(np.uint8)
    skel = skeletonize(binary_fg.astype(bool)).astype(np.uint8)
    skel_u8 = img_as_ubyte(skel)

    # black-on-white for display
    skel_black_on_white = np.ones_like(dilated) * 255
    skel_black_on_white[skel_u8 > 0] = 0
    skel_gray = skel_black_on_white.astype(np.uint8)

    # reuse the already-skeletonized result instead of running skeletonize a 3rd time
    skel_img = skel

    # ── endpoint detection ──
    results: list[dict] = []
    for d in detections_with_bbox:
        if d["cls"] == 11:  # skip text
            continue
        x1, y1, x2, y2 = d["bbox_xyxy"]
        if d["cls"] in (1, 6, 10):  # crossover, junction, terminal
            ep, method = _endpoint_for_class1_single(
                skel_img, x1, y1, x2, y2, margin=CLS1_MARGIN)
            results.append({
                "id": d["cls"], "conf": d["conf"],
                "bbox_xyxy": (x1, y1, x2, y2),
                "endpoint_xy": ep, "method": method,
            })
        else:
            inter = _skeleton_intersections_with_bbox_border(
                skel_img, x1, y1, x2, y2, tol=BORDER_TOL)
            if len(inter) == 0:
                roi = skel_img[y1:y2 + 1, x1:x2 + 1]
                ys, xs = np.where(roi > 0)
                pts = ([tuple(map(int, pt))
                        for pt in np.stack([x1 + xs, y1 + ys], axis=1)]
                       if len(xs) > 0 else [])
                method = "fallback_internal"
            else:
                pts, method = _get_all_border_intersections(skel_img, inter)
            results.append({
                "id": d["cls"], "conf": d["conf"],
                "bbox_xyxy": (x1, y1, x2, y2),
                "endpoints": pts, "method": method,
            })

    # ── extract + merge endpoints ──
    all_nodes = _extract_all_endpoints(results, global_merge_dist=GLOBAL_MERGE_DIST)

    # ── build adjacency ──
    if len(all_nodes) > 0:
        adjacency = _build_adjacency(skel_img, all_nodes, radius=NODE_RADIUS)

        # discard internal box connections
        for r in results:
            if "endpoints" in r and len(r.get("endpoints", [])) >= 2:
                for (p1, p2) in combinations(r["endpoints"], 2):
                    i1 = _find_node_index(p1, all_nodes, FIND_NODE_DIST)
                    i2 = _find_node_index(p2, all_nodes, FIND_NODE_DIST)
                    if i1 != -1 and i2 != -1:
                        adjacency[i1].discard(i2)
                        adjacency[i2].discard(i1)

        num_components = _count_components(adjacency)
    else:
        adjacency = {}
        num_components = 0

    # ── dissolve crossover nodes ──
    if len(all_nodes) > 0:
        _dissolve_crossovers(results, all_nodes, adjacency, FIND_NODE_DIST)
        num_components = _count_components(adjacency)

    # ── node-to-component mapping ──
    node_to_components = {}
    if len(all_nodes) > 0:
        node_to_components = _build_node_component_map(
            results, all_nodes, adjacency, GLOBAL_MERGE_DIST,
        )

    # ── serialise graph ──
    edges: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for i, nbrs in adjacency.items():
        for j in nbrs:
            edge = tuple(sorted([i, j]))
            if edge not in seen:
                seen.add(edge)
                edges.append(list(edge))

    # Build edge payloads
    edges_payload = []
    for e in edges:
        src, tgt = e[0], e[1]
        edge_entry: dict[str, Any] = {
            "source": src,
            "target": tgt,
        }
        # Attach linked component info per endpoint
        src_comps = node_to_components.get(src, [])
        tgt_comps = node_to_components.get(tgt, [])
        if src_comps or tgt_comps:
            edge_entry["linked_components"] = {
                "source_components": [
                    {
                        "cls": results[ri]["id"],
                        "name": (COMPONENT_NAMES[results[ri]["id"]]
                                 if results[ri]["id"] < len(COMPONENT_NAMES)
                                 else f"class_{results[ri]['id']}"),
                        "bbox": list(results[ri]["bbox_xyxy"]),
                    }
                    for ri in src_comps
                ],
                "target_components": [
                    {
                        "cls": results[ri]["id"],
                        "name": (COMPONENT_NAMES[results[ri]["id"]]
                                 if results[ri]["id"] < len(COMPONENT_NAMES)
                                 else f"class_{results[ri]['id']}"),
                        "bbox": list(results[ri]["bbox_xyxy"]),
                    }
                    for ri in tgt_comps
                ],
            }
        edges_payload.append(edge_entry)

    graph_payload = {
        "nodes": [{"id": i, "x": int(pt[0]), "y": int(pt[1])}
                  for i, pt in enumerate(all_nodes)],
        "edges": edges_payload,
        "num_components": num_components,
    }

    # ── serialise detections ──
    det_payload = []
    for d in detections_with_bbox:
        name = (COMPONENT_NAMES[d["cls"]]
                if d["cls"] < len(COMPONENT_NAMES) else f"class_{d['cls']}")
        det_payload.append({
            "cls": d["cls"],
            "name": name,
            "conf": d["conf"],
            "bbox": list(d["bbox_xyxy"]),
        })

    # ── serialise results (make tuples into lists) ──
    results_payload = []
    for r in results:
        entry: dict[str, Any] = {
            "cls": r["id"],
            "name": (COMPONENT_NAMES[r["id"]]
                     if r["id"] < len(COMPONENT_NAMES) else f"class_{r['id']}"),
            "conf": r["conf"],
            "bbox": list(r["bbox_xyxy"]),
            "method": r["method"],
        }
        if "endpoint_xy" in r:
            entry["endpoint"] = list(r["endpoint_xy"]) if r["endpoint_xy"] else None
        if "endpoints" in r:
            entry["endpoints"] = [list(p) for p in r["endpoints"]]
        results_payload.append(entry)

    # ── render images ──
    images = {
        "skeleton_png": _numpy_to_b64png(skel_gray),
        "overlay_png": _render_overlay(skel_gray, results, skel=skel_img),
        "bbox_png": _render_bbox_overlay(skeleton, detections_with_bbox),
        "adjacency_graph_png": (
            _render_adjacency_graph(all_nodes, adjacency)
            if len(all_nodes) > 0 else None
        ),
    }

    return {
        "image_size": {"width": IMG_W, "height": IMG_H},
        "detections": det_payload,
        "results": results_payload,
        "graph": graph_payload,
        "images": images,
    }
