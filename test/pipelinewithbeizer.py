"""
Run the circuit analysis pipeline step by step, saving an image after each stage.
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from pipeline import (
    _parse_yolo_text, _yolo_to_xyxy, _endpoint_for_class1_single,
    _skeleton_intersections_with_bbox_border, _get_all_border_intersections,
    _extract_all_endpoints, _build_adjacency, _find_node_index,
    _count_components, _disk_mask,
    COMPONENT_NAMES, DEFAULT_PARAMS,
)
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──
IMAGE_PATH = "/Users/mac/Downloads/LinedetectionforJustimprovement/finalfordem.JPG"
LABEL_PATH = "/Users/mac/Downloads/LinedetectionforJustimprovement/finalfordem8.txt"
OUT_DIR    = "/Users/mac/Downloads/LinedetectionforJustimprovement/steps_outputfinalbeizer"
os.makedirs(OUT_DIR, exist_ok=True)

p = DEFAULT_PARAMS

# ── Step 0: Load original image ──
print("Step 0: Loading original image...")
img_original = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img_original is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")
IMG_H, IMG_W = img_original.shape
print(f"  Image size: {IMG_W} x {IMG_H}")
cv2.imwrite(os.path.join(OUT_DIR, "step0_original.png"), img_original)

# ── Step 1: Parse YOLO detections & draw bounding boxes on original ──
print("Step 1: Parsing YOLO detections and drawing bounding boxes...")
with open(LABEL_PATH) as f:
    label_text = f.read()

yolo_dets = _parse_yolo_text(label_text)
detections_with_bbox = []
for d in yolo_dets:
    xyxy = _yolo_to_xyxy(d["xc"], d["yc"], d["w"], d["h"], IMG_W, IMG_H)
    detections_with_bbox.append({**d, "bbox_xyxy": xyxy})

vis_dets = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
class_colors = {
    0: (0, 165, 255), 1: (255, 0, 255), 2: (0, 255, 0), 3: (0, 255, 255),
    4: (255, 0, 0), 5: (128, 128, 255), 6: (0, 0, 255), 7: (255, 128, 0),
    8: (0, 200, 200), 9: (200, 200, 0), 10: (0, 0, 255), 11: (180, 180, 180),
    12: (255, 100, 100), 13: (0, 255, 128), 14: (128, 0, 255),
}
for d in detections_with_bbox:
    x1, y1, x2, y2 = d["bbox_xyxy"]
    color = class_colors.get(d["cls"], (128, 128, 128))
    cv2.rectangle(vis_dets, (x1, y1), (x2, y2), color, 2)
    lbl = COMPONENT_NAMES[d["cls"]] if d["cls"] < len(COMPONENT_NAMES) else f"cls{d['cls']}"
    if d["conf"] is not None:
        lbl = f"{lbl} {d['conf']:.2f}"
    cv2.putText(vis_dets, lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
cv2.imwrite(os.path.join(OUT_DIR, "step1_detections.png"), vis_dets)
print(f"  Found {len(detections_with_bbox)} detections")

# ── Step 2: Denoise + Binary Threshold ──
print("Step 2: Denoising + binary thresholding...")
denoised = cv2.medianBlur(img_original, 1)
_, simple_bin = cv2.threshold(denoised, p["binary_thresh"], 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(OUT_DIR, "step2_binary.png"), simple_bin)

# ── Step 3: Invert ──
print("Step 3: Inverting binary image...")
inverted = cv2.bitwise_not(simple_bin)
cv2.imwrite(os.path.join(OUT_DIR, "step3_inverted.png"), inverted)

# ── Step 4: Initial skeletonization ──
print("Step 4: Initial skeletonization...")
skeleton = skeletonize(inverted > 0)
skel_display = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255
skel_display[skeleton] = 0
cv2.imwrite(os.path.join(OUT_DIR, "step4_skeleton_initial.png"), skel_display)

# ── Step 5: Erase text regions from skeleton ──
print("Step 5: Erasing text regions (cls 11) from skeleton...")
skeleton_no_text = skeleton.copy()
for d in detections_with_bbox:
    if d["cls"] == 11:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        skeleton_no_text[y1:y2, x1:x2] = False
skel_nt_display = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255
skel_nt_display[skeleton_no_text] = 0
cv2.imwrite(os.path.join(OUT_DIR, "step5_skeleton_no_text.png"), skel_nt_display)

# ── Step 6: Dilate + re-skeletonize ──
print("Step 6: Dilating and re-skeletonizing...")
img_u8 = skeleton_no_text.astype(np.uint8) * 255
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.bitwise_not(cv2.dilate(img_u8, kernel, iterations=1))
binary_fg = (dilated < 128).astype(np.uint8)
skel = skeletonize(binary_fg.astype(bool)).astype(np.uint8)
skel_u8 = img_as_ubyte(skel)

skel_black_on_white = np.ones_like(dilated) * 255
skel_black_on_white[skel_u8 > 0] = 0
skel_gray = skel_black_on_white.astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "step6_skeleton_final.png"), skel_gray)

skel_img = skel  # the final skeleton used for endpoint detection

# ── Step 7: Bounding boxes on skeleton ──
print("Step 7: Bounding boxes overlaid on skeleton...")
vis_skel_bbox = cv2.cvtColor(skeleton_no_text.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
for d in detections_with_bbox:
    x1, y1, x2, y2 = d["bbox_xyxy"]
    color = class_colors.get(d["cls"], (128, 128, 128))
    cv2.rectangle(vis_skel_bbox, (x1, y1), (x2, y2), color, 2)
    lbl = COMPONENT_NAMES[d["cls"]] if d["cls"] < len(COMPONENT_NAMES) else f"cls{d['cls']}"
    if d["conf"] is not None:
        lbl = f"{lbl} {d['conf']:.2f}"
    fs, ft = 0.5, 1
    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    cv2.rectangle(vis_skel_bbox, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
    cv2.putText(vis_skel_bbox, lbl, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)
cv2.imwrite(os.path.join(OUT_DIR, "step7_skeleton_bboxes.png"), vis_skel_bbox)

# ── Step 8: Endpoint detection ──
print("Step 8: Detecting endpoints at bounding box borders...")
results = []
for d in detections_with_bbox:
    if d["cls"] == 11:
        continue
    x1, y1, x2, y2 = d["bbox_xyxy"]
    if d["cls"] in (1, 6):
        ep, method = _endpoint_for_class1_single(
            skel_img, x1, y1, x2, y2, margin=p["cls1_margin"])
        results.append({
            "id": d["cls"], "conf": d["conf"],
            "bbox_xyxy": (x1, y1, x2, y2),
            "endpoint_xy": ep, "method": method,
        })
    else:
        inter = _skeleton_intersections_with_bbox_border(
            skel_img, x1, y1, x2, y2, tol=p["border_tol"])
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

# Draw endpoints on skeleton
vis_endpoints = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2BGR)
for r in results:
    x1, y1, x2, y2 = r["bbox_xyxy"]
    cv2.rectangle(vis_endpoints, (x1, y1), (x2, y2), (0, 0, 139), 1)
    if "endpoint_xy" in r and r["endpoint_xy"]:
        cv2.circle(vis_endpoints, r["endpoint_xy"], 5, (0, 0, 255), -1)
    if "endpoints" in r:
        for pt in r["endpoints"]:
            cv2.circle(vis_endpoints, pt, 5, (0, 0, 255), -1)
# Show skeleton in blue
vis_endpoints[skel_img > 0] = (255, 0, 0)
cv2.imwrite(os.path.join(OUT_DIR, "step8_endpoints.png"), vis_endpoints)
print(f"  Detected endpoints for {len(results)} components")

# ── Step 9: Extract & merge all nodes ──
print("Step 9: Extracting and merging global node set...")
all_nodes = _extract_all_endpoints(results, global_merge_dist=p["global_merge_dist"])
print(f"  Total unique nodes: {len(all_nodes)}")

vis_nodes = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2BGR)
vis_nodes[skel_img > 0] = (255, 0, 0)
for i, (nx, ny) in enumerate(all_nodes):
    cv2.circle(vis_nodes, (nx, ny), 6, (0, 0, 255), -1)
    cv2.putText(vis_nodes, str(i), (nx + 6, ny - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 0), 1)
cv2.imwrite(os.path.join(OUT_DIR, "step9_nodes.png"), vis_nodes)

# ── Step 10: Build adjacency graph ──
print("Step 10: Building adjacency graph via BFS on skeleton...")
if len(all_nodes) > 0:
    adjacency = _build_adjacency(skel_img, all_nodes, radius=p["node_radius"])

    # remove intra-box connections
    for r in results:
        if "endpoints" in r and len(r.get("endpoints", [])) >= 2:
            for (p1, p2) in combinations(r["endpoints"], 2):
                i1 = _find_node_index(p1, all_nodes, p["find_node_dist_thresh"])
                i2 = _find_node_index(p2, all_nodes, p["find_node_dist_thresh"])
                if i1 != -1 and i2 != -1:
                    adjacency[i1].discard(i2)
                    adjacency[i2].discard(i1)

    num_components = _count_components(adjacency)
else:
    adjacency = {}
    num_components = 0

edges = []
seen = set()
for i, nbrs in adjacency.items():
    for j in nbrs:
        edge = tuple(sorted([i, j]))
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)
print(f"  Edges: {len(edges)}, Connected components: {num_components}")

# Render adjacency graph with matplotlib
fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
drawn_edges = set()
for i, neighbors in adjacency.items():
    x1, y1 = all_nodes[i]
    for j in neighbors:
        edge = tuple(sorted([i, j]))
        if edge in drawn_edges:
            continue
        drawn_edges.add(edge)
        x2, y2 = all_nodes[j]
        ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.7)

xs = [n[0] for n in all_nodes]
ys = [n[1] for n in all_nodes]
ax.scatter(xs, ys, s=150, c="red", zorder=5, edgecolors="black", linewidths=1.5)
for i, (x, y) in enumerate(all_nodes):
    ax.annotate(str(i), (x, y), textcoords="offset points",
                xytext=(8, 8), fontsize=10, fontweight="bold")
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Adjacency Graph: {len(all_nodes)} nodes, {len(drawn_edges)} edges, "
             f"{num_components} components")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step10_adjacency_graph.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Step 11: Dissolve crossover nodes using dot-product pairing ──
print("Step 11: Dissolving crossover nodes via dot-product pairing...")
import math

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

# 1. Identify crossover node indices (cls == 1)
crossover_node_indices = []
for r in results:
    if r["id"] == 1 and "endpoint_xy" in r and r["endpoint_xy"] is not None:
        idx = _find_node_index(r["endpoint_xy"], all_nodes, p["find_node_dist_thresh"])
        if idx != -1:
            crossover_node_indices.append(idx)

print(f"  Found {len(crossover_node_indices)} crossover node(s): {crossover_node_indices}")

dissolved_count = 0
for a_idx in crossover_node_indices:
    neighbors = list(adjacency.get(a_idx, set()))
    if len(neighbors) != 4:
        print(f"  Crossover node {a_idx} has {len(neighbors)} neighbors (need 4), skipping.")
        continue

    ax, ay = all_nodes[a_idx]

    # Compute normalized vectors from A to each neighbor
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
        print(f"  Crossover node {a_idx}: zero-length vector, skipping.")
        continue

    # 2. Test all 3 pair combinations of the 4 neighbors
    b, c, d, e = neighbors
    pairings = [
        ((b, c), (d, e)),
        ((b, d), (c, e)),
        ((b, e), (c, d)),
    ]

    best_score = -float("inf")
    best_pairing = None
    for (p1, p2), (p3, p4) in pairings:
        # Dot product for each pair of vectors from A
        dot1 = _dot(vecs[p1], vecs[p2])  # want close to -1
        dot2 = _dot(vecs[p3], vecs[p4])  # want close to -1
        # Score: sum of both dots (most negative = best straight-line match)
        score = dot1 + dot2
        if score < best_score or best_pairing is None:
            best_score = score
            best_pairing = ((p1, p2), (p3, p4))

    (pair1_a, pair1_b), (pair2_a, pair2_b) = best_pairing
    dot1 = _dot(vecs[pair1_a], vecs[pair1_b])
    dot2 = _dot(vecs[pair2_a], vecs[pair2_b])
    print(f"  Crossover node {a_idx}: best pairing "
          f"({pair1_a}-{pair1_b} dot={dot1:.3f}, {pair2_a}-{pair2_b} dot={dot2:.3f})")

    # 3. Only dissolve if both dot products indicate roughly opposite directions
    DOT_THRESHOLD = -0.7
    if dot1 > DOT_THRESHOLD or dot2 > DOT_THRESHOLD:
        print(f"    Dot products not opposite enough (threshold {DOT_THRESHOLD}), skipping.")
        continue

    # 4. Dissolve: remove node A and rewire
    # Remove edges A-neighbor
    for n_idx in neighbors:
        adjacency[n_idx].discard(a_idx)
    del adjacency[a_idx]

    # Create new direct edges
    adjacency[pair1_a].add(pair1_b)
    adjacency[pair1_b].add(pair1_a)
    adjacency[pair2_a].add(pair2_b)
    adjacency[pair2_b].add(pair2_a)
    dissolved_count += 1
    print(f"    Dissolved! New edges: {pair1_a}-{pair1_b}, {pair2_a}-{pair2_b}")

print(f"  Dissolved {dissolved_count} crossover node(s)")

# Rebuild edge list after dissolution
edges = []
seen = set()
for i, nbrs in adjacency.items():
    for j in nbrs:
        edge = tuple(sorted([i, j]))
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)

num_components = _count_components(adjacency)

# Render post-dissolution adjacency graph
fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
drawn_edges = set()
for i, neighbors in adjacency.items():
    x1, y1 = all_nodes[i]
    for j in neighbors:
        edge = tuple(sorted([i, j]))
        if edge in drawn_edges:
            continue
        drawn_edges.add(edge)
        x2, y2 = all_nodes[j]
        ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.7)

# Draw only remaining (non-dissolved) nodes
remaining = sorted(adjacency.keys())
xs = [all_nodes[i][0] for i in remaining]
ys = [all_nodes[i][1] for i in remaining]
ax.scatter(xs, ys, s=150, c="red", zorder=5, edgecolors="black", linewidths=1.5)
for i in remaining:
    x, y = all_nodes[i]
    ax.annotate(str(i), (x, y), textcoords="offset points",
                xytext=(8, 8), fontsize=10, fontweight="bold")
# Mark dissolved crossover positions with an X
for a_idx in crossover_node_indices:
    if a_idx not in adjacency:
        cx, cy = all_nodes[a_idx]
        ax.scatter([cx], [cy], s=200, c="orange", marker="X", zorder=6,
                   edgecolors="black", linewidths=1.5)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"After Crossover Dissolution: {len(remaining)} nodes, {len(drawn_edges)} edges, "
             f"{num_components} components  (dissolved {dissolved_count})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step11_crossover_dissolved.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Step 12: Bézier curve fitting with deviation gatekeeper ──
print("Step 12: Fitting straight lines / Bézier curves to skeleton edges...")
from collections import deque

def _bfs_path_between_nodes(skel, nodes, node_i, node_j, radius):
    """BFS along skeleton pixels from node_i to node_j, returning the pixel path."""
    H, W = skel.shape
    disk = _disk_mask(radius)
    # Build set of pixels belonging to the target node j's disk
    target_disk = set()
    tx, ty = nodes[node_j]
    for dy, dx in disk:
        py, px = ty + dy, tx + dx
        if 0 <= py < H and 0 <= px < W:
            target_disk.add((py, px))

    # Start BFS from the border of node_i's disk
    sx, sy = nodes[node_i]
    start_disk = set()
    for dy, dx in disk:
        py, px = sy + dy, sx + dx
        if 0 <= py < H and 0 <= px < W:
            start_disk.add((py, px))

    visited = set(start_disk)
    # parent map for path reconstruction
    parent = {}
    queue = deque()

    # Seed: skeleton neighbors just outside node_i's disk
    for dy, dx in disk:
        py, px = sy + dy, sx + dx
        if 0 <= py < H and 0 <= px < W:
            for npy in range(py - 1, py + 2):
                for npx in range(px - 1, px + 2):
                    if npy == py and npx == px:
                        continue
                    if 0 <= npy < H and 0 <= npx < W:
                        if (npy, npx) not in visited and skel[npy, npx] > 0:
                            visited.add((npy, npx))
                            parent[(npy, npx)] = None  # start sentinel
                            queue.append((npy, npx))

    found = None
    while queue:
        cy, cx = queue.popleft()
        if (cy, cx) in target_disk:
            found = (cy, cx)
            break
        for npy in range(cy - 1, cy + 2):
            for npx in range(cx - 1, cx + 2):
                if npy == cy and npx == cx:
                    continue
                if 0 <= npy < H and 0 <= npx < W:
                    if (npy, npx) not in visited and skel[npy, npx] > 0:
                        visited.add((npy, npx))
                        parent[(npy, npx)] = (cy, cx)
                        queue.append((npy, npx))

    if found is None:
        return None

    # Reconstruct path (pixel coords as (x, y))
    path = []
    cur = found
    while cur is not None:
        cy, cx = cur
        path.append((cx, cy))
        cur = parent[cur]
    path.reverse()

    # Prepend node_i center and append node_j center
    path.insert(0, (sx, sy))
    path.append((tx, ty))
    return path


def _perpendicular_distance(px, py, x1, y1, x2, y2):
    """Perpendicular distance from point (px,py) to line through (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dx * (y1 - py) - dy * (x1 - px)) / math.sqrt(len_sq)


def _max_deviation(path, x1, y1, x2, y2):
    """Max perpendicular distance of path points to the straight line P_start-P_end."""
    max_d = 0.0
    for px, py in path:
        d = _perpendicular_distance(px, py, x1, y1, x2, y2)
        if d > max_d:
            max_d = d
    return max_d


def _fit_cubic_bezier(path):
    """
    Fit a cubic Bézier to the path using least squares.
    P0 and P3 are fixed to the first and last path points.
    Returns (P0, P1, P2, P3) as 2D tuples.
    """
    pts = np.array(path, dtype=np.float64)  # shape (N, 2)
    n = len(pts)
    P0 = pts[0]
    P3 = pts[-1]

    if n <= 2:
        # Degenerate: just return a straight-line Bézier
        P1 = P0 + (P3 - P0) / 3.0
        P2 = P0 + 2.0 * (P3 - P0) / 3.0
        return tuple(P0), tuple(P1), tuple(P2), tuple(P3)

    # Parameterise by cumulative chord length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 1e-9:
        P1 = P0 + (P3 - P0) / 3.0
        P2 = P0 + 2.0 * (P3 - P0) / 3.0
        return tuple(P0), tuple(P1), tuple(P2), tuple(P3)
    t = cum / total  # t[0]=0, t[-1]=1

    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    # Rearrange: pts - (1-t)^3 P0 - t^3 P3 = 3(1-t)^2 t P1 + 3(1-t) t^2 P2
    # Let b1(t) = 3(1-t)^2 t,  b2(t) = 3(1-t) t^2
    # System:  [b1, b2] @ [P1; P2] = rhs   (for each x,y independently)

    omt = 1.0 - t  # one minus t
    b1 = 3.0 * omt ** 2 * t          # shape (N,)
    b2 = 3.0 * omt * t ** 2          # shape (N,)

    # RHS = pts - (1-t)^3 * P0 - t^3 * P3
    rhs = pts - np.outer(omt ** 3, P0) - np.outer(t ** 3, P3)  # (N, 2)

    # Build matrix A (N x 2)
    A = np.column_stack([b1, b2])  # (N, 2)

    # Solve for [P1x, P2x] and [P1y, P2y] via least squares
    sol_x, _, _, _ = np.linalg.lstsq(A, rhs[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, rhs[:, 1], rcond=None)

    P1 = np.array([sol_x[0], sol_y[0]])
    P2 = np.array([sol_x[1], sol_y[1]])

    return tuple(P0), tuple(P1), tuple(P2), tuple(P3)


def _evaluate_bezier(P0, P1, P2, P3, num_points=50):
    """Evaluate a cubic Bézier at num_points values of t in [0,1]."""
    P0, P1, P2, P3 = [np.array(p) for p in [P0, P1, P2, P3]]
    ts = np.linspace(0, 1, num_points)
    omt = 1.0 - ts
    curve = (np.outer(omt ** 3, P0) +
             np.outer(3 * omt ** 2 * ts, P1) +
             np.outer(3 * omt * ts ** 2, P2) +
             np.outer(ts ** 3, P3))
    return curve  # shape (num_points, 2)


# Build the node_id_map needed for path tracing
node_id_map = np.full((IMG_H, IMG_W), -1, dtype=np.int32)
disk = _disk_mask(p["node_radius"])
for i, (nx, ny) in enumerate(all_nodes):
    for dy, dx in disk:
        py, px = ny + dy, nx + dx
        if 0 <= py < IMG_H and 0 <= px < IMG_W:
            node_id_map[py, px] = i

DEVIATION_THRESHOLD = 15  # pixels (increased so real lines aren't classified as curves)

# Collect edges from current adjacency (post-dissolution)
edge_list = []
seen_edges = set()
for i, nbrs in adjacency.items():
    for j in nbrs:
        e = tuple(sorted([i, j]))
        if e not in seen_edges:
            seen_edges.add(e)
            edge_list.append(e)

edge_curves = {}  # edge -> { "type": "line"|"bezier", "path": [...], ... }
straight_count = 0
bezier_count = 0
no_path_count = 0

for (i, j) in edge_list:
    path = _bfs_path_between_nodes(skel_img, all_nodes, i, j, p["node_radius"])
    if path is None or len(path) < 2:
        edge_curves[(i, j)] = {"type": "line", "path": None}
        no_path_count += 1
        continue

    x1, y1 = path[0]
    x2, y2 = path[-1]
    max_dev = _max_deviation(path, x1, y1, x2, y2)

    if max_dev < DEVIATION_THRESHOLD:
        edge_curves[(i, j)] = {"type": "line", "path": path, "max_dev": max_dev}
        straight_count += 1
    else:
        P0, P1, P2, P3 = _fit_cubic_bezier(path)
        edge_curves[(i, j)] = {
            "type": "bezier", "path": path,
            "max_dev": max_dev,
            "P0": P0, "P1": P1, "P2": P2, "P3": P3,
        }
        bezier_count += 1

print(f"  Edges: {len(edge_list)} total — "
      f"{straight_count} straight, {bezier_count} Bézier, {no_path_count} no-path")

# ── Step 12: Link line segments to component bounding boxes ──
print("\nStep 12: Linking line-segment endpoints to component bounding boxes...")
import math as _math
from matplotlib.lines import Line2D

LINK_DIST_THRESH = p["global_merge_dist"] + 3  # a little slack beyond the merge radius

node_to_components = {i: [] for i in range(len(all_nodes))}  # node_idx -> list of result indices

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
            d = _math.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
            if d < best_dist:
                best_dist = d
                best_node = n_idx
        if best_node != -1 and best_dist <= LINK_DIST_THRESH:
            if r_idx not in node_to_components[best_node]:
                node_to_components[best_node].append(r_idx)

print("\n  ── Node-to-Component Mapping ──")
for n_idx in sorted(node_to_components.keys()):
    comp_list = node_to_components[n_idx]
    if n_idx not in adjacency:
        continue
    nx, ny = all_nodes[n_idx]
    if comp_list:
        comp_strs = []
        for r_idx in comp_list:
            r = results[r_idx]
            cls_name = COMPONENT_NAMES[r["id"]] if r["id"] < len(COMPONENT_NAMES) else f"cls{r['id']}"
            comp_strs.append(f"{cls_name}(bbox={r['bbox_xyxy']})")
        print(f"  Node {n_idx} ({nx},{ny}) → {', '.join(comp_strs)}")
    else:
        print(f"  Node {n_idx} ({nx},{ny}) → [no linked component]")

print("\n  ── Edge-to-Component Linkages (Line ↔ BBox) ──")
edge_links = []
edges_final = []
seen_e = set()
for i, nbrs in adjacency.items():
    for j in nbrs:
        e = tuple(sorted([i, j]))
        if e not in seen_e:
            seen_e.add(e)
            edges_final.append(e)

for (i, j) in edges_final:
    comps_i = node_to_components.get(i, [])
    comps_j = node_to_components.get(j, [])

    def _comp_label(r_idx):
        r = results[r_idx]
        cls_name = COMPONENT_NAMES[r["id"]] if r["id"] < len(COMPONENT_NAMES) else f"cls{r['id']}"
        return f"{cls_name}(bbox={r['bbox_xyxy']})"

    label_i = ", ".join(_comp_label(ri) for ri in comps_i) if comps_i else "[wire junction]"
    label_j = ", ".join(_comp_label(rj) for rj in comps_j) if comps_j else "[wire junction]"

    curve_info = edge_curves.get((i, j), edge_curves.get((j, i), {}))
    curve_type = curve_info.get("type", "unknown")

    link_info = {
        "edge": (i, j),
        "node_i_comps": comps_i,
        "node_j_comps": comps_j,
        "curve_type": curve_type,
    }
    edge_links.append(link_info)
    print(f"  Edge {i}─{j} [{curve_type}]:  {label_i}  ↔  {label_j}")

print(f"\n  Total linked edges: {len(edge_links)}")

# 12c. Visualise: bounding boxes + line segments + labels
vis_link = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2BGR)
vis_link[skel_img > 0] = (200, 200, 200)

for r_idx, r in enumerate(results):
    x1, y1, x2, y2 = r["bbox_xyxy"]
    cls_name = COMPONENT_NAMES[r["id"]] if r["id"] < len(COMPONENT_NAMES) else f"cls{r['id']}"
    color = class_colors.get(r["id"], (128, 128, 128))
    cv2.rectangle(vis_link, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis_link, cls_name, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

for link in edge_links:
    i, j = link["edge"]
    ix, iy = all_nodes[i]
    jx, jy = all_nodes[j]
    cv2.line(vis_link, (ix, iy), (jx, jy), (255, 0, 0), 2)

for n_idx in sorted(adjacency.keys()):
    nx, ny = all_nodes[n_idx]
    comp_list = node_to_components[n_idx]
    cv2.circle(vis_link, (nx, ny), 6, (0, 0, 255), -1)
    if comp_list:
        cls_name = COMPONENT_NAMES[results[comp_list[0]]["id"]]
        label = f"N{n_idx}:{cls_name}"
    else:
        label = f"N{n_idx}"
    cv2.putText(vis_link, label, (nx + 8, ny - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 0), 1)

cv2.imwrite(os.path.join(OUT_DIR, "step12_segment_bbox_links.png"), vis_link)

# 12d. Matplotlib figure for linkage
fig, ax = plt.subplots(figsize=(14, 14), facecolor="white")

for r_idx, r in enumerate(results):
    x1, y1, x2, y2 = r["bbox_xyxy"]
    cls_name = COMPONENT_NAMES[r["id"]] if r["id"] < len(COMPONENT_NAMES) else f"cls{r['id']}"
    color_rgb = tuple(c / 255 for c in class_colors.get(r["id"], (128, 128, 128))[::-1])
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor=color_rgb, facecolor=(*color_rgb, 0.1))
    ax.add_patch(rect)
    ax.text(x1, y1 - 3, cls_name, fontsize=7, color=color_rgb, fontweight="bold")

for link in edge_links:
    i, j = link["edge"]
    ix, iy = all_nodes[i]
    jx, jy = all_nodes[j]
    has_i = len(link["node_i_comps"]) > 0
    has_j = len(link["node_j_comps"]) > 0
    if has_i and has_j:
        color = "limegreen"
    elif has_i or has_j:
        color = "orange"
    else:
        color = "gray"
    ax.plot([ix, jx], [iy, jy], "-", color=color, linewidth=2.5, alpha=0.8)
    mx, my = (ix + jx) / 2, (iy + jy) / 2
    ax.text(mx, my, f"{i}─{j}", fontsize=6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))

remaining = sorted(adjacency.keys())
for n_idx in remaining:
    nx, ny = all_nodes[n_idx]
    comp_list = node_to_components[n_idx]
    node_color = "red" if comp_list else "gray"
    ax.scatter([nx], [ny], s=120, c=node_color, zorder=7,
               edgecolors="black", linewidths=1.2)
    if comp_list:
        cls_name = COMPONENT_NAMES[results[comp_list[0]]["id"]]
        ax.annotate(f"N{n_idx}:{cls_name}", (nx, ny), textcoords="offset points",
                    xytext=(8, -10), fontsize=7, fontweight="bold", color="darkgreen")
    else:
        ax.annotate(f"N{n_idx}", (nx, ny), textcoords="offset points",
                    xytext=(8, -10), fontsize=7, color="gray")

ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Step 12 — Line Segments Linked to Component Bounding Boxes\n"
             f"{len(edge_links)} edges, {len(remaining)} nodes")
ax.grid(True, alpha=0.3)

legend_elems = [
    Line2D([0], [0], color="limegreen", linewidth=2.5, label="Both ends linked to components"),
    Line2D([0], [0], color="orange", linewidth=2.5, label="One end linked"),
    Line2D([0], [0], color="gray", linewidth=2.5, label="No component link (wire-to-wire)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Node with component"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Node without component"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step12_segment_bbox_links_graph.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Step 13 (FINAL): Bézier curve fitting with component linkage labels ──
print("\nStep 13 (FINAL): Bézier / straight-line fitting with component linkage...")

fig, ax = plt.subplots(figsize=(14, 14), facecolor="white")

# Draw bounding boxes
for r_idx, r in enumerate(results):
    x1, y1, x2, y2 = r["bbox_xyxy"]
    cls_name = COMPONENT_NAMES[r["id"]] if r["id"] < len(COMPONENT_NAMES) else f"cls{r['id']}"
    color_rgb = tuple(c / 255 for c in class_colors.get(r["id"], (128, 128, 128))[::-1])
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor=color_rgb, facecolor=(*color_rgb, 0.08))
    ax.add_patch(rect)
    ax.text(x1, y1 - 3, cls_name, fontsize=7, color=color_rgb, fontweight="bold")

# Draw edges with Bézier / straight distinction
for (i, j), info in edge_curves.items():
    xi, yi = all_nodes[i]
    xj, yj = all_nodes[j]

    if info["type"] == "line" or info["path"] is None:
        ax.plot([xi, xj], [yi, yj], "-", color="limegreen", linewidth=2, alpha=0.8)
    else:
        curve_pts = _evaluate_bezier(info["P0"], info["P1"], info["P2"], info["P3"],
                                     num_points=80)
        ax.plot(curve_pts[:, 0], curve_pts[:, 1], "-", color="dodgerblue",
                linewidth=2, alpha=0.8)
        cp = np.array([info["P0"], info["P1"], info["P2"], info["P3"]])
        ax.plot(cp[:, 0], cp[:, 1], ":", color="gray", linewidth=0.8, alpha=0.5)
        ax.scatter(cp[1:3, 0], cp[1:3, 1], s=30, c="orange", zorder=6,
                   marker="D", edgecolors="black", linewidths=0.5)

    # Midpoint edge label with component names
    mx, my = (xi + xj) / 2, (yi + yj) / 2
    ax.text(mx, my, f"{i}─{j}", fontsize=5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.7, edgecolor="none"))

# Draw nodes labelled with component names
remaining = sorted(adjacency.keys())
for n_idx in remaining:
    nx, ny = all_nodes[n_idx]
    comp_list = node_to_components[n_idx]
    node_color = "red" if comp_list else "gray"
    ax.scatter([nx], [ny], s=130, c=node_color, zorder=7,
               edgecolors="black", linewidths=1.3)
    if comp_list:
        cls_name = COMPONENT_NAMES[results[comp_list[0]]["id"]]
        ax.annotate(f"N{n_idx}:{cls_name}", (nx, ny), textcoords="offset points",
                    xytext=(8, -10), fontsize=7, fontweight="bold", color="darkgreen")
    else:
        ax.annotate(f"N{n_idx}", (nx, ny), textcoords="offset points",
                    xytext=(8, -10), fontsize=7, color="gray")

ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Step 13 (FINAL) — Bézier Fitting + Component Linkage\n"
             f"{straight_count} straight (green), {bezier_count} Bézier (blue)  "
             f"[threshold={DEVIATION_THRESHOLD}px]  |  {len(remaining)} nodes")
ax.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0], [0], color="limegreen", linewidth=2, label="Straight line"),
    Line2D([0], [0], color="dodgerblue", linewidth=2, label="Bézier curve"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="orange",
           markersize=6, label="Control point (P1/P2)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
           markersize=8, label="Node with component"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=8, label="Node without component"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step13_final_bezier_with_links.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nDone! All step images saved to: {OUT_DIR}")
print("Files:")
for fname in sorted(os.listdir(OUT_DIR)):
    print(f"  {fname}")
