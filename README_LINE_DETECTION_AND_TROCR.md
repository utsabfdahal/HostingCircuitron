# Line Detection & TrOCR — Technical Reference

> Detailed documentation covering every stage of the skeleton-based wire/line
> detection pipeline and the TrOCR / CRNN optical-character-recognition
> subsystems used in CIRCUITRON.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Line Detection Pipeline](#2-line-detection-pipeline)
   - 2.1 [Entry Point & Parameters](#21-entry-point--parameters)
   - 2.2 [Stage 1 — Binary Thresholding & Skeletonization](#22-stage-1--binary-thresholding--skeletonization)
   - 2.3 [Stage 2 — Text-Region Erasure](#23-stage-2--text-region-erasure)
   - 2.4 [Stage 3 — Dilation & Re-Skeletonization](#24-stage-3--dilation--re-skeletonization)
   - 2.5 [Stage 4 — Endpoint Detection](#25-stage-4--endpoint-detection)
   - 2.6 [Stage 5 — Global Endpoint Merging](#26-stage-5--global-endpoint-merging)
   - 2.7 [Stage 6 — Adjacency Graph via Multi-Head BFS](#27-stage-6--adjacency-graph-via-multi-head-bfs)
   - 2.8 [Stage 7 — Remove Internal Component Connections](#28-stage-7--remove-internal-component-connections)
   - 2.9 [Stage 8 — Crossover Dissolution](#29-stage-8--crossover-dissolution)
   - 2.10 [Stage 9 — Component-to-Node Linkage](#210-stage-9--component-to-node-linkage)
   - 2.11 [Output Format](#211-output-format)
3. [Bézier Curve Extension](#3-bézier-curve-extension)
   - 3.1 [BFS Path Tracing Between Nodes](#31-bfs-path-tracing-between-nodes)
   - 3.2 [Deviation Analysis](#32-deviation-analysis)
   - 3.3 [Cubic Bézier Fitting (Least-Squares)](#33-cubic-bézier-fitting-least-squares)
   - 3.4 [Bézier Evaluation](#34-bézier-evaluation)
4. [TrOCR System](#4-trocr-system)
   - 4.1 [Model Architecture & Checkpoint](#41-model-architecture--checkpoint)
   - 4.2 [OCRService Class](#42-ocrservice-class)
   - 4.3 [Single-Image Recognition](#43-single-image-recognition)
   - 4.4 [Batch Recognition](#44-batch-recognition)
   - 4.5 [Full Extraction Pipeline](#45-full-extraction-pipeline)
   - 4.6 [Confidence Scoring](#46-confidence-scoring)
5. [Custom CRNN (Fast Alternative)](#5-custom-crnn-fast-alternative)
   - 5.1 [Architecture](#51-architecture)
   - 5.2 [Character Set](#52-character-set)
   - 5.3 [VGG Feature Extractor](#53-vgg-feature-extractor)
   - 5.4 [Bidirectional LSTM](#54-bidirectional-lstm)
   - 5.5 [CTC Greedy Decoder](#55-ctc-greedy-decoder)
   - 5.6 [Image Preprocessing](#56-image-preprocessing)
   - 5.7 [CustomOCRService Class](#57-customocrservice-class)
6. [OCR Mode Selection](#6-ocr-mode-selection)
7. [Proximity Mapping (Text → Component)](#7-proximity-mapping-text--component)
8. [Orchestration — Unified Pipeline](#8-orchestration--unified-pipeline)
9. [Standalone Test Scripts](#9-standalone-test-scripts)
10. [Training Notebooks](#10-training-notebooks)
11. [File Reference](#11-file-reference)
12. [Dependencies](#12-dependencies)
13. [Tunable Parameters & Constants](#13-tunable-parameters--constants)

---

## 1. Architecture Overview

The full analysis flow for a hand-drawn circuit image is:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Raw Image  │────▶│  YOLO v7     │────▶│  15-class         │
│  (bytes)    │     │  Detection   │     │  bounding boxes   │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                          ┌────────────────────────┼─────────────────┐
                          │                        │                 │
                    ┌─────▼──────┐          ┌──────▼──────┐   ┌──────▼──────┐
                    │  Text      │          │  Components │   │  Junctions  │
                    │  Boxes     │          │  (R,C,L,…)  │   │  Crossovers │
                    │  (cls 11)  │          │             │   │  Terminals  │
                    └─────┬──────┘          └──────┬──────┘   └──────┬──────┘
                          │                        │                 │
                    ┌─────▼──────┐          ┌──────▼──────┐          │
                    │  TrOCR /   │          │  Proximity  │          │
                    │  CRNN OCR  │─────────▶│  Mapper     │          │
                    │            │          │  (text→comp)│          │
                    └────────────┘          └──────┬──────┘          │
                                                   │                 │
                                            ┌──────▼─────────────────▼──────┐
                                            │  Line Detection Pipeline      │
                                            │  (Skeleton + Multi-Head BFS)  │
                                            │                               │
                                            │  Binary Threshold             │
                                            │  → Skeletonize                │
                                            │  → Erase Text Regions         │
                                            │  → Dilate + Re-Skeletonize    │
                                            │  → Endpoint Detection         │
                                            │  → Global Merge               │
                                            │  → Multi-Head BFS Graph       │
                                            │  → Remove Internal Edges      │
                                            │  → Dissolve Crossovers        │
                                            │  → Component-to-Node Map      │
                                            └──────┬────────────────────────┘
                                                   │
                                            ┌──────▼──────┐
                                            │  Circuit    │
                                            │  JSON       │
                                            │  (nodes,    │
                                            │  edges,     │
                                            │  components)│
                                            └─────────────┘
```

**YOLO classes used by line detection:**

| Class | Name                 | Role in Line Detection                           |
|-------|----------------------|--------------------------------------------------|
| 0     | capacitor            | Border intersection endpoints                    |
| 1     | crossover            | Center snap endpoint + crossover dissolution      |
| 2     | diode                | Border intersection endpoints                    |
| 3     | gnd                  | Border intersection endpoints                    |
| 4     | inductor             | Border intersection endpoints                    |
| 5     | integrated_circuit   | Border intersection endpoints                    |
| 6     | junction             | Center snap endpoint                             |
| 7     | operational_amplifier| Border intersection endpoints                    |
| 8     | resistor             | Border intersection endpoints                    |
| 9     | switch               | Border intersection endpoints                    |
| 10    | terminal             | Center snap endpoint                             |
| 11    | text                 | Erased from skeleton before line detection        |
| 12    | transistor           | Border intersection endpoints                    |
| 13    | voltage              | Border intersection endpoints                    |
| 14    | vss                  | Border intersection endpoints                    |

---

## 2. Line Detection Pipeline

### 2.1 Entry Point & Parameters

**File:** `test/pipeline.py`  
**Function:** `analyze(image_bytes, label_text, params=None)`

```
Parameters
----------
image_bytes  : bytes     – Raw image file content (JPEG / PNG)
label_text   : str       – YOLO output in normalized format:
                           "cls xc yc w h confidence\n..."
params       : dict      – Override any of DEFAULT_PARAMS (see below)

Returns
-------
dict with keys:
  "detections" : list     – Raw YOLO bounding boxes
  "results"    : list     – Enriched with endpoint data
  "graph"      : dict     – {"nodes": [...], "edges": [...], "num_components": int}
  "images"     : dict     – Base64-encoded diagnostic PNGs
```

**Default parameters** (all in `DEFAULT_PARAMS`):

| Parameter              | Default | Purpose                                                    |
|------------------------|---------|------------------------------------------------------------|
| `binary_thresh`        | 110     | Grayscale threshold for binarization                       |
| `skel_thresh`          | 128     | Skeleton threshold (reserved, currently unused)            |
| `node_radius`          | 25      | Pixel radius of disk around each node for BFS seeding      |
| `cls1_margin`          | 8       | Extra margin (px) when searching for junction/crossover/terminal endpoints |
| `border_tol`           | 1       | Pixel tolerance for the annular band at bbox borders       |
| `global_merge_dist`    | 5.0     | Merge endpoints within this many pixels into a single node |
| `find_node_dist_thresh`| 5       | Max distance to match an endpoint to an existing node      |

---

### 2.2 Stage 1 — Binary Thresholding & Skeletonization

The raw image is converted to a 1-pixel-wide skeleton that represents wire paths.

**Algorithm:**

1. **Median blur** (kernel size 1) — light denoising without destroying thin lines.
2. **Binary threshold** at `BINARY_THRESH` (default 110) — pixels below 110 become foreground (ink), above become background.
3. **Bit inversion** — scikit-image's `skeletonize()` expects foreground = True.
4. **Morphological skeletonization** — Zhang–Suen thinning via `skimage.morphology.skeletonize()`.

```python
denoised    = cv2.medianBlur(img_original, 1)
_, simple_bin = cv2.threshold(denoised, BINARY_THRESH, 255, cv2.THRESH_BINARY)
inverted    = cv2.bitwise_not(simple_bin)
skeleton    = skeletonize(inverted > 0)       # Boolean ndarray, True = wire
```

**Output:** Boolean 2D array where `True` pixels lie on thinned wire centerlines.

---

### 2.3 Stage 2 — Text-Region Erasure

Before building the wire graph, all text-class (class 11) bounding boxes are blanked out on the skeleton. This prevents OCR text pixels from being mistaken for wires.

```python
for d in detections_with_bbox:
    if d["cls"] == 11:      # text
        x1, y1, x2, y2 = d["bbox_xyxy"]
        skeleton[y1:y2, x1:x2] = False
```

This operates in-place on the boolean skeleton array.

---

### 2.4 Stage 3 — Dilation & Re-Skeletonization

After text erasure, small gaps may appear where text overlapped wires. A single pass of morphological dilation fills micro-gaps, then re-skeletonization restores the 1-pixel-wide skeleton.

```python
img_u8  = skeleton.astype(np.uint8) * 255
kernel  = np.ones((3, 3), np.uint8)
dilated = cv2.bitwise_not(cv2.dilate(img_u8, kernel, iterations=1))

binary_fg = (dilated < 128).astype(np.uint8)
skel      = skeletonize(binary_fg.astype(bool)).astype(np.uint8)
skel_u8   = img_as_ubyte(skel)
```

- **Dilation kernel:** 3×3 square, 1 iteration
- **Result:** Clean 1-pixel skeleton with text artifacts removed

---

### 2.5 Stage 4 — Endpoint Detection

Two distinct strategies are used depending on the YOLO class of the detection.

#### Strategy A: Center-snap (classes 1, 6, 10 — crossover, junction, terminal)

These components represent wire meeting points. The endpoint is the skeleton pixel closest to the bounding-box center.

**Function:** `_endpoint_for_class1_single(skel, x1, y1, x2, y2, margin=8)`

```
1. Compute bounding-box center: (xc, yc)
2. Expand search region by `margin` pixels on all sides
3. Find all skeleton pixels within the expanded region
4. Return the one nearest (Euclidean) to (xc, yc)
```

**Output:** A single `(x, y)` tuple — the junction's wire-attachment point.

#### Strategy B: Border intersections (all other component classes)

Components like resistors, capacitors, etc. have two or more leads that cross the bounding-box border. The algorithm finds where the skeleton intersects the bbox perimeter.

**Function:** `_skeleton_intersections_with_bbox_border(skel, x1, y1, x2, y2, tol=1)`

```
1. Create an annular binary mask along the bbox border (width = tol pixels)
2. Inner rectangle (bbox shrunk by tol) is excluded → only the border ring remains
3. Extract all skeleton pixels that fall inside this ring
4. Return as an (N, 2) array of (x, y) coordinates
```

These raw border-hit pixels are then clustered:

**Function:** `_get_all_border_intersections(skel, inter_xy, merge_dist=8.0)`

```
1. For each raw point, check if it's within merge_dist of any existing cluster
2. If yes → average into that cluster's centroid
3. If no  → start a new cluster
4. Return list of merged (x, y) centroids — these are the component's terminals
```

**Output:** List of 2+ `(x, y)` tuples representing where the component's leads connect to wires.

---

### 2.6 Stage 5 — Global Endpoint Merging

All endpoints from all detections are collected into a single global list. Nearby points (from different components but physically the same junction) are merged.

**Function:** `_extract_all_endpoints(results, global_merge_dist=5.0)`

```
1. Collect every endpoint_xy and every entry in endpoints[] from all results
2. For each point:
   a. Compute distances to all existing unique points (vectorized via NumPy)
   b. If min distance > global_merge_dist → add as new unique node
   c. Otherwise → discard (already represented by an existing node)
3. Return list of unique (x, y) node positions
```

**Output:** `all_nodes` — the final list of wire junction / endpoint coordinates. Each becomes a graph vertex.

---

### 2.7 Stage 6 — Adjacency Graph via Multi-Head BFS

This is the core algorithm that determines which nodes are connected by wires.

#### Step 1: Build Node ID Map

**Function:** `_build_node_id_map(skel, nodes, radius=25)`

A full-resolution integer array (`node_id_map`) is created, initialized to `-1`. For each node, all pixels within a circular disk of radius `radius` are stamped with that node's index.

```python
node_id_map = np.full((H, W), -1, dtype=np.int32)
for i, (nx, ny) in enumerate(nodes):
    for dy, dx in disk_offsets:          # precomputed circle of radius 25
        py, px = ny + dy, nx + dx
        if 0 <= py < H and 0 <= px < W:
            node_id_map[py, px] = i
```

This map lets BFS quickly determine "which node am I near?" in O(1) per pixel.

#### Step 2: Disk Mask Generation

**Function:** `_disk_mask(radius)`

Pre-computes all `(dy, dx)` offsets within a circle:

```python
offsets = []
for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
        if dy*dy + dx*dx <= radius*radius:
            offsets.append((dy, dx))
```

For `radius=25`, this produces ~1,963 offset pairs.

#### Step 3: BFS from Each Node

**Function:** `_bfs_neighbors_for_node(node_idx, nodes, skel, node_id_map, radius)`

```
1. Mark all pixels in node_idx's disk as "visited"
2. Seed the BFS queue with skeleton pixels just outside the disk boundary
   (i.e., 8-connected neighbors of disk pixels that lie on the skeleton)
3. BFS loop:
   a. Pop pixel (cy, cx) from queue
   b. Check node_id_map[cy, cx]:
      - If it's another node → record that node as a neighbor, DON'T expand further
      - If it's -1 → expand to all 8-connected skeleton neighbors not yet visited
4. Return set of discovered neighbor node indices
```

**Connectivity:** 8-way (includes diagonal neighbors) — essential for following diagonal wire paths in the skeleton.

#### Step 4: Symmetrize

```python
for i in range(len(nodes)):
    neighbors = _bfs_neighbors_for_node(i, ...)
    for j in neighbors:
        adjacency[i].add(j)
        adjacency[j].add(i)         # Ensure symmetry
```

**Output:** `adjacency: dict[int, set[int]]` — the wire-connectivity graph.

---

### 2.8 Stage 7 — Remove Internal Component Connections

Multi-terminal components (e.g., a resistor with both leads detected) will have both endpoints as separate nodes — but the wire between them runs *through* the component body, not around it. These false edges are removed.

```python
for r in results:
    if "endpoints" in r and len(r.get("endpoints", [])) >= 2:
        for (p1, p2) in combinations(r["endpoints"], 2):
            i1 = _find_node_index(p1, all_nodes, FIND_NODE_DIST)
            i2 = _find_node_index(p2, all_nodes, FIND_NODE_DIST)
            if i1 != -1 and i2 != -1:
                adjacency[i1].discard(i2)
                adjacency[i2].discard(i1)
```

`_find_node_index` returns the closest node within `find_node_dist_thresh` pixels, or `-1` if none match.

---

### 2.9 Stage 8 — Crossover Dissolution

**Problem:** A crossover (class 1) is where two wires visually cross without being electrically connected. The skeleton produces a single junction point with 4 neighbors. Naively, this creates a 4-way connection — incorrect.

**Function:** `_dissolve_crossovers(results, all_nodes, adjacency, find_dist)`

**Algorithm — Dot-Product Pairing:**

```
1. Identify crossover nodes: class 1 detections whose endpoint matches a graph node
2. Filter to those with exactly 4 neighbors
3. For each crossover node A with neighbors {B, C, D, E}:
   a. Compute normalized direction vectors: A→B, A→C, A→D, A→E
   b. Enumerate all 3 possible pairings of 4 neighbors into 2 pairs:
      • (B,C) + (D,E)
      • (B,D) + (C,E)
      • (B,E) + (C,D)
   c. For each pairing, score = dot(v_p1, v_p2) + dot(v_p3, v_p4)
      The best pairing minimizes the score (most opposite = dot ≈ -1)
   d. Validate: both pair dot products must be < DOT_THRESHOLD (-0.7)
4. Dissolve:
   a. Remove crossover node A and all its edges
   b. Wire pair1's two neighbors directly together
   c. Wire pair2's two neighbors directly together
```

**Geometric intuition:** Two wires crossing form an "X". The opposite arms of the X produce dot products near −1, while adjacent arms produce values near 0 or +1. The algorithm picks the pairing where both pairs are nearly antiparallel.

**Constants:**
- `DOT_THRESHOLD = -0.7` — minimum opposition required; values closer to 0 are rejected as ambiguous

**Helper functions:**
- `_normalized_vector(ax, ay, bx, by)` — returns unit vector from (ax,ay) to (bx,by), or `None` if magnitude < 1e-9
- `_dot(v1, v2)` — 2D dot product

---

### 2.10 Stage 9 — Component-to-Node Linkage

**Function:** `_build_node_component_map(results, all_nodes, adjacency, merge_dist)`

Maps each graph node to the component(s) whose endpoints fall within `link_dist = merge_dist + 3` pixels (default: 5 + 3 = 8 px).

```
For each detection result:
    For each of its endpoints:
        Find the nearest graph node within link_dist
        Add this result's index to that node's component list
```

**Output:** `node_to_components: dict[int, list[int]]` — node index → list of result indices.

This mapping is used downstream to build the `Connection` records for the frontend (component terminal → node).

---

### 2.11 Output Format

The `analyze()` function returns a dict with these top-level keys:

#### `detections` — Raw YOLO Boxes

```json
[
    {
        "cls": 8,
        "name": "resistor",
        "conf": 0.92,
        "bbox": [120, 45, 210, 95]
    }
]
```

#### `results` — Enriched with Endpoints

```json
[
    {
        "cls": 8,
        "name": "resistor",
        "conf": 0.92,
        "bbox": [120, 45, 210, 95],
        "method": "all_border_hits_cleaned",
        "endpoints": [[118, 70], [212, 70]]
    },
    {
        "cls": 1,
        "name": "crossover",
        "conf": 0.87,
        "bbox": [300, 200, 320, 220],
        "method": "cls1_center_snap",
        "endpoint_xy": [310, 210]
    }
]
```

**Method values:**
- `"cls1_center_snap"` — junction/crossover/terminal found via center-nearest
- `"all_border_hits_cleaned"` — standard border intersection + merge
- `"fallback_internal"` — border search failed, fell back to internal skeleton point

#### `graph` — Wire Connectivity

```json
{
    "nodes": [
        {"id": 0, "x": 118, "y": 70},
        {"id": 1, "x": 212, "y": 70},
        {"id": 2, "x": 310, "y": 210}
    ],
    "edges": [
        {
            "source": 0,
            "target": 1,
            "linked_components": {
                "source_components": [{"cls": 8, "name": "resistor", "bbox": [...]}],
                "target_components": [{"cls": 0, "name": "capacitor", "bbox": [...]}]
            }
        }
    ],
    "num_components": 1
}
```

`num_components` is the number of connected components (graph-theory sense) — a fully connected circuit has `num_components = 1`.

#### `images` — Diagnostic Visualizations (Base64 PNG)

```json
{
    "original": "data:image/png;base64,...",
    "skeleton": "data:image/png;base64,...",
    "endpoints": "data:image/png;base64,...",
    "graph": "data:image/png;base64,..."
}
```

---

## 3. Bézier Curve Extension

**File:** `test/pipelinewithbeizer.py`

An extended version of the line detection pipeline that fits cubic Bézier curves to wire edges for smoother visual rendering.

### 3.1 BFS Path Tracing Between Nodes

**Function:** `_bfs_path_between_nodes(skel, nodes, node_i, node_j, radius)`

For each edge in the adjacency graph, traces the exact skeleton pixel path connecting the two endpoint nodes.

```
1. Build a disk around node_i (start) and node_j (target)
2. Mark start disk as visited
3. Seed BFS with skeleton pixels just outside the start disk
4. BFS along skeleton with 8-connectivity, recording parent pointers
5. Stop when a pixel inside the target disk is reached
6. Reconstruct path via parent chain
7. Prepend node_i center, append node_j center
```

**Output:** Ordered list of `(x, y)` pixel coordinates tracing the wire from node_i to node_j.

### 3.2 Deviation Analysis

**Function:** `_max_deviation(path, x1, y1, x2, y2)`

Measures the maximum perpendicular distance of any path point from the straight line connecting the two endpoints.

```
For each point (px, py) in path:
    d = |dx*(y1-py) - dy*(x1-px)| / sqrt(dx² + dy²)
    max_d = max(max_d, d)
```

Where `dx = x2 - x1`, `dy = y2 - y1`.

**Classification threshold:**

```python
DEVIATION_THRESHOLD = 15    # pixels

if max_dev < DEVIATION_THRESHOLD:
    → classify as "line" (straight wire)
else:
    → classify as "curve" → fit Bézier
```

### 3.3 Cubic Bézier Fitting (Least-Squares)

**Function:** `_fit_cubic_bezier(path)`

Fits a cubic Bézier curve `B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃` to the traced pixel path using least-squares optimization.

**Algorithm:**

```
1. Fix P₀ = path[0] (start) and P₃ = path[-1] (end)
2. Parameterize by cumulative chord length:
   t[i] = cumulative_distance[i] / total_distance
   → t ∈ [0, 1]
3. Compute Bézier basis functions:
   b₁(t) = 3(1-t)²t    (influence of P₁)
   b₂(t) = 3(1-t)t²    (influence of P₂)
4. Form the right-hand side:
   rhs = path - (1-t)³·P₀ - t³·P₃
5. Build design matrix A = [b₁, b₂]   shape (N, 2)
6. Solve for P₁ and P₂ separately in x and y via np.linalg.lstsq:
   A · [P₁ₓ, P₂ₓ]ᵀ = rhsₓ
   A · [P₁ᵧ, P₂ᵧ]ᵀ = rhsᵧ
```

**Degenerate cases:** If the path has ≤ 2 points or total chord length < 1e-9, control points are placed at the 1/3 and 2/3 positions along the straight line.

**Output:** Tuple `(P₀, P₁, P₂, P₃)` — four 2D control points defining the cubic Bézier.

### 3.4 Bézier Evaluation

**Function:** `_evaluate_bezier(P0, P1, P2, P3, num_points=50)`

Generates `num_points` sample coordinates along the fitted curve:

```python
ts = np.linspace(0, 1, num_points)
omt = 1.0 - ts
curve = (omt³ · P₀) + (3·omt²·t · P₁) + (3·omt·t² · P₂) + (t³ · P₃)
```

**Output:** `(num_points, 2)` ndarray of sample `(x, y)` positions — used for rendering smooth curves on the schematic.

---

## 4. TrOCR System

### 4.1 Model Architecture & Checkpoint

| Property              | Value                                                          |
|-----------------------|----------------------------------------------------------------|
| Architecture          | Vision Encoder-Decoder (ViT encoder + GPT-2 decoder)          |
| Base model            | `microsoft/trocr-small-printed`                                |
| Fine-tuned checkpoint | `OCRmodel/trocrfinetuned/checkpoint-epoch-2/`                  |
| Checkpoint files      | `config.json`, `generation_config.json`, `model.safetensors`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json` |
| Base model files      | `config.json`, `generation_config.json`, `preprocessor_config.json`, `sentencepiece.bpe.model`, `special_tokens_map.json`, `tokenizer_config.json` |
| Max generated tokens  | 16                                                             |
| Precision             | float16 on CUDA, float32 on CPU                               |
| Device                | Auto-detected (CUDA preferred, CPU fallback)                   |

**File:** `test/ocr_service.py`

### 4.2 OCRService Class

```python
class OCRService:
    def __init__(self, model_id: str, device: Optional[str] = None)
```

Key design decisions:
- **Lazy loading:** Model is not loaded until the first `recognize()` / `extract_texts()` call. This avoids startup delay if OCR is never needed.
- **Singleton pattern:** `get_ocr_service()` returns a module-level singleton so the model is only loaded once per process.
- **Half precision:** On CUDA devices, `model.half()` is called for 2× inference speed.
- **Eval mode:** `model.eval()` disables dropout and batch-norm updates.

```python
_instance: Optional[OCRService] = None

def get_ocr_service(model_id: str = _DEFAULT_MODEL_ID) -> OCRService:
    global _instance
    if _instance is None:
        _instance = OCRService(model_id=model_id)
    return _instance
```

### 4.3 Single-Image Recognition

**Method:** `recognize(pil_image: Image.Image) -> (str, float)`

```
1. Preprocess image via TrOCRProcessor → pixel_values tensor
2. Cast to float16 if on CUDA
3. Generate tokens with max_new_tokens=16, output_scores=True
4. Decode token IDs → text string
5. Compute confidence from mean token log-probability (see §4.6)

Returns: (recognized_text, confidence_score)
```

### 4.4 Batch Recognition

**Method:** `_recognise_batch(pil_images: List[Image.Image]) -> List[(str, float)]`

Processes multiple images in a single forward pass for efficiency:

```
1. Processor handles dynamic padding across variable-width images
2. Single generate() call for entire batch
3. batch_decode() produces all text strings at once
4. Per-sample confidence computed by iterating output.scores per batch index
```

This is significantly faster than calling `recognize()` in a loop because:
- Single GPU kernel launch for the encoder
- Batched attention in the decoder
- No Python loop overhead per image

### 4.5 Full Extraction Pipeline

**Method:** `extract_texts(image_bgr, text_boxes) -> List[Dict]`

The main entry point called by the unified pipeline:

```
1. For each text_box:
   a. Clamp bbox to image dimensions
   b. Crop the BGR sub-image
   c. Convert BGR → RGB → PIL Image
   d. Add to batch list
2. Run _recognise_batch() on all valid crops (single forward pass)
3. Return list of dicts, each containing:
   - All original text_box fields
   - "ocr_text": recognized string
   - "ocr_confidence": float [0, 1]
```

Crops with zero area are handled gracefully (text="" , confidence=0.0).

### 4.6 Confidence Scoring

TrOCR confidence is derived from the mean token-level log-probability:

```
1. For each generated token (excluding <bos> and <eos>):
   a. Apply log_softmax to the vocab logits
   b. Select the log-prob of the chosen token ID
2. Average all selected log-probs
3. Exponentiate: confidence = exp(mean_log_prob)
```

This gives a value in `[0, 1]` where:
- **~0.95+** = very confident (clear, well-formed text)
- **~0.5–0.8** = moderate confidence (smudged, small, or unusual text)
- **< 0.3** = low confidence (likely misread)

---

## 5. Custom CRNN (Fast Alternative)

**File:** `test/custom_ocr.py`  
**Weights:** `OCRmodel/crnn_last.pth`

A lightweight CNN-BiLSTM-CTC model that runs 5–10× faster than TrOCR, trading some accuracy for speed.

### 5.1 Architecture

```
Input: (B, 1, 32, W)  — Grayscale, height normalized to 32, variable width
                ↓
┌──────────────────────────────┐
│  VGG Feature Extractor       │
│  Conv2d(1→32) + ReLU + Pool  │
│  Conv2d(32→64) + ReLU + Pool │
│  Conv2d(64→128)×2 + Pool     │
│  Conv2d(128→256)×2 + Pool    │
│  Conv2d(256→256) + ReLU      │
│  Output: (B, 256, H/16, W/4) │ (for VGG with output_channel=256)
└──────────────┬───────────────┘
               ↓  (or 512 channels with output_channel=512)
┌──────────────────────────────┐
│  AdaptiveAvgPool2d           │
│  Collapse height → 1         │
│  Output: (B, 512, 1, W/4)   │
│  → squeeze → (B, W/4, 512)  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  BiLSTM Layer 1              │
│  Input: 512, Hidden: 256     │
│  Output: (B, W/4, 256)       │
├──────────────────────────────┤
│  BiLSTM Layer 2              │
│  Input: 256, Hidden: 256     │
│  Output: (B, W/4, 256)       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  Linear (256 → 91)           │
│  91 = 90 chars + 1 CTC blank │
│  Output: (W/4, B, 91)        │
│  (permuted for CTC loss)     │
└──────────────────────────────┘
```

### 5.2 Character Set

90 printable characters plus the CTC blank token (index 0):

```
!"#$%&()*+,-./0123456789:<=>?@
ABCDEFGHIJKLMNOPQRSTUVWXYZ^_`
abcdefghijklmnopqrstuvwxyz~§µ×ßäöüΩ
```

Notable inclusions for electronics:
- `Ω` — ohm symbol
- `µ` — micro prefix
- `×` — multiplication sign
- Upper/lowercase letters for unit prefixes (k, M, m, p, n, etc.)

### 5.3 VGG Feature Extractor

**Class:** `VGG_FeatureExtractor(input_channel=1, output_channel=512)`

A 5-block VGG-style CNN that progressively downsamples height by 16× and width by 4×:

| Block | Layers                              | Output Channels | Spatial Change      |
|-------|-------------------------------------|-----------------|---------------------|
| 1     | Conv3×3 + ReLU + MaxPool(2,2)       | 64              | H/2 × W/2          |
| 2     | Conv3×3 + ReLU + MaxPool(2,2)       | 128             | H/4 × W/4          |
| 3     | Conv3×3×2 + ReLU + MaxPool(2,1)(2,1)| 256             | H/8 × W/4          |
| 4     | Conv3×3×2 + BN + ReLU + MaxPool(2,1)(2,1) | 512      | H/16 × W/4         |
| 5     | Conv2×2 + ReLU                      | 512             | (H/16−1) × (W/4−1) |

**Key design:** Blocks 3 and 4 use asymmetric pooling `(2,1)` — they reduce height but preserve width. This is critical for OCR because the width dimension corresponds to the character sequence axis.

### 5.4 Bidirectional LSTM

**Class:** `BidirectionalLSTM(input_size, hidden_size, output_size)`

Two stacked BiLSTM layers model left-to-right and right-to-left context:

```python
self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
self.linear = nn.Linear(hidden_size * 2, output_size)
```

- Layer 1: 512 → BiLSTM(256) → 256
- Layer 2: 256 → BiLSTM(256) → 256

The forward+backward hidden states are concatenated (512) then projected back to `output_size` via a linear layer.

### 5.5 CTC Greedy Decoder

**Function:** `_greedy_decode_with_confidence(output: torch.Tensor) -> List[(str, float)]`

Converts the CTC output probability matrix into text:

```
1. Apply softmax along the class dimension → (T, B, C) probabilities
2. Take argmax → predicted class per timestep
3. Collapse repeated characters: "aaa_bb_c__" → "abc"
   (CTC blank = index 0, repeated non-blank = collapse)
4. Track max probability of each selected character
5. Confidence = mean of selected character probabilities
```

**CTC blank token:** Index 0. The model outputs blank between characters and for "no character" timesteps. Consecutive identical non-blank predictions are collapsed into one character.

### 5.6 Image Preprocessing

**Class:** `_ResizeKeepAspectRatio(height=32)`

Resizes the input image to a fixed height of 32 pixels while preserving the aspect ratio (variable width). This is critical because:
- Component value text can be very short ("5V") or long ("100kΩ")
- Fixed-width resizing would squash/stretch characters

**Full transform pipeline:**

```python
transforms.Compose([
    _ResizeKeepAspectRatio(32),           # → (32, proportional_width)
    transforms.ToTensor(),                # → [0, 1] float tensor
    transforms.Normalize((0.5,), (0.5,)), # → [-1, 1] range
])
```

**For BGR crops** from OpenCV:

```python
def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
    tensor = _transform(pil_img)
    return tensor.unsqueeze(0)            # → (1, 1, 32, W)
```

### 5.7 CustomOCRService Class

**Class:** `CustomOCRService(model_path, device)`

Mirrors the `OCRService` API for drop-in replacement:

- **`_ensure_loaded()`** — Loads CRNN weights from `OCRmodel/crnn_last.pth` via `torch.load(weights_only=True)`, sets eval mode
- **`recognise(pil_image)`** — Single-image: grayscale → transform → forward → CTC decode → `(text, confidence)`
- **`extract_texts(image_bgr, text_boxes)`** — Per-crop inference (no batching because variable-width images require per-sample processing)

**Singleton:**

```python
_custom_instance: Optional[CustomOCRService] = None

def get_custom_ocr_service(model_path: str = _DEFAULT_MODEL_PATH) -> CustomOCRService:
    global _custom_instance
    if _custom_instance is None:
        _custom_instance = CustomOCRService(model_path=model_path)
    return _custom_instance
```

---

## 6. OCR Mode Selection

**File:** `test/unified_pipeline.py`, function `_get_ocr()`

The pipeline supports two OCR backends, selected by a string parameter:

| Mode     | Backend           | Model                                    | Speed   | Accuracy  |
|----------|-------------------|------------------------------------------|---------|-----------|
| `"fast"` | Custom CRNN       | `OCRmodel/crnn_last.pth`                 | 5–10×   | Good      |
| `"slow"` | TrOCR Transformer | `OCRmodel/trocrfinetuned/checkpoint-epoch-2/` | 1×  | Best      |

```python
def _get_ocr(ocr_mode: str = "fast"):
    if ocr_mode == "slow":
        return get_ocr_service()           # TrOCR (VisionEncoderDecoder)
    return get_custom_ocr_service()        # CRNN (VGG + BiLSTM + CTC)
```

Default is `"fast"` (CRNN) for interactive use. `"slow"` (TrOCR) is available for higher-accuracy batch processing or when misreads are detected.

---

## 7. Proximity Mapping (Text → Component)

**File:** `test/proximity_mapper.py`

After OCR recognizes text values from the image, each text result must be assigned to its nearest component. This is done via minimum bounding-box edge-to-edge distance.

**Function:** `map_text_to_components(components, text_detections, max_distance=250.0)`

**Algorithm:**

```
For each text detection:
    For each component:
        distance = min edge-to-edge gap between bboxes
        (0 if overlapping)
    Assign text to component with smallest distance (if < max_distance)
    Set component["value"] = ocr_text
    Set component["value_confidence"] = ocr_confidence
    Set component["mapped_text_bbox"] = text bbox
```

**Why edge-to-edge?** Centre-to-centre distance fails when a small text label like "10k" sits right next to a large component bbox. Edge distance correctly identifies proximity regardless of component/text size.

**Edge distance formula:**

```python
dx = max(0, max(a[0] - b[2], b[0] - a[2]))   # gap in x
dy = max(0, max(a[1] - b[3], b[1] - a[3]))   # gap in y
distance = sqrt(dx² + dy²)
```

Returns 0 when boxes overlap.

---

## 8. Orchestration — Unified Pipeline

**File:** `test/unified_pipeline.py`

The `_run_pipeline_raw()` function orchestrates all ML stages in sequence:

```
Step 1: Decode image bytes → BGR ndarray
Step 2: YOLO detection → 15-class bounding boxes
        Split into: text_boxes, junction_dets, component_dets
Step 3: OCR (TrOCR or CRNN) on text_boxes → text_results with ocr_text
Step 4: Proximity mapping → assign text→component → component["value"]
Step 5: Line detection (pipeline.analyze) → graph {nodes, edges}
Step 6: Assemble raw response dict
```

The public function `process_circuit_image()` wraps `_run_pipeline_raw()` and reshapes output into the `Circuit` Pydantic schema:

```
Step 7: Map raw components → Component dicts with terminal info
Step 8: Graph vertices → Node dicts with position
Step 9: Graph edges → Connection dicts (component_terminal → node)
Step 10: Wire edges → Edge dicts (node → node)
```

---

## 9. Standalone Test Scripts

### `run_trocr_test.py`

End-to-end test of YOLO + TrOCR on a single image:

```
1. Load image from disk
2. Run YOLO detection → print all detections
3. Filter text-class detections
4. Load TrOCR (checkpoint-epoch-2)
5. Run OCR on all text crops
6. Print results: text, confidence, bbox
7. Print full detection summary with OCR annotations
```

### `visualize_results.py`

Generates an annotated image with overlaid bounding boxes and OCR labels:

```
1. Run YOLO detection
2. Run TrOCR on text detections
3. Draw colored bounding boxes per class:
   - resistor: green, capacitor: orange, inductor: magenta
   - diode: cyan, voltage: light orange, gnd: gray
   - junction: yellow, text: dark orange, transistor: pink
   - switch: dark yellow, terminal: light green
4. For text boxes: display OCR result instead of "text" label
5. Save annotated JPEG (quality 95)
```

---

## 10. Training Notebooks

### `OCRmodel/EasyOCR.ipynb` — CRNN Training

| Setting       | Value                                    |
|---------------|------------------------------------------|
| Character set | 90 chars (includes Ω, µ, ×, ß)          |
| Image height  | 32 px (aspect-ratio-preserved width)     |
| Batch size    | 128                                      |
| Epochs        | 3                                        |
| Loss          | `CTCLoss(blank=0)`                       |
| Architecture  | VGG + 2×BiLSTM + Linear(91)             |

Contains: `WordImageDataset`, `ResizeKeepAspectRatio`, `collate_fn` (pads variable-width images), full training loop with validation, `greedy_decoder`.

### `customOCR/anveshanout (1).ipynb` — Alternative CRNN

Similar structure with slightly different CNN channel configurations. Also includes experimental YOLO inference cells.

---

## 11. File Reference

| File | Purpose | Key Exports |
|------|---------|-------------|
| `test/pipeline.py` | Line detection core | `analyze()` |
| `test/pipelinewithbeizer.py` | Extended with Bézier curves | `analyze()` (extended) |
| `test/ocr_service.py` | TrOCR wrapper | `OCRService`, `get_ocr_service()` |
| `test/ocr_engine.py` | Compatibility shim | Re-exports `OCRService` as `OCREngine` |
| `test/custom_ocr.py` | CRNN fast OCR | `CustomOCRService`, `CRNN`, `get_custom_ocr_service()` |
| `test/unified_pipeline.py` | Full orchestration | `process_circuit_image()`, `_run_pipeline_raw()` |
| `test/yolo_detector.py` | YOLO wrapper | `detect()`, `detect_parsed()`, `COMPONENT_NAMES` |
| `test/proximity_mapper.py` | Text-to-component assignment | `map_text_to_components()` |
| `test/schemas.py` | Pydantic models | `Circuit`, `Component`, `Node`, `Connection`, `Edge` |
| `run_trocr_test.py` | Standalone YOLO+TrOCR test | (script) |
| `visualize_results.py` | Annotated image generator | (script) |
| `OCRmodel/EasyOCR.ipynb` | CRNN training notebook | (notebook) |
| `OCRmodel/crnn_last.pth` | CRNN trained weights | (binary) |
| `OCRmodel/trocrfinetuned/checkpoint-epoch-2/` | TrOCR fine-tuned weights | (directory) |
| `OCRmodel/trocr-small-printed/` | TrOCR base model files | (directory) |

---

## 12. Dependencies

Core packages required for line detection and OCR:

| Package          | Version   | Used For                                          |
|------------------|-----------|---------------------------------------------------|
| `torch`          | 2.10.0    | CRNN inference, TrOCR inference                   |
| `torchvision`    | 0.25.0    | Image transforms for CRNN preprocessing           |
| `transformers`   | 5.2.0     | TrOCR model loading (VisionEncoderDecoderModel)    |
| `scikit-image`   | 0.26.0    | `skeletonize()` for wire detection                |
| `opencv-python`  | 4.13.0.92 | Image I/O, thresholding, morphology, cropping      |
| `numpy`          | 2.4.2     | Array operations throughout                       |
| `pillow`         | 12.1.1    | PIL Image handling for OCR                        |
| `matplotlib`     | 3.10.8    | Diagnostic visualization images                   |
| `ultralytics`    | latest    | YOLOv7 detection model                           |
| `scipy`          | 1.17.1    | Used internally by scikit-image                   |

---

## 13. Tunable Parameters & Constants

### Line Detection

| Constant / Param       | Location               | Default  | Description |
|------------------------|------------------------|----------|-------------|
| `BINARY_THRESH`        | `pipeline.py`          | 110      | Grayscale threshold; lower = more ink pixels detected |
| `node_radius`          | `DEFAULT_PARAMS`       | 25 px    | BFS disk radius; larger catches more distant skeleton branches |
| `cls1_margin`          | `DEFAULT_PARAMS`       | 8 px     | Search expansion for junction/crossover/terminal centers |
| `border_tol`           | `DEFAULT_PARAMS`       | 1 px     | Annular band width for border intersection detection |
| `global_merge_dist`    | `DEFAULT_PARAMS`       | 5.0 px   | Merge threshold for nearby endpoints |
| `find_node_dist_thresh`| `DEFAULT_PARAMS`       | 5 px     | Max distance to match endpoint → graph node |
| `merge_dist` (local)   | Various functions      | 8.0 px   | Per-component border-hit clustering radius |
| `DOT_THRESHOLD`        | Crossover dissolution  | −0.7     | Min antiparallel score for crossover wire pairing |
| `link_dist`            | Component-node map     | 8 px     | merge_dist + 3; max distance for component→node assignment |
| Dilation kernel        | Stage 3                | 3×3, 1 iter | Gap-filling after text erasure |

### Bézier Extension

| Constant               | Location                   | Default  | Description |
|------------------------|----------------------------|----------|-------------|
| `DEVIATION_THRESHOLD`  | `pipelinewithbeizer.py`    | 15 px    | Max deviation for classifying edge as straight |
| `num_points`           | `_evaluate_bezier()`       | 50       | Sample points along fitted curve |

### TrOCR

| Constant               | Location               | Default  | Description |
|------------------------|------------------------|----------|-------------|
| `_MAX_NEW_TOKENS`      | `ocr_service.py`       | 16       | Max decoder output length |
| `_DEFAULT_MODEL_ID`    | `ocr_service.py`       | `OCRmodel/trocrfinetuned/checkpoint-epoch-2/` | Model path |
| float16                | `_ensure_loaded()`     | On CUDA  | Half-precision for speed |

### CRNN

| Constant               | Location               | Default  | Description |
|------------------------|------------------------|----------|-------------|
| `_IMG_HEIGHT`          | `custom_ocr.py`        | 32       | Input image height (width is proportional) |
| `output_channel`       | `CRNN.__init__()`      | 512      | VGG output channels |
| `hidden_size`          | `CRNN.__init__()`      | 256      | BiLSTM hidden dimension |
| `num_class`            | `CRNN.__init__()`      | 91       | 90 chars + 1 CTC blank |
| `_DEFAULT_MODEL_PATH`  | `custom_ocr.py`        | `OCRmodel/crnn_last.pth` | Trained weights |

### Proximity Mapping

| Constant               | Location                   | Default  | Description |
|------------------------|----------------------------|----------|-------------|
| `max_distance`         | `map_text_to_components()` | 250 px   | Max edge-to-edge distance for text→component assignment |
