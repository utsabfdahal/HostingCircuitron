# CIRCUITRON

**Hand-drawn circuit diagrams → interactive digital schematics.**

CIRCUITRON is an end-to-end ML-powered application that converts photographs or scans of hand-drawn circuit diagrams into fully editable, simulatable digital schematics rendered inside [CircuitJS1](https://www.falstad.com/circuit/). The user uploads an image, reviews and corrects the ML detections, and receives a live SPICE-level simulation — all in the browser. An embedded AI assistant powered by Lightning AI (DeepSeek-V3.1) helps users understand, debug, and improve their circuits in real time.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
  - [Stage 1 — YOLO Component Detection](#stage-1--yolo-component-detection)
  - [Stage 2 — OCR Text Recognition (Dual Mode)](#stage-2--ocr-text-recognition-dual-mode)
  - [Stage 3 — Proximity Mapping](#stage-3--proximity-mapping)
  - [Stage 4 — Skeleton & Graph Analysis](#stage-4--skeleton--graph-analysis)
- [AI Chat Assistant](#ai-chat-assistant)
- [CircuitJS1 Integration & Backend Migration](#circuitjs1-integration--backend-migration)
  - [Why Backend Export?](#why-backend-export)
  - [CJS Text Format](#cjs-text-format)
  - [Coordinate Rescaling](#coordinate-rescaling)
  - [Component Type Mapping](#component-type-mapping)
  - [Terminal Assignment & Wiring](#terminal-assignment--wiring)
- [API Reference](#api-reference)
- [Frontend Workflow](#frontend-workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Manual Setup](#manual-setup)
- [Configuration & Thresholds](#configuration--thresholds)
- [Model Files](#model-files)
- [Tech Stack](#tech-stack)

---

## Features

- **15-class YOLO detection** — resistors, capacitors, diodes, voltage sources, transistors, op-amps, ICs, inductors, switches, junctions, terminals, ground, Vss, crossovers, and text labels.
- **Dual OCR modes** — choose between **Fast** (custom CRNN — lightweight CNN + BiLSTM with CTC decoding) and **Accurate** (TrOCR — transformer-based, higher accuracy) for reading component values.
- **Skeleton-based wire tracing** — binary thresholding, morphological skeletonisation, multi-head BFS to build the full connectivity graph.
- **Real-time threshold preview** — client-side canvas rendering lets you tune the binary threshold *before* analysis begins.
- **Interactive review & edit** — six diagnostic views (annotated, original+overlays, skeleton, wire overlay, detection boxes, adjacency graph) with toggle-able SVG overlays and editable component/text/junction lists.
- **Live CircuitJS1 simulation** — the final schematic is loaded into an embedded CircuitJS1 iframe with run/pause, timestep control, SVG/TXT/JSON export, and an optional oscilloscope panel.
- **AI Circuit Assistant** — a floating chat window (powered by Lightning AI / DeepSeek-V3.1) that understands your circuit's components and connections, answers questions, identifies issues, and suggests improvements.
- **Backend-driven CJS export** — coordinate rescaling, grid snapping, and component-code mapping all happen server-side, faithfully reproducing the graph topology as clean CircuitJS1 netlists.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend  (Next.js)                      │
│                                                              │
│   Upload ──► Review / Edit ──► Simulate (CircuitJS1 iframe)  │
│                                    └── AI Chat (floating)    │
│   Real-time threshold preview (client-side Canvas)           │
│   OCR mode toggle: ⚡ Fast (CRNN) │ 🔬 Accurate (TrOCR)     │
│   SVG overlays, editable sidebar, toast notifications        │
└────────────────────────┬─────────────────────────────────────┘
                         │  REST  (CORS-enabled)
                         │  POST /analyze, /re-analyze, /finalize,
                         │        /export-cjs, /chat
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Backend  (FastAPI)                        │
│                                                              │
│   main.py ─── unified_pipeline.py (orchestrator)             │
│                 ├── yolo_detector.py    (Stage 1: YOLO)      │
│                 ├── custom_ocr.py       (Stage 2a: CRNN Fast)│
│                 ├── ocr_service.py      (Stage 2b: TrOCR)    │
│                 ├── proximity_mapper.py (Stage 3: matching)   │
│                 ├── pipeline.py         (Stage 4: skeleton)   │
│                 └── circuit_to_cjs()    (CJS export)         │
│                                                              │
│   chat_service.py ─── Lightning AI / DeepSeek-V3.1           │
│   schemas.py ─── Pydantic models (Circuit, AnalysisPreview)  │
└────────────────────────┬─────────────────────────────────────┘
                         │  Disk I/O
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Model Weights                            │
│                                                              │
│   yolov7new/best.pt              (YOLOv7, 15-class)         │
│   OCRmodel/trocrfinetuned/       (TrOCR-small, fine-tuned)   │
│   customOCR/crnn_last (1).pth    (CRNN, fast mode)           │
│   circuitjs1/site/               (GWT-compiled simulator)    │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cktonutssab/
├── README.md                  # This file
├── requirements.txt           # Python dependencies (pip)
├── start.sh                   # One-command launcher script
├── .env                       # API keys (git-ignored)
│
├── test/                      # Backend (FastAPI + ML pipeline)
│   ├── __init__.py
│   ├── main.py                # FastAPI app & endpoints
│   ├── schemas.py             # Pydantic request/response models
│   ├── unified_pipeline.py    # Pipeline orchestrator + CJS export
│   ├── yolo_detector.py       # YOLO detection wrapper (ultralytics)
│   ├── ocr_service.py         # TrOCR batch inference (Accurate mode)
│   ├── custom_ocr.py          # CRNN fast OCR (Fast mode)
│   ├── ocr_engine.py          # OCR compatibility shim
│   ├── proximity_mapper.py    # Text → component greedy matcher
│   ├── pipeline.py            # Skeleton extraction & BFS graph
│   └── chat_service.py        # AI chat via Lightning AI API
│
├── frontend/                  # Frontend (Next.js + React)
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── app/
│   │   ├── page.tsx           # Main 3-step workflow UI + AI Chat
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── types/
│   │   └── circuit.ts         # TypeScript interfaces
│   └── public/
│
├── yolov7new/
│   └── best.pt                # YOLOv7 weights (15-class)
│
├── OCRmodel/
│   ├── crnn_last.pth          # CRNN weights (fast OCR)
│   └── trocrfinetuned/
│       └── checkpoint-epoch-2/  # Fine-tuned TrOCR-small
│
├── customOCR/                 # CRNN training artifacts
│   ├── crnn_last (1).pth      # CRNN model weights
│   └── easyOCRstuff/
│       ├── character_set.txt  # 90-char set for CRNN
│       └── imageswithlabels/  # Training data
│
├── circuitjs1/                # CircuitJS1 simulator (GWT source + compiled)
│   ├── site/
│   │   ├── circuitjs.html     # Simulator HTML entry point
│   │   └── circuitjs1/        # Compiled GWT JS modules
│   ├── src/                   # Java/GWT source code
│   └── build.gradle
│
└── tests/                     # Circuit test files (.txt)
```

---

## ML Pipeline

The backend runs a four-stage pipeline on every uploaded image:

```
Image → YOLO → OCR (Fast or Accurate) → Proximity Mapping → Skeleton/Graph → Circuit JSON → CJS Text
```

### Stage 1 — YOLO Component Detection

**Module**: `test/yolo_detector.py`
**Model**: Ultralytics YOLOv7s &nbsp;|&nbsp; **Weights**: `yolov7new/best.pt`

Detects 15 classes of circuit elements:

| Index | Class                  | Description                          |
|-------|------------------------|--------------------------------------|
| 0     | `capacitor`            | Capacitor symbol                     |
| 1     | `crossover`            | Wire crossing (no electrical contact)|
| 2     | `diode`                | Diode / LED                          |
| 3     | `gnd`                  | Ground symbol                        |
| 4     | `inductor`             | Inductor / coil                      |
| 5     | `integrated_circuit`   | IC / chip package                    |
| 6     | `junction`             | Wire junction (electrical contact)   |
| 7     | `operational_amplifier`| Op-amp triangle symbol               |
| 8     | `resistor`             | Resistor (zigzag or box)             |
| 9     | `switch`               | Switch                               |
| 10    | `terminal`             | Named terminal / port                |
| 11    | `text`                 | Text label (fed to OCR)              |
| 12    | `transistor`           | BJT / MOSFET                         |
| 13    | `voltage`              | DC or AC voltage source              |
| 14    | `vss`                  | Negative supply rail                 |

**Inference settings**: `imgsz=640`, `conf=0.25`, `iou=0.45`, `save=False`.

Output is YOLO-format label text (normalised `class xc yc w h confidence`), which is also parsed into structured dicts with pixel-coordinate bounding boxes.

### Stage 2 — OCR Text Recognition (Dual Mode)

Users can choose between two OCR backends from the UI:

#### ⚡ Fast Mode — Custom CRNN

**Module**: `test/custom_ocr.py`
**Model**: CRNN (VGG Feature Extractor + Bidirectional LSTM + CTC)
**Weights**: `customOCR/crnn_last (1).pth`

| Property | Details |
|----------|---------|
| Architecture | VGG-style (6 conv layers, 512 output channels) → AdaptiveAvgPool → 2× BiLSTM (hidden=256) → Linear (→91 classes) |
| Character set | 90 characters including digits, letters, EE symbols (`Ω`, `µ`, `×`) |
| Input | Grayscale, resized to height 32 px (aspect ratio preserved) |
| Decoding | CTC greedy decode with blank-index filtering |
| Speed | ~5–10× faster than TrOCR (no transformer overhead) |
| Device | CUDA (auto) or CPU |

#### 🔬 Accurate Mode — TrOCR

**Module**: `test/ocr_service.py`
**Model**: HuggingFace TrOCR-small fine-tuned
**Checkpoint**: `OCRmodel/trocrfinetuned/checkpoint-epoch-2`

| Property | Details |
|----------|---------|
| Architecture | Vision Encoder-Decoder (ViT encoder + GPT-2 decoder) |
| Inference | Batch (all crops in one forward pass) |
| Precision | float16 on GPU for 2× speed |
| Max tokens | 16 (sufficient for labels like `10k`, `100uF`, `5V`) |
| Confidence | Mean log-probability of generated tokens |

Both modes produce the same output format: `(ocr_text, ocr_confidence)` per detected text region.

### Stage 3 — Proximity Mapping

**Module**: `test/proximity_mapper.py`

A greedy one-to-one assignment algorithm that matches each OCR text region to its nearest component:

- Uses **edge-to-edge distance** (not centre-to-centre) so small labels adjacent to large components match correctly.
- Maximum matching distance: **250 px**.
- Components receive `value`, `value_confidence`, and `matched_text_bbox` fields.

### Stage 4 — Skeleton & Graph Analysis

**Module**: `test/pipeline.py`

Extracts the wire connectivity graph from the image:

1. **Preprocessing** — median blur → binary threshold (user-adjustable, default 110) → invert → erase text regions.
2. **Skeletonisation** — morphological dilation → `skimage.morphology.skeletonize` → threshold cleanup.
3. **Endpoint detection** — for two-terminal components: find skeleton intersections at the bounding-box border; for junctions/terminals: find nearest skeleton point.
4. **Endpoint merging** — cluster endpoints within 5 px globally.
5. **Multi-head BFS** — walk the skeleton from each node outward, discover which nodes are connected, prune internal-component edges.
6. **Output** — node positions, adjacency edges, and four diagnostic images (skeleton, overlay, bbox, adjacency graph) as base64 PNGs.

---

## CircuitJS1 Integration & Backend Migration

### Why Backend Export?

Earlier iterations attempted to build CircuitJS1 text on the frontend. This was migrated to the backend for several reasons:

| Concern | Frontend Approach (old) | Backend Approach (current) |
|---------|------------------------|---------------------------|
| **Coordinate system** | Raw pixel coords passed directly — component symbols rendered tiny on large images because CircuitJS1 draws fixed-size symbols regardless of coordinate span. | `circuit_to_cjs()` rescales all coordinates to a compact ~500 px range, keeping symbols proportional to wires. |
| **Grid alignment** | No grid snapping — CJS elements drifted off-grid, causing broken connections. | All coordinates snapped to a 48 px grid (`round(x/48)*48`), matching CJS's native grid. |
| **Component mapping** | Hard-coded on the client, hard to maintain across 15 classes. | Centralised `_type_to_cjs()` mapping lives next to the pipeline logic; adding a new component type is a one-line change. |
| **Terminal wiring** | Relied on the frontend to figure out which node connects to which terminal — duplicated graph logic. | Backend already has the full connectivity graph; terminal assignment, connection records, and wire generation are computed once and exported directly. |
| **Parameter encoding** | Frontend had to know CJS parameter syntax for every component type. | `_cjs_params()` encodes resistance/capacitance/inductance/voltage values with correct CJS parameter ordering. |
| **Zero-length guard** | Not enforced — zero-length components were invisible in CJS. | If two nodes snap to the same grid point, the component is extended by one grid cell (48 px) to remain visible. |
| **Diagonal wires** | Passed through as-is. | Preserved faithfully — wires are drawn directly from source to target node, maintaining the graph's natural geometry. |

The `/export-cjs` endpoint accepts a `Circuit` JSON and returns `{"cjs_text": "..."}` ready to be URL-encoded and loaded into the CircuitJS1 iframe.

### CJS Text Format

CircuitJS1 uses a plain-text netlist format. Each line is one element:

```
$ 1 0.000005 10.20027730826997 63 10 62 5e-11    ← header (timestep, speed, etc.)
r 192 144 288 144 0 1000                           ← 1kΩ resistor, horizontal
c 288 144 288 240 0 0.000001 0                     ← 1µF capacitor, vertical
v 192 240 192 144 0 0 40 5 0 0 0.5                 ← 5V DC source
w 288 144 384 144 0                                ← wire
g 288 240 288 288 0                                ← ground
x 192 130 0 12 V1                                  ← text label
```

**Element line format**: `{code} {x1} {y1} {x2} {y2} {params...}`

Where `(x1,y1)` and `(x2,y2)` are the two endpoints and `code` identifies the element type. Two elements are electrically connected when they share an exact endpoint coordinate.

### Coordinate Rescaling

Raw image coordinates (e.g. 0–2700 px on a high-res photo) cause component symbols to appear tiny in CircuitJS1, which renders symbols at a fixed visual size regardless of coordinate span.

The `circuit_to_cjs()` function rescales all positions:

```
1.  Collect all node + component positions.
2.  Compute bounding box (min_x, min_y, max_x, max_y).
3.  scale = min(500 / max(span_x, span_y),  1.0)
4.  For each coordinate:
      new_x = (x - min_x) * scale + 100          ← 100 px offset from origin
      new_y = (y - min_y) * scale + 100
      snapped_x = round(new_x / 48) * 48         ← snap to 48 px grid
```

This maps the entire circuit into a compact ~100–600 px region where component symbols are readable relative to wire lengths.

**Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `_CJS_GRID` | 48 | Grid snap resolution (matches CJS internal grid) |
| `_CJS_TARGET_SIZE` | 500 | Maximum coordinate span after rescaling |
| `_CJS_OFFSET` | 100 | Margin from top-left corner |

### Component Type Mapping

The `_type_to_cjs()` function maps the 15 YOLO class names to CircuitJS1 element codes:

| YOLO Class | CJS Code | CJS Element |
|------------|----------|-------------|
| `resistor` | `r` | Resistor |
| `capacitor` | `c` | Capacitor |
| `inductor` | `l` | Inductor |
| `voltage` | `v` | Voltage source |
| `diode` | `d` | Diode |
| `transistor` | `t` | NPN transistor |
| `gnd` | `g` | Ground |
| `vss` | `V` | Voltage rail |
| `switch` | `s` | Switch |
| `operational_amplifier` | `a` | Op-amp |
| `integrated_circuit` | `x` | Label (placeholder) |
| `junction` | `w` | Wire (junction point) |
| `terminal` | `w` | Wire (terminal point) |
| `crossover` | `w` | Wire (crossover) |
| `text` | *(skipped)* | — |

**Parameter encoding** (`_cjs_params()`):

| Code | Params | Example |
|------|--------|---------|
| `r` | `0 {ohms}` | `0 1000` → 1kΩ |
| `c` | `0 {farads} 0` | `0 1e-6 0` → 1µF |
| `l` | `0 {henrys} 0` | `0 0.001 0` → 1mH |
| `v` | `0 0 40 {volts} 0 0 0.5` | DC source |
| `d` | `2 default` | Default diode |
| `t` | `0 1 -5.43...` | Default NPN |
| `a` | `2 15 -15 1000000 0 0 0 1` | Ideal op-amp |
| `w` | `0` | Wire |
| `g` | `0` | Ground |

### Terminal Assignment & Wiring

1. For each component, the backend finds graph nodes that fall inside (or within 15 px of) the bounding box.
2. Each node is assigned as a terminal (`R1_t1`, `R1_t2`).
3. A `Connection` record links `(component_id, terminal_id) → node_id`.
4. Components are placed between their two connected nodes; single-connection components get a synthetic second endpoint offset by one grid cell (`_CJS_GRID = 48`). All endpoints are snapped to the 48 px grid.
5. After snapping, the component's final pin coordinates are written back to `node_used_coords` so that the wire generation phase connects to the exact same points.
6. Node-to-node edges become CJS wire (`w`) lines drawn directly from source to target, preserving any natural diagonal or slope from the graph.
7. If two nodes collapse to the same grid point, a zero-length guard extends the component by one grid cell to keep it visible.

---

## API Reference

All endpoints accept/return JSON unless noted. The backend runs on **port 8000**.

### `POST /upload`

One-shot pipeline: image → Circuit JSON.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `UploadFile` | — | Circuit image (multipart form) |
| `ocr_mode` | `string` (Form) | `"fast"` | `"fast"` (CRNN) or `"slow"` (TrOCR) |
| **Returns** | `Circuit` | — | Complete schematic JSON |

### `POST /analyze`

Step 1: image → ML preview for user review.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `UploadFile` | — | Circuit image (multipart form) |
| `binary_thresh` | `int` (Form) | `110` | Binary threshold for skeleton extraction |
| `ocr_mode` | `string` (Form) | `"fast"` | `"fast"` (CRNN) or `"slow"` (TrOCR) |
| **Returns** | `AnalysisPreview` | — | Detections + annotated image + diagnostic images |

### `POST /re-analyze`

Fast re-run of skeleton/graph analysis with a new threshold. Skips YOLO and OCR (uses cached results from `/analyze`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `session_id` | `string` | — | Session ID from `/analyze` response |
| `binary_thresh` | `int` | `110` | New threshold value |
| **Returns** | `object` | — | Updated graph, detections, diagnostic images |

### `POST /finalize`

Step 2: user-edited detections → final Circuit JSON.

| Field | Type | Description |
|-------|------|-------------|
| `body` | `EditedAnalysis` | User-corrected components, texts, junctions, graph |
| **Returns** | `Circuit` | Final schematic with connections and edges |

### `POST /export-cjs`

Convert Circuit JSON to CircuitJS1 text format.

| Field | Type | Description |
|-------|------|-------------|
| `body` | `Circuit` | Complete circuit JSON |
| **Returns** | `{"cjs_text": string}` | CircuitJS1 netlist text |

### `POST /chat`

AI-powered circuit analysis and Q&A.

| Field | Type | Description |
|-------|------|-------------|
| `circuit` | `Circuit` | Current circuit state for context |
| `message` | `string` | User's question |
| `history` | `ChatMessage[]` (optional) | Previous messages for multi-turn context (last 6 kept) |
| **Returns** | `{"reply": string}` | AI assistant's response |

The chat endpoint sends a concise circuit summary (component list, node/edge counts) plus the conversation history to Lightning AI's **DeepSeek-V3.1** model. The system prompt identifies the AI as an expert electrical-engineering assistant for CIRCUITRON.

---

## AI Chat Assistant

**Module**: `test/chat_service.py`
**Model**: Lightning AI / DeepSeek-V3.1
**API**: `https://lightning.ai/api/v1/chat/completions`

The AI Chat Assistant is a conversational feature available during the Simulate step (Step 3). It provides:

- **Circuit understanding** — "What does this circuit do?"
- **Issue identification** — "Are there any problems with this circuit?"
- **Improvement suggestions** — "How can I improve this design?"
- **EE education** — Explanations of electrical concepts in context

**How it works:**

1. The current `Circuit` object (components, nodes, connections, edges) is summarised into a compact text representation.
2. The summary is included in the system prompt so the AI has full context.
3. Up to 3 previous exchanges (6 messages) are included for follow-up questions.
4. The request is sent to Lightning AI's API with the `DeepSeek-V3.1` model.

**Configuration:**

The API key is stored in `.env` at the project root (git-ignored):

```env
LIGHTNING_AI_API_KEY=your-key-here
```

---

## Frontend Workflow

The UI is a single-page React application with three steps:

### Step 1 — Upload

1. Drag-and-drop or file-picker to select an image.
2. **OCR mode toggle**: choose between ⚡ **Fast** (CRNN) and 🔬 **Accurate** (TrOCR) OCR mode for text recognition.
3. **Real-time threshold preview**: a side-by-side canvas shows the original image and a binary-thresholded version. The threshold slider (range 30–230) updates the preview instantly via client-side canvas pixel manipulation.
4. Click **"Analyze Circuit"** to send the image, chosen threshold, and OCR mode to `POST /analyze`.
5. A progress animation with rotating circuit facts plays during processing.

### Step 2 — Review & Edit

1. **Multi-view canvas** with six views: Annotated, Original + Overlays, Skeleton, Wire Overlay, Detection Boxes, Adjacency Graph.
2. **SVG overlays** toggled independently: Components (green), Texts (yellow), Junctions (red), Graph (purple/magenta).
3. **OCR mode toggle** available in the toolbar for re-analysis.
4. **Threshold slider** (debounced 400ms) calls `/re-analyze` to update the graph without re-running YOLO/OCR.
5. **Editable sidebar**: modify component types/values, edit OCR text, delete false detections.
6. Click **"Generate Schematic"** to call `/finalize` then `/export-cjs`.

### Step 3 — Simulate

1. CircuitJS1 loads in an iframe with the generated CJS text.
2. **Controls**: Run/Pause, Timestep ×½/×2, Reset, Export TXT/SVG/JSON.
3. **Optional oscilloscope**: live voltage/current waveforms with cursor readouts and trace highlighting.
4. **AI Chat Assistant**: a floating chat window ( button, bottom-right) for asking questions about the circuit, powered by DeepSeek-V3.1.

---

## Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- **Model weights** placed at:
  - `yolov7new/best.pt` (YOLOv7 15-class)
  - `OCRmodel/trocrfinetuned/checkpoint-epoch-2/` (TrOCR-small fine-tuned)
  - `customOCR/crnn_last (1).pth` (CRNN fast OCR)
- **Lightning AI API key** in `.env` (for AI Chat feature)
- (Optional) NVIDIA GPU with CUDA for faster inference

### Quick Start

```bash
chmod +x start.sh
./start.sh
```

This will:
1. Create a Python virtual environment (`.venv/`) if it doesn't exist.
2. Install all Python dependencies from `requirements.txt`.
3. Install frontend Node.js dependencies (`npm install`).
4. Start the FastAPI backend on **http://localhost:8000**.
5. Start the Next.js dev server on **http://localhost:3000**.

Press `Ctrl+C` to stop both servers.

### Environment Variables

Create a `.env` file in the project root:

```env
LIGHTNING_AI_API_KEY=your-lightning-ai-api-key-here
```

This key is required for the AI Chat Assistant feature. The `.env` file is git-ignored.

### Manual Setup

**Backend:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn test.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev -- -p 3000
```

Open **http://localhost:3000** in your browser.

---

## Configuration & Thresholds

### User-Adjustable

| Parameter | Range | Default | Where |
|-----------|-------|---------|-------|
| Binary threshold | 30–230 | 110 | Upload preview slider & Review slider |

### Detection Thresholds (code-level)

| Parameter | Value | File |
|-----------|-------|------|
| YOLO confidence | 0.25 | `yolo_detector.py` |
| YOLO IoU (NMS) | 0.45 | `yolo_detector.py` |
| YOLO image size | 640 | `yolo_detector.py` |
| Node merge distance | 5.0 px | `pipeline.py` |
| Node BFS radius | 25 px | `pipeline.py` |
| Text→component max distance | 250 px | `proximity_mapper.py` |
| CJS grid size | 48 px | `unified_pipeline.py` |
| CJS target coordinate span | 500 px | `unified_pipeline.py` |

---

## Model Files

| Model | Path | Format | Purpose |
|-------|------|--------|---------|
| YOLOv7 weights | `yolov7new/best.pt` | PyTorch (Ultralytics) | 15-class component detection |
| TrOCR checkpoint | `OCRmodel/trocrfinetuned/checkpoint-epoch-2/` | HuggingFace | OCR text recognition (Accurate mode) |
| CRNN weights | `customOCR/crnn_last (1).pth` | PyTorch | Fast OCR (CRNN mode) |

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Next.js | 14.2 |
| | React | 18.3 |
| | TypeScript | 5.4 |
| | TailwindCSS | 3.4 |
| **Backend** | FastAPI | 0.134 |
| | Uvicorn | 0.41 |
| | Pydantic | 2.12 |
| **Detection** | Ultralytics (YOLOv7) | latest |
| **OCR (Accurate)** | HuggingFace Transformers (TrOCR) | 5.2 |
| **OCR (Fast)** | Custom CRNN (VGG + BiLSTM + CTC) | — |
| **AI Chat** | Lightning AI / DeepSeek-V3.1 | — |
| **Image Processing** | OpenCV | 4.13 |
| | scikit-image | 0.26 |
| **Deep Learning** | PyTorch | 2.10 |
| | torchvision | 0.25 |
| **HTTP Client** | httpx | 0.28 |
| **Simulation** | CircuitJS1 | GWT-compiled |

---

## License

CircuitJS1 is licensed under the GNU General Public License. See `circuitjs1/COPYING.txt` for details. The rest of the CIRCUITRON project is proprietary.
