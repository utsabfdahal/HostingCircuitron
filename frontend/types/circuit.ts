// TypeScript interfaces mirroring the backend Pydantic schemas (schemas.py)

export interface Position {
  x: number;
  y: number;
}

// ── Final schematic models ──────────────────────────────────────────────────

export interface Component {
  id: string;
  type: string;
  label: string;
  value: string;
  unit: string;
  position: Position;
  rotation: number;
  terminals: string[];
}

export interface Node {
  id: string;
  position: Position;
}

export interface Connection {
  from_component: string;
  from_terminal: string;
  to_node: string;
}

export interface Edge {
  source: string;  // node id, e.g. "n0"
  target: string;  // node id, e.g. "n3"
}

export interface Circuit {
  circuit_id: string;
  components: Component[];
  nodes: Node[];
  connections: Connection[];
  edges: Edge[];
}

// ── Preview / Analysis models ───────────────────────────────────────────────

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedComponent {
  id: string;
  cls: number;
  type: string;
  name: string;
  confidence: number;
  bbox: BBox;
  position: Position;
  value: string;
  matched_text: string;
}

export interface DetectedText {
  id: number;
  bbox: BBox;
  ocr_text: string;
  ocr_confidence: number;
}

export interface DetectedJunction {
  id: number;
  type: string;
  bbox: BBox;
  confidence: number;
  position: Position;
}

export interface GraphNode {
  id: number;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  path: Position[];
  linked_components?: {
    source_components: { cls: number; name: string; bbox: number[] }[];
    target_components: { cls: number; name: string; bbox: number[] }[];
  };
}

export interface DiagnosticImages {
  skeleton_png: string;          // base64 PNG
  overlay_png: string;           // base64 PNG
  bbox_png: string;              // base64 PNG
  adjacency_graph_png: string;   // base64 PNG
}

export interface AnalysisPreview {
  session_id: string;
  image_width: number;
  image_height: number;
  annotated_image: string;   // base64 PNG
  original_image: string;    // base64 PNG
  components: DetectedComponent[];
  texts: DetectedText[];
  junctions: DetectedJunction[];
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  diagnostic_images: DiagnosticImages;
}

export interface EditedAnalysis {
  image_width: number;
  image_height: number;
  components: DetectedComponent[];
  texts: DetectedText[];
  junctions: DetectedJunction[];
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
}
