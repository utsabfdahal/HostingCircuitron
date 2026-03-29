"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Check, X, Info, Zap, Upload, Search, FileText, GitBranch,
  ArrowLeft, ArrowRight, Pause, Play, ChevronDown, ChevronUp,
  RotateCcw, Image as ImageIcon, Clipboard, Activity, MessageCircle, Lightbulb,
  ArrowUp,
} from "lucide-react";
import type {
  AnalysisPreview,
  Circuit,
  DetectedComponent,
  DetectedJunction,
  DetectedText,
  EditedAnalysis,
  GraphEdge,
  GraphNode,
} from "@/types/circuit";

// ─── API ────────────────────────────────────────────────────────────────────

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── CircuitJS1 JS API types ────────────────────────────────────────────────

interface CircuitJSElement {
  getType(): string;
  getPostCount(): number;
  getInfo(): string;
  getVoltageDiff(): number;
  getCurrent(): number;
  getVoltage(n: number): number;
  getLabelName?(): string;
}

interface CircuitJSAPI {
  setSimRunning(run: boolean): void;
  isRunning(): boolean;
  getTime(): number;
  getTimeStep(): number;
  getMaxTimeStep(): number;
  setMaxTimeStep(dt: number): void;
  getNodeVoltage(label: string): number;
  setExtVoltage(label: string, v: number): void;
  getElements(): CircuitJSElement[];
  exportCircuit(): string;
  getCircuitAsSVG(): void;
  importCircuit(text: string, subcircuitsOnly?: boolean): void;
  onupdate?: (sim: CircuitJSAPI) => void;
  ontimestep?: (sim: CircuitJSAPI) => void;
  onanalyze?: (sim: CircuitJSAPI) => void;
  onsvgrendered?: (sim: CircuitJSAPI, svg: string) => void;
}

interface CircuitJSWindow extends Window {
  CircuitJS1: CircuitJSAPI;
  oncircuitjsloaded?: () => void;
}

// ─── Toast ──────────────────────────────────────────────────────────────────

type ToastType = "success" | "error" | "info";
interface ToastItem {
  id: number;
  message: string;
  type: ToastType;
}

let toastId = 0;

function ToastContainer({ toasts, onDismiss }: { toasts: ToastItem[]; onDismiss: (id: number) => void }) {
  return (
    <div className="fixed bottom-6 right-6 z-[999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-lg shadow-2xl border backdrop-blur-md text-sm animate-slide-up
            ${t.type === "success" ? "bg-emerald-900/80 border-emerald-600/50 text-emerald-100" : ""}
            ${t.type === "error" ? "bg-red-900/80 border-red-600/50 text-red-100" : ""}
            ${t.type === "info" ? "bg-blue-900/80 border-blue-600/50 text-blue-100" : ""}
          `}
        >
          <span>
            {t.type === "success" && <Check className="w-4 h-4" />}
            {t.type === "error" && <X className="w-4 h-4" />}
            {t.type === "info" && <Info className="w-4 h-4" />}
          </span>
          <span className="flex-1">{t.message}</span>
          <button onClick={() => onDismiss(t.id)} className="opacity-60 hover:opacity-100 ml-2"><X className="w-3 h-3" /></button>
        </div>
      ))}
    </div>
  );
}

function useToast() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const push = useCallback((message: string, type: ToastType = "info") => {
    const id = ++toastId;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, push, dismiss };
}

// ─── Workflow steps ─────────────────────────────────────────────────────────

type Step = "upload" | "review" | "schematic";

const stepMeta: Record<Step, { label: string; icon: string }> = {
  upload: { label: "Upload", icon: "1" },
  review: { label: "Review & Edit", icon: "2" },
  schematic: { label: "Schematic", icon: "3" },
};

// ─── Circuit facts shown during processing ──────────────────────────────────

const CIRCUIT_FACTS = [
  "The first integrated circuit was built by Jack Kilby in 1958 — it was about the size of a pencil tip.",
  "A single modern CPU chip can contain over 50 billion transistors.",
  "Circuit diagrams were first standardized by the IEC in 1909.",
  "The longest running electronic circuit is the Oxford Electric Bell, ringing since 1840.",
  "Kirchhoff's circuit laws, formulated in 1845, are still the foundation of all circuit analysis today.",
  "The term 'bug' in electronics originated when a moth got trapped in a Harvard relay computer in 1947.",
  "A human brain operates on roughly 20 watts — less power than a light bulb, yet outperforms most computers at pattern recognition.",
  "The world's smallest transistor is just 1 nanometre long — about the width of 5 atoms.",
  "Nikola Tesla's AC motor designs from 1888 are still used in nearly every electric motor worldwide.",
  "Resistor colour codes were standardized in the 1920s and haven't changed since.",
  "The speed of electricity through a copper wire is approximately 2/3 the speed of light.",
  "Claude Shannon's 1937 master's thesis showed how Boolean algebra could optimize telephone switching circuits — sparking the digital age.",
  "Superconductors can carry current with literally zero resistance when cooled near absolute zero.",
  "A capacitor was first discovered in 1745 as the 'Leyden jar' — it could store enough charge to knock a person down.",
  "The oscilloscope, invented in 1897, is still the single most important tool in circuit debugging.",
  "Moore's Law predicted transistor count doubling every ~2 years — and held true for over 50 years.",
  "SPICE (Simulation Program with Integrated Circuit Emphasis) was created at UC Berkeley in 1973 and is still the gold standard for circuit simulation.",
  "The shortest possible circuit is a superconducting loop — electricity in it can flow forever without any energy source.",
];

// ─── Reusable icon button ───────────────────────────────────────────────────

function IconBtn({
  children,
  onClick,
  disabled,
  title,
  variant = "default",
  className = "",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
  variant?: "default" | "primary" | "success" | "danger" | "warning";
  className?: string;
}) {
  const base = "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed";
  const variants: Record<string, string> = {
    default: "bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.08] text-gray-200",
    primary: "bg-indigo-600 hover:bg-indigo-500 text-white",
    success: "bg-emerald-600 hover:bg-emerald-500 text-white",
    danger: "bg-red-600/80 hover:bg-red-500 text-white",
    warning: "bg-amber-600 hover:bg-amber-500 text-white",
  };
  return (
    <button onClick={onClick} disabled={disabled} title={title} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}

// ─── Divider ────────────────────────────────────────────────────────────────

function Divider() {
  return <div className="w-px h-5 bg-white/10 mx-0.5" />;
}

// ─── Step 1: Upload ─────────────────────────────────────────────────────────

function UploadStep({
  onAnalyzed,
  toast,
}: {
  onAnalyzed: (data: AnalysisPreview, thresholdUsed: number) => void;
  toast: (msg: string, type: ToastType) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(0);
  const [factIndex, setFactIndex] = useState(0);
  const [threshold, setThreshold] = useState(110);
  const [ocrMode, setOcrMode] = useState<"fast" | "slow">("fast");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalDataUrl, setOriginalDataUrl] = useState<string>("");
  const [grayPixels, setGrayPixels] = useState<Uint8Array | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fake progress while analyzing
  useEffect(() => {
    if (!loading) { setProgress(0); return; }
    let p = 0;
    const iv = setInterval(() => {
      p += Math.random() * 12;
      if (p > 92) p = 92;
      setProgress(p);
    }, 400);
    return () => clearInterval(iv);
  }, [loading]);

  // Rotate circuit facts during loading
  useEffect(() => {
    if (!loading) { setFactIndex(0); return; }
    setFactIndex(Math.floor(Math.random() * CIRCUIT_FACTS.length));
    const iv = setInterval(() => {
      setFactIndex((prev) => (prev + 1) % CIRCUIT_FACTS.length);
    }, 5000);
    return () => clearInterval(iv);
  }, [loading]);

  // When a file is selected, decode it and extract grayscale pixels
  const handleFileSelected = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        toast("Please upload an image file", "error");
        return;
      }
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        setOriginalDataUrl(dataUrl);
        const img = new Image();
        img.onload = () => {
          const w = img.width;
          const h = img.height;
          setImgDims({ w, h });
          // Draw to offscreen canvas to get pixel data
          const off = document.createElement("canvas");
          off.width = w;
          off.height = h;
          const ctx = off.getContext("2d")!;
          ctx.drawImage(img, 0, 0);
          const rgba = ctx.getImageData(0, 0, w, h).data;
          // Convert to grayscale
          const gray = new Uint8Array(w * h);
          for (let i = 0; i < w * h; i++) {
            const r = rgba[i * 4];
            const g = rgba[i * 4 + 1];
            const b = rgba[i * 4 + 2];
            gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
          }
          setGrayPixels(gray);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    },
    [toast]
  );

  // Render thresholded image whenever threshold or grayPixels change
  useEffect(() => {
    if (!grayPixels || !canvasRef.current) return;
    const { w, h } = imgDims;
    const canvas = canvasRef.current;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const val = grayPixels[i] > threshold ? 255 : 0;
      imgData.data[i * 4] = val;
      imgData.data[i * 4 + 1] = val;
      imgData.data[i * 4 + 2] = val;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [grayPixels, threshold, imgDims]);

  // Start analysis with the selected file + threshold
  const doAnalyze = useCallback(
    async () => {
      if (!selectedFile) return;
      setLoading(true);
      try {
        const form = new FormData();
        form.append("file", selectedFile);
        form.append("binary_thresh", String(threshold));
        form.append("ocr_mode", ocrMode);
        const res = await fetch(`${API}/analyze`, { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail ?? "Analysis failed");
        }
        setProgress(100);
        const data: AnalysisPreview = await res.json();
        toast(`Detected ${data.components.length} components, ${data.texts.length} text labels`, "success");
        onAnalyzed(data, threshold);
      } catch (e: any) {
        toast(`Upload error: ${e.message}`, "error");
      } finally {
        setLoading(false);
      }
    },
    [onAnalyzed, toast, threshold, selectedFile, ocrMode]
  );

  const resetFile = useCallback(() => {
    setSelectedFile(null);
    setOriginalDataUrl("");
    setGrayPixels(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // ── Phase 2: threshold preview (file selected, not yet analyzing) ──
  if (selectedFile && grayPixels && !loading) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Top bar */}
        <div className="flex items-center gap-3 px-4 py-2 bg-gray-900/70 border-b border-white/[0.06] shrink-0">
          <button
            onClick={resetFile}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-gray-300 bg-white/[0.06] hover:bg-white/[0.1] transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Change Image
          </button>

          <div className="flex items-center gap-2 ml-4">
            <label className="text-xs text-gray-400 font-medium whitespace-nowrap">
              Skeleton Threshold
            </label>
            <input
              type="range"
              min={30}
              max={230}
              step={1}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-40 h-1 accent-blue-500 cursor-pointer"
              title={`Binary threshold: ${threshold}`}
            />
            <span className="text-xs font-mono text-gray-300 w-7 text-center tabular-nums">
              {threshold}
            </span>
          </div>

          <div className="flex-1" />

          {/* OCR mode toggle in top bar */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-gray-500 font-medium">OCR:</span>
            <div className="flex rounded-lg bg-white/[0.04] border border-white/[0.08] p-0.5">
              <button
                onClick={() => setOcrMode("fast")}
                className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors ${
                  ocrMode === "fast"
                    ? "bg-gray-600 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                Fast
              </button>
              <button
                onClick={() => setOcrMode("slow")}
                className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors ${
                  ocrMode === "slow"
                    ? "bg-gray-600 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                Accurate
              </button>
            </div>
          </div>

          <span className="text-gray-500 font-mono text-xs tabular-nums">
            {imgDims.w} x {imgDims.h}
          </span>

          <button
            onClick={doAnalyze}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors"
          >
            Analyze Circuit <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Side-by-side preview */}
        <div className="flex-1 flex overflow-hidden">
          {/* Original */}
          <div className="flex-1 flex flex-col border-r border-white/[0.06] overflow-auto bg-black/40">
            <div className="px-3 py-1.5 bg-gray-900/60 border-b border-white/[0.06] text-[11px] font-semibold text-gray-400 uppercase tracking-wider shrink-0">
              Original
            </div>
            <div className="flex-1 overflow-auto flex items-start justify-center p-2">
              <img src={originalDataUrl} alt="Original" className="max-w-full max-h-full object-contain" draggable={false} />
            </div>
          </div>
          {/* Thresholded */}
          <div className="flex-1 flex flex-col overflow-auto bg-black/40">
            <div className="px-3 py-1.5 bg-gray-900/60 border-b border-white/[0.06] text-[11px] font-semibold text-gray-400 uppercase tracking-wider shrink-0">
              Thresholded (Binary)
            </div>
            <div className="flex-1 overflow-auto flex items-start justify-center p-2">
              <canvas ref={canvasRef} className="max-w-full max-h-full object-contain" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Phase 3: analyzing (loading) ──
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-10 px-6">
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-20 h-20">
            <svg className="w-20 h-20 -rotate-90" viewBox="0 0 64 64">
              <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="4" />
              <circle
                cx="32" cy="32" r="28" fill="none" stroke="url(#prog-grad)" strokeWidth="4"
                strokeDasharray={`${progress * 1.76} 176`}
                strokeLinecap="round"
                className="transition-all duration-300"
              />
              <defs>
                <linearGradient id="prog-grad" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#6366f1" />
                </linearGradient>
              </defs>
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-lg font-bold text-gray-300">
              {Math.round(progress)}%
            </span>
          </div>
          <span className="text-sm text-gray-300 animate-pulse">Analyzing circuit...</span>
        </div>

        <div className="w-[460px] text-center px-6 py-4 rounded-xl bg-white/[0.03] border border-white/[0.08] transition-all duration-500">
          <div className="flex items-center justify-center gap-2 mb-2">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Did you know?</span>
          </div>
          <p key={factIndex} className="text-sm text-gray-300 leading-relaxed animate-slide-up">
            {CIRCUIT_FACTS[factIndex]}
          </p>
        </div>
      </div>
    );
  }

  // ── Phase 1: initial upload prompt ──
  return (
    <div className="flex flex-col items-center justify-center h-full gap-10 px-6">
      {/* Logo area */}
      <div className="text-center space-y-3">
        <div className="flex items-center justify-center gap-3">
          <div className="w-11 h-11 rounded-lg bg-indigo-600 flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-white">
            Circuitron
          </h1>
        </div>
        <p className="text-gray-400 max-w-md leading-relaxed text-sm">
          Upload a hand-drawn circuit diagram to detect components,
          read labels, and trace wires — then get an interactive,
          simulatable schematic.
        </p>
      </div>

      {/* Drop zone */}
      <label
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFileSelected(file);
        }}
        className={`group relative flex flex-col items-center justify-center w-[420px] h-56 rounded-xl
          border-2 border-dashed transition-colors cursor-pointer
          ${dragOver
            ? "border-indigo-400 bg-indigo-500/10"
            : "border-gray-600 bg-white/[0.02] hover:border-gray-400 hover:bg-white/[0.04]"
          }`}
      >
        <div className="flex flex-col items-center gap-3">
          <div className="w-12 h-12 rounded-lg bg-white/[0.06] border border-white/10 flex items-center justify-center">
            <Upload className="w-6 h-6 text-gray-400" />
          </div>
          <div className="text-center">
            <span className="block text-sm text-gray-200 font-medium">
              Drop your circuit image here
            </span>
            <span className="block text-xs text-gray-500 mt-1">
              or click to browse — PNG, JPG, HEIC
            </span>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFileSelected(f);
          }}
        />
      </label>

      {/* OCR mode selector */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500 font-medium">OCR Mode:</span>
        <div className="flex rounded-lg bg-white/[0.04] border border-white/[0.08] p-0.5">
          <button
            onClick={() => setOcrMode("fast")}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              ocrMode === "fast"
                ? "bg-gray-600 text-white"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Fast
          </button>
          <button
            onClick={() => setOcrMode("slow")}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              ocrMode === "slow"
                ? "bg-gray-600 text-white"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Accurate
          </button>
        </div>
        <span className="text-[10px] text-gray-600 max-w-[180px] leading-tight">
          {ocrMode === "fast"
            ? "Lightweight & quick"
            : "Higher accuracy, slower"}
        </span>
      </div>

      {/* Feature pills */}
      <div className="flex gap-2.5 flex-wrap justify-center">
        {[
          [Search, "Component Detection"],
          [FileText, "Text Recognition"],
          [GitBranch, "Wire Tracing"],
          [Zap, "Live Simulation"],
        ].map(([Icon, label]) => {
          const LIcon = Icon as React.ElementType;
          return (
            <span key={label as string} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white/[0.03] border border-white/[0.06] text-xs text-gray-500">
              <LIcon className="w-3.5 h-3.5" /> {label as string}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─── Step 2: Review & Edit ──────────────────────────────────────────────────

function ReviewStep({
  preview,
  initialThreshold,
  onConfirm,
  onBack,
  toast,
}: {
  preview: AnalysisPreview;
  initialThreshold: number;
  onConfirm: (edited: EditedAnalysis) => void;
  onBack: () => void;
  toast: (msg: string, type: ToastType) => void;
}) {
  const [components, setComponents] = useState<DetectedComponent[]>(preview.components);
  const [texts, setTexts] = useState<DetectedText[]>(preview.texts);
  const [junctions, setJunctions] = useState<DetectedJunction[]>(preview.junctions);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>(preview.graph_nodes);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>(preview.graph_edges);
  const [overlayComponents, setOverlayComponents] = useState(true);
  const [overlayTexts, setOverlayTexts] = useState(true);
  const [overlayJunctions, setOverlayJunctions] = useState(true);
  const [overlayGraph, setOverlayGraph] = useState(false);

  // Threshold slider state
  const [threshold, setThreshold] = useState(initialThreshold);
  const [thresholdLoading, setThresholdLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Diagnostic images (can be updated by re-analysis)
  const [diagImages, setDiagImages] = useState(preview.diagnostic_images);

  type ViewMode = "annotated" | "original" | "skeleton" | "overlay" | "bbox" | "adjacency";
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");

  const viewImageMap: Record<ViewMode, string> = {
    annotated: preview.annotated_image,
    original: preview.original_image,
    skeleton: diagImages?.skeleton_png ?? "",
    overlay: diagImages?.overlay_png ?? "",
    bbox: diagImages?.bbox_png ?? "",
    adjacency: diagImages?.adjacency_graph_png ?? "",
  };

  // Re-analyze with new threshold (debounced, skips YOLO/OCR)
  const reAnalyze = useCallback(
    async (newThreshold: number) => {
      setThresholdLoading(true);
      try {
        const res = await fetch(`${API}/re-analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: preview.session_id,
            binary_thresh: newThreshold,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail ?? "Re-analysis failed");
        }
        const data = await res.json();

        // Update graph + images from re-analysis
        const graph = data.graph ?? {};
        if (graph.nodes) {
          setGraphNodes(
            graph.nodes.map((n: any) => ({ id: n.id, x: n.x, y: n.y }))
          );
        }
        if (graph.edges) {
          setGraphEdges(
            graph.edges.map((e: any) => ({
              source: e.source,
              target: e.target,
              path: (e.path ?? []).map((p: any) => ({ x: p.x, y: p.y })),
              linked_components: e.linked_components,
            }))
          );
        }
        const imgs = data.images ?? {};
        setDiagImages({
          skeleton_png: imgs.skeleton_png ?? "",
          overlay_png: imgs.overlay_png ?? "",
          bbox_png: imgs.bbox_png ?? "",
          adjacency_graph_png: imgs.adjacency_graph_png ?? "",
        });
      } catch (e: any) {
        toast(`Threshold update error: ${e.message}`, "error");
      } finally {
        setThresholdLoading(false);
      }
    },
    [preview.session_id, toast]
  );

  const handleThresholdChange = useCallback(
    (val: number) => {
      setThreshold(val);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => reAnalyze(val), 400);
    },
    [reAnalyze]
  );

  const rawImg = viewImageMap[viewMode];
  const imgSrc = !rawImg
    ? ""
    : rawImg.startsWith("data:")
    ? rawImg
    : `data:image/png;base64,${rawImg}`;
  const showSvgOverlay = viewMode === "original";

  const handleConfirm = () => {
    onConfirm({
      image_width: preview.image_width,
      image_height: preview.image_height,
      components,
      texts,
      junctions,
      graph_nodes: graphNodes,
      graph_edges: graphEdges,
    });
  };

  const updateComponent = (idx: number, field: string, val: string) => {
    setComponents((prev) =>
      prev.map((c, i) => (i === idx ? { ...c, [field]: val } : c))
    );
  };
  const deleteComponent = (idx: number) => {
    setComponents((prev) => prev.filter((_, i) => i !== idx));
    toast("Component removed", "info");
  };
  const updateText = (idx: number, val: string) =>
    setTexts((prev) =>
      prev.map((t, i) => (i === idx ? { ...t, ocr_text: val } : t))
    );
  const deleteText = (idx: number) => {
    setTexts((prev) => prev.filter((_, i) => i !== idx));
    toast("Text label removed", "info");
  };
  const deleteJunction = (idx: number) => {
    setJunctions((prev) => prev.filter((_, i) => i !== idx));
    toast("Junction removed", "info");
  };

  return (
    <div className="flex h-full">
      {/* Left: image viewer */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 backdrop-blur-md border-b border-white/[0.06] text-xs shrink-0">
          <IconBtn onClick={onBack}><ArrowLeft className="w-3.5 h-3.5" /> Back</IconBtn>

          <Divider />

          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as ViewMode)}
            className="bg-white/[0.06] border border-white/[0.08] text-white rounded-md px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="annotated">Annotated</option>
            <option value="original">Original + Overlays</option>
            <option value="skeleton">Skeleton</option>
            <option value="overlay">Wire Overlay</option>
            <option value="bbox">Detection Boxes</option>
            <option value="adjacency">Adjacency Graph</option>
          </select>

          {viewMode === "original" && (
            <>
              <Divider />
              {([
                [overlayComponents, setOverlayComponents, "Components", "text-emerald-400"],
                [overlayTexts, setOverlayTexts, "Texts", "text-amber-400"],
                [overlayJunctions, setOverlayJunctions, "Junctions", "text-rose-400"],
                [overlayGraph, setOverlayGraph, "Graph", "text-purple-400"],
              ] as [boolean, (v: boolean) => void, string, string][]).map(([checked, setter, label, color]) => (
                <label key={label} className={`flex items-center gap-1.5 cursor-pointer select-none ${color}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setter(e.target.checked)}
                    className="rounded w-3 h-3 accent-current"
                  />
                  <span className="text-[11px] font-medium">{label}</span>
                </label>
              ))}
            </>
          )}

          <Divider />

          {/* Threshold slider */}
          <div className="flex items-center gap-2">
            <label className="text-[11px] text-gray-400 font-medium whitespace-nowrap" title="Binary threshold for skeleton extraction. Lower = more detail, higher = less noise.">
              Threshold
            </label>
            <input
              type="range"
              min={30}
              max={230}
              step={5}
              value={threshold}
              onChange={(e) => handleThresholdChange(Number(e.target.value))}
              className="w-24 h-1 accent-blue-500 cursor-pointer"
              title={`Binary threshold: ${threshold}`}
            />
            <span className="text-[11px] font-mono text-gray-300 w-7 text-center tabular-nums">
              {threshold}
            </span>
            {thresholdLoading && (
              <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            )}
          </div>

          <div className="flex-1" />

          <span className="text-gray-500 font-mono tabular-nums">
            {preview.image_width} × {preview.image_height}
          </span>

          <IconBtn onClick={handleConfirm} variant="primary">
            Generate Schematic <ArrowRight className="w-3.5 h-3.5" />
          </IconBtn>
        </div>

        {/* Image canvas */}
        <div className="relative flex-1 overflow-auto bg-black/40">
          <img src={imgSrc} alt="Circuit analysis" className="block max-w-full" draggable={false} />
          <svg
            className="absolute top-0 left-0"
            width={preview.image_width}
            height={preview.image_height}
            style={{ pointerEvents: "none" }}
          >
            {showSvgOverlay && overlayGraph && graphEdges.map((e, i) => {
              const src = graphNodes.find((n) => n.id === e.source);
              const tgt = graphNodes.find((n) => n.id === e.target);
              if (!src || !tgt) return null;

              // Determine color based on component linkage
              const hasSourceComp = (e.linked_components?.source_components?.length ?? 0) > 0;
              const hasTargetComp = (e.linked_components?.target_components?.length ?? 0) > 0;
              const edgeColor = hasSourceComp && hasTargetComp
                ? "#22c55e"  // both ends linked - green
                : hasSourceComp || hasTargetComp
                ? "#f59e0b"  // one end linked - amber
                : "#d946ef"; // no component link - purple

              if (e.path && e.path.length > 2) {
                const pts = e.path.map((p) => `${p.x},${p.y}`).join(" ");
                return (
                  <polyline key={`edge-${i}`} points={pts}
                    stroke={edgeColor} strokeWidth={2} fill="none" opacity={0.7} />
                );
              }

              return (
                <line key={`edge-${i}`} x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
                  stroke={edgeColor} strokeWidth={1.5} opacity={0.6} />
              );
            })}
            {showSvgOverlay && overlayGraph && graphNodes.map((n, i) => {
              // Check if this node has any linked components
              const linkedEdges = graphEdges.filter(
                (e) => e.source === n.id || e.target === n.id
              );
              const hasComponent = linkedEdges.some((e) => {
                const lc = e.linked_components;
                if (!lc) return false;
                if (e.source === n.id && lc.source_components?.length) return true;
                if (e.target === n.id && lc.target_components?.length) return true;
                return false;
              });
              return (
                <circle key={`gnode-${i}`} cx={n.x} cy={n.y} r={hasComponent ? 4.5 : 3.5}
                  fill={hasComponent ? "#ef4444" : "#a855f7"}
                  stroke="#1a1a2e" strokeWidth={1} />
              );
            })}
            {showSvgOverlay && overlayComponents && components.map((c, i) => {
              // Find edges linked to this component's bbox
              const compBbox = c.bbox;
              const linkedEdgeIndices: number[] = [];
              graphEdges.forEach((e, ei) => {
                const lc = e.linked_components;
                if (!lc) return;
                const matches = [
                  ...lc.source_components ?? [],
                  ...lc.target_components ?? [],
                ].some((lcomp) => {
                  if (!lcomp.bbox || lcomp.bbox.length < 4) return false;
                  return (
                    Math.abs(lcomp.bbox[0] - compBbox.x1) < 5 &&
                    Math.abs(lcomp.bbox[1] - compBbox.y1) < 5 &&
                    Math.abs(lcomp.bbox[2] - compBbox.x2) < 5 &&
                    Math.abs(lcomp.bbox[3] - compBbox.y2) < 5
                  );
                });
                if (matches) linkedEdgeIndices.push(ei);
              });
              const isLinked = linkedEdgeIndices.length > 0;
              return (
                <g key={`comp-${i}`}>
                  <rect x={compBbox.x1} y={compBbox.y1}
                    width={compBbox.x2 - compBbox.x1} height={compBbox.y2 - compBbox.y1}
                    fill={isLinked ? "rgba(34,197,94,0.10)" : "rgba(34,197,94,0.04)"}
                    stroke={isLinked ? "#22c55e" : "#6b7280"}
                    strokeWidth={isLinked ? 2 : 1.5}
                    rx={4}
                    strokeDasharray={isLinked ? undefined : "4,3"}
                  />
                  <text x={compBbox.x1 + 4} y={compBbox.y1 - 5}
                    fill={isLinked ? "#22c55e" : "#9ca3af"}
                    fontSize={11} fontWeight="600" fontFamily="monospace">
                    {c.id} {c.type}{c.value ? ` (${c.value})` : ""}
                    {isLinked ? ` •${linkedEdgeIndices.length}` : ""}
                  </text>
                  {/* Connection indicators - small dots where edges touch this bbox */}
                  {isLinked && linkedEdgeIndices.map((ei) => {
                    const edge = graphEdges[ei];
                    const srcNode = graphNodes.find((n) => n.id === edge.source);
                    const tgtNode = graphNodes.find((n) => n.id === edge.target);
                    const touchPoints: { x: number; y: number }[] = [];
                    if (srcNode && srcNode.x >= compBbox.x1 - 10 && srcNode.x <= compBbox.x2 + 10 &&
                        srcNode.y >= compBbox.y1 - 10 && srcNode.y <= compBbox.y2 + 10) {
                      touchPoints.push({ x: srcNode.x, y: srcNode.y });
                    }
                    if (tgtNode && tgtNode.x >= compBbox.x1 - 10 && tgtNode.x <= compBbox.x2 + 10 &&
                        tgtNode.y >= compBbox.y1 - 10 && tgtNode.y <= compBbox.y2 + 10) {
                      touchPoints.push({ x: tgtNode.x, y: tgtNode.y });
                    }
                    return touchPoints.map((tp, tpi) => (
                      <circle key={`comp-${i}-edge-${ei}-tp-${tpi}`}
                        cx={tp.x} cy={tp.y} r={4}
                        fill="#22c55e" stroke="white" strokeWidth={1.5} opacity={0.9} />
                    ));
                  })}
                </g>
              );
            })}
            {showSvgOverlay && overlayTexts && texts.map((t, i) => (
              <g key={`txt-${i}`}>
                <rect x={t.bbox.x1} y={t.bbox.y1}
                  width={t.bbox.x2 - t.bbox.x1} height={t.bbox.y2 - t.bbox.y1}
                  fill="rgba(251,191,36,0.08)" stroke="#fbbf24" strokeWidth={1.5} rx={2}
                />
                <text x={t.bbox.x1 + 2} y={t.bbox.y1 - 4} fill="#fbbf24" fontSize={10} fontWeight="500">
                  {t.ocr_text}
                </text>
              </g>
            ))}
            {showSvgOverlay && overlayJunctions && junctions.map((j, i) => (
              <g key={`junc-${i}`}>
                <circle cx={j.position.x} cy={j.position.y} r={8} fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.7} />
                <circle cx={j.position.x} cy={j.position.y} r={3} fill="#ef4444" opacity={0.9} />
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Right: editable sidebar */}
      <aside className="w-[400px] min-w-[340px] border-l border-white/[0.06] bg-gray-900/60 backdrop-blur-sm overflow-y-auto">
        {/* Components */}
        <SidebarSection
          title="Components"
          count={components.length}
          color="emerald"
          defaultOpen
        >
          <div className="space-y-2">
            {components.map((c, i) => (
              <div key={c.id} className="rounded-lg border border-white/[0.06] bg-white/[0.03] p-2.5 hover:bg-white/[0.05] transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-emerald-400 text-xs font-semibold">{c.id}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-500 bg-white/[0.06] px-1.5 py-0.5 rounded-full">
                      {(c.confidence * 100).toFixed(0)}%
                    </span>
                    <button onClick={() => deleteComponent(i)} className="text-red-400/60 hover:text-red-400 transition-colors" title="Delete"><X className="w-3.5 h-3.5" /></button>
                  </div>
                </div>
                <div className="grid grid-cols-[60px_1fr] gap-x-2 gap-y-1.5 text-xs">
                  {(["type", "value", "name"] as const).map((field) => (
                    <React.Fragment key={field}>
                      <label className="text-gray-500 capitalize leading-6">{field}</label>
                      <input
                        className="bg-white/[0.06] border border-white/[0.06] rounded-md px-2 py-1 text-white text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                        value={(c as any)[field] ?? ""}
                        onChange={(e) => updateComponent(i, field, e.target.value)}
                      />
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ))}
            {components.length === 0 && <EmptyMsg>No components detected</EmptyMsg>}
          </div>
        </SidebarSection>

        {/* OCR Texts */}
        <SidebarSection title="OCR Texts" count={texts.length} color="amber" defaultOpen>
          <div className="space-y-1.5">
            {texts.map((t, i) => (
              <div key={t.id} className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] p-2 hover:bg-white/[0.05] transition-colors">
                <input
                  className="flex-1 bg-white/[0.06] border border-white/[0.06] rounded-md px-2 py-1 text-amber-200 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-amber-500/50"
                  value={t.ocr_text}
                  onChange={(e) => updateText(i, e.target.value)}
                />
                <span className="text-[10px] text-gray-500 bg-white/[0.06] px-1.5 py-0.5 rounded-full w-8 text-center shrink-0">
                  {(t.ocr_confidence * 100).toFixed(0)}%
                </span>
                <button onClick={() => deleteText(i)} className="text-red-400/60 hover:text-red-400 transition-colors" title="Delete"><X className="w-3.5 h-3.5" /></button>
              </div>
            ))}
            {texts.length === 0 && <EmptyMsg>No text detected</EmptyMsg>}
          </div>
        </SidebarSection>

        {/* Junctions */}
        <SidebarSection title="Junctions" count={junctions.length} color="rose">
          <div className="space-y-1.5">
            {junctions.map((j, i) => (
              <div key={j.id} className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] p-2 hover:bg-white/[0.05] transition-colors">
                <span className="flex-1 text-gray-300 text-xs">
                  <span className="text-rose-400 font-medium">{j.type}</span> at ({j.position.x.toFixed(0)}, {j.position.y.toFixed(0)})
                </span>
                <span className="text-[10px] text-gray-500 bg-white/[0.06] px-1.5 py-0.5 rounded-full">{(j.confidence * 100).toFixed(0)}%</span>
                <button onClick={() => deleteJunction(i)} className="text-red-400/60 hover:text-red-400 transition-colors" title="Delete"><X className="w-3.5 h-3.5" /></button>
              </div>
            ))}
            {junctions.length === 0 && <EmptyMsg>No junctions detected</EmptyMsg>}
          </div>
        </SidebarSection>

        {/* Graph */}
        <SidebarSection title="Graph" count={graphNodes.length} color="purple">
          <div className="text-xs text-gray-400 space-y-1">
            <div className="flex gap-4">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-purple-400" /> {graphNodes.length} nodes</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-0.5 bg-purple-400" /> {graphEdges.length} edges</span>
            </div>
            <p className="text-gray-500">Skeleton graph from wire traces.</p>
          </div>
        </SidebarSection>

        {/* Diagnostic images gallery */}
        <SidebarSection title="Visualisations" count={0} color="cyan" defaultOpen>
          <div className="grid grid-cols-2 gap-2">
            {([
              ["skeleton", "Skeleton", diagImages?.skeleton_png],
              ["overlay", "Wire Overlay", diagImages?.overlay_png],
              ["bbox", "Detection", diagImages?.bbox_png],
              ["adjacency", "Adjacency", diagImages?.adjacency_graph_png],
            ] as [string, string, string | undefined][]).map(([key, label, b64]) =>
              b64 ? (
                <button
                  key={key}
                  className={`group rounded-lg border overflow-hidden transition-all duration-200
                    ${viewMode === key
                      ? "border-cyan-500/60 ring-1 ring-cyan-500/30"
                      : "border-white/[0.06] hover:border-white/[0.15]"
                    }`}
                  onClick={() => setViewMode(key as ViewMode)}
                >
                  <img
                    src={b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`}
                    alt={label}
                    className="w-full aspect-[4/3] object-cover"
                    draggable={false}
                  />
                  <div className="px-2 py-1 text-[10px] text-gray-400 bg-black/40 group-hover:text-gray-200 transition-colors">
                    {label}
                  </div>
                </button>
              ) : null
            )}
          </div>
        </SidebarSection>
      </aside>
    </div>
  );
}

// ─── Sidebar helpers ────────────────────────────────────────────────────────

const sectionColors: Record<string, string> = {
  emerald: "text-emerald-400",
  amber: "text-amber-400",
  rose: "text-rose-400",
  purple: "text-purple-400",
  cyan: "text-cyan-400",
};

function SidebarSection({
  title,
  count,
  color,
  defaultOpen,
  children,
}: {
  title: string;
  count: number;
  color: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <details open={defaultOpen} className="group border-b border-white/[0.04]">
      <summary className={`flex items-center gap-2 px-4 py-2.5 cursor-pointer select-none hover:bg-white/[0.03] transition-colors ${sectionColors[color] ?? "text-white"}`}>
        <svg className="w-3 h-3 transition-transform group-open:rotate-90 text-gray-500" fill="none" viewBox="0 0 12 12">
          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-semibold text-xs">{title}</span>
        {count > 0 && (
          <span className="ml-auto text-[10px] bg-white/[0.06] text-gray-400 px-1.5 py-0.5 rounded-full font-mono">
            {count}
          </span>
        )}
      </summary>
      <div className="px-3 pb-3">{children}</div>
    </details>
  );
}

function EmptyMsg({ children }: { children: React.ReactNode }) {
  return <p className="text-gray-600 text-center py-4 text-xs">{children}</p>;
}

// ─── Step 3: CircuitJS1 Schematic Editor ────────────────────────────────────

// ─── Simulation Scope: live time-series waveforms ───────────────────────────

const SCOPE_COLORS = [
  "#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#fb923c",
  "#60a5fa", "#e879f9", "#4ade80", "#f87171",
];

interface ScopeTrace {
  label: string;
  kind: "V" | "I";
  color: string;
  data: { t: number; v: number }[];
}

function SimulationScope({
  sim,
  running,
}: {
  sim: CircuitJSAPI | null;
  running: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tracesRef = useRef<ScopeTrace[]>([]);
  const [traces, setTraces] = useState<ScopeTrace[]>([]);
  const [maxPoints] = useState(500);
  const [hoveredTrace, setHoveredTrace] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const animRef = useRef<number>(0);
  const lastSampleRef = useRef<number>(0);

  // Discover elements when sim is available
  useEffect(() => {
    if (!sim) return;
    const elements = sim.getElements();
    const newTraces: ScopeTrace[] = [];
    let colorIdx = 0;

    elements.forEach((el: CircuitJSElement, i: number) => {
      let name = `E${i}`;
      try { if (typeof el.getLabelName === "function") name = el.getLabelName() || name; } catch { /* GWT obfuscated */ }
      // Add voltage trace
      newTraces.push({
        label: `${name} V`,
        kind: "V",
        color: SCOPE_COLORS[colorIdx % SCOPE_COLORS.length],
        data: [],
      });
      colorIdx++;
      // Add current trace
      newTraces.push({
        label: `${name} I`,
        kind: "I",
        color: SCOPE_COLORS[colorIdx % SCOPE_COLORS.length],
        data: [],
      });
      colorIdx++;
    });

    tracesRef.current = newTraces;
    setTraces([...newTraces]);
  }, [sim]);

  // Sample data from sim on each animation frame
  useEffect(() => {
    if (!sim) return;

    const sample = () => {
      if (!sim.isRunning()) {
        animRef.current = requestAnimationFrame(sample);
        return;
      }

      const now = sim.getTime();
      // Avoid duplicate samples at same time
      if (now === lastSampleRef.current) {
        animRef.current = requestAnimationFrame(sample);
        return;
      }
      lastSampleRef.current = now;

      const elements = sim.getElements();
      const tr = tracesRef.current;

      elements.forEach((el: CircuitJSElement, i: number) => {
        const vTraceIdx = i * 2;
        const iTraceIdx = i * 2 + 1;
        if (vTraceIdx < tr.length) {
          tr[vTraceIdx].data.push({ t: now, v: el.getVoltageDiff() });
          if (tr[vTraceIdx].data.length > maxPoints) tr[vTraceIdx].data.shift();
        }
        if (iTraceIdx < tr.length) {
          tr[iTraceIdx].data.push({ t: now, v: el.getCurrent() });
          if (tr[iTraceIdx].data.length > maxPoints) tr[iTraceIdx].data.shift();
        }
      });

      setTraces([...tr]);
      animRef.current = requestAnimationFrame(sample);
    };

    animRef.current = requestAnimationFrame(sample);
    return () => cancelAnimationFrame(animRef.current);
  }, [sim, maxPoints]);

  // Draw waveforms on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Background
    ctx.fillStyle = "#0d0d1a";
    ctx.fillRect(0, 0, W, H);

    // Visible traces
    const visibleTraces = traces.filter((_, i) => hoveredTrace === null || hoveredTrace === i);
    if (visibleTraces.length === 0 || visibleTraces.every((t) => t.data.length === 0)) {
      ctx.fillStyle = "#4b5563";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText(running ? "Waiting for data…" : "Paused — no data yet", W / 2, H / 2);
      return;
    }

    // Figure out time range across all data
    let tMin = Infinity, tMax = -Infinity;
    let vMin = Infinity, vMax = -Infinity;
    visibleTraces.forEach((tr) => {
      tr.data.forEach((d) => {
        if (d.t < tMin) tMin = d.t;
        if (d.t > tMax) tMax = d.t;
        if (isFinite(d.v)) {
          if (d.v < vMin) vMin = d.v;
          if (d.v > vMax) vMax = d.v;
        }
      });
    });

    if (tMax === tMin) tMax = tMin + 1e-6;
    const vRange = vMax - vMin || 1;
    const vPad = vRange * 0.1;
    vMin -= vPad;
    vMax += vPad;

    const pad = { top: 20, right: 60, bottom: 30, left: 10 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const tToX = (t: number) => pad.left + ((t - tMin) / (tMax - tMin)) * plotW;
    const vToY = (v: number) => pad.top + (1 - (v - vMin) / (vMax - vMin)) * plotH;

    // Grid lines
    ctx.strokeStyle = "#1f2937";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      const val = vMax - (i / 4) * (vMax - vMin);
      let label: string;
      if (Math.abs(val) >= 1000) label = (val / 1000).toFixed(1) + "k";
      else if (Math.abs(val) >= 1) label = val.toFixed(2);
      else if (Math.abs(val) >= 0.001) label = (val * 1000).toFixed(1) + "m";
      else label = (val * 1e6).toFixed(0) + "µ";
      ctx.fillText(label, W - pad.right + 50, y + 3);
    }

    // Zero line
    if (vMin < 0 && vMax > 0) {
      const zeroY = vToY(0);
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(W - pad.right, zeroY); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Time axis label
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    const tRange = tMax - tMin;
    let timeUnit = "s", tScale = 1;
    if (tRange < 1e-3) { timeUnit = "µs"; tScale = 1e6; }
    else if (tRange < 1) { timeUnit = "ms"; tScale = 1e3; }
    ctx.fillText(`${(tMin * tScale).toFixed(1)} – ${(tMax * tScale).toFixed(1)} ${timeUnit}`, W / 2, H - 4);

    // Draw traces
    visibleTraces.forEach((tr) => {
      if (tr.data.length < 2) return;
      ctx.strokeStyle = tr.color;
      ctx.lineWidth = hoveredTrace !== null ? 2 : 1.2;
      ctx.globalAlpha = hoveredTrace !== null && traces.indexOf(tr) !== hoveredTrace ? 0.15 : 1;
      ctx.beginPath();
      tr.data.forEach((d, j) => {
        const x = tToX(d.t);
        const y = vToY(isFinite(d.v) ? d.v : 0);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // Cursor crosshair
    if (mousePos && mousePos.x >= pad.left && mousePos.x <= W - pad.right) {
      ctx.strokeStyle = "#4b5563";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(mousePos.x, pad.top); ctx.lineTo(mousePos.x, pad.top + plotH); ctx.stroke();
      ctx.setLineDash([]);

      // Value readouts
      const cursorT = tMin + ((mousePos.x - pad.left) / plotW) * (tMax - tMin);
      let readY = pad.top + 4;
      visibleTraces.forEach((tr) => {
        // Find nearest data point
        let closest = tr.data[0];
        let bestDist = Infinity;
        tr.data.forEach((d) => {
          const dist = Math.abs(d.t - cursorT);
          if (dist < bestDist) { bestDist = dist; closest = d; }
        });
        if (closest) {
          ctx.fillStyle = tr.color;
          ctx.font = "10px monospace";
          ctx.textAlign = "left";
          const vStr = Math.abs(closest.v) >= 1 ? closest.v.toFixed(3)
            : Math.abs(closest.v) >= 0.001 ? (closest.v * 1000).toFixed(2) + "m"
            : (closest.v * 1e6).toFixed(1) + "µ";
          ctx.fillText(`${tr.label}: ${vStr}`, pad.left + 4, readY += 13);
        }
      });
    }
  }, [traces, hoveredTrace, mousePos, running]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  const handleMouseLeave = useCallback(() => setMousePos(null), []);

  return (
    <div className="flex flex-col h-full">
      {/* Legend */}
      <div className="flex flex-wrap gap-1.5 px-3 py-2 border-b border-white/[0.06] shrink-0 max-h-[100px] overflow-y-auto">
        {traces.map((tr, i) => (
          <button
            key={i}
            className={`text-[10px] px-1.5 py-0.5 rounded font-mono transition-all border ${
              hoveredTrace === i
                ? "bg-white/10 border-white/20"
                : hoveredTrace !== null
                ? "opacity-30 border-transparent"
                : "border-transparent hover:bg-white/5"
            }`}
            style={{ color: tr.color }}
            onMouseEnter={() => setHoveredTrace(i)}
            onMouseLeave={() => setHoveredTrace(null)}
          >
            {tr.kind === "V" ? <Zap className="w-3 h-3 inline" /> : <Activity className="w-3 h-3 inline" />} {tr.label}
          </button>
        ))}
        {traces.length === 0 && (
          <span className="text-gray-600 text-[10px]">No elements detected yet</span>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
      </div>
    </div>
  );
}

function SchematicStep({
  circuit,
  cjsText,
  onBack,
  toast,
}: {
  circuit: Circuit;
  cjsText: string;
  onBack: () => void;
  toast: (msg: string, type: ToastType) => void;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [sim, setSim] = useState<CircuitJSAPI | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [running, setRunning] = useState(true);
  const [simTime, setSimTime] = useState(0);
  const [elementCount, setElementCount] = useState(0);
  const [showGraph, setShowGraph] = useState(false);

  // ── AI Chat state ───────────────────────────────────────────────────
  const [chatOpen, setChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, chatLoading]);

  const sendChat = useCallback(async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;

    const userMsg = { role: "user" as const, content: msg };
    setChatHistory((h) => [...h, userMsg]);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          circuit,
          message: msg,
          history: chatHistory.slice(-6),
          cjs_text: cjsText,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Chat failed");
      }
      const { reply } = await res.json();
      setChatHistory((h) => [...h, { role: "assistant", content: reply }]);
    } catch (e: any) {
      setChatHistory((h) => [...h, { role: "assistant", content: `⚠ Error: ${e.message}` }]);
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading, circuit, chatHistory, cjsText]);

  const iframeSrc = `/circuitjs/circuitjs.html?cct=${encodeURIComponent(cjsText)}`;

  const onIframeLoad = useCallback(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;
    const win = iframe.contentWindow as CircuitJSWindow | null;
    if (!win) return;

    win.oncircuitjsloaded = () => {
      const api = win.CircuitJS1;
      setSim(api);
      setLoaded(true);

      if (cjsText) {
        api.importCircuit(cjsText, false);
      }

      api.onupdate = (s) => {
        setSimTime(s.getTime());
        setRunning(s.isRunning());
      };
      api.onanalyze = (s) => {
        setElementCount(s.getElements().length);
      };

      toast("Circuit loaded in simulator", "success");
    };

    if (win.CircuitJS1) {
      win.oncircuitjsloaded?.();
    }
  }, [cjsText, toast]);

  const toggleRun = () => {
    if (!sim) return;
    sim.setSimRunning(!running);
    setRunning(!running);
  };

  const handleReimport = () => {
    if (!sim) return;
    sim.importCircuit(cjsText);
    toast("Circuit re-imported", "info");
  };

  const downloadFile = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  const cid = circuit.circuit_id.slice(0, 8);

  const handleExportCircuit = () => {
    if (!sim) return;
    downloadFile(new Blob([sim.exportCircuit()], { type: "text/plain" }), `circuit-${cid}.txt`);
    toast("Exported as .txt", "success");
  };

  const handleExportSVG = () => {
    if (!sim) return;
    sim.onsvgrendered = (_s, svg) => {
      downloadFile(new Blob([svg], { type: "image/svg+xml" }), `circuit-${cid}.svg`);
      toast("Exported as SVG", "success");
    };
    sim.getCircuitAsSVG();
  };

  const handleExportJSON = () => {
    downloadFile(
      new Blob([JSON.stringify(circuit, null, 2)], { type: "application/json" }),
      `circuit-${cid}.json`
    );
    toast("Exported as JSON", "success");
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 px-4 py-2 bg-gray-900/80 backdrop-blur-md border-b border-white/[0.06] text-xs shrink-0">
        <IconBtn onClick={onBack}><ArrowLeft className="w-3.5 h-3.5" /> Review</IconBtn>

        <Divider />

        {/* Simulation controls */}
        <IconBtn onClick={toggleRun} disabled={!loaded} variant={running ? "danger" : "success"}>
          {running ? <><Pause className="w-3.5 h-3.5" /> Pause</> : <><Play className="w-3.5 h-3.5" /> Run</>}
        </IconBtn>
        <IconBtn onClick={() => { if (sim) sim.setMaxTimeStep(sim.getMaxTimeStep() / 2); }} disabled={!loaded} title="Halve timestep">
          <ChevronDown className="w-3.5 h-3.5" /> ½
        </IconBtn>
        <IconBtn onClick={() => { if (sim) sim.setMaxTimeStep(sim.getMaxTimeStep() * 2); }} disabled={!loaded} title="Double timestep">
          <ChevronUp className="w-3.5 h-3.5" /> 2×
        </IconBtn>

        <Divider />

        <IconBtn onClick={handleReimport} disabled={!loaded} variant="warning" title="Re-import original circuit">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </IconBtn>

        <Divider />

        {/* Export group */}
        <IconBtn onClick={handleExportCircuit} disabled={!loaded}><FileText className="w-3.5 h-3.5" /> .txt</IconBtn>
        <IconBtn onClick={handleExportSVG} disabled={!loaded}><ImageIcon className="w-3.5 h-3.5" /> SVG</IconBtn>
        <IconBtn onClick={handleExportJSON}><Clipboard className="w-3.5 h-3.5" /> JSON</IconBtn>



        <div className="flex-1" />

        {/* Status bar */}
        {loaded ? (
          <div className="flex items-center gap-3 text-gray-400 font-mono tabular-nums">
            <span>t = {simTime.toExponential(2)}s</span>
            <span className="text-gray-600">·</span>
            <span>{elementCount} elements</span>
            <span className="text-gray-600">·</span>
            {running ? (
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400">running</span>
              </span>
            ) : (
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <span className="text-amber-400">paused</span>
              </span>
            )}
          </div>
        ) : (
          <span className="text-gray-500 animate-pulse flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            Loading simulator…
          </span>
        )}
      </div>

      {/* Main content: CircuitJS1 iframe */}
      <div className="flex-1 flex overflow-hidden">
        {/* CircuitJS1 iframe */}
        <iframe
          ref={iframeRef}
          src={iframeSrc}
          onLoad={onIframeLoad}
          className="border-0 w-full"
          allow="fullscreen"
          title="Circuit Editor & Simulator"
        />
      </div>

      {/* ── AI Chat floating window ──────────────────────────────────── */}
      {chatOpen && (
        <div className="absolute bottom-20 right-6 w-[380px] h-[480px] bg-gray-900/95 backdrop-blur-md border border-white/[0.08] rounded-xl shadow-2xl flex flex-col z-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center gap-2 px-4 py-3 border-b border-white/[0.06] bg-gray-800/60 shrink-0">
            <span className="w-6 h-6 rounded-md bg-indigo-600 flex items-center justify-center"><MessageCircle className="w-3.5 h-3.5 text-white" /></span>
            <span className="text-sm font-semibold text-white/90">Circuit Assistant</span>
            <div className="flex-1" />
            <button
              onClick={() => setChatHistory([])}
              className="text-gray-500 hover:text-gray-300 text-xs px-1.5 py-0.5 rounded hover:bg-white/[0.06] transition-colors"
              title="Clear chat"
            >
              Clear
            </button>
            <button
              onClick={() => setChatOpen(false)}
              className="text-gray-500 hover:text-white leading-none px-1 rounded hover:bg-white/[0.06] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scrollbar-thin">
            {chatHistory.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center gap-3 opacity-60">
                <Lightbulb className="w-8 h-8 text-gray-400" />
                <p className="text-sm text-gray-400">Ask anything about your circuit</p>
                <div className="flex flex-wrap justify-center gap-1.5">
                  {["What does this circuit do?", "Any issues?", "Suggest improvements"].map((q) => (
                    <button
                      key={q}
                      onClick={() => { setChatInput(q); }}
                      className="text-[10px] px-2 py-1 rounded-md bg-white/[0.04] border border-white/[0.08] text-gray-400 hover:text-white hover:border-gray-500 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {chatHistory.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[85%] px-3 py-2 rounded-xl text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-blue-600/30 text-blue-100 rounded-br-md whitespace-pre-wrap"
                    : "bg-white/[0.06] text-gray-200 rounded-bl-md prose prose-invert prose-sm max-w-none"
                }`}>
                  {m.role === "user" ? m.content : <ReactMarkdown>{m.content}</ReactMarkdown>}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-white/[0.06] px-3 py-2 rounded-lg rounded-bl-md flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="px-3 py-3 border-t border-white/[0.06] bg-gray-900/80 shrink-0">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                placeholder="Ask about this circuit…"
                className="flex-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
                disabled={chatLoading}
              />
              <button
                onClick={sendChat}
                disabled={chatLoading || !chatInput.trim()}
                className="w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 flex items-center justify-center text-white text-sm disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Chat toggle FAB */}
      <button
        onClick={() => setChatOpen(!chatOpen)}
        className={`absolute bottom-6 right-6 w-12 h-12 rounded-full shadow-lg flex items-center justify-center text-lg transition-colors z-50 ${
          chatOpen
            ? "bg-gray-700 hover:bg-gray-600 text-white"
            : "bg-indigo-600 hover:bg-indigo-500 text-white"
        }`}
        title="Circuit Assistant"
      >
        {chatOpen ? <X className="w-5 h-5" /> : <MessageCircle className="w-5 h-5" />}
      </button>
    </div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────────────

export default function WorkspacePage() {
  const [step, setStep] = useState<Step>("upload");
  const [preview, setPreview] = useState<AnalysisPreview | null>(null);
  const [circuit, setCircuit] = useState<Circuit | null>(null);
  const [cjsText, setCjsText] = useState<string>("");
  const [finalizing, setFinalizing] = useState(false);
  const [uploadThreshold, setUploadThreshold] = useState(110);
  const { toasts, push: toast, dismiss: dismissToast } = useToast();

  const handleAnalyzed = useCallback((data: AnalysisPreview, thresholdUsed: number) => {
    setPreview(data);
    setUploadThreshold(thresholdUsed);
    setStep("review");
  }, []);

  const handleConfirm = useCallback(async (edited: EditedAnalysis) => {
    setFinalizing(true);
    try {
      const res = await fetch(`${API}/finalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(edited),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Finalize failed");
      }
      const data: Circuit = await res.json();
      setCircuit(data);

      const cjsRes = await fetch(`${API}/export-cjs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!cjsRes.ok) {
        const err = await cjsRes.json().catch(() => ({ detail: cjsRes.statusText }));
        throw new Error(err.detail ?? "CJS export failed");
      }
      const { cjs_text } = await cjsRes.json();
      setCjsText(cjs_text);

      toast(`Schematic generated — ${data.components.length} components, ${data.nodes.length} nodes`, "success");
      setStep("schematic");
    } catch (e: any) {
      toast(`Error: ${e.message}`, "error");
    } finally {
      setFinalizing(false);
    }
  }, [toast]);

  const goBackToUpload = useCallback(() => { setStep("upload"); setPreview(null); setCircuit(null); setCjsText(""); }, []);
  const goBackToReview = useCallback(() => { setStep("review"); setCircuit(null); setCjsText(""); }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-[#111116] text-white overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 border-b border-white/[0.06] shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-2 mr-4">
          <div className="w-6 h-6 rounded-md bg-indigo-600 flex items-center justify-center">
            <Zap className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight text-white/90">Circuitron</span>
        </div>

        {/* Step indicator */}
        <nav className="flex items-center gap-1">
          {(["upload", "review", "schematic"] as Step[]).map((s, i) => {
            const meta = stepMeta[s];
            const isActive = step === s;
            const isPast = (["upload", "review", "schematic"] as Step[]).indexOf(step) > i;
            return (
              <React.Fragment key={s}>
                {i > 0 && (
                  <svg className={`w-4 h-4 ${isPast ? "text-indigo-500" : "text-gray-700"}`} fill="none" viewBox="0 0 16 16">
                    <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
                <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs transition-colors
                  ${isActive ? "bg-indigo-600/20 text-indigo-300 font-semibold ring-1 ring-indigo-500/30" : ""}
                  ${isPast ? "text-indigo-400/60" : ""}
                  ${!isActive && !isPast ? "text-gray-600" : ""}
                `}>
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold
                    ${isActive ? "bg-indigo-600 text-white" : ""}
                    ${isPast ? "bg-indigo-500/20 text-indigo-400" : ""}
                    ${!isActive && !isPast ? "bg-white/[0.06] text-gray-600" : ""}
                  `}>
                    {isPast ? <Check className="w-2.5 h-2.5" /> : meta.icon}
                  </span>
                  {meta.label}
                </span>
              </React.Fragment>
            );
          })}
        </nav>

        <div className="flex-1" />

        {finalizing && (
          <span className="flex items-center gap-2 text-xs text-gray-400">
            <span className="inline-block w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
            Generating schematic…
          </span>
        )}
      </header>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {step === "upload" && <UploadStep onAnalyzed={handleAnalyzed} toast={toast} />}
        {step === "review" && preview && <ReviewStep preview={preview} initialThreshold={uploadThreshold} onConfirm={handleConfirm} onBack={goBackToUpload} toast={toast} />}
        {step === "schematic" && circuit && cjsText && <SchematicStep circuit={circuit} cjsText={cjsText} onBack={goBackToReview} toast={toast} />}
      </div>

      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
