"use client";

import React, { useEffect, useRef, useState, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowLeft, Pause, Play, ChevronDown, ChevronUp } from "lucide-react";

// ── Types for CircuitJS1 JS API ─────────────────────────────────────────────

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

// Extend the iframe's contentWindow
interface CircuitJSWindow extends Window {
  CircuitJS1: CircuitJSAPI;
  oncircuitjsloaded?: () => void;
}

// ── Page Component ──────────────────────────────────────────────────────────

function SimulateContent() {
  const searchParams = useSearchParams();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [sim, setSim] = useState<CircuitJSAPI | null>(null);
  const [running, setRunning] = useState(true);
  const [simTime, setSimTime] = useState(0);
  const [elementCount, setElementCount] = useState(0);
  const [loaded, setLoaded] = useState(false);

  // Build the iframe src with the circuit text from query params
  const circuitText = searchParams.get("cjs");
  const iframeSrc = circuitText
    ? `/circuitjs/circuitjs.html?startCircuitText=${encodeURIComponent(circuitText)}`
    : "/circuitjs/circuitjs.html";

  // Set up the CircuitJS1 JS API callback once the iframe loads
  const onIframeLoad = useCallback(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    const win = iframe.contentWindow as CircuitJSWindow | null;
    if (!win) return;

    // CircuitJS1 calls oncircuitjsloaded when the GWT module is ready
    win.oncircuitjsloaded = () => {
      const api = win.CircuitJS1;
      setSim(api);
      setLoaded(true);

      api.onupdate = (s) => {
        setSimTime(s.getTime());
        setRunning(s.isRunning());
      };

      api.onanalyze = (s) => {
        setElementCount(s.getElements().length);
      };
    };

    // If CircuitJS1 already loaded before we set the callback
    if (win.CircuitJS1) {
      win.oncircuitjsloaded?.();
    }
  }, []);

  // Controls
  const toggleRun = () => {
    if (!sim) return;
    sim.setSimRunning(!running);
    setRunning(!running);
  };

  const handleExportCircuit = () => {
    if (!sim) return;
    const text = sim.exportCircuit();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "circuit.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportSVG = () => {
    if (!sim) return;
    sim.onsvgrendered = (_s, svg) => {
      const blob = new Blob([svg], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "circuit.svg";
      a.click();
      URL.revokeObjectURL(url);
    };
    sim.getCircuitAsSVG();
  };

  const handleDoubleTimestep = () => {
    if (!sim) return;
    sim.setMaxTimeStep(sim.getMaxTimeStep() * 2);
  };

  const handleHalveTimestep = () => {
    if (!sim) return;
    sim.setMaxTimeStep(sim.getMaxTimeStep() / 2);
  };

  const goBack = () => {
    window.history.back();
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      {/* Top toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border-b border-gray-700 text-xs shrink-0">
        <button
          onClick={goBack}
          className="px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 font-semibold"
        >
          <ArrowLeft className="w-3.5 h-3.5 inline" /> Back to Schematic
        </button>

        <div className="w-px h-5 bg-gray-700 mx-1" />

        <button
          onClick={toggleRun}
          disabled={!loaded}
          className={`px-3 py-1.5 rounded font-semibold ${
            running
              ? "bg-red-700 hover:bg-red-600"
              : "bg-green-700 hover:bg-green-600"
          } disabled:opacity-40`}
        >
          {running ? <><Pause className="w-3.5 h-3.5 inline" /> Pause</> : <><Play className="w-3.5 h-3.5 inline" /> Run</>}
        </button>

        <button
          onClick={handleHalveTimestep}
          disabled={!loaded}
          className="px-2 py-1.5 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40"
          title="Halve timestep"
        >
          <ChevronDown className="w-3.5 h-3.5 inline" /> ½
        </button>
        <button
          onClick={handleDoubleTimestep}
          disabled={!loaded}
          className="px-2 py-1.5 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40"
          title="Double timestep"
        >
          <ChevronUp className="w-3.5 h-3.5 inline" /> 2×
        </button>

        <div className="w-px h-5 bg-gray-700 mx-1" />

        <button
          onClick={handleExportCircuit}
          disabled={!loaded}
          className="px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-40"
        >
          Export .txt
        </button>
        <button
          onClick={handleExportSVG}
          disabled={!loaded}
          className="px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-40"
        >
          Export SVG
        </button>

        <div className="flex-1" />

        {loaded ? (
          <span className="text-gray-400 font-mono">
            t = {simTime.toExponential(3)}s · {elementCount} elements ·{" "}
            {running ? (
              <span className="text-green-400">running</span>
            ) : (
              <span className="text-yellow-400">paused</span>
            )}
          </span>
        ) : (
          <span className="text-gray-500 animate-pulse">
            Loading simulator…
          </span>
        )}
      </div>

      {/* CircuitJS1 iframe — full remaining height */}
      <iframe
        ref={iframeRef}
        src={iframeSrc}
        onLoad={onIframeLoad}
        className="flex-1 w-full border-0"
        allow="fullscreen"
        title="Circuit Simulator"
      />
    </div>
  );
}

export default function SimulatePage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen bg-gray-950 text-gray-400">
        Loading simulator…
      </div>
    }>
      <SimulateContent />
    </Suspense>
  );
}
