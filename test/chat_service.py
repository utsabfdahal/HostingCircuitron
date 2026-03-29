"""
CIRCUITRON – AI Chat Service (Lightning AI / DeepSeek-V3.1).

Provides conversational circuit-analysis assistance by sending a concise
representation of the current circuit plus recent chat history to the
Lightning AI API.
"""

from __future__ import annotations

import os
from typing import List, Dict

import httpx

from .schemas import Circuit


_API_URL = "https://lightning.ai/api/v1/chat/completions"
_MODEL = "lightning-ai/DeepSeek-V3.1"

_SYSTEM_PROMPT = (
    "You are an expert electrical-engineering assistant embedded in the "
    "CIRCUITRON app — a tool that converts hand-drawn circuit diagrams into "
    "digital schematics.  You help users understand their circuits, debug "
    "issues, suggest improvements, and explain electrical concepts.\n\n"
    "Keep answers concise and practical.  Use standard EE notation.  "
    "When referring to components, use the IDs shown in the circuit data "
    "(e.g. R1, C2, V1).\n\n"
    "## CircuitJS (CJS) Netlist Format Reference\n"
    "The CJS text below describes the circuit topology.  Each line is one element:\n"
    "  <type> <x1> <y1> <x2> <y2> <flags> [params...]\n"
    "Common type codes:\n"
    "  r = resistor, c = capacitor, l = inductor, v = voltage source,\n"
    "  i = current source, d = diode, t = BJT (NPN), T = BJT (PNP),\n"
    "  f = N-MOSFET, F = P-MOSFET, a = op-amp, w = wire,\n"
    "  s = switch, S = SPDT switch, L = logic input, M = logic output,\n"
    "  g = ground, R = rail, p = probe, O = output, x = text, 172 = pot\n"
    "Coordinates are on a 16-px grid.  Lines starting with '$' set global "
    "options (flags, timeStep, etc.).  '38' lines are labels.\n"
    "Use this netlist to understand exact wiring and component values."
)


def _summarise_circuit(circuit: Circuit) -> str:
    """Build a short text summary of the circuit for the AI context window."""
    lines: list[str] = []
    if circuit.components:
        lines.append("Components:")
        for c in circuit.components:
            val = f" = {c.value}" if c.value else ""
            lines.append(f"  • {c.id} ({c.type}){val}")
    lines.append(f"\nNodes: {len(circuit.nodes)}")
    lines.append(f"Connections: {len(circuit.connections)}")
    if circuit.connections:
        lines.append("Connection details:")
        for conn in circuit.connections:
            lines.append(
                f"  {conn.from_component} [{conn.from_terminal}] → {conn.to_node}"
            )
    if circuit.edges:
        lines.append("Wire segments (node-to-node):")
        for e in circuit.edges:
            lines.append(f"  {e.source} — {e.target}")
    return "\n".join(lines)


async def get_chat_response(
    circuit: Circuit,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
    cjs_text: str | None = None,
) -> str:
    """
    Send a chat request to Lightning AI and return the assistant reply.

    Parameters
    ----------
    circuit      : Current Circuit object for context.
    user_message : The latest user question.
    history      : Last few messages ``[{"role": ..., "content": ...}, ...]``
                   for multi-turn context.  Keep this short (2-3 exchanges).
    cjs_text     : Raw CircuitJS netlist text for full topology context.
    """
    api_key = os.getenv("LIGHTNING_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LIGHTNING_AI_API_KEY is not set. "
            "Add it to the .env file in the project root."
        )

    circuit_context = _summarise_circuit(circuit)

    cjs_block = ""
    if cjs_text:
        cjs_block = f"\n\n--- CJS Netlist ---\n{cjs_text}\n---"

    system_content = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"--- Current circuit ---\n{circuit_context}\n---"
        f"{cjs_block}"
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Append recent history (last 3 exchanges = 6 messages max)
    if history:
        messages.extend(history[-6:])

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": _MODEL,
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _API_URL, headers=headers, json=payload, timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]
