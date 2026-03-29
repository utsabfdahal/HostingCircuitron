#!/usr/bin/env bash
# -----------------------------------------------------------------
#  CIRCUITRON -- one-command project launcher
#  Usage:  ./start.sh
# -----------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
FRONTEND="$ROOT/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Colours for pretty output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${CYAN}[CIRCUITRON]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }

# -- 1. Python virtual environment ------------------------------------
if [ ! -d "$VENV" ]; then
  info "Creating Python virtual environment..."
  python3 -m venv "$VENV"
  ok "Virtual environment created at $VENV"
fi

info "Activating virtual environment..."
source "$VENV/bin/activate"

# -- 2. Python dependencies -------------------------------------------
info "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ROOT/requirements.txt"
ok "Python dependencies installed"

# -- 3. Node.js dependencies ------------------------------------------
if [ ! -d "$FRONTEND/node_modules" ]; then
  info "Installing frontend dependencies..."
  (cd "$FRONTEND" && npm install)
  ok "Frontend dependencies installed"
else
  ok "Frontend dependencies already installed"
fi

# -- 4. Start backend (FastAPI / uvicorn) -----------------------------
info "Starting backend on port ${BACKEND_PORT}..."
uvicorn test.main:app \
  --host 0.0.0.0 \
  --port "${BACKEND_PORT}" \
  --reload &
BACKEND_PID=$!
ok "Backend started (PID ${BACKEND_PID})"

# -- 5. Start frontend (Next.js dev server) ---------------------------
info "Starting frontend on port ${FRONTEND_PORT}..."
(cd "$FRONTEND" && npm run dev -- -p "${FRONTEND_PORT}") &
FRONTEND_PID=$!
ok "Frontend started (PID ${FRONTEND_PID})"

# -- 6. Cleanup on exit ----------------------------------------------
cleanup() {
  echo ""
  info "Shutting down..."
  kill "${BACKEND_PID}" 2>/dev/null || true
  kill "${FRONTEND_PID}" 2>/dev/null || true
  ok "All processes stopped."
}
trap cleanup EXIT INT TERM

# -- 7. Wait ----------------------------------------------------------
echo ""
echo -e "${GREEN}==========================================================${NC}"
echo -e "${GREEN}  CIRCUITRON is running!${NC}"
echo -e "  Backend:   ${CYAN}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  Frontend:  ${CYAN}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "${GREEN}==========================================================${NC}"
echo ""
info "Press Ctrl+C to stop all services."
wait
