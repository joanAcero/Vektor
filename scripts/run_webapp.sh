#!/usr/bin/env bash
# run_webapp.sh
# -------------
# Desktop entrypoint for the Flask UI. Resolves the repo root from its own
# location (same convention as run_daily.sh) so it survives a repo move.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="${VEKTOR_VENV_DIR:-$REPO_DIR/vektor-venv}/bin/python"
HOST=127.0.0.1
PORT=5000
URL="http://$HOST:$PORT/"

cd "$REPO_DIR"

port_open() { (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; }

# Idempotencia: si ya hay un servidor escuchando, no lances otro (fallaría con
# EADDRINUSE) — abre el navegador y sal.
if port_open; then
    xdg-open "$URL" >/dev/null 2>&1 &
    echo "Ya estaba corriendo en $URL"
    exit 0
fi

"$VENV_PY" webapp.py &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT INT TERM

# Espera activa hasta que el puerto acepte conexiones (máx ~15 s). Un `sleep 3`
# ciego falla el día que un import tarde más de lo normal.
for _ in $(seq 60); do
    port_open && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "El servidor murió al arrancar."; wait $SERVER_PID; }
    sleep 0.25
done

xdg-open "$URL" >/dev/null 2>&1 &
wait $SERVER_PID