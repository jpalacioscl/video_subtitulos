#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "Entorno virtual no encontrado. Ejecuta primero: ./setup.sh"
    exit 1
fi

PORT=${PORT:-5001}
LLAMA_SERVER=/home/jaime/.local/bin/llama-server
LLAMA_MODEL=/mnt/datos/llmlocal/qwenlocal/models/Qwen3.5-9B-Q4_K_M.gguf
LLAMA_PORT=8080
LLAMA_STARTED_BY_US=false

# ── Matar instancia anterior de esta app ────────────────────────────────────
OLD_PID=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "[run] Deteniendo instancia anterior (puerto $PORT, PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

# ── Arrancar llama-server si no está activo ─────────────────────────────────
if curl -s "http://localhost:$LLAMA_PORT/v1/models" &>/dev/null; then
    echo "[LLM] llama-server ya está activo en el puerto $LLAMA_PORT."
else
    echo "[LLM] Iniciando llama-server con Qwen3.5-9B..."
    "$LLAMA_SERVER" \
        --model "$LLAMA_MODEL" \
        --host 0.0.0.0 \
        --port "$LLAMA_PORT" \
        --ctx-size 32768 \
        --n-gpu-layers 99 \
        --jinja \
        --chat-template-kwargs '{"enable_thinking":false}' \
        --log-disable \
        > /tmp/llama-server.log 2>&1 &

    LLAMA_PID=$!
    LLAMA_STARTED_BY_US=true

    echo -n "[LLM] Esperando que el modelo cargue"
    for i in $(seq 1 60); do
        sleep 2
        if curl -s "http://localhost:$LLAMA_PORT/v1/models" &>/dev/null; then
            echo " listo."
            break
        fi
        echo -n "."
        if [ $i -eq 60 ]; then
            echo ""
            echo "[LLM] ⚠ El servidor tardó demasiado. La app arrancará sin LLM."
            echo "      Ver logs en: /tmp/llama-server.log"
            LLAMA_STARTED_BY_US=false
        fi
    done
fi

# ── Al salir, detener llama-server si lo arrancamos nosotros ────────────────
cleanup() {
    echo ""
    echo "[run] Deteniendo app..."
    if [ "$LLAMA_STARTED_BY_US" = true ] && [ -n "${LLAMA_PID:-}" ]; then
        echo "[LLM] Deteniendo llama-server (PID $LLAMA_PID)..."
        kill "$LLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Iniciar la app Flask ────────────────────────────────────────────────────
source venv/bin/activate
echo "[app] http://localhost:$PORT"
PORT=$PORT python app.py
