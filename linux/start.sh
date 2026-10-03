#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3.8 via TensorFold on consumer NVIDIA GPUs
# by Screwed Up Tech — screwedup.tech

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Load .env
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Defaults
MODEL="${MODEL:-dense}"
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3072}"
THINKING="${THINKING:-off}"
DRAFT="${DRAFT:-none}"
GPU_MEM_GB="${GPU_MEM_GB:-14.7}"
NO_UPDATE_CHECK="${NO_UPDATE_CHECK:-1}"
VISION="${VISION:-off}"
TF_VENV="${TF_VENV:-.venv}"
LANE_KERNELS="${LANE_KERNELS:-auto}"
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-0}"

# Check venv
if [[ ! -f "$TF_VENV/bin/python" ]]; then
    echo "[ERROR] TensorFold venv not found. Run ./linux/setup.sh first."
    exit 1
fi

# Select model
if [[ "$MODEL" == "flashnext" ]]; then
    MODEL_PATH="${FLASHNEXT_MODEL_DIR:-models/Qwen3.8-Flash-Next-exl3-2.05bpw}"
    MODEL_ID="${MODEL_ID:-qwen3.8-flash-next-exl3-tensorfold}"
    echo "[*] Model: Qwen3.8 Flash Next (MoE + MTP)"
else
    MODEL_PATH="${DENSE_MODEL_DIR:-models/Qwen3.8-27B-EXL3-2.0bpw}"
    MODEL_ID="${MODEL_ID:-qwen3.8-27b-exl3-2bpw-tensorfold}"
    echo "[*] Model: Qwen3.8-27B dense (EXL3 2.0bpw)"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    echo "Run ./linux/setup.sh to download it."
    exit 1
fi

# Check VRAM
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[*] Free VRAM: ${FREE:-?} MiB"

# Build args
ARGS=(
    -m tensorfold serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --name "$MODEL_ID"
    --context "$CONTEXT_SIZE"
    --lane-kernels "$LANE_KERNELS"
)

[[ "$THINKING" == "off" ]] && ARGS+=(--no-thinking)
[[ "$DRAFT" == "none" ]] && ARGS+=(--no-drafts)
[[ "$NO_UPDATE_CHECK" == "1" ]] && ARGS+=(--no-update-check)

if [[ "$PROMPT_CACHE_GIB" != "0" ]]; then
    ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
fi

# Flash Next specific
if [[ "$MODEL" == "flashnext" ]]; then
    MTP_DRAFTS="${MTP_DRAFTS:-6}"
    MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
    SSD_EXPERTS_GIB="${SSD_EXPERTS_GIB:-50}"
    PLE_ON_SSD="${PLE_ON_SSD:-1}"

    ARGS+=(--mtp-drafts "$MTP_DRAFTS")
    ARGS+=(--mtp-confidence "$MTP_CONFIDENCE")
    ARGS+=(--ssd-experts "$SSD_EXPERTS_GIB")

    # Remove --no-drafts for Flash Next (MTP is the drafter)
    ARGS=("${ARGS[@]/--no-drafts/}")
fi

echo
echo "============================================================"
echo "  Qwen3.8 on TensorFold — Consumer NVIDIA GPU"
echo "  by Screwed Up Tech — screwedup.tech"
echo "============================================================"
echo "  Model:    $MODEL_PATH"
echo "  Endpoint: http://$HOST:$PORT/v1"
echo "  Context:  $CONTEXT_SIZE tokens"
echo "  Thinking: $THINKING"
echo "  Draft:    $DRAFT"
echo "============================================================"
echo

# Background mode
if [[ "${1:-}" == "-b" || "${1:-}" == "--background" ]]; then
    LOG="logs/tensorfold-$(date +%Y%m%d-%H%M%S).log"
    mkdir -p logs
    echo "[*] Starting in background, log: $LOG"
    nohup "$TF_VENV/bin/python" "${ARGS[@]}" >"$LOG" 2>&1 &
    echo "[*] PID: $!"
    echo "[*] Tail with: tail -f $LOG"
else
    exec "$TF_VENV/bin/python" "${ARGS[@]}"
fi
