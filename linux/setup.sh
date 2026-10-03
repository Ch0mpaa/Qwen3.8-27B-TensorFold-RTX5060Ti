#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8 TensorFold setup — creates venv, installs engine, downloads model
# by Screwed Up Tech — screwedup.tech

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "============================================================"
echo "  Qwen3.8 TensorFold Setup — Consumer NVIDIA GPUs"
echo "  by Screwed Up Tech — screwedup.tech"
echo "============================================================"
echo

# Load .env
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

MODEL="${MODEL:-dense}"
TF_VENV="${TF_VENV:-.venv}"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3.11+ required. Install it first."
    exit 1
fi

# Check NVIDIA
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Install NVIDIA driver 570+."
    exit 1
fi

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
echo "[*] GPU compute capability: $CC"

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[*] GPU VRAM: ${VRAM} MiB"

# Create venv
if [[ ! -f "$TF_VENV/bin/python" ]]; then
    echo "[*] Creating Python venv..."
    python3 -m venv "$TF_VENV"
    "$TF_VENV/bin/pip" install --upgrade pip -q
fi

# Install TensorFold
echo "[*] Installing TensorFold..."
"$TF_VENV/bin/pip" install tensorfold huggingface_hub -q 2>&1 | grep -v "already satisfied"

# Download model
if [[ "$MODEL" == "flashnext" ]]; then
    DIR="${FLASHNEXT_MODEL_DIR:-models/Qwen3.8-Flash-Next-exl3-2.05bpw}"
    echo "[*] Downloading Qwen3.8 Flash Next EXL3 2.05bpw (62.8 GB)..."
    "$TF_VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('turboderp/Qwen3.8-Flash-Next-exl3', revision='2.05bpw_h4_ng4', local_dir='$DIR')
print('Done.')
"
else
    DIR="${DENSE_MODEL_DIR:-models/Qwen3.8-27B-EXL3-2.0bpw}"
    echo "[*] Downloading Qwen3.8-27B EXL3 2.0bpw (6.8 GB)..."
    "$TF_VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw', local_dir='$DIR')
print('Done.')
"
fi

echo
echo "============================================================"
echo "  Setup complete! Run ./linux/start.sh to serve."
echo "============================================================"
