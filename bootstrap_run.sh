#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
HF_REPO="${HF_REPO:-Karayakar/Orpheus-TTS-Turkish-PT-5000}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PORT="${PORT:-5400}"

cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
python -m pip install \
  flask \
  "huggingface_hub[cli]" \
  librosa \
  numpy \
  scipy \
  snac \
  soundfile \
  transformers \
  accelerate \
  bitsandbytes

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${HF_REPO}" \
    --repo-type model \
    --local-dir "${ROOT_DIR}" \
    --local-dir-use-symlinks False
elif command -v hf >/dev/null 2>&1; then
  hf download "${HF_REPO}" \
    --repo-type model \
    --local-dir "${ROOT_DIR}" \
    --local-dir-use-symlinks False
else
  python -m huggingface_hub.commands.huggingface_cli download "${HF_REPO}" \
    --repo-type model \
    --local-dir "${ROOT_DIR}" \
    --local-dir-use-symlinks False
fi

export ORPHEUS_MODEL_DIR="${ROOT_DIR}"
export ORPHEUS_DEVICE="${ORPHEUS_DEVICE:-cuda}"
export PORT

python inference.py
