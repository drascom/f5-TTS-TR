#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

# Load project-local environment variables if present (e.g., HF_TOKEN).
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

HF_REPO="${HF_REPO:-Karayakar/Orpheus-TTS-Turkish-PT-5000}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PORT="${PORT:-5400}"

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

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    repo_type="model",
    local_dir="${ROOT_DIR}",
)
PY

export ORPHEUS_MODEL_DIR="${ROOT_DIR}"
export ORPHEUS_DEVICE="${ORPHEUS_DEVICE:-cuda}"
export PORT

python inference.py
