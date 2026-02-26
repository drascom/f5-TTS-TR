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

is_lfs_pointer() {
  local file="$1"
  [[ -f "$file" ]] && head -n 1 "$file" 2>/dev/null | grep -q "version https://git-lfs.github.com/spec/v1"
}

need_download=0
required_files=(
  "${ROOT_DIR}/config.json"
  "${ROOT_DIR}/model.safetensors.index.json"
  "${ROOT_DIR}/model-00001-of-00003.safetensors"
  "${ROOT_DIR}/model-00002-of-00003.safetensors"
  "${ROOT_DIR}/model-00003-of-00003.safetensors"
  "${ROOT_DIR}/tokenizer.json"
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]] || is_lfs_pointer "$file"; then
    need_download=1
    break
  fi
done

if [[ "$need_download" -eq 1 ]]; then
  echo "Model files missing or pointers detected. Downloading from Hugging Face..."
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    repo_type="model",
    local_dir="${ROOT_DIR}",
)
PY
else
  echo "Required model files already exist. Skipping download."
fi

export ORPHEUS_MODEL_DIR="${ROOT_DIR}"
export ORPHEUS_DEVICE="${ORPHEUS_DEVICE:-cuda}"
export PORT

python inference.py
