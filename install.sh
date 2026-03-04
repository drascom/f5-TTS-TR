#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

if [[ -d .git ]]; then
  branch="$(git rev-parse --abbrev-ref HEAD)"
  echo "Updating repository on branch: ${branch}"
  git fetch --all --prune
  git pull --ff-only origin "${branch}"
fi

if [[ ! -f "${ROOT_DIR}/.env" && -f "${ROOT_DIR}/.env.example" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "Created .env from .env.example"
fi

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.env"
  set +a
fi

if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv ffmpeg
  elif [[ "${EUID}" -eq 0 ]]; then
    apt-get update
    apt-get install -y python3-venv ffmpeg
  else
    echo "Skipping apt dependencies (no sudo/root): python3-venv ffmpeg"
  fi
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
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
  bitsandbytes \
  gunicorn

HF_REPO="${HF_REPO:-Karayakar/Orpheus-TTS-Turkish-PT-5000}"

python - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(r"${ROOT_DIR}")
required = [
    root / "config.json",
    root / "model.safetensors.index.json",
    root / "model-00001-of-00003.safetensors",
    root / "model-00002-of-00003.safetensors",
    root / "model-00003-of-00003.safetensors",
    root / "tokenizer.json",
]

def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 4096:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False

missing = [p for p in required if (not p.exists() or p.stat().st_size <= 1024 or is_lfs_pointer(p))]
if missing:
    print("Model files missing/invalid. Downloading from Hugging Face...")
    snapshot_download(
        repo_id="${HF_REPO}",
        repo_type="model",
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
else:
    print("Model files already present. Skipping download.")
PY

echo ""
echo "Install complete."
echo "Run the API with:"
echo "  ./start.sh"
