#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
INSTALL_MARKER="${ROOT_DIR}/.installed"
SERVICE_NAME="tts"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_USER="${SUDO_USER:-$(id -un)}"

cd "${ROOT_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

setup_systemd_service() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemd not found. Skipping service setup."
    return
  fi

  local service_content
  service_content="[Unit]
Description=Turkish TTS API Service
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"

  if command -v sudo >/dev/null 2>&1; then
    echo "${service_content}" | sudo tee "${SERVICE_FILE}" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}"
    sudo systemctl restart "${SERVICE_NAME}"
  elif [[ "${EUID}" -eq 0 ]]; then
    echo "${service_content}" > "${SERVICE_FILE}"
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    systemctl restart "${SERVICE_NAME}"
  else
    echo "No sudo/root permissions. Skipping service setup for ${SERVICE_NAME}."
  fi
}

if [[ -f "${INSTALL_MARKER}" && -d "${VENV_DIR}" ]]; then
  choice="run"
  if [[ -t 0 ]]; then
    echo "Installation already detected (${INSTALL_MARKER})."
    read -r -p "Choose action [run/reinstall] (default: run): " input_choice
    input_choice="$(echo "${input_choice}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${input_choice}" == "reinstall" || "${input_choice}" == "run" ]]; then
      choice="${input_choice}"
    fi
  fi

  if [[ "${choice}" == "run" ]]; then
    echo "Skipping reinstall."
    if command -v systemctl >/dev/null 2>&1 && { systemctl list-unit-files 2>/dev/null || true; } | grep -q "^${SERVICE_NAME}\.service"; then
      if command -v sudo >/dev/null 2>&1; then
        sudo systemctl restart "${SERVICE_NAME}"
      elif [[ "${EUID}" -eq 0 ]]; then
        systemctl restart "${SERVICE_NAME}"
      else
        echo "No sudo/root permissions. Starting app directly..."
        exec "${ROOT_DIR}/start.sh"
      fi
      echo "Service restarted: ${SERVICE_NAME}"
      exit 0
    fi

    echo "Starting app directly..."
    exec "${ROOT_DIR}/start.sh"
  fi
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

cat > "${INSTALL_MARKER}" <<EOF
installed_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python=$(python3 --version 2>&1)
EOF

setup_systemd_service

echo "Service installed/enabled: ${SERVICE_NAME}"
echo "Check status with: sudo systemctl status ${SERVICE_NAME}"
