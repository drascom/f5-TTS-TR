#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

export ORPHEUS_MODEL_DIR="${ORPHEUS_MODEL_DIR:-${ROOT_DIR}}"
export ORPHEUS_DEVICE="${ORPHEUS_DEVICE:-cuda}"
export PORT="${PORT:-5400}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing virtualenv at ${VENV_DIR}. Run ./bootstrap_run.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

if command -v gunicorn >/dev/null 2>&1; then
  exec gunicorn -w "${GUNICORN_WORKERS:-1}" -k gthread --threads "${GUNICORN_THREADS:-4}" -b "0.0.0.0:${PORT}" inference:app --timeout "${GUNICORN_TIMEOUT:-300}"
fi

exec python inference.py
