# Orpheus Turkish TTS API

Simple Flask API (with a lightweight web UI) for Turkish text-to-speech using the Orpheus model.

## Folder Structure

```
.
├── inference.py        # Flask API application
├── templates/index.html # Browser UI for testing TTS
├── install.sh          # Setup/update script (deps + model + .env)
├── start.sh            # Run API (gunicorn if available, else python)
├── .env.example        # Environment template
├── config.json
├── model.safetensors.index.json
├── tokenizer_config.json
└── ... model/tokenizer files
```

## API Endpoints

- `GET /` (web UI)
- `GET /health`
- `POST /generate` (returns WAV file)
- `POST /generate-json` (returns JSON with file path)

Example request:

```bash
curl -X POST "http://127.0.0.1:5400/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Merhaba, bu bir testtir."}' \
  --output test.wav

# Open browser test UI
open http://127.0.0.1:5400/
```

## Quick Start (Server)

```bash
git clone https://github.com/drascom/f5-TTS-TR.git
cd f5-TTS-TR
./install.sh
```

`install.sh` creates and enables a systemd service named `tts` and starts it.

Check service and API health:

```bash
sudo systemctl status tts
curl -s http://127.0.0.1:5400/health
```

## Configuration

Copy `.env.example` to `.env` and update values as needed.  
`install.sh` does this automatically if `.env` does not exist.

Common variables:
- `PORT` (default: `5400`)
- `ORPHEUS_DEVICE` (`cuda` or `cpu`)
- `ORPHEUS_MODEL_DIR` (default: project root)
- `OUTPUT_DIR` (default: `./inference`)
- `HF_REPO` (default: `Karayakar/Orpheus-TTS-Turkish-PT-5000`)
- `ORPHEUS_MAX_NEW_TOKENS` / `ORPHEUS_MIN_NEW_TOKENS` to control output length floor/cap

## Notes

- `install.sh` updates the local git checkout (`git pull --ff-only`).
- If model shards are missing, `install.sh` downloads them from Hugging Face.
- For GPU servers, keep `ORPHEUS_DEVICE=cuda`.
- After first successful setup, `install.sh` writes a local `.installed` marker.
- If you run `install.sh` again, it asks `run` or `reinstall` (non-interactive default: `run`).
- The service is installed as `tts.service` and enabled to auto-start after reboot.
