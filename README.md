# Orpheus Turkish TTS API

Simple Flask API for Turkish text-to-speech using the Orpheus model.

## Folder Structure

```
.
├── inference.py        # Flask API application
├── install.sh          # Setup/update script (deps + model + .env)
├── start.sh            # Run API (gunicorn if available, else python)
├── .env.example        # Environment template
├── config.json
├── model.safetensors.index.json
├── tokenizer_config.json
└── ... model/tokenizer files
```

## API Endpoints

- `GET /health`
- `POST /generate` (returns WAV file)
- `POST /generate-json` (returns JSON with file path)

Example request:

```bash
curl -X POST "http://127.0.0.1:5400/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Merhaba, bu bir testtir."}' \
  --output test.wav
```

## Quick Start (Server)

```bash
git clone https://github.com/drascom/f5-TTS-TR.git
cd f5-TTS-TR
./install.sh
./start.sh
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

## Notes

- `install.sh` updates the local git checkout (`git pull --ff-only`).
- If model shards are missing, `install.sh` downloads them from Hugging Face.
- For GPU servers, keep `ORPHEUS_DEVICE=cuda`.
- After first successful setup, `install.sh` writes a local `.installed` marker.
- If you run `install.sh` again, it asks `run` or `reinstall` (non-interactive default: `run`).
