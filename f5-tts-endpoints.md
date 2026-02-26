# F5 TTS API Endpoints (Updated)

Base URL:
- `https://tts.drascom.uk`

Audio response type:
- `audio/wav`

JSON error format:
```json
{
  "detail": "error message"
}
```

## 1) Health

### `GET /health`
Service and model settings check.

Example:
```bash
curl -s https://tts.drascom.uk/health
```

Success response:
```json
{"status":"ok"}
```

Possible errors:
- `500` if model/checkpoint/vocab/reference WAV is missing or invalid.

## 2) Coqui-Compatible Endpoints

### `GET /languages`
Returns supported language list.

Example:
```bash
curl -s https://tts.drascom.uk/languages
```

Response:
```json
{"languages":["tr"]}
```

### `GET /speakers`
Returns available speaker IDs.

Example:
```bash
curl -s https://tts.drascom.uk/speakers
```

Response:
```json
{"speakers":["default"]}
```

### `GET /tts`
Generates WAV with query params.

Query params:
- `text` (required, min length: 1)
- `language-id` (optional, default: `tr`, currently kept for compatibility)
- `speaker-id` (optional; accepted values: `default`, `tr`, `main`)
- `speaker-wav` (optional; absolute/local server path to WAV file)

Example:
```bash
curl -G "https://tts.drascom.uk/tts" \
  --data-urlencode "text=Merhaba, bu bir testtir." \
  --data-urlencode "language-id=tr" \
  --data-urlencode "speaker-id=default" \
  --output test_get.wav
```

### `POST /tts`
Generates WAV with JSON body (Coqui field names supported).

Request body:
```json
{
  "text": "Merhaba, bu bir testtir.",
  "language-id": "tr",
  "speaker-id": "default",
  "speaker-wav": "/app/custom_reference.wav"
}
```

Notes:
- `language-id` is accepted for compatibility; current model is Turkish.
- `speaker-id` validates only `default`, `tr`, `main`.
- `speaker-wav` must point to an existing file on the server/container.

Example:
```bash
curl -X POST "https://tts.drascom.uk/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Merhaba, bu bir testtir.","language-id":"tr","speaker-id":"default"}' \
  --output test_post.wav
```

Success response:
- Binary WAV audio (`audio/wav`)

## 3) OpenAI-Compatible Endpoint

### `POST /v1/audio/speech`
OpenAI-style speech endpoint.

Request body:
```json
{
  "input": "Merhaba, OpenAI uyumlu endpoint testi.",
  "voice": "default"
}
```

Notes:
- `input` is required.
- `voice` maps to `speaker-id` and supports `default`, `tr`, `main`.

Example:
```bash
curl -X POST "https://tts.drascom.uk/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Merhaba, OpenAI uyumlu endpoint testi.","voice":"default"}' \
  --output test_openai.wav
```

Success response:
- Binary WAV audio (`audio/wav`)

## 4) Common Error Cases

- `400` text/input is empty.
- `400` `speaker-id`/`voice` is unknown.
- `400` `speaker-wav` path does not exist.
- `500` model file is missing, invalid, or a Git LFS pointer file.

## 5) Performance Notes

- First request is slower because model is loaded lazily.
- Inference runs with a global lock, so requests are processed one at a time.
- CPU mode is significantly slower than CUDA.

Recommendations:
- Use GPU with `F5_DEVICE=cuda`.
- Keep texts shorter.
- Keep service warm to avoid cold-start overhead.
