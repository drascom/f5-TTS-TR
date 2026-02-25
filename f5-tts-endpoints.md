# F5 TTS API Endpoints

Base URL:
- `https://tts.drascom.uk`

Content type:
- Request: `application/json`
- Response (audio): `audio/wav`

## 1) Health Check

### `GET /health`
Service ayakta mı kontrol eder.

Example:
```bash
curl -s https://tts.drascom.uk/health
```

Success response:
```json
{"status":"ok"}
```

## 2) Coqui-Compatible Endpoints

### `GET /api/languages`
Desteklenen dilleri döner.

Example:
```bash
curl -s https://tts.drascom.uk/api/languages
```

Response:
```json
{"languages":["tr"]}
```

### `GET /api/speakers`
Desteklenen speaker listesini döner.

Example:
```bash
curl -s https://tts.drascom.uk/api/speakers
```

Response:
```json
{"speakers":["default"]}
```

### `GET /api/tts`
Query param ile TTS üretir.

Query params:
- `text` (required)
- `language-id` (optional, default: `tr`)
- `speaker-id` (optional, default: `default`)
- `speaker-wav` (optional, server path; verilmezse varsayılan referans ses kullanılır)

Example:
```bash
curl -G "https://tts.drascom.uk/api/tts" \
  --data-urlencode "text=Merhaba, bu bir testtir." \
  --data-urlencode "language-id=tr" \
  --output test_get.wav
```

### `POST /api/tts`
JSON body ile TTS üretir.

Request body:
```json
{
  "text": "Merhaba, bu bir testtir.",
  "language-id": "tr",
  "speaker-id": "default"
}
```

Example:
```bash
curl -X POST "https://tts.drascom.uk/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Merhaba, bu bir testtir.","language-id":"tr","speaker-id":"default"}' \
  --output test_post.wav
```

Success response:
- Binary WAV audio (`audio/wav`)

## 3) OpenAI-Compatible Endpoint

### `POST /v1/audio/speech`
OpenAI benzeri TTS endpoint.

Request body:
```json
{
  "input": "Merhaba, OpenAI uyumlu endpoint testi.",
  "voice": "default"
}
```

Example:
```bash
curl -X POST "https://tts.drascom.uk/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Merhaba, OpenAI uyumlu endpoint testi.","voice":"default"}' \
  --output test_openai.wav
```

Success response:
- Binary WAV audio (`audio/wav`)

## Error Format

Error durumunda JSON döner:
```json
{
  "detail": "error message"
}
```

Örnek:
- Model dosyası Git LFS pointer ise detaylı hata mesajı döner.

## Performance Notes (Why it can be slow)

- İlk istek daha yavaştır (model + vocoder yüklenir).
- Uzun `text` daha uzun inference süresi üretir.
- GPU yoksa CPU oldukça yavaş olabilir.

Hız için öneriler:
- NVIDIA GPU ile çalıştırın (`F5_DEVICE=cuda`).
- Metni daha kısa cümlelere bölün.
- Container'ı sıcak tutun (ilk yükleme sonrası istekler hızlanır).
- HF cache volume kullanın (tekrar indirme ve cold start maliyeti azalır).
