import io
import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from f5_tts.api import F5TTS


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    # LFS pointer files are tiny text files starting with this marker.
    if not path.exists() or path.stat().st_size > 4096:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256)
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _resolve_device(requested_device: str) -> str:
    device = (requested_device or "cpu").strip().lower()

    if device == "cuda":
        if torch.cuda.is_available():
            # safetensors/torch loader is more reliable with an explicit CUDA index.
            return "cuda:0"
        return "cpu"

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"
        return "cpu"

    if device in {"cpu", "meta"}:
        return device

    return "cpu"


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("F5_MODEL_NAME", "F5TTS_Base")
    ckpt_file: Path = Path(os.getenv("F5_CKPT_FILE", str(BASE_DIR / "f5_tts_turkish_800000.safetensors")))
    vocab_file: Path = Path(os.getenv("F5_VOCAB_FILE", str(BASE_DIR / "vocab.txt")))
    reference_wav: Path = Path(os.getenv("F5_REFERENCE_WAV", str(BASE_DIR / "tr.wav")))
    reference_text: str = os.getenv("F5_REFERENCE_TEXT", "")
    device: str = _resolve_device(os.getenv("F5_DEVICE", "cpu"))
    remove_silence: bool = os.getenv("F5_REMOVE_SILENCE", "false").lower() == "true"


class CoquiTTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker_wav: str | None = Field(default=None, alias="speaker-wav")
    speaker_id: str | None = Field(default=None, alias="speaker-id")
    language_id: str | None = Field(default="tr", alias="language-id")

    model_config = {"populate_by_name": True}


class OpenAITTSRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str | None = None


def _resolve_reference_wav(settings: Settings, speaker_wav: str | None, speaker_id: str | None) -> Path:
    if speaker_wav:
        path = Path(speaker_wav)
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"speaker_wav not found: {speaker_wav}")

    # Keep Coqui-compatible field while defaulting to a single local speaker.
    if speaker_id and speaker_id not in {"default", "tr", "main"}:
        raise HTTPException(status_code=400, detail=f"unknown speaker_id: {speaker_id}")

    return settings.reference_wav


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    if not settings.ckpt_file.exists():
        raise RuntimeError(f"Checkpoint file not found: {settings.ckpt_file}")
    if _looks_like_git_lfs_pointer(settings.ckpt_file):
        raise RuntimeError(
            "Checkpoint file is a Git LFS pointer, not a real safetensors model. "
            "Download the actual model file (expected size ~1.3GB) and place it at "
            f"{settings.ckpt_file}."
        )
    if not settings.vocab_file.exists():
        raise RuntimeError(f"Vocab file not found: {settings.vocab_file}")
    if not settings.reference_wav.exists():
        raise RuntimeError(f"Reference wav not found: {settings.reference_wav}")

    return settings


@lru_cache(maxsize=1)
def get_engine() -> F5TTS:
    settings = get_settings()
    try:
        return F5TTS(
            model=settings.model_name,
            ckpt_file=str(settings.ckpt_file),
            vocab_file=str(settings.vocab_file),
            device=settings.device,
        )
    except OSError as e:
        if settings.device != "cpu" and "No such device" in str(e):
            logger.warning(
                "Model init failed on device '%s' (%s). Retrying on cpu.",
                settings.device,
                e,
            )
            return F5TTS(
                model=settings.model_name,
                ckpt_file=str(settings.ckpt_file),
                vocab_file=str(settings.vocab_file),
                device="cpu",
            )
        raise


inference_lock = threading.Lock()
app = FastAPI(title="F5 Turkish Coqui-Compatible API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok"}


@app.get("/languages")
def api_languages() -> dict[str, list[str]]:
    return {"languages": ["tr"]}


@app.get("/speakers")
def api_speakers() -> dict[str, list[str]]:
    return {"speakers": ["default"]}


def _synthesize(text: str, speaker_wav: str | None = None, speaker_id: str | None = None) -> bytes:
    try:
        settings = get_settings()
        engine = get_engine()
    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    ref_wav_path = _resolve_reference_wav(settings, speaker_wav, speaker_id)

    with inference_lock:
        wav, sample_rate, _ = engine.infer(
            ref_file=str(ref_wav_path),
            ref_text=settings.reference_text,
            gen_text=text.strip(),
            remove_silence=settings.remove_silence,
        )

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, wav, sample_rate, format="WAV")
    return audio_buffer.getvalue()


@app.get("/tts")
def tts_get(
    text: str = Query(..., min_length=1),
    speaker_wav: str | None = Query(default=None, alias="speaker-wav"),
    speaker_id: str | None = Query(default=None, alias="speaker-id"),
    language_id: str | None = Query(default="tr", alias="language-id"),
) -> Response:
    _ = language_id  # kept for Coqui compatibility
    audio = _synthesize(text=text, speaker_wav=speaker_wav, speaker_id=speaker_id)
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts")
def tts_post(payload: CoquiTTSRequest) -> Response:
    _ = payload.language_id  # kept for Coqui compatibility
    audio = _synthesize(text=payload.text, speaker_wav=payload.speaker_wav, speaker_id=payload.speaker_id)
    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/audio/speech")
def openai_compatible_tts(payload: OpenAITTSRequest) -> Response:
    audio = _synthesize(text=payload.input, speaker_id=payload.voice)
    return Response(content=audio, media_type="audio/wav")
