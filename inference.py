import os
import threading
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from huggingface_hub import snapshot_download
from scipy.io.wavfile import write
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(os.getenv("ORPHEUS_MODEL_DIR", Path(__file__).resolve().parent)).resolve()
PROMPT_DATA_DIR = Path(os.getenv("PROMPT_DATA_DIR", MODEL_DIR / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", MODEL_DIR / "inference")).resolve()
DEVICE = os.getenv("ORPHEUS_DEVICE", "cuda")
MAX_NEW_TOKENS = int(os.getenv("ORPHEUS_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("ORPHEUS_TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("ORPHEUS_TOP_K", "10"))
TOP_P = float(os.getenv("ORPHEUS_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("ORPHEUS_REPETITION_PENALTY", "1.9"))
tokenizer = None
model = None
snac_model = None
init_lock = threading.Lock()


def load_orpheus_tokenizer(model_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def load_snac() -> SNAC:
    return SNAC.from_pretrained("hubertsiuzdak/snac_24khz")


def load_orpheus_auto_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map=DEVICE,
    )
    if DEVICE.startswith("cuda"):
        model.cuda()
    return model


def prepare_inputs(text_prompts: list[str], tokenizer):
    start_tokens = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)

    all_modified_input_ids = []
    for prompt in text_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        second_input_ids = torch.cat([start_tokens, input_ids, end_tokens], dim=1)
        all_modified_input_ids.append(second_input_ids)

    all_padded_tensors = []
    all_attention_masks = []
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        attention_mask = torch.cat(
            [
                torch.zeros((1, padding), dtype=torch.int64),
                torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64),
            ],
            dim=1,
        )
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)

    input_ids = torch.cat(all_padded_tensors, dim=0).to(DEVICE)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(DEVICE)
    return input_ids, attention_mask


def run_inference(model, input_ids, attention_mask):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            eos_token_id=128258,
        )
        generated_ids = torch.cat([generated_ids, torch.tensor([[128262]], device=DEVICE)], dim=1)
        return generated_ids


def redistribute_codes(code_list, snac_model):
    layer_1 = []
    layer_2 = []
    layer_3 = []

    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    return snac_model.decode(codes)


def convert_tokens_to_speech(generated_ids, snac_model):
    token_to_find = 128257
    token_to_remove = 128258
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        new_length = (row.size(0) // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    return [redistribute_codes(code_list, snac_model) for code_list in code_lists]


def to_wav_from(samples: list) -> list[np.ndarray]:
    processed_samples = []
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().squeeze().to("cpu").numpy()
        else:
            sample = np.squeeze(sample)
        processed_samples.append(sample)
    return processed_samples


def save_wav(samples: list[np.ndarray], sample_rate: int, filename: str):
    write(filename, sample_rate, samples[0].astype(np.float32))


app = Flask(__name__)


def ensure_initialized():
    global tokenizer, model, snac_model
    if tokenizer is not None and model is not None and snac_model is not None:
        return

    with init_lock:
        if tokenizer is not None and model is not None and snac_model is not None:
            return

        required = [
            MODEL_DIR / "config.json",
            MODEL_DIR / "model.safetensors.index.json",
            MODEL_DIR / "model-00001-of-00003.safetensors",
            MODEL_DIR / "model-00002-of-00003.safetensors",
            MODEL_DIR / "model-00003-of-00003.safetensors",
            MODEL_DIR / "tokenizer.json",
        ]
        if not all(p.exists() and p.stat().st_size > 1024 for p in required):
            snapshot_download(
                repo_id="Karayakar/Orpheus-TTS-Turkish-PT-5000",
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
            )

        tokenizer = load_orpheus_tokenizer(str(MODEL_DIR))
        model = load_orpheus_auto_model(str(MODEL_DIR))
        snac_model = load_snac()


@app.get("/health")
def health():
    ensure_initialized()
    return {"status": "ok", "device": DEVICE}


@app.post("/generate")
def generate_file():
    ensure_initialized()
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    requested_output_path = (payload.get("output_path") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    texts = [text]
    input_ids, attention_mask = prepare_inputs(texts, tokenizer)
    generated_ids = run_inference(model, input_ids, attention_mask)
    samples = convert_tokens_to_speech(generated_ids, snac_model)
    wav_forms = to_wav_from(samples)

    if requested_output_path:
        raw_path = Path(requested_output_path)
        if raw_path.is_absolute():
            out_file = raw_path
        else:
            out_file = OUTPUT_DIR / raw_path

        if out_file.suffix.lower() != ".wav":
            out_file = out_file / f"output_{int(datetime.now().timestamp())}.wav"
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / f"output_{int(datetime.now().timestamp())}.wav"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_wav(wav_forms, 24000, str(out_file))

    return send_file(str(out_file), mimetype="audio/wav", as_attachment=True, download_name=out_file.name)


@app.post("/generate-json")
def generate_json():
    ensure_initialized()
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    requested_output_path = (payload.get("output_path") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    texts = [text]
    input_ids, attention_mask = prepare_inputs(texts, tokenizer)
    generated_ids = run_inference(model, input_ids, attention_mask)
    samples = convert_tokens_to_speech(generated_ids, snac_model)
    wav_forms = to_wav_from(samples)

    if requested_output_path:
        raw_path = Path(requested_output_path)
        if raw_path.is_absolute():
            out_file = raw_path
        else:
            out_file = OUTPUT_DIR / raw_path

        if out_file.suffix.lower() != ".wav":
            out_file = out_file / f"output_{int(datetime.now().timestamp())}.wav"
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / f"output_{int(datetime.now().timestamp())}.wav"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_wav(wav_forms, 24000, str(out_file))

    return jsonify({"status": "success", "file": str(out_file)})


if __name__ == "__main__":
    ensure_initialized()
    app.run(host="0.0.0.0", debug=False, port=int(os.getenv("PORT", "5400")))
