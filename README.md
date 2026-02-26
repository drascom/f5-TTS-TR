---
license: mit
language:
- tr
base_model:
- canopylabs/orpheus-3b-0.1-pretrained
pipeline_tag: text-to-speech
tags:
- karayakar
- Turkish
- Turkce
- TTS
- Orpheus
- Text-to-Speech
---


# Orpheus TTS Turkish Model

Orpheus TTS Turkish Pretrain (step 2000) 
model is trained based on "canopylabs/orpheus-3b-0.1-pretrained".

Syntethic voice data over 60 hrs used for initial training.
+160hrs additional Syntethic voice data mixed in training.
400 Emoji (real voice) data used for emoji support. 

you can interact with the model - Flask API


# Emotion Support 

Model supports below emotions in the text.
```
<laugh> – gülme

<chuckle> – kıkırdama

<sigh> – iç çekme

<cough> – öksürme

<sniffle> – <burnunu çekme>

<groan> – inleme

<yawn> – esneme

<gasp> – nefesi kesilme / şaşkınlıkla soluma
```


# API 

Flask configured to run on port 5400 (you can change in the below script)

```
POST http://127.0.0.1:5400/generate HTTP/1.1
User-Agent: Fiddler
content-type: application/json
Host: 127.0.0.1:5400
Content-Length: 110

{
    "text": "Merhaba, orpheusTTS Turkce deneme"
}

```

# Create Environment

windows:
```
#create virtual environment
python -m venv venv 
venv\Scripts\activate

python inference.py



```

# Training

```
For training with your own data, you can check
train.py
config.yaml

```





# inference.py
(please install the necessary libraries)

```
# respective torch from https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install snac pathlib torch transformers huggingface_hub librosa numpy scipy torchaudio Flask jsonify


```

```
import os
from snac import SNAC
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer,BitsAndBytesConfig
from huggingface_hub import snapshot_download
import librosa
import numpy as np
from scipy.io.wavfile import write
import torchaudio
from flask import Flask, jsonify, request

modelLocalPath="D:\\...\\Karayakar\\Orpheus-TTS-Turkish-PT-5000"


def load_orpheus_tokenizer(model_id: str = modelLocalPath) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id,local_files_only=True, device_map="cuda")
    return tokenizer

def load_snac():
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    return snac_model

def load_orpheus_auto_model(model_id: str = modelLocalPath):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,local_files_only=True, device_map="cuda")
    model.cuda()
    return model



def tokenize_audio(audio_file_path, snac_model):
    audio_array, sample_rate = librosa.load(audio_file_path, sr=24000)
    waveform = torch.from_numpy(audio_array).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    waveform = waveform.unsqueeze(0)

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

    return all_codes


def prepare_inputs(
    fpath_audio_ref,
    audio_ref_transcript: str,
    text_prompts: list[str],
    snac_model,
    tokenizer,
):

    
    start_tokens = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)


    all_modified_input_ids = []
    for prompt in text_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        #second_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, input_ids, end_tokens], dim=1)
        second_input_ids = torch.cat([start_tokens, input_ids, end_tokens], dim=1)
        all_modified_input_ids.append(second_input_ids)

    all_padded_tensors = []
    all_attention_masks = []
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64),
                                    torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)

    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    input_ids = all_padded_tensors.to("cuda")
    attention_mask = all_attention_masks.to("cuda")
    return input_ids, attention_mask



def inference(model, input_ids, attention_mask):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.2,
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.9,
            num_return_sequences=1,
            eos_token_id=128258,

        )

        generated_ids = torch.cat([generated_ids, torch.tensor([[128262]]).to("cuda")], dim=1) # EOAI

        return generated_ids


def convert_tokens_to_speech(generated_ids, snac_model):
    token_to_find = 128257
    token_to_remove = 128258
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
    else:
        cropped_tensor = generated_ids

    _mask = cropped_tensor != token_to_remove
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7 
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    my_samples = []
    for code_list in code_lists:
        samples = redistribute_codes(code_list, snac_model)
        my_samples.append(samples)

    return my_samples


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
        torch.tensor(layer_3).unsqueeze(0)
    ]
    audio_hat = snac_model.decode(codes)
    return audio_hat 


def to_wav_from(samples: list) -> list[np.ndarray]:
    """Converts a list of PyTorch tensors (or NumPy arrays) to NumPy arrays."""
    processed_samples = []

    for s in samples:
        if isinstance(s, torch.Tensor):  
            s = s.detach().squeeze().to('cpu').numpy()
        else:  
            s = np.squeeze(s)

        processed_samples.append(s)

    return processed_samples


def zero_shot_tts(fpath_audio_ref, audio_ref_transcript, texts: list[str], model, snac_model, tokenizer):
    print(f"fpath_audio_ref {fpath_audio_ref}")
    print(f"audio_ref_transcript {audio_ref_transcript}")
    print(f"texts {texts}")
    inp_ids, attn_mask = prepare_inputs(fpath_audio_ref, audio_ref_transcript, texts, snac_model, tokenizer)
    print(f"input_id_len:{len(inp_ids)}")
    gen_ids = inference(model, inp_ids, attn_mask)
    samples = convert_tokens_to_speech(gen_ids, snac_model)
    wav_forms = to_wav_from(samples)
    return wav_forms


def save_wav(samples: list[np.array], sample_rate: int, filenames: list[str]):
    """ Saves a list of tensors as .wav files.

    Args:
        samples (list[torch.Tensor]): List of audio tensors.
        sample_rate (int): Sample rate in Hz.
        filenames (list[str]): List of filenames to save.
    """
    wav_data = to_wav_from(samples)

    for data, filename in zip(wav_data, filenames):
        write(filename, sample_rate, data.astype(np.float32))
        print(f"saved to {filename}")


def get_ref_audio_and_transcript(root_folder: str):
    root_path = Path(root_folder)
    print(f"root_path   {root_path}")
    out = []
    for speaker_folder in root_path.iterdir():
        if speaker_folder.is_dir():  # Ensure it's a directory
            wav_files = list(speaker_folder.glob("*.wav"))
            txt_files = list(speaker_folder.glob("*.txt"))

            if wav_files and txt_files:
                ref_audio = wav_files[0]  # Assume only one .wav file per folder
                transcript = txt_files[0].read_text(encoding="utf-8").strip()
                out.append((ref_audio, transcript))

    return out

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    process_data(content)
    rresponse = {
        'received': content,
        'status': 'success'
    }
    response= jsonify(rresponse)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response



def process_data(jsonText):
    texts = [f"{jsonText['text']}"]
    #print(f"texts:{texts}")
    #print(f"prompt_pairs:{prompt_pairs}")
    for fpath_audio, audio_transcript in prompt_pairs:
        print(f"zero shot: {fpath_audio} {audio_transcript}")
        wav_forms = zero_shot_tts(fpath_audio, audio_transcript, texts, model, snac_model, tokenizer)

        import os
        from pathlib import Path
        from datetime import datetime
        out_dir = Path(fpath_audio).parent / "inference"
        #print(f"out_dir:{out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)  #
        timestamp_str = str(int(datetime.now().timestamp()))
        file_names = [f"{out_dir.as_posix()}/{Path(fpath_audio).stem}_{i}_{timestamp_str}.wav" for i, t in enumerate(texts)]
        #print(f"file_names:{file_names}")
        save_wav(wav_forms, 24000, file_names)
        
      

if __name__ == "__main__":
    tokenizer = load_orpheus_tokenizer()
    model = load_orpheus_auto_model()
    snac_model = load_snac()
    prompt_pairs = get_ref_audio_and_transcript("D:\\AI_APPS\\Orpheus-TTS\\data")
    print(f"snac_model loaded")
    app.run(debug=True,port=5400)
   


```