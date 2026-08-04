#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump golden reference tensors for the ASR models so the Rust engine can be
parity-gated against the real HuggingFace implementations.

Stage 1 (this pass): the log-mel FRONT ENDS.
    - NemotronAsrStreamingFeatureExtractor  (librosa slaney mel, n_fft 512/win 400,
      Hann periodic=False, constant pad, preemphasis 0.97, log(x+2^-24), no norm)
    - Qwen3ASRFeatureExtractor               (transformers slaney mel, n_fft 400,
      Hann periodic=True, reflect pad, log10 + dynamic-range compression)

Everything is written as little-endian f32 raw blobs plus a JSON manifest of shapes
under resources/asr/golden/<group>/. Deterministic input (fixed seed) so the Rust
side feeds byte-identical samples.

Usage:  python tools/asr_dump_reference.py frontend
"""
import json
import os
import sys

import numpy as np

RES = os.environ.get("BRAIN_ASR_MIRROR", "/data/workspace/resources/asr")
GOLD = os.path.join(RES, "golden")


def save(group: str, name: str, arr, manifest: dict):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    d = os.path.join(GOLD, group)
    os.makedirs(d, exist_ok=True)
    arr.tofile(os.path.join(d, name + ".f32"))
    manifest[name] = list(arr.shape)


def make_waveform(seconds=2.0, sr=16000, seed=1234):
    """Deterministic pseudo-speech: sum of a few swept sinusoids + mild noise, in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    for f0, f1, amp in [(120, 180, 0.5), (440, 400, 0.3), (900, 1300, 0.2), (2100, 1800, 0.1)]:
        inst = f0 + (f1 - f0) * (t / seconds)
        phase = 2 * np.pi * np.cumsum(inst) / sr
        sig += amp * np.sin(phase)
    sig += 0.01 * rng.standard_normal(n)
    # gentle amplitude envelope so it isn't perfectly stationary
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    sig = sig * env
    sig = sig / np.max(np.abs(sig)) * 0.9
    return sig.astype(np.float32)


def dump_frontend():
    import torch
    from transformers import AutoProcessor

    wav = make_waveform()
    man = {"sampling_rate": 16000}
    save("frontend", "waveform", wav, man)

    # ---- Nemotron ----
    proc_n = AutoProcessor.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
    fe_n = proc_n.feature_extractor
    out_n = fe_n(wav, sampling_rate=16000, return_tensors="pt", center=True)
    # (1, T, 128)
    save("frontend", "nemotron_mel", out_n["input_features"][0].numpy(), man)
    save("frontend", "nemotron_mel_filters", fe_n.mel_filters.numpy(), man)  # [128, 257]
    save("frontend", "nemotron_hann", torch.hann_window(fe_n.win_length, periodic=False).numpy(), man)
    man["nemotron"] = dict(n_fft=fe_n.n_fft, hop=fe_n.hop_length, win=fe_n.win_length,
                           preemphasis=fe_n.preemphasis, n_mels=fe_n.feature_size)

    # ---- Qwen3-ASR ----
    proc_q = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    fe_q = proc_q.feature_extractor
    feat_q = fe_q(wav, sampling_rate=16000, return_tensors="pt")
    # Qwen returns input_features (B, 128, T) and input_features_mask
    fk = "input_features" if "input_features" in feat_q else list(feat_q.keys())[0]
    arr_q = feat_q[fk][0].numpy()
    save("frontend", "qwen_mel", arr_q, man)
    # valid (non-padding) frame count so the Rust test compares only real frames
    mask_key = next((k for k in feat_q if "mask" in k), None)
    qvalid = int(np.asarray(feat_q[mask_key][0]).sum()) if mask_key else arr_q.shape[-1]
    man["qwen_valid_frames"] = qvalid
    save("frontend", "qwen_mel_filters", np.asarray(fe_q.mel_filters), man)  # [201, 128]
    save("frontend", "qwen_hann", torch.hann_window(fe_q.n_fft, periodic=True).numpy(), man)
    man["qwen"] = dict(n_fft=fe_q.n_fft, hop=fe_q.hop_length, n_window=fe_q.n_window,
                       n_mels=fe_q.feature_size, mel_out_shape=list(arr_q.shape))

    with open(os.path.join(GOLD, "frontend", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print("frontend goldens written:")
    print(json.dumps(man, indent=2))


def dump_qwen_encoder():
    """Run the Qwen3-ASR audio tower + projector in fp32 on the test waveform and
    dump the encoder output (post ln_post) and the projected audio embeddings."""
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    wav = make_waveform()
    man = {}
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    feat = proc.feature_extractor(
        wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    fk = "input_features"
    mask_key = next(k for k in feat if "mask" in k)  # "attention_mask"
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf", dtype=torch.float32
    ).eval()

    with torch.no_grad():
        enc = model.model.audio_tower(
            input_features=feat[fk].to(torch.float32),
            input_features_mask=feat[mask_key],
        )
        hidden = enc.last_hidden_state  # (n_audio, 1024)
        proj = model.model.multi_modal_projector(hidden)  # (n_audio, 2048)

    save("qwen_encoder", "input_features", feat[fk][0].numpy(), man)
    save("qwen_encoder", "input_features_mask", feat[mask_key][0].numpy().astype("float32"), man)
    save("qwen_encoder", "encoder_out", hidden.numpy(), man)
    save("qwen_encoder", "audio_embeds", proj.numpy(), man)
    man["n_audio"] = int(hidden.shape[0])
    man["enc_dim"] = int(hidden.shape[1])
    man["proj_dim"] = int(proj.shape[1])
    with open(os.path.join(GOLD, "qwen_encoder", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print("qwen encoder goldens written:", json.dumps(man, indent=2))


def dump_qwen_decode():
    """Full Qwen3-ASR transcription on a real clip: dump the prompt input_ids,
    the audio placeholder positions, this clip's mel + audio_embeds, the greedy
    output_ids and the decoded transcription — the end-to-end brain parity target."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    wav_path = os.path.join(RES, "audio", "librispeech_mr_quilter.wav")
    wav, sr = sf.read(wav_path)
    wav = wav.astype(np.float32)
    assert sr == 16000

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf", dtype=torch.float32
    ).eval()

    inputs = proc.apply_transcription_request(audio=wav, language="en", sampling_rate=16000)
    inputs = {k: (v.to(torch.float32) if torch.is_floating_point(v) else v) if torch.is_tensor(v) else v
              for k, v in inputs.items()}

    man = {}
    input_ids = inputs["input_ids"][0].cpu().numpy()
    audio_token_id = model.config.audio_token_id
    audio_pos = np.nonzero(input_ids == audio_token_id)[0]
    save("qwen_decode", "input_ids", input_ids.astype("float32"), man)
    man["audio_token_id"] = int(audio_token_id)
    man["n_audio"] = int(len(audio_pos))
    man["audio_pos_first"] = int(audio_pos[0])
    man["audio_pos_contiguous"] = bool(np.all(np.diff(audio_pos) == 1))
    man["prompt_len"] = int(len(input_ids))

    with torch.no_grad():
        enc = model.model.audio_tower(
            input_features=inputs["input_features"].to(torch.float32),
            input_features_mask=inputs["input_features_mask"],
        )
        audio_embeds = model.model.multi_modal_projector(enc.last_hidden_state)
        save("qwen_decode", "audio_embeds", audio_embeds.cpu().numpy(), man)
        save("qwen_decode", "input_features", inputs["input_features"][0].cpu().numpy(), man)
        save("qwen_decode", "input_features_mask", inputs["input_features_mask"][0].cpu().numpy().astype("float32"), man)

        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen = out[0, input_ids.shape[0]:].cpu().numpy()
    save("qwen_decode", "output_ids", gen.astype("float32"), man)
    text = proc.decode(out[:, input_ids.shape[0]:], return_format="transcription_only")[0]
    man["transcription"] = text
    man["output_len"] = int(len(gen))

    with open(os.path.join(GOLD, "qwen_decode", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print("qwen decode goldens written:", json.dumps({k: v for k, v in man.items() if not isinstance(v, list)}, indent=2))


def dump_nemotron():
    """Nemotron 3.5 ASR on the real clip: dump input_features, per-stage encoder
    activations (subsampling, block 0, encoder last_hidden, projected pooler),
    the greedy output token ids and the transcription — the FastConformer/RNN-T
    brain parity target."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

    wav, sr = sf.read(os.path.join(RES, "audio", "librispeech_mr_quilter.wav"))
    wav = wav.astype(np.float32)
    assert sr == 16000
    mid = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    proc = AutoProcessor.from_pretrained(mid)
    model = Nemotron3_5AsrForRNNT.from_pretrained(mid, dtype=torch.float32).eval()

    inputs = proc(wav, sampling_rate=16000, language="en")
    man = {}
    caught = {}
    enc = model.encoder
    h1 = enc.subsampling.register_forward_hook(lambda m, i, o: caught.__setitem__("subsampling", o.detach()))
    h2 = enc.layers[0].register_forward_hook(lambda m, i, o: caught.__setitem__("block0", (o[0] if isinstance(o, tuple) else o).detach()))
    h3 = enc.register_forward_hook(lambda m, i, o: caught.__setitem__("enc_last", o.last_hidden_state.detach()))

    with torch.no_grad():
        feats = model.get_audio_features(
            input_features=inputs["input_features"].to(torch.float32),
            attention_mask=inputs.get("attention_mask"),
            prompt_ids=inputs["prompt_ids"],
            num_lookahead_tokens=inputs.get("num_lookahead_tokens"),
        )
    for h in (h1, h2, h3):
        h.remove()
    save("nemotron", "input_features", inputs["input_features"][0].numpy(), man)
    save("nemotron", "subsampling", caught["subsampling"][0].numpy(), man)
    save("nemotron", "block0", caught["block0"][0].numpy(), man)
    save("nemotron", "enc_last", caught["enc_last"][0].numpy(), man)
    save("nemotron", "pooler", feats.pooler_output[0].numpy(), man)  # [T, 640]
    man["prompt_id"] = int(inputs["prompt_ids"][0])
    man["num_lookahead_tokens"] = int(inputs.get("num_lookahead_tokens", 3))

    with torch.no_grad():
        out = model.generate(**{k: (v if not torch.is_tensor(v) else v) for k, v in inputs.items()}, max_new_tokens=256)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    seq = seq.cpu().numpy()
    save("nemotron", "output_ids", seq.astype("float32"), man)
    text = proc.batch_decode(out.sequences if hasattr(out, "sequences") else out, skip_special_tokens=True)[0]
    man["transcription"] = text
    with open(os.path.join(GOLD, "nemotron", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print("nemotron goldens:", json.dumps({k: v for k, v in man.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "frontend"
    if stage == "frontend":
        dump_frontend()
    elif stage == "qwen_encoder":
        dump_qwen_encoder()
    elif stage == "qwen_decode":
        dump_qwen_decode()
    elif stage == "nemotron":
        dump_nemotron()
    else:
        raise SystemExit(f"unknown stage {stage!r}")
