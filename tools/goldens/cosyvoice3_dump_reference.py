#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump CosyVoice 3 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) reference tensors
for brain's parity ladder.

This is the CosyVoice 3 sibling of `cosyvoice_dump_reference.py` (CosyVoice 2)
- same self-validation discipline, same manifest.json/per-component-meta.json
structure, same f32-safetensors-style raw dump convention, same monkeypatch-
for-reproducibility tricks - extended to CosyVoice 3's real topology, which
was read line-by-line against the live reference source before writing this
(porting.md §0-1), not assumed from the summary that scoped this milestone:

  - `CosyVoice3LM` (`cosyvoice/llm/llm.py`): `sos = speech_token_size + 0`,
    `eos_token = +1`, `task_id = +2`, `fill_token = +3`, all read from
    `speech_embedding` (a `speech_token_size + 200` wide table) - NOT a
    separate `llm_embedding` table the way `Qwen2LM`/CosyVoice2 uses.
    `llm_decoder` projects to `speech_token_size + 200`, not `+3`. Verified by
    reading `Qwen2LM.inference()` (shared code both classes call through
    `self.__class__.__name__` dispatch, not overridden) line-by-line.
  - `CosyVoice3LM.inference()` hard-asserts token id `151646`
    (`<|endofprompt|>`) is present in the concatenated `prompt_text + text`.
    Verified empirically (not assumed) that this id is stable: the base
    `CosyVoice-BlankEN` Qwen2.5 tokenizer's `vocab.json` has exactly 151643
    entries and its `tokenizer_config.json` carries exactly 3
    `added_tokens_decoder` entries (151643/4/5 = endoftext/im_start/im_end);
    `CosyVoice3Tokenizer.__init__` calls `add_special_tokens` with
    `<|endofprompt|>` FIRST in its `additional_special_tokens` list after
    those three, so HF's tokenizer deterministically assigns it 151646 - the
    same id CosyVoice2's tokenizer also lands on, which is exactly why the
    reference hardcodes it instead of looking it up. This dumper follows the
    real `example.py`'s own zero-shot convention (grepped, not guessed) and
    prepends `"You are a helpful assistant.<|endofprompt|>"` to the prompt
    text; `CosyVoiceFrontEnd.text_normalize` has a builtin escape hatch for
    this (`if '<|' in text and '|>' in text: text_frontend = False`), so no
    normalizer touches the literal token.
  - `CausalMaskedDiffWithDiT` (`cosyvoice/flow/flow.py`): `input_size=80`
    (the token embedding table is `Embedding(6561, 80)`, not 512), genuinely
    NO encoder field - just `pre_lookahead_layer` then
    `h.repeat_interleave(token_mel_ratio, dim=1)` - confirmed absent, not
    merely unused, by reading `CausalMaskedDiffWithDiT.__init__`/`.inference`
    end to end. `ConditionalCFM`/`CausalConditionalCFM`
    (`cosyvoice/flow/flow_matching.py`) are the exact same shared classes
    CosyVoice2 uses - same `forward_estimator(x, mask, mu, t, spks, cond,
    streaming)` call convention regardless of whether `self.estimator` is the
    UNet (`CausalConditionalDecoder`) or the `DiT` - confirmed by reading
    `DiT.forward`'s signature, which matches exactly. The fixed CFM noise
    buffer (`rand_noise`, `set_all_random_seed(0); torch.randn([1,80,15000])`)
    is therefore bit-for-bit the SAME mechanism as CosyVoice2's, not a new
    one to port.
  - `CausalHiFTGenerator` (`cosyvoice/hifigan/generator.py`): causal convs
    throughout (`conv_pre` is a right-looking `CausalConv1d`, `ups[i]` are
    `CausalConv1dUpsample` - nearest-upsample + causal conv, NOT
    `ConvTranspose1d`). Its `inference()` signature is `(speech_feat,
    finalize=True)` - genuinely NO `cache_source` parameter (CosyVoice2's
    `HiFTGenerator.inference` takes one; this dumper's CV2 sibling passes
    `cache_source=torch.zeros(1,1,0)` - that call would fail here, verified
    by reading the real signature, not by trial and error). Its
    `f0_predictor` is explicitly upcast to float64 for the duration of the
    call (`self.f0_predictor.to(torch.float64)` - a real, load-bearing
    precision requirement the reference's own comment calls out: "f0_predictor
    precision is crucial for causal inference").
  - **The RNG story is DIFFERENT from CosyVoice2's, verified by reading
    `SineGen2.__init__`/`_f02sine`/`forward` and `SourceModuleHnNSF.__init__`/
    `.forward` line-by-line**: when `causal=True` (CosyVoice3's case) AND
    `self.training is False` (eval mode), the module reads from FIXED
    buffers (`self.rand_ini`, `self.sine_waves`, `self.uv`) that are drawn
    ONCE, at `__init__` time (plain `torch.rand(...)` attribute assignments,
    never `register_buffer`'d, so never saved in `hift.pt`) - NOT redrawn on
    every `inference()` call the way CosyVoice2's non-causal `SineGen2`
    redraws from the global RNG every time. Consequence: calling
    `hift.inference()` twice on the SAME already-constructed model instance
    is bit-exact WITHOUT reseeding between the two calls (this dumper's own
    self-validation proves exactly that, see `dump_hift`) - reproducibility
    instead depends on the global RNG state at MODEL CONSTRUCTION time
    (`load_hyperpyyaml(...)` builds `hift` eagerly), which is why `main()`
    seeds before that call, not immediately before `hift.inference()` the way
    the CosyVoice2 dumper does.

Reference source: `github.com/FunAudioLLM/CosyVoice` (branch `main`, shared
with the CosyVoice2 dumper - verified this session that `main` already
carries `CosyVoice3LM`/`CausalMaskedDiffWithDiT`/`CausalHiFTGenerator`, no
second clone needed), fetched by `resources/cosyvoice/fetch.py` into
`resources/cosyvoice/source/`. Checkpoint:
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, fetched by the same script's `--cv3`
mode into `resources/cosyvoice/weights3/` (two files, `campplus.onnx` and all
of `CosyVoice-BlankEN/`, are hardlinked from the CosyVoice2 fetch rather than
re-downloaded - verified byte-identical by sha256 against the CV3 repo's own
reported LFS hashes, see `fetch.py`'s `HARDLINK_FROM_CV2` table).

Needs the same scratch venv `resources/cosyvoice/.venv` the CosyVoice2 dumper
uses, PLUS one extra package the DiT estimator imports that CosyVoice2's own
code never touches:

    resources/cosyvoice/.venv/bin/pip install x-transformers

(`cosyvoice.flow.DiT.dit`/`.modules` import `RotaryEmbedding`/
`apply_rotary_pos_emb` from `x_transformers.x_transformers` - checked this
venv's installed package list first, per this file's own instructions:
everything else CosyVoice2's dumper already required - torch/torchaudio,
transformers==4.51.3, onnxruntime, soundfile, numpy, safetensors,
huggingface_hub, conformer==0.3.2, hyperpyyaml, omegaconf, openai-whisper,
inflect, librosa, einops, tiktoken, hydra-core, rootutils, rich, lightning,
gdown, matplotlib, wget, pyworld, pyarrow - is untouched by CosyVoice3's
import chain too).

Usage:
    tools/goldens/cosyvoice3_dump_reference.py \\
        --weights resources/cosyvoice/weights3 \\
        --source resources/cosyvoice/source \\
        --out testdata/golden/cosyvoice3

Swedish Embedded AB implements solutions for porting reference PyTorch TTS
pipelines to from-scratch GPU kernels for its clients. If your team needs
expertise in bringing a streaming speech model to an edge-deployable engine,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import sha256_of, source_block  # noqa: E402

SEED = 0
CHECKPOINT = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TAG = "real"  # no --tiny tier yet - mirrors the CosyVoice2 dumper's own note

# The reference repo's own `example.py` zero-shot convention for CosyVoice3:
# `<|endofprompt|>` must appear in prompt_text/text (see module docstring).
# `zero_shot_prompt.wav`'s own caption ("希望你以后能够做的比我还好呦。") is
# reused unchanged from the CosyVoice2 dumper - same asset, same speaker.
DEFAULT_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
DEFAULT_TTS_TEXT = "收到好友从远方寄来的生日礼物。"
DEFAULT_AR_TOKENS = 32  # cap on captured AR-decoded speech tokens - CPU-tractable "handful", not a full utterance


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def write_i32(path, arr):
    np.asarray(arr, dtype=np.int32).reshape(-1).tofile(path)


def sha256_bytes(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


# ---------------------------------------------------------------------------
# setup: patched wav loader (same torchcodec/ffmpeg gotcha as the CV2 dumper)
# ---------------------------------------------------------------------------

def install_soundfile_wav_loader():
    import soundfile as sf
    import torchaudio
    import cosyvoice.utils.file_utils as fu
    import cosyvoice.cli.frontend as frontend_mod

    def load_wav_sf(wav, target_sr, min_sr=16000):
        data, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data.T)
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, f"wav sample rate {sample_rate} must be >= {min_sr}"
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    fu.load_wav = load_wav_sf
    frontend_mod.load_wav = load_wav_sf
    return load_wav_sf


def frontend_load(frontend, wav_path, sr):
    import cosyvoice.utils.file_utils as fu
    return fu.load_wav(wav_path, sr)


# ---------------------------------------------------------------------------
# component 1: mel front end (matcha.utils.audio.mel_spectrogram) - IDENTICAL
# config to CosyVoice2 (n_fft=1920, 80 mels, hop=480, win=1920, fmin=0,
# fmax=8000, center=False - verified against cosyvoice3.yaml's own
# `feat_extractor`/`mel_spec_transform1` blocks, byte-for-byte the same
# numbers as cosyvoice2.yaml).
# ---------------------------------------------------------------------------

def mel_spectrogram_numpy(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax=None):
    """Independent reimplementation of `matcha.utils.audio.mel_spectrogram`
    (reflect-pad n_fft/2-ish, magnitude STFT, Slaney mel + Slaney norm,
    log(clamp(x, 1e-5))) - the dumper's second independent path for the
    self-validation porting.md requires.

    `fmax=None` (librosa's own default -> sr/2) matches CosyVoice3's real
    `cosyvoice3.yaml` `feat_extractor`/`mel_spec_transform1` blocks - a real,
    VERIFIED-not-assumed divergence from CosyVoice2's `fmax: 8000`: the
    milestone brief's own architecture summary claimed the mel front end was
    unchanged between generations, and it is wrong on this one field (caught
    by this exact self-validation failing at cosine 0.973 with fmax=8000
    hardcoded, before this fix)."""
    from librosa.filters import mel as librosa_mel_fn

    pad = (n_fft - hop_size) // 2
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    window = _hann_periodic(win_size)  # torch.hann_window default periodic=True
    n_frames = 1 + (len(y_pad) - n_fft) // hop_size
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        start = t * hop_size
        frame = y_pad[start:start + n_fft]
        windowed = np.zeros(n_fft, dtype=np.float64)
        windowed[: len(window)] = frame[: len(window)] * window
        spec[:, t] = np.fft.rfft(windowed)
    magnitude = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ magnitude
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype(np.float32)


def _hann_periodic(n):
    return np.hanning(n + 1)[:-1]


def dump_mel(out_dir, feat_extractor, wav):
    # NOTE real, verified-against-cosyvoice3.yaml divergence from CosyVoice2:
    # fmax is `null` (None -> librosa default sr/2), NOT 8000. See
    # mel_spectrogram_numpy's own docstring for how this was caught.
    mel_params = dict(n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
                       win_size=1920, fmin=0, fmax=None)
    with torch.no_grad():
        mel_torch = feat_extractor(wav)  # (1, num_mels, T)
    mel_np = mel_spectrogram_numpy(wav.squeeze(0).numpy().astype(np.float64), **mel_params)

    a, b = mel_torch.squeeze(0).numpy(), mel_np
    cos = float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.abs(a - b).max())

    prefix = os.path.join(out_dir, f"mel_{TAG}")
    write_f32(prefix + "_in.f32", wav)
    write_f32(prefix + "_out.f32", mel_torch)
    write_f32(prefix + "_out_independent.f32", mel_np)
    ident = {k: int(v) for k, v in mel_params.items() if v is not None}
    ident["fmax_is_none"] = 1  # source_block's identity only accepts ints - record the None-ness as a flag
    meta = {
        "in_shape": list(wav.shape), "out_shape": list(mel_torch.shape),
        "architecture_note": "fmax=None (sr/2), NOT 8000 like CosyVoice2 - verified against the real "
                              "cosyvoice3.yaml feat_extractor/mel_spec_transform1 blocks, a real divergence the "
                              "milestone brief's summary got wrong (it claimed this front end was unchanged)",
        "self_validation": {"method": "torch matcha.utils.audio.mel_spectrogram vs independent numpy "
                                       "reflect-pad+Slaney-mel reimplementation of the documented formula",
                             "cosine": cos, "max_abs_diff": max_abs, "pass": bool(cos > 0.9999 and max_abs < 1e-2)},
        "source": source_block(checkpoint=CHECKPOINT, identity=ident),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"mel[{TAG}]: in {tuple(wav.shape)} -> out {tuple(mel_torch.shape)} "
          f"cosine={cos:.7f} max_abs_diff={max_abs:.3e}")
    return meta


# ---------------------------------------------------------------------------
# component 2: CAM++ x-vector (onnxruntime) - byte-identical checkpoint to
# CosyVoice2's (verified by sha256 in fetch.py's HARDLINK_FROM_CV2 table).
# ---------------------------------------------------------------------------

def dump_campplus(out_dir, frontend, wav_path, weights_dir):
    import torchaudio.compliance.kaldi as kaldi

    speech = frontend_load(frontend, wav_path, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)

    emb1 = frontend._extract_spk_embedding(wav_path)
    emb2 = frontend._extract_spk_embedding(wav_path)
    exact = torch.equal(emb1, emb2)

    prefix = os.path.join(out_dir, f"campplus_{TAG}")
    write_f32(prefix + "_in.f32", feat)
    write_f32(prefix + "_out.f32", emb1)
    meta = {
        "in_shape": list(feat.shape), "out_shape": list(emb1.shape),
        "self_validation": {"method": "run the onnxruntime session twice on identical input, assert bit-exact "
                                       "(CPU determinism, not a math cross-check - CAM++ has no second reference "
                                       "implementation in this repo)",
                             "bit_exact": bool(exact), "pass": bool(exact)},
        "source": source_block(checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "campplus.onnx")],
                                identity={"spk_embed_dim": int(emb1.shape[-1])}),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"campplus[{TAG}]: in {tuple(feat.shape)} -> out {tuple(emb1.shape)} bit_exact={exact}")
    return meta, emb1


# ---------------------------------------------------------------------------
# component 3: S3Tokenizer v3 FSQ token ids (onnxruntime, 12-layer MinMo
# encoder - the encoder itself is a deferred milestone per s3tokenizer's own
# roadmap gap; this dumper only needs the ONNX graph's token OUTPUT).
# ---------------------------------------------------------------------------

def dump_s3tokenizer(out_dir, frontend, wav_path, weights_dir):
    import whisper

    speech = frontend_load(frontend, wav_path, 16000)
    feat = whisper.log_mel_spectrogram(speech, n_mels=128)

    tok1, len1 = frontend._extract_speech_token(wav_path)
    tok2, len2 = frontend._extract_speech_token(wav_path)
    exact = tok1.tolist() == tok2.tolist()

    prefix = os.path.join(out_dir, f"s3tokenizer_{TAG}")
    write_f32(prefix + "_in.f32", feat)
    write_i32(prefix + "_tokens.i32", tok1)
    meta = {
        "in_shape": list(feat.shape), "num_tokens": int(tok1.shape[1]),
        "self_validation": {"method": "run the onnxruntime session twice on identical input, assert token ids "
                                       "match exactly (these are discrete ids - exact match is the real gate, "
                                       "not cosine)",
                             "exact_match": bool(exact), "pass": bool(exact)},
        "source": source_block(checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "speech_tokenizer_v3.onnx")],
                                identity={"num_mels": 128, "speech_token_size": 6561}),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"s3tokenizer[{TAG}]: in {tuple(feat.shape)} -> {tok1.shape[1]} tokens, exact_match={exact}")
    return meta, tok1, len1


# ---------------------------------------------------------------------------
# component 4: LM (CosyVoice3LM.inference via the shared Qwen2LM.inference,
# real prompt, REAL ras_sampling reseeded) - see module docstring for the
# sos/task_id-from-speech_embedding and 151646-assertion findings.
# ---------------------------------------------------------------------------

def _run_llm_capped(llm, model_input, max_tokens, capture_hidden=None):
    """One capped call to the real `Qwen2LM.inference()` generator (shared by
    `CosyVoice3LM` - not overridden), same rationale as the CosyVoice2
    dumper's twin function: greedy decoding degenerates (confirmed there),
    so this uses the real `ras_sampling`, reproducible via reseeding torch's
    global RNG right before the call."""
    torch.manual_seed(SEED)
    handle = None
    if capture_hidden is not None:
        def hook(module, args, kwargs, output):
            capture_hidden.append(output.hidden_states[-1].detach().clone())
        handle = llm.llm.model.register_forward_hook(hook, with_kwargs=True)
    try:
        gen = llm.inference(
            text=model_input["text"], text_len=model_input["text_len"],
            prompt_text=model_input["prompt_text"], prompt_text_len=model_input["prompt_text_len"],
            prompt_speech_token=model_input["llm_prompt_speech_token"],
            prompt_speech_token_len=model_input["llm_prompt_speech_token_len"],
            embedding=model_input["llm_embedding"],
        )
        tokens = []
        for tok in gen:
            tokens.append(tok)
            if len(tokens) >= max_tokens:
                break
    finally:
        if handle is not None:
            handle.remove()
    return tokens


def dump_llm(out_dir, model, model_input, weights_dir, max_tokens):
    llm = model.llm
    captures = []
    tokens = _run_llm_capped(llm, model_input, max_tokens, capture_hidden=captures)
    assert len(captures) >= 1, "no forward_one_step calls captured - LM inference produced 0 tokens"
    prefill_hidden = captures[0]

    # CosyVoice3's own speech-token logits are `llm_decoder(hidden_states)`
    # (896 -> speech_token_size+200=6761) - NOT the HF Qwen2ForCausalLM's own
    # `.logits` (896 -> ~151936 text vocab). Same real-vs-unused-head
    # distinction the CosyVoice2 dumper already documented, just a wider
    # special-token tail (200, not 3).
    with torch.inference_mode():
        prefill_logits = llm.llm_decoder(prefill_hidden)

    tokens2 = _run_llm_capped(llm, model_input, max_tokens)
    exact = tokens == tokens2

    prefix = os.path.join(out_dir, f"llm_{TAG}")
    write_i32(prefix + "_text.i32", model_input["text"])
    write_i32(prefix + "_prompt_text.i32", model_input["prompt_text"])
    write_f32(prefix + "_prefill_hidden.f32", prefill_hidden)
    write_f32(prefix + "_prefill_logits.f32", prefill_logits)
    write_i32(prefix + "_ar_tokens.i32", np.array(tokens, dtype=np.int32))
    meta = {
        "prefill_hidden_shape": list(prefill_hidden.shape), "prefill_logits_shape": list(prefill_logits.shape),
        "num_ar_tokens": len(tokens), "ar_tokens_capped_at": max_tokens,
        "sampling": "real ras_sampling (top_p=0.8, top_k=25, win_size=10, tau_r=0.1), reseeded to SEED for reproducibility",
        "endofprompt_token_id": 151646,
        "sos_task_id_source": "speech_embedding (NOT llm_embedding - CosyVoice3LM has no llm_embedding table)",
        "self_validation": {"method": "reseed torch to SEED, run the real capped inference() generator twice "
                                       "independently, assert the sampled token sequences match exactly",
                             "tokens_run1": tokens, "tokens_run2": tokens2, "pass": bool(exact)},
        "source": source_block(
            checkpoint=CHECKPOINT,
            files=[os.path.join(weights_dir, "llm.pt"), os.path.join(weights_dir, "CosyVoice-BlankEN/model.safetensors")],
            identity={"llm_input_size": 896, "llm_output_size": 896, "speech_token_size": 6561,
                      "llm_decoder_out_features": int(prefill_logits.shape[-1])},
        ),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"llm[{TAG}]: prefill_hidden {tuple(prefill_hidden.shape)}, {len(tokens)} AR tokens, "
          f"reseed_determinism={exact}")
    return meta, tokens


# ---------------------------------------------------------------------------
# component 5: flow (CausalMaskedDiffWithDiT.inference, DiT CFM estimator,
# CFM Euler solver - the SAME ConditionalCFM/CausalConditionalCFM classes
# CosyVoice2 uses; only the estimator module and the (encoder-free) condition
# assembly differ. See module docstring.)
# ---------------------------------------------------------------------------

def _solve_euler_capturing(self, x, t_span, mu, mask, spks, cond, streaming=False, _capture=None):
    """Line-for-line copy of `ConditionalCFM.solve_euler` with one addition:
    it appends the post-step latent to `_capture`. Identical to the
    CosyVoice2 dumper's twin function - `ConditionalCFM`/`CausalConditionalCFM`
    are literally the same shared classes, not reimplemented per generation."""
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)
    x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
    t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
    spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = spks
        cond_in[0] = cond
        dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        if _capture is not None:
            _capture.append(x.detach().clone())
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x.float()


def dump_flow(out_dir, model, model_input, ar_tokens, weights_dir):
    flow = model.flow
    decoder = flow.decoder
    estimator = decoder.estimator  # DiT, not CausalConditionalDecoder

    # rand_noise self-validation: same fixed seed(0) buffer mechanism as
    # CosyVoice2 (CausalConditionalCFM.__init__ is the literal same class).
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(SEED)
    rand_noise_replay = torch.randn([1, 80, 50 * 300])
    noise_exact = torch.equal(decoder.rand_noise, rand_noise_replay)

    # forward_pre_hook: capture (mu, mask, spks, cond) exactly as the real
    # `flow.inference()` produced them - no encoder here (CosyVoice3 has
    # none), just pre_lookahead_layer + repeat_interleave, verified absent by
    # reading CausalMaskedDiffWithDiT.__init__/.inference (module docstring).
    pre_capture = {}

    def pre_hook(module, args, kwargs):
        pre_capture.update(kwargs)

    post_capture = {}

    def post_hook(module, args, kwargs, output):
        post_capture["feat_full"] = output[0].detach().clone()

    h1 = decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = decoder.register_forward_hook(post_hook, with_kwargs=True)

    embedding_capture = {}

    def emb_hook(module, args, output):
        embedding_capture["embedding"] = output.detach().clone()

    h3 = flow.spk_embed_affine_layer.register_forward_hook(emb_hook)

    # DiT-specific internals: InputEmbedding's concatenated input (x, cond,
    # text_embed=mu, spks) + its output, and TimestepEmbedding's (t -> emb)
    # - captured from the FIRST Euler step's forward only (a real forward the
    # module actually computed, not hand-assembled), self-validated by an
    # independent recompute from the captured inputs against the captured
    # output further below.
    dit_ie_capture = {}

    def ie_hook(module, args, kwargs, output):
        if "x" not in dit_ie_capture:
            dit_ie_capture["x"] = args[0].detach().clone()
            dit_ie_capture["cond"] = args[1].detach().clone()
            dit_ie_capture["text_embed"] = args[2].detach().clone()
            dit_ie_capture["spks"] = args[3].detach().clone()
            dit_ie_capture["out"] = output.detach().clone()

    dit_te_capture = {}

    def te_hook(module, args, output):
        if "t_in" not in dit_te_capture:
            dit_te_capture["t_in"] = args[0].detach().clone()
            dit_te_capture["out"] = output.detach().clone()

    h4 = estimator.input_embed.register_forward_hook(ie_hook, with_kwargs=True)
    h5 = estimator.time_embed.register_forward_hook(te_hook)

    euler_steps = []
    orig_solve_euler = type(decoder).solve_euler
    entry_args = {}

    def patched(self, x, t_span, mu, mask, spks, cond, streaming=False):
        entry_args.update(x=x.clone(), t_span=t_span.clone(), mu=mu.clone(), mask=mask.clone(),
                          spks=spks.clone(), cond=cond.clone(), streaming=streaming)
        return _solve_euler_capturing(self, x, t_span, mu, mask, spks, cond, streaming, _capture=euler_steps)

    decoder.solve_euler = types.MethodType(patched, decoder)

    token = torch.tensor([ar_tokens], dtype=torch.int32)
    token_len = torch.tensor([len(ar_tokens)], dtype=torch.int32)
    feat, _ = flow.inference(
        token=token, token_len=token_len,
        prompt_token=model_input["flow_prompt_speech_token"], prompt_token_len=model_input["flow_prompt_speech_token_len"],
        prompt_feat=model_input["prompt_speech_feat"], prompt_feat_len=model_input["prompt_speech_feat_len"],
        embedding=model_input["flow_embedding"], streaming=False, finalize=True,
    )

    h1.remove(); h2.remove(); h3.remove(); h4.remove(); h5.remove()
    decoder.solve_euler = orig_solve_euler  # restore before the independent replay

    # self-validation #1: last captured Euler step vs the decoder module's
    # own forward-hook-captured return value.
    step_match = torch.equal(post_capture["feat_full"], euler_steps[-1].float())

    # self-validation #2: independent second call to the UNPATCHED
    # solve_euler with the exact entry args, bit-exact against the patched
    # run's last step.
    with torch.inference_mode():
        replay = orig_solve_euler(decoder, entry_args["x"], entry_args["t_span"], entry_args["mu"],
                                  entry_args["mask"], entry_args["spks"], entry_args["cond"],
                                  streaming=entry_args["streaming"])
    replay_match = torch.equal(replay, euler_steps[-1].float())

    # self-validation #3 (DiT-specific): recompute InputEmbedding/
    # TimestepEmbedding from the captured inputs and assert the recomputed
    # output matches the hook-captured one bit-exactly - proves the hook
    # captured a real, reproducible forward, not an artifact of hook timing.
    with torch.inference_mode():
        ie_recompute = estimator.input_embed(dit_ie_capture["x"], dit_ie_capture["cond"],
                                             dit_ie_capture["text_embed"], dit_ie_capture["spks"])
        te_recompute = estimator.time_embed(dit_te_capture["t_in"])
    ie_match = torch.equal(ie_recompute, dit_ie_capture["out"])
    te_match = torch.equal(te_recompute, dit_te_capture["out"])

    prefix = os.path.join(out_dir, f"flow_{TAG}")
    write_f32(prefix + "_conds.f32", pre_capture["cond"])
    write_f32(prefix + "_mu.f32", pre_capture["mu"])
    write_f32(prefix + "_embedding.f32", embedding_capture["embedding"])
    write_f32(prefix + "_euler_steps.f32", torch.stack(euler_steps, dim=0))
    write_f32(prefix + "_mel_out.f32", feat)
    write_f32(prefix + "_rand_noise_slice.f32", decoder.rand_noise[:, :, :100])
    write_f32(prefix + "_dit_input_embed_in_x.f32", dit_ie_capture["x"])
    write_f32(prefix + "_dit_input_embed_in_cond.f32", dit_ie_capture["cond"])
    write_f32(prefix + "_dit_input_embed_in_text_embed.f32", dit_ie_capture["text_embed"])
    write_f32(prefix + "_dit_input_embed_in_spks.f32", dit_ie_capture["spks"])
    write_f32(prefix + "_dit_input_embed_out.f32", dit_ie_capture["out"])
    write_f32(prefix + "_dit_time_embed_in.f32", dit_te_capture["t_in"])
    write_f32(prefix + "_dit_time_embed_out.f32", dit_te_capture["out"])

    meta = {
        "conds_shape": list(pre_capture["cond"].shape), "mu_shape": list(pre_capture["mu"].shape),
        "embedding_shape": list(embedding_capture["embedding"].shape),
        "euler_steps_shape": [len(euler_steps)] + list(euler_steps[0].shape),
        "mel_out_shape": list(feat.shape), "n_timesteps": 10,
        "rand_noise_full_shape": list(decoder.rand_noise.shape),
        "rand_noise_full_sha256": sha256_bytes(decoder.rand_noise.numpy().tobytes()),
        "rand_noise_slice_note": "first 100 of 1,200,000 values dumped; full buffer verified by sha256 only",
        "dit_input_embed_out_shape": list(dit_ie_capture["out"].shape),
        "dit_time_embed_out_shape": list(dit_te_capture["out"].shape),
        "architecture_note": "no encoder (CausalMaskedDiffWithDiT has none - verified absent, not just unused): "
                              "condition assembly is pre_lookahead_layer(token_emb) then "
                              "repeat_interleave(token_mel_ratio); estimator is a 22-layer adaLN-zero DiT, not the "
                              "CausalConditionalDecoder UNet CosyVoice2 uses",
        "self_validation": {
            "rand_noise_two_ways": {
                "method": "decoder.rand_noise (read from the loaded model instance) vs "
                          "set_all_random_seed(0); torch.randn([1,80,15000]) recomputed independently",
                "bit_exact": bool(noise_exact), "pass": bool(noise_exact),
            },
            "captured_vs_module_output": {
                "method": "last captured Euler step (via the monkeypatched solve_euler) vs the decoder module's "
                          "own forward-hook-captured return value",
                "bit_exact": bool(step_match), "pass": bool(step_match),
            },
            "capture_wrapper_vs_original": {
                "method": "independent second call to the UNPATCHED solve_euler with the exact entry args "
                          "captured at the patched call's entry, compared bit-exact against the patched run's "
                          "last step",
                "bit_exact": bool(replay_match), "pass": bool(replay_match),
            },
            "dit_input_embed_recompute": {
                "method": "recompute estimator.input_embed(x, cond, text_embed, spks) from the captured inputs, "
                          "compare bit-exact against the hook-captured output",
                "bit_exact": bool(ie_match), "pass": bool(ie_match),
            },
            "dit_time_embed_recompute": {
                "method": "recompute estimator.time_embed(t) from the captured input, compare bit-exact against "
                          "the hook-captured output",
                "bit_exact": bool(te_match), "pass": bool(te_match),
            },
        },
        "source": source_block(
            checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "flow.pt")],
            identity={"input_size": 80, "output_size": 80, "vocab_size": 6561,
                      "token_mel_ratio": 2, "pre_lookahead_len": 3, "n_timesteps": 10,
                      "dit_dim": 1024, "dit_depth": 22, "dit_heads": 16, "dit_dim_head": 64},
        ),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"flow[{TAG}]: mu {tuple(pre_capture['mu'].shape)}, mel_out {tuple(feat.shape)}, "
          f"noise_exact={noise_exact} step_match={step_match} replay_match={replay_match} "
          f"ie_match={ie_match} te_match={te_match}")
    return meta, feat


# ---------------------------------------------------------------------------
# component 6: HiFT vocoder (CausalHiFTGenerator.inference, ISTFT head) -
# see module docstring for the fixed-buffer-at-construction RNG finding,
# which changes the self-validation strategy vs the CosyVoice2 dumper.
# ---------------------------------------------------------------------------

def dump_hift(out_dir, model, mel_feat, weights_dir):
    hift = model.hift
    capture = {}
    orig_istft = hift._istft

    def patched_istft(magnitude, phase):
        capture["magnitude"] = magnitude.detach().clone()
        capture["phase"] = phase.detach().clone()
        return orig_istft(magnitude, phase)

    hift._istft = patched_istft

    # CausalHiFTGenerator.inference has NO cache_source parameter (a real,
    # verified topology difference from CosyVoice2's HiFTGenerator.inference
    # - see module docstring) - and NO reseed here: SineGen2(causal=True)'s
    # noise buffers were fixed once at model construction (main() seeds
    # before that), so two calls on this same instance should already be
    # bit-exact without reseeding in between. That claim IS the
    # self-validation below, not an assumption.
    with torch.no_grad():
        speech1, source1 = hift.inference(speech_feat=mel_feat, finalize=True)
    mag1, phase1 = capture["magnitude"], capture["phase"]

    with torch.no_grad():
        speech2, _ = hift.inference(speech_feat=mel_feat, finalize=True)
    hift._istft = orig_istft

    reseed_exact = torch.equal(speech1, speech2)

    prefix = os.path.join(out_dir, f"hift_{TAG}")
    write_f32(prefix + "_speech_feat_in.f32", mel_feat)
    write_f32(prefix + "_magnitude.f32", mag1)
    write_f32(prefix + "_phase.f32", phase1)
    write_f32(prefix + "_waveform.f32", speech1)
    meta = {
        "speech_feat_in_shape": list(mel_feat.shape), "magnitude_shape": list(mag1.shape),
        "phase_shape": list(phase1.shape), "waveform_shape": list(speech1.shape),
        "gotcha": "DIFFERENT from CosyVoice2's HiFT gotcha (verified by reading SineGen2/SourceModuleHnNSF "
                  "line-by-line, not assumed): when causal=True (CosyVoice3's CausalHiFTGenerator) and "
                  "self.training is False, SineGen2/SourceModuleHnNSF read FIXED buffers (rand_ini, sine_waves, "
                  "uv) drawn ONCE at __init__ time (plain tensor attributes, never registered as nn.Buffers, so "
                  "never saved in hift.pt) rather than redrawing from the global RNG on every call. Two "
                  "inference() calls on the SAME model instance are therefore bit-exact WITHOUT reseeding between "
                  "them - reproducibility instead depends on the global RNG state at MODEL CONSTRUCTION time, "
                  "which is why this dumper's main() seeds before load_hyperpyyaml() builds the model, not "
                  "immediately before hift.inference() the way the CosyVoice2 dumper does.",
        "self_validation": {"method": "call inference() twice on the same already-constructed model instance, "
                                       "WITHOUT reseeding in between, assert the two waveforms are bit-exact - "
                                       "proves the fixed-buffer-at-construction claim above, not the "
                                       "reseed-before-every-call claim CosyVoice2's dumper proves",
                             "bit_exact": bool(reseed_exact), "pass": bool(reseed_exact)},
        "source": source_block(
            checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "hift.pt")],
            identity={"in_channels": 80, "base_channels": 512, "nb_harmonics": 8, "sampling_rate": 24000,
                      "n_fft": 16, "hop_len": 4, "conv_pre_look_right": 4},
        ),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"hift[{TAG}]: speech_feat {tuple(mel_feat.shape)} -> waveform {tuple(speech1.shape)} "
          f"no_reseed_exact={reseed_exact}")
    return meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="resources/cosyvoice/weights3")
    ap.add_argument("--source", required=True, help="resources/cosyvoice/source")
    ap.add_argument("--out", required=True, help="testdata/golden/cosyvoice3")
    ap.add_argument("--prompt-wav", default=None, help="defaults to <source>/asset/zero_shot_prompt.wav")
    ap.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    ap.add_argument("--tts-text", default=DEFAULT_TTS_TEXT)
    ap.add_argument("--ar-tokens", type=int, default=DEFAULT_AR_TOKENS)
    args = ap.parse_args()

    weights_dir = os.path.abspath(args.weights)
    source_dir = os.path.abspath(args.source)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    prompt_wav = args.prompt_wav or os.path.join(source_dir, "asset", "zero_shot_prompt.wav")

    sys.path.insert(0, source_dir)
    sys.path.insert(0, os.path.join(source_dir, "third_party", "Matcha-TTS"))

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    # Seed BEFORE model construction, not just before hift.inference() - see
    # module docstring's RNG finding: CausalHiFTGenerator's NSF noise buffers
    # are drawn once, right here, when load_hyperpyyaml() builds `hift`.
    torch.manual_seed(SEED)

    from hyperpyyaml import load_hyperpyyaml
    with open(os.path.join(weights_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN")})

    install_soundfile_wav_loader()
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from cosyvoice.cli.model import CosyVoice3Model

    frontend = CosyVoiceFrontEnd(configs["get_tokenizer"], configs["feat_extractor"],
                                 os.path.join(weights_dir, "campplus.onnx"),
                                 os.path.join(weights_dir, "speech_tokenizer_v3.onnx"),
                                 "", configs["allowed_special"])

    model = CosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16=False)
    model.load(os.path.join(weights_dir, "llm.pt"), os.path.join(weights_dir, "flow.pt"),
               os.path.join(weights_dir, "hift.pt"))
    print(f"loaded llm/flow/hift from {weights_dir} (strict state_dict load)")

    # the real frontend pipeline, exactly as CosyVoice3(CosyVoice2).
    # inference_zero_shot builds it - not hand-assembled model inputs. The
    # `<|endofprompt|>` literal in prompt_text disables text_frontend
    # automatically (CosyVoiceFrontEnd.text_normalize's own `'<|' in text`
    # escape hatch), matching example.py's own usage.
    prompt_text = frontend.text_normalize(args.prompt_text, split=False, text_frontend=True)
    model_input = frontend.frontend_zero_shot(args.tts_text, prompt_text, prompt_wav, 24000, "")

    results = {}

    # -- mel (independent front-end tap, not tied to the tokenizer pipeline)
    wav24 = frontend_load(frontend, prompt_wav, 24000)
    results["mel"] = dump_mel(out_dir, configs["feat_extractor"], wav24)

    results["campplus"] = dump_campplus(out_dir, frontend, prompt_wav, weights_dir)[0]
    results["s3tokenizer"] = dump_s3tokenizer(out_dir, frontend, prompt_wav, weights_dir)[0]

    llm_meta, ar_tokens = dump_llm(out_dir, model, model_input, weights_dir, args.ar_tokens)
    results["llm"] = llm_meta

    flow_meta, flow_mel = dump_flow(out_dir, model, model_input, ar_tokens, weights_dir)
    results["flow"] = flow_meta

    results["hift"] = dump_hift(out_dir, model, flow_mel, weights_dir)

    import transformers
    import onnxruntime

    manifest = {
        "checkpoint": CHECKPOINT,
        "note": "CosyVoice 3 goldens - the CosyVoice2 dumper's own manifest recorded these as a deliberate "
                "follow-up; this is that follow-up. Covers: mel front end, CAM++ x-vector, S3Tokenizer v3 FSQ "
                "tokens, CosyVoice3LM prefill+AR tokens, CausalMaskedDiffWithDiT flow (incl. DiT-internal "
                "InputEmbedding/TimestepEmbedding taps), CausalHiFTGenerator vocoder.",
        "tag": TAG,
        "seed": SEED,
        "run_params": {
            "prompt_wav": os.path.relpath(prompt_wav, source_dir),
            "prompt_wav_sha256": sha256_of(prompt_wav),
            "prompt_text": args.prompt_text, "tts_text": args.tts_text,
            "ar_tokens_cap": args.ar_tokens,
        },
        "library_versions": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "onnxruntime": onnxruntime.__version__, "numpy": np.__version__,
        },
        "components": results,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    all_pass = []
    for comp, meta in results.items():
        sv = meta.get("self_validation", {})
        checks = sv.values() if all(isinstance(v, dict) for v in sv.values()) else [sv]
        for c in checks:
            if isinstance(c, dict) and "pass" in c:
                all_pass.append((comp, c["pass"]))
    print("\nself-validation summary:")
    for comp, ok in all_pass:
        print(f"  {comp}: {'PASS' if ok else 'FAIL'}")
    if not all(ok for _, ok in all_pass):
        print("\nSOME SELF-VALIDATION CHECKS FAILED - see manifest.json / *_meta.json", file=sys.stderr)
        return 1
    print(f"\nwrote goldens to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
