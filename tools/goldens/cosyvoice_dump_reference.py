#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`) reference tensors for
brain's parity ladder.

CosyVoice 3 goldens are a DELIBERATE follow-up, not covered by this script -
see the ledger note this dumper's own docstring and `manifest.json` both
carry.

Reference source: `github.com/FunAudioLLM/CosyVoice` (branch `main`), cloned
by `resources/cosyvoice/fetch.py` into `resources/cosyvoice/source/` together
with its `third_party/Matcha-TTS` submodule (`matcha.utils.audio.
mel_spectrogram` and `matcha.models.components.{decoder,transformer,
flow_matching}` are imported directly by the flow decoder - not optional).
Checkpoint: `FunAudioLLM/CosyVoice2-0.5B`, fetched by the same script into
`resources/cosyvoice/weights/`.

This dumper needs a SCRATCH venv this repo's own `requirements.txt`
deliberately does not carry (see its "NOT pip-installable" block for the
general convention) - CosyVoice's own `requirements.txt` pulls in a training
stack (deepspeed/tensorrt/grpc/gradio/...) this dumper never touches; the
actual runtime import chain (frontend -> Qwen2LM -> CausalMaskedDiffWithXvec
-> HiFTGenerator, plus Matcha-TTS's own `matcha.utils` package, which eagerly
imports hydra/lightning/rich/matplotlib/gdown/wget/pyworld/pyarrow for
training-only helpers this dumper never calls but Python still executes at
import time) is:

    python3 -m venv resources/cosyvoice/.venv
    resources/cosyvoice/.venv/bin/pip install torch torchaudio torchcodec \\
        --index-url https://download.pytorch.org/whl/cpu
    resources/cosyvoice/.venv/bin/pip install \\
        "transformers==4.51.3" onnxruntime soundfile numpy safetensors \\
        huggingface_hub conformer==0.3.2 hyperpyyaml omegaconf \\
        openai-whisper inflect librosa diffusers==0.29.0 einops tiktoken \\
        hydra-core rootutils rich lightning gdown matplotlib wget pyworld \\
        pyarrow

Two real, load-bearing reference gotchas this dumper works around/documents
rather than silently papering over:

  - `torchaudio.load(..., backend='soundfile')` no longer honors the
    `backend=` kwarg on the torchaudio version this venv resolves - it always
    routes through `torchcodec`, which needs system ffmpeg .so's this box
    does not have. Worked around by monkeypatching
    `cosyvoice.utils.file_utils.load_wav` (and the name `cosyvoice.cli.
    frontend` already imported into its own namespace) to read via
    `soundfile` directly - same resample-to-target-sr behavior, no ffmpeg.
  - `HiFTGenerator`'s NSF harmonic source (`SourceModuleHnNSF` ->
    `SineGen2._f02sine`, since `sinegen_type='2'` for CosyVoice2's 24kHz -
    ground truth confirmed live against source) draws fresh `torch.rand`/
    `torch.randn` from the GLOBAL RNG on every call when `causal=False`
    (the non-streaming path this dumper exercises) - i.e. real HiFT inference
    output is NOT reproducible run-to-run without the caller reseeding first.
    This dumper calls `torch.manual_seed(SEED)` immediately before
    `hift.inference(...)` to freeze it, and self-validates that choice by
    calling it a second time under the same reseed and asserting bit-exact
    agreement (documented in `hift_real_meta.json`).

Usage:
    tools/goldens/cosyvoice_dump_reference.py \\
        --weights resources/cosyvoice/weights \\
        --source resources/cosyvoice/source \\
        --out testdata/golden/cosyvoice

Swedish Embedded AB implements solutions for porting reference PyTorch TTS
pipelines to from-scratch GPU kernels for its clients. If your team needs
expertise in bringing a streaming speech model to an edge-deployable engine,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import argparse
import copy
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
CHECKPOINT = "FunAudioLLM/CosyVoice2-0.5B"
TAG = "real"  # no --tiny tier yet: crates/cosyvoice::config doesn't exist to sync dims against (see report)

DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"  # the reference repo's own asset/zero_shot_prompt.wav caption
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
# setup: patched wav loader (see module docstring - torchcodec/ffmpeg gotcha)
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


# ---------------------------------------------------------------------------
# component 1: mel front end (matcha.utils.audio.mel_spectrogram)
# ---------------------------------------------------------------------------

def mel_spectrogram_numpy(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """Independent reimplementation of `matcha.utils.audio.mel_spectrogram`
    (reflect-pad n_fft/2-ish, magnitude STFT, Slaney mel + Slaney norm,
    log(clamp(x, 1e-5))) built from the documented formula, not by calling
    the library - this IS the dumper's "second independent path" for the
    self-validation porting.md requires; the reference repo offers no other
    implementation of this exact front end to diff against."""
    from librosa.filters import mel as librosa_mel_fn

    pad = (n_fft - hop_size) // 2
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    window = np.hanning(win_size + 1)[:-1] if False else _hann_periodic(win_size)  # torch.hann_window default periodic=True
    n_frames = 1 + (len(y_pad) - n_fft) // hop_size
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        start = t * hop_size
        frame = y_pad[start:start + n_fft]
        windowed = np.zeros(n_fft, dtype=np.float64)
        # win_size == n_fft here (1920 == 1920); window is centered trivially.
        windowed[: len(window)] = frame[: len(window)] * window
        spec[:, t] = np.fft.rfft(windowed)
    magnitude = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ magnitude
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype(np.float32)


def _hann_periodic(n):
    # torch.hann_window(n) defaults to periodic=True: this is NOT
    # np.hanning(n) (which is the symmetric/non-periodic window) - it is
    # np.hanning(n+1)[:-1], matching torch's periodic definition exactly.
    return np.hanning(n + 1)[:-1]


def dump_mel(out_dir, feat_extractor, wav):
    mel_params = dict(n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
                       win_size=1920, fmin=0, fmax=8000)
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
    meta = {
        "in_shape": list(wav.shape), "out_shape": list(mel_torch.shape),
        "self_validation": {"method": "torch matcha.utils.audio.mel_spectrogram vs independent numpy "
                                       "reflect-pad+Slaney-mel reimplementation of the documented formula",
                             "cosine": cos, "max_abs_diff": max_abs, "pass": bool(cos > 0.9999 and max_abs < 1e-2)},
        "source": source_block(checkpoint=CHECKPOINT, identity={k: int(v) for k, v in mel_params.items()}),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"mel[{TAG}]: in {tuple(wav.shape)} -> out {tuple(mel_torch.shape)} "
          f"cosine={cos:.7f} max_abs_diff={max_abs:.3e}")
    return meta


# ---------------------------------------------------------------------------
# component 2: CAM++ x-vector (onnxruntime)
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


def frontend_load(frontend, wav_path, sr):
    import cosyvoice.utils.file_utils as fu
    return fu.load_wav(wav_path, sr)


# ---------------------------------------------------------------------------
# component 3: S3Tokenizer FSQ token ids (onnxruntime)
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
        "source": source_block(checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "speech_tokenizer_v2.onnx")],
                                identity={"num_mels": 128, "speech_token_size": 6561}),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"s3tokenizer[{TAG}]: in {tuple(feat.shape)} -> {tok1.shape[1]} tokens, exact_match={exact}")
    return meta, tok1, len1


# ---------------------------------------------------------------------------
# component 4: LM (Qwen2LM.inference, real prompt, REAL ras_sampling reseeded)
# ---------------------------------------------------------------------------

def _run_llm_capped(llm, model_input, max_tokens, capture_hidden=None):
    """One capped call to the real `Qwen2LM.inference()` generator - the real
    `ras_sampling` (nucleus + repetition-guard), not patched to greedy.
    Greedy was tried first and rejected: argmax decoding of CosyVoice's
    speech-token distribution gets stuck repeating a single token forever
    (confirmed empirically - 32/32 identical tokens, producing a near-silent
    degenerate mel/waveform through flow+hift), which is exactly the failure
    mode `ras_sampling`'s win_size/tau_r repetition guard exists to avoid.
    Reproducibility instead comes from reseeding torch's global RNG right
    before the call (nucleus_sampling's only randomness is `.multinomial()`
    draws from it)."""
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

    # CosyVoice's own speech-token logits are `llm_decoder(hidden_states)`
    # (896 -> speech_token_size+3=6564) - NOT the HF Qwen2ForCausalLM's own
    # `.logits` output (896 -> ~151936 text vocab), which `Qwen2Encoder.
    # forward_one_step` computes internally but `Qwen2LM` never reads. Dumping
    # the unused HF head would both waste ~70MB and mislabel what "logits"
    # means for this model - so this is the real, consumed projection.
    with torch.inference_mode():
        prefill_logits = llm.llm_decoder(prefill_hidden)

    # self-validation: reseed to SEED and run a second, independent capped
    # generator call; assert the sampled token sequence matches exactly.
    # `ras_sampling`'s nucleus step draws from torch's global RNG, so this is
    # the "capture the exact RNG-consumed sequence" reproducibility path
    # porting.md names as the alternative to greedy patching.
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
# component 5: flow (CausalMaskedDiffWithXvec.inference, CFM Euler solver)
# ---------------------------------------------------------------------------

def _solve_euler_capturing(self, x, t_span, mu, mask, spks, cond, streaming=False, _capture=None):
    """Line-for-line copy of `ConditionalCFM.solve_euler` (`resources/
    cosyvoice/source/cosyvoice/flow/flow_matching.py`) with one addition: it
    appends the post-step latent to `_capture` - the "monkeypatched
    solve_euler... step-end callback" hook point porting.md §1 names as the
    cheapest way to get per-Euler-step latents without touching torch RNG."""
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

    # rand_noise self-validation: two independent recomputations of the fixed
    # seed(0) buffer CausalConditionalCFM.__init__ generated at construction.
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(SEED)
    rand_noise_replay = torch.randn([1, 80, 50 * 300])
    noise_exact = torch.equal(decoder.rand_noise, rand_noise_replay)

    # forward_pre_hook: capture (mu, mask, spks, cond) exactly as the real
    # `flow.inference()` produced them, not hand-assembled.
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

    h1.remove(); h2.remove(); h3.remove()
    decoder.solve_euler = orig_solve_euler  # restore before the independent replay

    # self-validation #1: the module's own captured full-length output must
    # equal my capturing wrapper's last recorded step (same computation, two
    # different read points - proves the capture wrapper didn't perturb it).
    step_match = torch.equal(post_capture["feat_full"], euler_steps[-1].float())

    # self-validation #2: an independent second call to the ORIGINAL
    # (unpatched) solve_euler with the exact entry args, compared bit-exact
    # against the patched run - proves the capturing wrapper is
    # computationally identical to the reference, not just structurally.
    # entry_args tensors were captured inside flow.inference()'s
    # @torch.inference_mode() and stay "inference tensors" forever - replay
    # under the same mode rather than fighting the autograd-tracking guard.
    with torch.inference_mode():
        replay = orig_solve_euler(decoder, entry_args["x"], entry_args["t_span"], entry_args["mu"],
                                  entry_args["mask"], entry_args["spks"], entry_args["cond"],
                                  streaming=entry_args["streaming"])
    replay_match = torch.equal(replay, euler_steps[-1].float())

    prefix = os.path.join(out_dir, f"flow_{TAG}")
    write_f32(prefix + "_conds.f32", pre_capture["cond"])
    write_f32(prefix + "_mu.f32", pre_capture["mu"])
    write_f32(prefix + "_embedding.f32", embedding_capture["embedding"])
    write_f32(prefix + "_euler_steps.f32", torch.stack(euler_steps, dim=0))
    write_f32(prefix + "_mel_out.f32", feat)
    write_f32(prefix + "_rand_noise_slice.f32", decoder.rand_noise[:, :, :100])

    meta = {
        "conds_shape": list(pre_capture["cond"].shape), "mu_shape": list(pre_capture["mu"].shape),
        "embedding_shape": list(embedding_capture["embedding"].shape),
        "euler_steps_shape": [len(euler_steps)] + list(euler_steps[0].shape),
        "mel_out_shape": list(feat.shape), "n_timesteps": 10,
        "rand_noise_full_shape": list(decoder.rand_noise.shape),
        "rand_noise_full_sha256": sha256_bytes(decoder.rand_noise.numpy().tobytes()),
        "rand_noise_slice_note": "first 100 of 1,200,000 values dumped; full buffer verified by sha256 only",
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
                          "last step - proves the capturing wrapper computes identically to the reference",
                "bit_exact": bool(replay_match), "pass": bool(replay_match),
            },
        },
        "source": source_block(
            checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "flow.pt")],
            identity={"input_size": 512, "output_size": 80, "vocab_size": 6561,
                      "token_mel_ratio": 2, "pre_lookahead_len": 3, "n_timesteps": 10},
        ),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"flow[{TAG}]: mu {tuple(pre_capture['mu'].shape)}, mel_out {tuple(feat.shape)}, "
          f"noise_exact={noise_exact} step_match={step_match} replay_match={replay_match}")
    return meta, feat


# ---------------------------------------------------------------------------
# component 6: HiFT vocoder (HiFTGenerator.inference, ISTFT head)
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

    # HiFT's NSF source branch draws fresh global-RNG randomness every call
    # when causal=False (see module docstring) - reseed to freeze it.
    torch.manual_seed(SEED)
    with torch.no_grad():
        speech1, source1 = hift.inference(speech_feat=mel_feat, cache_source=torch.zeros(1, 1, 0))
    mag1, phase1 = capture["magnitude"], capture["phase"]

    torch.manual_seed(SEED)
    with torch.no_grad():
        speech2, _ = hift.inference(speech_feat=mel_feat, cache_source=torch.zeros(1, 1, 0))
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
        "gotcha": "SourceModuleHnNSF's SineGen2 (sinegen_type='2' for 24kHz CosyVoice2, causal=False) draws "
                  "fresh torch.rand/torch.randn from the GLOBAL RNG every inference() call - real HiFT output "
                  "is NOT reproducible run-to-run unless the caller reseeds first. This dumper calls "
                  "torch.manual_seed(SEED) immediately before hift.inference().",
        "self_validation": {"method": "reseed to SEED, run inference() twice, assert the two waveforms are "
                                       "bit-exact - proves the reseed-to-freeze approach actually reproduces",
                             "bit_exact": bool(reseed_exact), "pass": bool(reseed_exact)},
        "source": source_block(
            checkpoint=CHECKPOINT, files=[os.path.join(weights_dir, "hift.pt")],
            identity={"in_channels": 80, "base_channels": 512, "nb_harmonics": 8, "sampling_rate": 24000,
                      "n_fft": 16, "hop_len": 4},
        ),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"hift[{TAG}]: speech_feat {tuple(mel_feat.shape)} -> waveform {tuple(speech1.shape)} "
          f"reseed_exact={reseed_exact}")
    return meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="resources/cosyvoice/weights")
    ap.add_argument("--source", required=True, help="resources/cosyvoice/source")
    ap.add_argument("--out", required=True, help="testdata/golden/cosyvoice")
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
    torch.manual_seed(SEED)

    from hyperpyyaml import load_hyperpyyaml
    with open(os.path.join(weights_dir, "cosyvoice2.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN")})

    install_soundfile_wav_loader()
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from cosyvoice.cli.model import CosyVoice2Model

    frontend = CosyVoiceFrontEnd(configs["get_tokenizer"], configs["feat_extractor"],
                                 os.path.join(weights_dir, "campplus.onnx"),
                                 os.path.join(weights_dir, "speech_tokenizer_v2.onnx"),
                                 "", configs["allowed_special"])

    model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=False)
    model.load(os.path.join(weights_dir, "llm.pt"), os.path.join(weights_dir, "flow.pt"),
               os.path.join(weights_dir, "hift.pt"))
    print(f"loaded llm/flow/hift from {weights_dir} (strict state_dict load)")

    # the real frontend pipeline, exactly as CosyVoice2.inference_zero_shot
    # builds it - not hand-assembled model inputs.
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
        "note": "CosyVoice 3 goldens are a deliberate follow-up - NOT covered by this dumper "
                "(no cosyvoice3.yaml/speech_tokenizer_v3.onnx fetched, no CausalMaskedDiffWithDiT dump). "
                "A gate that never runs is worse than no gate: this is a recorded gap, not silent coverage.",
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
