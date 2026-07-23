# Text-to-Speech (Qwen3-TTS) in brain

End-to-end, pure-Rust voice synthesis: **text (+ an optional reference voice) →
24 kHz waveform**, running on the same portable WGSL engine as the rest of
brain. This is a faithful, parity-verified re-implementation of the
**Qwen3-TTS-12Hz-0.6B** stack — no Python, no PyTorch at inference time.

---

## Architecture

The pipeline is four parity-verified components wired into an autoregressive
voice-synthesis loop (`crates/tts/src/pipeline.rs`):

```
  text ──► Qwen BPE ──► Talker prompt ─┐
 (ref voice) ──► ECAPA x-vector ───────┤
                                       ▼
                            ┌──────────────────────┐
                            │  Talker (Qwen3 dense) │  samples codebook-0 / frame
                            └──────────┬───────────┘
                                       ▼
                            ┌──────────────────────┐
                            │  MTP code predictor   │  fills codebooks 1..15
                            └──────────┬───────────┘
                                       ▼  [T, 16] codec codes
                            ┌──────────────────────┐
                            │  Mimi-style codec     │  decode → 24 kHz waveform
                            └──────────────────────┘
```

- **Talker** (`crates/tts`, `TalkerModel`/`TalkerGen`) — a Qwen3-style dense
  decoder (28 layers, `d_model=1024`, GQA 16/8 heads, SwiGLU, RMSNorm). It
  autoregressively samples the **codebook-0** acoustic token per 12 Hz frame,
  conditioned on projected text embeddings, a speaker x-vector, and optional
  in-context reference codes. It uses **M-RoPE** (interleaved,
  `mrope_section=[24,20,20]`), not the half-split NeoX RoPE used elsewhere.
- **MTP** (multi-token-prediction) code predictor — a small 5-layer Qwen3 block
  that fills the residual **codebooks 1..15** from codebook-0 each frame.
- **Codec** (`crates/codec`, `Codec`) — a **Mimi/Moshi-style** 12 Hz neural
  audio codec. Decode path: `[T,16]` codes → split-residual VQ dequant (1
  semantic + 15 acoustic codebooks) → 8-layer sliding-window GQA transformer
  (window 72) → 2 conv-transpose upsamples → SEANet decoder (SnakeBeta residual
  units, upsample rates 8·5·4·3) → 24 kHz waveform.
- **Speaker encoder** (`crates/speaker`, `SpeakerEncoder`) — an **ECAPA-TDNN**
  x-vector network (initial TDNN → 3× SE-Res2Net blocks → multi-layer feature
  aggregation → attentive statistics pooling → 1×1 conv → 1024-dim embedding).
  Produces the timbre vector for voice cloning. Front-end is the shared
  `audio::mel` log-mel (24 kHz, 128 mels).
- **Audio I/O** (`crates/audio`) — WAV read/write, 1-D conv/transposed-conv
  builders, linear resampling, and the STFT/mel front-end shared by the codec
  and speaker encoder.

---

## CLI: `brain tts {import, clone, synth}`

All flags below are taken from `crates/cli/src/tts_cli.rs`.

### 1. Import — convert the HF checkpoint to brain checkpoints

```bash
brain tts import \
  --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  [--codec-ckpt   <dir>]   # defaults to <ckpt>/speech_tokenizer
  [--speaker-ckpt <dir>]   # defaults to <ckpt> (speaker lives with the Talker)
  [--out-dir out/tts]
```

Writes four brain checkpoints into `--out-dir`:

```
out/tts/talker.weights   out/tts/mtp.weights
out/tts/codec.weights    out/tts/speaker.weights
```

### 2. Synth — speaker-free text-to-speech (no reference voice)

```bash
brain tts synth \
  --text "Hello from brain, running entirely in Rust." \
  --out out.wav \
  [--weights-dir out/tts] [--ckpt <hf_dir>] [--lang english] \
  [--max-frames 256] [--temp 0.0] [--top-k 0] [--seed 0]
```

### 3. Clone — synthesize in the timbre of a reference voice

```bash
# x-vector-only path (pure brain, no external codes needed):
brain tts clone \
  --text "This sentence is spoken in the reference speaker's voice." \
  --ref voice.wav --ref-text "transcript of voice.wav" \
  --out demo.wav \
  [--weights-dir out/tts] [--ckpt <hf_dir>] [--lang english \
   --max-frames 256 --temp 0.0 --top-k 0 --seed 0]

# In-context (ICL) path — additionally conditions on external [T,16] reference
# codec codes (brain's decode-only codec cannot produce these in-tree):
brain tts clone --text "..." --ref voice.wav --ref-text "..." \
  --ref-codes codes.bin --out demo.wav
```

Default voice-clone uses the **x-vector-only** path (timbre from the reference's
speaker embedding), which runs end-to-end inside brain. Supplying `--ref-codes`
(an external `[T,16]` u32 codes file: 8-byte LE count header + u32 data) switches
to the **ICL** path, which also conditions on `--ref-text` + the reference codes.

Output is always mono 24 kHz f32 PCM WAV.

---

## CPU / GPU / NPU

**See [`ACCELERATION.md`](acceleration.md)** for the full optimization write-up
(what changed, why, validated numbers) and the backend speed comparison.

- **NPU** (`--device npu`, OpenVINO) — the fast path and the default for the 1.7B
  models: a resident **KV-cache INT8 Talker** (~7–19× faster/frame than the old
  cache-free path; ~26× faster prompt prefill) plus the **stateful streaming
  codec**. `brain tts serve` keeps the compiled graphs resident across requests.
- **CPU** (`--device npu BRAIN_TTS_TALKER=cpu`) — the portable fallback: a
  KV-cache CPU Talker (AVX2/FMA) with the codec still on the NPU. Fits in host
  RAM; slower on the talker.
- **GPU** (Vulkan/wgpu, `BRAIN_DEVICE=vulkan`) — the shared `gpu-core`
  training/forward backend and the path for the smaller (0.6B) models. The 1.7B
  Talker decoder is multi-GB and is **not** uploaded to the GPU backend (the NPU
  host path exists precisely to avoid that), so Vulkan is validated for
  correctness (`make parity`) rather than used for 1.7B synthesis.

Env knobs and the parity gate are documented in [`ACCELERATION.md`](acceleration.md).

---

## Parity vs. the HF reference

The Rust components were validated tensor-for-tensor against the official
Qwen3-TTS implementation:

| Component        | Metric                          | Result        |
| ---------------- | ------------------------------- | ------------- |
| Codec decode     | max-abs error vs. HF waveform   | **3.7e-2**    |
| Codec decode     | log-mel error                   | **4.6e-3**    |
| Speaker encoder  | x-vector cosine similarity      | **1.0**       |
| Talker           | sampled codebook-0 top-1 match  | **exact**     |

### Evaluating your own syntheses

`crates/eval/src/tts.rs` provides reference-free + reference-based audio metrics
(fp32, dependency-light) for regression-testing synthesis quality:

- `mel_cepstral_distortion(pred, ref, sr)` — **MCD** (dB). Per-frame DCT of the
  log-mel into mel-cepstra, Euclidean distance over coefficients `1..=24`
  (the `c0` energy term excluded), averaged and scaled by `10/ln(10)·√2`.
  Frames are **length-clip aligned** (index-for-index up to `min(n_pred,n_ref)`)
  — exact for comparing a synthesis against its own ground truth; a DTW
  alignment would be needed for differing speaking rates and is intentionally
  omitted to keep the evaluator deterministic and dependency-free.
- `speaker_similarity(pred, ref, speaker_weights, sr)` — cosine similarity of the
  two utterances' ECAPA x-vectors (`speaker::SpeakerEncoder::embed_wav`). The
  "is it the same voice" number; needs a loaded speaker checkpoint.
- `log_mel_l1(pred, ref, sr)` — mean absolute log-mel difference (length-clipped);
  a cheap, model-free structural distance.
- `TtsMetrics { mcd, speaker_sim, log_mel_l1 }` + `tts_eval(pred, ref, sr,
  speaker_weights)` compute all three in one call (speaker term is `NaN` when no
  checkpoint is supplied).

---

## Streaming serving (`brain run`)

The event-driven HFSM controller (`crates/runtime`) understands TTS, mirroring
the text path:

- **Events** (`crates/events`): `UserSynthRequest { text, ref_audio, ref_text,
  language }` drives a request; the controller streams back `AudioChunk {
  pcm_b64, sample_rate, seq, done }` events — base64 little-endian f32 PCM,
  ordered by `seq`, with a terminal `done:true`. Both encode/decode through the
  same JSONL protocol as `user_text` / `brain_text_chunk`.
- **Runtime**: a `SynthModel` seam (parallel to `InferModel`) produces the
  waveform; an `AudioStreamPump` slices it into chunks; a `Synthesizing` HFSM
  state (reachable from `Operational` on `UserSynthRequest`) emits one
  `audio_chunk` per RTC tick and returns to `Idle` on completion. The existing
  text/detect paths are unchanged.

> **Wiring a real TTS model into `brain run`.** A `SynthModel` adapter holds the
> loaded `tts::TtsPaths` + `tts::GenOpts` and, in `synth()`, calls
> `tts::pipeline::synth` (no reference) or `tts::pipeline::clone` (with
> `ref_audio` / `ref_text`), returning the 24 kHz waveform. It is **not** wired
> into `brain run` yet, to keep the runtime crate's build/deps light (a real
> model pulls in the whole codec + speaker + talker graph). The seam +
> `FakeSynthModel` test (`crates/runtime/tests/controller.rs`) exercise the full
> `UserSynthRequest → AudioChunk… → done` flow without a real model.

---

## Current limitations

- **The `gpu-core` (Vulkan/wgpu/JIT) Talker is cache-free** — it re-runs the full
  prefix each step, so it's for the 0.6B / parity / training path. The fast
  synthesis paths (NPU and `BRAIN_TTS_TALKER=cpu`) use a KV-cache — see
  [`ACCELERATION.md`](acceleration.md).
- **MTP + codec are the remaining per-clip cost** after the Talker win — a fused
  single-infer MTP graph and further codec work are the next levers.
- **Codec windowed-mask for `T > 72`** is pending — the sliding-window codec
  transformer currently assumes the full sequence fits the window; long-form
  decode needs the windowed attention mask.
- **Training is Phase 7** — this is the inference stack. From-scratch / LoRA
  fine-tuning of the Talker, MTP, and codec is tracked separately.

---

## Pointer snippet for the top-level README

Paste this section into the repository `README.md` (kept here so this doc owns
the wording; the top-level README is edited in a separate task):

```markdown
## Text-to-Speech (Qwen3-TTS)

Pure-Rust voice synthesis: text (+ an optional reference voice) → 24 kHz audio,
via a parity-verified Qwen3-TTS stack (Talker + MTP + Mimi-style codec + ECAPA
speaker encoder). See [readme.md](readme.md) for the
architecture, the `brain tts {import,clone,synth}` commands, parity results, and
streaming serving over the `brain run` event protocol.
```
