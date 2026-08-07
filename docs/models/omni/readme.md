# Qwen3-Omni-30B-A3B in brain

An omni-modal model: **text, audio and vision/video in → text and speech out**,
running on the same portable WGSL engine as the rest of brain. This is a
faithful, parity-verified re-implementation of `Qwen3OmniMoeForConditionalGeneration`
(`Qwen/Qwen3-Omni-30B-A3B-Instruct`) — the only released variant with the Talker
(speech-output) path.

35.25 B params total (70.5 GB bf16 upstream); only ~3 B are active per token
(30B-A3B = 30 B total routed weights, ~3 B active), but int8-quantized storage
still needs the full parameter count resident, ~35 GB.

---

## Architecture

Three components chained end to end (`crates/omni`):

```
  text ───────────────────────────────┐
  audio ──► AuT (audio tower) ────────┤
  image/video ──► ViT (vision tower)──┤
                                       ▼
                            ┌──────────────────────┐
                            │   Thinker (MoE LLM)   │  48 layers, 128 experts/top-8
                            └──────────┬───────────┘  text out; hidden layer 24 → Talker
                                       ▼
                            ┌──────────────────────┐
                            │  Talker (MoE, +MTP)   │  20 layers, 128 experts/top-6
                            └──────────┬───────────┘  [T,16] codec codes
                                       ▼
                            ┌──────────────────────┐
                            │      Code2Wav         │  RVQ decode → 24 kHz waveform
                            └──────────────────────┘
```

- **Audio tower (AuT)** (`crates/omni`, reusing `qwen-asr`'s Whisper/Qwen-omni-style
  encoder at Omni's scale: 32 layers, `d_model=1280`, 20 heads, `ffn=5120`,
  128 mel bins) — conv-stem + windowed transformer, `n_window_infer=800`,
  projects to the Thinker's hidden width (2048).
- **Vision tower** (`crates/qwenvl`, extended to Omni's scale: depth 27,
  hidden 1152, `gelu_pytorch_tanh`, DeepStack taps `[8,16,24]`) — ViT +
  PatchMerger + DeepStack, `spatial_merge_size=2`, temporal patching for video.
- **Thinker** — a Qwen3-MoE decoder (128 experts, top-8, no shared expert,
  `use_qk_norm`, GQA 32/4) with **3-axis interleaved M-RoPE** over text,
  audio (`position_id_per_seconds=13`), image and video positions, spliced
  with multimodal embeddings via the shared `model::vlm` seam.
- **Talker** (`crates/tts`, extended to MoE: 20 layers, 128 experts top-6,
  shared_expert 768) — consumes the Thinker's hidden state at
  `accept_hidden_layer=24` (not just token embeddings) plus a per-speaker
  embedding (`chelsie`/`ethan`/`aiden`), autoregressively sampling codebook-0
  acoustic tokens at 12.5 Hz.
- **MTP code predictor** (`crates/tts`, unchanged shape — already 5 layers /
  16 codebooks) — fills residual codebooks 1..15 from codebook-0 each frame.
- **Code2Wav** (`crates/codec`, extended: `hidden_size=1024`,
  `intermediate_size=3072`, mean-pooled multi-codebook embedding input instead
  of RVQ dequant) — 8-layer sliding-window (72) GQA pre-transformer → ConvNeXt
  upsample `[2,2]` → SEANet decoder `[8,5,4,3]` + SnakeBeta → 24 kHz waveform.
  `chunked_decode(chunk_size=300, left_context_size=25)` for streaming.

See `docs/models/omni/status.md` for the measured, chronological build ledger.

## Sparse MoE core

Both the Thinker and the Talker are top-k MoE. brain's only prior HF-importable
MoE forward (`crates/glm`) evaluated **every** expert densely — 16× wasted work
at 128 experts / top-8. This model motivated a real sparse MoE core
(`model::moe`: router → top-k → gather-by-expert → grouped GEMM → scatter-add,
fp32 and int8/DP4A), which `glm` was migrated onto in the same workstream.
See `docs/lessons.md` for the write-up.

## Serving

Reachable over all three of brain's surfaces — D-Bus (`Run`/`Subscribe` with
typed audio/image blobs), the OpenAI-compatible HTTP API (`/v1/chat/completions`
with `image_url`/`input_audio` content parts, `/v1/audio/speech`,
`/v1/audio/transcriptions`), and the Anthropic-compatible API (`/v1/messages`
with `image` content blocks) — scheduled by `residency` with per-turn lifetime
management: the Thinker stays resident across a conversation, the audio/vision
towers are built, used and dropped per turn, and the Talker + Code2Wav are only
built once the Thinker's text response is available.

`examples/omni.py` exercises every transport × modality-in × modality-out
combination locally.

## Known limitations

- NPU export (`crates/npu`) is wired for the audio tower, vision tower and
  Code2Wav, and validated as far as CPU-side OpenVINO parity — this development
  box has no NPU, so an actual device run has not happened. See status.md.
- Only `Qwen3-Omni-30B-A3B-Instruct` is supported; `-Thinking` and `-Captioner`
  have no Talker and are out of scope for now.
