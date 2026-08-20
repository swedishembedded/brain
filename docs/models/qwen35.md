# Qwen3.8-27B (dense hybrid GDN/GQA decoder + MTP + vision)

A 64-layer dense hybrid decoder - the dense sibling of
[`qwen35moe`](qwen35moe.md): 3:1 Gated DeltaNet (chunked linear-attention) to
GQA layers, a sigmoid attention-output gate, partial RoPE + M-RoPE on the GQA
layers, but a plain dense SwiGLU MLP on every layer instead of a sparse MoE.
Adds a single-layer multi-token-prediction (MTP) head sharing the token
embedding and LM head, and a spliced Qwen3-VL-style vision tower (ViT +
PatchMerger, reused unchanged) for image input. `reasoning_effort` is a
chat-template concept, not an architectural one - this port has no verified
prompt-injection convention for it yet (see below).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| LoRA fine-tune         | [x] (rank/alpha adapters on all 12 targetable GDN/GQA/MLP projections) |
| MTP head               | [x] (structural - no reference oracle available, gradchecked and overfit-tested only) |
| Vision input            | [x] (real-dims parity-tested tower; end-to-end splice is structural only - no reference oracle for the composite) |
| CLI                    | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] (paged, single-GPU, one active sequence at a time on the GPU - see "Hardware and limits") |

## Getting the weights

Model id: `brain/qwen35`, upstream `Qwen/Qwen3.8-27B-FP8`. Weights ship as
DeepSeek-V3-style blockwise FP8 (E4M3 + BF16 `weight_scale_inv`), dequantized
host-side at import. No GGUF importer for this architecture - the real
checkpoint is safetensors-only.

## Running it

```bash
brain qwen35 infer --weights qwen35.safetensors --tokenizer tokenizer.json --prompt "..."
```

Serving (HTTP/D-Bus, paged continuous batching):

```bash
BRAIN_QWEN35_WEIGHTS=qwen35.safetensors BRAIN_QWEN35_TOKENIZER=tokenizer.json brain serve
```

## LoRA training

Two LoRA paths exist, at very different scales:

- **Resident (small/merged checkpoints)** - `crate::finetune::finetune`
  builds a fully resident `Qwen35` with every weight in one host `HashMap`,
  so the checkpoint must fit in RAM as fp32 - not the real 27B checkpoint on
  a RAM-constrained box, but fine for a `Qwen35Config::tiny()`-scale fixture
  or a future smaller release.
- **Streaming (the real 27B checkpoint)** - `crates/qwen35/src/stream_train.rs`
  streams every layer's weights from disk TWICE per step (once forward, once
  in reverse for backward), so only a small window of layers is ever
  resident. This is the path that actually works against
  `Qwen/Qwen3.8-27B-FP8` on this project's own reference hardware (no
  discrete GPU, ~20 GiB usable RAM). It is exposed today ONLY via a
  standalone binary (`stream_train_step`) - `generate`'s `streaming=true`
  path has no adapter-loading wired in yet, so a trained streaming adapter
  cannot currently be used through the normal `generate` action; only the
  binary's own built-in before/after check exercises it.

### The "artificial GPU memory limit"

There is no literal "set a ceiling in GB" flag - the actual knob is
`--window-budget N`, the number of decoder layers' worth of weights kept
resident on the device at once (everything else stays on disk, streamed in
via `crates/weightset` as needed). Memory cost is roughly `N x per-layer
size`: ~373-383 MB/layer at int8 (the plain inference path), or ~4x that at
fp32 (training needs fp32 - see below), so `--window-budget 2` (the
trainer's own default) costs on the order of a few GB device-resident
regardless of the checkpoint's real ~28 GB total size. The plain streaming
inference path (`crate::stream::generate`, behind `generate --streaming
true`) hardcodes this window to 4 internally rather than exposing it as a
request parameter yet.

### Why training runs on the CPU backend, not the GPU

Backward needs the frozen base weight in fp32 (int8 backward-through-the-
weight is not wired up anywhere in this engine yet), and the resident
`lm_head` at fp32 (`vocab x d_model x 4` bytes, ~4.74 GiB as ONE buffer)
exceeds this box's Vulkan/wgpu adapter's real `max_buffer_size` (2047 MiB, a
hard driver limit - confirmed by an actual `wgpu` validation error, not a
host-RAM shortage). `stream_train_step` therefore builds its `Gpu` via
`Gpu::new_cpu(...)` rather than the default GPU adapter.

### Example: a real streamed LoRA fine-tune against the real checkpoint

Each phase is its own short-lived process invocation - a real
forward+backward step against the real checkpoint takes tens of minutes on
this reference hardware, and this development environment kills background
processes well before one combined run would finish. The tiny LoRA adapter
state round-trips through a small safetensors file
(`--adapter-in`/`--adapter-out`) between phases, so a training run can span
several short process invocations without losing progress:

```bash
DIR=/data/workspace/resources/qwen3.8
CORPUS=/data/workspace/resources/qwen35_finetune/corpus.txt  # any real text file

# 1. Completion BEFORE training (adapter is zero-init, a provable no-op, so
#    this is genuinely the base model's own behaviour).
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase before \
  --adapter-out /tmp/lora.safetensors \
  --prompt "The capital of France is" --max-new 3

# 2. One real training step (rank 4 / alpha 8 adapters on all 12 targetable
#    leaves, 16 training tokens, window-budget 2). For a second step, rerun
#    with --step 2 and --adapter-in pointing at this step's --adapter-out.
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase step --step 1 \
  --corpus "$CORPUS" --window-budget 2 --rank 4 --alpha 8 --lr 0.05 \
  --adapter-out /tmp/lora_step1.safetensors

# 3. Completion AFTER training, loading the trained adapter.
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase after \
  --adapter-in /tmp/lora_step1.safetensors --adapter-out /tmp/lora_final.safetensors \
  --prompt "The capital of France is" --max-new 3
```

**Real measured numbers from this exact command shape** (CPU backend,
prompt `"The capital of France is"`, `max_new=3`, greedy,
`window_budget=2`, `rank=4 alpha=8`, `lr=0.05`, `n=16` training tokens):

| Phase | Loss | Wall-clock |
|---|---|---|
| BEFORE | - | 17.4-18.1 min |
| step 1 | 2.417521 | 35.71 min |
| step 2 | 0.071535 (33x drop) | 35.58 min |
| AFTER | - | 17.40 min |

- **BEFORE**: `"The capital of France is"` -> `" Paris.\n"`
- **AFTER**: `"The capital of France is"` -> `"emelemelemel"`

The adapter visibly, dramatically changed the model's output - that is what
this proves (the streaming forward+backward machinery genuinely works end
to end against the real checkpoint), not that 2 steps at `lr=0.05` on 16
tokens of unrelated text produces a good fine-tune. Two steps on a 16-token
batch hard-overfits by design; a real fine-tune needs a real dataset and a
far more conservative step count/learning rate. Each phase above costs tens
of minutes on this reference hardware - a profiling/optimization pass to
bring that down is in progress (see "Hardware and limits").

## Hardware and limits

Two distinct execution paths exist for this model - keep them separate when
reading claims below, they have very different capabilities:

- **The resident paged-serving engine** (`crate::serve::Engine`, behind HTTP/
  D-Bus/`brain serve`) needs the WHOLE model resident, is single-GPU, fp32
  weights and fp32 KV cache only (no int8 weight or KV tier wired into THIS
  path), and processes one truly active sequence at a time on the GPU -
  several sequences may be resident and interleaved by the scheduler across
  iterations, but never batched together into one GPU dispatch. Prefill
  replays the prompt one token at a time rather than a fast batched forward.
  No LoRA adapter folding into this path yet. No multi-GPU sharding wired in
  either (the underlying `model::Shardable` capability exists and is gated
  by its own test, `crates/qwen35/tests/shard_parity.rs`, which self-skips
  without 2+ discrete GPUs - not available on every box).
- **The streaming path** (`crate::stream`/`crate::stream_train`, behind
  `generate --streaming true` and the standalone `stream_train_step`
  binary - see "LoRA training" above) runs the REAL checkpoint on a machine
  with no discrete GPU by holding only a small window of layers resident at
  once, with a genuine int8 (DP4A) weight tier for inference. It is real-
  weight validated end to end (real tokenizer, real embeddings, real
  sampling, an MTP-accelerated greedy decode mode, and streamed LoRA
  fine-tuning) but is NOT integrated with the resident serving engine above
  - no HTTP/D-Bus/paged-batching surface, no multi-GPU sharding, and
  currently no way to load a trained adapter back into it. It is also, on
  this project's own reference hardware, VERY slow (tens of minutes per
  decoder pass) - a profiling/optimization pass is in progress to bring
  this down; treat the current numbers in "LoRA training" above as a
  starting point, not a ceiling.

`reasoning_effort` (xhigh/medium/low) is not implemented: no verified
Qwen3.8 prompt-injection convention was found to build it against without
guessing - only the existing `enable_thinking` boolean is wired, reused from
`qwen3::chat`.
