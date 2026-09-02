# Qwen3.8-27B (dense hybrid GDN/GQA decoder + MTP + vision)

A 64-layer dense hybrid decoder - the dense sibling of
[`qwen35moe`](qwen35moe.md): 3:1 Gated DeltaNet (chunked linear-attention) to
GQA layers, a sigmoid attention-output gate, partial RoPE + M-RoPE on the GQA
layers, but a plain dense SwiGLU MLP on every layer instead of a sparse MoE.
Adds a single-layer multi-token-prediction (MTP) head sharing the token
embedding and LM head, and a spliced Qwen3-VL-style vision tower (ViT +
PatchMerger, reused unchanged) for image input. `reasoning_effort` and the
chat-template flavor (`qwen3.8` default, `qwen3` opt-out) are chat-template
concepts, not architectural ones - both are wired through the shared chat
path (see the note at the end of this page).

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

A **GGUF** of the same model IS servable, though - see "Serving the real
checkpoint from its Q8_0 GGUF" below. That path reads
`unsloth/Qwen3.8-27B-GGUF`'s `Q8_0` file directly and never writes an fp32
intermediate.

## Running it

```bash
brain qwen35 infer --weights qwen35.safetensors --tokenizer tokenizer.json --prompt "..."
```

Serving (HTTP/D-Bus, paged continuous batching):

```bash
BRAIN_QWEN35_WEIGHTS=qwen35.safetensors BRAIN_QWEN35_TOKENIZER=tokenizer.json brain serve
```

## Serving the real checkpoint from its Q8_0 GGUF (multi-GPU, INT8)

Model id: `unsloth/Qwen3.8-27B-Q8_0`
(`crates/qwen35/src/int8_gguf_resident.rs`). A SEPARATE resident from
`brain/qwen35` above, not a mode of it - the two coexist and are registered
independently.

> **Status: correct.** The model plans across two 24 GiB P40s, loads with no
> fp32 intermediate, decodes at 7.44-7.57 tok/s (M22) and is bit-stable
> under greedy sampling; a factual greedy continuation of `"The capital city
> of France is"` produces `" Paris. Paris is the largest city in"`. M21 left
> this RED - the GGUF conversion stores every GDN leaf indexed by value head
> in a different head-order convention than brain (and the reference HF
> model) expect; M23 found and fixed it
> (`crates/qwen35/src/int8_gguf_resident.rs`'s `GdnHeadOrder`). See
> `.agents/roadmap/qwen35.md` (M23) for the full investigation and what it
> ruled out on the way there.

```bash
BRAIN_QWEN35_GGUF=/path/to/Qwen3.8-27B-Q8_0.gguf brain serve
# or drop the file at <models-dir>/unsloth/Qwen3.8-27B/Q8_0.gguf and it is
# discovered with no env var at all.
BRAIN_QWEN35_GGUF_CTX=2048   # per-sequence prompt+max_new cap (default 2048)
```

What it does differently:

- **No fp32 intermediate anywhere.** `checkpoint::gguf::MmapGguf` is a
  `TensorSource`, so `Qwen35::new_i8_shard` streams each leaf out of the
  mapping and re-quantizes it to brain's group-wise INT8 on the way to the
  card. Peak host use is one tensor. (The offline `brain import` route would
  need ~108 GB of disk for the fp32 conversion.)
- **Layer-sharded across as many cards as it needs**, by
  `model::shard::plan_fewest_devices` over real per-layer byte costs. On a
  box with two 24 GiB Tesla P40s and the default 2 GiB/card reserve it plans
  **27.05 GiB total: layers 0..34 on gpu0 (13.67 GiB), 34..64 on gpu1 (13.38
  GiB)**. One card is correctly reported infeasible rather than attempted.
- **The endpoints are not in a shard.** Both `[248320, 5120]` tables are 5.09
  GB as fp32, which is over a P40's `max_buffer_size` AND 2.4x its 2047 MiB
  storage-binding limit - not a tight fit, an impossible one. The embedding
  is read one row at a time from the mapping
  (`MmapGguf::tensor_range`, ~20 KiB peak, never uploaded); the `lm_head` is
  INT8 (1.42 GB) and projected by `crate::stream::head_logits_on`, the same
  head epilogue the streaming path uses.
- **No MTP.** `cfg.mtp` is forced `false` - `Qwen35::new_impl_on` requires MTP
  to sit on a whole shard, which a multi-card split is not. The GGUF's
  `blk.64.*` block is excluded by `gguf_import::classify` itself.
- **`ssm_a` is un-transformed on read.** llama.cpp's converter stores
  `-exp(A_log)`; brain's `gdn_decay_gate.wgsl` wants `A_log`. Both the
  offline importer (`Mapped::Transformed`) and this resident
  (`SsmALogFix`) apply `gguf::import::ElemOp::LnNeg`. Importing verbatim
  makes the Gated-DeltaNet decay gate up to 260x too strong and the model
  stops integrating context - it was found by the real end-to-end gate, not
  by any structural check. See `.agents/rules/lessons.md` #70.
- **Text only, one sequence per dispatch, per-token prefill** - same shape as
  `crate::serve::Engine` and for the same reasons.

Gated by `crates/qwen35/tests/gguf_resident_real.rs` (real file + real cards,
self-skipping loudly without either).

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
  `Qwen/Qwen3.8-27B-FP8` on a machine with no discrete GPU and far less RAM
  than the checkpoint's own footprint. It is exposed today ONLY via a
  standalone binary (`stream_train_step`) - `generate`'s `streaming=true`
  path has no adapter-loading wired in yet, so a trained streaming adapter
  cannot currently be used through the normal `generate` action; only the
  binary's own built-in before/after check exercises it.

### The "artificial GPU memory limit"

There is no literal "set a ceiling in GB" flag - the actual knob is
`--window-budget N`, the number of decoder layers' worth of weights kept
resident on the device at once (everything else stays on disk, streamed in
via `crates/weightset` as needed). Memory cost is roughly `N x per-layer
size`: ~419-431 MB/layer at int8 (the plain inference path), or ~4x that at
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
exceeds a Vulkan/wgpu adapter's `max_buffer_size` (2047 MiB on any
non-NVIDIA Linux adapter - see below - confirmed by an actual `wgpu`
validation error, not a host-RAM shortage). `stream_train_step` therefore
builds its `Gpu` via
`Gpu::new_cpu(...)` rather than the default GPU adapter.

Splitting the resident `lm_head` into several sub-`max_buffer_size` device
buffers (so training could stay on the GPU backend) was investigated and
rejected for now: `matmul.wgsl`'s output write and `matmul_dx.wgsl`'s `dY`
read both index their buffer as `row * n + col`, where `n` is the SAME
value as the dispatch's own output/reduction width - neither kernel has a
row-stride parameter distinct from the per-dispatch width. A vocabulary
row-chunk therefore cannot be written into (or read back out of) its correct
column range of one full-width `[tokens, vocab]` logits/gradient buffer
without a WGSL-level change (a distinct row-stride parameter, or a
column-scatter/gather kernel - neither exists today for this shape). That is
real new kernel work, not a call-the-same-kernel-N-times change, so it is
out of proportion for what is purely a training-throughput improvement (the
CPU backend already trains correctly end to end on the real checkpoint).
Two other real-weight paths in this engine (`crates/sam1/tests/parity.rs`,
`crates/deepseek2/tests/common/real_lm.rs`) hit the same class of
`max_buffer_size` ceiling and use the same fallback - pinning the CPU
backend for the affected pass is this engine's established answer to "one
tensor is bigger than one device buffer can be," not a one-off workaround.

The 2047 MiB figure is not a conservative driver reporting its own real
ceiling - it is `wgpu-hal`'s own policy, unconditionally applied to every
Linux/Android adapter from a non-NVIDIA vendor regardless of what the
hardware actually reports (`wgpu-hal` 29.0.4 and 30.0.0 - the latest
release as of this writing - both clamp `max_buffer_size` to `i32::MAX`
bytes in `src/vulkan/adapter.rs`'s `!is_nvidia` branch, "prevent very large
buffers on mesa and most android devices"). Raw Vulkan on an Intel Arc
(Meteor Lake) adapter, tested here, reports a real `maxBufferSize` of 4 GiB
(`vulkaninfo`), well above what `wgpu` ever advertises - so bumping the
`wgpu` dependency version cannot move this number, and `--device gpu`
fails identically on any current `wgpu` release, on any non-NVIDIA Linux
GPU.

`stream_train_step --device vulkan` (native Vulkan via `crates/backend-vulkan`,
ash + naga, bypassing `wgpu-hal` entirely) was tried as a way around the
`wgpu` clamp specifically. It is a genuine dead end, not just an unsupported
path: the 4.74 GiB `lm_head` buffer allocation SUCCEEDS (this backend does
not enforce the reported 4 GiB `maxBufferSize` either), and a forward-only
`--phase before` run completes correctly (real-weight output matching the
CPU backend's own `" Paris."`) - but the heavier backward
dispatch pattern of `--phase step` crashes the GPU outright with
`ERROR_DEVICE_LOST` (`crates/vulkan/src/context.rs`'s wait call), not a
catchable validation error. This is exactly the class of instability
`wgpu-hal`'s comment warns its clamp exists to prevent. Do not route around
the `wgpu` buffer-size wall via the native Vulkan backend for this model -
it trades a clean, immediate failure for an unrecoverable one.

### Example: a real streamed LoRA fine-tune against the real checkpoint

Each phase below is its own short-lived process invocation - a real
forward+backward step against the real checkpoint is slow on reference
hardware with no discrete GPU, and this development environment kills
background processes before one combined run would finish. The tiny LoRA
adapter state round-trips through a small safetensors file
(`--adapter-in`/`--adapter-out`) between phases, so a training run can span
several short process invocations without losing progress.

A tiny, concrete example corpus - 20 repetitions of one deliberately-novel
fact, so a hard-overfit run's effect is unambiguous:

```bash
CORPUS=[path/to/corpus.txt]
printf 'The capital of France is Leon.\n%.0s' {1..20} > "$CORPUS"
```

```bash
DIR=[path/to/qwen3.8]          # a downloaded Qwen/Qwen3.8-27B-FP8 checkpoint dir

# 1. Completion BEFORE training (adapter is zero-init, a provable no-op, so
#    this is genuinely the base model's own behaviour).
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase before \
  --adapter-out [path/to/lora.safetensors] \
  --prompt "The capital of France is" --max-new 3

# 2. One real training step (rank 4 / alpha 8 adapters on all 12 targetable
#    leaves, 16 training tokens, window-budget 2). For a second step, rerun
#    with --step 2 and --adapter-in pointing at this step's --adapter-out.
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase step --step 1 \
  --corpus "$CORPUS" --window-budget 2 --rank 4 --alpha 8 --lr 0.05 \
  --adapter-out [path/to/lora_step1.safetensors]

# 3. Completion AFTER training, loading the trained adapter.
cargo run --release -p brain-qwen35 --bin stream_train_step -- \
  --dir "$DIR" --phase after \
  --adapter-in [path/to/lora_step1.safetensors] --adapter-out [path/to/lora_final.safetensors] \
  --prompt "The capital of France is" --max-new 3
```

A real run of this shape (CPU backend, `window_budget=2`, `rank=4 alpha=8`,
`lr=0.05`, `n=16` training tokens, a different but comparably-sized real
corpus) produced a clear, decreasing loss trajectory across 2 steps - a
33x drop from step 1 to step 2 - and the AFTER completion for the SAME
prompt visibly, dramatically diverged from the BEFORE completion. That is
what this proves: the streaming forward+backward machinery genuinely works
end to end against the real checkpoint, not that 2 steps at `lr=0.05` on 16
tokens produces a good fine-tune. A few steps on a tiny repeated-fact batch
hard-overfits by design; a real fine-tune needs a real dataset and a far
more conservative step count/learning rate. Each phase above is slow on
reference hardware - a profiling/optimization pass is in progress (see
"Hardware and limits").

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
  without 2+ discrete GPUs - not available on every box). At the real 27B
  shape this path cannot load the real checkpoint on a 24 GB card at all: its
  fp32 `tok.weight`/`lm_head.weight` are each over `max_buffer_size`. Use the
  GGUF resident below for the real weights.
- **The INT8 multi-GPU GGUF resident** (`crate::int8_gguf_resident`, model id
  `unsloth/Qwen3.8-27B-Q8_0`, behind HTTP/D-Bus/`brain serve`) - the path that
  actually serves the REAL checkpoint on real hardware: INT8 weights read
  straight from the released Q8_0 GGUF, layer-sharded across as many cards as
  its real per-layer bytes need, host-staged residual between cards. Also one
  sequence per dispatch and per-token prefill; no MTP, no vision, no LoRA.
  See "Serving the real checkpoint from its Q8_0 GGUF" above.
- **The streaming path** (`crate::stream`/`crate::stream_train`, behind
  `generate --streaming true` and the standalone `stream_train_step`
  binary - see "LoRA training" above) runs the REAL checkpoint on a machine
  with no discrete GPU by holding only a small window of layers resident at
  once, with a genuine int8 (DP4A) weight tier for inference. It is real-
  weight validated end to end (real tokenizer, real embeddings, real
  sampling, an MTP-accelerated greedy decode mode, and streamed LoRA
  fine-tuning) but is NOT integrated with the resident serving engine above
  - no HTTP/D-Bus/paged-batching surface, no multi-GPU sharding, and
  currently no way to load a trained adapter back into it. It is also very
  slow on a machine with no discrete GPU (every decoder pass re-streams
  every layer's weights from disk) - a profiling/optimization pass is in
  progress to bring this down.

`reasoning_effort` (xhigh/medium/low) is wired through the shared chat path
(`qwen3::chat::parse_request`, defaulting to `xhigh` when thinking is
enabled), validated against the real Qwen3.8 chat template. The template
flavor is selectable too: requests default to this model's own Qwen3.8
template (XML `<function=...>` tool-call payloads, prefilled open `<think>`,
live `preserve_thinking`), and `template_flavor: "qwen3"` opts back into the
Qwen3-era JSON tool-call form.
