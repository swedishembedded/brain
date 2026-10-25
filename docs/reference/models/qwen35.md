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
checkpoint from its GGUF" below. That path reads
`unsloth/Qwen3.8-27B-GGUF`'s `Q8_0` file (or any other quant) directly and
never writes an fp32 intermediate.

## Running it

```bash
brain qwen35 infer --weights qwen35.safetensors --tokenizer tokenizer.json --prompt "..."
```

Serving (HTTP/D-Bus, paged continuous batching): drop the checkpoint (and its
sibling `tokenizer.json`) under the models directory and `brain serve` finds
it with no env var at all - resolved through `qwen35::spec::Qwen35Spec` (a
`ModelCard.family == "qwen35"` `.safetensors`, or a `general.architecture ==
"qwen35"` GGUF, paired with a co-located, vocab-compatible tokenizer; the
server's own brain-checkpoint resident takes the `.safetensors` side only, through
`Qwen35Fp32Spec`, because each released GGUF is a model of its own; it holds the
weights as the file stores them, f32, bf16 or packed int8, and a demoted
instance keeps only the mapped file). A
`--weights`/`--tokenizer` flag on `brain qwen35 infer` still names either
file outright, and prints every real candidate (with the flag to pick it) if
the models directory holds more than one.

```bash
brain serve
```

## Serving the real checkpoint from its GGUF (multi-GPU, INT8)

Model id: the file's store id - `unsloth/Qwen3.8-27B-GGUF` for
`<models-dir>/unsloth/Qwen3.8-27B-GGUF/Q8_0.gguf`, `<vendor>/<repo>-<QUANT>`
when one repo directory holds several quants
(`crates/qwen35/src/int8_gguf_resident.rs`). A SEPARATE resident from
`brain/qwen35` above, not a mode of it - the two coexist and are registered
independently.

> **Status: correct.** The model plans onto however many cards it needs at
> INT8, or one card at Q4, and is bit-stable under greedy sampling; a factual
> greedy continuation of `"The capital city of France is"` produces `" Paris.
> Paris is the largest city in"`. Prefill is chunked at both tiers. M21 left
> this RED - the GGUF conversion stores every GDN leaf indexed by value head
> in a different head-order convention than brain (and the reference HF model)
> expect; M23 found and fixed it (`crates/qwen35/src/int8_gguf_resident.rs`'s
> `GdnHeadOrder`).
>
> Decode speed depends on the backend and the GPU, and the numbers recorded
> while this resident was brought up came from pre-tensor-core cards, so they
> say nothing about a current one. On an NVIDIA GPU, pass `--backend cuda`
> (see `docs/guides/hardware/cuda.md`); the default wgpu path leaves its tensor cores
> idle. Measure on the machine you serve from.

```bash
# Every qwen35 GGUF under the models directory is served as its own model, with
# no per-model flag: the scan lists it, the scheduler loads it on first use,
# and it is placed on the GPUs --device allows. A file that could never load
# (no GPU allowed, weights that do not fit, no embedded tokenizer) is not
# listed; `brain serve` logs why.
brain --device gpu0,gpu1 serve --openai

# Server-wide defaults for all of them (all optional):
#   --ctx 24576       fix the context. Default: the longest the cards
#                            hold once the weights are placed
#   --weight-format q4         per-leaf quantization tier: "i8", "q4", or
#                            "q4,in_proj_a.weight=f32,in_proj_b.weight=f32"
#                            to pin specific tensors at fp32; a malformed
#                            grammar is refused rather than served at a
#                            silently wrong precision. Default: each tensor as
#                            the checkpoint stores it - Q4_K and Q5_K stay
#                            K-quants, every other format is held at int8
#   --kv-cache int8         how the 16 GQA layers store K/V: "f32", "bf16"
#                            (half the bytes), "int8" (one scale per token and
#                            kv head, ~3.9x fewer bytes) or "fp8" (E4M3, the
#                            same bytes and scales as int8). Default: chosen
#                            from the hardware (see "Defaults" below)
#   --max-batch N     concurrent decode slots (default 8)
#   --exact-decode    make a decode row's tokens and per-token logits
#                            independent of which sequences share its step
#                            (default off; costs one extra weight pass per
#                            group and turns speculation off)
#   --ctx N      context for the resident served from a brain
#                            checkpoint (default 4096)
#   --max-batch N
#                            decode slots for that resident (default 4)
#   --speculative MODE       speculative decoding: auto (the checkpoint's own
#                            MTP head when it has one, otherwise off), off,
#                            mtp, ngram (prompt lookup, no second model), dflash
#                            (refused: no DFlash2 draft checkpoint is configured).
#                            Default: chosen from the hardware
```

**Weights as stored.** Without `--weight-format` every linear is held in the
encoding the checkpoint stores it in (`int8_gguf_resident::held_tiers`, over
`gguf::upload`): a Q4_K tensor stays a K-quant (about 4.6 bits a weight, served
by the native `matmul_kq4_gemv` at decode), a Q8_0 tensor repacks to int8
exactly, and a Q6_K or float tensor is held at int8. A Q4_K_M release mixes
Q4_K and Q6_K by layer and is held per tensor, about 16 GiB of layers where
int8 would need 25.5. The Gated-DeltaNet input projections move their blocks
whole into brain's value-head order; `out_proj`'s value heads are narrower than
a Q4_K super-block, so a Q4_K `out_proj` is held at int8.

**Defaults.** A setting left unset is chosen from the detected hardware by the
rules in `catalog::resident_qwen35` (`KV_DEFAULTS`, `SPECULATION_DEFAULTS`,
`PREFIX_SLOT_DEFAULTS`, over `gpu_core::defaults`): each rule ranks its
candidates by expected throughput, and the first one whose memory plan fits is
kept - the KV tier first (int8, then bf16, then f32), then speculation (`auto`
where every card runs the CUDA verify kernels, else `off`), then the prompt
cache (one per concurrent sequence, at least 4; then 4, 2, 1, 0). "Fits" means
the plan places at a fixed `--ctx`, or, with the context left to the cards,
that at least 32768 tokens (or the trained window, when shorter) remain. The
context is then the longest the cards hold beside all of it. `brain serve`
logs each choice and why (`brain: <id>: default ...`). Adding a default is
adding a rule or a candidate where its throughput ranks it.

What it does differently:

- **Speculative decoding** (`--speculative`, `model::serve::speculate`). The
  generic scheduler asks a drafter for up to seven proposals per sequence,
  scores the input and the proposals of every sequence in ONE batched step
  (`Qwen35GgufEngine::forward_batched_verify`), keeps the longest prefix the
  model agrees with and rolls the rest back: the paged KV by truncating the
  block table, the 48 Gated DeltaNet layers' recurrent state from a snapshot
  the `gdn_verify_pool` kernel takes after every row but the last
  (`gdn_commit_pool` puts the chosen one back). The drafter is the checkpoint's
  own multi-token-prediction head (`int8_gguf_resident/mtp.rs`: one
  full-attention layer, `eh_proj`, the shared head; about 2 ms per proposal) or <!-- perf-number: measured result; reproduced by the named model binary/test (`brain perf run` for timings) -->
  prompt lookup. The window adapts per sequence from the proposals recently
  accepted, and a step with more sequences than the measured crossover
  (`--speculative` off the fast path) is the plain batched step. Greedy output
  is the plain output wherever the batch composition does not change the
  arithmetic (a request decoded alone is token-for-token identical on 32 real
  prompts); the engine's attention splits its key range by batch size, so
  plain decode of the same request already differs between batch sizes on many
  prompts, and speculative decode differs from plain at that rate and no more.
  Sampled requests use rejection sampling, which leaves the distribution
  unchanged. The recurrent-state scratch costs 150 MB per verify row
  (`SpecServing::scratch_rows`, 32 rows by default), charged beside each
  recurrent layer on its card, and the MTP head beside the head; neither is
  charged where it will not be allocated (`SpecServing::cost`). Measurements: `.agents/knowledge/` entry on speculative decoding
  and `qwen35_spec_sweep`.

- **A compact KV cache** (`model::kv_tier`, `--kv-cache`). At 128k
  context the 16 GQA layers' KV cache is the whole bound on the batch - 16 GiB
  per sequence in f32 against 27 GiB of INT8 weights - so it can be stored
  narrower: `bf16` (8 GiB per sequence at 128k) or `int8` with one f32 scale
  per (token, kv-head) row (4.06 GiB), or `fp8` (E4M3 elements under the same
  per-row scale, so the same bytes as int8; the tensor-core prefill decodes
  the E4M3 bytes to fp16 while it stages each tile). The attention arithmetic stays f32; each
  element is rounded once when it is written and widened as it is read, by the
  paged-KV kernel family (`paged_kv_append_batched_word`/
  `paged_kv_append_i8_clipped_batched`, the `paged_decode_*` scores/apply and the
  fused head_dim-256 prefill, whose bf16 and int8 loads are template variants of
  the f32 source), on every backend. The planner charges the real bytes, so one
  97 GiB GH200 admits 4 / 8 / 15 sequences of 128k at f32 / bf16 / int8. Decode
  attention at head_dim 256 is the fused split-key kernel
  (`paged_flash_decode_gqa_hd256`, a GQA group of 6 or 8) in every tier - one
  layer at 128k is 0.83 / 0.54 / 0.42 ms for one sequence against 63 ms for the <!-- perf-number: measured result; reproduced by the named model binary/test (`brain perf run` for timings) -->
  scores/softmax/apply triad it replaced (`.agents/roadmap/qwen35.md` M30). The
  recurrent state of the 48 GDN layers (150 MiB per sequence) is not part of the
  cache and stays f32. Default stays `f32`. Gates: per-kernel rounding bounds in
  `crates/model/tests/kv_tier.rs`, logits agreement through chunked prefill and
  batched decode in `crates/qwen35/tests/kv_tier.rs`, and top-1 agreement over
  20 real prompts x 64 steps in `crates/qwen35/tests/gguf_kv_tier_real.rs`. `brain perf run
  longctx` reports the tier as `kv_precision`.

- **No fp32 intermediate anywhere.** `checkpoint::gguf::MmapGguf` is a
  `TensorSource`, so `Qwen35::new_i8_shard` streams each leaf out of the
  mapping and re-quantizes it to brain's group-wise INT8 on the way to the
  card. Peak host use is one tensor. (The offline `brain import` route would
  need ~108 GB of disk for the fp32 conversion.)
- **Layer-sharded across as many cards as it needs**, by
  `model::shard::plan_fewest_devices` over real per-layer byte costs. On a
  box with two 24 GiB cards and the default 2 GiB/card reserve it plans
  **27.05 GiB total: layers 0..34 on gpu0 (13.67 GiB), 34..64 on gpu1 (13.38
  GiB)**. A card count that cannot hold the layers is reported infeasible
  rather than attempted; a card with room for all of them takes them all.
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
- **`ssm_a` and the value-head order are undone on read.** llama.cpp's
  converter stores `-exp(A_log)` (brain's `gdn_decay_gate.wgsl` wants
  `A_log`) and stores every value-head-indexed Gated-DeltaNet leaf group-major
  (brain: sub-major). Both the offline importer (`Mapped::Transformed`) and
  this resident (`gguf::GdnFixSource`) apply `gguf::import::ElemOp::GdnVHeads`
  (`gguf::gdn_order`). Importing verbatim makes the decay gate up to 260x too
  strong and points every head at another head's state - it was found by the
  real end-to-end gate, not by any structural check. See `.agents/knowledge/`
  #70.
- **Continuous batching.** The resident is a `model::serve::PagedDecoder`
  (`Qwen35GgufEngine`, one KV slot per sequence) driven by the generic
  `model::serve::Scheduler` through `llmserve`, like every brain LLM:
  `--max-batch N` (default 8) requests decode in ONE batched step per
  iteration - the weight reads are paid once for all of them - and requests
  join and leave between steps, including ones that arrive while the group is
  running (`Instance::run_open`). Each sequence has its own sampling
  parameters, seed, stop strings and cancellation. The recurrent state of the
  48 GDN layers is a per-layer pool addressed by slot (the native
  `gdn_decode_pool` launch), prefill of a joining sequence works on a private
  copy and is scattered into its row. `--ctx` and `--kv-cache` size the
  slots; a configuration whose slots do not fit the cards is refused with the
  arithmetic and the flags to change. Measured numbers: `.agents/knowledge/`
  #313 and `make perf/model/qwen35-batch`. One KV block per sequence (block =
  context length) wastes the unused tail of a short sequence's window;
  sequences are not paged below context granularity.
- **Prefix reuse.** An agent resends its whole history every turn. The
  resident keeps the prefix-slot setting (default chosen from the hardware, see
  "Defaults"; 0 = off) finished
  prompts: the sequence's KV rows (decoding never rewrites the prompt's rows, so
  the block is simply retained) plus a snapshot of every Gated-DeltaNet layer's
  recurrent state and conv history taken the moment the prefill ended, which a
  recurrent layer needs because it keeps a state, not rows. A prompt that starts
  with a cached one copies the rows in, restores the snapshot and prefills only
  its new tokens (an identical prompt needs no compute); the longer prompt then
  supersedes the shorter entry. Exact-token match, LRU. Each cached prompt costs
  one sequence slot, charged into the placement. Gate: `crates/qwen35/tests/
  gguf_resident_real.rs` (real checkpoint: same continuation as a cold prefill).
- **Prefill between decode steps.** A prompt goes in the prefill-slice setting
  tokens (default 512) per scheduler iteration, with a decode step of every
  running sequence in between. the prefill-batch setting (default 0, off)
  hands the engine the next slices of several prefilling sequences as one
  forward of up to N rows: every projection and the MLP run once over all the
  rows, each sequence's rows go through its own token mixer (its state row
  gathered and scattered around the chunk kernels, its KV window). The
  attention of a prefill round runs on the fp16 tensor-core kernel for an f32,
  bf16 or int8 KV cache alike (`--kv-cache`); the int8 cache used to fall
  back to the portable kernel, which was most of the round. Measured numbers,
  and what bounds the rest of the round: `.agents/knowledge/` #319.
- **Text only, per-token prefill when chunking is not profitable** - same
  shape as `crate::serve::Engine` and for the same reasons.

Gated by `crates/qwen35/tests/gguf_resident_real.rs` (real file + real cards,
self-skipping loudly without either).

## LoRA training

Three LoRA paths exist, at very different scales:

- **Frozen bf16 / NF4 base (the 27B on one GPU)** - `brain qwen35 finetune
  --base-format bf16|nf4|fp8` streams the FP8 checkpoint directory tensor by tensor
  (`crate::fp8_source`, bit-identical to the eager importers, about 1 GiB of host
  memory for the whole 27B) into a frozen device base; the flags and tier limits
  are in [`docs/guides/training/lora.md`](../../guides/training/lora.md#qwen38-27b-brain-qwen35-finetune),
  and `brain qwen35 sample-adapter` decodes the trained adapter on the same frozen base.
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
  the file's store id, behind HTTP/D-Bus/`brain serve`) - the path that
  actually serves the REAL checkpoint on real hardware: INT8 weights read
  straight from the released GGUF, layer-sharded across as many cards as
  its real per-layer bytes need, host-staged residual between cards. The
  running sequences decode in one batched step and prompts prefill in chunked
  rounds. On CUDA devices a prompt is read in on each card's normal stream
  while the decode steps run on a high-priority one, so the sequences already
  generating keep generating while a new prompt is prefilled. Text only (no
  vision). See "Serving the real checkpoint from its GGUF" above.
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
