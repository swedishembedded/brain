# qwen35moe - roadmap

Scope: Qwen3.5-35B-A3B hybrid MoE decoder (`crates/qwen35moe`).

Qwen3.5-35B-A3B: a hybrid decoder mixing Gated DeltaNet (chunked linear-attention) layers with GQA full-attention layers, a 256-expert sparse MoE on every layer, and a natively-multimodal vision tower.

## Not yet done

- [ ] The prefill/training tape's RMSNorm BACKWARD (`rms_inv`/`rmsnorm_dx`) is still the per-element kernel; only the forward selects the coalesced `rmsnorm_rows` (measured 16.3x on one decode token's norms).
- [ ] Multi-GPU INT8/INT4 residency serving (the full `PagedDecoder` engine), including layer-range sharding across GPUs for the served path - the full model does not fit a single GPU's memory.
- [ ] Prefix-cache reuse and speculative decode in the serving engine (batched decode and chunked prefill are done; see "Serving on the shared paged engine").
- [ ] Paged KV cache for served decode (int8 and fp8 KV tiers exist, one whole block per sequence).
- [ ] INT4 quantized inference (weight quantization) for this model specifically.
- [ ] LoRA adapter folding into the served model, and int8 weights on the serving path (currently only reachable via a separate, non-serving entry point).
- [ ] Incremental/KV-cache decode-time image splice for multimodal generation - only whole-sequence prefill with an image is supported today.
- [ ] Video (multi-frame) multimodal input - only single-image input is supported.
- [ ] A working NPU run: export currently reaches "compiles" only, at a fixed sequence length, text-only (no vision export, no dynamic sequence length).
- [ ] Overlapped (GPipe-style) pipeline execution - the sharded pipeline currently runs strictly sequentially, one GPU active at a time.
- [ ] An INT4/INT8 frozen base with a trainable (backward-capable) path - needs a dequantizing backward matmul kernel that doesn't exist yet.
- [ ] Register-tiled (non-naive) GEMM for weight gradients - a performance gap, not a correctness one.
- [ ] Numerical parity against the reference implementation - only structural correctness of the forward pass is currently established.

The dense (non-MoE) sibling and its MTP head are **not** in scope for this
crate - they are `crates/qwen35` (llama.cpp `LLM_ARCH_QWEN35`, distinct from
this crate's `LLM_ARCH_QWEN35MOE`), a separate architecture with its own
roadmap. See `.agents/roadmap/qwen35.md`.

A full-precision import of this model is impractical at its parameter count and does not fit alongside everything else on typical development hardware, so quantized (INT8/INT4) device buffers must be constructed directly from the compressed checkpoint format rather than via an intermediate full-precision file.

## Serving on the shared paged engine

Goal (serving-runtime items 3, G2-G4): the 35B-A3B is served by the machinery
that serves the 27B, not by a second fast stack. Written 2026-10-29 after a
source audit and a baseline on one GH200; numbers are the released Q8_0 GGUF
with int8 weights and fp32 KV on gpu1 (`brain perf run longctx --target
qwen35moe:<gguf> --context 8192 --ladder 1,2,4,8,16 --prefill 1024,4096
--steps 8`, `brain perf run sweep --target qwen35moe:<gguf> --workload chat
--ladder 1,4,16`).

### Where the engine stands today (corrects the older gap text)

`crates/qwen35moe/src/serve.rs` is not the correctness-first loop its module
header still says. It already has batched decode (one `[bsz, d]` set of
dispatches, a grouped int8 MoE sublayer, a recorded decode tape per batch
shape), chunked prefill (256-row rounds), a device head, f32/bf16/int8/fp8 KV
tiers and int8 weights read from the GGUF through `gguf_load::source`. What it
lacks is the shared runtime, not speed per token:

| baseline | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| longctx 8K decode, total tok/s | 183.4 | 292.9 | 432.6 | 607.6 | 986.4 |
| ms per step | 5.45 | 6.83 | 9.25 | 13.17 | 16.22 |

Prefill 2986 tok/s (1K prompt), 2790 tok/s (4K). Chat sweep (request level,
no prefix cache, cold prefill of every prompt): concurrency 1/4/16 gives
155.6/273.6/263.2 output tok/s, inter-token p99 5.1/359/803 ms, TTFA p99
435/415/7256 ms. The plateau between 4 and 16 clients and the inter-token p99
are prefill slices stalling decode and recomputed shared prefixes, the two
things the 27B's engine fixed (serving-runtime items 8 and 9).

The gaps that remain (serving-runtime G2, G4): one `max_seq_len` block per
sequence (the 8K target reserves 335 MB of f32 KV per resident sequence: 10
GQA layers x 2 x 512 x 4 B x 8192), the recurrent-state row is
`table.blocks()[0]` so sequence identity IS the first KV page, no prefix cache
(`reclaim_prefix` returns 0), no speculation, no adapter bank, no layer
sharding, a decode window of 1, and no `brain serve` path for the GGUF at all
(`catalog::resident_qwen35moe` takes brain-native safetensors or an HF
directory; a GGUF is only reachable through `perf` and the `qwen35moe` CLI).

### The two architectures, layer by layer

| | Qwen3.8-27B (`qwen35`) | Qwen3.6-35B-A3B (`qwen35moe`) |
|---|---|---|
| layers | 64 = 48 Gated-DeltaNet + 16 GQA | 40 = 30 GDN + 10 GQA |
| layer pattern | `full_attention_interval` 4 | the same |
| d_model | 5120 | 2048 |
| GQA | 24 heads, 4 KV heads, head_dim 256 | 16 heads, 2 KV heads, head_dim 256 |
| GDN | 16 K heads, 48 V heads, dims 128/128, conv 4 | 16 K heads, 32 V heads, dims 128/128, conv 4 |
| mixer weights, rope, norms | identical structure | identical structure |
| FFN | dense SwiGLU, intermediate 17408 | router `[256, d]` + 256 experts (gate/up `[512, d]`, down `[d, 512]`), top-8, plus one shared expert (ff 512) scaled by `sigmoid(shared_gate . x)` |
| MTP | one extra layer + `fc_e`/`fc_h`, speculation | none in the GGUF (nextn block dropped) |
| vision | 27-block tower, no DeepStack | tower with DeepStack, separate mmproj; text serving needs neither |
| GGUF experts | n/a | stacked `blk.N.ffn_{gate,up,down}_exps` `[E, n, k]`, already a fused bank layout |
| per-token GQA KV (f32) | 16 x 2 x 1024 x 4 B = 128 KiB | 10 x 2 x 512 x 4 B = 40 KiB |
| per-sequence GDN state | 48 x (48x128x128 f32 + conv hist) ~ 150 MB | 30 x (32x128x128 f32 + conv hist) ~ 63 MB |

Everything outside the FFN has the same shape family and differs only in
dimensions. The mixers, caches, paging, snapshots and scheduling do not know
which FFN a layer has.

### What in the 27B resident and engine is generic, and what is not

`Qwen35GgufEngine` (`int8_gguf_resident/engine.rs`, `impl PagedDecoder`) reaches
the model through `Qwen35GgufInstance` and its stage models
(`qwen35::model::Qwen35`, one per card). Calls that touch the layer stack:
`stage_models` (16 sites), `kv_layout`, `set_blocks`, `pump_prefill`,
`prefill_hidden{,_multi}`, `store_prefilled`, `snapshot_prefill_state`,
`seed_state_from_snapshot`, head argmax/top-k, `copy_block_rows`.

- Generic: block tables and the block allocator, the snapshot prefix cache and
  its branch points, background prefill, the prefill policy, CUDA-graph decode
  recording, the device-to-device stage hand-off, the layer-sharded placement
  and its cost model (`layer_cost`, `serving_cost`: they count GQA and GDN
  layers by type and the FFN by its weight bytes), and the scheduler contract.
- Architecture specific, and only these: (1) the FFN of a layer
  (`Qwen35::mlp_fwd_pre`, called from decode at one place and from prefill at
  one place; the training and MTP paths call it too); (2) the FFN weights and
  their names (`blocks.{l}.mlp.{gate,up,down}.weight` against router, expert
  banks and shared expert), their GGUF mapping (`gguf_import::classify` against
  `qwen35moe::import::classify`) and their byte cost; (3) the MTP drafter and
  the adapter sites on `gate`/`up`/`down`, which a MoE layer does not have
  (an adapter bank over the mixer sites only is already a configuration of the
  bank); (4) the config constructor.

The decode, prefill and verify code in `qwen35::model::Qwen35` is a second
7600-line implementation of the layer stack next to `qwen35moe::model::Qwen35`'s
5000. They agree on everything but the FFN. The shared engine needs one stack,
and the one with the paged KV, pooled caches, multi-sequence prefill, verify
rows and stage hand-off is `qwen35`'s.

### Decision: an FFN kind in the layer stack, not a second engine and not a trait over whole models

A trait over "a model the engine drives" would make the engine generic but leave
two layer stacks, which is the duplication. The smallest seam that gives one
implementation is the one the dense code already has: the FFN call.

1. `qwen35::config::Qwen35Config` gets `moe: Option<MoeFfn>` (`n_experts`,
   `top_k`, `moe_intermediate_size`, `shared_expert_intermediate_size`), the
   shape other optional blocks (`mtp`, `swa_window`, `anchor_memory`) already
   take. `None` is today's dense model, bit-unchanged.
2. `Qwen35::ffn_fwd_pre(l, xn2, pre, epilogue, n)` replaces the direct
   `mlp_fwd_pre` calls in the decode and prefill layer loops. It selects
   `mlp_fwd_pre` or the MoE sublayer by `cfg.moe`. It takes and returns what
   `mlp_fwd_pre` does (the optionally pre-quantised activation, the output
   buffer), so the fused norm-and-quantise front and the residual-add epilogue
   stay as they are.
3. The MoE sublayer is `model::moe_i8`, the shared op that already exists
   (router top-k on the device, `moe_i8_gemv_gather` at decode,
   `moe_route_{count,scan,emit}` + `moe_i8_grouped` at prefill, native int8
   tensor cores on CUDA, `moe_swiglu_quant`, `moe_slot_combine`). Today it
   serves DeepSeek-V2 only; `qwen35moe::model::moe_sublayer_i8` is a second copy
   of it that also carries the shared expert as an extra slot. Step 1 below makes
   the shared op carry the shared expert and deletes the copy.
4. Weights: the stage loader builds an `I8Experts` per MoE layer from the GGUF's
   stacked tensors (a repack of Q8_0 blocks, no fp32 copy), the router and shared
   gate as one f32 `[E+1, d]` weight, the shared expert as bank block `E`.
   `layer_cost`/`serving_cost` charge those bytes.
5. `qwen35moe` keeps training, the HF/NVFP4 import and the fp32 reference. Its
   serving `Engine`, `serve.rs` and `catalog::resident_qwen35moe` retire when
   the shared engine reproduces its greedy tokens and meets its numbers; the
   catalog serves the 35B GGUF through `resident_qwen35`.

Not decided here, measured later: whether the NVFP4 HF checkpoint (an `Nv`
expert tier with float activations) moves onto the same seam or stays on the old
engine until `moe_i8` has an NVFP4 sibling.

### Kernels the MoE FFN needs on CUDA

Already present and bit-gated against the WGSL reference: `moe_router_topk`
(CUDA), `moe_i8_grouped_mma` (int8 `mma.sync.m16n8k32`, group-32 weight scales
times group-32 activation scales, identical result to the WGSL kernel),
`moe_i8_gemv_gather`, route count/scan/emit, `moe_swiglu_quant`,
`moe_slot_combine`. Missing for the shared engine to beat the old one:

- Tuner coverage. The int8 tuner picks only `matmul_i8_*` classes; the MoE
  GEMV and grouped kernels are fixed choices (`GROUPED_MIN_ROWS` = 7 swept once
  on a decode step). They are candidates for the same per-device, per-shape
  selection (kernels.md, section F) once the shared engine can measure them.
- Prefill slots beyond one 8-slot tile per expert re-read the weight row per
  tile; a wgmma-class grouped kernel is the prefill roof lever (the dense int8
  prefill sits at 15-23 % of the tensor-core roof, knowledge 372, so this is
  not first).
- Decode rows past 32: the shared-expert and router matmuls go through the
  dense `ops.matmul` selection already; the routed experts at 33-128 rows are
  grouped GEMMs whose tile occupancy is data dependent. Measure before adding.

### Incremental plan, each step lands green

| # | step | gate |
|---|---|---|
| 1 | `model::moe_i8` carries the shared-expert slot (`SlotShape.shared`), `qwen35moe` calls it and drops its copy | `model/tests/moe_i8_slots.rs` extended with a shared slot against the host reference; qwen35moe's moe sublayer and decode tests unchanged; the 35B greedy tokens before and after identical |
| 2 | `ffn_fwd_pre` is the one FFN call of the `qwen35` layer loops, dense bit-unchanged | `qwen35/tests/gguf_conformance.rs` on gpu1 and the CPU lib suite, unchanged |
| 3 | `Qwen35Config::moe`, MoE weights in the stage loader, GGUF mapping, byte cost; a tiny random-weight MoE GGUF fixture served by `Qwen35GgufEngine` | `model::serve::conformance` suite on the fixture (the same suite `qwen35moe/tests/serve_conformance.rs` runs against the old engine), parity with `qwen35moe::serve::Engine` on the same weights |
| 4 | the real 35B GGUF through the shared engine: paged KV (256-row blocks), state row independent of KV pages, snapshot prefix cache, background prefill | greedy tokens equal the old engine's on 8 prompts; `longctx` and chat `sweep` rerun against the baseline above |
| 5 | catalog serves the GGUF via the shared resident, `qwen35moe` serving engine and `resident_qwen35moe` deleted | `brain serve` end to end on the GGUF; serving-runtime G2/G4 closed |
| 6 | speculation and adapters on the MoE (adapter bank over mixer sites), layer sharding, tuner coverage of the MoE kernels | per item, in serving-runtime |

Step 4 changes what is measured, so its acceptance is a number: the chat sweep
at concurrency 16 must beat 263 output tok/s and the 4-client inter-token p99
must fall under the 359 ms above, with longctx single-stream decode within 5 %
of 183 tok/s.
