<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# ripple - roadmap

Scope: Ripple-65B (`RicursiveForCausalLM`, `model_type: ricursive`) inference, serving and training, with latent sparse attention on its global layers.

Swedish Embedded AB implements long-context serving of dense coding models
for its clients. If your team needs expertise in porting and serving large
hybrid-attention decoders, you can procure our services by sending an email
to info@swedishembedded.com.

## Checkpoint facts (read from the published files, 2026-11-07)

Sources: `ricursiveAI/Ripple-65B-Preview` `config.json`,
`modeling_ricursive.py`, `model.safetensors.index.json`; NVFP4 release
`ricursiveAI/Ripple-65B-Preview-NVFP4` `config.json`. Local copies under
`~/resources/models/ripple-65b-preview/`.

| property | value |
|---|---|
| layers | 80; `layer_types` = 4 `sliding_attention` then 1 `full_attention`, repeated (16 global: 4, 9, ..., 79) |
| width | hidden 5120, FFN 40960, vocab 100352, untied `lm_head` |
| attention | 80 query heads, 8 KV heads (group 10), head_dim 128, SDPA scale `1/sqrt(128)`, no bias |
| local layers | RoPE (half-split, theta 10000, `rope_scaling` none), sliding window 1024 (key `j` visible to query `i` iff `i-1023 <= j <= i`) |
| global layers | full causal attention, **no RoPE** (NoPE) |
| norms | RMSNorm eps 1e-6, computed in f32, weight applied after casting back; PeriNorm: `input`, `post_attention`, `pre_feedforward`, `post_feedforward` per block; per-head `q_norm`/`k_norm` over head_dim before RoPE |
| embedding | `embed_tokens(ids) * embed_scale` (71.554 = sqrt 5120, from config; a folded checkpoint sets it null) |
| value residual | `v = v_proj(lerp(x0, xn, sigmoid(value_res_lambda)))`, `x0` = scaled embeddings (un-normed, layer-0 input), `xn` = this layer's `input_layernorm` output; every layer, including layer 0 |
| gated attention | `attn * sigmoid(float(g_proj(xn)))`, `g_proj` 5120 -> 10240 (elementwise, per channel), applied before `o_proj` |
| MLP | `down(silu(min(gate, 20)) * clamp(up, -20, 20))` |
| block | `h += post_attn_norm(attn)`; `h += post_ffn_norm(mlp(pre_ffn_norm(h)))`; final `norm`, `lm_head` |
| tensors | 1203: 15 per layer (`q,k,v,o,g_proj`, `q_norm`, `k_norm`, `value_res_lambda` [1], 4 norms, 3 MLP) + embed, norm, lm_head |
| positions | `max_position_embeddings` 2097152; advertised 1M |
| tokenizer | `PreTrainedTokenizerFast` (100k BPE), ChatML `<\|im_start\|>`, tool calls in XML tags (`chat_template.jinja`) |
| NVFP4 release | modelopt NVFP4 W4A4, group 16, static input scales, **MLP only**: every `self_attn*`, `embed_tokens` and `lm_head` stay bf16 |
| drafter | `Ripple-65B-Preview-DFlash2` (a Qwen3-architecture DFlash2 drafter) |

The global layers' Q/K are RMS-normed and un-rotated, so the LSA index reads
them as they are (`Rope::None`); nothing has to be un-rotated.

## What brain already has, and what it lacks

| need | in brain today | gap |
|---|---|---|
| hybrid 4:1 sliding/full decoder, PeriNorm, QK-norm | `crates/gemma4` (5:1 sliding/full, sandwich norms, QK-norm, windowed GQA kernel `K_GQA_SCORES_WIN`) | Gemma4 is forward-only, no paged engine; value residual, `g_proj` gate and clipped SwiGLU exist nowhere |
| HF arch registry, safetensors import, modelopt NVFP4 metadata | `crates/arch`, `crates/checkpoint` (`packed/hfquant.rs` reads `modelopt` NVFP4) | a `ricursive` row and importer; NVFP4 activation (A4) static input scales checked against what the NVFP4 GEMM consumes |
| paged serving under the generic `Scheduler` | `PagedDecoder` engines: `qwen3/serve.rs` (growing KV), `qwen35/serve_ring.rs` (fixed ring per sequence) | no engine mixes a 1024-row ring for local layers with growing paged KV for global ones; prefix reuse must share global pages and snapshot the rings |
| LSA index, selection, sparse paged read | `model::lsa` (per-layer `Option` projections, `Rope::None`, flat and tree reads, int8/fp8 KV) | wired into `qwen3/serve.rs` only: `set_lsa`, crossover, kernel set and memory charge live in the Qwen3 engine and must move to a shared place before a second engine can use them |
| sparse attention kernels for 80/8/128 | per-KV-head `paged_sparse_flash_gqa` with a `GROUP` knob compiled for 2, 4, 5, 8; whole-row kernel requires `64 % n_heads == 0` | group 10 not compiled; whole-row prefill refuses 80 heads (roadmap `lsa.md`: templated workgroup) |
| D-LSA distillation | `model::lsa::distill` (host math) and `brain lsa` | model-agnostic capture of a layer's Q/K, so a Ripple indexer can be distilled over the 16 global layers; device distillation (`lsa.md`) |
| speculative decoding | DFlash2 in `crates/qwen35` | binding the Ripple DFlash2 drafter to a Ripple target |
| training | gradcheck, LoRA, `ModelFit` | backward for value residual lerp, sigmoid gate, clipped SwiGLU (clamp masks), PeriNorm order |

KV memory (bf16, per token): global 16 x 2 x 8 x 128 x 2 B = 64 KiB, so
64 GiB at 1M tokens; local rings 64 x 2 x 8 x 128 x 2 B x 1024 = 256 MiB
per sequence regardless of length. LSA cuts the attention reads, not this
storage; int8/fp8 KV halves it.

## Plan

Correctness first, on the CPU backend (`BRAIN_DEVICE=cpu`), while the GPUs
serve other work; CUDA parity and performance follow on the same gates.

1. [x] `crates/ricursive`: config, `ARCHS` row, importer that refuses
       missing/unknown tensors, f32 forward on any backend. Gate: a
       random-weight tiny config (2 local + 1 global, window smaller than
       the prompt) matches `modeling_ricursive.py` logits; real-weight
       per-layer parity on layers 0 (local) and 4 (global) plus embed/head,
       fetched by tensor (no full 130 GB download: the project drive has
       ~60 GB free). Done on the CPU backend; CUDA parity is step 7.
2. [x] Paged hybrid engine under the generic `Scheduler`: ring KV for local
       layers, paged KV for global ones; serving conformance suite; decode
       equals full-recompute logits across the window boundary; prefix
       reuse. Done on the CPU backend as `ricursive::RicursiveEngine`; not in
       the catalog and not reachable from the CLI yet.
3. [x] LSA lifted out of the Qwen3 engine into a shared engine-side
       component (`model::lsa::binding`); Ripple's 16 global layers read
       sparsely (`RicursiveEngine::set_lsa`); whole-row and group-10 kernels
       compiled. Written and statically validated on the CPU backend, which
       executes none of the LSA kernels: the device gates (full-budget
       equivalence, prefix reuse, batching, real layers) skip there and are
       still to be run on a GPU.
4. [x] D-LSA indexer capture and distillation, generic: any engine meeting
       `model::lsa::capture::AttentionCapture` (Qwen3, Ripple) is captured,
       PCA/random-initialised, distilled and evaluated by `brain lsa`;
       Ripple's indexer covers its 16 global layers. Tooling and a tiny-scale
       gate done; the real long-context run is the GPU phase below.
5. [x] Weight tiers (bf16 as stored, fp8, int8, the official NVFP4 release as
       published), compact KV (bf16, int8, fp8) for the rings and the global
       pools, byte-exact placement over cards, the engine in pipeline stages.
       Done and measured on the CPU backend and the software Vulkan adapter;
       no card has run it (step 7).
6. [ ] Backward, gradcheck, LoRA; N-LSA after D-LSA.
7. [ ] CUDA parity and the measured long-context numbers (both GH200s,
       when free).

## Measured

Step 1, CPU backend (`BRAIN_DEVICE=cpu`), f32 forward against the official
`modeling_ricursive.py` run in fp32 on CPU (torch 2.14, transformers 5.21; the
reference imports unchanged). Error is `max|got - want| / max|want|` per tap,
over the embeddings, each layer's `o_proj` output and residual stream, the final
norm and the logits.

| gate | taps | worst scaled error | tolerance | largest absolute error |
|---|---|---|---|---|
| tiny random weights (3 layers sliding/sliding/full, window 4, 12 tokens, clamp 0.5 firing on both branches) | 9 | 8.2e-7 | 1e-5 | 1.1e-5 (`attn_out.2`, scale 34) |
| real layers 0 and 4 (width 5120, FFN 40960, 13 tokens, bf16 weights promoted to f32, real embed/norm/`lm_head`) | 7 | 6.2e-6 (logits) | 5e-5 | 1.3e-4 (`final_hidden`, scale 43); logits 8.1e-5 at scale 13 |

The tiny fixture is dumped with a random seed (recorded in its manifest), so its
worst case varies a little between dumps. The real run takes about 8 s on the
shared CPU node. Mutating the window by one key, or doubling the gate cap,
fails the tiny gate at scaled errors 0.5 and 0.12.

Step 2, serving engine, CPU backend, f32 (scaled error as above, per logits row):

| gate | rows | worst scaled error | tolerance |
|---|---|---|---|
| tiny: prefill then 14 teacher-forced decode steps against recomputing the whole prefix, five (engine chunk, prompt slice) pairs, window 4 wrapped three times, 4-row blocks | 5 x 15 | 6.7e-7 | 1e-5 |
| tiny: three concurrent sequences (6, 9 and 13 prompt tokens) decoding 8 steps in one batch, against each alone and against recompute | 24 x 2 | 1.1e-6 (batched vs alone: 0, bit-identical) | 1e-5 |
| tiny: prefix of 20 and of 18 tokens (past the window, block-aligned and not) resumed by a longer prompt and by an exact repeat; the filing sequence keeps decoding through the adoption | 2 x 6 | 1.0e-6 | 1e-5 |
| real layers 0 and 4: prefill 9 tokens in chunks of 4, decode 4, against the official reference's logits | 5 | 7.5e-6 | 5e-5 |

The shared conformance suite (`model::serve::conformance::run_all`) passes on
the tiny model with prefix reuse on. Dropping the ring slack, or skipping the
ring snapshot on a prefix hit, fails the tiny gates (the missing snapshot by
9e-2 scaled).

Not measured: CUDA or Vulkan parity (GPUs were busy), any prompt longer than 13
real tokens, a real window-exclusion case on real weights (13 tokens is below
the 1024 window; the window is gated by the tiny model only), layers other than
0 and 4, throughput of any kind.

Step 3, CPU backend. The LSA kernels are workgroup-cooperative and the CPU JIT
cannot execute them (`@cpu no`), so on this box: the refusals (an indexer that
covers a sliding layer, other geometry, rotated heads, a tree, an engine opened
without the kernels, and the backend's own refusal by name) pass; every variant
of the per-KV-head kernel (groups 1 to 10) and of the whole-row kernel (16, 32,
40, 64, 80 heads and two others) parses and validates under naga within 16 KiB of
workgroup memory; Qwen3's serve conformance and LSA tests build and pass
by skipping their device parts. Not run, written and waiting for a GPU:
`ricursive` tests `a_budget_past_the_context_equals_dense_through_prefill_chunks_and_decode`,
`a_small_budget_reads_fewer_keys_than_dense_attention`,
`prefix_reuse_and_batching_match_cold_and_alone_runs_with_sparse_attention`,
`sparse_attention_at_a_full_budget_decodes_real_layers_0_and_4_as_the_reference`,
and the `brain-model` `paged_sparse` device tests for the new geometries. The
engine-level sparse-equals-oracle gate is the kernel's (per layer, against
`lsa::attend`), not a Ripple logits comparison.

Software Vulkan adapter (lavapipe, `BRAIN_DEVICE=software`, 2026-11-07): the
`lsa` gates and the tiny-model `parity` and `serve` gates of `brain-ricursive`
run on the wgpu path and agree with the CPU backend (11 of 14 tests pass; LSA
full-budget worst scaled error 6.6e-7 against tolerance 1e-5, prefill and
decode, chunked; warm vs cold prefill 0 to 5.2e-7). The three real-weight tests
(`real_layers_0_and_4_match_the_reference`, the serving-engine one and the
sparse full-budget one) do not run there: the real checkpoint binds a 200 MiB
weight matrix and lavapipe caps a storage binding at 128 MiB, so
`create_bind_group` rejects it. They stay CPU-JIT or card gates; the sparse one
needs a card (it skips on the JIT).

Step 4, D-LSA tooling. `AttentionCapture` (geometry with the index-time
rope, `sparse_capable`, a chunked dense capture of keys and the queries at
requested positions with LSA set aside, `position_logprobs`, `set_lsa`) is
implemented by `qwen3::serve::Engine` and `RicursiveEngine`; random, PCA and
distilled indexers get `None` for non-capable layers. CPU backend unless noted:

| gate | result |
|---|---|
| capture of the tiny global layer vs the step-1 forward's normed, unrotated Q and K | max_abs 1.1e-6 (1.6e-6 on the software adapter), sliding layers `None` |
| tiny Ripple, 24 synthetic texts of 64 tokens, held-out retention of dense attention at top_p 3, top_k 6 (window and sinks alone 0.245) | random 0.403, PCA 0.422, distilled 0.412; distilled beats random on all five dumps tried (+0.006 to +0.015); PCA stays ahead of distilled at this scale |
| same, software adapter, next-token loss vs dense (dense 8.399) | gap random 0.211, distilled 0.155 |
| real layers 0+4 (`brain lsa`, 3 documents of 80 tokens from this repo, 2 epochs, cluster 8, window 8) | mean KL 0.758 to 0.264; retained share random 0.509, PCA 0.513, distilled 0.513 (window and sinks 0.217) |
| PCA indexer for the real global layer, saved and loaded | geometry `rope_base` 0, `sparse_layers` [1] (layer 4 of the checkpoint is layer 1 of the two-layer mapping), equal after the round trip |

Host cost of distilling a 65B global layer (one thread, Ripple's 80 heads over
8 KV heads of 128): the teacher costs 46 ms per query at 8k keys, so about
0.19 s per query at 32k; 64 queries on each of 16 layers is about 190 s per
32k-token sample, plus the retention evaluation at the same cost, because
`teacher_targets` runs layer after layer on one thread (the Adam step is
parallel over layers). Across 16 cores that is about 12 s. The capture itself
no longer holds every query: it keeps the sampled positions (all positions of
a 32k text would be 1.3 GB a layer, 21 GB for 16), the keys are 134 MB a
layer. The blocker the engine was (f32 weights, about 260 GB for 80 layers) is gone with
step 5: the NVFP4 release is 52 GiB resident.

Step 5, weight tiers, KV tiers, placement, stages. The engine holds each linear at
the tier the checkpoint stores (`resolve_tiers`: bf16 stays bf16 and the NVFP4
release keeps its MLP in NVFP4 and the rest in bf16) or at a `--weight-format`
policy, through the shared `upload_from_source` and `Ops`; K and V at a
`model::kv_tier::KvTier` through `KvKernels` (the paged triad over a per-row
scaled pool); the layers in `Stage`s opened from a `model::shard::StreamPlan`.
`ricursive::POLICY` is what the model declares (weights f32, bf16, f16, int8, fp8
with per-leaf rules; KV f32, bf16, int8, fp8; sparse attention). A tier the
device cannot hold is an error naming the tensor and the tier it would have used
(the CPU JIT has no packed-int8 dot, so int8 is refused there and runs on the
software Vulkan adapter).

NVFP4 semantics. The release stores, per MLP linear: `weight` U8 `[n, k/2]` (E2M1
codes, even element in the low nibble), `weight_scale` F8_E4M3 `[n, k/16]`,
`weight_scale_2` F32 scalar and `input_scale` F32 scalar; attention, `embed_tokens`,
`lm_head`, norms and `value_res_lambda` are bf16; there is no `hf_quant_config.json`
(404), the quantization is declared in `config.json` (`quant_method: modelopt`,
`quant_algo: NVFP4`, static 4-bit input activations, group 16, every `self_attn*`,
`embed_tokens` and `lm_head` ignored). 1923 tensors, `total_size` 57,213,636,640 bytes
(`placement::footprint` costs 57,216,944,320: 5.8e-5 off). Brain executes it
float-read, `Weight::Micro` (`NVFP4`): the tile decode yields `E2M1 x scale_2 x E4M3`
and multiplies the f32 activations (W4A16 on every backend; on CUDA the library GEMM
reads bf16 activations by default, `LinearActivations::AUTO`). It does not quantize
activations to FP4, so it is at least as precise as the W4A4 reference. The static
`input_scale`s are read and checked at open (`import::activation_scales`: exactly one
finite positive scalar each, present for every quantized weight or for none) and not
applied. Backends: the CPU JIT and the software adapter run the tile decode and
the dense GEMM; CUDA has the native NVFP4 kernels (`LinearActivations`), not
validated here.

Weight tiers, tiny model (3 layers, 15 rows: 10 prefill and 14 decode steps with window 4
wrapped, logits against the f32 engine; `tests/it/tiers.rs`):

| tier | engine vs f32 engine on the tier's own decode (scaled) | mean KL vs original f32 | top-1 vs original |
|---|---|---|---|
| bf16 | 0 (CPU), 1.2e-6 (software) | 9.7e-5 | 93.3% |
| f16 | 0 | 1.2e-6 | 100% |
| fp8 e4m3 (128 blocks) | 0 (CPU), 7.8e-7 (software) | 2.2e-2 | 73.3% |
| int8 (software adapter only; W8A8) | 2.3e-2 (activations quantized too) | 1.7e-3 | 100% |
| NVFP4 MLP (fixture encoder, random weights) | 6.1e-7 (CPU), 1.1e-6 (software) | 0.11 | 60% |

The tiny model has random weights, which 4 and 8 bit formats do not preserve as they do
trained ones; the tier numbers that mean something are the real-tensor ones below. "The
tier's own decode" is the device read-back of the loaded weight (`Ops::read_host`); the
codec's host `Weight::round_trip` differs from the loader's FP8 quantizer by one step on some
elements (the loader rounds block scales to bf16), which is why the reference is the
read-back.

NVFP4 on real tensors (layers 0 and 4 of `ricursiveAI/Ripple-65B-Preview-NVFP4`
fetched by range reads beside the bf16 partial, 3.4 GB; CPU backend;
`tests/it/real_nvfp4.rs`; 115 tokens of an English and Rust paragraph tokenized greedily
over the release vocabulary):

| check | result |
|---|---|
| device decode vs an independent ModelOpt decode (E2M1 x E4M3 x `weight_scale_2`), 60 rows over six matrices | worst error 1.0e-7 of the row range |
| NVFP4 vs the bf16 files they were quantized from: hidden stream relative L2 error | 4.2e-2 after layer 0, 4.2e-2 after layer 4 |
| same, next-token distribution | mean KL 3.6e-3 (last position 8.3e-4), top-1 agreement 87.8% (101 of 115), worst scaled logits error 5.4e-2 |
| static input scales against the activations the MLP saw (gate and up share a scale) | layer 0: observed amax 1.32 of calibrated 1.84 (72%); layer 4: 2.02 of 2.50 (81%): no clipping a W4A4 reference would apply on this text |
| engine on the NVFP4 files vs the prefill model on the same files (9-token prefill, 4 decodes) | worst scaled error 4.1e-6 |
| resident weights of the two layers | bf16 x11 1.58 GiB, NVFP4 x6 0.66 GiB (those six matrices in bf16 are 2.3 GiB) |

A two-layer stack of an 80-layer model is not a language model: these are relative damage
measures of the formats, not task metrics. The 80-layer NVFP4 versus bf16 comparison needs
the whole release (57 GB) and a card, and is not measured.

KV tiers, tiny model, same run, against the f32 cache (rings wrapped three times, global
pool across 4-row blocks):

| KV | worst scaled logits error | mean KL | top-1 | bytes of the pools (tiny) |
|---|---|---|---|---|
| bf16 | 4.5e-3 | 1.8e-5 | 100% | half of f32 |
| int8 | 1.5e-2 | 1.4e-4 | 93.3% | 0.31 of f32 (1 byte + a 4-byte scale per 16 elements) |
| fp8 e4m3 | 5.7e-2 | 2.4e-3 | 86.7% | same as int8 |

Prefix reuse (prefix of 20 and 18 tokens resumed by a longer prompt) reproduces the cold
logits of the same tier exactly (scaled error 0) for bf16, int8 and fp8: the ring snapshot and
the shared blocks are copied as stored, scales included. Latent sparse attention at a budget
past the context over an int8 or fp8 cache answers as the dense read of that cache:
3.4e-7 and 4.2e-7 (software adapter). A bf16 cache with sparse attention, and sparse attention on
a staged engine, are refused by name. The D-LSA capture needs the f32 cache (it reads
unquantized keys) and works across stages.

Stages. `RicursiveEngine::from_source_staged` opens one device handle per stage under the
plan's placement scope, loads only the stage's layers, keeps block tables, ring slots and the
prefix cache on the host (block `b` is block `b` on every stage) and carries the residual stream
across a boundary with `Gpu::copy_to` (peer copy where the pair has a link, host-staged
otherwise). Two stages reproduce the single stage exactly on the CPU backend (scaled error 0,
including a prefix-reuse resume across the boundary and the D-LSA capture). What the CPU can
and cannot do: `--device cpu0-7,cpu8-15` is one compute set (a core mask), not two devices, so
the two-stage test opens two independent CPU `Gpu` handles, whose boundary copy goes through host
memory ("device-to-device transfers on the cpu backend go through host memory"). A real peer copy,
the `with_gpu` placement of a stage on a card and `StreamPlan::cards` (the free-memory probe) have
not run: all physical GPUs are in use by other jobs. The one other serving engine I found that splits a sequence's decode
over cards is `Qwen35GgufResident` (`qwen35/int8_gguf_resident.rs`), whose stage seam this engine follows; `model::shard::Shardable` is the training pipeline's trait (it
needs `Model`, a backward) and is not implemented for Ripple: the adapter is `placement`
(`LayerBytes`, `ShardCost`) until step 6.

Placement, published configuration on 96 GiB cards (`capacity::headroom` leaves 83.8 GiB usable
of 96; `placements_of_the_published_configuration_on_96_gib_cards` prints the whole table), one
sequence, latent sparse attention charged (`lsa_device_bytes` of a default indexer), the dense
paged triad capped at 8192 keys (`ServeConfig::dense_keys`), 512-row prefill chunks. Every
row therefore assumes the global layers read sparsely; a dense read at 128k needs
`dense_keys` 131072, whose score scratch is 2 x 4 x rows x 80 x 131072 bytes (5.4 GB at 64-row
chunks, 43 GB at 512), and past that the dense triad is not an option. Fixed bytes per stage 2.9
to 3.2 GiB (score scratch, activations, tile budget) are on top of the stage bytes:

| weights | KV | context | cards | per card (stage bytes) |
|---|---|---|---|---|
| bf16 (119.7 GiB) | bf16 | 128k | 2 | 63.7 + 64.8 GiB |
| bf16 | int8 | 128k | 2 | 61.6 + 62.8 GiB |
| bf16 | int8 | 1M | 2 | 76.7 + 77.9 GiB (33.0 GiB of KV) |
| bf16 | bf16 | 1M | 3 | 63.7 + 63.3 + 58.9 GiB (64.0 GiB of KV) |
| NVFP4 release (52.3 GiB) | int8 | 128k | 1 | 57.0 GiB |
| NVFP4 | bf16 | 128k | 1 | 61.1 GiB |
| NVFP4 | int8 | 1M | 2 | 43.7 + 43.5 GiB |
| NVFP4 | bf16 | 1M | 2 | 59.3 + 59.1 GiB |
| fp8 (59.9 GiB) | int8 | 128k | 1 | 64.6 GiB |
| int8 (67.3 GiB) | int8 | 128k | 1 | 72.1 GiB |

**Staged engines cannot read sparsely yet** (open items), so the multi-card rows are placements of
bytes, not configurations this engine serves at those contexts: a staged engine reads dense, which
caps its context by the score scratch. One card holds the NVFP4 release with an int8 cache up to about 843k tokens (bf16 cache 442k;
fp8 weights 614k and 319k). Rings cost 0.4 GiB (bf16) or 0.2 GiB (int8) a slot of the whole stack
whatever the context. The weights cost matches the published `total_size` and the 65B parameter
count (`placement` unit tests); the fixed per-stage bytes are an estimate (the activations of the
widest forward), not a measurement, and the allocator overhead is `capacity::headroom`'s 14%.

Not measured or not run: any card (CUDA parity of the tiers, the native NVFP4 kernels, peer copies,
throughput of any tier or KV format, loading time), the 80-layer release end to end, int8 on the
CPU backend (refused), the `brain lsa` staging path (`StreamPlan::cards` probes free card memory),
and the host-RAM figure of the embedding gather (it reads one row per token from the mapped file).

### GPU phase runbook

```
brain lsa distill --arch ricursive --checkpoint <Ripple-65B directory> \
    --corpus <directory of source files> --out ripple-lsa.safetensors \
    --tokens 32768 --train-docs 48 --epochs 3 --queries 64 --top-k 1024 --top-p 256 --device gpu0
brain lsa eval --arch ricursive --checkpoint <directory> --corpus <directory> \
    --indexer random,pca,ripple-lsa.safetensors --tokens 32768 --skip-docs 48 --held-docs 6
```

Needs the host parallelism above for the teacher. Serve with the artifact through the
engine's `set_lsa`; the catalog entry comes later.

Which tier and placement (all numbers above; the first run on a card checks each against the
engine's `kv_bytes`/`weight_report` and `nvidia-smi`):

* **1 GH200 (96 GiB), D-LSA distillation and evaluation, serving up to ~0.8M tokens**: the NVFP4
  release loaded as published (`--checkpoint <Ripple-65B-Preview-NVFP4 directory>`, no
  `--weights`), `--device gpu0`. 52.3 GiB of weights. Distillation and the capture need an f32
  cache (`brain lsa distill`: 128 KiB a token over the 16 global layers, 4 GiB at 32k tokens, fine). Serving
  and `prefill` timing take `--kv-cache int8` (33 KiB a token: the card holds about 843k tokens
  with sparse attention charged) or `bf16` (442k). `eval` and `set_lsa` need one stage.
* **2 GH200, the bf16 release (129 GB)**: `--device gpu0,gpu1`, no `--weights`: the layers are staged
  over the fewest cards that hold them (bf16 + int8 cache: 2 cards, 62 + 63 GiB at 128k). Dense
  attention (`dense_keys` set to the context, 64-row chunks), scoring and the D-LSA capture work
  across stages; latent sparse attention does not, so a 1M context on several cards waits for the
  open item below and the one-card NVFP4 limit (843k tokens, int8 cache, sparse) is the longest
  context served today. `--weights fp8` or `int8` from the bf16 release
  quantizes at load (fp8 59.9 GiB, int8 67.3 GiB resident, 1 card to 128k).
* First checks on a card, in order: the NVFP4 tile decode and the library bf16 GEMM against the CPU
  numbers in this file (`real_nvfp4` with `--device gpu0`; set `BRAIN_REQUIRE_CAPABILITIES`), the int8
  tier (`tiers` with `--device gpu0`), peer copy vs host-staged boundary on two cards, then throughput
  per tier with `brain perf` (`kernels.md` section F) before any tuning.

## Open items for the later steps

- Fused dense attention for group 10: `KvKernels::flash_decode` is specialised for head width 256 and
  `(128, 1)`; Ripple (128, group 10) keeps the paged triad, whose score scratch is why
  `ServeConfig::dense_keys` exists. Only a sparse engine can run long contexts today.
- Latent sparse attention across stages (one binding per stage needs per-stage index pages in the block table): without it a multi-card engine reads its global layers dense and the long-context rows of the placement table (1M tokens on 2 or 3 cards) are not servable.
- Activation quantization to FP4 (W4A4 as the release was calibrated) as an opt-in beside the float-read default
  would need the static `input_scale`s applied; the checks and the coverage numbers above are the groundwork.
- The embedding is gathered on the host a row at a time from the mapped file (no device copy); a bf16 device table
  would save the host read at 1 GiB.

- Step 1's `Layer::record` still materializes `[heads, t, t]` scores; the
  engine does not (its scratch is `rows x heads x keys`, rows being one
  prefill chunk or the decode batch, keys the window or the sequence limit).
  It dispatches the paged decode triad because the fused paged flash kernels
  are not CPU-capable; on a GPU the fused prefill and decode kernels (global
  layers) and a windowed variant for the rings are the candidates, selected
  by capability, with the triad as the reference.
- The ring is `window + prefill_chunk` rows: a chunk writes all its rows
  before any attends. A sliding row's block table is a list of its ring rows
  (`block_size` 1), uploaded per forward; a windowed paged kernel that walks
  the ring itself would drop that table.
- Prefix reuse is at the filed length only (a snapshot of the rings plus the
  shared global blocks); branch points (`SnapshotPrefixCache::branch_point`,
  `hint_shared_prefix`) are not wired, so prompts that share a document and
  then diverge do not reuse it.
- LSA (step 3) can sit on the engine's global layers: their K is already
  RMS-normed and unrotated in the pool, and `BlockTable` carries index pages.
  The engine's `attend` for a global layer is the one place a sparse read
  replaces the dense triad; the lifted component needs the block pool, the
  block tables and `cap` from the engine, and a selection that falls back to
  dense below a crossover.
- The value-residual lerp uses `edm_mix` with per-row coefficients
  `(1 - s, s)` (an f32 rounding apart from `x0 + s (xn - x0)`); a fused
  QK-norm + RoPE + lerp kernel is the decode-path candidate.
- The global layers' Q/K leave `Layer::record` RMS-normed and unrotated, which
  is exactly what the LSA index (step 3) reads.
- Real-weight inputs live under `testdata/ricursive/ripple-65b-partial/`; they
  are not in the fetch mirror yet, so the real-weight test skips by name on a
  box without them.
