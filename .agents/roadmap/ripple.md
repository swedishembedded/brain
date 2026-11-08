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
| drafter | `Ripple-65B-Preview-DFlash2` (`DFlash2DraftModel`, `model_type: qwen3`, 4.04 B parameters in one 8.08 GB bf16 `model.safetensors`; 81 tensors; layers 1, 20, 39, 58, 77 of the target are its taps) |

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
| speculative decoding | DFlash2 in `crates/qwen35` (bound to its GGUF single-stream path) | done as step 6b: the draft is the shared `crates/dflash2`, the engine retains the taps, the scheduler's verify takes its proposals |
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
       ~60 GB free). Done on the CPU backend; CUDA parity is step 8.
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
       no card has run it (step 8).
5b. [x] Latent sparse attention across stages (one binding per stage device over the
       global layers it owns, index pages in per-device domains of the shared block
       tables) and fused paged flash attention for Ripple's group of 10 (global layers
       and the rings' windows, all KV tiers). Measured below on the software Vulkan
       adapter; no card has run it.
6. [x] Served like Qwen3, through the generic paths and no parallel stack: catalog entry
       (`brain/ricursive`, `RicursiveSpec`), `RicursiveResident` (memory from
       `ricursive::placement`, stages from the device selection, host tier on `--device cpu`),
       the checkpoint's tokenizer and `chat_template.jinja` rendered by the shared engine
       (byte-identical to `apply_chat_template`), reasoning and tool calls through the shared
       scanner (streaming deltas), `--weight-format`/`--kv-cache`/`--sparse-attention`/`--ctx`/
       `--max-batch` checked against `ricursive::POLICY`, `brain ricursive generate`, a pull that
       fetches `chat_template.jinja`, batched speculative verify with prompt-lookup drafting.
       End to end on the CPU backend and the software adapter (`tests/e2e/ripple.bats`). The
       No card has served it (step 8). DFlash2 speculation is step 6b.
6b. [x] DFlash2 speculative decoding, through the generic scheduler: the draft model out of `qwen35` into a
       target-agnostic crate (`crates/dflash2`: per-layer sliding, full and NoPE attention, a ring of window plus one
       block per sequence, one batched forward over the blocks of several sequences, safetensors or GGUF), the
       engine contract for hidden-state taps (`PagedDecoder::enable_taps`/`read_taps`, a `TapPool` addressed by the
       KV block tables), `RicursiveEngine::set_drafter`, `--speculative dflash` and `--draft-model`, the draft and
       the taps charged in the placement, and speculation beside `--sparse-attention`. Measured below on the CPU
       backend and the software adapter; no card has run it (step 8).
7. [x] Backward for every Ripple op, gradchecked and compared with PyTorch autograd; the `Model` and `Shardable`
       faces through the generic trainer, pipeline and objectives; LoRA/QLoRA through `model::adapter` served by
       the engine; full fine-tuning; N-LSA on the global layers; `brain finetune` and the SDK fine-tuners. Measured
       on the CPU backend and the software Vulkan adapter below; no card has trained it (step 8).
7c. [x] Adapters served from a bank over one shared base (generalised `model::adapter::bank` / `catalog::adapter_registry`,
       no Ripple copy): per-request selection by model id, a batch mixing base and several adapters on any number of
       stages, `--adapter`, `--adapter-manifest`, `--watch-adapters`, model-store adapters, `TextGenerationPipeline::adapter`.
7b. [x] `brain serve --adapter` for Ripple, training across pipeline stages from the memory plan (generic
       `model::plan_stage_layout`), the chunked head, native sparse attention as a model-agnostic trainer option.
       Open: a bank of adapter slots, norms and `value_res_lambda` beside an adapter (Open items).
8. [ ] CUDA parity and the measured long-context numbers (both GH200s,
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
(`LayerBytes`, `ShardCost`) until step 7.

Placement, published configuration on 96 GiB cards (`capacity::headroom` leaves 83.8 GiB usable
of 96; `placements_of_the_published_configuration_on_96_gib_cards` prints the whole table), one
sequence, 512-row prefill chunks, fused attention (no score slab; the triad would add
2 x 4 x rows x 80 x keys bytes, 43 GB at 512 rows and 128k keys), latent sparse attention charged
per global layer and per stage (`LsaCharge`: 0.2 GiB at 128k, 1.5 GiB at 1M tokens). Fixed bytes per
stage (activations of the widest forward, the tile budget, the index scratch) are 0.5 to 1.0 GiB on
top of the stage bytes:

| weights | KV | context | cards | per card (stage bytes) |
|---|---|---|---|---|
| bf16 (119.7 GiB) | bf16 | 128k | 2 | 63.6 + 64.8 GiB |
| bf16 | int8 | 128k | 2 | 61.6 + 62.8 GiB |
| bf16 | int8 | 1M | 2 | 76.6 + 77.8 GiB (33.0 GiB of KV) |
| bf16 | bf16 | 1M | 3 | 63.6 + 63.2 + 58.8 GiB (64.0 GiB of KV) |
| NVFP4 release (52.3 GiB) | int8 | 128k | 1 | 57.0 GiB |
| NVFP4 | bf16 | 128k | 1 | 61.1 GiB |
| NVFP4 | int8 | 1M | 2 | 43.6 + 43.4 GiB |
| NVFP4 | bf16 | 1M | 2 | 59.2 + 59.0 GiB |
| fp8 (59.9 GiB) | int8 | 128k | 1 | 64.6 GiB |
| int8 (67.3 GiB) | int8 | 128k | 1 | 72.0 GiB |

These rows are configurations the engine serves: the multi-card ones read the global layers sparsely through
one binding per stage (below), or densely through the fused kernels. One card holds the NVFP4 release with an
int8 cache up to about 921k tokens with the sparse index charged (bf16 cache 479k; fp8 weights 692k and 360k).
Rings cost 0.4 GiB (bf16) or 0.2 GiB (int8) a slot of the whole stack whatever the context. The weights
cost matches the published `total_size` and the 65B parameter count (`placement` unit tests); the fixed
per-stage bytes are an estimate, not a measurement, and the allocator overhead is `capacity::headroom`'s 14%.

Step 5b, latent sparse attention across stages (software Vulkan adapter, two `Gpu` handles of lavapipe
coexist; a synthetic four-layer model, global layers 1 and 3 on different stages, 26 prompt tokens in
chunks of 5 and 14 decodes; `tests/it/staged_lsa.rs`). `BlockTable` index pages gain a domain
(`index_pages_in`), `LsaAttention::set_domain` and `LsaBinding::attach_in_domain` give each binding its own,
`Indexer::restricted_to` hands each stage its layers, and `LsaBinding::device_bytes_split` charges per
layer and per device; page allocation, release and snapshot retention run on every stage's binding from the
shared host-side tables, and no ricursive special case touches the binding.

| gate | result |
|---|---|
| two-stage sparse vs single-stage sparse (top_p 2, top_k 3: 796 of 1640 causal keys) | logits scaled error 0 (bit-identical), same keys selected |
| full budget across stages vs the dense engine | 8.3e-7, every causal key selected |
| prefix reuse across stages with the index on (24 filed, resumed to 30) | warm equals the single-stage warm run exactly; each stage rebuilds its own pages (2 for the resumed prompt against 1 on one stage), the cache holds 6 pages against 3, reclaiming frees all |

Pages are per stage, so the index pool costs the same bytes per global layer wherever the layer sits.

Step 5b, fused attention. `paged_flash_decode_gqa_hd256` took at most 8 query heads per kv head and only
256-wide rows at the compact tiers. Its per-head arrays are now sized by `GMAX` (specialised when the group
passes 8), the softmax runs 8 heads per round of its 128 threads, and the compact tiers' row scale follows the
head width (`HDS`); `kv_tier::kernel_list_for` and `KvKernels::resolve_for` register a model's own shape
(`check_flash_shape`: power-of-two head width 16 to 256, group 1 to 128). `RicursiveEngine` reads a global
layer with `flash_prefill_hd256` for a chunk of one sequence and the split-key decode kernel for a decode
batch, and the rings' windows with the decode kernel (a row per window), wherever the device has workgroup
barriers; the CPU JIT keeps the triad (with a warning that names the device), as does
`ServeConfig::fused_attention: false`.

| gate (software adapter) | result |
|---|---|
| fused decode vs f64 attention over the decoded planes, f32 / bf16 / int8 / fp8, (20,2,128) group 10, (40,4,128), (24,2,64) group 12 and the built-in shapes | worst relative error 1.3e-6 to 3.1e-6 |
| engine, fused vs triad, 5 layers, group 10 at head_dim 128, window 4 wrapped, prefill chunks of 5 and 14 decodes | f32 1.5e-6; bf16 2.4e-4, int8 5.3e-4, fp8 1.4e-6 |
| same, distance to the f32 cache | triad / fused: bf16 6.31e-3 / 6.28e-3, int8 2.82e-2 / 2.81e-2, fp8 8.03e-2 / 8.03e-2 |
| shared conformance suite (continuous batching, no head-of-line blocking, prefix reuse) on the fused kernels | passes |
| dense read past a 16-key score scratch | the triad refuses by name; the fused engine answers as the triad reference (6.4e-7) |

The tier rows of the engine comparison are not noise of the kernels (they are exact against f64 above): the
triad's bf16, int8 and fp8 variants are gated at 1e-4 per operation, and five random layers amplify that.

Context the dense path then supports (fused kernels, no index, no score slab; only the cache bounds it, found
by bisection in the same test; the planes index with 32 bits, so one global layer holds at most 4.19M tokens):

| weights | KV | 1 card | 2 cards | 3 cards |
|---|---|---|---|---|
| NVFP4 release | bf16 | 491k | 1.85M | 2.89M |
| NVFP4 release | int8 | 962k | 3.60M | 5.62M |
| bf16 | bf16 | does not fit | 741k | 1.95M |
| bf16 | int8 | does not fit | 1.44M | 3.78M |

A dense decode token reads the whole cache (64 GiB of bf16 at 1M tokens), so these are capacity limits;
decode speed at those lengths is the sparse path's reason and is not measured here.

Not measured or not run: any card (CUDA parity of the tiers, the native NVFP4 kernels, the CUDA build of the
widened fused kernel, peer copies, throughput of any tier, KV format or attention path, loading time), the 80-layer release end to end, int8 on the
CPU backend (refused), the `brain lsa` staging path (`StreamPlan::cards` probes free card memory), the fused kernels' speed against the triad,
and the host-RAM figure of the embedding gather (it reads one row per token from the mapped file).

### Step 6: served like Qwen3 (CPU backend and the software Vulkan adapter, 2026-11-07)

Gates, all in this repository: `crates/catalog/tests/it/ricursive_chat.rs`,
`ricursive_served.rs`, `crates/serving/src/model_dir.rs` tests, `crates/ricursive/tests/it/serve.rs`,
`crates/data/src/qwen_chat.rs` tests and `tests/e2e/ripple.bats`.

| gate | result |
|---|---|
| tokenizer vs the `tokenizers` library (`tokenizer.json`, 100278-entry BPE; code, unicode, whitespace runs, special tokens, digit runs, contractions, empty text) | 18 texts, ids and decode identical |
| rendered prompt vs `transformers.apply_chat_template` on the published `chat_template.jinja` | 14 conversations byte-identical (system, multi-turn, reasoning, `clear_thinking`, tools, tool calls with string arguments, parallel calls, tool results, generation prompt) |
| reply scan against the rules of `inference/server/ripple_format.py` (reasoning, content, calls; chunked 1 character to whole; the Python needs SGLang and was read, not run) | 14 layouts equal, plus streamed deltas; 7 replies parsed and sent back re-render to the generated tokens |
| engine verify vs the full forward, two sequences in one call, rollback at 4 acceptance lengths | argmax equal at every row, continuation equal within 1e-5 scaled error |
| shared conformance suite with speculation (any drafter: right, partly right, wrong) | passes: output equals plain greedy, accepts and rejects exercised |
| prompt-lookup speculation through the resident (cycle model, 3 requests, 24 tokens each) | output equal to plain greedy; 54 proposed, 54 accepted over 5 verify steps |
| scripted tiny Ripple over HTTP (CPU backend): tool call, stream, tool result sent back, `chat_template_kwargs` | pass; 7 completion tokens, one call, reasoning split out |
| same model with a distilled indexer on the software adapter | pass; `lsa_sparse_row_reads` > 0 through the resident; a bad indexer is refused at startup |
| real layers 0 and 4 (two-layer model, 5.2 GB bf16) served on the CPU backend | built in 2 s at the first request; a 173-token tools prompt plus 4 tokens answered in 11 s end to end |

Automatic context on the real config (the largest power-of-two halving of 2 Mi that places on
cards of 96 GiB with 4 live and 2 cached sequences, `ricursive::placement::footprint`): NVFP4
release on one card 262144 tokens at a bf16 cache, 524288 at int8; on two cards 1 Mi and 2 Mi.
bf16 release: one card does not hold it; two cards 524288 at bf16, 1 Mi at int8. These are the
plan's numbers (no card has checked them).

Shared-code defects found and fixed on the way: `from_model_dir` preferred the template inline
in `tokenizer_config.json` to `chat_template.jinja` (Ripple's inline one has no tools); a pull
did not fetch `chat_template.jinja`; the OpenAI surface sorted a tool schema's members before the
template printed them and gave a template that serializes a call's arguments the JSON text, not
the object; every JSON tool call was reported as a discarded tool-call tail; `--device software`
left the host budget at zero so nothing could be placed.

### Step 6b: the DFlash2 drafter (CPU backend and the software Vulkan adapter, 2026-11-08)

The drafter (`ricursiveAI/Ripple-65B-Preview-DFlash2`, fetched with `brain pull` to
`~/project/shared/models-qhf/ripple-65b-preview-dflash2/`: `config.json` and an 8,075,513,104-byte `model.safetensors`)
has 81 tensors of 4,037,752,064 parameters, all bf16. Against the Qwen3.8-27B drafter this repository first ported
(`qwen35::dflash2`), what differs:

| | Qwen3.8-27B draft (GGUF) | Ripple draft (safetensors) |
|---|---|---|
| tensor names | `blk.N.attn_q.weight`, `enc.output_norm`, `selector_predecessor.weight` ... | `layers.N.self_attn.q_proj.weight`, `hidden_norm`, `candidate_selector.predecessor_codebook` ... (both spellings live in `dflash2::source::Tensor`) |
| layers | all sliding, window 2048, all rotate | 4 sliding (window 4096, RoPE theta 10000) then 1 full NoPE (`layer_types`, `no_rope_layers: [4]`) |
| dynamic convolutions | two taps | four taps (`conv_kernel_size` 4), groups of 16, bases `[2, 4, 5120]`, projections `5120 -> 2560` |
| selector | top 16 | top 32, rank 256, codebooks `[100352, 256]` |
| width, heads | 5120, 24/4 (head 256) per the GGUF | 5120, 80/8 of 128, FFN 40960, `q_norm`/`k_norm` over the head (plain RMSNorm gains, not `1 + w`) |
| target layers | GGUF stores them one higher (hidden-states tuple index) | `target_layer_ids` are decoder layers: the output of layer `i` |
| attention | non-causal within the window both ways | non-causal; the full layer reads every position |
| config keys | `dflash.*` metadata | `dflash_config`, `layer_types`, `no_rope_layers`, `rope_parameters.rope_theta`, `is_causal: false`, `draft_vocab_size` |

It is NOT Ripple's PeriNorm/value-residual block: the layers carry only `input_layernorm`, `post_attention_layernorm`,
the two head norms, the three MLP linears and the two convolution sublayers.

| gate | result |
|---|---|
| draft forward on the REAL 4.04 B weights vs `tools/goldens/dflash2_reference_forward.py real` (float32 numpy over the bf16 values; 24 synthetic tap rows of 5 x 5120 with heavy-tailed channels, the anchor ` the` (token 279), the shared embedding and `lm_head` from `layers-0-4.safetensors`), CPU backend | layers 0 to 4 scaled error 3.2e-6, 1.0e-5, 2.5e-5, 2.4e-5, 2.4e-5; post-norm rows 3.1e-5; head logits (15 x 100352) 2.7e-5; the selector's 15-token path identical; the embedding rows exact. Load 7.42 GiB in 4.7 s, 24 context rows in 1.2 s, one block in 8.6 s (288 host cores) |
| the same on a tiny random draft (window 6, three layers, one full NoPE, block 5), context of 14 and 17 positions so the rings wrap; context fed whole, in pieces, and extended after a denoise; two sequences in one forward | per layer, post-norm rows and logits at most 2e-6, the selector paths identical |
| retained taps vs the step-1 forward's hidden state at the tapped layers (tiny Ripple, `dflash::retained_taps_equal...`) through chunked prefill, decode, a verify of four rows of which two are kept, and a prefix hit whose tail block is copied | within bf16 rounding (scaled error bound 5e-3) at every position |
| speculation with the draft equals plain greedy (4 sequences of 9, 15, 30 and 6 tokens, 40 new tokens each, target window 4, draft window 6) | all four outputs equal; 14 sequence-rounds, 19 proposals, 8 verify steps; the random draft is right only by chance (0 accepted), so rejection and rollback are what ran |
| the same after a prefix hit (22 tokens filed, 31 resumed) | equal |
| the shared conformance suite (batching, short request not held, prefix reuse, scripted drafters that are right, partly right and wrong) with the draft attached and taps retained for every row | passes |
| sparse attention: the scripted-drafter conformance run beside the index (software adapter, cluster blocks of 8 keys) | equal to plain greedy with the same index; with `spec_rows_limit` off the output diverges (`91 15 68 ... 116 51` against `... 102 39`), so the limit is what makes it exact |
| sparse attention with the draft model proposing (software adapter) | equal to plain greedy with the same index; the global layer read sparsely |
| e2e (`tests/e2e/ripple.bats`): the scripted tiny Ripple with the fixture's random draft over HTTP, `--speculative dflash` with the draft found beside the target; again on the software adapter with `--sparse-attention` | the same reply and tool call as the plain server; the log reports `spec_rounds`, `spec_accept_rate`, `spec_tokens_per_round`, `spec_verify_steps` |
| sampled requests (temperature 0.9, top-k 20, seeded) with the draft attached | run to the end through the rejection-sampling verify, repeatable under one seed |
| a draft model that is not there, or built for another target | refused by name; the model is not served |

Memory the plan charges (`placement::tests::the_draft_model_is_charged...`, NVFP4 target, int8 KV, one sequence): the
draft is 7.42 GiB of weights plus its caches (1.13 GiB at 128k tokens, 2.13 GiB at 256k; four sequences at 256k: 8.5
GiB) on the last stage's card, and the target's retained hidden states are 50 KiB a pooled position on the host (6.2 GiB
at 128k, 12.5 GiB at 256k). On one 96 GiB card the longest dense context falls from about 962k tokens to about 581k with
the draft attached.

Shared-code defects found and fixed on the way: speculation was off beside latent sparse attention (a verify that
crossed a cluster block of the index read a state a plain decode builds later: now the scheduler bounds each verify,
`spec_rows_limit`, and the verify prepares the index as a chunk); the dynamic-convolution kernel could not run
several blocks in one dispatch; `brain pull` refused the draft ("unsupported architecture") and the model scan
called a pulled draft an unservable family; the `ricursive` registry row declared none of the training capabilities
the capability matrix records.

Not verified: the draft with REAL hidden states of the 80-layer target (no card here holds it, so every number above is
the draft's forward, not its acceptance), the acceptance rate and speedup, and two conventions the published files do
not settle. The target taps are taken as the OUTPUT of decoder layer `i` (as the Qwen3.8 port established, and as the
GGUF's one-higher index shows), and the block's noise embedding is the target's `embed_tokens` row as stored, WITHOUT
`embed_scale` (the README says "token embeddings"; the scaled embedding is what the target's first layer reads). Both
are one line to flip (`RicursiveEngine::tap_layers` offset, `DraftHost::embed`) and the first run on a card settles them
by `spec_accept_rate`: a wrong convention shows as an acceptance near the random draft's.

### Step 7: trained through the generic paths (CPU backend and the software Vulkan adapter, 2026-11-07)

Gates: `crates/ricursive/tests/it/train_*.rs`, `crates/gradcheck/tests/it/ricursive_fd.rs`,
`crates/sdk/tests/it/chat_finetune_ricursive.rs`, `tests/e2e/ripple.bats`, the references from
`tools/goldens/ricursive_dump_reference.py` (`grads`, `real-grads`, `real-lora`; torch 2.15,
transformers 5.22, fp32 on the CPU). `ricursive::Ripple` is a `model::Model` (embedding scale, value-residual
lerp with its own gate, f32 sigmoid attention gate, one- and two-sided clamps with zero gradient where they
saturate, per-head QK norm, PeriNorm order, windowed and global attention, untied head) and a
`model::Shardable`; only the residual between layers is kept (`model::saved`, f32/bf16/fp8 boundaries) and
each layer is rebuilt in the backward.

| gate | result |
|---|---|
| finite differences, whole tiny model, every parameter, 8 random directions, seeds 3 and 11 (`ricursive_every_parameter_matches_finite_differences`) | 206 checks inside atol 4e-3 / rtol 8e-2; worst relative error 2.5e-2 on directions with a gradient over 0.05 |
| finite differences entry by entry: the embedding (it sums every layer's value path, 80 of them in the real model) and each layer's value-residual gate | worst absolute error 2.4e-4 at gradients to 0.4; the feed-forward clamp is off in this check (a finite difference across its kink is wrong, see the gradcheck commit) and is held by the directional check, the kernel tests and autograd |
| every feature moves the loss (clamp, window, value residual, gate, embedding scale) | each by more than 1e-3 |
| tiny 3-layer model (sliding, sliding, global; window 4 < 12 tokens; clamp 0.5 firing) vs PyTorch autograd of the official `modeling_ricursive.py` | loss exact to 1e-5; all 48 parameter gradients, worst scaled error 4.5e-5 |
| three AdamW steps with global-norm clipping vs `torch.optim.AdamW` | losses to 2e-7, parameters to 2.5e-5 |
| real layers 0 and 4 (5.2 GB bf16 as f32), one backward vs autograd | loss 9.5e-7; gradient norms 1.4e-5; 4096 sampled entries per tensor 4.8e-5 |
| real layers 0 and 4 with a rank-8 adapter on all eight linears of both layers (`one_lora_step_over_real_layers_0_and_4_matches_pytorch`, 47 s forward and backward, 105 s the test) | loss 13.992954 vs 13.992969 (1.0e-6); all 32 factors' gradients in full, worst scaled error 4.3e-5; after one AdamW step 36 of 3,145,728 factor entries moved differently (first-step Adam moves each entry by the learning rate, so a gradient within rounding of zero can flip), loss 11.53493 vs 11.534856 (6.5e-6) |
| a run learns a repeated sentence through the shared fit loop; a run stopped and resumed ends bit for bit where the uninterrupted one does | loss 3.28 -> 0.0000 in 40 steps; resumed equal |
| adapter gradient = merged-weight gradient through the factors (`dB = s dW A^T`, `dA = s B^T dW`) | worst scaled error 3.4e-6 over 48 factors |
| zero-init adapter equals the base; a run learns and leaves the base untouched | adapter loss 4.1951 -> 0.0551; base tensors identical |
| a trained adapter attached to `RicursiveEngine` vs the training model's logits | scaled error 2.3e-7 |
| adapter over a bf16, FP8, NF4 base vs an f32 build over the decoded values (QLoRA) | loss and gradients identical (0.0); int8 needs a packed-int8 product (skips on this backend) |
| adapter over the ModelOpt NVFP4 release (MLP resident in NVFP4, attention and head as stored) vs the decoded values | loss 4.219043 both, worst adapter gradient 1.0e-6 |
| two-stage `model::Pipeline<Ripple>` vs one stage (cut between the second sliding layer and the global layer, so the value residual reaches across it) | loss identical; every gradient equal (worst 6.4e-8, the embedding, which both stages hold); four AdamW steps to 7.8e-7; also with an adapter and with bf16 boundaries |
| N-LSA on the global layer, full budget vs dense (`train_nlsa`, software adapter, 2 x 64 tokens) | loss 3.759443 both, worst gradient difference 3.2e-7 of the largest; window-only 3.771009 |
| N-LSA, sparse budget (window 8, sinks 2, top_k 16, top_p 8) | 40 AdamW steps: loss 3.7757 -> 0.0004 |
| SDK `ChatFineTune` (LoRA), `ChatFullFineTune`, `PreferenceFineTune` on a tiny Ripple | loss falls in each; the DPO margin starts at ln 2; an adapter file holds adapter tensors only; a variant this model does not train and a held-out set it does not score are refused by name |
| training head at every packed tier (`model::head_ce`: F32, BF16, NF4, FP8, int8 through the tile decode of `Ops::matmul_rows`; NVFP4 shares the path) vs the f32 reference over the decoded table, whole and in vocabulary windows | loss within 1e-4, hidden gradient within 1e-4, chunking and windowing change nothing (1e-5); a Ripple adapter over an FP8 base holds `lm_head` at FP8 (loss 4.2025094 both, worst gradient error 1.8e-6), no longer widened to bf16 |
| training head: loss and gradient a chunk of 64 rows at a time vs the batch whole (2 x 64 rows) | loss identical (3.887103), worst gradient error 1.8e-7; the parity, gradcheck and real-layer gates above run through it |
| two stages through the SDK trainer (`stages(2)`; the generic layout planner `model::plan_stage_layout`) vs one, LoRA and full fine-tune, CPU | losses before and after equal within 1e-5 and 1e-4 relative, every adapter factor within 1e-4; `brain ricursive finetune --stages 2` prints the layout and trains the one-stage loss |
| adapter bank, one batch of a base row and the rows of two adapters over one resident base (`train_bank`), one stage and two | every row equals that row decoded alone (0.0) and the training model's forward with that adapter (worst 4.3e-7); the adapters change the answer (scaled difference 1.0); loading and unloading while serving (eviction among 3 adapters in 2 slots, draining, purging) |
| the shared serving conformance suite with the bank on (batching, no head-of-line blocking, prefix reuse, speculation, adapters never share KV with the base, embedded prompts apart) | passes |
| the catalog resident: two adapters and the base from ONE instance (`RicursiveResident`, `AdaptedInstance`), a batch of the three vs each alone vs the training forward's greedy tokens | equal token for token; an adapter of another family, base, layer, linear, rank 0 or above the bank's rank cap, or a server with no bank, is refused by name; the bank's bytes are in the placement |
| adapters offered, replaced and withdrawn on a running executor (`LiveAdapters`, the data plane of the manifest and the watcher) | answers at once beside the base; a new release under a card id replaces the old; a withdrawn adapter is an unknown model; the base is unchanged |
| model-store adapters (`discover`) | two adapters of a Ripple base are served as views of one resident, batching with it and budgeted as it |
| `tests/e2e/ripple.bats`: two adapters (one pinned, one in the store) and the base answer chat requests from one `brain serve`; the instance is built once; an adapter promoted into a `--watch-adapters` directory is listed and answers while the server runs | pass |
| `TextGenerationPipeline::builder(dir).adapter(file)` over a Ripple | the adapter answers what the training forward does; `attach_adapter` switches to another and `detach_adapter` returns to the base bit for bit, no reload |
| native sparse attention as a trainer option (`--native-sparse`, `native_sparse_attention`) at a full budget vs dense, software adapter, Ripple and Qwen3 | Ripple LoRA losses (3.9267719, 2.2625196) dense vs (3.9267719, 2.2625198) sparse; Qwen3 equal within 1e-3; refused by name on the CPU backend ("workgroup-barrier") and by architectures that do not declare it |
| policy gradient on a tiny Ripple (`ChatFineTune::policy_gradient`, rollouts with an `advantage`), the rewarded actions and the penalised ones scored as held-out sets before and after on the resident model (`ScoredFit`, CPU backend) | rewarded actions' loss 3.8021 -> 3.4217, penalised 3.8529 -> 4.1263; first-step ratio 1 +- 1e-4; adapter card regime `policy_gradient_lora`; SFT held-out loss 3.8228 -> 0.8712 |
| GRPO against a verifiable reward (share of even tokens) on a tiny Ripple over a bf16, NF4 and FP8 base (`rl` `objective_fits`, software adapter, 40 steps, 96 samples each side) | mean reward 0.534 -> 0.844, 0.539 -> 0.802, 0.557 -> 0.839; only the adapters move |
| the trainer's token log-probabilities vs the serving engine's, the engine under a trained adapter in its bank, sampler temperature 0.7, 11 positions of tokens the adapter was not fitted to (`train_lora`) | worst difference 1.9e-6 |
| `brain ricursive finetune` / `brain finetune --arch ricursive` end to end (`tests/e2e/ripple.bats`, `--device cpu`) | a LoRA run on the scripted model lowers the loss and writes the adapter, a second run continues it at its own rank and a different rank is refused; a full run writes a checkpoint |

How a stage is cut: with the value residual every stage holds the embedding table and recomputes `x0` from the
tokens, so nothing but the residual crosses the cut; each stage turns its share of `x0`'s gradient into an
embedding gradient of its own and the pipeline sums the copies as it does for any tensor held by several
stages (`Residuals::ingest` stores a stage's input boundary in the narrow stream). N-LSA reuses
`model::lsa::native` as Qwen3 does, with `Rope::None` for the NoPE index space (the selector takes the rope as a
parameter now) and `window_selection` moved there from Qwen3.

Shared-code defects found and fixed on the way: the Ripple entry-by-entry gradient check failed at the first
commit that introduced it (a finite difference across the clamp's kink); `NativeSelector` assumed RoPE.

CLI and SDK: `brain finetune BASE --dataset FILE --out DIR [--lora RANK] [--arch ID] [--base-format F]
[--activations P] [--continue ADAPTER]` (model-agnostic flags; `brain ricursive finetune ...` is the same
command) over `ChatFineTune` (supervised LoRA, written as `adapter.safetensors` with a card) and
`ChatFullFineTune`; `PreferenceFineTune` and the rest of the SDK take `.arch("ricursive")`. Ripple's trainer
declares its frozen-base formats (`f32, bf16, int8, nf4, fp8` and `nvfp4` as the release stores it) in
`RippleTrainer::LORA_BASES` and an f32 optimizer.

### GPU phase runbook

```
brain lsa distill --arch ricursive --checkpoint <Ripple-65B directory> \
    --corpus <directory of source files> --out ripple-lsa.safetensors \
    --tokens 32768 --train-docs 48 --epochs 3 --queries 64 --top-k 1024 --top-p 256 --device gpu0
brain lsa eval --arch ricursive --checkpoint <directory> --corpus <directory> \
    --indexer random,pca,ripple-lsa.safetensors --tokens 32768 --skip-docs 48 --held-docs 6
```

Needs the host parallelism above for the teacher. Serve with the artifact through
`brain serve --sparse-attention` (below).

Training (step 7), after `brain pull`, one to two GH200s. Every command below ran on the CPU backend at
tiny scale; none has run on a card.

```
# LoRA over the NVFP4 release on 1 GH200 (52 GiB of weights; the adapter, its AdamW state and the activations
# of one row are small beside it). --base-format nvfp4 holds the release as published.
brain pull ricursiveAI/Ripple-65B-Preview-NVFP4
brain finetune ricursiveAI/Ripple-65B-Preview-NVFP4 --arch ricursive --dataset chat.jsonl --out run-lora \
    --lora 16 --alpha 32 --base-format nvfp4 --activations recompute --steps 200 --lr 1e-4 --max-block 4096 \
    --device gpu0

# The bf16 release (129 GB) does not fit one card: the trainer lays the model out over the selected cards from the
# memory plan (model::plan_stage_layout), prints the layout ("pipeline of 2 stages: gpu0 layers 0..40, ...") and
# trains through model::Pipeline<Ripple>; --stages N asks for exactly N stages.
brain finetune ricursiveAI/Ripple-65B-Preview --arch ricursive --dataset chat.jsonl --out run-lora-bf16 \
    --lora 16 --base-format bf16 --activations bf16 --max-block 4096 --device gpu0,gpu1

# N-LSA continued training of the dense release: the global layers read their selection only (a model trained
# dense reads it as a restriction: start from a budget near the context and tighten it). Needs barrier kernels.
brain finetune ricursiveAI/Ripple-65B-Preview-NVFP4 --arch ricursive --dataset long.jsonl --out run-nlsa \
    --lora 16 --base-format nvfp4 --native-sparse --lsa-window 4096 --lsa-sink 4 --lsa-top-k 16384 --lsa-top-p 1024 \
    --lsa-cluster-block 512 --max-block 65536 --device gpu0

# serve the result beside its base (its own instance; a model id of the adapter's card id, see --adapter-id):
brain serve --models-dir <models> --openai 8080 --adapter run-lora/adapter.safetensors --device gpu0
# the adapter also exports as a PEFT directory (ricursive::train::adapter::export_peft)
```

First checks on a card, in order: `train_lora`, `train_pipeline` and `train_tiers::a_chunked_head_*` with `--device gpu0` / `gpu0,gpu1` against
the CPU numbers above; the real-weight step (`train_real_lora`, set the testdata link) on a card against the
CPU loss 13.992954 and gradients; one `brain finetune` step with `--kernel-profile` against the GH200 roofline
(`kernels.md` section F) before any tuning, with the head in view (below).

Serving (step 6), after `brain pull`:

```
# 1 GH200, NVFP4 release, int8 KV, the distilled D-LSA indexer
brain pull ricursiveAI/Ripple-65B-Preview-NVFP4
brain serve --openai --device gpu0 --kv-cache int8 --sparse-attention ripple-lsa.safetensors

# the same without the indexer, prompt-lookup speculation on
brain serve --openai --device gpu0 --kv-cache int8 --speculative ngram

# 1 GH200, NVFP4 target + the DFlash2 drafter + int8 KV (the draft is found beside the target; the startup log prints
# "default draft model ..." and the placement with the draft's 7.4 GiB and caches charged to the card, the taps to
# the host; add --sparse-attention ripple-lsa.safetensors to read the global layers sparsely beside it)
brain pull ricursiveAI/Ripple-65B-Preview-DFlash2
brain serve --openai --device gpu0 --kv-cache int8 --speculative dflash --verbose 2

# 2 GH200, the bf16 release (staged over both cards; add --kv-cache int8 for 1M tokens)
brain pull ricursiveAI/Ripple-65B-Preview
brain serve --openai --device gpu0,gpu1
```

First checks for the drafter, in order: `make test/one P=brain-dflash2 T=it ARGS=tiny` with `--device gpu0` (the tiny draft
against the reference on a card, the fused paged decode attention in place of the CPU's triad), then `real` (needs
`testdata/dflash2/ripple-65b-preview-dflash2` and the golden of `tools/goldens/dflash2_reference_forward.py real`), then
the same request twice, plain and `--speculative dflash`, and the replies must be identical; `spec_accept_rate` in the
`--verbose 2` request lines settles the two conventions listed under step 6b (tap layers, unscaled embedding) and is the
number to record, with tokens/s against plain decode at batch 1 and 4.

First checks: the startup log's placement line and defaults against `nvidia-smi` after the
first request (the numbers above are the plan's); `BRAIN_BIN=./target/release/brain bats
tests/e2e/ripple.bats` is CPU only, so on the card send the tool-call request of that file to
the served id and compare the reply with the CPU run of the same prompt; then
`brain lsa prefill` timings against `brain serve` latency.

Which tier and placement (all numbers above; the first run on a card checks each against the
engine's `kv_bytes`/`weight_report` and `nvidia-smi`):

* **1 GH200 (96 GiB), D-LSA distillation and evaluation, serving up to ~0.8M tokens**: the NVFP4
  release loaded as published (`--checkpoint <Ripple-65B-Preview-NVFP4 directory>`, no
  `--weights`), `--device gpu0`. 52.3 GiB of weights. Distillation and the capture need an f32
  cache (`brain lsa distill`: 128 KiB a token over the 16 global layers, 4 GiB at 32k tokens, fine). Serving
  and `prefill` timing take `--kv-cache int8` (33 KiB a token: the card holds about 921k tokens
  with sparse attention charged, 962k dense) or `bf16` (479k, 491k dense).
* **2 GH200, the bf16 release (129 GB), or 1M+ tokens**: `--device gpu0,gpu1`, no `--weights`: the layers
  are staged over the fewest cards that hold them (bf16 + int8 cache: 2 cards, 62 + 63 GiB at 128k, 77 + 78
  GiB at 1M). Sparse attention (`set_lsa`, `eval`), the D-LSA capture and scoring all work across stages;
  dense reads run the fused kernels (to 1.44M tokens at bf16 weights and an int8 cache). The NVFP4 release
  on two cards with an int8 cache holds 1M tokens at 43.6 + 43.4 GiB, or 3.6M dense. `--weights fp8` or
  `int8` from the bf16 release quantizes at load (fp8 59.9 GiB, int8 67.3 GiB resident, 1 card to 128k).
* First checks on a card, in order: the fused kernels against the triad (`fused` with `--device gpu0`), the
  staged sparse read (`staged_lsa` with `--device gpu0`, then across two cards for the peer copy), the NVFP4 tile decode and the library bf16 GEMM against the CPU
  numbers in this file (`real_nvfp4` with `--device gpu0`; set `BRAIN_REQUIRE_CAPABILITIES`), the int8
  tier (`tiers` with `--device gpu0`), peer copy vs host-staged boundary on two cards, then throughput
  per tier with `brain perf` (`kernels.md` section F) before any tuning.

## Open items for the later steps

- DFlash2, step 6b. The draft's K/V caches are f32, one slot per live sequence, a full layer holding every position
  (2.1 GiB a sequence at 256k): a cache slot above the device's binding limit is refused by name at load, so very long
  contexts need a paged or compact (bf16/int8) draft cache; the dense fallback attention (no workgroup barriers) allocates
  score slabs of `rows x heads x keys` for the full layer. The taps are bf16 on the host (50 KiB a position, 13 GiB at
  256k) and cross to the draft through a host copy per round. The drafter reads one slot per request id; a finished
  request's slot is reclaimed only when another needs it. The shared engine contract is implemented by Ripple only:
  Qwen3.8 (`qwen35::dflash2`) still drives the draft single-stream, behind its own tap path, and Qwen3's engine does
  not retain taps. Sampled decoding takes the scheduler's rejection sampling (`verify_proposals`); with the draft it ran
  (seeded, repeatable, verify steps taken) but its distribution was not measured against plain sampling.

- Adapter serving, step 7c. The bank is allocated per stage with `slots` x `rank_cap` x (in + out) f32 per targeted
  linear (`--adapter-slots`, `--adapter-rank-cap`, `--adapter-targets`; Ripple's default is all eight linears, 4 slots of
  rank 16): 3.75 GiB at 65B for four rank-16 slots on every linear (48 MiB a layer, computed from the released config; 1.9 GiB for two slots, 15 GiB for four at rank 64), charged to the stage that holds each layer; to be measured on a card.
  Not wired: the grouped correction of the linears that read one activation (q, k, v, g and gate, up: two dispatches per
  linear instead of per group, `AdapterServing::input_groups`), and KV-clip calibration warnings do not apply (no calibration
  yet). A sequence running under an adapter keeps its K and V in the adapter's namespace, so a prefix is shared among rows of
  the same adapter only.
- Training: norms and `value_res_lambda` (and `g_proj` as a full tensor) train alongside an adapter only in a full
  fine-tune. The adapter policy (`model::adapter`) has no "also train these tensors" notion: an adapter file holds
  factors only, and the engine and the serving path apply factors only. Adding it means a `modules_to_save` list in
  the adapter file's card, the tensors in the file (PEFT represents them as `modules_to_save`), the trainer's roles,
  the engine applying them and the continue path reading them: a change to the shared format, not to Ripple.
- Training: an adapter trains over an NVFP4 base only as the ModelOpt release stores it (nothing quantizes into
  NVFP4), and a full fine-tune needs a float base (f32 weights, AdamW on the device); layer-streamed and
  resident bf16/fp8 full fine-tuning (`Model::set_resident`) are not wired, so the 65B cannot be fully
  fine-tuned yet (a pipeline of float stages writes a full checkpoint, `Shardable::save_stages`).
- Training: on the CPU backend `matmul_dx` with a very large contraction (the head's, vocabulary 152k) is a
  naive strided loop, 25 s a step for a toy model, so toy tests use a 32-token vocabulary; a GPU does not
  have the problem.
- N-LSA is a trainer option (`--native-sparse`, `.native_sparse_attention(cfg, long_range)`) for models that
  declare it (Qwen3 and Ripple); its index space is the NoPE queries and keys as they are (no learned projection),
  so an indexer distilled for serving (D-LSA) is a separate artifact from the N-LSA model. Not measured at
  context lengths where it pays (the Qwen3 crossover is near 8k tokens); a sparse-budget pipeline stage is built but
  only the one-stage run is checked against dense.
- Real-weight training numbers on a card (tokens/s against the GH200 roofline, the two-card pipeline's bubble,
  N-LSA's crossover against the dense flash path at Ripple's head dimension) are step 8.

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
