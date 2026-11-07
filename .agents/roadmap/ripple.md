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
| D-LSA distillation | `model::lsa::distill` (host math) and the `qwen_lsa` tool | model-agnostic capture of a layer's Q/K, so a Ripple indexer can be distilled over the 16 global layers; device distillation (`lsa.md`) |
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
3. [ ] LSA lifted out of the Qwen3 engine into a shared engine-side
       component; Ripple's 16 global layers read sparsely; group-10 sparse
       kernel; gate: full-budget selection equals dense output.
4. [ ] D-LSA indexer capture and distillation for Ripple's global layers.
5. [ ] NVFP4 release loading (MLP W4A4, attention bf16), quantized KV.
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

## Open items for the later steps

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
