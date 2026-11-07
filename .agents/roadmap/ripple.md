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

1. [ ] `crates/ricursive`: config, `ARCHS` row, importer that refuses
       missing/unknown tensors, f32 forward on any backend. Gate: a
       random-weight tiny config (2 local + 1 global, window smaller than
       the prompt) matches `modeling_ricursive.py` logits; real-weight
       per-layer parity on layers 0 (local) and 4 (global) plus embed/head,
       fetched by tensor (no full 130 GB download: the project drive has
       ~60 GB free).
2. [ ] Paged hybrid engine under the generic `Scheduler`: ring KV for local
       layers, paged KV for global ones; serving conformance suite; decode
       equals full-recompute logits across the window boundary; prefix
       reuse.
3. [ ] LSA lifted out of the Qwen3 engine into a shared engine-side
       component; Ripple's 16 global layers read sparsely; group-10 sparse
       kernel; gate: full-budget selection equals dense output.
4. [ ] D-LSA indexer capture and distillation for Ripple's global layers.
5. [ ] NVFP4 release loading (MLP W4A4, attention bf16), quantized KV.
6. [ ] Backward, gradcheck, LoRA; N-LSA after D-LSA.
7. [ ] CUDA parity and the measured long-context numbers (both GH200s,
       when free).

## Measured

Nothing yet.
