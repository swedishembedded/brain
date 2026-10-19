<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 329. Which 27B inference GEMMs run on tensor cores, and what turning the library tier on would change

Qwen35's inference `Ops` is built pinned to `GemmPrecision::Fp32Reference` (`Ops::new`); only a trainer that states a
precision (`TrainOpts::gemm_precision`, default Auto) turns the vendor-library tier on. For the served 27B (Q8_0
GGUF, `TierPolicy::uniform(I8)`, int8 paged KV) that means:

| work | kernel | tensor cores today |
|---|---|---|
| every projection and MLP linear, prefill rounds (int8 weights, dynamically int8-quantised activations) | `matmul_i8_mma` (`_m16`, `_m128` tiles) | yes (int8 `mma.sync`) |
| the same linears at decode batch 1-8 | `matmul_i8_gemv`, `_gemv_multi` | no: dp4a on CUDA cores, memory bound; from batch 16 `matmul_i8_mma_m16` |
| paged flash prefill, head_dim 256 | `flash_prefill_f16_hd256(_kv8)` (version 2 opt-in) | yes (fp16 `mma.sync`, fp32 accumulate) |
| GDN chunk recurrence and prep (prefill) | `gdn_chunk_loop_tc`, `gdn_chunk_prep_tok_tc` | yes (three-pass TF32) |
| paged flash decode | `paged_flash_decode_gqa_hd256` | no: fp32 CUDA cores |
| GDN decode step, norms, activations, quant epilogues | native/portable elementwise | no (not GEMMs) |

Turning the library tier on for inference would change nothing for the int8 linears: `library_weight` serves only
bf16 and NF4 weights, so an I8 weight never reaches the library. It would (a) route any f32-weight linear above the
decode regime through TF32 (none exist under a uniform I8 policy; `--qwen35-tier q4,in_proj_a.weight=f32,...` would have
them), and (b) make the Hopper int8-weight bf16-activation kernel (W8A16, `matmul_i8w_bf16a_wgmma`, reverted from main)
selectable for prefill-sized rows: it needs `gemm.is_active()` for the bf16 cast of the shared activation. A
"wgmma on" run with the tier off ran the int8 `mma.sync` kernels, so window5's identical on/off tables measured
nothing; the engine-level speed and accuracy of W8A16 are NOT yet measured in this tree (the window9 run script lives outside the repository; no number is recorded here) (decision input: library
tier and W8A16 by default for the 27B, correctness over speed).
