<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 478. A fused weight saved launches on the GEMV path and cost three copies of its output on the library path

**Lesson:** concatenating weights (`wq;wk;wv`, `gate;up`) so one product serves several linears pays only where the
product streams the weight once per row and the launch count is the cost. On a library GEMM over many rows the product
is compute-bound either way, and the wide output then has to be copied apart (`concat_split`): one read and one write
of the whole output per part, for nothing. Give each part its own product over a sub-range of the fused weight, into
the buffer its reader takes, and share the activation's bf16 cast; and produce the next product's bf16 operand in the
pass that computes it (`silu_mul_bf16`) instead of casting it in a pass of its own.

## What happened

After knowledge 476 the Qwen3-4B serving engine (bf16 weights, int8 KV, `perf run serve --input 8192 --output 128`,
GH200) spent 12.2 ns of glue kernels per prefill token, more than a third of it in `concat_split`: per layer of an
8K-row chunk the gate/up split read and wrote 1.27 GB and the QKV split 0.4 GB, then `silu_mul` read the two halves
again and the down product's `cast_f32_bf16` re-read the whole fp32 activation. The engine fused the weights in
`Engine::build` for the GEMV decode path; since the bf16 tier (knowledge 468) every product of one row or more runs on
the vendor library (`LIBRARY_MIN_ROWS_BF16`), so the copies ran in every prefill and every decode step.

## What to do

- `Engine::linear_parts`: on the bf16 library path a fused weight is one product per part over its sub-range
  (`library::Operand::range` in words, two bf16 a word), writing `q`/`k`/`v` and `gate`/`up` directly, after one cast.
  Adapter corrections and a QKV bias address the fused output's columns, so those keep the fused product and the split.
- `Engine::swiglu_into_bf16_down`: `silu_mul_bf16` writes `h` and the down product's bf16 operand in one pass, and the
  product reads it with `forward_bf16`.
- Gate: `crates/qwen3/tests/it/engine_glue.rs` (no `concat_split`, no `silu_mul` in a bf16 library prefill or decode
  tape); it fails before the change.
- The split products run other library kernels (n of 1024 for k/v) than the fused one; measured, the decode step's
  kernels got faster, not slower (below), so no row-count threshold was needed.

## Evidence

Same command, gpu3 of a GH200 node, nsys `-t cuda,nvtx --cuda-graph-trace=node`, glue = `concat_split`, `silu_mul*`,
`cast_f32_bf16`, `qknorm_rope_fused`, `add_rmsnorm_fwd`, `kv_append*`, norms, embedding and argmax kernels, attributed to
the NVTX phase their launch ran in:

| | before | after |
|---|---|---|
| c=16: req/s, TTFT p50 / p95 (no profiler) | 2.75, 524 / 4262 ms | 3.07, 493 / 3626 ms |
| c=64: req/s, TTFT p50 / p95 (no profiler) | 3.08, 10.5 / 17.2 s | 3.46, 9.3 / 15.2 s |
| c=64: prefill kernel time, glue per prefill token | 33.5 s, 12.2 ns | 27.8 s, 5.8 ns |
| c=64: decode step kernel time, glue per step | 13.4 ms, 2.06 ms | 12.9 ms, 1.22 ms |

`serve_conformance`, `engine_exact_decode`, `engine_half_weights`, `engine_prefill_mixed`, `engine_prefill_hd128`,
`lsa_engine`, `engine_adapter_bank`, `engine_variants`, `engine_score_continuation`, `engine_rollout`, `lora_serve_fold`
and the `serve::` unit tests pass on CUDA; `batched_prefill_matches_per_token_decode` fails as before (knowledge 476).
