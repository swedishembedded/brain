<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 486. On AArch64 every native CPU route was gated on AVX2, so each f32 GEMM ran as the per-element JIT loop

**Lesson:** A capability gate named after one ISA ("is AVX2 present") is a gate that every other ISA fails. Gate a
fast path on the capability it needs (a vector tier for the f32 GEMM), give each ISA its implementation of it, and
measure the backend on the hosts it actually runs on, not only on the one the fast path was written for.

## What happened

`backend_cpu` routes the GEMM-shaped kernels (every `matmul*`, `matmul_dx*`, `matmul_dw*`, the GQA attention
trio and its backward) to native host code, and decided whether to do so with `fast_conv::avx2_available()`. On a
Grace (Neoverse V2, 288 cores) host that is false, so the JIT kept every kernel it could compile and each GEMM ran as
one invocation per output element. A 5.3M-parameter Ouro (hidden 256, 4 layers, 4 passes) trained at 71 tokens/s on
rows of 1024, batch 8 (`samples/models/ouro`, `--kernel-profile`: the attention backward and `matmul_dw` at the
top). The native routes that did run there (the tiled GEMMs the JIT cannot compile) fell back to a scalar dot
product, and the GQA backward natives walked a per-row axpy over the materialized scores on one core per head.

## What changed

- `fast_ops::neon_gemm`: a 4x4-tiled NEON f32 GEMM (`C (+)= A·Bᵀ`) with `k` in panels, rows split over tasks
  sized by the work (a fan-out to all 288 threads cost more than a per-head attention product), columns split for a
  GEMV, bit-identical to its unsplit form; `matmul_abt`, `matmul_dx` and `matmul_dw` reach it on AArch64 (the
  backward ones through one transpose per call instead of streaming the whole activation once per output row).
- Native routes are taken whenever the host has a vector tier for the f32 GEMM (AVX2 or NEON).
- The GQA natives are threaded over heads, and the backward ones are per-head GEMMs (`dS·K`, `dSᵀ·Q`, `Pᵀ·dO`,
  over the causal part only, as the kernels sum it) instead of per-row axpy loops.
- Same machine, same seed (`samples/models/ouro`, step rate after the first step, other tenants active): the
  default size at rows of 1024, batch 8, 71 -> 2365 tokens/s; the 1.3M size at rows of 512, 147 -> 4900. The step
  losses agree with the JIT run to four decimals; the NEON GEMM is gated against an f64 reference
  (`neon_gemm::tests`), the attention natives against their scalar references.
- Every element is reduced in one order (one 4-lane FMA chain, a lane sum, the scalar tail) whether it falls in a
  full 4x4 tile or on an edge, each edge shape its own instance of the tile. The first version reduced edge elements
  through a separate four-accumulator dot, so a GEMV (`n == 1`, every element an edge) differed in the last bit from
  the same column of a GEMM, which `hostmath::linear_rows` must equal exactly
  (`linear_rows_is_bit_identical_to_a_per_row_matvec_loop`).
