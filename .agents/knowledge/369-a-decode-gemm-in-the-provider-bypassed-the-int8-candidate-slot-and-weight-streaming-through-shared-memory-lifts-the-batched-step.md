<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 369. A decode GEMM in the CUDA provider bypassed the int8 candidate slot, and weight streaming through shared memory lifts the batched 27B step

**Lesson:** A tuned candidate table is worth nothing if a hard-wired route sits in front of it, so
check the profile for the kernel names the table can choose. Weight streaming through shared
memory with a load structure built for the memory system lifts the batched decode step.

## What happened
Question: after the tuner of #368 found no candidate better than the hand choice, why did a
measured-better candidate never reach the model, and what does a load structure built for the
memory system buy?

Finding 1 (a defect): `CudaProvider` accepted every int8 request above `I8_GEMV_MAX_ROWS` (8 rows)
and ran the tiled `matmul_i8_mma_m16/m32` GEMM. At 9-32 rows the model therefore never entered the
`matmul_i8_gemv` slot the autotuner chooses in (the step profile at 20 streams showed only
`native:matmul_i8_mma_m32`).

## Why
The selector already said the GEMV family owns the decode regime on a device with a compute
capability (`select::i8_gemv_max_rows`), but the provider kept its own rule in front of the slot.

Finding 2: `cu/matmul_i8_stream.cu` streams weights, scales and x by `cp.async` through a ring of
128-byte-per-row stages (a warp loads four full cache lines per instruction).

- It feeds `mma.m16n8k32` through `ldmatrix` (XOR-swizzled tiles, no bank conflicts).
- It splits K over a thread-block cluster whose members add their partials in rank order through
  distributed shared memory (deterministic, no workspace; the cluster size comes from the launch,
  so one source serves splits of 1, 2, 4 and 8).
- Narrow (64 rows a block) and wide (128) variants are separate candidates.

Rows 17-32 stay at 30-60 %: each MMA costs a thread about 12 FP instructions of group-scale fixup,
so the kernel is issue-bound there. Stage count (3 to 8) changes nothing: the loads are not the
limit.

## What to do
The provider now uses the same rule as the selector (`i8_gemv_max_rows_at`). A tuned table is worth
nothing if a hard-wired route sits in front of it: check the profile for the kernel names the
table can choose.

Trap: the store key hashes the candidate sources, so adding candidates retunes; the grid with 12
candidates tunes in about 10 s. `brain roofline` honours `--device` since b0f04f5b5; measure with
`BRAIN_PIPELINE_CACHE_DIR` pointing at a fresh directory when the old store may have been measured
under load (a stale store reports "from the store").

## Evidence
GH200 idle card, device-timed, weights past the L2:
- large matrices (K 5120 x N 16384) 2.2-2.8 TB/s (60-77 % of the 3.66 TB/s read roof) against
  0.7-1.8 before;
- 5-16 rows at K 17408 x N 4096 2.1-2.4 TB/s.

Decode step (`qwen35_decode_profile`, 16 steps, kv int8, context 2048, same window, base binary
against new): 12/20/32 streams 496/615/850 tok/s -> 556/702/883 (best runs 573/732/956).
Run-to-run noise on this part is about 5 %. The 6-step runs showed outliers in one batch; use 16
steps.
