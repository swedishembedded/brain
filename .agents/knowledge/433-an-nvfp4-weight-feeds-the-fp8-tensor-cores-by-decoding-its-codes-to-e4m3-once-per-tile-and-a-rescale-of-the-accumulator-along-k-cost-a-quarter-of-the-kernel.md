<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 433. An NVFP4 weight feeds the FP8 tensor cores by decoding its codes to E4M3 once per tile, and a rescale of the accumulator along K cost a quarter of the kernel

Question (the brief of 2026-11-02): the NVFP4 products from 256 rows up decoded the tile to bf16 and called the library (17408 x 5120 gate/up at 8192 rows 2284 us against FP8's 1213; the 64-row build 253 us
against FP8's 41; the head at 64 rows 2.5 ms against 386 us). sm_90 has no FP4 tensor cores. What feeds the tensor cores directly from the 4-bit codes, at 33 to 8192 rows and at the 9 to 32 rows of a decode batch?

## The operand precision (stated, tested)

- E2M1 code x E4M3 block scale has up to five significant bits; E4M3 holds four. The weight operand is `RN_e4m3(code * scale / 8)`: one rounding to nearest even (relative error at most 2^-4, zero for the codes 0, 0.5, 1, 2, 4 and for
  half of the scale mantissas; 33 % of random weights are inexact, worst relative error 0.0476 measured), and the 1/8 keeps the largest product (6 x 448) inside E4M3's 448. The epilogue multiplies the 8 back with the tensor's f32 scale.
  bf16 would be exact, but runs the tensor cores at half the rate and needs twice the shared memory per tile.
- The activation is E4M3 with ONE f32 scale per row (`quant_e4m3_row`, amax / 448), not one per 128 of K. A scale that varies along K needs the whole accumulator multiplied by the ratio of two scales at every 128 of K: 64 to 128
  multiplies a thread a stage, which with the accumulators idle around them cost a quarter of the kernel (2018 us with the rescale and its dependent waits, 1532 us with the multiply removed, same tile, same decode). E4M3 is a
  floating-point format: the row's outliers do not cost the small elements their three mantissa bits, only the ones under 2^-9 x amax / 448.
- Tests: `nvfp4_wgmma_native` holds each kernel to the f64 product of exactly those operands (the wgmma GEMM to 2e-3 of the sum of absolute terms, the FP8 tensor core accumulates with fewer than 24 bits: worst 5.6e-4; the `mma.sync`
  builds accumulate in full f32: worst 1.9e-7), ragged shapes, every row written (NaN-poisoned outputs). `qops_nvfp4_wgmma` holds the model path to the decoded weight at 0.2 of the root of the sum of squared terms (two 2^-4 roundings).

## The kernels (`cu/matmul_nvfp4_wgmma.cu`, `cu/matmul_nvfp4_mmae.cu`, shared decode in `cu/nvfp4_e4m3_decode.cuh`)

- Decode: a table of 256 entries, one per scale byte, holds the E4M3 bytes of the eight E2M1 magnitudes times that scale / 8 (4 `mul.f16x2` + 4 `cvt.rn.satfinite.e4m3x2.f16x2`, built once a block); eight codes (one word) become eight
  E4M3 bytes with four `prmt` (the three magnitude bits of four codes are the selectors of four table bytes; the signs by two more permutes: a selector with bit 3 set replicates the selected byte's top bit, and the sign of an
  odd nibble is already a byte's top bit, of an even one after a shift by four) merged by xor. About 1.4 instructions a weight. The first version, unrolled over two chunks, ran at 11.5 weights a cycle an SM
  (a warp issues one dependent instruction every ~7 cycles with two warps per scheduler); fed to the FP8 consumers it bound the stage.
- `matmul_nvfp4_wgmma`: 512 threads, a persistent grid over 256 x 128 tiles. Two consumer warpgroups (`wgmma m64n128k32`, each over two 64-row blocks, 128 accumulator registers, `setmaxnreg` 168), two decoder warpgroups (88
  registers) that turn each stage's raw code and scale tiles (TMA) into the 128-byte-swizzled E4M3 operand in shared memory, and lane 0 of the first consumer warp as producer (so no warp is spent on loads and the registers go to the
  accumulators). A 256-row tile halves the decode per FLOP against 128 rows (each weight tile is decoded once per tile of rows: 32 times at 8192 rows). Ring of 3 stages of 58 KiB.
- `matmul_nvfp4_mmae{8,16,32}`: the bf16 `mma.sync` builds' structure (weight rows as the 16-row dimension, a lane owns a 32-weight chunk, eight K-slices a block) with `mma.m16n8k32.e4m3` and the same decode: 2 to 32 rows.

## Kernel times (gpu0, device events, weights cycled past the L2 below 129 rows, us; `kernel_times_against_fp8`, ignored)

| shape | rows | wgmma | bf16 mma (before) | E4M3 mma | best FP8 |
|---|---|---|---|---|---|
| gate/up 17408x5120 | 8 | 88 | 54 | 37 | 33 |
| | 16 | 112 | 65 | 53 | 35 |
| | 32 | 106 | 140 | 118 | 33 |
| | 64 | 87 | 254 | | 36 |
| | 256 | 100 | | | 70 |
| | 1024 | 260 | | | 166 |
| | 4096 | 872 | | | 633 |
| | 8192 | 1566 | | | 1271 |
| down 5120x17408 | 16 | 116 | 60 | 44 | 34 |
| | 32 | 117 | 112 | 75 | 33 |
| | 64 | 117 | 205 | | 35 |
| | 256 | 142 | | | 73 |
| | 1024 | 294 | | | 178 |
| | 4096 | 740 | | | 647 |
| | 8192 | 1514 | | | 1308 |
| lm_head 248320x5120 | 8 | 1011 | 624 | 461 | 375 |
| | 16 | 1809 | 1203 | 614 | 388 |
| | 32 | 960 | 1476 | 1455 | 384 |
| | 64 | 685 | 3547 | | 389 |
| | 256 | 905 | | | 652 |
| | 1024 | 3579 | | | 2488 |

Whole products as the model runs them (activation quantisation included, `qops_nvfp4_dense_shapes`, base = 69356a3c3 before this change, 1 row = the GEMV of the day), us:
gate/up 2 / 8 / 16 / 32 / 64 / 256 / 1024 / 4096 / 8192 rows: base 56 / 58 / 70 / 144 / 258 / 236 / 411 / 1183 / 2289, now 43 / 41 / 55 / 86 / 86 / 104 / 252 / 843 / 1615 (FP8: 36 / 38 / 42 / 38 / 42 / 67 / 178 / 600 / 1212).
down: base 56 / 57 / 65 / 117 / 214 / 233 / 430 / 1284 / 2387, now 42 / 40 / 48 / 81 / 126 / 156 / 352 / 927 / 1818 (FP8 40 / 39 / 43 / 39 / 45 / 84 / 208 / 689 / 1352).
lm_head 2 / 8 / 16 / 32 / 64 / 256 / 1024 rows: base 518 / 541 / 651 / 1364 / 2482 / 5843 / 15165, now 364 / 367 / 522 / 606 / 619 / 745 / 2683 (FP8 350 / 370 / 387 / 382 / 385 / 581 / 2092).

## Selection (`Ops::nvfp4_e4m3`, then the bf16 builds as the fallback)

2 to 16 rows (to 32 where the weight has under 96 tiles of 128 rows: the 5120-row down projection has 40, a block each, so the grid idles at a few rows) on the E4M3 `mma.sync` builds; from 17 (33) rows on the wgmma GEMM.
Not the autotuner's table: the candidates are three launch forms with three activation forms (f32 GEMV, bf16, E4M3 per row) and the tuner's grid, problem builder and oracle (`native_tune`) know int8 and FP8 weights only. Open.

## Parity (`qwen35_fp8_parity`, 8 x ~1300 tokens, 32 steps, int8 KV, the NVFP4 release against its own bf16 decode)

Before: 98.83 % greedy agreement, mean rel-L2 1.85e-2, mean KL 1.17e-3. Now: 99.61 %, 2.75e-2, **1.685e-3**, no flip above margin 0.5. The agreement is better, the KL is not: the E4M3 activations of the MLP cost what they cost the FP8 release
(FP8 against its bf16 decode: 97.92 %, 2.75e-2, 1.76e-3). The bar "no worse than 1.17e-3" is not met; meeting it needs bf16 activations, i.e. a bf16 `wgmma` (half the rate) and a 2x tile.

## Served (gpu0, bf16 KV, f32 state, no speculation, two interleaved rounds each, base = b79ecf10c)

| cell | tok/s base -> now | mean TTFT ms | mean TPOT ms |
|---|---|---|---|
| 8K x 1 | 67.2 -> 71.5 | 936 -> 700 | 11.3 -> 11.3 |
| 8K x 16 | 218.5 -> 260.7 | 4805 -> 3976 | 54.1 -> 45.5 |
| 32K x 16 | 62.7 -> 74.4 | 18638 -> 15527 | 180.3 -> 152.3 |
| 8K x 64 | 241.9 -> 307.8 | 15325 -> 12637 | 200.2 -> 155.6 |

vLLM 0.32 (matched run of 2026-11-01): 74.9 | 783 | 10.3, 266.5 | 4374 | 42.8, 70.5 (32K x 16). Stress (`back_to_back_launches_are_deterministic_and_do_not_fault`, ignored): 20 launches a batch at 4096 x 17408 x 5120, 2048 x 5120 x 17408, 300 x 4098 and 33 x 62, 40 s each,
outputs poisoned between batches, every batch bit-identical to the first; `dmesg | grep -ci xid` unchanged across the stress and the four served runs.

## Traps

- A TMA box of 16 bytes (the scale tile, two stages' worth of scale bytes) must START on a 16-byte boundary in the global row: coordinate `8 * kt` is an illegal instruction at the second stage. The stage loads the aligned window `(kt / 2) * 16`
  and the decoder reads `8 * (kt & 1)` into it. Bisecting by removing one load at a time found it (ptxas and the tensor-map test cannot).
- `wgmma` groups serialise ("C7520/C7514", every group followed by a wait) when a divergent branch encloses the issue (a conditional product per block, or the producer lane's branch: make the role a `__shfl_sync` result) and when two
  groups on DIFFERENT accumulators are kept in flight with `wait_group 1`; groups on the SAME accumulators pipeline (`wait_group 1` after each stage's commit, the previous stage's slot released behind it). The no-rescale design is what
  made that legal.
- The 128 x 128 tile with 8 decoder warps spent 570 us of 1895 on decode that the 8 consumer warps could not hide; the same decode behind a 256-row tile is hidden (1831 -> 1537 with the row scale).
- Two worktrees sharing one cargo target dir overwrite each other's `brain-gpu-core` and `kernels-cuda` artifacts (a stale rlib without the new symbols): give the base worktree its own target.

## Open

- The autotuner: candidates (`gemv_m1/m2/m4`, `mma8..64`, `mmae8..32`, `wgmma`) timed per shape class instead of the thresholds above (`NVFP4_WGMMA_MIN_ROWS`, `NVFP4_WGMMA_FILL_TILES`), which are measured but fixed.
- 16 rows is still 53 us (gate/up) against FP8's 35, and the 5120-row down projection fills 40 of 132 multiprocessors in the wgmma GEMM (117 us at 33 to 64 rows against FP8's 35 to 45): split-K or a 64-row tile.
- Prefill at 8192 rows: 1566 us against FP8's 1271 (the tile rate of the 128 x 256 FP8 GEMM is higher than 256 x 128: the consumers wait for the decode and the ring is 3 stages).
- The silu product and the norms still write f32 for NVFP4 readers (`E4m3Readers`): a fused E4M3 row-scaled producer would spare the separate `quant_e4m3_row` pass (~5 % of a 8192-row product).
- `kernels-cuda` `the_shipped_registry_is_well_formed` fails on the base for the `matmul_nvfp4_gemv*` and `matmul_fp8_gemv*` entries (they share a source with the `mma.sync` builds and declare min_cc 5.0).
