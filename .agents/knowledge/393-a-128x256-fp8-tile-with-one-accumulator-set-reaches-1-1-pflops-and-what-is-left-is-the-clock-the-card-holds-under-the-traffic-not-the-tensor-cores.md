<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 393. A 128x256 FP8 tile with one accumulator set reaches 1.1 PFLOPS, and what is left below 1.2 is the clock the card holds under the traffic, not the tensor cores or the issue slots

Question: knowledge 387 stopped at 0.95 PFLOPS for the block-scaled FP8 GEMM (`matmul_fp8_wgmma_tma`, 128 x 128 tile, `wgmma m64n128k32`) and named the next step: an N of 256 per `wgmma` with the scale fold done on one accumulator set. What does it reach, and what bounds it?

Built (`matmul_fp8_wgmma_tma_n256`, `cu/matmul_fp8_wgmma_tma_n256.cu`): a 128 x 256 tile, two consumer warpgroups of 64 rows each issuing `wgmma m64n256k32` (128 accumulator registers, so 128 + 128 for a separate promoted sum do not fit). The 128-wide block scales are applied as a **ratio on the running sum**: the accumulator holds `sum / s_k` for the scale `s_k = sx[row, k] * sw[block, k]` of the last 128 of K, a block starts with one multiply of the 128 registers by `s_{k-1} / s_k`, and the tensor core adds the block's products straight into them; the epilogue multiplies by the last scale. A zero scale keeps the previous one (`!(s > 0)`), a row past M has scale 0 and never stores. The two 128-column weight tiles are two TMA loads into one 256-row swizzled tile; the output leaves through shared memory and a TMA store (16 x 16-float boxes, a third device-built map, 64-byte swizzle; `N` a multiple of 4), two boxes in flight per warp. Four stages of 48 KiB + 1 KiB scales, `setmaxnreg` 232 / 40, grouped tile order unchanged (group 16 was the best of 4..64).

Measured (gpu1, `make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_prefill_tflops"`, weights cycled past the L2, admitted by the tuner's rule on the ragged shapes; TFLOPS, `matmul_fp8_wgmma_tma` -> `_n256`):

| shape (K x N) | M = 768 | 2048 | 8192 | 31000 |
|---|---|---|---|---|
| gate/up 5120 x 34816 | 894 -> 1030 | 969 -> 1128 | 948 -> 1119 | 818 -> 981 |
| down 17408 x 5120 | 906 -> 1144 | 992 -> 1060 | 952 -> 1086 | 803 -> 1053 |
| qkv 5120 x 10240 | 838 -> 1034 | 912 -> 1117 | 951 -> 1091 | 928 -> 1108 |
| qkv 5120 x 12288 | 805 -> 859 | 931 -> 1137 | 952 -> 1102 | 878 -> 1057 |
| out 6144 x 5120 | 822 -> 987 | 906 -> 968 | 945 -> 1142 | 938 -> 1127 |

Accuracy: the tuner's error against the f64 reference of the stored weights is the input rounding error for both kernels (gate/up 2048 rows: rms 0.5928 vs 0.5969, input 0.5922); `qwen35_fp8_parity` (8 x 1500 tokens, 32 steps) on two texts: base 98.05 % / KL 2.51e-3 and 98.05 % / 1.74e-3, now 98.05 % / 3.16e-3 and 99.22 % / 1.50e-3 (no flip above margin 0.5): inside the run-to-run spread of the numbers earlier ledgers record. The tuner picks it for the large classes (10 of 96 classes; the 128 x 128 kernel keeps the small-N and 129-512-row classes, where 128 x 256 tiles fill fewer waves).

What bounds it (ncu, same binary, `fp8_one_candidate` at 8192 x 5120 x 34816, ablations by compiling work out; durations in ms, SM clock from `sm__cycles_elapsed.avg.per_second`):

1. **The tensor pipe is not the limit.** The same consumer loop with every load, barrier and store removed and zeros in shared memory runs 1.49 ms at 1.83 GHz (1.96 PFLOPS, 99 % of the 4096 MAC/clk/SM peak, the 128 multiplies per block fully hidden by the other warpgroup). With real operand bytes in shared memory, still no loads, it runs 1.90 ms at 1.59 GHz (1.74 without the rescale). **Cycles per tile, not instruction slots, are close to the floor: 77 % of ideal for the full kernel against 89 % for the loop alone.**
2. **The clock drops with the work around the tensor cores.** Full tile flow without the epilogue (loads on): 2.39 ms at 1.36 GHz; the full kernel 2.60-2.65 ms at 1.35 GHz. Zero data and no traffic hold 1.83 GHz, real operands 1.59, real operands with the TMA flow 1.36: the card is power limited (the module limit here is 660 W, `SW Power Capping` active) and every joule of L2-to-SM traffic, the rescale and the output stores is paid in clock. Lower-entropy weights (small exponents only) did not move the TFLOPS (1.14 PFLOPS at 2048 x gate/up), so it is the movement, not the tensor toggling.
3. **Latency-hiding changes did nothing.** Scale ratios computed while the previous block's `wgmma` run, one empty-barrier arrival a warp, a lagged second warpgroup, strictly alternating issue (`bar.sync`), scales fetched in 4 KiB chunks of 16-byte copies instead of 128 four-byte copies a stage (ncu counted ~117 bank-conflict wavefronts a stage on the 4-byte copies), half the weight bytes loaded, a 2-block cluster multicasting the weight tile (the L2-to-SM bytes down by a third; `mbarrier.arrive.release.cluster` compiles to `MEMBAR.ALL.GPU` and cost 33 %, the plain arrive is free): all within 1-2 % of 2.60 ms. Where the work is energy bound a stall that only idles a warp costs nothing.
4. **Where the remaining 2.60 against 1.90 goes** (ms, same conditions): rescale 0.2-0.27 (128 `FMUL` per thread per 128 of K, inherent to block-scaled FP8: one multiply per output per block), output epilogue 0.2-0.3 (1.14 GB of f32 written; a direct `STG`, a shared-memory-transposed `STG.128` and the TMA store are within 3 % of each other), and the load flow.

So 1.2 PFLOPS is not reachable on this card at this power setting by restructuring the loop; the 8-10 % each of the rescale and the f32 output are the next candidates (a bf16 output would halve the stores but changes the contract of every FP8 linear), and wave quantisation on the 768-row shapes (gate/up 6.18 waves -> 7, down/out 120 tiles of 132 SMs, qkv 12288 2.18 -> 3) needs a K split of the tail tiles.

Training: the FP8 GEMM is not on the NF4 training path; nothing to gradcheck.
