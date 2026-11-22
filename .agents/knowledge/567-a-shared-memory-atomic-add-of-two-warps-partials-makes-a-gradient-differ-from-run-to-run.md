<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 567. A shared-memory atomicAdd of two warps' partials makes a gradient differ from run to run

**Lesson:** a float `atomicAdd` sums in whatever order the contributions land, so a kernel that adds two or more
partials into one slot that way is not repeatable, even in shared memory and even with only two contributors:
`(x + a) + b` and `(x + b) + a` round differently. Gated DeltaNet's fused chunk backward
(`gdn_chunk_bwd_post_tc.cu`) added the row dots of a row tile's two column-half warps into `d_exp_g_cs`,
`d_decay_scale` and `d_beta`, and the decay mask's row and column sums into `d_g_cs`, with `atomicAdd`. The last bits
of those gradients, and of every gradient behind them, changed between runs of one binary, so a parameter-sharded
Qwen3.8-27B step's gradient norm did not repeat (step 1: 7.720721, 7.720708, 7.7207236 in three runs), against the
"repeatable" level the distributed plan promises. Each accumulator now keeps one slot per column half, written by
exactly one thread in program order, and the halves are summed in a fixed order after the block synchronises; the
row and column sums of the mask have slots of their own. 1 KiB more shared memory per block, no measured cost.

**How to apply:** a kernel on a training path that combines partials uses a fixed-order reduction (per-producer
slots, then an ordered sum; a warp shuffle tree; a block reduction indexed by warp), never `atomicAdd` on floats.
`grep 'atomicAdd' crates/kernels-cuda/cu/` before claiming a path repeatable; the NVFP4 decode products
`matmul_nvfp4_mmae.cu` and `fp8_nvfp4_gemv.cu` still meet their K-slices with shared atomics (a served NVFP4 decode is
therefore not bit-repeatable). Repeatability is tested by running the same inputs many times and comparing bits
(`gdn_mixer_tc` `the_tensor_core_backward_repeats_bit_for_bit` failed at its first repeat before the fix).

## Evidence (2026-11-22, n442, 4 GH200)

`qwen35` `measure_sharded_parameters_rank`, Qwen3.8-27B-FP8 cut to 4 layers, 4 ranks over NCCL, block 1024, window
2, running clip, 4 steps: four runs before gave four different norm sequences; three runs after gave
7.7207165 / 5.7851524 / 4.954092 / 4.6867476 every time. 8 layers, two-pass clip: two runs before differ at step 1
(8.5234375, 8.523462), two after agree (8.523543 ...); steps 2-4 took 1.12-1.16 s before and 1.11-1.19 s after.
Qwen3-4B (no GDN layer) parameter-sharded on 4 cards was already bit-identical across two server processes.
