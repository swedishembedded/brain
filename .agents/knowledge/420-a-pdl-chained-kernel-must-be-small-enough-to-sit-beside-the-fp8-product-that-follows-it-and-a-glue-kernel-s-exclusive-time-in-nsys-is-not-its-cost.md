<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 420. A PDL-chained kernel must be small enough to sit beside the FP8 product that follows it, and a glue kernel's exclusive time in nsys is not its cost

**Lesson:** A kernel chained by programmatic dependent launch ahead of a persistent product must
keep its shared memory and register footprint small, or the product's block cannot sit beside it and
the wait moves to the product. Judge a decode change by the step period, because a kernel's
exclusive time in nsys charges overlap to the kernel launched first.

## What happened

Question (the brief of 2026-11-01): the decode step's norm kernels (`add_rms_quant_e4m3` 624 us,
`add_rms_quant` 356 us a step at 8K x 16, 9.6 and 5.4 us a launch) against vLLM's fused add +
rmsnorm at ~3.4 us. They run one 256-thread block a row, 16 blocks on 132 SMs.

A 1024-thread block a row (a float4 a thread, the squares exchanged through 20 KiB of shared memory
so the old chain order is kept bit for bit) took the isolated kernel from 4.65 to 4.05 us (ncu, warm
L2) and the step's exclusive e4m3 time from 611 to 280 us.

The decode-only step got **slower**: topk-to-topk period median 17934 -> 18273 us (nsys), served
median ITL unchanged (17.49 / 17.47 -> 17.40 / 17.46 ms), and the following
`matmul_fp8_wgmma_swap_tma_bn136c1` grew by the time the norm lost (5197 -> 5432 us exclusive).

## Why

- The FP8 product is launched with programmatic dependent launch while the norm still runs and keeps
  `8 x (64 x 128 + 136 x 128 + 528) + 1056` = 210 KiB of the SM's 227 KiB of shared memory for its
  weight ring. A norm block with 20 KiB of shared memory (or 1024 threads x 46 registers) cannot be
  resident beside it, so the product's block on that SM waits for the norm block to finish: the wait
  the early launch exists to hide moves to the product.
- The old 256-thread block with 32 bytes of shared memory fitted. So: **a kernel launched into the
  shadow of a persistent product keeps its footprint (shared memory under ~15 KiB, registers under
  ~20 K a block) small**, and gets its parallelism from more blocks, not bigger ones.
- nsys exclusive time (`step3.py`'s method, a kernel's span minus what overlaps an earlier kernel)
  charges the overlapped part of a PDL-launched kernel to the kernel launched first, so the sum is
  the step but the split is not a cost: a faster norm showed up as a slower product. **Judge a
  decode change by the step period** (the gap between two `brain_topk_rows` starts, decode-only
  steps) and by served median ITL, not by one kernel's exclusive column. The served cell's tok/s is
  mostly prefill and mixed rounds (8K x 16: median ITL 17.4 ms against a mean TPOT of 39 ms) and
  does not resolve a 2 % decode change.

## What to do

`add_rms_quant_e4m3_cluster`: a cluster of five 256-thread blocks a row (4 KiB of shared memory),
block k takes the 1024-element span k, a thread one float4; the squares go to shared memory, one
cluster barrier, and thread t of every block reads thread t's five float4s from the five blocks and
adds them in the old order (element by element, spans ascending), then the old shuffle tree and
eight-warp fold.

Bit for bit the one-block form's `xn`, E4M3 bytes and scales (a host model of the order is the
gate). `Fused::AddRmsQuantE4m3` picks the cluster form for <= 32 rows on a device with block
clusters and the one-block form otherwise (a launch of rows x 5 blocks is 1.4 x slower at 64 rows
and 2.5 x at 256, isolated). Kernel time (events, 20 back-to-back launches, 5120 wide): 16 rows 6.1
-> 5.5 us. Step period median at 8K x 16 (nsys): 17934 -> 17588 us.

### Rules

- A cluster kernel that reads a peer's shared memory ends with `barrier.cluster.arrive.release`
  after the peer loads and `wait` before exit. With a *relaxed* arrive the first version passed
  every gate at <= 16 rows and died with `unspecified launch failure` (Xid) at 130 rows x 5 blocks:
  a peer exited, and freed its shared memory, while a load of it was in flight. Only launches with
  more clusters than run at once showed it. The ignored
  `cluster_form_survives_back_to_back_launches` runs 500-2000 launches at 1, 16, 32 and 130 rows and
  2560/5120 wide.
- The f32 `add_rms_quant` (65 launches a step, the mixer fronts of the FP8 build) as a five-block
  cluster (a lane's 80 squares read through the cluster, 64 threads add each chain) is slower than
  the one block it replaces: isolated 8.3 against 6.6 us at 16 rows, step period 17642 against 17544
  us; not shipped. Its chain (80 + 64 dependent adds, kept for bit-identity with the WGSL chain) is
  the floor.
