<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 407. The FP8 prefill GEMM's gap to the reference was tensor-pipe idle time around the block scales, not L2 traffic or the epilogue, and reading them in the consumers a block ahead closed most of it

Question (knowledge 393, the brief of 2026-11-01): `matmul_fp8_wgmma_tma_n256` runs 1.1 PFLOPS at 4096 rows where the reference (DeepGEMM `sm90_fp8_gemm_1d2d`, 256 x 128 tile, 3 stages, a 2-block cluster multicasting one operand) runs 1.25-1.29. The same-clock ncu comparison showed 42 % more instructions and 31 % more L2 bytes. Which of the two costs cycles?

All numbers: gpu1, 4096 x 5120 x 34816, ncu `--clock-control base` (SM 1.14-1.19 GHz), `fp8_one_candidate`. The wall-clock probe moves 3-10 % between runs while the other card runs a benchmark (power and clock), so a change is judged on ncu cycles and counters, not on the probe alone.

## What the counters say

| | brain 128 x 256 | reference |
|---|---|---|
| duration at ~1.17 GHz | 1.49 ms (1.37 after items 1 and 2; elapsed cycles 1.80 M -> 1.59 M) | 1.30 ms (1.56 M cycles) |
| `sm__pipe_tensor_cycles_active` (of elapsed) | 76.7 % (85.1 % after item 2) | 88.6 % |
| executed instructions | 402 M | 284 M |
| `wgmma` instructions | 5.57 M (m64n256k32) | 11.1 M (m64n128k32) |
| L2 bytes | 9.5 GB | 7.1 GB |
| shared-memory bank-conflict wavefronts | 14.1 M (cp.async scales 8.0, epilogue stores 6.1) | 0.8 M |
| stall samples in `WARPGROUP.DEPBAR` / FMUL-or-FFMA | 33 % / 24 % | 39 % / 26 % |

The two consumer loops are the same shape (4 chained wgmma, wait, 64-128 multiplies of the accumulators per block, wait on the next stage): the reference also serialises a warpgroup's promote after its products. The difference is in what the tensor pipe does between them.

Ablations on the same kernel (ncu cycles, base clock, results rejected by the admission rule where they drop work):

| variant | duration | SM active cycles |
|---|---|---|
| shipped before this change | 1.49 ms | 1.715 M |
| no scale logic at all (no scale loads, ratios or accumulator multiplies; products, TMA flow and epilogue kept) | 1.38 ms | 1.545 M (tensor 89 %, the reference's figure) |
| epilogue stores removed (on the cluster variant of item 3, 1.48 ms / 1.70 M with them) | 1.49 ms | 1.718 M |
| only the 128 accumulator multiplies removed | 1.38 ms | |

So the scale path (loads, ratio, 128 multiplies, the fences around them) costs 11 % of the cycles and the f32 epilogue none in this loop, and a loop with no scale work sits at the reference's tensor utilisation. The reference's promote costs it close to nothing for the same 128 FFMA per 256 columns of a block; what it does with the 128 FFMA in the shadow of the other warpgroup's products is not something the instruction mix explains (issue slots busy: 43 % brain, 35 % reference).

## What was tried

Kept in two commits: SM active cycles 1.715 M -> 1.648 M (instruction diet) -> 1.550 M (scales from global memory), elapsed duration at the base clock 1.49 -> 1.37 ms against the reference's 1.30 ms; in served 4096-row terms gate/up 1306 -> 1237 us against the reference's 1134-1175 us on the same card.
1. **Instruction diet (kept).** `rcp.approx.ftz` instead of `__fdividef` (denormal rescaling is ten instructions a quotient and can not apply to a product of two normal scales), the stage's four wgmma descriptors as the first one's plus constants, and the warpgroup role through a shuffle so the descriptor arithmetic stays in the uniform datapath: 402 M -> 330 M instructions, SM active cycles 1.715 M -> 1.648 M (-3.9 %). The consumer loop is now ~108 -> ~70 instructions besides the 128 multiplies.
2. **Block scales loaded by the consumers from global memory, one block ahead (kept).** The producer staged each stage's 128 activation and 2 weight scales through 4-byte `cp.async` (8 M bank-conflict wavefronts, 33 arrivals on the full barrier, 1 KiB of ring a stage) and the consumers read them back from shared memory before every block's ratio. Now each consumer thread loads its own four scales (two rows, two weight column blocks) into registers one block before use, and the next tile's first block before the epilogue; the barrier is the TMA's alone and the stage is the two tiles. Bit-identical sums, 351 M instructions (more than the diet alone: 4 loads and the prefetch bookkeeping) but 1.648 M -> 1.550 M active cycles, because the work left the producer's critical path and the shared-memory data pipe. The producer-side preparation of item 5 failed for the same reason it is a producer job: its loads and stores sit between the stage's release and its fill.
3. **2-block cluster multicast, fence-free (not kept).** Each block of a pair loads its own 128 rows of x and half of the 256 weight rows, the tensor memory accelerator multicasts the half to both (`cp.async.bulk.tensor ... .multicast::cluster`, mask 0b11), a stage is rewritten only after the consumers of BOTH blocks released it: each consumer warp arrives on its own empty barrier and on the peer's through `mapa` + `mbarrier.arrive.shared::cluster` (no `.release.cluster`, so no `MEMBAR.ALL.GPU`; one arrival a warp, count 16), and all threads end with a cluster barrier so no block leaves while the peer may still arrive on it. It runs, passes the admission rule (ragged M and N, odd row-tile counts: a phantom second tile is zero-filled and never stored) and cuts the L2 bytes 9.44 -> 7.94 GB (reference 7.13), and the duration does not move (1.47 -> 1.48 ms, probe within noise, better on some shapes, 5-15 % worse on others). L2-to-SM traffic is therefore not what limits this kernel at this size; a candidate that ties the base would only add tuning time and a registry row, so it is not committed.
4. **Warpgroups taking turns on the tensor core** (named barriers 1 and 2 of 256 threads: warpgroup A scales its sums only after B has issued, so A's multiplies overlap B's products): 1.48 ms, unchanged. The loss is not the two warpgroups scaling in lockstep.
5. **Scale ratios computed by the producer** (it reads `sx`, `sw`, keeps the previous scale per row and half, and writes ratios and scales to the stage; consumers only load four ratios): -11 % instructions but +10 % cycles (the consumers wait three times as long on the full barrier). The producer's global loads sit on the stage's critical path.
6. **Ratios of the next block computed (and its barrier waited) before the wgmma wait**: 1.69 ms, +15 %. The early wait delays the stage release, the refill and with it the next block.
7. **Splitting the 256 columns into two accumulator halves with their own commit groups** (`m64n128k32` x 2, `wait_group 1`, so one half's multiplies overlap the other half's products inside a warpgroup): correct, but ptxas reports C7514 (serialises every wgmma: "non wgmma instructions reading accumulator registers of a wgmma between start and end of the pipeline stage") however the loop is peeled, and the serialised kernel is 30 % slower. It is a compiler analysis limit for loop-carried `wait_group N>0`, not a hardware one; peeling the first block and fencing the operands did not change it.

## Rules

- A GEMM that must be measured to a few percent is measured in ncu cycles at the base clock (instructions, active cycles, tensor-pipe percentage), with the reference profiled the same way; the wall-clock probe is for the final table.
- A multicast that cuts the L2 bytes by a sixth is not a speed-up unless the L2 is what the kernel waits for. Check `sm__pipe_tensor_cycles_active` before building the cluster protocol.
- Anything the producer computes for the consumers adds its latency to the stage's fill; scale preparation belongs on the consumer side or in a prior kernel.

## Open

What is left to the reference is 5 % of the duration at the base clock (1.37 against 1.30 ms) and the multiply-and-wait structure itself (why the same 128 accumulator multiplies a block hide better behind its 256 x 128 tile of two 64-row waves). Candidates not yet tried: the reference's 256 x 128 tile (two 64-row waves a warpgroup on `m64n128k32`, 64 + 128 accumulator registers, FFMA promote with the block scale of the activation row only), a single named barrier and one store fence for the whole epilogue (the reference does one `STSM` batch, one fence and one TMA store a tile, we fence 16 times), and the multicast again on top of the scale change.

Training: the FP8 GEMM is not on the NF4 training path; nothing to gradcheck.
