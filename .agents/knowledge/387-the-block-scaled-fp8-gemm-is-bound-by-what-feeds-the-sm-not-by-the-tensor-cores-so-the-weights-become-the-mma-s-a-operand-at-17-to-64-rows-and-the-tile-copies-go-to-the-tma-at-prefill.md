<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 387. The block-scaled FP8 GEMM is bound by what feeds the SM, not by the tensor cores, so the weights become the MMA's A operand at 17 to 64 rows and the tile copies go to the TMA at prefill

Question: knowledge 386 left the FP8 GEMM family at 1.2-1.7 TB/s for 17-64 rows (`dq_stream_r2` / `matmul_fp8_wgmma`, 12 + 7.8 ms of a 34 ms batch-64 decode
step) and at 616-970 TFLOPS for prefill (`matmul_fp8_wgmma`, 128 x 128 tile, `cp.async` ring, one `__syncthreads` a stage). What bounds each, and what moves it?

Measured (gpu1, `make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_batch_and_prefill_rate"`, weights cycled past the L2,
device events, so every figure carries the ~6 us launch floor of the event timing; ncu on the same binary with `fp8_one_candidate`, `--clock-control none`):

1. **17-64 rows: the 128-row x tile was half padding and the grid did not fill the card.** At 64 rows `matmul_fp8_wgmma` runs one 128-row tile of x per 128 weight
   rows: 272 blocks of one block an SM on 132 SMs (2.06 waves, so 3) for the 34816-row gate/up, 80 blocks for the 5120-row down projection. The swapped form
   (`matmul_fp8_wgmma_swap_*`, `cu/matmul_fp8_wgmma_swap.cu`) makes a warpgroup own 64 *weight* rows (the MMA's M) against the 32 or 64 rows of x (the MMA's N,
   `wgmma m64n64k32`, 32 accumulator registers): no padding, 64 weight rows a block and three blocks an SM, the weight copies of the first stages issued before
   `griddepcontrol.wait`. The K = 17408 / N = 5120 shapes get a K split over a cluster of 2 or 4 blocks summed in rank order through distributed shared memory
   (deterministic, 18.5 us instead of 26 at 64 rows for 6144 x 5120).

   | 64 rows, shape (K x N) | before (us) | after (us) | after, TB/s |
   |---|---|---|---|
   | 5120 x 34816 | 89.1 | 73.9 | 2.4 |
   | 5120 x 17408 | 57.5 | 36.7 | 2.4 |
   | 17408 x 5120 | 85.7 | 35.6 | 2.5 |
   | 5120 x 10240 | 30.6 | 25.6 | 2.0 |
   | 6144 x 5120 | 34.3 | 18.5 | 1.7 |

   Served (FP8 27B, int8 KV, tiny prompts, 384 outputs): median ITL at 32 / 64 sequences 24.05 / 33.23 -> 18.15 / 27.33 ms, output tok/s 1255 / 1786 -> 1636 / 2130.
2. **The ceiling of the `cp.async` forms is what the SM ingests, ~5.5-6 TB/s, not the HBM.** The swapped kernel at 64 rows streams the weights at 2.4 TB/s *and* the
   x tile (one block re-reads all 64 rows of x per 64 weight rows: as many L2 bytes as the weights); 128-weight-row blocks (x bytes 0.5 of the weights') and deeper
   rings were not faster. Prefill at 128 x 128 x 128 per stage is 128 FLOP per byte copied: 650-700 TFLOPS is 5.2-5.9 TB/s of `cp.async` traffic.
3. **A warp-specialised `cp.async` pipeline did not move prefill** (producer warpgroup, two consumer warpgroups, `cp.async.mbarrier.arrive.noinc` full barriers and
   an empty barrier instead of `__syncthreads`: 620-700 TFLOPS, the same as the barrier kernel); a 256 x 128 tile (two 64-row wgmma a consumer, 128 + 64 accumulator
   registers under `setmaxnreg`) was slower; ncu `--clock-control none` of the specialised kernel at M = 2048 gate/up: 1.09 ms, 0.99 without the scale fold,
   0.68 with the tile copies removed. The data path, not the barrier or the fold, was the limit.
4. **The tensor memory accelerator is the data path that fits.** `matmul_fp8_wgmma_tma`: a persistent grid (at most 132 blocks), one producer thread issuing two
   `cp.async.bulk.tensor.2d` loads a stage (the 128-byte swizzle and the zero fill of rows past the edge are the tensor map's), two consumer warpgroups on
   mbarriers, the producer running ahead across tile boundaries, tiles walked in groups of 16 row tiles so the L2 keeps the operands live. brain launches kernels with
   plain pointers, so a block builds its two maps on the device: the driver's encoding of everything the shape does not change is the kernel's 16-word
   `BRAIN_FWT_TEMPLATE` (held to `cuTensorMapEncodeTiled` by `backend-cuda/tests/tensormap_template.rs`), and `tensormap.replace` sets the global address, the two
   dimensions and the row stride (in BYTES: a stride of `bytes >> 4`, which the replace instruction's documentation suggests, decodes wrong and the result is garbage
   with no fault), `tensormap.cp_fenceproxy` publishes it, one slot per SM (a block owns its SM, so no two live blocks share a slot).

   | TFLOPS (us) | M = 512 | 768 | 2048 | 8192 |
   |---|---|---|---|---|
   | gate/up 5120 x 34816, before | 664 (275) | 696 (393) | 661 (1105) | 602 (4849) |
   | gate/up, TMA | 842-866 (211-217) | 880-910 (301-312) | 927-942 (775-788) | 840-940 (3100-3480) |
   | down 17408 x 5120, before | 487 | 715 | 738 | 570 |
   | down, TMA | 615 | 899-938 | 947-967 | 731-890 |

   31K-token TTFT 4.46 -> 4.11 s (served, same client); 16 / 64 sequences 525 / 712 -> 545 / 800 output tok/s.
5. **Not reached: 1.2 PFLOPS.** Without the fold the TMA kernel runs 1.08 PFLOPS (675 us against 759 at 2048 x 5120 x 34816), so the loop is not tensor-bound
   either: `wgmma m64n128k32` reads 2 KB of A and 4 KB of B from shared memory for 64 cycles of tensor work, ~96 of the 128 bytes a cycle the port delivers, before the
   TMA's own writes. Two accumulator sets that overlap the fold with the next stage's wgmma spilled (192 accumulator registers) and ran 15 % slower. The next step is
   an N of 192 or 256 per wgmma (fewer operand bytes a FLOP) with the scale fold done on a single accumulator set (pre-scale the carried sum by the inverse of the block
   scale, accumulate in the tensor core, rescale: no second register set), at the cost of bit-identity with the current kernel.

Training: `matmul_fp8_wgmma*` is the FP8 serving linear; the W8K NF4 train probe (NF4 base, bf16 compute) does not dispatch it. The probe is reported in the ledger.
