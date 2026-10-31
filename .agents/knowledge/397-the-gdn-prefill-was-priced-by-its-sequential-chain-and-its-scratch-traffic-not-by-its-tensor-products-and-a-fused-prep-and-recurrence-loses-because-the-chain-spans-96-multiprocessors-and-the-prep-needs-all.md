<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 397. The GDN prefill was priced by its sequential chain and its scratch traffic, not by its tensor products, and a fused prep and recurrence loses because the chain spans 96 multiprocessors and the prep needs all of them

Question: the 27B's Gated DeltaNet prefill (prep + recurrence) cost 0.146 us/token/layer at 8K tokens (1.2 ms); the reference server's single fused
kernel is ~0.04. What moves it, GH200, one 8K segment (48 heads x 128 chunks of 64) and a 700-row round of three segments?

Measured, us a layer at 8K (before -> after): prep 588 -> 239, recurrence 610 -> 228, convolution 323 -> 195; prep + recurrence 0.146 -> 0.057 us/token.
700-row round: prep 67 -> 30, recurrence 60 -> 33 (0.19 -> 0.09 us/token).
- Moving the prep's 64x64 solve onto tensor cores (blocked forward substitution over four 16-row blocks, fp32 diagonal solves, bf16 hi/lo products with
  the accumulator pair repacked as the next block row's A operand) alone gave 591 -> 564 us. The instruction profile said why: 40 % of the prep's
  instructions were the L2 normalisation of query and key, done three times (once per value head of a key head) and stalled on global loads. Normalised
  once in the convolution (eight channels a thread, 16-byte accesses), the prep copies the rows by `cp.async`: 564 -> 323.
- The recurrence is a chain of 128 dependent chunks per head. One block of two warpgroups a (segment, head) is 48 multiprocessors of 132; a block a 64-column
  half is 96 and halves each block's tensor work and u bytes: 394 -> 282. Folding the row scalings into the recurrence (on the query product's accumulator
  and on the bf16 operand of the state update) removes two of five scratch tensors; u as fp16 in the order the threads hold it (four 16-byte stores in the
  prep, one contiguous copy in the recurrence) halves the largest: 282 -> 242. Issuing the next chunk's staging, the state's decay and the output store
  under the wgmma groups: 242 -> 228. The per-chunk chain is ~1.8 us (3300 cycles) of dependent products, packs and barriers; the products themselves are
  ~35 % of it.
- A fused kernel (prep for chunk j + 1 beside the recurrence for chunk j, scratch in shared memory) was costed and not written: the recurrence has 96
  chains, the prep 6144 independent blocks at three an SM. Run inside the chains, the prep would cost ~6 us of an SM a chunk against the chain's 1.8, so 128
  chunks x ~8 us on 96 multiprocessors is ~1 ms, four times the two-kernel sum. What fusion would buy is the ~0.7 GB the scratch moves through DRAM (it is
  250 MB at 8K, five times the L2); that needs the prep running concurrently on other multiprocessors with flags between roles, not a per-head kernel.
- A clock64 trace of one prep block alone on a multiprocessor: 12.9 k cycles, 1.9 k to the gates (global-load latency), 2.5 k the Gram products and
  decay epilogue, 2.4 k the diagonal solves, 5.5 k the block-row products and stores. A round of 12 chunks x 48 heads is two waves of ~14 us blocks:
  the 700-row round is bound by the block's latency, not by its bytes (the 320-row recurrence is 6 us of fixed cost plus 2.2 us a chunk).
- Wrong turns: a block that keeps the prep's scaled copies and staged them with a global load on the critical path (tid 0 waiting on g_cs each chunk) was
  slower than the unfolded form until the decay vectors came through the copy; launch bounds for three prep blocks an SM were worth 10 %, the epilogue
  and A-load micro-optimisations nothing (the kernel was latency, not issue, bound).

Quality: the prep against the exact TF32 form (relative rms) u 2.1e-3 -> 1.5e-3, w 2.9e-3 -> 2.4e-3, intra 3.0e-3 unchanged; the tuner's end-to-end
error 4.0e-3 -> 3.7e-3 (limit 7.8e-3); the folded recurrence with fp16 u against the `mma.sync` one 1.7e-3 (output) / 2.1e-3 (state). `qwen35_fp8_parity`
(8 x 1500 tokens, 32 steps, two texts) base 98.83 % / KL 1.73e-3 and 98.44 % / 2.07e-3, new 98.83 % / 1.51e-3 and 98.83 % / 1.97e-3.

Served (gpu0 base, gpu1 new, same client) c = 1 / 4 / 16 / 64 output tok/s 80.5 / 250.6 / 570.4 / 858.9 -> 80.3 / 256.0 / 563.8 / 864.2; 31K TTFT
3.47 -> 3.59 s (run-to-run; the prefill layer is a small share of an iteration). The train-form forward and backward keep their TF32 kernels: the
backward reads T, raw attn0 and the decay mask the blocked solve never forms; the checkpointed forward takes the faster prep and recurrence through the
tuner (W8K probe 1608 -> 1614 tok/s, same losses).
