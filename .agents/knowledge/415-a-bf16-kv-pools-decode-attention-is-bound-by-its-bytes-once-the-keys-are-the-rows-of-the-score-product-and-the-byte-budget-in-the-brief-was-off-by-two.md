<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 415. A bf16 KV pool's decode attention is bound by its bytes once the keys are the rows of the score product, and the byte budget in the brief was off by two

Question (the brief of 2026-11-01): the decode kernel over a bf16 pool was the TF32 kernel (widen each element through registers, `m16n8k8` at half rate); the 8-bit
kernel reaches 112 us a layer over half the bytes. Can the 8-bit design, with bf16 operands, reach the memory roof?

## The byte budget

8K keys x 16 sequences x 4 kv heads x 256 x 2 B is **268 MB per plane, 537 MB for K and V**, not 268 MB: the roof is 158 us at 3.4 TB/s, not 77 us. (The int8 pool is
272 MB with its scales; its 112 us is 2.4 TB/s.) A target of "77 us" was unreachable by construction; judge the kernel by the achieved rate.

## What the kernel is (`paged_flash_decode_gqa_hd256_kvbf16.cu`)

- **Operands.** A bf16 key or value is an exact `mma.sync.m16n8k16 ... bf16` operand; nothing is rounded. Q and P are the f32 side, split into bf16 parts. Never to f16
  (|x| > 65504 would be infinity: a spec runs keys of 1e5).
- **Three parts, not two.** Parts carry 8 bits each. Two parts (16 bits) fail the large-score spec (queries x10, scores in the tens: the exponent's absolute error is the score's
  relative error times the score): 3.2e-5 worst at unit scale (three parts: 1.9e-6) and a failure at x10. Three parts (24 bits) pass at 2.5e-5, against 9.3e-5-1.0e-4 for the TF32 kernel at the same
  shapes, and 0.6-2.3e-6 against 4.5e-6 at unit scale.
- **Keys are the rows.** The 8-bit kernel puts the 6 heads of a group in the rows of the score product, padded to 16 (10 of 16 rows wasted, two key tiles). Here S^T = K . Q^T: 16 keys
  are the rows, the 6 heads pad to 8 columns, one `mma` per k-step and part (48 per chunk for three parts, against 96). The accumulator holds (key gid | gid + 8, heads 2 tig | 2 tig + 1),
  the transpose of the value product's B operand (keys 2 tig .. | head gid), and `movmatrix.trans.b16` turns one into the other in registers (6 per chunk). As a side effect the running
  maximum and sum belong to the lane's own two heads, as do the rows of the output accumulator: the per-chunk correction needs no shuffle.
- **Ring of half chunks.** A chunk's keys (8 KiB) and values (8 KiB) are separate `cp.async` units; a warp's ring holds three of them (96 KiB a block, two blocks an SM), so a
  chunk's values load while its keys are computed. A 16-key bf16 chunk is 16 KiB: the 8-bit kernel's whole-chunk stage would have allowed one block of four warps an SM.
  Whole-sector loads as in the 8-bit kernel (knowledge 401): key piece w at bytes 64 w + 16 tig, value piece p at bytes 128 p + 16 gid.
- **Split cut.** `flash_splits_kv8` with 2 blocks an SM was measured against fixed cuts of 1-32 tiles a split at 8K x 16 / 64 and 32K x 16: the model's choice is within 2 % of the best fixed cut
  at 16 x 8K (186 us kernel) and 64 x 8K (689 us), and finer cuts lose (1 tile 233 / 860 us). The combine is 8-15 us.

## Kernel time (gpu1 idle, best of 9, split pass + combine, us a layer)

| cell | TF32 kernel (before) | new | roof at 3.4 TB/s | new, % of roof |
|---|---|---|---|---|
| 8K x 1 | 32.4 | 27.6 | 10 (16 sequence-heads cannot fill the card) | 36 |
| 8K x 16 | 225.3 | 194.4 | 158 | 81 |
| 8K x 64 | 840.6 | 704.1 | 632 | 90 |
| 32K x 1 | 72.3 | 65.9 | 39 (low occupancy) | 60 |
| 32K x 16 | 830.1 | 698.7 | 632 | 90 |

The int8 kernel on the same card: 120.7 (8K x 16), 401 (8K x 64), 398.9 (32K x 16). `decode_attention_matrix_cell_report` prints this table, with the time of each kernel.

## Served (gpu1, FP8, bf16 KV, f32 GDN state, no speculation, default prompts, interleaved twice, mean of two)

| cell | base tok/s | new tok/s | base mean TPOT ms | new mean TPOT ms |
|---|---|---|---|---|
| 8K x 1 | 70.6 | 70.6 | 11.9 | 11.9 |
| 8K x 16 | 287.0 | 295.7 (+3.0 %) | 42.0 | 40.7 |
| 32K x 16 | 82.8 | 83.5 (+0.8 %) | 137.4 | 135.7 |
| 8K x 64 | 348.4 | 355.4 (+2.0 %) | 138.4 | 135.9 |

The decode step is mostly not this kernel: 16 full-attention layers x 31 us saved is 0.5 ms of a 42 ms step at 8K x 16. The attention family in the reference comparison (4.11 s against 0.75 s) is
dominated by prefill, not by this kernel.

## Rules

- The kernel was admitted on agreement with exact f64 attention over ragged, scattered and shared-prefix batches, on large scores and on elements beyond the f16 range, with NaN-poisoned outputs and
  partial records, AND on 2,000 back-to-back launches at 16 x 8K, 64 x 1K and 1 x 32K returning the first launch's bits (`native_decode_survives_back_to_back_launches_at_served_shapes`, ignored;
  `dmesg` Xid count unchanged, 596 before and after).
- `i8_gemv_multi_native`, `qops_frozen_lora_grad` and `qops_gemv_small_m` fail on origin/main as well (unrelated, found while running the gpu-core and model suites).
