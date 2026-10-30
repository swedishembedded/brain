<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 385. A mixed iteration paid for its glue in launches and host time, not in the GEMM, so the segments, the gates and the top-k head are one pass each

Question: after knowledge 382-384 a c = 16 mixed iteration (16 decode rows plus ~700 prompt rows, FP8 27B, int8 KV, gpu1) was ~119 ms of span for ~105 ms
of kernels, the FP8 GEMM 58 ms of it. Where is the rest, and how much of it is removable without touching the GEMM?

Measured (nsys `-t cuda,nvtx`, the median of the 69 mixed iterations of a 48-request c = 16 run; the first table row is the profile before, the second after):

| per iteration | before | after |
|---|---|---|
| span / kernels | 119.4 / 105.5 ms (~690 rows) | 100.7 / 93.9 ms (~768 rows) |
| ms per 1000 rows (span) | 173 | 131 (-24 %) |
| `concat_split` | 1209 launches 5.80 ms | 293 launches 1.91 ms |
| GDN prep + recurrence + conv (+ hist) | 184.5 x 3 launches + 185 hist, 11.0 ms | 48 x 3 launches, 8.7 ms |
| `splice` + `region_copy` + pool gather/scatter | 288 + 185 + 321 launches, 8.1 ms | 59 + 0 + 91 launches, 1.65 ms |
| gated norm chain (`rmsnorm_rows` + `silu` + `mul`) over the GDN rows | 28 + 11 + 80 us a layer | `gdn_gated_norm` 19 us |
| top-k head of a sampled step (32 candidates) | 2200 us | 209 us |

What the findings were:

1. **One launch a layer needs data-driven addressing, not bigger kernels.** The GDN mixer ran every prompt segment, and the full-chunk and remainder piece of
   each, as its own conv, prep and recurrence on four `concat_split` row copies, with the sequence's state gathered from the pool before and scattered back
   after. `gdn_conv_silu_seg`, `gdn_chunk_prep_seg_bf16` and `gdn_chunk_loop_seg_bf16` take one table per round (segment rows and pool slot, chunk token rows and
   length), read and write the round's token-major tensors in place, and keep the state and conv history in the pool rows themselves. The recurrence reads
   and writes only its own 32-column slice, so in-place needs no scratch; the conv block that owns a segment's first rows reads the old history into its window
   and then overwrites it (no other block of the segment reads it). The prep and loop entries of the single-sequence path are now two thin wrappers over the same
   device bodies (`brain_gdn_prep_chunk`, `brain_gdn_loop_walk`), so training and the per-segment fallback keep their launches and arithmetic. Output equals the
   per-segment path bit for bit on state and history (`gdn_segments`).
2. **The elementwise glue was slow, not just numerous.** The WGSL-translated `mul` took 80 us for 768 x 6144 words (about 120 GB/s: 64-thread blocks, one word a
   thread); the gated norm is three tensors read once and one written, 19 us. A decode row's projections lead every row block, so the fused pooled decode step reads
   them in place and writes its rows into the gated block.
3. **A fused E4M3 quantisation is not free in a kernel that already runs one block of 1024 threads an SM.** Tried: `quant_epilogue` (the MLP product, 17408 wide)
   and `add_rms_quant` writing the E4M3 bytes and scales of their product beside it, bit-equal to `quant_e4m3_128`. ncu on the 768-row MLP product: 118 us as the
   plain epilogue, 215 us with the pack (46 registers, one block an SM, 50 % occupancy; the epilogue has no other block to hide the extra two IEEE divisions and the
   group-wide max behind), against 22 us for the standalone quantiser it was to replace; a barrier per 128-thread group instead of the block's changed nothing. The
   `add_rms_quant` pack cost what the launch it removed did (+0.70 ms against 128 x 6 us). Neither landed. What did: the gated norm, where a head row is exactly one
   128-group and the warp already holds the row (0.74 -> 0.85 ms against 48 x 11 us and the launch gap).
4. **The prompt rows of an iteration should fill the GEMM's last tile.** The latency-target fit left 636-716 rows; with a 128-row tile that is six tiles, the last
   one 10-80 % full, the same cost as 768 rows. `MixedLimits::row_quantum` rounds the prompt rows up to the tile (capped by the room in the forward).
5. **Host time inside the iteration.** The first kernel of a mixed forward starts 4-6 ms after `serve.mixed` opens: the embedding rows of the round are decoded on
   the host (`ckpt.embed_rows`: 704 random-token rows 3.7 ms warm, 5.4 cold) before stage 0 can start, and the round's whole `[rows, d_model]` block was read back
   (0.9 ms) for the head. The read-back is now the decode rows and each prompt's last row; the decode of the rows goes straight into the result (3.7 -> 2.7 ms).
   Open: a bf16 upload with the widening on the device, or the embedding of a prompt's rows ahead of its slices.
6. **The sampled head.** The device top-k ran `argmax_part`, `argmax_final`, `topk_extract_step` once per candidate, ~69 us a round whatever the row count, so the
   32 candidates of a sampled step cost 2.2 ms (6 % of a c = 16 run's wall clock, 12 % of a single-stream token). `topk_rows_f32` is one block of 1024 threads a
   row, a thread owns a contiguous run read as 16-byte vectors and rescans it only when it owned a winner: 209 us (a first version with strided runs rescanned
   at one 4 KB stride a load and took 1279 us).

Served (same client, FP8 27B, int8 KV; the node is shared, c = 16 ranges 425-451 between runs of the same binary): c = 1 / 4 / 16 / 64 output tok/s
53 / 208 / 395 / 568 -> 57.9 / 239 / 425-451 / 655; 31K TTFT 4.6 -> 4.45 s. `qwen35_fp8_parity` (8 x ~1500 tokens of source text, 32 steps): 99.61 % greedy
agreement, mean KL 1.38e-3, one flip at margin 0.031; W8K NF4 train probe 1033 tokens/s (shared kernels refactored, not changed).

Where the iteration is now (ms of 100.7): FP8 GEMM 61.4 (the 816-block gate/up shape 223 us at 616 TFLOPS against 970 for the down projection of the same FLOPs:
7 waves at 88 % and a short K), `quant_epilogue_cluster` 3.6 + `quant_e4m3_128` 2.6 + `add_rms_quant` 2.0, GDN prep 3.4 + recurrence 3.2 + conv 2.1 +
decode 2.6, GQA glue (`concat_split` 1.9, `sigmoid` / `mul` / `splice` 2.4, `rmsnorm_rows` 0.7, `kv_append` 0.5), attention 2.8, host gaps 8.6 (embedding 4-5).
