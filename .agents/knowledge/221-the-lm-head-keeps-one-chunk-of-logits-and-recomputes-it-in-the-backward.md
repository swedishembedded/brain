<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 221. The LM head keeps one chunk of logits and recomputes it in the backward

`n x V` fp32 logits are 993 KB per token at V = 248320: the 4 GiB binding is
gone at about 4,300 tokens and the card at a few tens of thousands. `model::head_ce::HeadCe`
holds `rows_per_chunk x V` only (logits, their gradient, and the statistics),
whatever the token count. The forward computes a chunk's logits and per-token
loss; the backward recomputes each chunk's logits (one more GEMM per chunk),
runs the two-pass CE gradient (row statistics once, then an elementwise pass)
and applies the head. A single chunk that the forward just computed is not
recomputed.

Things that bite:

- The head weight can also exceed one storage binding (the fp32 table is
  5.09 GB). It is cut into row windows; each window start must be a 256-byte
  offset in every buffer it slices. For NF4 the scale plane has one word per
  32 elements, so window starts are multiples of `2048 / gcd(2048, d)` rows
  (the plan computes this per tier). Passes within a window start at multiples
  of 64 columns, which is aligned for every tier when `d % 32 == 0`.
- Chunk starts are binding offsets into the per-token buffers (labels, loss), so
  `rows_per_chunk` must be a multiple of 64 unless one chunk covers the call.
- `rows_per_chunk * V` must stay below 2^32 (the CE kernels index with u32).
- The logit gradient order is: CE gradient, caller scale (device buffer for a
  recorded tape, constant for immediate calls), logit scale, per-token weights.
  Weights go through `scale_row`, which is not in place, so the caller owns a
  `rows x V` scratch.
- Measured on one GH200 (CUDA backend, debug build of the host side), 4096
  tokens, 248320 x 5120 bf16 head, 1024-row chunks: forward 0.9 s, backward
  2.6 s, 2.39 GiB of scratch; 512-row chunks: 1.4 s / 3.7 s, 1.45 GiB. The
  loss agrees across chunk sizes.
