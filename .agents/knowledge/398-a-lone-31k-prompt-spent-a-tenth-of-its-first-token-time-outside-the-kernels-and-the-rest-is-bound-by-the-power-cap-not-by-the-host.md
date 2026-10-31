<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 398. A lone 31K prompt spent a tenth of its first-token time outside the kernels, and what is left is bound by the power cap, not by the host

Question: FP8 Qwen3.8-27B, one uncached 32.5K-token prompt, 3.50-3.60 s to the first token against 2.32 s for the reference server. Where does every millisecond
go from the request's arrival, and what does removing the host side leave?

Timeline of one request (client send = 0, `nsys` NVTX plus wall-clock stamps in a scratch build; ms, before -> after):

| phase | before | after |
|---|---|---|
| HTTP body, JSON, template | 1 | 1 |
| tokenise (32.4K tokens, one thread) | 87 | 14 |
| prefill slices (2048-row rounds -> 4096) | 3402 (16 rounds) | 2927 (8 rounds) |
| of which idle card between kernels | 276 | 29 |
| first token out after the last slice | ~25 | ~25 |
| first token at the client | 3.50-3.60 s | 2.93-3.00 s |

The "missing 0.5 s" was not outside the kernel stream: it was the slice path of a LONE prompt. One sequence's slice (`prefill_slice`) decoded every row's embedding
on the host, uploaded the f32 block, read all 2048 x 5120 rows of every round back (only the last is used) and gathered/scattered the recurrent work state around
it: 9 ms of idle card per round (3.3 ms after the last layer, 5.4 ms before the first), and it ran the unfused kernel set (`mul` + `sigmoid` attention gate,
`rmsnorm_rows`, the chunk GDN kernels). The multi-sequence forward the server already ran for two prompts gathers the embedding on the card, reads one row, keeps the
state in the pool row and has the fused kernels; a lone prompt over `DECODE_REGIME_MAX_ROWS` now takes it. Kernel time per request 3224 -> 2993 ms, idle 276 -> 64.

- **Round width.** The FP8 GEMMs with N = 5120 (down, output projections) are 320 tiles of 128 x 256 at 2048 rows, 2.4 waves of 132 multiprocessors run as 3; 640 tiles
  at 4096 are 4.8 run as 5. 3.30 -> 3.10 s; 8192 rows is 3.09 s and costs the default context 18 % (4096 costs 6 %: 774144 -> 729088 tokens). Only a lone prompt gets
  the 4096 (`MixedLimits::lone_prompt_rows`): a burst of 1K prompts at that width delayed the first of them.
- **Tokeniser.** The pre-token cut is serial (6 ms); the merges of each piece are independent, so runs of >= 2048 pieces are encoded on up to 16 threads and joined in
  order: 87 -> 14 ms. A faster hasher and borrowed pair keys for the merge table, and merge ranges instead of strings, moved it 87 -> 73 ms only: the cost is the
  cache misses of 250K-entry tables, so the parallel form is the lever.
- **Attention split.** From 2048 rows the grid is already 2.9+ waves; unsplit (no partial sums, no merge launch, longest key range first) is 2-12 % faster per
  call (4096-row round at 4K/8K/16K/24K/32K context: 0.918/2.079/4.183/6.246/8.189 -> 0.815/1.881/3.960/5.978/7.973 ms, 385 TFLOPS causal at 32K).
- **Not kept.** `silu_mul_quant_e4m3_128` with fast division and `__expf` (it is ALU-bound by IEEE divisions, 232 us for 285 MB at 4096 rows): first token unchanged
  (2.98/2.99/2.93 against 2.99/3.00/2.93 s), numerics no longer bit-identical to the pair it replaces; reverted.

What bounds the rest (nsys, 8 rounds of 3.0 s; ms, scaled to the whole request): FP8 GEMMs ~1600 (54 %, 1.01-1.04 PFLOPS in the server against 1.12-1.14 isolated at 8192
rows), attention ~580 (20 %, 385 TFLOPS), GDN ~190, norms/quant/glue ~390. `nvidia-smi` at 100 ms during the prefill: SM clock mean 1.62 GHz (1.28-1.98), 125 of 171
samples with the SW power cap active (0x4), the card drawing ~510 W. The prefill is energy-bound: every byte moved and every redundant instruction is paid in clock
(knowledge 393 measured the same inside the GEMM). The host side is now 14 ms of tokenising and ~25 ms after the last round; 2.2 s needs ~0.7 s off a stream that is
already 97 % kernel time, i.e. a GEMM at ~1.3 PFLOPS (the reference server's 8K-row rate) and an attention kernel past 500 TFLOPS, both at the same power. Candidates by
measured weight: GEMM energy per FLOP (f32 output of the N = 5120 products, the rescale, multicast without the cluster-wide barrier), attention K/V widening shared across
the 6 heads of a kv head (the widening is ~30 % of the kernel, knowledge 383 item 4), the q/k/v split + q/k norm + rope + int8 append of the 16 GQA layers (four kernels,
~0.6 ms a layer at 4096 rows).
