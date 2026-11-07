<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 402. A 4096-row round spends a fifth of its time in glue, and a full-attention layer's glue moved the query five times

**Lesson:** The glue kernels around the GEMMs and attention can take a fifth of a large round, and a
layer that moves the same tensor through several separate passes is far off its byte roof. Fusing
those passes while staying bit-identical recovered most of it.

## What happened
Question (knowledge 400): which non-GEMM, non-attention kernels of a 4096-row mixed round are off
their byte roof, and by how much?

Method: nsys of `brain serve` (afa0cb1de, FP8 27B, int8 KV, `--act-precision` default bf16) at 8Kx16
on gpu1 (73 rounds of 4096 rows), kernel time from the sqlite export; one layer's kernel sequence
printed in order. Roof 3.4 TB/s. Per kernel at 4096 rows (in situ, us a launch; bytes are the
kernel's own):

| kernel (layer kind) | us | bytes | % of roof | launches a round |
|---|---|---|---|---|
| `silu_mul_quant_e4m3_128` (MLP, bf16 gate/up in, E4M3 out) | 235 | 358 MB | 45 | 64 |
| `add_rms_quant_e4m3` (norm front end, xn stored or not) | 98-113 | 273-357 MB | 80-100 | 128 |
| `quant_epilogue_cluster` mode 2 + `quant_e4m3_128` (attention gate, 16 layers) | 391 + 44 | 350 + 125 MB | 22, 83 | 16 |
| 2 x `concat_split` (query and gate halves out of the projection) | 172 + 171 | 200 MB each | 35 | 16 |
| `rmsnorm_rows` q (24 heads) + k, `rope2d_partial` q + k | 128 + 21 + 32 + 7 | 200 + 200 MB | 45 | 16 |
| `cast_f32_bf16` (GDN layer: xn to bf16 for the a/b gates) | 38 | 126 MB | 97 | 48 |
| `gdn_conv_silu_seg_vec` | 123 | ~170 MB | 41 | 48 |
| `gdn_gated_norm` (E4M3 out) | 88 | ~125 MB | 42 | 48 |
| `flash_prefill_combine` | 180 | | | 16 |

A GDN layer plus MLP is 4.4 ms of kernels at 4096 rows, GEMMs 62 %. An attention layer's glue (the
rows above, not the attention itself) is about 1.3 ms, more than a GDN layer's.

## Why
The attention layer was the largest single piece. The query projection's output interleaves
`[query | gate]` per head, and the chain:

- split it with two copies;
- normed and rotated the query in two more passes;
- then read the gate copy in the output gate's epilogue (a decode-regime kernel, 1024-thread blocks
  a row, at 0.9 TB/s over 4096 rows).

`silu_mul_quant_e4m3_128` is issue-bound, not byte-bound, and that is why it stays above 60 %. Exact
`expf` and two correctly rounded divisions an element are ~45 instructions; at 4 issue slots an
SM-cycle that is the kernel's time (SM 73-75 % busy, DRAM 45 %). The bytes are not the limit;
staying bit-identical to the pair of kernels it replaces is.

## What to do
Built (bit-identical to what they replace, each held by a native-vs-chain gate):

- `gqa_qk_prep`: query half + key heads -> RMS norm (the 64-lane order of `rmsnorm_rows`) -> partial
  rotation in one launch, the gate left in place. 506 us -> 104 us.
- `gate_mul_quant_e4m3_128`: `ctx * sigmoid(gate)` with the gate read in place from the projection,
  E4M3 rows out, the fp32 product only when an adapter or a backward reads it. 435 us -> 71 us (93 %
  of the roof).
- E4M3 quantisers (`e4m3_quant_common.cuh`): the reciprocal of the block's scale once, three fused
  operations an element instead of a division (a range check and a slow-path call as well).
- For `silu_mul_quant_e4m3_128` that is 62 instructions an element (`expf` 9, the SwiGLU division 13
  with its range check, the quantiser's division 13, the amax and the stores), 138.5 M -> 127.1 M
  warp instructions at 4096 rows; 195 -> 184 us on the bench.
