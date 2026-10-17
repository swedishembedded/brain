<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 317. Element-wise glue folds and an NF4 decode cache take repeated passes out of the 27B training step (measured by count and parity; the idle-card clock is pending)

Question: after knowledge 316 (device step about 7.2 s, glue and decode still ~1.5 s), which of the
remaining passes outside the GDN mixer can go without a custom GEMM? Everything here was measured on
a card shared with a decode server, so the evidence is launch counts, bit-level parity and traffic
arithmetic, not device milliseconds. Figures are **VERIFIED-BY-RUN** unless marked **DERIVED**.

## What changed (series `t4-glue-1017`)

- `swiglu_bwd_fused`: `silu_bwd_da` and `silu_bwd_db` in one pass (3 reads + 2 writes per element
  instead of 5 + 2). `block::swiglu_bwd` takes it on CUDA.
- `add_rmsnorm_fwd` and `rmsnorm_dx_add`: the residual add folded into the RMSNorm that follows it
  (forward), and into the RMSNorm input gradient (backward, frozen norm weight). Both norms of a layer,
  forward, rebuilt forward and backward.
- `silu_mul_bf16`: `silu(gate)*up` written as fp32 and, in the same pass, as the packed bf16 operand of
  the down GEMM; `Ops::act_f32_with_bf16` seeds the activation's cached cast with it.
- `gdn_chunk_loop(_train)_tc` with `heads_tok`: the recurrence output token-major, no
  `gdn_layout_permute` after it.
- `DecodedWeightScope`: each whole NF4 weight is decoded to bf16 once per layer pass and shared by the
  segments, MLP row chunks, rebuilt forward and input-gradient products. The vendor library takes bf16
  operands from memory, so a decode in its prologue is not available; the repeats are what goes.

## Evidence

| change | check | result |
|---|---|---|
| SwiGLU backward | f64 transcription, totals 4096 / 1000 / 1 / 7 / 262,147 | 4.7e-7 absolute |
| add + RMSNorm fwd, RMSNorm dx + add | f64, d = 5120 / 64 / 20 | 1.2e-7 / 3.0e-7 / 2.0e-7 |
| `silu_mul_bf16` | f64 for the product; bf16 half equals RNE of the fp32 product of the same launch | 1.5e-6; exact |
| token-major recurrence output | training and inference forwards, batch 1 and 2, vs chunk-major | bit-identical |
| decode scope | 4-layer 4096-token NF4 probe | decode launches 121 to 62, loss identical (13.4194), peak 16.4 to 15.7 GiB |

qwen35 `lora_{recompute,nf4_base,gdn_segments,bf16_residuals}` and `train_tier_trajectory`, gdn
`gdn_mixer_{tc,segmented}` stay green. (One trajectory run failed once at 21 s beside the decode
server and passed on five reruns; no cause recorded.)

## DERIVED effect on the 8192-token step

From the knowledge 316 profile: the two SwiGLU backward kernels 186 ms to about 130 ms; `add2` 141 ms
mostly gone plus one read of each normalised tensor; one fp32 read + bf16 write of the [n, 17408]
product per MLP pass; the recurrence permute 95 ms; the NF4 decode 307 ms to roughly 90 ms (2,960
launches to about a quarter). Together about 0.5 s of a 7.2 s step. An idle-card run of the whole tree
is queued in `runs/t-gdn2/final.sh` form and must replace this paragraph.

## Not done

`take_rows` / `put_rows` (the MLP row-chunk copies, `concat_split` 205 ms and `splice_add` 215 ms) are
bandwidth-bound copies at about 28 us per 42 MB: they only go away if the library GEMM binds a row
window of the activation (it accepts operand offsets) and the adapter and norm paths do the same;
that is a wider change in `qwen35::model` than a kernel.
