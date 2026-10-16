<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 312. One wait per MLP pass lifts the 27B step to about 630 tokens/s, and the step is bound by device time, not launches

Question: after knowledge 311 (621 tokens/s NF4 on the real checkpoint), which of the
remaining levers on the 2.7 s host gap and the 3.9 s non-GEMM device time pays, within a
short budget? Idle card throughout (`other_contexts_start=0`, `other_contexts_end=0`, and an
empty `nvidia-smi` compute list before the CLI run). Figures are **VERIFIED-BY-RUN** unless
marked **DERIVED**.

## Change taken (Q2 in `throughput-27b.md`)

`mlp_fwd_rows` and `mlp_bwd_saved` waited for the device after every row chunk (about 4
waits per MLP per pass at 2048-row chunks). They now wait once after the loop: the chunk
transients are released together, so the host keeps enqueueing the next chunk while the
device works. The loss is unchanged to the fourth digit (13.4607 / 13.4012 / 13.3244, as in
knowledge 311 within the fp32 summation noise of its sweep kernel); the chunks' transients now live until
the pass ends (a few hundred MiB at 2048 rows; not isolated by a same-build comparison).

| | before (311) | after |
|---|---:|---:|
| probe W8K NF4 (random weights, 64 layers, rank 16, seg 2048, rows 2048) step 2-3 s | 13.42 | 13.03-13.08 |
| probe tokens/s | 611 | 626-629 (warm 628) |
| probe peak MiB | 49,973 (311) | 44,789 (the probe's sampled figure; not the same sampler as 311's row) |
| real checkpoint NF4 (CLI defaults, batch 1, seq 8192) step 2 / 3 s | 13.2 / 13.2 | 12.9 / 12.9 |
| real checkpoint tokens/s | 621 | **635** |

The "before" probe figure is the one recorded in 311, not a same-session rerun; the gain
(about 2.5 percent) is within what two idle runs differ by, so treat it as small.
Real-checkpoint loss on the 100,000 random-token corpus: 12.8938, 12.8947, 12.8905
(timing run only). The bf16 base was not re-measured (DERIVED: the same loops, a similar gain).

## What the profile says (probe, NF4, one timed step)

Device time 10.74 s of a 13.07 s step. Library GEMMs 2.65 s (21,896 calls), `bmm_tiled`
1.01 s, the three GDN recurrence kinds 1.79 s, layout permutes 0.52 s, `l2norm_scale` 0.34 s,
`scale_row` 0.31 s, NF4 dequantisation 0.30 s, `mul` 0.27 s, `splice_add` 0.26 s. The host
thread is busy for the whole step (`cpu_s` equals `step_s`).

## Launch count is not the bound (settings experiment)

Raising the segment and MLP chunk cuts the launches but not the time: seg 4096 / rows 4096
48,575 dispatches, 12.73-12.81 s, 642 tokens/s, peak 57.7 GiB; seg 8192 / rows 4096 37,151
dispatches, 13.03-13.09 s, 627 tokens/s, peak 62.9 GiB. Halving the launches bought 2 percent
for 13 GiB, so the defaults stay. The 35 us per launch of 311 is therefore mostly waiting
for device work that is already queued, and per-layer graphs (K8) would not move the step
much: the device itself is the bottleneck. A submission holding a library GEMM also cannot
be captured (see `throughput-27b.md` K1), so K8 needs the GEMMs as capturable nodes first.

## What is not done

1,000 tokens/s is not reached (best real-checkpoint 635, NF4). With the device busy 10.7 s of
13 s, 1,000 tokens/s needs the device total at or under about 7 s: the library GEMMs
(2.65 s) are at the card's practical rate, so the rest (about 8 s of GDN recurrence,
`bmm_tiled`, layout permutes, norms, element-wise kernels) has to fall by roughly half,
which only fused whole-tensor GDN forward/backward kernels (K7, the fused GDN backward of
knowledge 311) can deliver. DERIVED, not run: K4 (head CE gradient in the forward) removes
one head GEMM pass of the 1.27 G-parameter head, on the order of 0.2 s of the 13 s step.
