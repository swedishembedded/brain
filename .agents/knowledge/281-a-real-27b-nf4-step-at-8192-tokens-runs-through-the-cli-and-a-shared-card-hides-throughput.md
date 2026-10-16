<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 281. A real 27B NF4 step at 8192 tokens runs through the CLI; a shared card hides throughput

Measured 2026-10-16, GH200 shared with other jobs (a neighbour process of about 18 GiB at high utilisation
was present for most of it), release binaries.

**Real checkpoint, CLI** (`brain qwen35 finetune CORPUS --base Qwen3.8-27B-FP8 --mode lora --base-dtype nf4
--batch 1 --seq-len 8192 --lora-rank 16 --warmup 1 --steps 1`, corpus of 10,640 tokens): the long-sequence
defaults were chosen automatically and printed (GDN segment 128, MLP rows 512, bf16 residuals). Load 161 s,
whole run 4,833 s including the initial-loss forward, loss 0.0333 -> 0.0304 (the corpus is four facts
repeated, so the loss is low from the start). Peak device memory 35,341 MiB (per-process, sampled every 2 s
from nvidia-smi), host RSS high-water 7,503 MiB (8,132 MiB in an earlier attempt). Two earlier attempts were
cut off by an external time limit and are not counted.

**Why the wall time is not a throughput figure.** The probe at the same shape (64 layers, random weights,
8192 tokens, NF4) took 1,765 s for one step (5 tokens/s) in this window, against 405 s (20 tokens/s)
recorded in an earlier quiet window (knowledge 234). A 4-layer probe at 4096 tokens ranged 20 s to 50 s per
step across repeats of the same configuration. Swept over GDN segment 128/512/1024/2048 twice, each
configuration's two repeats differed by more than any difference between configurations (102 vs 207,
143 vs 122, 132, 154 vs 127 tokens/s), so no setting was shown faster and none was changed.

**Where the launches are** (`BRAIN_PROFILE=1`, 4 layers, 4096 tokens, segment 128, device time inflated by
per-launch timing): 70 kernel kinds, 39.5 s device over 50.7 s wall for one step. Largest: `matmul_reg3` NF4
5.1 s (314 calls), `concat_split` 4.8 s (2,724 calls), `bmm_tiled` 2.6 s (4,608 calls), `matmul_reg3` bf16
2.4 s, `matmul_dx_reg` NF4 2.4 s, `splice_add` 1.6 s (2,493 calls), `region_copy` 1.6 s, `gdn_layout_permute`
1.5 s. The NF4 and bf16 GEMMs are about 30% together; the rest is thousands of small per-segment launches,
which is where a larger segment or fused segment kernels would act. This is a ranking under contention, not
a speed-up.

Rule: do not tune throughput on a card another job is using; rank by device time over replays of the same
program (`program_table`) or wait for a quiet card.
