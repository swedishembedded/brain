<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 392. A prompt's start cost two device waits and its first token a head pass of its own, and the mixed iteration's scheduling curve is flat, so the per-row cost is the lever

Question: after knowledge 391, brain serves c=64 at ~750 tok/s against 1203 for the reference server. Is the schedule (mixed iterations against decode steps plus larger prompt chunks) or the host/idle time worth taking?

Measured (FP8 27B, int8 KV, `ref/run.sh`, distinct seed per concurrency, `fast` profile binary, both cards at the same time):

1. **Scheduling curve** (prompt-row floor of a mixed iteration, `MIN_MIXED_PROMPT_ROWS`; c=64 output tok/s / TTFT p50 / ITL p50 / ITL p99; c=16 tok/s):
   256: 713 / 2104 ms / 65 / 107, c=16 510; 512 (now): 754 / 497 / 87 / 124, c=16 529; 1024: 756 / 715 / 24.6 / 194, c=16 533; 2048: 775 / 931 / 24.6 / 255, c=16 538.
   Throughput moves 3 % over an 8x range of the floor: the per-row cost (GEMM 62 us a row, non-GEMM 56 us) is the same in a 600-row and a 2000-row iteration. At 1024+ rows the schedule alternates mixed iterations with decode steps (ITL p50 24.6 ms) at the price of TTFT and p99. Not changed.
2. **Idle in the mixed iteration** (nsys, median of 144 iterations, span 86.1 ms, kernels 82.6): 0.5 ms between `serve.mixed` opening and the first kernel, 0.8 ms in front of the first layer, 3.5 ms of launch gaps inside (largest: 14 us after each `add_rms_quant`), 1.2 ms between iterations. Of the host time in front of a forward, **1.05 ms per started prompt was `reset_work_state` + `store_prefilled`** (a 150 MB clear, 48 scatters and `poll_wait`) and **1.02 ms per finished prompt was `admit_greedy`** (a head pass of one row re-reading the 2.5 GB head). Both are now gone from the iteration: a cold prompt's pool row is scattered from one shared zero row with no wait (`zero_pool_row`), and a finished prompt's last hidden row rides in the decode rows' head pass (`MixedOutcome::first`; the pass asks for candidates when a finishing prompt samples). nsys: `mixed.begin_slices` 0.68 -> 0.21 ms, `serve.accept` 0.33 -> 0.01 ms, iteration 90.6 -> 90.0 ms. Served c=64 745 vs 741 tok/s (inside the 1 % run-to-run band: the saving is 0.6 %).
3. **What stays idle**: ~0.8 ms in front of the first layer (position tables on the host), ~0.6 ms between iterations, ~3.5 ms of launch gaps: asynchronous scheduling can take at most 1.4 ms (1.5 %) of the first two, because the next iteration's decode inputs depend on the sampled tokens.

Open: the 54 % of the iteration that is the FP8 GEMM, the GDN segment walk (8.7 ms) and decode-row state (5.9 ms), the quant chain (8 ms: `silu_mul_quant` 48 us against a 22 us read/write roof, `add_rms_quant` 19 us against 9), and running decode on a reserved SM slice beside the prefill (not measured here).
