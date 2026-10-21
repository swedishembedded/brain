<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 346. A training row padded to the longest record cost the full block, so a 27B monitor evaluation took 690 s

Symptom: the CE8 diagnostic (27B bf16 base, rank 16, 16 monitor records) spent ~680 s per evaluation (~42 s per
record) against ~26 s for a training step of two examples, and the same 27B ran its older monitor evaluations in ~40 s.

Cause (profiled, not guessed): the 16 monitor records are prepared into 186 examples (one per supervised turn), and every
example is a row padded to the run's block (14 990 tokens, the longest training record). The forward pass computed the whole
row whatever the example's length (3.4-4.7 s per example, flat, with 33-200 supervised tokens), 186 x 3.7 s = 690 s. It
was neither the best-parameter save (no patience in the run, nothing is held or spilled) nor precision: GPU kernels were
busy tensor-core GEMMs. Training paid the same padding (about 2.7x: the mean example is 5.6k tokens).

Fix: `Qwen35::set_batch` runs a one-sequence training batch only up to its last supervised position, rounded up to whole
GDN chunks (`run_len`). The model is causal, so the tail changes no earlier activation and adds no loss or gradient (the
same argument as the existing padding of an awkward `t`). Weighted-loss, MTP, multi-sequence and inference builds keep
the full row. Test: `a_batch_supervised_only_early_runs_only_its_prefix` (loss and every gradient equal a build sized to
the shorter row, also under segmented GDN, a bf16 residual stream and chunked MLP).

Measured (27B bf16, card 1, same 186 examples, block 14 990): evaluation 690 s to ~245 s (3.7 s to 1.3 s per example);
training steps 26 s to 5-15 s; step-1 loss unchanged (0.7447), step-5 held-out 0.1326 to 0.1324.
