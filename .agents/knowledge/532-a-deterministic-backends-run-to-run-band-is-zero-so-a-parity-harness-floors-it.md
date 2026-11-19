<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 532. A deterministic backend's run-to-run band is zero, so a step-parity harness floors it at the round-off bound

**Lesson:** "within the old path's own run-to-run band" is a tolerance only where the old path varies. On the CPU backend
two runs of one step are bit-identical, so the band is zero and a replacement that only reorders a reduction fails it.
Floor the band at the binary32 agreement bound of the step's longest reduction (`brain_testutil::rounding`), report the
raw run-to-run band beside it, and give the old path a way to run inside the new code, or the comparison is not of the
two paths at all.

## What happened

2026-11-19, the tier-1 harness (`brain_testutil::step_parity`) on eight real SFT records: the old path run twice gave a
band of exactly 0 in loss, log-probabilities, gradients and post-step weights. Trimming weighted rows (A1) sums each
adapter's weight gradient over 5376 positions instead of 8192, so its gradients differed by 3.1e-7 (relative L2), a
reordering, not a defect, and a zero band would have refused it.

## What to do

- The band is `max(run-to-run, agreement_bound(k))` with `k` the longest reduction of the step (rows of a micro-batch plus
  the micro-batches accumulated); the harness prints run-to-run, band and new difference, so a test log carries all three.
- Keep the old path reachable after the change. Two ways that cost no API:
  - a trimming change: a weight-zero sentinel target at each row's last position, with the step's denominator pinned to the
    real supervised count, keeps the whole block computed and adds nothing to the loss or gradient;
  - a restructured objective (KTO's two-row batches): write the old step out in the test from public pieces.
- Check that the harness can fail: scoring the lazy KTO reference under the live adapter instead of the start-of-run one
  moved the log-probabilities by 0.083 and failed the test, while the post-step weights alone (1.5e-5) stayed inside
  the band. Compare every field; weights alone hide a wrong reference.

## Evidence

`qwen3` `step_parity::a_weighted_batch_trains_trimmed_as_over_the_whole_block`, `rl` `kto_step_parity` (CPU,
`Fp32Reference` GEMMs).
