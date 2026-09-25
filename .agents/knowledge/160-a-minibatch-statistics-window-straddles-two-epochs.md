<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 160. A minibatch statistics window straddles two epochs

`splat::opt` collected density-control evidence over "the last N iterations
of the stage", N sized to one epoch of views. With full-batch fitting that is
every view N times. With minibatches drawn from a fresh permutation each
epoch, the last N iterations are the tail of one epoch and the head of the
next - two different orders - so some views were measured twice and others
not at all. A gaussian visible only in the missed views has zero evidence and
reads as dead.

It did not look like a sampling bug; it looked like a scene full of invisible
gaussians, and the first suspects (sub-pixel starting splats under the Mip
filter, tile-instance clamping) were each fixed and measured to change
nothing.

**Rule:** a statistic that must cover every view is measured over every view,
explicitly - `opt.rs` now runs the credit pass once per stage over all views
against the stage's final parameters. Never infer coverage from an iteration
count when the batch order is permuted.
