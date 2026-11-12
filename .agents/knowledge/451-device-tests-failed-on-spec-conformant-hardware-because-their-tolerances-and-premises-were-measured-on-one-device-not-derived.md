<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 451. Device tests failed on a conformant GPU because their tolerance or premise was measured on one device instead of derived

**Lesson:** a tolerance is derived from the computation (`testutil::rounding` for sums, the WGSL accuracy table for
`sin`/`cos`/`pow`), and a test's premise ("the device is still busy", "this is one submit") is checked against the
mechanisms that can falsify it. Both kinds were measured once on one card and then failed on the next conformant one.

## What happened

* **`sin`/`cos` are only specified to `2^-11` absolute on `[-pi, pi]`** (W3C WGSL, "Floating Point Accuracy"); outside
  that range nothing is promised. The interleaved-RoPE test asked `1e-5` at angles up to 5 rad and failed on an Intel
  Arc at `3e-5`, which is conformant. The bound is now `(|e|+|o|) * 2^-11` plus the `pow` and rounding terms, with the
  angle kept inside `[-pi, pi]`.
* **A tolerance scaled by the measured quantity collapses when it cancels.** `rel < 1e-2` against `max(|d|, 1e-3)` for a
  directional derivative (a sum of signed terms) has a tolerance of `1e-5` whenever the terms cancel, below the f32
  noise of a central difference (`u S / eps`). One draw in about a thousand failed; the bound is now built from the
  absolute sum `S` of the terms.
* **`Backend::stats().submits` counted queue submissions, and a driver workaround splits a batch into one per step.**
  The Intel ANV sliced-binding workaround (`flush_serialized`) made a 25-step MoE layer report 25 submits against a
  budget of 3. `submits` now counts batches and `queue_submits` the physical count.
* **`submit` itself waits for the device once a batch reaches `max_unsynced_workgroups`** (the preemption valve). A
  test that queued 256 bandwidth-bound dispatches to "keep the device busy" had drained it inside `submit`, so the
  "work still outstanding" premise was false on every run. Make the work long in its inner loop (a naive `matmul`), not in
  its workgroup count, and assert the count stays under the valve.

## Rule

State the derivation of a tolerance in the test. Before asserting a timing or counting premise, read the code between the
call and the clock for a flush, a wait or a split.
