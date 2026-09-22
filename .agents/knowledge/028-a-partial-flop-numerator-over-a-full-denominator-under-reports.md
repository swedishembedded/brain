<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 28. A partial FLOP numerator over a full denominator under-reports in silence

`vqgan_bench`'s `WHOLE PASS` row summed FLOPs across every row regardless of
whether `gpu_core::cost` had a formula for it, then divided by the whole-pass
time. A third of the VQGAN backward's kernel kinds had no formula, so the
published "backward = 5.4% of peak" was computed from a numerator missing a
third of the graph.

This is the mirror image of the failure `cost` was designed to prevent. An
uncovered kernel already reported `-` per row (never a zero that reads as slow),
but the pass-level total had no such guard - and a *pass* rate is the number a
reader quotes. The fix is that a partly covered pass reports **no rate at all**
and names the kinds it is missing:

    WHOLE PASS   457.42 ms   1404   (rate unavailable - no cost formula for: mse_grad, masked_l1_grad)

which turned an invisible accounting hole into a two-line work item. With every
kind covered, the honest numbers are **forward 356.8 GFLOP/s (3.4%) and backward
638.1 GFLOP/s (6.0%) of the measured roof.**

Two structural notes:

* The coverage test was a hand-maintained *list* of kernel names, so it could
  only fail when someone remembered to extend it - it could not stop a new
  kernel landing unmeasurable. It is now backed by a **ratchet** over the whole
  kernel table that fails when coverage falls.
* Name the uncovered kinds in the profile output, never just count them. They
  are usually cheap enough to fall outside the printed top rows, which is
  exactly where a missing formula hides.
