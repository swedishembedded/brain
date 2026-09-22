<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 112. Two backends can disagree about a language's guarantees, and both stay quiet

WGSL guarantees two things about work-group execution that neither a barrier-
splitting CPU JIT nor a naive CUDA emission gives for free, and neither one
fails loudly when it is missing:

* **An early `return` is permanent.** The rule that a pre-barrier `return` must
  be work-group-uniform is what makes it legal; it also means that invocation
  does no further work at all, including after the barrier. A compiler that
  models a kernel as two per-invocation loops split at the barrier gives the
  return the scope of the FIRST loop only, so the invocations a padded-grid
  guard (`if (w >= p.n_wg) { return; }`) exists to exclude run the second half
  anyway and write their output. In CUDA the same construct fails the other
  way: a thread that really returns cannot arrive at `__syncthreads()`, so the
  threads that did arrive wait on a barrier that can never complete.
* **`var<workgroup>` is zero-initialised** at the start of every work-group.
  Back it with a stack slot allocated once per kernel, or with `__shared__`,
  and the second work-group reads the first one's values. A reduction whose
  tail lanes never write their slot folds that stale value into its sum.

Both are silent-wrong-number defects, and both hid in this tree because no test
over-dispatched a padded grid or left a work-group slot unwritten. The
generalisation: when you retarget a kernel language, enumerate the guarantees
the SOURCE language makes that the target does not, and write a test per
guarantee - the kernels themselves will not tell you, because they are written
assuming the guarantee holds.
