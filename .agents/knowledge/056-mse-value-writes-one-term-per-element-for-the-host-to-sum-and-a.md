<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 56. `mse_value` writes one term PER ELEMENT for the host to sum, and a 1-element buffer makes the loss silently mean "element 0"

Closing `check_unet` produced a first run where 239 of 263 tensors failed with
`rel_err` clustered at 0.96-1.00 and `analytic / numeric` ratios scattered
between 1.5x and 190x. The scatter is what made it look like a backward defect:
a uniform factor would have said "normalisation", and the ratios growing with
distance from the output said "contributions being lost or doubled per layer".

Both readings were wrong. The **backward was fine**; the loss head was not.
`mse_value.wgsl` (like `ce_value`) writes `(pred[i] - tgt[i])^2 / n` into
`out[i]`, one invocation per element, and the CALLER sums - the per-element
division is what keeps the host reduction a plain sum. The trainer allocated
`gpu.storage(1)` and dispatched one thread, so `forward()` returned the first
element's term alone. Nothing errors: the buffer is big enough for what was
dispatched, and the value is a real, finite, plausible-looking float. The
finite-difference sweep then measured `d(element 0's term)/d(w)` against an
analytic `d(mean over all 256)/d(w)`, and each parameter influences element 0
by its own arbitrary amount - hence the scatter.

**The thing to carry**: when a finite-difference check fails *broadly* -
including on the tensors closest to the loss, where the chain is shortest -
suspect the objective before the adjoint. `conv_out.weight` failing at
`rel = 1.00` was the tell: one conv from the output, there is almost no chain
left to get wrong. Check that `loss()` computes what the analytic gradient is
the gradient OF, and specifically check the value kernel's own contract - both
`mse_value` and `ce_value` are per-element-plus-host-sum, not scalar-out.
`crates/vqgan/src/train.rs` had it right (`gpu.storage(te)`, `te` threads) and
was the working call site to copy, per the "copy a working dispatch" rule.
