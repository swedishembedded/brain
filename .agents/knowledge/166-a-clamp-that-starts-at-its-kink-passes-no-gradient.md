<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 166. A clamp that starts at its kink passes no gradient

The environment behind a splat scene (`splat::env`) is radiance
`E = max(0, sum_k Y_k c_k)`, clamped because radiance cannot be negative.
Its coefficients started at zero. The backward masks the gradient where the
clamp is active (`E > 0` is false at exactly zero), so no coefficient ever
received a gradient: the fitted environment came back black, and the test
still scored 31 dB - because the scene's gaussians had inflated to paint
the sky instead. Only comparing against a run without an environment (also
31 dB) showed the environment had done nothing.

The fix is to start where the function is live: uniform at the
photographs' mean colour. The same trap waits for any ReLU-style
non-negativity constraint initialized at zero - SH colours clamped at zero,
opacities through a hard floor.

**Rule:** never initialize a parameter exactly at a kink whose subgradient
the backward takes as zero; and gate a new model component against the
same fit WITHOUT it, not against an absolute score the rest of the model
can reach on its own.
