<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 11. Test RNGs are code too

`(*state >> 33) as f32 / 2^31 - 1.0` yields **[-1, 0)** - every sample negative,
so no test ever exercised the positive branch of an activation. Correct is
`>> 32`. Fixing it made depth's QARep gradcheck fail at rel 0.1119, which was a
*second* real finding: the analytic gradient was right and `eps` was the defect
(a ±1 direction over `n` components is an L2 step of `eps*sqrt(n)`). The knee was
measured across five values, `eps` dropped to 1e-4 and the tolerance **tightened**
8× - not loosened.
