<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 11. Test RNGs are code too

**Lesson:** Test RNGs are code too, and a broken one can hide real findings.

## What happened
`(*state >> 33) as f32 / 2^31 - 1.0` yields **[-1, 0)**. Every sample was negative, so no test
ever exercised the positive branch of an activation. The correct shift is `>> 32`.

Fixing it made depth's QARep gradcheck fail at rel 0.1119, which was a *second* real finding.

## Why
The analytic gradient was right and `eps` was the defect: a ±1 direction over `n` components is
an L2 step of `eps*sqrt(n)`.

## What to do
The knee was measured across five values, `eps` dropped to 1e-4 and the tolerance **tightened**
8×, not loosened.
