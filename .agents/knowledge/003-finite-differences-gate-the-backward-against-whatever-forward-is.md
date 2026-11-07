<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 3. Finite differences gate the backward against whatever forward is emitted

**Lesson:** Finite differences prove the derivative, never the objective: they gate the backward
against whatever forward is emitted, so a mis-weighted objective is self-consistent and passes.

## What happened
`check_vqgan` cannot see which term `beta` multiplies.

## What to do
Pin the objective by reading the reference implementation's source. There `beta` sits on the
**codebook** term, not on the commitment term a stale local comment claimed.
