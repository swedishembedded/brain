<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 114. A generated tier is held to the reference's answer, not to the language's

**Lesson:** A generated backend must reproduce what the portable reference computes, explicitly,
rather than inherit whichever behaviour its target happens to have. Otherwise the tiers disagree on
inputs no kernel is supposed to produce and every comparison between them becomes negotiable.

## What happened
The PTX ISA clamps a shift amount at or above the word width (`x << 32` is 0). x86, and so the
Cranelift reference, masks it modulo 32 (`x << 32` is `x`).

WGSL itself calls that range indeterminate, so there is no "correct" answer to inherit.

## Why
That is exactly why a generated backend may not inherit whichever behaviour its target happens to
have. If it does, the two tiers disagree on an input no kernel is supposed to produce.

## What to do
Reproduce what the portable reference computes, explicitly: `(b) & 31u` written into the emitted
code.

The same reasoning picks:

- `1.0f / sqrtf(x)` over `rsqrtf`
- `rintf` over `roundf` (WGSL rounds halves to even)
- the spec's own `mix` composition over the algebraically equal one
- `--fmad=false`

Each is a case where the faster or more natural target idiom is a DIFFERENT function, and the
difference is invisible until a parity assertion fails somewhere unrelated.
