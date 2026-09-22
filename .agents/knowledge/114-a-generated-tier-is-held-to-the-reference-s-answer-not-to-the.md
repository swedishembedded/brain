<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 114. A generated tier is held to the reference's answer, not to the language's

The PTX ISA clamps a shift amount at or above the word width (`x << 32` is 0);
x86, and so the Cranelift reference, masks it modulo 32 (`x << 32` is `x`).
WGSL itself calls that range indeterminate, so there is no "correct" answer to
inherit - which is exactly why a generated backend may not inherit whichever
behaviour its target happens to have. It has to reproduce what the portable
reference computes, explicitly (`(b) & 31u` written into the emitted code), or
the two tiers disagree on an input no kernel is supposed to produce and every
comparison between them becomes negotiable.

The same reasoning picks `1.0f / sqrtf(x)` over `rsqrtf`, `rintf` over `roundf`
(WGSL rounds halves to even), the spec's own `mix` composition over the
algebraically equal one, and `--fmad=false`: each is a case where the faster or
more natural target idiom is a DIFFERENT function, and the difference is
invisible until a parity assertion fails somewhere unrelated.
