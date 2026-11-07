<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 118. A metadata gate that scans source text reads the header prose as code

**Lesson:** Any gate keyed on "does this source mention X" must decide, explicitly, whether prose
counts. Rewording the comment until the gate is quiet trades a documented design decision for a
green check.

## What happened
`kernels-cuda`'s registry check refuses a kernel whose source contains `__restrict__`, because
brain's device buffers alias by design.

The first native CUDA kernel to land failed that check, for a sentence in its own header explaining
that it deliberately carries no aliasing promise, and why. The scan could not tell an explanation
from a declaration.

This repo had already made and fixed the identical mistake one layer over. `workgroup_size_of` used
to take the first `@workgroup_size` anywhere in a WGSL file and therefore read the number out of ten
kernels' header prose.

## What to do
Strip `//` comments and scan what is left. That was the fix both times.

Do not reword the comment until the gate is quiet. The next kernel that genuinely needs to explain
the same thing would hit it again.
