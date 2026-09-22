<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 118. A metadata gate that scans source text reads the header prose as code

`kernels-cuda`'s registry check refuses a kernel whose source contains
`__restrict__`, because brain's device buffers alias by design. The first
hand-written kernel to land failed that check - for a sentence in its own
header explaining that it deliberately carries no aliasing promise, and why.
The scan could not tell an explanation from a declaration.

This repo had already made and fixed the identical mistake one layer over:
`workgroup_size_of` used to take the first `@workgroup_size` anywhere in a
WGSL file and therefore read the number out of ten kernels' header prose. The
fix is the same both times - strip `//` comments and scan what is left - and
the generalisation is that any gate keyed on "does this source mention X"
must decide, explicitly, whether prose counts. The tempting alternative
(reword the comment until the gate is quiet) trades a documented design
decision for a green check, and the next kernel that genuinely needs to
explain the same thing hits it again.
