<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 115. Materialise generated expressions at their emit point, not at their use

Translating a naga expression DAG to C++ text by substituting each expression's
text into its users is wrong in a way that only shows up with mutation: a
`Load` inlined at its use site floats past any store to the same location that
was emitted in between, so the kernel reads the new value where the source
reads the old one. naga marks evaluation order explicitly with `Statement::Emit`
ranges, and the fix is to bind each emitted expression to its own variable
there, exactly as the Cranelift path does. Hoisting those variable
DECLARATIONS to function scope additionally means no `goto` (WGSL's `continue`
is not C's - the continuing block still has to run) can ever jump across an
initialisation, which C++ forbids.
