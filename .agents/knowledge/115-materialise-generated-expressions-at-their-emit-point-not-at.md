<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 115. Materialise generated expressions at their emit point, not at their use

**Lesson:** When translating an expression DAG to text, bind each expression to its own variable
where the source emits it. Substituting text into the users breaks as soon as mutation is involved.

## What happened
Translating a naga expression DAG to C++ text by substituting each expression's text into its users
is wrong in a way that only shows up with mutation.

A `Load` inlined at its use site floats past any store to the same location that was emitted in
between. The kernel then reads the new value where the source reads the old one.

## Why
naga marks evaluation order explicitly with Emit statement ranges. Inlining at the use site
ignores them.

## What to do
Bind each emitted expression to its own variable at its `Statement::Emit` point, exactly as the
Cranelift path does.

Hoist those variable DECLARATIONS to function scope. Then no `goto` can ever jump across an
initialisation, which C++ forbids. WGSL's `continue` is not C's: the continuing block still has to
run.
