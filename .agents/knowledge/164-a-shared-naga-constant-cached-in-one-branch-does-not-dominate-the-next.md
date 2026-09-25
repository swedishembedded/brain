<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 164. A shared naga constant cached in one branch does not dominate the next

naga gives a literal, a module `const`, and a `let` bound to either ONE
expression handle that every use shares, and it never puts that handle in a
`Statement::Emit` range - it has no program point of its own. The CPU JIT
(`crates/wgsl-cpu`) memoised every expression handle the first time it
evaluated it, so such a constant was materialised in whichever Cranelift block
used it first and that `Value` was reused everywhere after. When the first
use sat inside a loop body or one arm of an `if` and a later use in a block
that the first does not dominate, Cranelift's verifier rejected the whole
kernel: `uses value v40 from non-dominating inst40`.

It was a hard compile failure, never a wrong number, so it never reached a
model - it reached kernel authors instead. `resize_bicubic_dx.wgsl` was
restructured around it, and its header carried "a `let` bound to a bare
literal must be first used in a block that dominates all its other uses" as a
standing rule for everyone editing the file: a JIT defect turned into a
kernel-writing convention. Nothing about the WGSL was wrong; the same source
ran on the GPU.

The fix is in the cache, not the kernels: only expressions some `Emit`
evaluates are memoised (`inline::Frame::emitted`); everything else is
re-materialised at each use, where it trivially dominates, and Cranelift's
egraph pass merges the duplicate constants again. The regression test is
`a_shared_constant_is_usable_across_non_dominating_blocks` in
`crates/wgsl-cpu/tests/aggregates.rs` (`let c = 2.0;` and a `const K` used in
a loop body and then in both arms of two later `if`s), which failed with the
verifier error above before the change.

**Rule:** when a backend rejects valid WGSL that another backend runs, fix the
backend and record it - do not write the workaround into the kernel as a
convention. A per-kernel rule that exists only because of one compiler's bug
outlives the bug and is followed by people who never saw it.
