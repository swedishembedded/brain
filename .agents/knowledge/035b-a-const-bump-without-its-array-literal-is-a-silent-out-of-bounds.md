<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 35b. A `const` bump without its array literal is a silent out-of-bounds write

`crates/kernels/wgsl/router_gate.wgsl` and `router_gate_train.wgsl` declare
`const MAX_EXPERTS: u32 = 128u;` - bumped up from 64 in an earlier commit,
with a comment explaining why (128-expert MoE routers were coming). But the
shader body's actual scratch storage, `var prob: array<f32, 64>` / `var used:
array<bool, 64>`, was a second, independent literal that the bump did not
touch. Both compiled cleanly (WGSL fixed-size arrays are just a length
literal, unrelated to `MAX_EXPERTS` unless something cross-checks them), and
every existing test kept passing, because `crates/model/tests/moe_sparse_parity.rs`
exercises `router_gate.wgsl` at a synthetic `n_experts: 8` - nowhere near 64.

The result: for any router with more than 64 experts, every expert index ≥ 64
read and wrote past the end of a 64-element array during the per-token
softmax/top-k/renormalize loop. WGSL has no bounds-checking panic for this -
it is UB that manifests as silently wrong probabilities for roughly the top
half of the expert set, not a crash, not a validation error, nothing that
shows up in a smoke test. Found only because one model's Thinker decoder
(128 experts) was validated against a real top-k-id/weight golden
end-to-end, not just via a same-composition sparse-vs-dense oracle test.

**The general shape to watch for**: any `const` that exists specifically to
size a fixed-length local/array, where the array's own literal is a second,
separately-typed place the same number has to be repeated. Grep for the
`const`'s name is not enough - grep for `array<`/`[N;` sitting near it and
confirm the literal actually reads the const (WGSL has no way to do that, so
"confirm by eye, then add a test at a size between the old and new bound" is
the actual mitigation, not a language fix). The existing `moe_sparse_parity.rs`
oracle test is exactly this class of gap: it proves the sparse and dense
paths agree with EACH OTHER, which is blind to a bug present identically in
both - a real-weight/real-golden test is what caught this one, not the
same-composition oracle.

**Recurrence, caught by a later audit of this lesson's own fix**: the
fix above touched `router_gate.wgsl`/`router_gate_train.wgsl` (`MAX_EXPERTS`
64→128) but never reached `router_gate_sigmoid.wgsl` - a THIRD kernel with
its OWN independent `const MAX_E: u32 = 64u;` and six of its own
`array<f32, 64>` scratch locals, never bumped. Worse: a status doc's own
written record of the fix above stated one model was "unaffected (uses the
separate `router_gate_sigmoid.wgsl`)" - true as far as it went, but the
person who wrote that line checked that the model used a DIFFERENT file, not
that the different file had its OWN safe cap. That model's published config
declares 256 routed experts through that router, so this sat as a live,
undetected out-of-bounds write behind a status doc that actively said the
opposite. Same root cause as this lesson's own `router_bwd.wgsl` (a SEPARATE
kernel from the pair above, `array<f32, 64>` `pr`/`dp` scratch, hard-capped at
exactly the same 64): every kernel that copy-pasted this softmax/top-k
scratch-array shape needed its own audit - fixing the first two instances did
not imply the third and fourth were safe, and the written record repeating
"unaffected" without re-deriving it made this LESS likely to be caught, not
more. Fixed by rewriting `router_bwd.wgsl` array-free (mirroring
`router_bwd_sigmoid.wgsl`'s already-unbounded style - that one was NEVER
capped, its own header comment claiming "E <= 64" was simply stale/wrong) and
adding a loud `assert!(n_routed_experts <= 64, ...)` in the model's
constructor for the `router_gate_sigmoid.wgsl` forward path specifically,
since an array-free top-k rewrite there is real kernel work (needs `top_k`
passes over `E`), not a literal bump - deliberately NOT re-bumping the
constant a second time, which is this lesson's own failure mode repeated.
**The sharper general shape this recurrence adds**: when a status/lessons
doc records "X is unaffected because it uses Y instead," that claim is only
as strong as whoever wrote it having ALSO checked Y's own limit - a written
record that sounds like verification but is actually just "different code
path, assumed safe" is worse than no record at all, because it stops the
next reader from re-deriving what was never actually checked.
