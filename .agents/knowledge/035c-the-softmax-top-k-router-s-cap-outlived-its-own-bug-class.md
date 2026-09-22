<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 35c. The softmax top-k router's cap outlived its own bug-class writeup

`35b`'s "recurrence" paragraph fixed `router_bwd.wgsl` array-free and added a
guard `assert!` to `router_gate_sigmoid.wgsl`'s forward path, but left
`router_gate.wgsl`/`router_gate_train.wgsl` - the plain-softmax top-k
router, used by every model that isn't the sigmoid-`noaux_tc` scheme -
still hard-capped at `MAX_EXPERTS = 128` via `var prob: array<f32, 128>` /
`var used: array<bool, 128>`. Nobody had needed more than 128 experts yet, so
the cap never got exercised past its own boundary and the writeup moved on
to the other two kernels.

A later model declares **256** routed experts through exactly this router
(plain softmax top-k, not sigmoid), which is the first real model in this
repo to hit it. Fixed the same array-free way `router_bwd.wgsl` already was:
neither kernel now caches anything in an array sized by `n_experts` - the
softmax numerator is stashed in the `gate`/`probs` OUTPUT buffer itself
(already sized `[rows, n_experts]`, so it doubles as scratch) instead of a
second buffer, and the only remaining `var<function>` array
(`sel_idx: array<u32, 32>`) is bounded by `top_k`, never by `n_experts` - the
exact fix shape `35b` names but a bound the `top_k`-side needed too (a
top-k selection loop that excludes already-picked experts via an `n_experts`-
sized `used[]` bool array has the identical failure shape as caching the
probabilities themselves; both are "an array sized by the wrong dimension").

Caught by `crates/model/tests/router_gate_expert_cap.rs`, mirroring
`router_bwd_expert_cap.rs`'s own shape: a real (non-same-composition) host
oracle at `n_experts` values below, one past, and far past the former cap
(8, 129, 256) - the class of test `35b` itself says is the only thing that
actually catches this, since a same-composition sparse-vs-dense oracle is
blind to a bug present identically on both sides of the comparison.

**`router_gate_sigmoid.wgsl` (the sigmoid-`noaux_tc` router) is still
behind its documented `assert!(n_routed_experts <= 64)` guard, unchanged by
this fix** - the model above doesn't use that router kind, and a full
array-free rewrite there needs its own group-limited top-k pass structure
(`n_group`/`topk_group`), which is real kernel work, not a bound swap. Anyone
raising that router past 64 routed experts needs to do that work first, not
bump the `assert!`.
