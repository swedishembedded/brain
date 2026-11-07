<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 35c. The softmax top-k router's cap outlived its own bug-class writeup

**Lesson:** The softmax top-k router's cap outlived its own bug-class writeup: a cap that is never
exercised past its boundary survives until a real model hits it.

## What happened
`35b`'s "recurrence" paragraph fixed `router_bwd.wgsl` array-free and added a guard `assert!` to
`router_gate_sigmoid.wgsl`'s forward path. It left `router_gate.wgsl`/`router_gate_train.wgsl`,
the plain-softmax top-k router used by every model that isn't the sigmoid-`noaux_tc` scheme, still
hard-capped at `MAX_EXPERTS = 128` via `var prob: array<f32, 128>` / `var used: array<bool, 128>`.

Nobody had needed more than 128 experts yet, so the cap never got exercised past its own boundary
and the writeup moved on to the other two kernels.

A later model declares **256** routed experts through exactly this router (plain softmax top-k, not
sigmoid), which is the first real model in this repo to hit it.

## What to do
Fix it the same array-free way `router_bwd.wgsl` already was.

- Neither kernel now caches anything in an array sized by `n_experts`. The softmax numerator is
  stashed in the `gate`/`probs` OUTPUT buffer itself (already sized `[rows, n_experts]`, so it
  doubles as scratch) instead of a second buffer.
- The only remaining `var<function>` array (`sel_idx: array<u32, 32>`) is bounded by `top_k`, never
  by `n_experts`.
- This is the exact fix shape `35b` names, but a bound the `top_k` side needed too. A top-k
  selection loop that excludes already-picked experts via an `n_experts`-sized `used[]` bool array
  has the identical failure shape as caching the probabilities themselves; both are "an array
  sized by the wrong dimension".

**`router_gate_sigmoid.wgsl` (the sigmoid-`noaux_tc` router) is still behind its documented
`assert!(n_routed_experts <= 64)` guard, unchanged by this fix.**

- The model above doesn't use that router kind.
- A full array-free rewrite there needs its own group-limited top-k pass structure
  (`n_group`/`topk_group`), which is real kernel work, not a bound swap.
- Anyone raising that router past 64 routed experts needs to do that work first, not bump the
  `assert!`.

## Evidence
Caught by `crates/model/tests/it/router_gate_expert_cap.rs`, mirroring
`router_bwd_expert_cap.rs`'s own shape: a real (non-same-composition) host oracle at `n_experts`
values below, one past, and far past the former cap (8, 129, 256). This is the class of test `35b`
itself says is the only thing that actually catches this, since a same-composition sparse-vs-dense
oracle is blind to a bug present identically on both sides of the comparison.
