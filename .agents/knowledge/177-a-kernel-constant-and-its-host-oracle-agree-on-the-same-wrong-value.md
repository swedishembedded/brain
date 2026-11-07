<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 177. A kernel constant and its host oracle agree on the same wrong value

**Lesson:** A kernel constant and its host oracle can agree on the same wrong value. Only an oracle
that takes the value from somewhere else can fail.

## What happened

GLM-DSA reads `rope_theta` from its config, and GLM-5 declares `8e6`. Four kernels and a host
reference nevertheless compiled the base in as `10000`:

- `rope_train` / `rope_train_bwd` (the forward and backward of every attention layer);
- `rope_sub` (the sparse indexer's partial-head rotation);
- glmdsa's inline decode twin `rope_train_at`;
- `distill.rs`'s host rotation, which computes the indexer's distillation gradients.

## Why

Every glmdsa test stayed green because every one of them compares one of these against another: the
KV-cache decode against the full recompute, and the indexer against the dense model. Both sides of
each comparison rotated at the same wrong base.

## What to do

Use an oracle that takes θ from somewhere else. `crates/model/tests/it/interleaved_rope_theta.rs`
rotates on the host at the caller's θ.

θ is now a Params field (`theta: f32`, the last slot) in all three shared kernels and in the inline
one, and `IdxDims` carries it into the host distillation. toymoe, whose config declares no base,
passes its own named `ROPE_THETA` of 10000, the same value its inference kernel `rope.wgsl` uses.

## Evidence

Before the fix the kernels missed that host rotation by 1.13 at θ=8e6.
