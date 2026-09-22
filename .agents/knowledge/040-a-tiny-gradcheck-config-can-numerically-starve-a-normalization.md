<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 40. A tiny gradcheck config can numerically starve a normalization op, producing a hollow - but *passing* - gradcheck

`gradcheck::check_qwen35` (qwen35moe's model-level backward integration -
renamed to `check_qwen35moe` once the dense `qwen35` sibling arch took the
plain name; this entry is left as originally written) passed
clean on first run: every parameter's finite-difference check fell inside
the workspace's `(4e-3, 8e-2)` gate. Every Gated-DeltaNet-layer parameter
(`A_log`, `dt_bias`, `in_proj_a.weight`, `in_proj_b.weight`) reported its
finite-difference numeric gradient as *exactly* `0.0`, not merely small -
and the tolerance's `denom.max(1e-3)` floor swallowed the distinction, since
the tiny *analytic* values (`~1e-13` to `~1e-17`) were dwarfed by the same
floor. A wholly wrong or entirely-disconnected gradient looks identical to
a genuinely tiny one once both round to the floor.

Confirmed by direct probing (bypass the checker, call `forward()` by hand):
perturbing `A_log` by ±10 - enormous relative to its own scale - left the
loss **bit-identical**, while the raw decay gate `g` it feeds was confirmed
to vary correctly (-0.001 to -157) under the same perturbation. The root
cause was three stages downstream: `Qwen35Config::tiny()`'s standard
`std=0.02` init, applied to `in_proj_qkv.weight` then a *depthwise* causal
conv1d (`groups=conv_dim`, so each output channel sums only `kernel=4`
terms - far less central-limit averaging than a `d_model`-wide matmul) then
SiLU, collapses `query`/`key`'s pre-normalization magnitude to `~1e-6` by
the time they reach `l2norm_scale.wgsl`. That kernel's `eps=1e-6` (tuned for
the REAL model's much wider `d_model=2048`, not a toy config) then
*dominates* the normalization denominator instead of the vector's own norm,
so the "normalized" output is just `x/sqrt(eps)` - still proportional to
the collapsed input, not an actual unit vector - and the entire chunked
recurrence downstream is swamped to `~1e-11` regardless of any decay/beta/
gate parameter. The standalone `model::gdn` unit tests
(`crates/model/tests/gdn_chunk_{fwd,bwd}.rs`) never hit this: they feed
`q`/`k`/`v`/`g`/`beta` directly at `std=1.0` with no upstream conv chain, so
this normalization/init interaction never had a chance to manifest there.

Fix: **do not touch `model::gdn`'s `eps` or the model's production
`std=0.02` init** - both are correct for the real model's scale. Instead, the
gradcheck harness itself (`qwen35_gradcheck_harness`, now
`qwen35moe_gradcheck_harness`, in
`crates/gradcheck/src/lib.rs`) overrides just `in_proj_qkv.weight`/
`conv1d.weight` to `std=1.0` post-init, restoring a real, non-degenerate
finite-difference signal (confirmed: `blocks.N.linear_attn.conv1d.weight`
and `A_log` both now show close analytic/numeric agreement instead of a
floor-dominated `0.0`). **The general shape**: whenever a tiny gradcheck
config feeds a fresh, small-std init through more than one cascaded
small-scale stage before an operation with a fixed additive epsilon
(L2-norm, RMSNorm, any `/(x+eps)` normalization), check whether the
pre-normalization magnitude has collapsed to epsilon's own scale - a
gradcheck that passes because the *true* gradient is unresolvably small is
indistinguishable, from the report alone, from one that passes because the
implementation is correct. When a parameter reports numeric `0.0` across
every check in a report, perturb it by hand outside the checker and confirm
the *loss itself* moves before trusting the pass.
