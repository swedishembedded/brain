<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 562. A finite difference of a routed model crosses top-k flips: pin the routing it took

**Lesson:** a mixture-of-experts loss is piecewise smooth: every top-k selection is a branch, and a perturbation that
flips one row's selection evaluates another function. A gradient check, a "one step lowers the loss" check or a
"two bases train alike" check over a deep routed model hits such flips far more often than the step size suggests:
Inkling 2's tiny looped config makes 192 routing decisions per batch (12 rows, 16 visits), and with random weights a step
of 1e-2 in one entry of a norm gain changed the loss by 0.09 (a central difference of -0.26 where the analytic and the
1e-3..1e-5 differences agreed at 0.0838), and a bf16 rounding of the base moved adapter gradients by up to 36%. None of
it was a backward defect. Pin the routing the reference forward took (routing replay: `taken_routing` /
`TokenObjectives::set_routing_replay`, the router's own weights over the current logits at the pinned experts, so the
router is still differentiated) for every comparison that perturbs the model, or, where the suite cannot pin (the
generic conformance suite), give the subject a config in which every expert routes (`top_k == n_experts`).

## Evidence (2026-11-22)

- `crates/gradcheck/src/inkling2.rs`: with the routing pinned, every parameter (directional), every shared core entry
  checked (elementwise) and every adapter tensor pass; unpinned, numeric derivatives of about ±4.6 appeared on random
  entries (a 0.09 loss jump over a 2e-2 interval).
- `crates/inkling2/tests/it/training.rs`, `a_bf16_base_trains_the_adapters_the_f32_base_does`: unpinned, per-tensor
  gradient differences of 15-36%; pinned and compared with an f32 build over the bf16-rounded values, at most 2.7e-4.
- `crates/rl/tests/conformance.rs`, `inkling2_subject`: the four-expert config failed C3 and one ORPO step
  intermittently; the two-of-two config passes the whole suite.
