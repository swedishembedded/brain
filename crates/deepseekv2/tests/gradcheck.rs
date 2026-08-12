// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Backprop correctness gate for the DeepSeek-V2-family MHA decoder.
//!
//! Finite-difference (`gradcheck::directional_check`) comparison over **every**
//! trainable tensor, through the blanket `CheckModel for model::Model` impl -
//! no bespoke harness, because [`deepseekv2::DeepseekV2`] implements
//! `model::Model` directly.
//!
//! What this covers that nothing else in the tree does:
//!
//! - **Plain causal MHA** at `n_kv_heads == n_heads` with NEOX (half-split)
//!   RoPE over the FULL `head_dim` - the layout the real checkpoint uses, and
//!   NOT the interleaved-pairs convention `crates/glm`'s MLA path applies.
//! - **The dense→MoE block schedule** (`n_dense_layers = 1`): block 0 is a plain
//!   SwiGLU MLP, block 1 is the sparse MoE, both in one backward.
//! - **The FUSED, UNWEIGHTED shared expert** - `model::moe::shared_expert_bwd`'s
//!   `None` arm over a single `n_shared * moe_ff`-wide SwiGLU, added with no
//!   gate. This crate is that arm's second caller.
//! - **`norm_topk_prob = false` and `routed_scaling != 1.0` through a real
//!   decoder block.** Those two `RouterKind::Softmax` fields are exactly why
//!   this decoder cannot reuse `crates/moe`'s router configuration, and a
//!   kernel-level check of `router_gate`/`router_bwd` alone does not prove the
//!   MODEL wires the same pair into both halves - a forward that scaled and a
//!   backward that did not is a silently wrong gradient, not a crash.
//!
//! ## Two fixture shapes, because neither alone is honest
//!
//! **Smooth (`top_k == n_experts`).** A hard top-k selection is a discontinuity
//! finite differences cannot see through: perturbing the router weight can flip
//! *which* experts a token selects, and the central difference then straddles a
//! kink. `check_moe`, `check_glm` and `check_qwen35` all take this mitigation,
//! and it is the only shape in which the **router weight's own** gradient can be
//! gated at all. Every tensor is checked here, router included.
//!
//! **Sparse (`top_k < n_experts`).** The shape the real checkpoint runs, and the
//! ONLY shape in which `norm_topk_prob` means anything: renormalising the
//! selected probabilities is an exact no-op when the selection is every expert
//! (`Z = Σ_all p = 1`), which was measured, not assumed - a `norm=true`/
//! `norm=false` pair at `top_k == n_experts` produced BIT-IDENTICAL per-tensor
//! reports, so gating the flag there would have been an inert test that always
//! passes. [`router_policy_flags_are_live`] asserts that separation directly
//! before any gradient claim rests on it. In the sparse variants the router
//! weight itself is excluded from the gate (it is the one tensor whose
//! perturbation can cross a selection boundary) - but every OTHER tensor still
//! receives its gradient *through* `d_x` from the router's backward, so the
//! `norm`/`scale` derivations are still covered end to end.

use deepseekv2::config::DeepseekV2Config;
use deepseekv2::model::{DeepseekV2, PIPELINES};
use gradcheck::{directional_check, Report};

/// The gradcheck fixture: the golden dumper's own tiny decoder dimensions
/// (`d_model` 12 = 3 heads x head_dim 4, 2 layers with block 0 dense at ff 21
/// and block 1 MoE at ff 7, 2 shared experts fused to 14, vocab 19), with the
/// expert count and the router policy as the knobs a variant sets.
fn cfg_of(n_experts: u32, top_k: u32, norm_topk_prob: bool, routed_scaling: f32) -> DeepseekV2Config {
    let mut cfg = DeepseekV2Config::tiny();
    cfg.shape.n_experts = n_experts;
    cfg.shape.top_k = top_k;
    cfg.norm_topk_prob = norm_topk_prob;
    cfg.routed_scaling = routed_scaling;
    cfg
}

/// **Why this fixture initialises every non-gain tensor at `std = 0.15` instead
/// of [`deepseekv2::init::init_weights`]'s production `0.02`.** Both numbers
/// below were measured on this fixture, not reasoned about in the abstract.
///
/// *The MoE branch was invisible.* At `std = 0.02` and `d_model = 12` the
/// routed-expert branch contributes ~2.6e-5 to a residual stream of ~2e-2 -
/// about 0.1 % - because three cascaded small-scale stages compound (`gate`/`up`
/// matmuls, SiLU's near-identity attenuation for small inputs, then a
/// `moe_ff = 7`-wide `down` matmul) and the top-k gate multiplies what survives
/// by another ~0.2. Two consequences, both observed:
/// every routed-expert tensor's numeric finite difference came back an exact
/// multiple of 2.384e-5 - one ULP of a loss near `ln(19)` divided by `2·eps`,
/// i.e. the FD noise floor - against analytic derivatives of only ~1e-4, so the
/// gate was passing on its `atol` floor rather than on agreement; and
/// [`router_policy_flags_are_live`] failed outright, because `norm_topk_prob`
/// moved the loss by 6e-7 on 2.9485 (≈2 ULP). At `0.15` those same derivatives
/// are ~1e-2 with relative errors ~1e-4 - resolved by three orders of magnitude,
/// and the router policies separate cleanly.
///
/// *`eps` was a huge perturbation.* `directional_check` steps every entry by
/// `±eps = 5e-3`, which is **25 %** of a `std = 0.02` tensor's own scale (and,
/// summed over `tok.weight`'s 228 entries, a step of norm 0.076 against a weight
/// of norm 0.30). The embedding's central difference was correspondingly
/// dominated by curvature, not by the derivative: `tok.weight` came back at
/// `analytic = -8.77e-1` vs `numeric = -7.82e-1`. At `0.15` the same `eps` is
/// 3.3 % of scale and that tensor lands inside the workspace gate.
///
/// The production init is not wrong for the real model's 1280-wide `d_model`;
/// this is a harness conditioning fix, the same resolution (and the same
/// reasoning) as `gradcheck::check_qwen35`'s own `in_proj_qkv`/`conv1d` rescale.
/// Norm gains keep their `1.0` identity init - rescaling those would test a
/// configuration no checkpoint ever has.
const FIXTURE_INIT_STD: f32 = 0.15;

fn harness(seed: u64, cfg: DeepseekV2Config) -> DeepseekV2 {
    let mut init = <DeepseekV2 as model::Model>::init_weights(&cfg, seed);
    let mut rng = data::rng::Lcg::new(seed ^ 0x9e37_79b9);
    let mut names: Vec<String> = init.keys().cloned().collect();
    names.sort(); // HashMap iteration order is not stable; the rescale must be reproducible
    for name in names {
        if name.ends_with("norm.weight") || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") {
            continue; // RMSNorm gains stay at the identity
        }
        for x in init.get_mut(&name).expect("named above").iter_mut() {
            *x = rng.scaled(FIXTURE_INIT_STD);
        }
    }
    let model = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 2, 6, &init, true);
    // 2 sequences x 6 positions, every position supervised (no IGNORE), so the
    // whole graph carries gradient.
    let v = cfg.vocab();
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % v).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % v).collect();
    model.set_batch(&x, &y);
    model
}

/// The workspace-standard combined abs+rel tolerance for fp32 directional FD on
/// a device (`crates/gradcheck`'s own `assert_grad_gate`), plus the
/// dead-gradient structural guard the abs floor cannot see through.
fn assert_grad_gate(report: &Report, what: &str) {
    let (atol, rtol) = (4e-3, 8e-2);
    let fails = report.failures(atol, rtol);
    assert!(
        fails.is_empty(),
        "{what}: gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(
        dead.is_empty(),
        "{what}: silently-DEAD gradients (analytic exactly 0, numeric nonzero -- a wrong or missing backward kernel, not a small derivative): {:?}",
        dead.iter().map(|c| (&c.param, c.analytic, c.numeric)).collect::<Vec<_>>()
    );
}

/// Drop the router weight from a report - the one tensor whose finite
/// difference can cross a top-k selection boundary in the sparse variants.
fn without_router(full: &Report) -> Report {
    Report { checks: full.checks.iter().filter(|c| !c.param.ends_with("mlp.router.weight")).cloned().collect() }
}

/// One forward+backward: the loss, and the analytic gradient of a routed-expert
/// weight (the tensor whose gradient the router's combine weight multiplies
/// directly).
fn loss_and_expert_grad(cfg: DeepseekV2Config) -> (f32, Vec<f32>) {
    let m = harness(7, cfg);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    (loss, m.read_grad("blocks.1.mlp.experts.0.down.weight"))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
}

/// **Before any gradient claim rests on them: prove the two router flags are
/// not inert.** A gate on a flag that changes nothing tells you nothing - and
/// one of these combinations genuinely IS a no-op, which is exactly why it has
/// to be pinned rather than assumed.
///
/// Checked on BOTH halves, because a flag can be live in the forward and
/// dropped on the way to the backward (the failure `model::moe`'s own doc warns
/// about: "a forward that silently defaults one of them is a gradient the
/// backward cannot check"):
///
/// - At `top_k < n_experts`, `norm_topk_prob` must move the loss AND the
///   routed-expert gradient.
/// - At `top_k == n_experts` it must move NEITHER: `Z = Σ_all p = 1`, so
///   `p / Z == p` exactly. This is the measured reason the smooth variants
///   below cannot be said to test the flag - an earlier revision of this file
///   claimed they did, and a `norm=true` / `norm=false` pair produced
///   bit-identical per-tensor reports.
/// - `routed_scaling` must move both, at either shape.
#[test]
fn router_policy_flags_are_live() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (sparse_raw, g_raw) = loss_and_expert_grad(cfg_of(5, 2, false, 1.0));
    let (sparse_norm, g_norm) = loss_and_expert_grad(cfg_of(5, 2, true, 1.0));
    let scale = g_raw.iter().chain(&g_norm).fold(0.0f32, |a, x| a.max(x.abs()));
    assert!(scale > 1e-6, "routed-expert gradient is ~0 ({scale:.3e}); the fixture cannot see the MoE branch at all");
    assert!(
        (sparse_raw - sparse_norm).abs() > 1e-4,
        "norm_topk_prob is INERT in the FORWARD at top_k=2 of 5 (raw={sparse_raw}, renormalised={sparse_norm}) -- \
         the fixture cannot distinguish the two router policies, so no gate on it means anything"
    );
    assert!(
        max_abs_diff(&g_raw, &g_norm) > 0.05 * scale,
        "norm_topk_prob is INERT in the BACKWARD: the routed-expert gradient is the same either way \
         (max|diff|={:.3e} vs scale {scale:.3e}) -- the forward's renormalisation is not reaching the backward",
        max_abs_diff(&g_raw, &g_norm)
    );

    let (smooth_raw, gs_raw) = loss_and_expert_grad(cfg_of(4, 4, false, 1.0));
    let (smooth_norm, gs_norm) = loss_and_expert_grad(cfg_of(4, 4, true, 1.0));
    assert_eq!(
        smooth_raw, smooth_norm,
        "at top_k == n_experts, renormalising a softmax over EVERY expert must be an exact no-op \
         (Z = 1); if these ever differ, the router kernel is doing something other than p/Z"
    );
    // The same no-op must hold in the backward, but only up to float
    // reassociation - the two branches reach it by different arithmetic.
    // `router_bwd.wgsl`'s `norm = 1` path forms `dp_f = d_gate_f/Z - sdp/Z²`
    // where `sdp = Σ_{e∈S} d_gate_e·p_e`; with `S` = every expert, `Z = 1` and
    // that is `d_gate_f - sdp`, a CONSTANT shift across all `f`. The softmax
    // backward that follows, `d_logit_i = p_i(dp_i - Σ_j p_j·dp_j)`, cancels
    // that shift exactly (`Σ_j p_j = 1`), landing on the same `p_i(d_gate_i -
    // sdp)` the `norm = 0` path computes directly. Equal in exact arithmetic,
    // a few ULP apart in fp32 - so this is a tolerance, not `assert_eq!`.
    let gs_scale = gs_raw.iter().chain(&gs_norm).fold(0.0f32, |a, x| a.max(x.abs()));
    assert!(
        max_abs_diff(&gs_raw, &gs_norm) <= 1e-4 * gs_scale,
        "at top_k == n_experts the two router backward branches must agree to float noise \
         (max|diff|={:.3e} vs scale {gs_scale:.3e})",
        max_abs_diff(&gs_raw, &gs_norm)
    );

    let (scaled, g_scaled) = loss_and_expert_grad(cfg_of(5, 2, false, 2.5));
    assert!(
        (scaled - sparse_raw).abs() > 1e-4,
        "routed_scaling is INERT in the FORWARD (scale=2.5 gives {scaled}, scale=1.0 gives {sparse_raw})"
    );
    assert!(
        max_abs_diff(&g_scaled, &g_raw) > 0.05 * scale,
        "routed_scaling is INERT in the BACKWARD: a forward-only scale is exactly this bug"
    );
}

/// Smooth shape, the checkpoint's own router policy (raw top-k probabilities,
/// unit scale). The only variant in which the **router weight's own** gradient
/// is gated, since it is the only one with no selection boundary for a finite
/// difference to cross. Everything else - MHA + NEOX RoPE, the dense block, the
/// routed experts, the fused unweighted shared expert, the untied head - is
/// gated here too.
#[test]
fn grads_match_finite_differences_smooth() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let report = directional_check(&harness(7, cfg_of(4, 4, false, 1.0)), 5e-3, 4, 7 ^ 0x1234);
    report.print();
    println!("deepseekv2 smooth (top_k=n_experts=4, raw gates): max_rel={:.3e}", report.max_rel());
    assert_grad_gate(&report, "deepseekv2 smooth");
}

/// Smooth shape with a routed scaling factor != 1.0. The scale multiplies the
/// gate in the forward and must multiply the logit gradient in the backward; a
/// forward-only scale (or a backward-only one) is off by exactly 2.5x on the
/// router and every routed-expert tensor, which an 8e-2 relative gate cannot
/// absorb.
#[test]
fn grads_match_finite_differences_smooth_with_routed_scaling() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let report = directional_check(&harness(7, cfg_of(4, 4, false, 2.5)), 5e-3, 4, 7 ^ 0x1234);
    report.print();
    println!("deepseekv2 smooth + routed_scaling=2.5: max_rel={:.3e}", report.max_rel());
    assert_grad_gate(&report, "deepseekv2 smooth (routed_scaling=2.5)");
}

/// **The real checkpoint's shape and policy**: `top_k = 2` of 5 experts, RAW
/// (un-renormalised) top-k softmax probabilities as combine weights.
/// `router_bwd.wgsl`'s `norm = 0` branch is a genuinely different (simpler)
/// derivation, not a limit of the other: with no `1/Z` factor there is no
/// quotient rule, so the cross-term coupling every selected expert to every
/// other one is ABSENT. Also the only variant that exercises
/// `moe_linear_gated`'s per-row early-exit for real - most rows are skipped by
/// most experts here.
#[test]
fn grads_match_finite_differences_sparse_raw_router() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let full = directional_check(&harness(7, cfg_of(5, 2, false, 1.0)), 5e-3, 4, 7 ^ 0x1234);
    full.print();
    let report = without_router(&full);
    println!("deepseekv2 sparse raw top-2-of-5: max_rel={:.3e} (router-inclusive {:.3e})", report.max_rel(), full.max_rel());
    assert_grad_gate(&report, "deepseekv2 sparse (norm_topk_prob=false)");
}

/// The same sparse shape with `norm_topk_prob = true` - the ONE configuration
/// that exercises `router_bwd.wgsl`'s renormalisation branch non-trivially
/// (`Z = Σ_selected p < 1`, so the `sdp / Z²` cross-term is live). Its gradient
/// reaches every upstream tensor through `d_x`, which is what this gate covers;
/// see [`router_policy_flags_are_live`] for the proof that this variant and the
/// one above are not the same computation.
#[test]
fn grads_match_finite_differences_sparse_renormalised_router() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let full = directional_check(&harness(7, cfg_of(5, 2, true, 1.0)), 5e-3, 4, 7 ^ 0x1234);
    full.print();
    let report = without_router(&full);
    println!("deepseekv2 sparse renormalised top-2-of-5: max_rel={:.3e} (router-inclusive {:.3e})", report.max_rel(), full.max_rel());
    assert_grad_gate(&report, "deepseekv2 sparse (norm_topk_prob=true)");
}

/// A forward must be finite and bit-reproducible at a fixed seed and batch -
/// the cheapest guard against an uninitialised or aliased scratch buffer, which
/// a gradient check (two forwards either side of a perturbation) can average
/// away.
#[test]
fn forward_is_finite_and_deterministic() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let m = harness(11, cfg_of(5, 2, false, 1.0));
    let a = m.forward();
    let b = m.forward();
    assert!(a.is_finite(), "loss must be finite, got {a}");
    assert_eq!(a, b, "forward must be deterministic across repeated submits");
    // A fresh 19-token vocab starts near ln(19) = 2.944; anything far off means
    // the graph is not actually a language-model head.
    assert!(a > 1.0 && a < 6.0, "fresh-init loss {a} is not near ln(vocab)");
}

/// `logits_all` rebuilds the forward tape at a shorter `t`; causal attention
/// means the shared prefix must produce the same logits either way. This is
/// what proves the MoE scratch (the shared `moe_acc`, the per-expert `MoeActs`)
/// is re-indexed at the smaller row count rather than reading stale rows from
/// the previous call.
#[test]
fn logits_all_is_stable_across_lengths() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = DeepseekV2Config::tiny();
    let init = <DeepseekV2 as model::Model>::init_weights(&cfg, 5);
    let v = cfg.vocab() as usize;
    let m = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 6, &init, false);
    let tokens: Vec<u32> = (0..6).map(|i| (i * 7 + 2) % 19).collect();
    let long = m.logits_all(&tokens);
    let short = m.logits_all(&tokens[..4]);
    assert!(long.iter().all(|x| x.is_finite()), "logits must be finite");
    let worst = short.iter().zip(&long[..4 * v]).fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(worst < 1e-4, "causal prefix logits diverged between lengths: maxabs={worst}");
}
