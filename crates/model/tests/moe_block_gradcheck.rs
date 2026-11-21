// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host central-finite-difference gradcheck for `model::moe`'s new backward
//! surface (`router_fwd_kind`/`router_bwd`/`expert_dgate`/`expert_bwd`/
//! `moe_layer_bwd`), modelled on `vit_block_gradcheck.rs`'s shape but using
//! `gradcheck::elementwise_check` directly (a `CheckModel` impl) rather than
//! a Python-generated golden -- this module IS the oracle (central
//! differences of the cached forward), matching the plan's own ask.
//!
//! This is the REAL correctness gate for the new backward, not a formality:
//! `check_moe`/`check_glm` (the model-level gradchecks) both use tiny
//! configs with `top_k == n_experts`, so `moe_linear_gated`'s early-exit
//! branch (a non-routed row skipping its K-reduction) is NEVER TAKEN in
//! either -- a model-level gradcheck alone cannot prove the sparse branch.
//! Every case here runs at BOTH `top_k < n_experts` (the branch neither
//! existing gate exercises) and `top_k == n_experts` (the dense-equivalent
//! case), for BOTH `RouterKind` arms.

use std::cell::RefCell;
use std::collections::HashMap;

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use gradcheck::{elementwise_check, CheckModel};
use model::glu::{expert_bwd_glu, expert_fwd_glu, GatedAct, GluBwdScratch, GluGrads, GluIds, GluLayout, GluScratch, GluSpec, GluWeights};
use model::moe::{
    expert_dgate, expert_fwd, experts_fwd_train, moe_layer_bwd, router_bwd, router_fwd_kind, ExpertBwdScratch, ExpertGrads, ExpertScratch, ExpertWeights,
    MoeActs, MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds, RouterKind, COMPACT_TRAIN_KERNELS,
};

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("router_gate", kernels::ROUTER_GATE),
    ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
    ("router_bwd", kernels::ROUTER_BWD),
    ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
    ("expert_counts", kernels::EXPERT_COUNTS),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),
    ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("glu_act", kernels::GLU_ACT),
    ("glu_act_bwd", kernels::GLU_ACT_BWD),
    ("scale_add", kernels::SCALE_ADD),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

const ROWS: u32 = 5;
const D: u32 = 6;
const FF: u32 = 4;
const E: u32 = 4;

/// One MoE layer, wired for `gradcheck::CheckModel`: `loss()` runs the
/// (gated) sparse forward via `router_fwd_kind`/`expert_fwd`, saving every
/// expert's activations into `MoeActs`; `backward()` runs `moe_layer_bwd`
/// against those saved activations. Weight params: `router.weight` and each
/// expert's `expertN.{gate,up,down}_w` -- `moe.router.bias` (SigmoidNoAuxTc
/// only) is intentionally NOT a checked param, matching real GLM (`Role::
/// Frozen`, never backprop'd).
struct MoeLayerCheck {
    g: Gpu,
    kind: RouterKind,
    shape: MoeShape,
    matmul: usize,
    matmul_dx: usize,
    matmul_dw: usize,
    fwd_ids: MoeIds,
    bwd_ids: MoeIdsBwd,
    router_bwd_ids: RouterBwdIds,

    x: DeviceBuffer,
    bias: Option<DeviceBuffer>,
    wloss: Vec<f32>,

    router_w: DeviceBuffer,
    router_grad: DeviceBuffer,
    expert_w: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
    expert_grad: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,

    logits: DeviceBuffer,
    gate: DeviceBuffer,
    probs: Option<DeviceBuffer>,
    acc: DeviceBuffer,
    /// Every expert's activations for every row, or (`compact`) for its routed rows only, rebuilt from each
    /// forward's routing.
    acts: RefCell<MoeActs>,
    compact: bool,

    d_gate: DeviceBuffer,
    d_router_logits: DeviceBuffer,
    fe: Option<DeviceBuffer>,
    d_expert_out: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_moe_acc: DeviceBuffer,
    d_x: DeviceBuffer,

    /// `Some` runs the experts through `model::glu` under this declared activation and layout (dense activations only).
    glu: Option<GluCase>,

    // shape sizes cached for read/write
    sizes: HashMap<String, usize>,
    // the last computed dx, exposed for a direct (non-CheckModel) spot-check
    last_dx: RefCell<Option<Vec<f32>>>,
}

/// The declared activation of a `model::glu` case and the per-expert buffers it saves and propagates through.
struct GluCase {
    spec: GluSpec,
    ids: GluIds,
    /// Per expert: the saved `gate_up` (`gate_pre` when separate), `up` (separate only), `h`, `expert_out`.
    saved: Vec<(DeviceBuffer, Option<DeviceBuffer>, DeviceBuffer, DeviceBuffer)>,
    d_gate_up: DeviceBuffer,
    /// The up gradient (separate), or an unused buffer distinct from `d_gate_up` (fused).
    d_up: DeviceBuffer,
}

impl GluCase {
    fn new(g: &Gpu, spec: GluSpec) -> GluCase {
        let (rows, ff, d) = (ROWS as u64, FF as u64, D as u64);
        let gu = if spec.is_fused() { 2 * ff } else { ff };
        let saved = (0..E)
            .map(|_| (g.storage(rows * gu), (!spec.is_fused()).then(|| g.storage(rows * ff)), g.storage(rows * ff), g.storage(rows * d)))
            .collect();
        GluCase {
            spec,
            ids: GluIds { act: idx(g, "glu_act"), act_bwd: idx(g, "glu_act_bwd") },
            saved,
            d_gate_up: g.storage(rows * gu),
            d_up: g.storage(if spec.is_fused() { 1 } else { rows * ff }),
        }
    }

    fn scratch(&self, e: usize) -> GluScratch<'_> {
        let (gate_up, up, h, expert_out) = &self.saved[e];
        GluScratch { gate_up, up: up.as_ref(), h, expert_out }
    }
}

impl MoeLayerCheck {
    fn new(kind: RouterKind, top_k: u32, seed: u64, compact: bool) -> MoeLayerCheck {
        Self::build(kind, top_k, seed, compact, None)
    }

    fn new_glu(kind: RouterKind, top_k: u32, seed: u64, spec: GluSpec) -> MoeLayerCheck {
        Self::build(kind, top_k, seed, false, Some(spec))
    }

    fn build(kind: RouterKind, top_k: u32, seed: u64, compact: bool, glu: Option<GluSpec>) -> MoeLayerCheck {
        let fused = glu.is_some_and(|s| s.is_fused());
        let pipes: Vec<(&str, &str)> = PIPES.iter().chain(COMPACT_TRAIN_KERNELS).copied().collect();
        let g = Gpu::new_cpu(&pipes);
        let shape = MoeShape { rows: ROWS, d_model: D, moe_ff: FF, n_experts: E, top_k };
        let fwd_ids = MoeIds {
            router_gate: idx(&g, if matches!(kind, RouterKind::Softmax { .. }) { "router_gate" } else { "router_gate_sigmoid" }),
            linear_gated: idx(&g, "moe_linear_gated"),
            silu_mul: idx(&g, "silu_mul"),
            scale_add: idx(&g, "scale_add"),
        };
        let bwd_ids = MoeIdsBwd {
            scale_add_dexp: idx(&g, "scale_add_dexp"),
            scale_add_dgate: idx(&g, "scale_add_dgate"),
            silu_da: idx(&g, "silu_bwd_da"),
            silu_db: idx(&g, "silu_bwd_db"),
            linear_dx: idx(&g, "moe_linear_gated_dx"),
            linear_dw: idx(&g, "moe_linear_gated_dw"),
            linear_gated: true,
        };
        let router_bwd_ids = RouterBwdIds {
            router_bwd: idx(&g, if matches!(kind, RouterKind::Softmax { .. }) { "router_bwd" } else { "router_bwd_sigmoid" }),
            expert_counts: matches!(kind, RouterKind::Softmax { .. }).then(|| idx(&g, "expert_counts")),
        };

        let mut r = Lcg::new(seed);
        let b = |g: &Gpu, v: &[f32]| g.storage_init("w", v);

        let x = b(&g, &r.vec_scaled((ROWS * D) as usize, 0.5));
        let bias = matches!(kind, RouterKind::SigmoidNoAuxTc { .. }).then(|| b(&g, &r.vec_scaled(E as usize, 0.3)));
        let wloss = r.vec_scaled((ROWS * D) as usize, 0.5);

        let router_w = b(&g, &r.vec_scaled((E * D) as usize, 0.4));
        let router_grad = g.storage((E * D) as u64);
        let mut expert_w = Vec::new();
        let mut expert_grad = Vec::new();
        for _ in 0..E {
            // A fused expert holds one `[2 * FF, D]` gate/up matrix in the first slot; the second slot is then unused.
            let gate_rows = if fused { 2 * FF } else { FF };
            let gw = b(&g, &r.vec_scaled((gate_rows * D) as usize, 0.4));
            let uw = if fused { g.storage(1) } else { b(&g, &r.vec_scaled((FF * D) as usize, 0.4)) };
            let dw = b(&g, &r.vec_scaled((D * FF) as usize, 0.4));
            expert_w.push((gw, uw, dw));
            expert_grad.push((g.storage((gate_rows * D) as u64), g.storage(if fused { 1 } else { (FF * D) as u64 }), g.storage((D * FF) as u64)));
        }

        let mut sizes = HashMap::new();
        sizes.insert("router.weight".to_string(), (E * D) as usize);
        for e in 0..E {
            sizes.insert(format!("expert{e}.gate_w"), ((if fused { 2 * FF } else { FF }) * D) as usize);
            if !fused {
                sizes.insert(format!("expert{e}.up_w"), (FF * D) as usize);
            }
            sizes.insert(format!("expert{e}.down_w"), (D * FF) as usize);
        }

        MoeLayerCheck {
            logits: g.storage((ROWS * E) as u64),
            gate: g.storage((ROWS * E) as u64),
            probs: matches!(kind, RouterKind::SigmoidNoAuxTc { .. }).then(|| g.storage((ROWS * E) as u64)),
            acc: g.storage((ROWS * D) as u64),
            acts: RefCell::new(MoeActs::new(&g, &shape)),
            compact,
            d_gate: g.storage((ROWS * E) as u64),
            d_router_logits: g.storage((ROWS * E) as u64),
            fe: matches!(kind, RouterKind::Softmax { .. }).then(|| g.storage(E as u64)),
            d_expert_out: g.storage((ROWS * D) as u64),
            d_h: g.storage((ROWS * FF) as u64),
            d_gate_pre: g.storage((ROWS * FF) as u64),
            d_up: g.storage((ROWS * FF) as u64),
            d_moe_acc: b(&g, &wloss), // loss = sum(acc .* wloss) => d(loss)/d(acc) = wloss, constant
            d_x: g.storage((ROWS * D) as u64),
            matmul: idx(&g, "matmul"),
            matmul_dx: idx(&g, "matmul_dx"),
            matmul_dw: idx(&g, "matmul_dw"),
            fwd_ids,
            bwd_ids,
            router_bwd_ids,
            x,
            bias,
            wloss,
            router_w,
            router_grad,
            expert_w,
            expert_grad,
            sizes,
            last_dx: RefCell::new(None),
            glu: glu.map(|spec| GluCase::new(&g, spec)),
            kind,
            shape,
            g,
        }
    }

    fn weight_buf(&self, name: &str) -> &DeviceBuffer {
        if name == "router.weight" {
            return &self.router_w;
        }
        let (e, field) = Self::parse_expert_name(name);
        match field {
            "gate_w" => &self.expert_w[e].0,
            "up_w" => &self.expert_w[e].1,
            "down_w" => &self.expert_w[e].2,
            _ => unreachable!(),
        }
    }

    fn grad_buf(&self, name: &str) -> &DeviceBuffer {
        if name == "router.weight" {
            return &self.router_grad;
        }
        let (e, field) = Self::parse_expert_name(name);
        match field {
            "gate_w" => &self.expert_grad[e].0,
            "up_w" => &self.expert_grad[e].1,
            "down_w" => &self.expert_grad[e].2,
            _ => unreachable!(),
        }
    }

    fn expert_weights(&self) -> Vec<ExpertWeights> {
        self.expert_w.iter().map(|(gw, uw, dw)| ExpertWeights { gate_w: gw.clone(), up_w: uw.clone(), down_w: dw.clone(), w_off: 0 }).collect()
    }

    fn parse_expert_name(name: &str) -> (usize, &str) {
        let rest = name.strip_prefix("expert").expect("param name");
        let (e_str, field) = rest.split_once('.').expect("param name");
        (e_str.parse().unwrap(), field)
    }
}

impl CheckModel for MoeLayerCheck {
    fn param_names(&self) -> Vec<String> {
        self.sizes.keys().cloned().collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.g.read(self.weight_buf(name), self.sizes[name])
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        self.g.write_f32(self.weight_buf(name), data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.g.read(self.grad_buf(name), self.sizes[name])
    }

    fn loss(&self) -> f32 {
        let g = &self.g;
        let mut steps = vec![g.step(self.matmul, &[&self.x, &self.router_w, &self.logits], &[ROWS, D, E], ROWS * E)];
        steps.push(router_fwd_kind(g, &self.fwd_ids, self.kind, &self.shape, &self.logits, self.bias.as_ref(), &self.gate, self.probs.as_ref()));
        g.submit(&[], &steps);

        if let Some(case) = &self.glu {
            for e in 0..E as usize {
                let (gw, uw, dw) = &self.expert_w[e];
                let w = GluWeights { gate_w: gw, up_w: (!case.spec.is_fused()).then_some(uw), down_w: dw, gate_off: 0, down_off: 0 };
                let steps = expert_fwd_glu(g, &self.fwd_ids, &case.ids, &case.spec, &self.shape, &self.x, &self.gate, &w, &case.scratch(e), &self.acc, e as u32, e != 0);
                g.submit(&[], &steps);
            }
        } else if self.compact {
            // Each expert holds and computes its routed rows only; every row's top-k experts add into `acc`.
            let acts = MoeActs::compact(g, &self.shape, &g.read(&self.gate, (ROWS * E) as usize));
            g.submit(&[&self.acc], &experts_fwd_train(g, &self.fwd_ids, &self.shape, &self.x, &self.gate, &self.expert_weights(), &acts, &self.acc));
            *self.acts.borrow_mut() = acts;
        } else {
            for e in 0..E as usize {
                let (gw, uw, dw) = &self.expert_w[e];
                let acts = self.acts.borrow();
                let scratch = acts.at(e);
                // `w_off = 0`: this harness gives every expert its own buffers.
                let fwd_steps = expert_fwd(g, &self.fwd_ids, &self.shape, &self.x, &self.gate, gw, uw, dw, &scratch, &self.acc, e as u32, 0, e != 0);
                g.submit(&[], &fwd_steps);
            }
        }

        let acc = g.read(&self.acc, (ROWS * D) as usize);
        let loss: f64 = acc.iter().zip(&self.wloss).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        loss as f32
    }

    fn zero_grads(&self) {
        let g = &self.g;
        g.write_f32(&self.router_grad, &vec![0.0; (E * D) as usize]);
        let fused = self.glu.as_ref().is_some_and(|c| c.spec.is_fused());
        for (gw, uw, dw) in &self.expert_grad {
            g.write_f32(gw, &vec![0.0; ((if fused { 2 * FF } else { FF }) * D) as usize]);
            g.write_f32(uw, &vec![0.0; if fused { 1 } else { (FF * D) as usize }]);
            g.write_f32(dw, &vec![0.0; (D * FF) as usize]);
        }
    }

    fn backward(&self) {
        let g = &self.g;
        // router weight's own dense-linear backward: FIRST write to d_x
        // (accumulate=0), caller-supplied per moe_layer_bwd's own doc.
        let router_weight_bwd = vec![
            g.step(self.matmul_dw, &[&self.d_router_logits, &self.x, &self.router_grad], &[ROWS, D, E], E * D),
            g.step(self.matmul_dx, &[&self.d_router_logits, &self.router_w, &self.d_x], &[ROWS, D, E, 0], ROWS * D),
        ];

        let expert_grads: Vec<ExpertGrads> = self.expert_grad.iter().map(|(gw, uw, dw)| ExpertGrads { gate_w: Some(gw), up_w: Some(uw), down_w: Some(dw) }).collect();
        let sb = ExpertBwdScratch { d_expert_out: &self.d_expert_out, d_h: &self.d_h, d_gate_pre: &self.d_gate_pre, d_up: &self.d_up };

        let expert_weights = self.expert_weights();
        // The compacted backward adds each expert's column into `d_gate`.
        g.submit(&[&self.d_gate], &[]);
        if let Some(case) = &self.glu {
            // `moe_layer_bwd`'s dense phases with the experts' phase C under the declared activation.
            let mut steps = Vec::new();
            for e in 0..E as usize {
                let s = case.scratch(e);
                let saved = ExpertScratch { gate_pre: s.gate_up, up: s.up.unwrap_or(s.gate_up), h: s.h, expert_out: s.expert_out };
                steps.push(expert_dgate(g, &self.bwd_ids, &self.shape, &saved, &self.d_moe_acc, &self.d_gate, e as u32));
            }
            steps.extend(router_bwd(g, &self.router_bwd_ids, self.kind, &self.shape, &self.logits, &self.gate, &self.d_gate, &self.d_router_logits, self.fe.as_ref()));
            steps.extend(router_weight_bwd);
            let sb = GluBwdScratch { d_expert_out: &self.d_expert_out, d_h: &self.d_h, d_gate_up: &case.d_gate_up, d_up: &case.d_up };
            let fused = case.spec.is_fused();
            for e in 0..E as usize {
                let ((gw, uw, dw), (ggw, guw, gdw)) = (&self.expert_w[e], &self.expert_grad[e]);
                let w = GluWeights { gate_w: gw, up_w: (!fused).then_some(uw), down_w: dw, gate_off: 0, down_off: 0 };
                let gr = GluGrads { gate_w: Some(ggw), up_w: (!fused).then_some(guw), down_w: Some(gdw) };
                expert_bwd_glu(g, &self.bwd_ids, &case.ids, &case.spec, &self.shape, &self.x, &self.gate, &w, &gr, &case.scratch(e), &sb, &self.d_moe_acc, &self.d_x, e as u32, true, &mut steps);
            }
            g.submit(&[], &steps);
            *self.last_dx.borrow_mut() = Some(g.read(&self.d_x, (ROWS * D) as usize));
            return;
        }
        let acts = self.acts.borrow();
        let steps = moe_layer_bwd(
            g,
            &self.router_bwd_ids,
            &self.bwd_ids,
            self.kind,
            &self.shape,
            &self.logits,
            &self.gate,
            self.fe.as_ref(),
            &self.d_gate,
            &self.d_router_logits,
            &router_weight_bwd,
            &self.x,
            &expert_weights,
            &expert_grads,
            &acts,
            &sb,
            &self.d_moe_acc,
            &self.d_x,
        );
        g.submit(&[], &steps);
        *self.last_dx.borrow_mut() = Some(g.read(&self.d_x, (ROWS * D) as usize));
    }
}

/// Also confirm the router-weight `.expect(...)`/panic paths never fire for
/// well-formed configs (compile-time proof by construction: this file would
/// not build if the API demanded something these calls don't supply) -- the
/// real assertions are the elementwise checks below.
///
/// The `rel_err < 5e-2 || abs_err < 1e-3` dual criterion (not a bare rel_err
/// bound) exists for a real, verified reason, not just generosity: top-k
/// expert selection is a HARD, discontinuous decision -- an `eps=5e-3`
/// perturbation of `router.weight` occasionally flips which expert wins a
/// borderline row's boundary, and the resulting numeric derivative silently
/// straddles that flip. Confirmed by direct inspection (not assumed): every
/// `Softmax` case's worst offenders are exclusively `router.weight` entries
/// with tiny analytic magnitude (~1e-4) and correspondingly tiny absolute
/// error (~1e-4) -- never an expert weight, and never large in an absolute
/// sense. Expert weights and the SigmoidNoAuxTc router both clear rel_err
/// ~2e-4 everywhere in the same run. A REAL bug would show up as either a
/// large absolute error, or a non-router-weight parameter failing -- this
/// gate would catch that; it does not paper over it.
fn run_case(kind: RouterKind, top_k: u32, seed: u64) {
    run_case_on(kind, top_k, seed, false);
}

fn run_case_on(kind: RouterKind, top_k: u32, seed: u64, compact: bool) {
    check_model(MoeLayerCheck::new(kind, top_k, seed, compact), kind, top_k);
}

fn check_model(m: MoeLayerCheck, kind: RouterKind, top_k: u32) {
    let mut checked = 0usize;
    let mut worst_rel = 0.0f32;
    for name in m.param_names() {
        let report = elementwise_check(&m, &name, 5e-3);
        for c in &report.checks {
            worst_rel = worst_rel.max(c.rel_err);
            assert!(
                c.rel_err < 5e-2 || c.abs_err < 1e-3,
                "{}: analytic {} vs numeric {} (rel {:.4}, abs {:.4}) [kind={}, top_k={}]",
                c.param,
                c.analytic,
                c.numeric,
                c.rel_err,
                c.abs_err,
                kind_name(kind),
                top_k
            );
        }
        checked += report.checks.len();
    }
    assert!(checked > 50, "too few gradients compared ({checked})");
    eprintln!("{}(top_k={top_k}): {checked} grads checked, worst rel_err {worst_rel:.4}", kind_name(kind));
}

fn kind_name(k: RouterKind) -> &'static str {
    match k {
        RouterKind::Softmax { .. } => "Softmax",
        RouterKind::SigmoidNoAuxTc { .. } => "SigmoidNoAuxTc",
    }
}

// ---- Softmax router (Qwen3-Omni Thinker/Talker's kind) ----

#[test]
fn softmax_router_gradcheck_top_k_less_than_n_experts() {
    // top_k < n_experts: exercises moe_linear_gated's early-exit branch for
    // real -- the branch a top_k==n_experts config can never take.
    run_case(RouterKind::Softmax { aux_coef: 0.01, z_coef: 0.001, norm_topk_prob: true, routed_scaling: 1.0 }, 2, 0xA0FF);
}

#[test]
fn softmax_router_gradcheck_top_k_equals_n_experts() {
    // top_k == n_experts: every row selects every expert (dense-equivalent
    // case) -- the shape check_moe/check_glm's tiny configs actually run.
    run_case(RouterKind::Softmax { aux_coef: 0.01, z_coef: 0.001, norm_topk_prob: true, routed_scaling: 1.0 }, E, 0xA0FE);
}

#[test]
fn softmax_router_gradcheck_unnormalized_and_scaled() {
    // norm_topk_prob=false + routed_scaling != 1: the second real combine-weight
    // branch of `router_gate.wgsl`/`router_bwd.wgsl`, added for DeepSeek-OCR's
    // decoder. The backward is NOT the normalised form with `Z := 1` -- it has
    // no `sdp/Z^2` cross-term at all (see `router_bwd.wgsl`'s header), so this
    // is the gate on a genuinely different expression, not a parameter sweep.
    //
    // `top_k == E`: with the renormalisation gone the gate weights are the raw
    // softmax probabilities, smooth in every direction, so FD is well
    // conditioned -- the same reason `check_moe` pins `top_k == n_experts`.
    run_case(RouterKind::Softmax { aux_coef: 0.01, z_coef: 0.001, norm_topk_prob: false, routed_scaling: 1.5 }, E, 0xA0FD);
}

// ---- SigmoidNoAuxTc router (GLM-5.2/DeepSeek-V3's kind) ----

#[test]
fn sigmoid_noaux_router_gradcheck_top_k_less_than_n_experts() {
    run_case(RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 }, 2, 0x519F);
}

#[test]
fn sigmoid_noaux_router_gradcheck_top_k_equals_n_experts() {
    run_case(RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 }, E, 0x519E);
}

#[test]
fn sigmoid_noaux_router_gradcheck_unnormalized() {
    // norm_topk_prob=false: a second real combine-weight branch
    // (router_gate_sigmoid.wgsl / router_bwd_sigmoid.wgsl's `norm` flag).
    run_case(RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: false, routed_scaling: 1.0 }, 2, 0x519D);
}

// ---- row-compacted activations (each expert holds its routed rows only) ----

/// The compacted training path is the same function: its gradients pass the finite-difference check, for both router
/// kinds with `top_k < n_experts` (where compaction drops rows).
#[test]
fn compacted_experts_pass_the_gradcheck() {
    run_case_on(RouterKind::Softmax { aux_coef: 0.01, z_coef: 0.001, norm_topk_prob: true, routed_scaling: 1.0 }, 2, 0xC0A1, true);
    run_case_on(RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 }, 2, 0xC0A2, true);
}

/// The compacted path computes what the dense one does: the layer output, every weight gradient and the input
/// gradient agree, while the saved activations shrink to the routed rows.
#[test]
fn compacted_experts_match_the_dense_path() {
    let kind = RouterKind::Softmax { aux_coef: 0.01, z_coef: 0.001, norm_topk_prob: true, routed_scaling: 1.0 };
    let (dense, compact) = (MoeLayerCheck::new(kind, 2, 0xC0A3, false), MoeLayerCheck::new(kind, 2, 0xC0A3, true));
    let (want, got) = (dense.loss(), compact.loss());
    assert!((want - got).abs() <= 1e-5 * want.abs().max(1.0), "loss {got} vs {want}");
    for m in [&dense, &compact] {
        m.zero_grads();
        m.backward();
    }
    for name in dense.param_names() {
        for (i, (a, b)) in dense.read_grad(&name).iter().zip(compact.read_grad(&name)).enumerate() {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1e-3), "{name}[{i}]: compact {b} vs dense {a}");
        }
    }
    let (dx, cx) = (dense.last_dx.borrow().clone().unwrap(), compact.last_dx.borrow().clone().unwrap());
    for (i, (a, b)) in dx.iter().zip(&cx).enumerate() {
        assert!((a - b).abs() <= 1e-5 * a.abs().max(1e-3), "d_x[{i}]: compact {b} vs dense {a}");
    }
}

// ---- declared expert activation and gate/up layout (`model::glu`) ----

/// Small caps keep the softcap in its nonlinear range at this harness's activation magnitudes (a few tenths), so the
/// check exercises `1 - tanh^2`, not the identity the caps degenerate to when large.
const SOFTCAP: GatedAct = GatedAct::SoftcapSigmoid { gate_cap: 0.6, up_cap: 0.9 };

fn glu_case(act: GatedAct, layout: GluLayout, seed: u64) {
    let kind = RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 };
    check_model(MoeLayerCheck::new_glu(kind, 2, seed, GluSpec { act, layout }), kind, 2);
}

#[test]
fn swiglu_over_a_concat_fused_bank_passes_the_gradcheck() {
    glu_case(GatedAct::SwiGlu, GluLayout::Concat, 0x61A1);
}

#[test]
fn swiglu_over_an_interleaved_fused_bank_passes_the_gradcheck() {
    glu_case(GatedAct::SwiGlu, GluLayout::Interleaved, 0x61A2);
}

#[test]
fn softcap_sigmoid_over_separate_projections_passes_the_gradcheck() {
    glu_case(SOFTCAP, GluLayout::Separate, 0x61A3);
}

#[test]
fn softcap_sigmoid_over_a_concat_fused_bank_passes_the_gradcheck() {
    glu_case(SOFTCAP, GluLayout::Concat, 0x61A4);
}

#[test]
fn softcap_sigmoid_over_an_interleaved_fused_bank_passes_the_gradcheck() {
    glu_case(SOFTCAP, GluLayout::Interleaved, 0x61A5);
}

/// The default spec is today's SwiGLU over separate projections, bit for bit: the layer output and every gradient equal
/// `expert_fwd` / `moe_layer_bwd`'s.
#[test]
fn default_glu_spec_reproduces_the_existing_expert_path_exactly() {
    let kind = RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 };
    let (old, new) = (MoeLayerCheck::new(kind, 2, 0x61A6, false), MoeLayerCheck::new_glu(kind, 2, 0x61A6, GluSpec::default()));
    assert_eq!(old.loss().to_bits(), new.loss().to_bits(), "loss differs");
    for m in [&old, &new] {
        m.zero_grads();
        m.backward();
    }
    for name in old.param_names() {
        let (a, b) = (old.read_grad(&name), new.read_grad(&name));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: gradient differs from the existing path");
    }
    assert_eq!(old.last_dx.borrow().as_ref().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()), new.last_dx.borrow().as_ref().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()), "d_x differs");
}

/// The two fused layouts are the same function of a permuted weight: an interleaved bank holds the concat bank's rows
/// in the order g0,u0,g1,u1,..., and gives the same layer output.
#[test]
fn interleaved_bank_computes_what_the_permuted_concat_bank_computes() {
    let kind = RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 1.5 };
    let concat = MoeLayerCheck::new_glu(kind, 2, 0x61A7, GluSpec { act: SOFTCAP, layout: GluLayout::Concat });
    let inter = MoeLayerCheck::new_glu(kind, 2, 0x61A7, GluSpec { act: SOFTCAP, layout: GluLayout::Interleaved });
    for e in 0..E {
        let name = format!("expert{e}.gate_w");
        let rows = concat.read_weight(&name); // [2 * FF, D]: gate rows then up rows
        let (ff, d) = (FF as usize, D as usize);
        let mut permuted = vec![0.0f32; rows.len()];
        for j in 0..ff {
            permuted[(2 * j) * d..(2 * j + 1) * d].copy_from_slice(&rows[j * d..(j + 1) * d]);
            permuted[(2 * j + 1) * d..(2 * j + 2) * d].copy_from_slice(&rows[(ff + j) * d..(ff + j + 1) * d]);
        }
        inter.write_weight(&name, &permuted);
        inter.write_weight(&format!("expert{e}.down_w"), &concat.read_weight(&format!("expert{e}.down_w")));
    }
    let (a, b) = (concat.loss(), inter.loss());
    assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "interleaved bank gives {b}, concat bank {a}");
}
