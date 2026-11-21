// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of routing replay in the MoE router forward (`model::moe::router_fwd_replay`, `router_gate_replay.wgsl`):
//! forcing a recorded top-k selection keeps the router's own weight formula over the current logits at the recorded
//! experts, leaves unrecorded rows as the router wrote them, reproduces the router's own gate to the bit when the
//! recorded selection is its own, and is differentiated correctly by the unchanged router backward (finite
//! differences of the device forward against the device backward, and an f64 reference).
//!
//! Swedish Embedded AB implements solutions for reinforcement learning on mixture-of-experts models that trains the
//! experts the serving engine actually ran for its clients. If your team needs expertise in MoE training numerics then
//! you can procure our services by sending an email to info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use gpu_core::{DeviceBuffer, Gpu};
use model::moe::{router_bwd, router_fwd_kind, router_fwd_replay, MoeIds, MoeShape, RouterBwdIds, RouterKind, RouterOverride, ROUTER_REPLAY_KERNEL};
use model::routing::{upload_experts, UNRECORDED};

const PIPES: &[(&str, &str)] = &[
    ("router_gate", kernels::ROUTER_GATE),
    ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
    ("router_bwd", kernels::ROUTER_BWD),
    ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
    ("expert_counts", kernels::EXPERT_COUNTS),
    ROUTER_REPLAY_KERNEL,
];

const E: usize = 16;
const K: usize = 4;
const ROWS: usize = 6;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel {name} not registered"))
}

fn kinds() -> [RouterKind; 3] {
    [
        RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 },
        RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: false, routed_scaling: 2.5 },
        RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 8.0 },
    ]
}

fn is_sigmoid(kind: RouterKind) -> bool {
    matches!(kind, RouterKind::SigmoidNoAuxTc { .. })
}

struct Router {
    g: Gpu,
    kind: RouterKind,
    shape: MoeShape,
    bias: DeviceBuffer,
}

impl Router {
    fn new(kind: RouterKind, bias: &[f32]) -> Router {
        let g = gpu_core::testgpu::dev(PIPES);
        let shape = MoeShape { rows: ROWS as u32, d_model: 1, moe_ff: 1, n_experts: E as u32, top_k: K as u32 };
        let bias = g.storage_init("bias", bias);
        Router { g, kind, shape, bias }
    }

    fn ids(&self) -> MoeIds {
        let gate = if is_sigmoid(self.kind) { "router_gate_sigmoid" } else { "router_gate" };
        MoeIds { router_gate: idx(&self.g, gate), linear_gated: 0, silu_mul: 0, scale_add: 0 }
    }

    /// The gate of `logits`: the router's own, then (when `forced` is given) the recorded selection forced on it.
    fn gate(&self, logits: &[f32], forced: Option<&[u32]>) -> Vec<f32> {
        let g = &self.g;
        let lb = g.storage_init("logits", logits);
        let (gate, probs) = (g.storage((ROWS * E) as u64), g.storage((ROWS * E) as u64));
        let mut steps = vec![router_fwd_kind(g, &self.ids(), self.kind, &self.shape, &lb, Some(&self.bias), &gate, Some(&probs))];
        let experts = forced.map(|f| upload_experts(g, f));
        if let Some(experts) = &experts {
            let over = RouterOverride { kernel: idx(g, ROUTER_REPLAY_KERNEL.0), experts };
            steps.push(router_fwd_replay(g, over, self.kind, &self.shape, &lb, Some(&self.bias), &gate));
        }
        g.submit(&[], &steps);
        g.read(&gate, ROWS * E)
    }

    /// `d loss / d logits` through the router backward, for `loss = sum d_gate * gate` with the gate `gate`.
    fn backward(&self, logits: &[f32], gate: &[f32], d_gate: &[f32]) -> Vec<f32> {
        let g = &self.g;
        let (lb, gb, db) = (g.storage_init("logits", logits), g.storage_init("gate", gate), g.storage_init("d_gate", d_gate));
        let (dl, fe) = (g.storage((ROWS * E) as u64), g.storage(E as u64));
        let (bwd, counts) = if is_sigmoid(self.kind) { (idx(g, "router_bwd_sigmoid"), None) } else { (idx(g, "router_bwd"), Some(idx(g, "expert_counts"))) };
        let ids = RouterBwdIds { router_bwd: bwd, expert_counts: counts };
        g.submit(&[], &router_bwd(g, &ids, self.kind, &self.shape, &lb, &gb, &db, &dl, Some(&fe)));
        g.read(&dl, ROWS * E)
    }
}

/// The experts a dense gate row selected, ascending.
fn selected(gate: &[f32], row: usize) -> Vec<u32> {
    (0..E).filter(|&e| gate[row * E + e] != 0.0).map(|e| e as u32).collect()
}

/// `K` distinct experts per row, different from `avoid`'s where possible, in an arbitrary order.
fn other_selection(rng: &mut Lcg, avoid: &[f32]) -> Vec<u32> {
    let mut out = Vec::new();
    for row in 0..ROWS {
        let own = selected(avoid, row);
        let mut pool: Vec<u32> = (0..E as u32).filter(|e| !own.contains(e)).collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, (rng.unit() * (i + 1) as f32) as usize % (i + 1));
        }
        // keep one of the router's own experts so the forced set overlaps it
        let mut pick = vec![own[0]];
        pick.extend_from_slice(&pool[..K - 1]);
        pick.reverse();
        out.extend(pick);
    }
    out
}

/// The router's weights in f64 at the forced experts of one row (selection fixed).
fn reference(kind: RouterKind, logits: &[f64], forced: &[u32]) -> Vec<f64> {
    match kind {
        RouterKind::Softmax { norm_topk_prob, routed_scaling, .. } => {
            let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|l| (l - mx).exp()).sum();
            let p: Vec<f64> = forced.iter().map(|&e| (logits[e as usize] - mx).exp() / sum).collect();
            let z: f64 = if norm_topk_prob { p.iter().sum::<f64>().max(1e-9) } else { 1.0 };
            p.iter().map(|v| v / z * f64::from(routed_scaling)).collect()
        }
        RouterKind::SigmoidNoAuxTc { norm_topk_prob, routed_scaling, .. } => {
            let s: Vec<f64> = forced.iter().map(|&e| 1.0 / (1.0 + (-logits[e as usize]).exp())).collect();
            let z: f64 = if norm_topk_prob { s.iter().sum::<f64>() + 1e-20 } else { 1.0 };
            s.iter().map(|v| v / z * f64::from(routed_scaling)).collect()
        }
    }
}

fn close(got: f64, want: f64, abs: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= abs + rel * want.abs(), "{what}: {got} vs {want}");
}

#[test]
fn replaying_the_routers_own_selection_reproduces_its_gate_bit_for_bit() {
    let mut rng = Lcg::new(random_seed());
    for kind in kinds() {
        let r = Router::new(kind, &rng.vec_scaled(E, 0.3));
        let logits = rng.vec_scaled(ROWS * E, 3.0);
        let own = r.gate(&logits, None);
        // the record lists the experts in ascending order, not in the order the router picked them
        let record: Vec<u32> = (0..ROWS).flat_map(|row| selected(&own, row)).collect();
        assert_eq!(record.len(), ROWS * K);
        let replayed = r.gate(&logits, Some(&record));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&replayed), bits(&own), "{}: replaying the router's own selection must not move a bit", if is_sigmoid(kind) { "sigmoid" } else { "softmax" });
    }
}

#[test]
fn a_forced_selection_takes_the_routers_formula_and_unrecorded_rows_keep_their_own() {
    let mut rng = Lcg::new(random_seed());
    for kind in kinds() {
        let r = Router::new(kind, &rng.vec_scaled(E, 0.3));
        let logits = rng.vec_scaled(ROWS * E, 3.0);
        let own = r.gate(&logits, None);
        let mut forced = other_selection(&mut rng, &own);
        // the last row is not recorded
        forced[(ROWS - 1) * K..].fill(UNRECORDED);
        let gate = r.gate(&logits, Some(&forced));
        for row in 0..ROWS - 1 {
            let ids = &forced[row * K..(row + 1) * K];
            let mut sorted = ids.to_vec();
            sorted.sort_unstable();
            assert_eq!(selected(&gate, row), sorted, "row {row}: the gate selects exactly the recorded experts");
            let l64: Vec<f64> = logits[row * E..(row + 1) * E].iter().map(|&v| f64::from(v)).collect();
            for (&e, w) in ids.iter().zip(reference(kind, &l64, ids)) {
                close(f64::from(gate[row * E + e as usize]), w, 1e-7, 1e-5, &format!("row {row} expert {e}"));
            }
        }
        let last = (ROWS - 1) * E;
        assert_eq!(gate[last..], own[last..], "an unrecorded row keeps the router's own gate");
    }
}

#[test]
fn the_router_backward_differentiates_the_forced_gate() {
    let mut rng = Lcg::new(random_seed());
    for kind in kinds() {
        let r = Router::new(kind, &rng.vec_scaled(E, 0.3));
        let logits = rng.vec_scaled(ROWS * E, 2.0);
        let forced = other_selection(&mut rng, &r.gate(&logits, None));
        let d_gate = rng.vec_scaled(ROWS * E, 1.0);
        let gate = r.gate(&logits, Some(&forced));
        let dlogits = r.backward(&logits, &gate, &d_gate);
        let loss_dev = |l: &[f32]| -> f64 { r.gate(l, Some(&forced)).iter().zip(&d_gate).map(|(g, d)| f64::from(*g) * f64::from(*d)).sum() };
        for row in 0..ROWS {
            let ids = &forced[row * K..(row + 1) * K];
            let l64: Vec<f64> = logits[row * E..(row + 1) * E].iter().map(|&v| f64::from(v)).collect();
            let loss_ref = |l: &[f64]| -> f64 { ids.iter().zip(reference(kind, l, ids)).map(|(&e, w)| w * f64::from(d_gate[row * E + e as usize])).sum() };
            for e in 0..E {
                // the f64 reference, selection fixed
                let h = 1e-5;
                let (mut up, mut dn) = (l64.clone(), l64.clone());
                up[e] += h;
                dn[e] -= h;
                let fd = (loss_ref(&up) - loss_ref(&dn)) / (2.0 * h);
                let got = f64::from(dlogits[row * E + e]);
                close(got, fd, 1e-5, 2e-3, &format!("row {row} logit {e} against the f64 reference"));
                // the device forward itself, central differences in f32
                if e % 5 == row % 5 {
                    let h = 1e-2f32;
                    let (mut up, mut dn) = (logits.clone(), logits.clone());
                    up[row * E + e] += h;
                    dn[row * E + e] -= h;
                    let fd = (loss_dev(&up) - loss_dev(&dn)) / (2.0 * f64::from(h));
                    close(got, fd, 2e-3, 2e-2, &format!("row {row} logit {e} against the device forward"));
                }
            }
        }
    }
}
