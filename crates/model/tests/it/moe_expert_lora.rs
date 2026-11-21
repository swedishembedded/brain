// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-rank fine-tuning of quantized mixture-of-experts models on their published weights
// for its clients. If your team needs expertise in training adapters over frozen NVFP4 and MXFP8 expert banks then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: stacked-expert LoRA (per expert `e`, `A_e [r, k]` and `B_e [n, r]` over an `[E, n, k]` bank) trains through the
//! row-compacted MoE path over a frozen microscaled bank ([`experts_fwd_train_adapted`], [`moe_layer_bwd_adapted`]):
//! - a softcap-sigmoid expert over a row-interleaved fused gate/up bank behind a sigmoid router, adapted on gate_up and
//!   down: the layer output is the f64 forward over the dequantized weights plus the adapters;
//! - every A/B entry's gradient is the f64 central difference of that forward (elementwise, every entry), ADDED into the
//!   gradient buffers; an expert no row routes to keeps its gradient buffers bit for bit;
//! - the input gradient, including the adapters' share, is the f64 central difference with the router differentiated
//!   and held constant;
//! - adapters the layer cannot take are refused by name.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::glu::{GatedAct, GluBwdScratch, GluLayout, GluSpec, GluWeights};
use model::moe::{
    experts_fwd_train_adapted, moe_layer_bwd_adapted, router_fwd_kind, ExpertAdapterGrads, ExpertAdapters, ExpertBank, ExpertLora, ExpertLoraGrads, ExpertSet, MoeActs, MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds,
    RouterGrad, RouterKind, COMPACT_TRAIN_KERNELS, EXPERT_LORA_KERNELS,
};
use model::ops::{micro_bank_kernels, MicroBank, MicroScaled};

use crate::tier_kit::build_bank;

const BASE: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
    ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
    ("scale_add", kernels::SCALE_ADD),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
];

fn device() -> Gpu {
    let mut pipes: Vec<(&'static str, &'static str)> = Vec::new();
    for &(name, src) in BASE.iter().chain(COMPACT_TRAIN_KERNELS).chain(EXPERT_LORA_KERNELS) {
        if !pipes.iter().any(|(n, _)| *n == name) {
            pipes.push((name, src));
        }
    }
    pipes.extend(micro_bank_kernels());
    gpu_core::testgpu::dev(Box::leak(pipes.into_boxed_slice()))
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

const ROWS: u32 = 7;
const D: u32 = 128;
const FF: u32 = 128;
const E: u32 = 4;
const TOP_K: u32 = 2;
const RANK: u32 = 2;
/// `alpha / rank` of both adapters.
const LORA_SCALE: f32 = 1.5;
const ROUTE_SCALE: f32 = 2.5;
/// Never routed: its selection bias sits below every other expert's score.
const IDLE: usize = 3;
const KIND: RouterKind = RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: ROUTE_SCALE };
const ACT: GatedAct = GatedAct::SoftcapSigmoid { gate_cap: 0.15, up_cap: 0.2 };

/// One stacked adapter on the host: `a [E, RANK, k]`, `b [E, n, RANK]`.
#[derive(Clone)]
struct HostLora {
    a: Vec<f32>,
    b: Vec<f32>,
    n: usize,
    k: usize,
}

impl HostLora {
    fn random(n: usize, k: usize, rng: &mut Lcg) -> HostLora {
        let r = RANK as usize;
        HostLora { a: rng.vec_scaled(E as usize * r * k, 0.1), b: rng.vec_scaled(E as usize * n * r, 0.1), n, k }
    }

    /// `scale * B_j (A_j v)`, in f64.
    fn delta(&self, j: usize, v: &[f64]) -> Vec<f64> {
        let r = RANK as usize;
        let t: Vec<f64> = (0..r).map(|q| (0..self.k).map(|i| f64::from(self.a[(j * r + q) * self.k + i]) * v[i]).sum()).collect();
        (0..self.n).map(|o| f64::from(LORA_SCALE) * (0..r).map(|q| f64::from(self.b[(j * self.n + o) * r + q]) * t[q]).sum::<f64>()).collect()
    }
}

/// The layer's inputs, the dequantized weights of its frozen banks and its adapters.
struct Layer {
    x: Vec<f32>,
    router_w: Vec<f32>,
    bias: Vec<f32>,
    wloss: Vec<f32>,
    /// `[E * 2 * FF, D]`, rows interleaved `g0, u0, g1, u1, ...`.
    gate_up: Vec<f32>,
    /// `[E * D, FF]`.
    down: Vec<f32>,
    lora_gate_up: HostLora,
    lora_down: HostLora,
}

type Routing = Vec<Vec<(usize, f64)>>;

fn softcap(v: f64, c: f64) -> f64 {
    c * (v / c).tanh()
}

fn route(l: &Layer, xr: &[f64]) -> Vec<(usize, f64)> {
    let (d, e) = (D as usize, E as usize);
    let score: Vec<f64> = (0..e).map(|j| 1.0 / (1.0 + (-(0..d).map(|i| f64::from(l.router_w[j * d + i]) * xr[i]).sum::<f64>()).exp())).collect();
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &b| (score[b] + f64::from(l.bias[b])).total_cmp(&(score[a] + f64::from(l.bias[a]))));
    let chosen = &order[..TOP_K as usize];
    let sum: f64 = chosen.iter().map(|&j| score[j]).sum::<f64>() + 1e-20;
    chosen.iter().map(|&j| (j, score[j] / sum * f64::from(ROUTE_SCALE))).collect()
}

/// Row `r`'s loss share through expert `j` at routing weight `w`: the adapted softcap-sigmoid FFN over the interleaved
/// bank, dotted with the row's loss weights.
fn expert_row_loss(l: &Layer, gu: &HostLora, dn: &HostLora, r: usize, xr: &[f64], j: usize, w: f64) -> f64 {
    let (d, ff) = (D as usize, FF as usize);
    let GatedAct::SoftcapSigmoid { gate_cap, up_cap } = ACT else { unreachable!() };
    let delta = gu.delta(j, xr);
    let proj = |row: usize| (0..d).map(|i| f64::from(l.gate_up[(j * 2 * ff + row) * d + i]) * xr[i]).sum::<f64>() + delta[row];
    let h: Vec<f64> = (0..ff)
        .map(|c| {
            let (gv, uv) = (proj(2 * c), proj(2 * c + 1));
            softcap(gv, f64::from(gate_cap)) / (1.0 + (-gv).exp()) * softcap(uv, f64::from(up_cap))
        })
        .collect();
    let dd = dn.delta(j, &h);
    (0..d).map(|o| ((0..ff).map(|c| f64::from(l.down[(j * d + o) * ff + c]) * h[c]).sum::<f64>() + dd[o]) * f64::from(l.wloss[r * d + o])).sum::<f64>() * w
}

/// The f64 loss `<wloss, out>` and the routing; `fixed` holds the routing constant.
fn host_forward(l: &Layer, x: &[f64], fixed: Option<&Routing>) -> (f64, Routing) {
    let d = D as usize;
    let mut routing = Vec::new();
    let mut loss = 0.0;
    for r in 0..ROWS as usize {
        let xr = &x[r * d..(r + 1) * d];
        let rt = fixed.map_or_else(|| route(l, xr), |f| f[r].clone());
        loss += rt.iter().map(|&(j, w)| expert_row_loss(l, &l.lora_gate_up, &l.lora_down, r, xr, j, w)).sum::<f64>();
        routing.push(rt);
    }
    (loss, routing)
}

/// The loss share of expert `j` under `routing`, with the adapters given: what one of `j`'s factor entries moves.
fn expert_loss(l: &Layer, gu: &HostLora, dn: &HostLora, x: &[f64], routing: &Routing, j: usize) -> f64 {
    let d = D as usize;
    routing.iter().enumerate().map(|(r, rt)| rt.iter().filter(|&&(e, _)| e == j).map(|&(_, w)| expert_row_loss(l, gu, dn, r, &x[r * d..(r + 1) * d], j, w)).sum::<f64>()).sum()
}

/// Every entry of one adapter's factor (`which`: gate_up or down, `a` or `b`) by f64 central differences: the routing
/// does not depend on the adapters, so only the owning expert's share moves.
fn host_factor_grad(l: &Layer, routing: &Routing, down: bool, factor_a: bool) -> Vec<f64> {
    let x: Vec<f64> = l.x.iter().map(|&v| f64::from(v)).collect();
    let base = if down { &l.lora_down } else { &l.lora_gate_up };
    let len = if factor_a { base.a.len() } else { base.b.len() };
    let per_expert = len / E as usize;
    let eps = 1e-3;
    (0..len)
        .map(|i| {
            let j = i / per_expert;
            let eval = |delta: f32| {
                let mut p = base.clone();
                let v = if factor_a { &mut p.a[i] } else { &mut p.b[i] };
                *v += delta;
                if down { expert_loss(l, &l.lora_gate_up, &p, &x, routing, j) } else { expert_loss(l, &p, &l.lora_down, &x, routing, j) }
            };
            // Perturb in f32 storage and divide by the step actually taken.
            let v0 = if factor_a { base.a[i] } else { base.b[i] };
            let (up, dn) = ((v0 + eps) - v0, v0 - (v0 - eps));
            (eval(up) - eval(-dn)) / f64::from(up + dn)
        })
        .collect()
}

fn selected(routing: &Routing) -> Vec<Vec<usize>> {
    routing.iter().map(|r| r.iter().map(|&(j, _)| j).collect()).collect()
}

/// `d loss / d x[i]` for `picks` by f64 central differences; `None` where a perturbation flips a row's selection.
fn host_dx(l: &Layer, fixed: Option<&Routing>, picks: &[usize]) -> Vec<(usize, Option<f64>)> {
    let x: Vec<f64> = l.x.iter().map(|&v| f64::from(v)).collect();
    let (_, base) = host_forward(l, &x, None);
    let eps = 1e-5;
    picks
        .iter()
        .map(|&i| {
            let (mut up, mut dn) = (x.clone(), x.clone());
            up[i] += eps;
            dn[i] -= eps;
            let ((lu, ru), (ld, rd)) = (host_forward(l, &up, fixed), host_forward(l, &dn, fixed));
            (i, (fixed.is_some() || (selected(&ru) == selected(&base) && selected(&rd) == selected(&base))).then(|| (lu - ld) / (2.0 * eps)))
        })
        .collect()
}

/// The device side: router, gate, the frozen banks, the adapters and their gradient buffers (pre-filled with `prior`).
struct DeviceLayer {
    g: Gpu,
    shape: MoeShape,
    x: DeviceBuffer,
    router_w: DeviceBuffer,
    bias: DeviceBuffer,
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    probs: DeviceBuffer,
    acc: DeviceBuffer,
    gate_up: MicroBank,
    down: MicroBank,
    spec: GluSpec,
    /// gate_up A, gate_up B, down A, down B.
    factors: [DeviceBuffer; 4],
}

impl DeviceLayer {
    fn new(g: &Gpu, kind: MicroScaled, rng: &mut Lcg) -> (DeviceLayer, Layer) {
        let (gate_up, gate_up_w) = build_bank(g, kind, E as usize, 2 * FF as usize, D as usize, rng);
        let (down, down_w) = build_bank(g, kind, E as usize, D as usize, FF as usize, rng);
        let mut bias = rng.vec_scaled(E as usize, 0.05);
        bias[IDLE] = -10.0;
        let host = Layer {
            x: rng.vec_scaled((ROWS * D) as usize, 1.0),
            router_w: rng.vec_scaled((E * D) as usize, 0.15),
            bias,
            wloss: rng.vec_scaled((ROWS * D) as usize, 1.0),
            gate_up: gate_up_w,
            down: down_w,
            lora_gate_up: HostLora::random(2 * FF as usize, D as usize, rng),
            lora_down: HostLora::random(D as usize, FF as usize, rng),
        };
        let factors = [&host.lora_gate_up.a, &host.lora_gate_up.b, &host.lora_down.a, &host.lora_down.b].map(|v| g.storage_init("lora", v));
        let dev = DeviceLayer {
            shape: MoeShape { rows: ROWS, d_model: D, moe_ff: FF, n_experts: E, top_k: TOP_K },
            x: g.storage_init("x", &host.x),
            router_w: g.storage_init("router_w", &host.router_w),
            bias: g.storage_init("bias", &host.bias),
            logits: g.storage((ROWS * E) as u64),
            gate: g.storage((ROWS * E) as u64),
            probs: g.storage((ROWS * E) as u64),
            acc: g.storage((ROWS * D) as u64),
            gate_up,
            down,
            spec: GluSpec { act: ACT, layout: GluLayout::Interleaved },
            factors,
            g: g.share(),
        };
        (dev, host)
    }

    fn set(&self) -> ExpertSet<'_> {
        ExpertSet { glu: self.spec, bank: ExpertBank::Micro { gate_up: &self.gate_up, up: None, down: &self.down } }
    }

    fn adapters(&self) -> ExpertAdapters<'_> {
        let f = &self.factors;
        ExpertAdapters {
            gate_up: Some(ExpertLora { a: &f[0], b: &f[1], rank: RANK, scale: LORA_SCALE }),
            up: None,
            down: Some(ExpertLora { a: &f[2], b: &f[3], rank: RANK, scale: LORA_SCALE }),
        }
    }

    fn fwd_ids(&self) -> MoeIds {
        let g = &self.g;
        MoeIds { router_gate: idx(g, "router_gate_sigmoid"), linear_gated: idx(g, "moe_linear_gated"), silu_mul: idx(g, "silu_mul"), scale_add: idx(g, "scale_add") }
    }

    fn bwd_ids(&self) -> MoeIdsBwd {
        let g = &self.g;
        MoeIdsBwd {
            scale_add_dexp: idx(g, "scale_add_dexp"),
            scale_add_dgate: idx(g, "scale_add_dgate"),
            silu_da: idx(g, "silu_bwd_da"),
            silu_db: idx(g, "silu_bwd_db"),
            linear_dx: idx(g, "moe_linear_gated_dx"),
            linear_dw: idx(g, "moe_linear_gated_dw"),
            linear_gated: true,
        }
    }

    fn forward(&self) -> (Vec<f32>, MoeActs) {
        let g = &self.g;
        let ids = self.fwd_ids();
        let steps = vec![
            g.step(idx(g, "matmul"), &[&self.x, &self.router_w, &self.logits], &[ROWS, D, E], ROWS * E),
            router_fwd_kind(g, &ids, KIND, &self.shape, &self.logits, Some(&self.bias), &self.gate, Some(&self.probs)),
        ];
        g.submit(&[], &steps);
        let acts = MoeActs::compact_for(g, &self.shape, &g.read(&self.gate, (ROWS * E) as usize), &self.spec);
        let steps = experts_fwd_train_adapted(g, &ids, &self.set(), &self.adapters(), &self.shape, &self.x, &self.gate, &acts, &self.acc).unwrap();
        g.submit(&[&self.acc], &steps);
        (g.read(&self.acc, (ROWS * D) as usize), acts)
    }

    /// The backward of `loss = <wloss, out>`: the input gradient, and the four factor gradients added onto `prior`.
    fn backward(&self, wloss: &[f32], acts: &MoeActs, full_router: bool, prior: &[Vec<f32>; 4]) -> (Vec<f32>, [Vec<f32>; 4]) {
        let g = &self.g;
        let (rows, d, ff, e) = (ROWS as u64, D as u64, FF as u64, E as u64);
        let d_acc = g.storage_init("d_acc", wloss);
        let (d_gate, d_logits, d_x, router_grad) = (g.storage(rows * e), g.storage(rows * e), g.storage(rows * d), g.storage(e * d));
        let sb_bufs = [g.storage(rows * d), g.storage(rows * ff), g.storage(rows * 2 * ff), g.storage(1)];
        let sb = GluBwdScratch { d_expert_out: &sb_bufs[0], d_h: &sb_bufs[1], d_gate_up: &sb_bufs[2], d_up: &sb_bufs[3] };
        let router_weight_bwd = [
            g.step(idx(g, "matmul_dw"), &[&d_logits, &self.x, &router_grad], &[ROWS, D, E], E * D),
            g.step(idx(g, "matmul_dx"), &[&d_logits, &self.router_w, &d_x], &[ROWS, D, E, 0], ROWS * D),
        ];
        let rids = RouterBwdIds { router_bwd: idx(g, "router_bwd_sigmoid"), expert_counts: None };
        let router = if full_router {
            RouterGrad::Full { ids: &rids, kind: KIND, logits: &self.logits, fe: None, d_gate: &d_gate, d_router_logits: &d_logits, router_weight_bwd: &router_weight_bwd }
        } else {
            RouterGrad::Detached
        };
        let grads = [0, 1, 2, 3].map(|i| g.storage_init("grad", &prior[i]));
        let adapter_grads = ExpertAdapterGrads { gate_up: Some(ExpertLoraGrads { a: &grads[0], b: &grads[1] }), up: None, down: Some(ExpertLoraGrads { a: &grads[2], b: &grads[3] }) };
        let steps = moe_layer_bwd_adapted(g, router, &self.bwd_ids(), &self.set(), &[], &self.adapters(), &adapter_grads, &self.shape, &self.x, &self.gate, acts, &sb, &d_acc, &d_x).unwrap();
        g.submit(&[&d_gate, &d_x], &steps);
        let lens = [0, 1, 2, 3].map(|i| prior[i].len());
        (g.read(&d_x, (ROWS * D) as usize), [0, 1, 2, 3].map(|i| g.read(&grads[i], lens[i])))
    }
}

fn rms(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = v.clone().count().max(1) as f64;
    (v.map(|w| w * w).sum::<f64>() / n).sqrt()
}

fn assert_dx(label: &str, got: &[f32], want: &[(usize, Option<f64>)]) {
    let scale = rms(want.iter().filter_map(|w| w.1));
    let mut compared = 0;
    for &(i, w) in want {
        let Some(w) = w else { continue };
        compared += 1;
        let g = got[i];
        assert!((f64::from(g) - w).abs() <= 2e-3 * scale + 2e-3 * w.abs(), "{label}: d_x[{i}] = {g}, f64 central difference {w} (rms {scale})");
    }
    assert!(compared * 10 >= want.len() * 9, "{label}: too few entries away from a routing flip ({compared} of {})", want.len());
}

/// Every entry of a factor gradient: the routed experts' entries are `prior + f64 central difference`, the idle
/// expert's are `prior` bit for bit.
fn assert_factor(label: &str, got: &[f32], prior: &[f32], want: &[f64]) {
    let per_expert = want.len() / E as usize;
    let scale = rms(want.iter().copied());
    assert!(scale > 0.0, "{label}: the oracle gradient is zero everywhere");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if i / per_expert == IDLE {
            assert_eq!(g.to_bits(), prior[i].to_bits(), "{label}[{i}]: expert {IDLE} has no routed row, its gradient must stay as it was");
            assert_eq!(w, 0.0, "{label}[{i}]: the oracle moves an unrouted expert");
            continue;
        }
        let added = f64::from(g) - f64::from(prior[i]);
        assert!((added - w).abs() <= 2e-3 * scale + 2e-3 * w.abs(), "{label}[{i}] (expert {}): added {added}, f64 central difference {w} (rms {scale})", i / per_expert);
    }
}

/// A frozen MXFP8 (and, where the device has workgroup reductions, NVFP4) bank layer adapted on gate_up and down:
/// forward, every factor gradient and the input gradient against the f64 oracle.
#[test]
fn stacked_expert_adapters_train_through_a_frozen_bank() {
    let g = device();
    let mut rng = Lcg::random();
    for kind in [MicroScaled::Mxfp8E4m3, MicroScaled::Nvfp4] {
        if kind == MicroScaled::Nvfp4 && !g.caps().workgroup_reductions {
            brain_testutil::skip_unavailable("the NVFP4 bank forward needs workgroup reductions; the MXFP8 bank covers the adapter path here");
            continue;
        }
        let (dev, host) = DeviceLayer::new(&g, kind, &mut rng);
        let (out, acts) = dev.forward();
        let x64: Vec<f64> = host.x.iter().map(|&v| f64::from(v)).collect();
        let (want_loss, routing) = host_forward(&host, &x64, None);
        assert!(routing.iter().all(|r| r.iter().all(|&(j, _)| j != IDLE)), "expert {IDLE} must be unrouted");
        let got_loss: f64 = out.iter().zip(&host.wloss).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        assert!((got_loss - want_loss).abs() <= 1e-4 * want_loss.abs().max(1.0), "{kind:?}: loss {got_loss}, f64 oracle {want_loss}");

        let prior = [host.lora_gate_up.a.len(), host.lora_gate_up.b.len(), host.lora_down.a.len(), host.lora_down.b.len()].map(|n| rng.vec_scaled(n, 0.01));
        let (full_dx, factor_grads) = dev.backward(&host.wloss, &acts, true, &prior);
        for (i, (label, down, a)) in [("gate_up.lora_a", false, true), ("gate_up.lora_b", false, false), ("down.lora_a", true, true), ("down.lora_b", true, false)].into_iter().enumerate() {
            assert_factor(&format!("{kind:?} {label}"), &factor_grads[i], &prior[i], &host_factor_grad(&host, &routing, down, a));
        }

        let picks: Vec<usize> = (0..(ROWS * D) as usize).filter(|i| i % 13 == (i / D as usize) % 13).collect();
        assert_dx(&format!("{kind:?} full router"), &full_dx, &host_dx(&host, None, &picks));
        let (detached_dx, detached_grads) = dev.backward(&host.wloss, &acts, false, &prior);
        assert_dx(&format!("{kind:?} detached router"), &detached_dx, &host_dx(&host, Some(&routing), &picks));
        assert!(detached_grads.iter().zip(&factor_grads).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())), "{kind:?}: the router's treatment does not move the adapter gradients");
    }
}

/// Adapters the layer cannot take: an up adapter over a fused gate/up bank, a rank of 0, gradient buffers for a
/// projection with no adapter, and adapters beside a dense (trainable) bank.
#[test]
fn adapters_the_layer_cannot_take_are_refused_by_name() {
    let g = device();
    let mut rng = Lcg::random();
    let (dev, host) = DeviceLayer::new(&g, MicroScaled::Mxfp8E4m3, &mut rng);
    let (_, acts) = dev.forward();
    let ids = dev.fwd_ids();
    let f = &dev.factors;
    let lora = ExpertLora { a: &f[0], b: &f[1], rank: RANK, scale: LORA_SCALE };
    let fwd = |set: &ExpertSet, adapters: &ExpertAdapters| experts_fwd_train_adapted(&g, &ids, set, adapters, &dev.shape, &dev.x, &dev.gate, &acts, &dev.acc).err();

    let err = fwd(&dev.set(), &ExpertAdapters { up: Some(lora), ..ExpertAdapters::default() }).expect("an up adapter over a fused bank is refused");
    assert!(err.contains("up adapter"), "{err}");
    let err = fwd(&dev.set(), &ExpertAdapters { gate_up: Some(ExpertLora { rank: 0, ..lora }), ..ExpertAdapters::default() }).expect("rank 0 is refused");
    assert!(err.contains("gate_up") && err.contains("rank 0"), "{err}");

    let dense_w = [GluWeights { gate_w: &f[0], up_w: None, down_w: &f[0], gate_off: 0, down_off: 0 }; E as usize];
    let dense = ExpertSet { glu: dev.spec, bank: ExpertBank::Dense(&dense_w) };
    let err = fwd(&dense, &dev.adapters()).expect("adapters beside a dense bank are refused");
    assert!(err.contains("dense bank"), "{err}");

    let (rows, d) = (ROWS as u64, D as u64);
    let bufs = [g.storage(rows * d), g.storage(rows * d), g.storage(rows * 2 * d), g.storage(1)];
    let sb = GluBwdScratch { d_expert_out: &bufs[0], d_h: &bufs[1], d_gate_up: &bufs[2], d_up: &bufs[3] };
    let d_acc = g.storage_init("d_acc", &host.wloss);
    let only_gate_up = ExpertAdapters { gate_up: Some(lora), ..ExpertAdapters::default() };
    let stray = ExpertAdapterGrads { down: Some(ExpertLoraGrads { a: &bufs[0], b: &bufs[1] }), ..ExpertAdapterGrads::default() };
    let err = moe_layer_bwd_adapted(&g, RouterGrad::Detached, &dev.bwd_ids(), &dev.set(), &[], &only_gate_up, &stray, &dev.shape, &dev.x, &dev.gate, &acts, &sb, &d_acc, &bufs[0]).err().expect("gradients for an absent adapter are refused");
    assert!(err.contains("down") && err.contains("no adapter"), "{err}");
}
