// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of quantized mixture-of-experts models on their published weights for its
// clients. If your team needs expertise in training through frozen NVFP4 and MXFP8 expert banks then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a mixture-of-experts layer whose routed experts are a frozen microscaled base (NVFP4 or MXFP8
//! [`MicroBank`]s, the published bytes) trains through the shared row-compacted path under a declared expert
//! activation and gate/up layout:
//! - the banks' transposed products (`gather_dx`, `grouped_dx`) are the f64 transposed product of the decoded weights,
//!   and the grouped form is the gather form's arithmetic to the bit;
//! - a softcap-sigmoid expert over a row-interleaved fused gate/up bank, behind a sigmoid router: the layer output is the
//!   f64 forward over the dequantized weights, and the input gradient is the f64 central difference of that forward,
//!   with the router differentiated ([`RouterGrad::Full`]) or held constant ([`RouterGrad::Detached`]);
//! - shapes that do not fit, and weight gradients asked of a frozen bank, are refused by name.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::glu::{GatedAct, GluBwdScratch, GluLayout, GluSpec};
use model::moe::{experts_fwd_train_set, moe_layer_bwd_set, router_fwd_kind, ExpertBank, ExpertSet, MoeActs, MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds, RouterGrad, RouterKind, COMPACT_TRAIN_KERNELS};
use model::ops::{micro_bank_kernels, MicroBank, MicroScaled, MICRO_GROUPED_SLOTS};

use crate::tier_kit::{build_bank, upload_u32};

const BASE: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
    ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),
    ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("scale_add", kernels::SCALE_ADD),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
];

fn device() -> Gpu {
    let mut pipes: Vec<(&'static str, &'static str)> = BASE.iter().chain(COMPACT_TRAIN_KERNELS).copied().collect();
    pipes.extend(micro_bank_kernels());
    gpu_core::testgpu::dev(Box::leak(pipes.into_boxed_slice()))
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn kinds() -> [MicroScaled; 2] {
    [MicroScaled::Nvfp4, MicroScaled::Mxfp8E4m3]
}

// ---- the transposed bank products --------------------------------------------------------------------------------

/// `[2 * (ne + 1)]` slot then tile offsets per expert and the slots in expert-major order, as the routing kernels write
/// them for a grouped product.
fn route(ids: &[u32], ne: u32) -> (Vec<u32>, Vec<u32>, u32) {
    let counts: Vec<u32> = (0..ne).map(|e| ids.iter().filter(|&&i| i == e).count() as u32).collect();
    let (mut slots, mut tiles) = (vec![0u32], vec![0u32]);
    for c in &counts {
        slots.push(slots.last().unwrap() + c);
        tiles.push(tiles.last().unwrap() + c.div_ceil(MICRO_GROUPED_SLOTS));
    }
    let total_tiles = *tiles.last().unwrap();
    let perm = (0..ne).flat_map(|e| (0..ids.len() as u32).filter(move |&s| ids[s as usize] == e)).collect();
    (slots.into_iter().chain(tiles).collect(), perm, total_tiles)
}

#[test]
fn bank_input_gradients_are_the_transposed_product_of_the_decoded_weights() {
    let g = device();
    let mut rng = Lcg::random();
    let (blocks, n, k) = (3usize, 40usize, 128usize);
    let ids: Vec<u32> = vec![2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1];
    let slots = ids.len();
    for kind in kinds() {
        let (bank, w) = build_bank(&g, kind, blocks, n, k, &mut rng);
        let dy = rng.vec_scaled(slots * n, 1.0);
        let prior = rng.vec_scaled(slots * k, 1.0);
        let (dyb, idb) = (g.storage_init("dy", &dy), upload_u32(&g, &ids));
        let (tab, perm, tiles) = route(&ids, blocks as u32);
        let (tabb, permb) = (upload_u32(&g, &tab), upload_u32(&g, &perm));
        let [gather, grouped, added] = [(); 3].map(|_| g.storage_init("dx", &prior));
        let steps = [
            bank.gather_dx(&g, &dyb, &idb, &gather, slots as u32, false).unwrap(),
            bank.grouped_dx(&g, &dyb, &tabb, &permb, &grouped, tiles + 2, blocks as u32, false).unwrap(),
            bank.gather_dx(&g, &dyb, &idb, &added, slots as u32, true).unwrap(),
        ];
        g.submit(&[], &steps);
        let [gather, grouped, added] = [gather, grouped, added].map(|b| g.read(&b, slots * k));
        for (s, &e) in ids.iter().enumerate() {
            for i in 0..k {
                let terms: Vec<f64> = (0..n).map(|c| f64::from(w[(e as usize * n + c) * k + i]) * f64::from(dy[s * n + c])).collect();
                let (want, scale) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
                let got = f64::from(gather[s * k + i]);
                assert!((got - want).abs() <= 2e-5 * scale + 1e-9, "{kind:?} gather_dx slot {s} col {i}: {got}, oracle {want}");
                let sum = f64::from(added[s * k + i]);
                let want_sum = want + f64::from(prior[s * k + i]);
                assert!((sum - want_sum).abs() <= 2e-5 * (scale + want_sum.abs()) + 1e-9, "{kind:?} accumulated slot {s} col {i}: {sum}, want {want_sum}");
            }
        }
        assert!(gather.iter().zip(&grouped).all(|(a, b)| a.to_bits() == b.to_bits()), "{kind:?}: the grouped input gradient is the gather form's arithmetic");
    }
}

// ---- a layer over a frozen bank: f64 oracle -----------------------------------------------------------------------

const ROWS: u32 = 7;
const D: u32 = 128;
const FF: u32 = 128;
const E: u32 = 4;
const TOP_K: u32 = 2;
const SCALE: f32 = 2.5;
const KIND: RouterKind = RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: SCALE };
/// Caps small enough that the softcaps work in their nonlinear range at these magnitudes.
const ACT: GatedAct = GatedAct::SoftcapSigmoid { gate_cap: 0.15, up_cap: 0.2 };

/// The layer's inputs and the dequantized weights of its frozen banks.
struct Layer {
    x: Vec<f32>,
    router_w: Vec<f32>,
    bias: Vec<f32>,
    wloss: Vec<f32>,
    /// `[E * 2 * FF, D]`, rows interleaved `g0, u0, g1, u1, ...`.
    gate_up: Vec<f32>,
    /// `[E * D, FF]`.
    down: Vec<f32>,
}

fn softcap(v: f64, c: f64) -> f64 {
    c * (v / c).tanh()
}

/// The f64 forward: routing (sigmoid scores, top-k on score + bias, unbiased scores renormalised then scaled), and the
/// selected experts' softcap-sigmoid FFN over the interleaved bank. `fixed` replaces the routing with a given one (the
/// router held constant). Returns `(loss, routing)`.
fn host_forward(l: &Layer, x: &[f64], fixed: Option<&[Vec<(usize, f64)>]>) -> (f64, Vec<Vec<(usize, f64)>>) {
    let (d, ff, e) = (D as usize, FF as usize, E as usize);
    let GatedAct::SoftcapSigmoid { gate_cap, up_cap } = ACT else { unreachable!() };
    let mut routing = Vec::new();
    let mut loss = 0.0;
    for r in 0..ROWS as usize {
        let xr = &x[r * d..(r + 1) * d];
        let route = match fixed {
            Some(f) => f[r].clone(),
            None => {
                let score: Vec<f64> = (0..e).map(|j| 1.0 / (1.0 + (-(0..d).map(|i| f64::from(l.router_w[j * d + i]) * xr[i]).sum::<f64>()).exp())).collect();
                let mut order: Vec<usize> = (0..e).collect();
                order.sort_by(|&a, &b| (score[b] + f64::from(l.bias[b])).total_cmp(&(score[a] + f64::from(l.bias[a]))));
                let chosen = &order[..TOP_K as usize];
                let sum: f64 = chosen.iter().map(|&j| score[j]).sum::<f64>() + 1e-20;
                chosen.iter().map(|&j| (j, score[j] / sum * f64::from(SCALE))).collect()
            }
        };
        for &(j, w) in &route {
            let proj = |row: usize| (0..d).map(|i| f64::from(l.gate_up[(j * 2 * ff + row) * d + i]) * xr[i]).sum::<f64>();
            let h: Vec<f64> = (0..ff)
                .map(|c| {
                    let (gv, uv) = (proj(2 * c), proj(2 * c + 1));
                    softcap(gv, f64::from(gate_cap)) / (1.0 + (-gv).exp()) * softcap(uv, f64::from(up_cap))
                })
                .collect();
            for o in 0..d {
                let y: f64 = (0..ff).map(|c| f64::from(l.down[(j * d + o) * ff + c]) * h[c]).sum();
                loss += w * y * f64::from(l.wloss[r * d + o]);
            }
        }
        routing.push(route);
    }
    (loss, routing)
}

fn selected(routing: &[Vec<(usize, f64)>]) -> Vec<Vec<usize>> {
    routing.iter().map(|r| r.iter().map(|&(j, _)| j).collect()).collect()
}

/// `d loss / d x[i]` for the entries `picks` by f64 central differences; `None` where a perturbation changes which
/// experts a row selects (the derivative does not exist there). `fixed` holds the routing constant.
fn host_dx(l: &Layer, fixed: Option<&[Vec<(usize, f64)>]>, picks: &[usize]) -> Vec<(usize, Option<f64>)> {
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

/// The device side of one layer: router, gate, compacted activations, the bank products and the backward scratch.
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
}

impl DeviceLayer {
    fn new(g: &Gpu, kind: MicroScaled, rng: &mut Lcg) -> (DeviceLayer, Layer) {
        let (gate_up, gate_up_w) = build_bank(g, kind, E as usize, 2 * FF as usize, D as usize, rng);
        let (down, down_w) = build_bank(g, kind, E as usize, D as usize, FF as usize, rng);
        let host = Layer {
            x: rng.vec_scaled((ROWS * D) as usize, 1.0),
            router_w: rng.vec_scaled((E * D) as usize, 0.15),
            bias: rng.vec_scaled(E as usize, 0.05),
            wloss: rng.vec_scaled((ROWS * D) as usize, 1.0),
            gate_up: gate_up_w,
            down: down_w,
        };
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
            g: g.share(),
        };
        (dev, host)
    }

    fn set(&self) -> ExpertSet<'_> {
        ExpertSet { glu: self.spec, bank: ExpertBank::Micro { gate_up: &self.gate_up, up: None, down: &self.down } }
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

    /// The forward: router, then the routed experts over the bank. Returns the layer output and the activations.
    fn forward(&self) -> (Vec<f32>, MoeActs) {
        let g = &self.g;
        let ids = self.fwd_ids();
        let steps = vec![
            g.step(idx(g, "matmul"), &[&self.x, &self.router_w, &self.logits], &[ROWS, D, E], ROWS * E),
            router_fwd_kind(g, &ids, KIND, &self.shape, &self.logits, Some(&self.bias), &self.gate, Some(&self.probs)),
        ];
        g.submit(&[], &steps);
        let acts = MoeActs::compact_for(g, &self.shape, &g.read(&self.gate, (ROWS * E) as usize), &self.spec);
        let steps = experts_fwd_train_set(g, &ids, &self.set(), &self.shape, &self.x, &self.gate, &acts, &self.acc).unwrap();
        g.submit(&[&self.acc], &steps);
        (g.read(&self.acc, (ROWS * D) as usize), acts)
    }

    /// The input gradient of `loss = <wloss, out>`, with the router differentiated or held constant.
    fn input_grad(&self, wloss: &[f32], acts: &MoeActs, full_router: bool) -> Vec<f32> {
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
        let steps = moe_layer_bwd_set(g, router, &self.bwd_ids(), &self.set(), &[], &self.shape, &self.x, &self.gate, acts, &sb, &d_acc, &d_x).unwrap();
        // `d_gate` is added into and, held constant, `d_x` too: both start at zero.
        g.submit(&[&d_gate, &d_x], &steps);
        g.read(&d_x, (ROWS * D) as usize)
    }
}

fn assert_grad(label: &str, got: &[f32], want: &[(usize, Option<f64>)]) {
    let scale = want.iter().filter_map(|w| w.1).map(|v| v * v).sum::<f64>().sqrt() / (want.len() as f64).sqrt();
    let mut compared = 0;
    for &(i, w) in want {
        let Some(w) = w else { continue };
        compared += 1;
        let g = got[i];
        assert!((f64::from(g) - w).abs() <= 2e-3 * scale + 2e-3 * w.abs(), "{label}: d_x[{i}] = {g}, f64 central difference {w} (rms {scale})");
    }
    assert!(compared * 10 >= want.len() * 9, "{label}: too few entries away from a routing flip ({compared} of {})", want.len());
}

/// The layer over each frozen bank kind: forward equals the f64 forward over the dequantized weights, the input
/// gradient equals the f64 central difference with the router differentiated and with it held constant.
#[test]
fn a_softcap_sigmoid_interleaved_bank_trains_through_the_compacted_path() {
    let g = device();
    let mut rng = Lcg::random();
    for kind in kinds() {
        let (dev, host) = DeviceLayer::new(&g, kind, &mut rng);
        if kind == MicroScaled::Nvfp4 && !g.caps().workgroup_reductions {
            // The NVFP4 bank's forward product cooperates across a workgroup; its input gradient is checked above.
            brain_testutil::skip_unavailable("the NVFP4 bank forward needs workgroup reductions; its input gradient kernels ran above");
            continue;
        }
        let (out, acts) = dev.forward();
        let x64: Vec<f64> = host.x.iter().map(|&v| f64::from(v)).collect();
        let (want_loss, routing) = host_forward(&host, &x64, None);
        let got_loss: f64 = out.iter().zip(&host.wloss).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        assert!((got_loss - want_loss).abs() <= 1e-4 * want_loss.abs().max(1.0), "{kind:?}: loss {got_loss}, f64 oracle {want_loss}");

        // Every row, a spread of columns: the central differences run the whole f64 forward twice per entry.
        let picks: Vec<usize> = (0..(ROWS * D) as usize).filter(|i| i % 13 == (i / D as usize) % 13).collect();
        let full = dev.input_grad(&host.wloss, &acts, true);
        assert_grad(&format!("{kind:?} full router"), &full, &host_dx(&host, None, &picks));
        let detached = dev.input_grad(&host.wloss, &acts, false);
        assert_grad(&format!("{kind:?} detached router"), &detached, &host_dx(&host, Some(&routing), &picks));
        assert!(full.iter().zip(&detached).any(|(a, b)| (a - b).abs() > 1e-4), "{kind:?}: the router's share of the gradient is not zero");
    }
}

#[test]
fn shapes_that_do_not_fit_and_gradients_of_a_frozen_bank_are_refused() {
    let g = device();
    let mut rng = Lcg::random();
    let (dev, host) = DeviceLayer::new(&g, MicroScaled::Mxfp8E4m3, &mut rng);
    let (_, acts) = dev.forward();
    let ids = dev.fwd_ids();
    // A separate-layout spec over a fused bank: the up bank is missing.
    let separate = ExpertSet { glu: GluSpec { act: ACT, layout: GluLayout::Separate }, ..dev.set() };
    let err = experts_fwd_train_set(&g, &ids, &separate, &dev.shape, &dev.x, &dev.gate, &acts, &dev.acc).err().expect("a missing up bank is refused");
    assert!(err.contains("up bank"), "{err}");
    // The bank's rows do not match the layer's expert width.
    let narrow = MoeShape { moe_ff: FF / 2, ..dev.shape };
    let err = experts_fwd_train_set(&g, &ids, &dev.set(), &narrow, &dev.x, &dev.gate, &acts, &dev.acc).err().expect("a mismatched bank is refused");
    assert!(err.contains("gate_up bank"), "{err}");
    // A frozen bank has no weight gradients to write.
    let (rows, d) = (ROWS as u64, D as u64);
    let bufs = [g.storage(rows * d), g.storage(rows * d), g.storage(rows * 2 * d), g.storage(1)];
    let sb = GluBwdScratch { d_expert_out: &bufs[0], d_h: &bufs[1], d_gate_up: &bufs[2], d_up: &bufs[3] };
    let grads = [model::glu::GluGrads { gate_w: Some(&bufs[0]), up_w: None, down_w: None }; E as usize];
    let d_acc = g.storage_init("d_acc", &host.wloss);
    let err = moe_layer_bwd_set(&g, RouterGrad::Detached, &dev.bwd_ids(), &dev.set(), &grads, &dev.shape, &dev.x, &dev.gate, &acts, &sb, &d_acc, &bufs[0]).err().expect("weight gradients of a frozen bank are refused");
    assert!(err.contains("frozen base"), "{err}");
}
