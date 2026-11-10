// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-checked GPU training for compact perception models for its clients.
// If your team needs expertise in porting a host trainer onto accelerators then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The device training tape against the host tape. One graph, written once against [`Graph`], is built on the host
//! `Tape<f32>` (the oracle, itself finite-difference checked in `train_ops`) and on the `DevTape`; forward values
//! and the gradient of every leaf must agree. A random weighting of each output makes every element matter. The
//! device is whatever the process selected (`BRAIN_DEVICE=cpu` runs the CPU backend's WGSL JIT).

use std::collections::HashMap;

use data::rng::Lcg;
use mirage::device::train::DevTape;
use mirage::device::Device;
use mirage::train::graph::Graph;
use mirage::train::ops::Seg;
use mirage::train::tape::{Id, Tape};
use paramstore::{ParamStore, Role};

/// Minimum cosine between a device result and the host result.
const MIN_COSINE: f64 = 0.99999;
/// Maximum |difference| relative to the largest host magnitude.
const MAX_REL: f64 = 1e-4;

fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Lcg::new(seed); // seed: filler compared between two paths
    (0..n).map(|_| rng.signed() * scale).collect()
}

fn device(test: &str) -> Option<Device> {
    crate::common::cap_threads();
    match Device::open() {
        Ok(d) => Some(d),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
            None
        }
    }
}

/// A store holding `params` (name, values), all trainable except the names `frozen_roles` lists.
fn store(dev: &Device, params: &[(&str, Vec<f32>)], frozen_roles: &[&str]) -> ParamStore {
    let init: HashMap<String, Vec<f32>> = params.iter().map(|(n, v)| (n.to_string(), v.clone())).collect();
    let roles = params.iter().map(|(n, v)| (n.to_string(), v.len(), if frozen_roles.contains(n) { Role::Frozen } else { Role::Trainable })).collect();
    ParamStore::new_with_roles(dev.gpu(), roles, &init)
}

/// What a graph builder returns: the leaves whose gradients are compared, the outputs, and the scalar loss.
struct Run {
    leaves: Vec<Id>,
    outs: Vec<Id>,
    loss: Id,
}

/// The device parameter `name`, or on the host a leaf holding `values`.
fn param(g: &mut dyn Graph<f32>, name: &str, values: &[f32]) -> Id {
    match g.resident_param(name) {
        Some(id) => id,
        None => g.leaf(values.to_vec()),
    }
}

/// `sum_i w_i * out_i` over a random weighting, so every output element enters the loss with its own weight.
fn weighted_loss(g: &mut dyn Graph<f32>, outs: &[Id], seed: u64) -> Id {
    let mut total: Option<Id> = None;
    for (i, &o) in outs.iter().enumerate() {
        let n = g.len(o);
        let w = g.leaf(noise(n, seed + i as u64, 1.0));
        let m = g.mul(o, w);
        let s = g.sum_weighted(m, 1.0);
        total = Some(match total {
            None => s,
            Some(t) => g.add(t, s),
        });
    }
    total.expect("a graph with outputs")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let (na, nb): (f64, f64) = (a.iter().map(|x| f64::from(*x).powi(2)).sum(), b.iter().map(|x| f64::from(*x).powi(2)).sum());
    if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn assert_close_cos(what: &str, got: &[f32], want: &[f32], max_rel: f64, min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-6);
    let max_err = got.iter().zip(want).fold(0f64, |m, (g, w)| m.max(f64::from((g - w).abs()))) / scale;
    let cos = cosine(got, want);
    eprintln!("{what}: cosine {cos:.8}, max relative error {max_err:.2e}");
    assert!(cos >= min_cos, "{what}: cosine {cos}");
    assert!(max_err <= max_rel, "{what}: max relative error {max_err:.3e}");
}

/// Build the graph on the host and on the device and compare every output and every leaf gradient.
/// The CPU backend runs the same f32 arithmetic as the host oracle; a GPU's selector may pick tensor-core GEMMs
/// (rounding at about 1e-3 of the largest value), so the bounds widen there (ledger, lane L2c).
fn bounds(dev: &Device, max_rel: f64) -> (f64, f64) {
    if dev.gpu().kind() == "cpu" {
        (max_rel, MIN_COSINE)
    } else {
        (max_rel.max(2e-3), 0.9999)
    }
}

fn compare(dev: &Device, name: &str, params: &ParamStore, build: impl Fn(&mut dyn Graph<f32>) -> Run) {
    compare_within(dev, name, params, MAX_REL, build);
}

/// [`compare`] with a chosen relative error bound.
fn compare_within(dev: &Device, name: &str, params: &ParamStore, max_rel: f64, build: impl Fn(&mut dyn Graph<f32>) -> Run) {
    let (max_rel, min_cos) = bounds(dev, max_rel);
    let mut host = Tape::<f32>::new();
    let hr = build(&mut host);
    let hgrads = host.backward(hr.loss);

    let mut tape = DevTape::new(dev, params).with_leaf_grads();
    let dr = build(&mut tape);
    for (i, (&h, &d)) in hr.outs.iter().zip(&dr.outs).enumerate() {
        assert_close_cos(&format!("{name} output {i}"), &tape.value(d), host.val(h), max_rel, min_cos);
    }
    tape.backward(dr.loss);
    assert_eq!(hr.leaves.len(), dr.leaves.len());
    for (i, (&h, &d)) in hr.leaves.iter().zip(&dr.leaves).enumerate() {
        let want = hgrads.get(h).map(<[f32]>::to_vec).unwrap_or_else(|| vec![0.0; host.len(h)]);
        let got = tape.grad(d).unwrap_or_else(|| vec![0.0; want.len()]);
        assert_close_cos(&format!("{name} gradient of leaf {i}"), &got, &want, max_rel, min_cos);
    }
}

fn no_params(dev: &Device) -> ParamStore {
    store(dev, &[], &[])
}

#[test]
fn sums_products_and_scales_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("elementwise") else { return };
    let params = no_params(&dev);
    let n = 131; // not a multiple of the workgroup size
    compare(&dev, "binary", &params, |g| {
        let (a, b) = (g.leaf(noise(n, 1, 1.0)), g.leaf(noise(n, 2, 1.0)));
        let (sum, prod, scaled) = (g.add(a, b), g.mul(a, b), g.scale(a, -0.37));
        // The same value on both sides of a product: the gradient reaches it twice.
        let square = g.mul(a, a);
        let outs = vec![sum, prod, scaled, square];
        let loss = weighted_loss(g, &outs, 10);
        Run { leaves: vec![a, b], outs, loss }
    });
}

#[test]
fn channel_scales_and_biases_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("channel ops") else { return };
    let params = no_params(&dev);
    let (rows, c) = (37, 11);
    compare(&dev, "channel", &params, |g| {
        let x = g.leaf(noise(rows * c, 3, 1.0));
        let (bias, chan, one) = (g.leaf(noise(c, 4, 1.0)), g.leaf(noise(c, 5, 1.0)), g.leaf(noise(1, 6, 1.0)));
        let (b, m, s) = (g.bias_add(x, bias), g.chan_mul(x, chan), g.scalar_mul(x, one));
        let outs = vec![b, m, s];
        let loss = weighted_loss(g, &outs, 20);
        Run { leaves: vec![x, bias, chan, one], outs, loss }
    });
}

#[test]
fn activations_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("activations") else { return };
    let params = no_params(&dev);
    let n = 70;
    compare(&dev, "activations", &params, |g| {
        let x = g.leaf(noise(n, 7, 3.0));
        let (sg, si, ge) = (g.sigmoid(x), g.silu(x), g.gelu_tanh(x));
        let outs = vec![sg, si, ge];
        let loss = weighted_loss(g, &outs, 30);
        Run { leaves: vec![x], outs, loss }
    });
}

#[test]
fn grad_scale_is_the_identity_forward_and_scales_only_the_gradient() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("grad_scale") else { return };
    let params = no_params(&dev);
    compare(&dev, "grad_scale", &params, |g| {
        let x = g.leaf(noise(45, 8, 1.0));
        let y = g.grad_scale(x, 0.1);
        let z = g.silu(y);
        let outs = vec![y, z];
        let loss = weighted_loss(g, &outs, 40);
        Run { leaves: vec![x], outs, loss }
    });
}

#[test]
fn the_loss_reduction_and_scalar_read_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("sum") else { return };
    let params = no_params(&dev);
    let x = noise(1000, 9, 1.0);
    let mut tape = DevTape::new(&dev, &params).with_leaf_grads();
    let xi = tape.leaf(x.clone());
    let s = tape.sum_weighted(xi, 0.25);
    let want: f64 = x.iter().map(|v| f64::from(*v)).sum::<f64>() * 0.25;
    assert!((tape.scalar(s) - want).abs() < 1e-4, "{} vs {want}", tape.scalar(s));
}

fn linear_run(g: &mut dyn Graph<f32>, w: &[f32], wt: &[f32], dims: (usize, usize, usize)) -> Run {
    let (rows, inn, out) = dims;
    let x = g.leaf(noise(rows * inn, 11, 1.0));
    let (wid, wtid) = (param(g, "w", w), param(g, "wt", wt));
    let (y, yt) = (g.linear(x, wid, rows, inn, out), g.linear_nt(x, wtid, rows, inn, out));
    let outs = vec![y, yt];
    let loss = weighted_loss(g, &outs, 50);
    Run { leaves: vec![x, wid, wtid], outs, loss }
}

#[test]
fn linear_and_linear_nt_match_the_host_with_store_parameters() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("linear") else { return };
    for (rows, inn, out) in [(7, 13, 9), (130, 70, 140)] {
        let (w, wt) = (noise(inn * out, 12, 0.5), noise(out * inn, 13, 0.5));
        let params = store(&dev, &[("w", w.clone()), ("wt", wt.clone())], &[]);
        compare(&dev, &format!("linear {rows}x{inn}x{out}"), &params, |g| linear_run(g, &w, &wt, (rows, inn, out)));
    }
}

#[test]
fn frozen_parameters_get_no_gradient_while_the_input_path_still_does() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("frozen") else { return };
    let (rows, inn, out) = (6, 10, 7);
    let (w, wt) = (noise(inn * out, 14, 0.5), noise(out * inn, 15, 0.5));
    // `w` is frozen after construction (it keeps a gradient buffer), `wt` is a frozen role (it has none).
    let mut params = store(&dev, &[("w", w.clone()), ("wt", wt.clone())], &["wt"]);
    params.freeze_where(|n| n == "w");

    let mut host = Tape::<f32>::new();
    let hr = linear_run(&mut host, &w, &wt, (rows, inn, out));
    let hgrads = host.backward(hr.loss);

    let mut tape = DevTape::new(&dev, &params);
    let x = tape.leaf(noise(rows * inn, 11, 1.0));
    // A constant input and frozen weights: nothing needs a gradient, so nothing is recorded for backward.
    let (wid, wtid) = (tape.resident_param("w").unwrap(), tape.resident_param("wt").unwrap());
    let y = tape.linear(x, wid, rows, inn, out);
    let yt = tape.linear_nt(x, wtid, rows, inn, out);
    assert_eq!(tape.backward_nodes(), 0, "constants and frozen parameters record no backward");
    let _ = (y, yt);

    // The trainable input path still gets its gradient through the frozen weights; the frozen ones stay zero.
    let mut tape = DevTape::new(&dev, &params).with_leaf_grads();
    let dr = linear_run(&mut tape, &w, &wt, (rows, inn, out));
    tape.backward(dr.loss);
    let (rel, cos) = bounds(&dev, MAX_REL);
    assert_close_cos("frozen: input gradient", &tape.grad(dr.leaves[0]).unwrap(), hgrads.get(hr.leaves[0]).unwrap(), rel, cos);
    assert!(params.read_grad(dev.gpu(), "w").iter().all(|&v| v == 0.0), "a frozen parameter accumulated a gradient");
    assert!(tape.grad(dr.leaves[1]).is_none() && tape.grad(dr.leaves[2]).is_none());
}

#[test]
fn stacked_layers_resolve_to_one_entry_each() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("layers") else { return };
    let (rows, inn, out) = (5, 9, 4);
    let layers: Vec<Vec<f32>> = (0..3).map(|l| noise(inn * out, 60 + l, 0.5)).collect();
    let entries: Vec<(String, Vec<f32>)> = layers.iter().enumerate().map(|(l, v)| (format!("stack#{l}"), v.clone())).collect();
    let refs: Vec<(&str, Vec<f32>)> = entries.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
    let params = store(&dev, &refs, &[]);
    compare(&dev, "layers", &params, |g| {
        let x = g.leaf(noise(rows * inn, 16, 1.0));
        let ids: Vec<Id> = (0..3).map(|l| g.resident_layer("stack", l).unwrap_or_else(|| g.leaf(layers[l].clone()))).collect();
        let outs: Vec<Id> = ids.iter().map(|&w| g.linear(x, w, rows, inn, out)).collect();
        let loss = weighted_loss(g, &outs, 70);
        let mut leaves = vec![x];
        leaves.extend(ids);
        Run { leaves, outs, loss }
    });
}

#[test]
fn adapter_pairs_resolve_through_the_registry() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("lora") else { return };
    let (a, b) = (noise(8 * 3, 17, 0.5), noise(3 * 5, 18, 0.5));
    let params = store(&dev, &[("k.lora_a", a.clone()), ("k.lora_b", b.clone())], &[]);
    let mut tape = DevTape::new(&dev, &params).with_lora("k", "k.lora_a", "k.lora_b", 3, 2.0);
    let (ia, ib, rank, scale) = tape.resident_lora("k").expect("registered");
    assert_eq!((rank, scale), (3, 2.0));
    assert_eq!(tape.value(ia), a);
    assert_eq!(tape.value(ib), b);
    assert!(tape.resident_lora("other").is_none());
}

#[test]
fn zero_centred_and_unit_rms_norms_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("norms") else { return };
    let params = no_params(&dev);
    for (rows, d) in [(5, 13), (4, 70)] {
        compare(&dev, &format!("rms {rows}x{d}"), &params, |g| {
            let x = g.leaf(noise(rows * d, 21, 2.0));
            let scale = g.leaf(noise(d, 22, 0.5));
            let (zc, unit) = (g.zc_norm(x, d, scale), g.rms_unit(x, d));
            let outs = vec![zc, unit];
            let loss = weighted_loss(g, &outs, 80);
            Run { leaves: vec![x, scale], outs, loss }
        });
    }
}

#[test]
fn masked_rows_are_zero_and_pass_no_gradient() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("mask_rows") else { return };
    let params = no_params(&dev);
    let (rows, d) = (9, 7);
    let valid = [true, false, true, true, false, false, true, false, true];
    compare(&dev, "mask_rows", &params, |g| {
        let x = g.leaf(noise(rows * d, 23, 1.0));
        let m = g.mask_rows(x, d, &valid);
        let loss = weighted_loss(g, &[m], 90);
        Run { leaves: vec![x], outs: vec![m], loss }
    });
}

#[test]
fn plans_with_overlapping_segments_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("plan") else { return };
    let params = no_params(&dev);
    let n = 53;
    compare(&dev, "plan", &params, |g| {
        let x = g.leaf(noise(n, 24, 1.0));
        let segs = vec![
            Seg::copy(0, 10, 20),
            // Overlaps the first segment in the destination and starts mid-way through it.
            Seg { dst: 15, src: 3, len: 17, w: 0.5f32 },
            // Repeats a source range already used (the transpose overlaps in the source).
            Seg { dst: 40, src: 10, len: 20, w: -1.5 },
            // Covers the same destination range as another one exactly.
            Seg { dst: 40, src: 33, len: 20, w: 2.0 },
            Seg::copy(70, 0, 0), // empty
            Seg { dst: 85, src: 50, len: 3, w: 1.0 },
        ];
        let y = g.plan(x, 90, segs);
        let loss = weighted_loss(g, &[y], 100);
        Run { leaves: vec![x], outs: vec![y], loss }
    });
}

#[test]
fn gathers_concatenations_and_slices_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("movement") else { return };
    let params = no_params(&dev);
    let d = 5;
    compare(&dev, "movement", &params, |g| {
        let x = g.leaf(noise(9 * d, 25, 1.0));
        let y = g.leaf(noise(4 * d, 26, 1.0));
        // Repeated and out-of-order rows (an embedding lookup of a sentence with a repeated token).
        let rows = g.gather_rows(x, d, &[3, 0, 3, 8, 3]);
        let cat = g.concat(x, y);
        let piece = g.slice(cat, 7, 29);
        let outs = vec![rows, cat, piece];
        let loss = weighted_loss(g, &outs, 110);
        Run { leaves: vec![x, y], outs, loss }
    });
}

#[test]
fn softmax_and_row_products_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("rows") else { return };
    let params = no_params(&dev);
    for (rows, d) in [(6, 3), (5, 70), (3, 200)] {
        compare(&dev, &format!("rows {rows}x{d}"), &params, |g| {
            let x = g.leaf(noise(rows * d, 27, 3.0));
            let y = g.leaf(noise(rows * d, 28, 1.0));
            let s = g.leaf(noise(rows, 29, 1.0));
            let (sm, dot, scaled) = (g.softmax_rows(x, d), g.row_dot(x, y, d), g.row_mul(s, y, d));
            let outs = vec![sm, dot, scaled];
            let loss = weighted_loss(g, &outs, 120);
            Run { leaves: vec![x, y, s], outs, loss }
        });
    }
}

use mirage::vision::ops::Map;

#[test]
fn depthwise_convolutions_match_the_host_at_every_stride() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("depthwise") else { return };
    let params = no_params(&dev);
    let m = Map { h: 9, w: 11, c: 6 };
    // (kh, kw, sy, sx): the stem's 3x3 stride 2x2, anisotropic strides, and 1xk / kx1 kernels.
    for (kh, kw, sy, sx) in [(3, 3, 1, 1), (3, 3, 2, 2), (3, 3, 2, 1), (1, 7, 1, 1), (1, 5, 1, 2), (3, 1, 2, 1)] {
        compare(&dev, &format!("depthwise {kh}x{kw} s{sy}x{sx}"), &params, |g| {
            let x = g.leaf(noise(m.h * m.w * m.c, 31, 1.0));
            let k = g.leaf(noise(kh * kw * m.c, 32, 0.5));
            let (y, o) = g.depthwise(x, m, k, kh, kw, sy, sx);
            assert_eq!(o.c, m.c);
            let loss = weighted_loss(g, &[y], 130);
            Run { leaves: vec![x, k], outs: vec![y], loss }
        });
    }
}

#[test]
fn dense_convolutions_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("conv_dense") else { return };
    let params = no_params(&dev);
    let m = Map { h: 9, w: 10, c: 5 };
    let cout = 7; // neither channel count is a multiple of the workgroup size
    for (kh, kw, sy, sx) in [(3, 3, 1, 1), (3, 3, 2, 2), (3, 3, 2, 1), (1, 1, 1, 1)] {
        compare(&dev, &format!("conv {kh}x{kw} s{sy}x{sx}"), &params, |g| {
            let x = g.leaf(noise(m.h * m.w * m.c, 33, 1.0));
            let k = g.leaf(noise(kh * kw * m.c * cout, 34, 0.5));
            let (y, _) = g.conv_dense(x, m, k, kh, kw, cout, sy, sx);
            let loss = weighted_loss(g, &[y], 140);
            Run { leaves: vec![x, k], outs: vec![y], loss }
        });
    }
}

#[test]
fn upsampling_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("upsample2") else { return };
    let params = no_params(&dev);
    let m = Map { h: 5, w: 7, c: 3 };
    compare(&dev, "upsample2", &params, |g| {
        let x = g.leaf(noise(m.h * m.w * m.c, 35, 1.0));
        let (y, o) = g.upsample2(x, m);
        assert_eq!((o.h, o.w), (10, 14));
        let loss = weighted_loss(g, &[y], 150);
        Run { leaves: vec![x], outs: vec![y], loss }
    });
}

#[test]
fn the_walsh_transform_is_the_host_sequency_transform_and_self_adjoint() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("walsh") else { return };
    let params = no_params(&dev);
    for (rows, n) in [(5, 8), (7, 64), (3, 512)] {
        compare(&dev, &format!("walsh {rows}x{n}"), &params, |g| {
            let x = g.leaf(noise(rows * n, 36, 1.0));
            let y = g.walsh(x, n);
            let z = g.walsh(y, n); // self-inverse: z == x
            let outs = vec![y, z];
            let loss = weighted_loss(g, &outs, 160);
            Run { leaves: vec![x], outs, loss }
        });
    }
}

#[test]
fn banded_channel_convolutions_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("band") else { return };
    let params = no_params(&dev);
    // Edge channels (k > n and k < n), and the real width with the real 65-tap kernel.
    for (rows, n, k) in [(5, 13, 5), (6, 64, 65), (40, 512, 65)] {
        compare(&dev, &format!("band {rows}x{n} k{k}"), &params, |g| {
            let u = g.leaf(noise(rows * n, 37, 1.0));
            let w = g.leaf(noise(n * k, 38, 0.3));
            let y = g.band(u, n, w, k);
            let loss = weighted_loss(g, &[y], 170);
            Run { leaves: vec![u, w], outs: vec![y], loss }
        });
    }
}

#[test]
fn global_response_norm_scale_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("grn") else { return };
    let params = no_params(&dev);
    let (rows, d) = (11, 70);
    let valid: Vec<bool> = (0..rows).map(|r| r % 4 != 3).collect();
    compare(&dev, "grn", &params, |g| {
        let y = g.leaf(noise(rows * d, 39, 1.0));
        let gamma = g.leaf(noise(d, 40, 0.5));
        let s = g.grn_scale(y, d, &valid, gamma);
        let loss = weighted_loss(g, &[s], 180);
        Run { leaves: vec![y, gamma], outs: vec![s], loss }
    });
}

/// The HConvGLU block as `Net::hconv` builds it: ZC norm, Walsh, gate and value bands, gelu gate, out band, Walsh.
fn hconv_block(g: &mut dyn Graph<f32>, rows: usize, n: usize, k: usize) -> Run {
    let x = g.leaf(noise(rows * n, 41, 1.0));
    let norm = g.leaf(noise(n, 42, 0.3));
    let (gate, value, out) = (g.leaf(noise(n * k, 43, 0.2)), g.leaf(noise(n * k, 44, 0.2)), g.leaf(noise(n * k, 45, 0.2)));
    let gate_bias = g.leaf(noise(n, 46, 0.3));
    let u = g.zc_norm(x, n, norm);
    let u = g.walsh(u, n);
    let a = g.band(u, n, gate, k);
    let v = g.band(u, n, value, k);
    let a = g.bias_add(a, gate_bias);
    let a = g.gelu_tanh(a);
    let a = g.mul(a, v);
    let y = g.band(a, n, out, k);
    let y = g.walsh(y, n);
    let loss = weighted_loss(g, &[y], 190);
    Run { leaves: vec![x, norm, gate, value, out, gate_bias], outs: vec![y], loss }
}

#[test]
fn an_hconv_block_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("hconv") else { return };
    let params = no_params(&dev);
    for (rows, n, k) in [(9, 64, 65), (24, 512, 65)] {
        compare(&dev, &format!("hconv {rows}x{n}"), &params, |g| hconv_block(g, rows, n, k));
    }
}

use mirage::train::seq::AttnShape;

#[test]
fn rotary_embeddings_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("rope") else { return };
    let params = no_params(&dev);
    let (rows, heads, hd) = (9, 3, 16);
    let angles: Vec<f64> = noise(rows * hd / 2, 51, 3.0).into_iter().map(f64::from).collect();
    compare(&dev, "rope", &params, |g| {
        let x = g.leaf(noise(rows * heads * hd, 52, 1.0));
        let y = g.rope(x, heads, hd, &angles);
        let loss = weighted_loss(g, &[y], 200);
        Run { leaves: vec![x], outs: vec![y], loss }
    });
}

fn attention_run(g: &mut dyn Graph<f32>, s: AttnShape) -> Run {
    let q = g.leaf(noise(s.tq * s.hq * s.qk, 53, 1.0));
    let k = g.leaf(noise(s.tk * s.hk * s.qk, 54, 1.0));
    let v = g.leaf(noise(s.tk * s.hk * s.vh, 55, 1.0));
    let o = g.attention(q, k, v, s);
    let loss = weighted_loss(g, &[o], 210);
    Run { leaves: vec![q, k, v], outs: vec![o], loss }
}

#[test]
fn masked_grouped_query_attention_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("attention") else { return };
    let params = no_params(&dev);
    let shapes = [
        // The decoder's causal self attention, a cross attention over a longer memory, the encoder's one-kv-head top.
        AttnShape { tq: 37, tk: 37, hq: 8, hk: 2, qk: 48, vh: 64, causal: true },
        AttnShape { tq: 21, tk: 130, hq: 8, hk: 8, qk: 48, vh: 64, causal: false },
        AttnShape { tq: 70, tk: 70, hq: 4, hk: 1, qk: 32, vh: 64, causal: false },
        AttnShape { tq: 5, tk: 3, hq: 2, hk: 2, qk: 3, vh: 5, causal: false },
        // A causal sequence longer than one 64-lane chunk over grouped key heads.
        AttnShape { tq: 150, tk: 150, hq: 8, hk: 2, qk: 48, vh: 64, causal: true },
        // Few queries over thousands of keys (the cell memory of a full canvas).
        AttnShape { tq: 6, tk: 4400, hq: 8, hk: 8, qk: 48, vh: 64, causal: false },
    ];
    for s in shapes {
        compare(&dev, &format!("attention tq{} tk{} hq{} hk{} causal {}", s.tq, s.tk, s.hq, s.hk, s.causal), &params, |g| attention_run(g, s));
    }
}

#[test]
fn causal_taps_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("causal_taps") else { return };
    let params = no_params(&dev);
    let (t, w) = (23, 37);
    for (kt, dil) in [(3, 1), (3, 4), (4, 30)] {
        compare(&dev, &format!("taps kt{kt} dil{dil}"), &params, |g| {
            let x = g.leaf(noise(t * w, 56, 1.0));
            let taps = g.leaf(noise(kt * w, 57, 0.5));
            let y = g.causal_taps(x, taps, w, dil);
            let loss = weighted_loss(g, &[y], 220);
            Run { leaves: vec![x, taps], outs: vec![y], loss }
        });
    }
}

#[test]
fn sinkhorn_matches_the_host_through_twenty_rounds() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("sinkhorn") else { return };
    let params = no_params(&dev);
    for (blocks, n, iters) in [(37, 4, 20), (5, 3, 7), (3, 4, 0)] {
        compare(&dev, &format!("sinkhorn {blocks}x{n} iters {iters}"), &params, |g| {
            let l = g.leaf(noise(blocks * n * n, 58, 1.5));
            let y = g.sinkhorn(l, n, iters);
            let loss = weighted_loss(g, &[y], 230);
            Run { leaves: vec![l], outs: vec![y], loss }
        });
    }
}

#[test]
fn lane_mixing_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("row_mix") else { return };
    let params = no_params(&dev);
    for (t, a, b, d) in [(9, 1, 4, 70), (7, 4, 4, 13), (5, 4, 1, 33)] {
        compare(&dev, &format!("row_mix {t} {a}x{b}x{d}"), &params, |g| {
            let h = g.leaf(noise(t * a * b, 59, 1.0));
            let x = g.leaf(noise(t * b * d, 60, 1.0));
            let y = g.row_mix(h, x, a, b, d);
            let loss = weighted_loss(g, &[y], 240);
            Run { leaves: vec![h, x], outs: vec![y], loss }
        });
    }
}

#[test]
fn kronecker_rotations_and_permutations_match_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("kron") else { return };
    let params = no_params(&dev);
    for (rows, fa, fb) in [(7, 4, 6), (3, 16, 32), (5, 3, 3)] {
        let n = fa * fb;
        // A rotation of the columns (a permutation with no fixed structure).
        let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % n as u32).collect();
        let perm = if (0..n as u32).all(|i| perm.iter().filter(|&&p| p == i).count() == 1) { perm } else { (0..n as u32).rev().collect() };
        compare(&dev, &format!("kron {rows}x{fa}x{fb}"), &params, |g| {
            let z = g.leaf(noise(rows * n, 61, 1.0));
            let a = g.leaf(noise(fa * fa, 62, 0.5));
            let b = g.leaf(noise(fb * fb, 63, 0.5));
            let k = g.kron(z, a, b, fa, fb);
            let p = g.permute_cols(k, n, &perm);
            let outs = vec![k, p];
            let loss = weighted_loss(g, &outs, 250);
            Run { leaves: vec![z, a, b], outs, loss }
        });
    }
}

/// Value bound for the losses (the brief's 1e-5).
const LOSS_REL: f64 = 1e-5;

/// A scalar loss as the single output (the random weighting then only multiplies it by one number).
fn loss_run(g: &mut dyn Graph<f32>, logits: Id, loss: Id) -> Run {
    let total = weighted_loss(g, &[loss], 300);
    Run { leaves: vec![logits], outs: vec![loss], loss: total }
}

#[test]
fn token_cross_entropy_matches_the_host_at_decoder_size() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("softmax_ce") else { return };
    let params = no_params(&dev);
    let (n, v) = (240, 8299);
    // Padding and prompt rows carry weight 0 and a target that must not be read (out of range here).
    let weights: Vec<f64> = (0..n).map(|i| if i % 7 == 3 { 0.0 } else { 0.5 + (i % 5) as f64 * 0.25 }).collect();
    let targets: Vec<u32> = (0..n).map(|i| if weights[i] == 0.0 { u32::MAX } else { ((i * 977) % v) as u32 }).collect();
    compare_within(&dev, "softmax_ce", &params, LOSS_REL, |g| {
        let x = g.leaf(noise(n * v, 301, 3.0));
        let l = g.softmax_ce(x, v, &targets, &weights);
        loss_run(g, x, l)
    });
}

#[test]
fn the_focal_heat_loss_matches_the_host_on_a_detection_map() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("focal_heat") else { return };
    let params = no_params(&dev);
    let cells = 64 * 64;
    // Peaks (target 1), Gaussian shoulders, background, and excluded padding cells.
    let target: Vec<f64> = (0..cells).map(|i| if i % 97 == 0 { 1.0 } else if i % 13 == 0 { 0.6 } else { 0.0 }).collect();
    let weight: Vec<f64> = (0..cells).map(|i| if i % 5 == 4 { 0.0 } else { 1.0 }).collect();
    compare_within(&dev, "focal_heat", &params, LOSS_REL, |g| {
        let x = g.leaf(noise(cells, 302, 3.0));
        let l = g.focal_heat(x, &target, &weight, 0.54, 42.0);
        loss_run(g, x, l)
    });
}

#[test]
fn the_weighted_l1_matches_the_host() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("l1") else { return };
    let params = no_params(&dev);
    let n = 64 * 64 * 4;
    let target: Vec<f64> = noise(n, 303, 1.0).into_iter().map(f64::from).collect();
    let weight: Vec<f64> = (0..n).map(|i| if i % 11 == 0 { 1.0 } else { 0.0 }).collect();
    compare_within(&dev, "l1", &params, LOSS_REL, |g| {
        // Some predictions equal their target exactly: the subgradient there is zero.
        let mut xv = noise(n, 304, 1.0);
        for i in (0..n).step_by(33) {
            xv[i] = target[i] as f32;
        }
        let x = g.leaf(xv);
        let l = g.l1(x, &target, &weight, 17.0);
        loss_run(g, x, l)
    });
}

#[test]
fn the_mask_loss_matches_the_host_on_a_128_square_region() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("bce_dice") else { return };
    let params = no_params(&dev);
    let n = 128 * 128;
    let target: Vec<f64> = (0..n).map(|i| if (i / 128 > 30 && i / 128 < 90) && (i % 128 > 20 && i % 128 < 100) { 1.0 } else { 0.0 }).collect();
    compare_within(&dev, "bce_dice", &params, LOSS_REL, |g| {
        let x = g.leaf(noise(n, 305, 4.0));
        let l = g.bce_dice(x, &target, 1.0, 0.7);
        loss_run(g, x, l)
    });
}

#[test]
fn ctc_matches_the_host_including_repeats_and_infeasible_labels() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("ctc") else { return };
    let params = no_params(&dev);
    let (frames, classes) = (80, 307);
    // A reader window: repeated labels need a blank between them; the last case cannot be aligned at all.
    let cases: [(usize, Vec<u32>); 4] = [
        (frames, (0..30).map(|i| 1 + (i * 37 % 300) as u32).collect()),
        (frames, vec![5, 5, 5, 9, 9, 120, 5]),
        (frames, vec![3]),
        (6, vec![4, 4, 4, 4]),
    ];
    for (k, (f, labels)) in cases.iter().enumerate() {
        // The values agree to f32 rounding; the gradient (a difference of two log-space sums) to about 4e-5.
        compare(&dev, &format!("ctc case {k}"), &params, |g| {
            let x = g.leaf(noise(f * classes, 306 + k as u64, 2.0));
            let l = g.ctc(x, classes, labels);
            loss_run(g, x, l)
        });
    }
}

// ---- the whole model graph, device against host ----

use std::collections::BTreeMap;

use mirage::device::train::params::{stacked_grads, upload_params};
use mirage::train::net::{params_from_checkpoint, Net};
use mirage::vision::ops::valid_cells;
use mirage::vision::memory::Slot;

const CANVAS_SIZE: (usize, usize) = (400, 300);

fn tiny_slots(cfg: &mirage::config::MirageConfig) -> Vec<Slot> {
    let base = cfg.text + mirage::config::LANGUAGE_TAGS;
    vec![
        Slot { bbox: [0.1, 0.1, 0.6, 0.5], token: base },
        Slot { bbox: [0.3, 0.2, 0.9, 0.7], token: base + 1 },
        Slot { bbox: [0.05, 0.6, 0.45, 0.95], token: base + cfg.detections },
    ]
}

/// Every head's output through a fixed random projection (as `train_graph`), then the supervised losses on the same
/// maps: one backward reaches the encoder from the detection, mask, reader and decoder heads.
fn whole_graph_loss<T: Graph<f32>>(net: &mut Net<'_, f32, T>) -> Id {
    let cfg = net.cfg;
    let (h, w) = CANVAS_SIZE;
    let canvas = net.constant(noise(cfg.canvas * cfg.canvas * 3, 71, 1.0));
    let enc = net.encoder(canvas, h, w);
    let dense = net.dense(&enc, h, w);
    let (protos, coefs) = net.mask_maps(dense.head_in, &enc, h, w);
    let valid = valid_cells(h, w, cfg.grid());
    let mem = net.decoder_memory(&enc, dense.head_in, &valid, &tiny_slots(cfg));
    let rows = net.tape.len(mem) / cfg.dim;
    let window = net.constant(noise(48 * 320 * 3, 5, 1.0));
    let read = net.reader(window);
    let tokens = [2u32, 24, 5, 17, 36, 9, 30, 1];
    let logits = net.decoder_logits(&tokens, mem, rows);
    let mut parts = Vec::new();
    for (i, id) in [dense.head, protos, coefs, read, logits].into_iter().enumerate() {
        let proj = net.constant(noise(net.tape.len(id), 100 + i as u64, 1.0));
        let prod = net.tape.mul(id, proj);
        parts.push(net.tape.sum_weighted(prod, 1.0));
    }
    // The text and reader losses on top of the projections, so the loss primitives sit in the backward too.
    let v = net.tape.len(logits) / tokens.len();
    let targets: Vec<u32> = tokens.iter().skip(1).chain(std::iter::once(&1)).copied().collect();
    let weights = vec![1.0 / tokens.len() as f64; tokens.len()];
    parts.push(net.tape.softmax_ce(logits, v, &targets, &weights));
    parts.push(net.tape.ctc(read, mirage::config::READER_CLASSES, &[3, 5, 5, 9]));
    let mut total = parts[0];
    for p in &parts[1..] {
        total = net.tape.add(total, *p);
    }
    total
}

/// Per group: concatenated device gradients, concatenated host gradients, worst per-tensor cosine, its tensor, tensor count.
type GroupStats = (Vec<f32>, Vec<f32>, f64, String, usize);

/// Parameter group of a checkpoint tensor name, for the table.
fn group(name: &str) -> String {
    let mut parts = name.split('/');
    let first = parts.next().unwrap_or(name);
    match (first, parts.next()) {
        ("stack", Some("layers")) => "decoder stack (layers)".to_string(),
        ("stack", _) => "decoder stack (routing, norm)".to_string(),
        _ => first.to_string(),
    }
}

#[test]
fn the_whole_graph_on_the_device_gives_the_host_gradients() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("whole graph") else { return };
    let ck = crate::common::tiny_checkpoint();
    let cfg = ck.config.clone();
    let mut params: mirage::train::net::Params<f32> = params_from_checkpoint(&ck);
    for v in params.values_mut().flatten() {
        *v *= 0.5;
    }

    let mut host = Net::<f32>::new(&cfg, &params, None);
    let hl = whole_graph_loss(&mut host);
    let host_loss = host.tape.val(hl)[0];
    let mut grads = host.tape.backward(hl);
    let want = host.param_grads(&mut grads);

    let store = upload_params(&dev, cfg.dec_layers, &params, |_| paramstore::Role::Trainable);
    let mut net = Net::resident(&cfg, DevTape::new(&dev, &store));
    let dl = whole_graph_loss(&mut net);
    let dev_loss = net.tape.scalar(dl);
    net.tape.backward(dl);
    let got = stacked_grads(&dev, &store);
    eprintln!("loss: host {host_loss:.6}, device {dev_loss:.6}");
    assert!((f64::from(host_loss) - dev_loss).abs() <= bounds(&dev, 1e-4).0 * f64::from(host_loss).abs().max(1.0), "loss differs");

    let mut groups: BTreeMap<String, GroupStats> = BTreeMap::new();
    for (name, w) in &want {
        let g = &got[name];
        assert_eq!(g.len(), w.len(), "{name}: gradient length");
        let e = groups.entry(group(name)).or_insert_with(|| (Vec::new(), Vec::new(), 1.0, String::new(), 0));
        e.0.extend_from_slice(g);
        e.1.extend_from_slice(w);
        // A gradient far below the group's scale (the residual routing's ~1e-18) has no resolvable direction.
        let peak = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let c = cosine(g, w);
        if c < 0.99 {
            eprintln!("  low-cosine tensor {name}: host peak gradient {peak:.2e}, cosine {c:.4}");
        }
        if c < e.2 {
            e.2 = c;
            e.3 = name.clone();
        }
        e.4 += 1;
    }
    eprintln!("{:<32} {:>8} {:>14} {:>14}  worst tensor", "group", "tensors", "group cosine", "worst cosine");
    let mut failures = Vec::new();
    for (name, (d, h, worst, worst_name, tensors)) in &groups {
        let c = cosine(d, h);
        eprintln!("{name:<32} {tensors:>8} {c:>14.8} {worst:>14.8}  {worst_name}");
        if c < bounds(&dev, 0.0).1.min(0.9999) {
            failures.push(format!("{name}: group cosine {c}"));
        }
    }
    assert!(failures.is_empty(), "device gradients deviate from the host: {failures:?}");
}
