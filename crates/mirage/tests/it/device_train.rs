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

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-6);
    let max_err = got.iter().zip(want).fold(0f64, |m, (g, w)| m.max(f64::from((g - w).abs()))) / scale;
    let cos = cosine(got, want);
    eprintln!("{what}: cosine {cos:.8}, max relative error {max_err:.2e}");
    assert!(cos >= MIN_COSINE, "{what}: cosine {cos}");
    assert!(max_err <= MAX_REL, "{what}: max relative error {max_err:.3e}");
}

/// Build the graph on the host and on the device and compare every output and every leaf gradient.
fn compare(dev: &Device, name: &str, params: &ParamStore, build: impl Fn(&mut dyn Graph<f32>) -> Run) {
    let mut host = Tape::<f32>::new();
    let hr = build(&mut host);
    let hgrads = host.backward(hr.loss);

    let mut tape = DevTape::new(dev, params).with_leaf_grads();
    let dr = build(&mut tape);
    for (i, (&h, &d)) in hr.outs.iter().zip(&dr.outs).enumerate() {
        assert_close(&format!("{name} output {i}"), &tape.value(d), host.val(h));
    }
    tape.backward(dr.loss);
    assert_eq!(hr.leaves.len(), dr.leaves.len());
    for (i, (&h, &d)) in hr.leaves.iter().zip(&dr.leaves).enumerate() {
        let want = hgrads.get(h).map(<[f32]>::to_vec).unwrap_or_else(|| vec![0.0; host.len(h)]);
        let got = tape.grad(d).unwrap_or_else(|| vec![0.0; want.len()]);
        assert_close(&format!("{name} gradient of leaf {i}"), &got, &want);
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
    assert_close("frozen: input gradient", &tape.grad(dr.leaves[0]).unwrap(), hgrads.get(hr.leaves[0]).unwrap());
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
