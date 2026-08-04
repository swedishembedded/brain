// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The imaging-workstream additions to `vision::blocks`: [`ConvTranspose`],
//! [`MaxPool`], [`LayerNorm2d`] and [`CXBlock`].
//!
//! Every block is checked TWICE — forward against an independent host reference
//! written straight from the operator's definition (never from the block's own
//! dispatch), and backward against finite differences of that same forward. An
//! oracle that shares code with the thing it checks proves nothing, and a
//! mismatched kernel param list is silently wrong rather than a crash: both of
//! these blocks bind kernels whose `Params` word order is easy to get subtly
//! wrong (`maxpool2d`'s `stride` sits BEFORE `pad`; `convtr2d`'s weight has the
//! INPUT channel outermost and always binds at the wrong layout).
//!
//! The correctness tests all run on the CPU backend (`Gpu::new_cpu`), so no GPU
//! is required. `layernorm2d_composition_cost` is the one exception — it is a
//! MEASUREMENT and uses the pooled test device, because a CPU-backend timing
//! would say nothing about the coalescing question it exists to settle.

use data::rng::Lcg;
use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{
    Act, ConvKernelIds, ConvTrSpec, ConvTranspose, Ctx, CxSpec, CXBlock, LayerNorm2d, Ln2dNames,
    MaxPool, PoolSpec, Shape,
};

/// Only the kernels these blocks dispatch. Deliberately not `kernels::ALL`: the
/// CPU backend JIT-compiles every registered kernel at device creation.
const PIPELINES: &[(&str, &str)] = &[
    // conv (CXBlock's depthwise + pointwise stages)
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_dx", kernels::CONV2D_GD_DX),
    ("conv2d_gd_dw", kernels::CONV2D_GD_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("bias_grad", kernels::BIAS_GRAD),
    // transposed conv
    ("convtr2d", kernels::CONVTR2D),
    ("convtr2d_dx", kernels::CONVTR2D_DX),
    ("convtr2d_dw", kernels::CONVTR2D_DW),
    // pooling
    ("maxpool2d", kernels::MAXPOOL2D),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    // LayerNorm2d: the permutations + BOTH LayerNorm variants, so the
    // `model::block` seam can pick the coalesced `*_rows` twins where the
    // device supports a workgroup reduction.
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats", kernels::LN_STATS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    // activations
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    // elementwise / layer scale
    ("scale_chan", kernels::SCALE_CHAN),
    ("scale_chan_dg", kernels::SCALE_CHAN_DG),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
];

fn ids() -> &'static ConvKernelIds {
    static IDS: std::sync::OnceLock<ConvKernelIds> = std::sync::OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

fn dev() -> Gpu {
    Gpu::new_cpu(PIPELINES)
}

fn store(gpu: &Gpu, params: Vec<(String, usize)>, seed: u64) -> ParamStore {
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        init.insert(n.clone(), Lcg::new(seed ^ (i as u64 * 0x9E37)).vec(*numel).iter().map(|v| 0.5 * v).collect());
    }
    ParamStore::new(gpu, params, &init)
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
            "{what}: element {i} differs — got {x}, want {y}"
        );
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose
// ---------------------------------------------------------------------------

/// torch's `ConvTranspose2d`, written from the definition: every input tap
/// (hi,wi,kh,kw) SCATTERS onto `ho = hi*stride - pad + kh*dilation`. The kernel
/// computes the same thing as a gather over the inverted map, so agreeing here
/// is a real cross-check of that inversion (and of the exact-division test the
/// gather form needs at stride > 1).
///
/// Weight layout is torch's `[Cin, Cout/G, K, K]`.
#[allow(clippy::too_many_arguments)]
fn convtr2d_ref(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    xs: Shape,
    spec: &ConvTrSpec,
    out: Shape,
) -> Vec<f32> {
    let (n, cin, h, wd) = (xs.n as usize, xs.c as usize, xs.h as usize, xs.w as usize);
    let (cout, ho, wo) = (out.c as usize, out.h as usize, out.w as usize);
    let (k, stride, pad, dil) = (spec.k as usize, spec.stride as usize, spec.pad as isize, spec.dilation as usize);
    let g = spec.groups as usize;
    let (cin_g, cout_g) = (cin / g, cout / g);
    let mut y = vec![0.0f32; n * cout * ho * wo];
    for ni in 0..n {
        for ci in 0..cin {
            let grp = ci / cin_g;
            for hi in 0..h {
                for wi in 0..wd {
                    let xv = x[((ni * cin + ci) * h + hi) * wd + wi];
                    for co_l in 0..cout_g {
                        let co = grp * cout_g + co_l;
                        for kh in 0..k {
                            let oh = hi as isize * stride as isize - pad + (kh * dil) as isize;
                            if oh < 0 || oh >= ho as isize {
                                continue;
                            }
                            for kw in 0..k {
                                let ow = wi as isize * stride as isize - pad + (kw * dil) as isize;
                                if ow < 0 || ow >= wo as isize {
                                    continue;
                                }
                                let wv = w[((ci * cout_g + co_l) * k + kh) * k + kw];
                                y[((ni * cout + co) * ho + oh as usize) * wo + ow as usize] += xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        for ni in 0..n {
            for co in 0..cout {
                for p in 0..ho * wo {
                    y[(ni * cout + co) * ho * wo + p] += b[co];
                }
            }
        }
    }
    y
}

/// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7) — the same series
/// `gelu_erf.wgsl` inlines, evaluated here in f64 so the reference is not a copy
/// of the kernel's fp32 rounding.
fn erf(x: f32) -> f32 {
    let (s, a) = (x.signum() as f64, x.abs() as f64);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (s * (1.0 - poly * (-a * a).exp())) as f32
}

fn convtr_case(xs: Shape, spec: ConvTrSpec, seed: u64) {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let ct = ConvTranspose::torch(&ctx, "deconv", xs, spec);
    let ps = store(&gpu, ct.param_list(), seed);
    let x = Lcg::new(seed ^ 0xABC).vec(xs.numel() as usize);
    let xb = gpu.storage_init("x", &x);
    ct.forward(&ctx, &ps, &xb);
    let got = gpu.read(ct.out(), ct.out_shape.numel() as usize);

    let w = ps.read_weight(&gpu, ct.names().weight.as_str());
    let b = spec.bias.then(|| ps.read_weight(&gpu, ct.names().bias.as_str()));
    let mut want = convtr2d_ref(&x, &w, b.as_deref(), xs, &spec, ct.out_shape);
    match spec.act {
        Act::None => {}
        // torch's `nn.GELU()`: 0.5x(1 + erf(x/sqrt 2)), via the same
        // Abramowitz & Stegun 7.1.26 series the kernel inlines.
        Act::GeluErf => want.iter_mut().for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2))),
        other => panic!("this reference covers Act::None and Act::GeluErf, not {other:?}"),
    }
    close(&got, &want, 2e-5, "convtr2d forward");
}

#[test]
fn convtranspose_forward_matches_a_scatter_reference() {
    // The plain 2x upsample SAM 2's mask decoder does twice.
    convtr_case(Shape::new(1, 3, 5, 4), ConvTrSpec::new(4, 2, 2, 0), 11);
    // stride 2 with padding — the regime where the gather form needs its exact
    // divisibility test and where `pad` crops real output positions.
    convtr_case(Shape::new(2, 3, 4, 4), ConvTrSpec::new(2, 3, 2, 1), 12);
    // output_padding un-crops the far-side band. It is NOT zero-fill, which is
    // exactly what a scatter reference proves.
    convtr_case(Shape::new(1, 2, 4, 4), ConvTrSpec::new(2, 3, 2, 1).with_out_pad(1), 13);
    // grouped + dilated + bias-free.
    convtr_case(Shape::new(1, 4, 5, 5), ConvTrSpec::new(6, 3, 1, 1).with_groups(2).with_dilation(2).bias_free(), 14);
    // with a GELU tail, the SAM 2 shape.
    convtr_case(Shape::new(1, 3, 4, 4), ConvTrSpec::new(3, 2, 2, 0).with_act(Act::GeluErf), 15);
}

/// `out_pad` must widen the output, not merely pad it: the extra row/column
/// carries real signal. A block that treated it as zero-fill would pass every
/// shape assertion and every "did it run" check.
#[test]
fn output_padding_band_is_not_zeros() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let xs = Shape::new(1, 2, 4, 4);
    let spec = ConvTrSpec::new(2, 3, 2, 1).with_out_pad(1).bias_free();
    let ct = ConvTranspose::torch(&ctx, "d", xs, spec);
    // Ho = (4-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 8. Without out_pad it would be 7.
    assert_eq!(ct.out_shape, Shape::new(1, 2, 8, 8));
    let ps = store(&gpu, ct.param_list(), 21);
    let xb = gpu.storage_init("x", &Lcg::new(9).vec(xs.numel() as usize));
    ct.forward(&ctx, &ps, &xb);
    let out = gpu.read(ct.out(), ct.out_shape.numel() as usize);
    let last_row: f32 = (0..8).map(|c| out[7 * 8 + c].abs()).sum();
    assert!(last_row > 1e-3, "the output_padding band came out all zeros ({last_row}) — it is not zero-fill");
}

// ---------------------------------------------------------------------------
// MaxPool
// ---------------------------------------------------------------------------

#[test]
fn maxpool_forward_matches_a_host_reference() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    for spec in [PoolSpec::same5(), PoolSpec::half(), PoolSpec::new(3, 2, 1)] {
        let xs = Shape::new(2, 3, 7, 6);
        let mp = MaxPool::new(&ctx, xs, spec);
        // Strictly negative input: a kernel that seeded its running max at 0.0
        // instead of the first in-bounds tap would let zero PADDING win, and
        // every-value-positive test data hides that.
        let x: Vec<f32> = Lcg::new(31).vec(xs.numel() as usize).iter().map(|v| -1.0 - v.abs()).collect();
        let xb = gpu.storage_init("x", &x);
        mp.forward(&ctx, &xb);
        let got = gpu.read(mp.out(), mp.out_shape.numel() as usize);

        let (h, w) = (xs.h as isize, xs.w as isize);
        let (k, st, pd) = (spec.k as isize, spec.stride as isize, spec.pad as isize);
        let mut want = Vec::new();
        for n in 0..xs.n as isize {
            for c in 0..xs.c as isize {
                for oh in 0..mp.out_shape.h as isize {
                    for ow in 0..mp.out_shape.w as isize {
                        let mut m = f32::NEG_INFINITY;
                        for kh in 0..k {
                            for kw in 0..k {
                                let (hi, wi) = (oh * st - pd + kh, ow * st - pd + kw);
                                if hi < 0 || hi >= h || wi < 0 || wi >= w {
                                    continue;
                                }
                                m = m.max(x[(((n * xs.c as isize + c) * h + hi) * w + wi) as usize]);
                            }
                        }
                        want.push(m);
                    }
                }
            }
        }
        close(&got, &want, 1e-6, &format!("maxpool2d k{} s{} p{}", spec.k, spec.stride, spec.pad));
    }
}

/// The backward must route each output's gradient to the ONE input that won its
/// window, and nowhere else. Checked against a host argmax, not against the
/// block's own `argmax` buffer.
#[test]
fn maxpool_backward_routes_gradient_to_the_winning_tap() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let xs = Shape::new(1, 2, 6, 6);
    let spec = PoolSpec::new(3, 2, 1);
    let mp = MaxPool::new(&ctx, xs, spec);
    let x = Lcg::new(41).vec(xs.numel() as usize);
    let xb = gpu.storage_init("x", &x);
    mp.forward(&ctx, &xb);
    let d_out = Lcg::new(42).vec(mp.out_shape.numel() as usize);
    let dob = gpu.storage_init("dy", &d_out);
    mp.backward(&ctx, &dob);
    let got = gpu.read(mp.d_in(), xs.numel() as usize);

    let (h, w) = (xs.h as isize, xs.w as isize);
    let (k, st, pd) = (spec.k as isize, spec.stride as isize, spec.pad as isize);
    let mut want = vec![0.0f32; xs.numel() as usize];
    let mut o = 0usize;
    for c in 0..xs.c as isize {
        for oh in 0..mp.out_shape.h as isize {
            for ow in 0..mp.out_shape.w as isize {
                let (mut best, mut bi) = (f32::NEG_INFINITY, usize::MAX);
                for kh in 0..k {
                    for kw in 0..k {
                        let (hi, wi) = (oh * st - pd + kh, ow * st - pd + kw);
                        if hi < 0 || hi >= h || wi < 0 || wi >= w {
                            continue;
                        }
                        let idx = ((c * h + hi) * w + wi) as usize;
                        if x[idx] > best {
                            best = x[idx];
                            bi = idx;
                        }
                    }
                }
                want[bi] += d_out[o];
                o += 1;
            }
        }
    }
    close(&got, &want, 1e-6, "maxpool2d_dx");
}

// ---------------------------------------------------------------------------
// LayerNorm2d
// ---------------------------------------------------------------------------

fn ln2d_ref(x: &[f32], gamma: &[f32], beta: &[f32], s: Shape, eps: f32) -> Vec<f32> {
    let (n, c, hw) = (s.n as usize, s.c as usize, (s.h * s.w) as usize);
    let mut y = vec![0.0f32; x.len()];
    for ni in 0..n {
        for p in 0..hw {
            let at = |ch: usize| x[(ni * c + ch) * hw + p];
            let mean = (0..c).map(at).sum::<f32>() / c as f32;
            let var = (0..c).map(|ch| (at(ch) - mean) * (at(ch) - mean)).sum::<f32>() / c as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for ch in 0..c {
                y[(ni * c + ch) * hw + p] = (at(ch) - mean) * inv * gamma[ch] + beta[ch];
            }
        }
    }
    y
}

#[test]
fn layernorm2d_normalizes_across_channels_not_across_space() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let s = Shape::new(2, 6, 5, 4);
    let eps = 1e-6;
    let ln = LayerNorm2d::new(&ctx, Ln2dNames::torch("norm"), s, eps);
    let ps = store(&gpu, ln.param_list(), 51);
    let x = Lcg::new(52).vec(s.numel() as usize);
    let xb = gpu.storage_init("x", &x);
    ln.forward(&ctx, &ps, &xb);
    let got = gpu.read(ln.out(), s.numel() as usize);
    let gamma = ps.read_weight(&gpu, "norm.weight");
    let beta = ps.read_weight(&gpu, "norm.bias");
    close(&got, &ln2d_ref(&x, &gamma, &beta, s, eps), 1e-5, "LayerNorm2d forward");

    // The axis is the trap. A LayerNorm over H*W instead of C would still be a
    // plausible-looking normalization; pin the axis by making one channel a
    // constant offset and checking the OTHER channels moved with it.
    let mut x2 = x.clone();
    let hw = (s.h * s.w) as usize;
    for p in 0..hw {
        x2[p] += 10.0; // image 0, channel 0 only
    }
    let x2b = gpu.storage_init("x2", &x2);
    ln.forward(&ctx, &ps, &x2b);
    let got2 = gpu.read(ln.out(), s.numel() as usize);
    let moved = (0..hw).any(|p| (got2[hw + p] - got[hw + p]).abs() > 1e-3);
    assert!(moved, "channel 1 did not react to channel 0's shift — the norm is not over C");
}

// ---------------------------------------------------------------------------
// finite-difference gradient checks
// ---------------------------------------------------------------------------

/// Central-difference check of `analytic` against `loss(x)`, where the loss is
/// `sum(out * seed_weights)` so every output element contributes.
fn fd_check(mut loss: impl FnMut(&[f32]) -> f32, base: &[f32], analytic: &[f32], tol: f32, what: &str) {
    let h = 2e-3f32;
    // Every element for small tensors; a strided sample for larger ones, so the
    // check stays O(seconds) without becoming a spot check of one corner.
    let stride = (base.len() / 24).max(1);
    let mut probe = base.to_vec();
    for i in (0..base.len()).step_by(stride) {
        probe[i] = base[i] + h;
        let lp = loss(&probe);
        probe[i] = base[i] - h;
        let lm = loss(&probe);
        probe[i] = base[i];
        let num = (lp - lm) / (2.0 * h);
        let scale = 1.0 + num.abs().max(analytic[i].abs());
        assert!(
            (num - analytic[i]).abs() <= tol * scale,
            "{what}: grad[{i}] analytic {} vs numeric {num}",
            analytic[i]
        );
    }
}

#[test]
fn convtranspose_backward_matches_finite_differences() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let xs = Shape::new(1, 3, 4, 4);
    let spec = ConvTrSpec::new(4, 3, 2, 1).with_act(Act::GeluErf);
    let ct = ConvTranspose::torch(&ctx, "d", xs, spec);
    let ps = store(&gpu, ct.param_list(), 61);
    let on = ct.out_shape.numel() as usize;
    let dw = Lcg::new(62).vec(on); // the loss's output weights

    let x = Lcg::new(63).vec(xs.numel() as usize);
    let xb = gpu.storage(xs.numel() as u64);
    let mut fwd = |xv: &[f32]| {
        gpu.write(&xb, bytemuck::cast_slice(xv));
        ct.forward(&ctx, &ps, &xb);
        gpu.read(ct.out(), on).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };

    // analytic grads
    gpu.write(&xb, bytemuck::cast_slice(&x));
    ct.forward(&ctx, &ps, &xb);
    let dob = gpu.storage_init("dy", &dw);
    let dxb = gpu.storage(xs.numel() as u64);
    ps.zero_grads(&gpu);
    ct.backward(&ctx, &ps, &xb, &dob, &dxb);
    let d_x = gpu.read(&dxb, xs.numel() as usize);
    let d_w = ps.read_grad(&gpu, ct.names().weight.as_str());
    let d_b = ps.read_grad(&gpu, ct.names().bias.as_str());

    fd_check(&mut fwd, &x, &d_x, 3e-2, "convtr2d_dx");

    let w0 = ps.read_weight(&gpu, ct.names().weight.as_str());
    let wb = ps.w(ct.names().weight.as_str()).clone();
    gpu.write(&xb, bytemuck::cast_slice(&x));
    let mut fwd_w = |wv: &[f32]| {
        gpu.write(&wb, bytemuck::cast_slice(wv));
        ct.forward(&ctx, &ps, &xb);
        gpu.read(ct.out(), on).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_w, &w0, &d_w, 3e-2, "convtr2d_dw");
    gpu.write(&wb, bytemuck::cast_slice(&w0));

    let b0 = ps.read_weight(&gpu, ct.names().bias.as_str());
    let bb = ps.w(ct.names().bias.as_str()).clone();
    let mut fwd_b = |bv: &[f32]| {
        gpu.write(&bb, bytemuck::cast_slice(bv));
        ct.forward(&ctx, &ps, &xb);
        gpu.read(ct.out(), on).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_b, &b0, &d_b, 3e-2, "convtranspose bias grad");
}

#[test]
fn layernorm2d_backward_matches_finite_differences() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let s = Shape::new(1, 8, 4, 3);
    let ln = LayerNorm2d::new(&ctx, Ln2dNames::torch("n"), s, 1e-6);
    let ps = store(&gpu, ln.param_list(), 71);
    let n = s.numel() as usize;
    let dw = Lcg::new(72).vec(n);
    let x = Lcg::new(73).vec(n);
    let xb = gpu.storage(n as u64);
    let mut fwd = |xv: &[f32]| {
        gpu.write(&xb, bytemuck::cast_slice(xv));
        ln.forward(&ctx, &ps, &xb);
        gpu.read(ln.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };

    gpu.write(&xb, bytemuck::cast_slice(&x));
    ln.forward(&ctx, &ps, &xb);
    let dob = gpu.storage_init("dy", &dw);
    let dxb = gpu.storage(n as u64);
    ps.zero_grads(&gpu);
    ln.backward(&ctx, &ps, &dob, &dxb);
    let d_x = gpu.read(&dxb, n);
    let d_g = ps.read_grad(&gpu, "n.weight");
    let d_b = ps.read_grad(&gpu, "n.bias");
    fd_check(&mut fwd, &x, &d_x, 3e-2, "layernorm_dx");

    gpu.write(&xb, bytemuck::cast_slice(&x));
    let g0 = ps.read_weight(&gpu, "n.weight");
    let gb = ps.w("n.weight").clone();
    let mut fwd_g = |gv: &[f32]| {
        gpu.write(&gb, bytemuck::cast_slice(gv));
        ln.forward(&ctx, &ps, &xb);
        gpu.read(ln.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_g, &g0, &d_g, 3e-2, "layernorm_dgamma");
    gpu.write(&gb, bytemuck::cast_slice(&g0));

    let b0 = ps.read_weight(&gpu, "n.bias");
    let bb = ps.w("n.bias").clone();
    let mut fwd_b = |bv: &[f32]| {
        gpu.write(&bb, bytemuck::cast_slice(bv));
        ln.forward(&ctx, &ps, &xb);
        gpu.read(ln.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_b, &b0, &d_b, 3e-2, "layernorm_dbeta");
}

// ---------------------------------------------------------------------------
// CXBlock
// ---------------------------------------------------------------------------

#[test]
fn cxblock_param_list_matches_the_reference_module() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let s = Shape::new(1, 8, 6, 6);
    let cx = CXBlock::new(&ctx, "neck.0", s, CxSpec::new(), true);
    let names: Vec<String> = cx.param_list().iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(
        names,
        vec![
            "neck.0.dwconv.weight",
            "neck.0.dwconv.bias",
            "neck.0.norm.weight",
            "neck.0.norm.bias",
            "neck.0.pwconv1.weight",
            "neck.0.pwconv1.bias",
            "neck.0.pwconv2.weight",
            "neck.0.pwconv2.bias",
            "neck.0.gamma",
        ]
    );
    let sizes: Vec<usize> = cx.param_list().iter().map(|(_, n)| *n).collect();
    // dwconv is DEPTHWISE: [C, 1, 7, 7], not [C, C, 7, 7].
    assert_eq!(sizes[0], 8 * 7 * 7);
    // pwconv1 is an nn.Linear(8, 32) in the reference — same flat layout as a
    // 1x1 conv weight [32, 8, 1, 1], which is why it loads without permuting.
    assert_eq!(sizes[4], 32 * 8);
    assert_eq!(sizes[6], 8 * 32);
    assert_eq!(sizes[8], 8, "LayerScale gamma is per-channel");

    // Without LayerScale there is no `gamma` tensor at all — a spurious one
    // would fail a strict checkpoint load.
    let plain = CXBlock::new(&ctx, "b", s, CxSpec::new().without_layer_scale(), true);
    assert!(plain.param_list().iter().all(|(n, _)| n != "b.gamma"));
}

/// The block is a residual: at `gamma = 0` it must be exactly the identity. This
/// pins the residual wiring independently of every kernel inside the branch.
#[test]
fn cxblock_is_the_identity_at_zero_layer_scale() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let s = Shape::new(1, 8, 5, 5);
    let cx = CXBlock::new(&ctx, "b", s, CxSpec::new(), true);
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in cx.param_list().iter().enumerate() {
        let v = if n.ends_with(".gamma") { vec![0.0; *numel] } else { Lcg::new(81 ^ i as u64).vec(*numel) };
        init.insert(n.clone(), v);
    }
    let ps = ParamStore::new(&gpu, cx.param_list(), &init);
    let x = Lcg::new(82).vec(s.numel() as usize);
    let xb = gpu.storage_init("x", &x);
    cx.forward(&ctx, &ps, &xb);
    close(&gpu.read(cx.out(), s.numel() as usize), &x, 1e-6, "CXBlock at gamma=0");
}

#[test]
fn cxblock_backward_matches_finite_differences() {
    let gpu = dev();
    let ctx = Ctx::new(&gpu, ids());
    let s = Shape::new(1, 6, 4, 4);
    // k=3/pad=1 keeps the finite-difference sweep cheap; the 7x7 default runs
    // the identical code path (conv2d_gd is general over K).
    let spec = CxSpec { k: 3, pad: 1, mlp_ratio: 2, ..CxSpec::new() };
    let cx = CXBlock::new(&ctx, "b", s, spec, true);
    let ps = store(&gpu, cx.param_list(), 91);
    let n = s.numel() as usize;
    let dw = Lcg::new(92).vec(n);
    let x = Lcg::new(93).vec(n);
    let xb = gpu.storage(n as u64);
    let mut fwd = |xv: &[f32]| {
        gpu.write(&xb, bytemuck::cast_slice(xv));
        cx.forward(&ctx, &ps, &xb);
        gpu.read(cx.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };

    gpu.write(&xb, bytemuck::cast_slice(&x));
    cx.forward(&ctx, &ps, &xb);
    let dob = gpu.storage_init("dy", &dw);
    let dxb = gpu.storage(n as u64);
    ps.zero_grads(&gpu);
    cx.backward(&ctx, &ps, &xb, &dob, &dxb);
    let d_x = gpu.read(&dxb, n);
    let d_gamma = ps.read_grad(&gpu, "b.gamma");
    let d_dwb = ps.read_grad(&gpu, "b.dwconv.bias");
    fd_check(&mut fwd, &x, &d_x, 4e-2, "CXBlock d_in (residual + branch)");

    gpu.write(&xb, bytemuck::cast_slice(&x));
    let g0 = ps.read_weight(&gpu, "b.gamma");
    let gb = ps.w("b.gamma").clone();
    let mut fwd_g = |gv: &[f32]| {
        gpu.write(&gb, bytemuck::cast_slice(gv));
        cx.forward(&ctx, &ps, &xb);
        gpu.read(cx.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_g, &g0, &d_gamma, 3e-2, "CXBlock LayerScale grad");
    gpu.write(&gb, bytemuck::cast_slice(&g0));

    // The depthwise conv is GROUPED and BIASED — the one combination with no
    // fused kernel, so its bias runs as a separate `add_chan_inplace` pass and
    // its gradient through the generic `bias_grad` reduce.
    let b0 = ps.read_weight(&gpu, "b.dwconv.bias");
    let bb = ps.w("b.dwconv.bias").clone();
    let mut fwd_b = |bv: &[f32]| {
        gpu.write(&bb, bytemuck::cast_slice(bv));
        cx.forward(&ctx, &ps, &xb);
        gpu.read(cx.out(), n).iter().zip(&dw).map(|(a, b)| a * b).sum::<f32>()
    };
    fd_check(&mut fwd_b, &b0, &d_dwb, 3e-2, "grouped+biased dwconv bias grad");
}

/// GELU has no code in the fused `conv_act*` / `bn_eval` activation selector,
/// and those kernels fall through to the IDENTITY on an unknown code. Pin that
/// `Act` reports it as unfusable rather than handing out a fabricated `4`.
#[test]
fn gelu_has_no_fused_activation_selector() {
    assert_eq!(Act::None.fused_code(), Some(0));
    assert_eq!(Act::Relu.fused_code(), Some(1));
    assert_eq!(Act::Silu.fused_code(), Some(2));
    assert_eq!(Act::Sigmoid.fused_code(), Some(3));
    assert_eq!(Act::Gelu.fused_code(), None, "a fabricated code would silently drop the activation");
    assert_eq!(Act::GeluErf.fused_code(), None);
}

/// Cross-backend parity on the pooled test device.
///
/// The correctness tests above all run the Cranelift CPU backend, which cannot
/// see a wgpu **usage-scope** violation — and two of these blocks bind a buffer
/// as `read_write` in a dispatch that follows one writing the same buffer
/// (`add_chan_inplace` after `convtr2d` / `conv2d_gd`, `add_inplace` at the end
/// of `CXBlock::backward`). Run the same graphs on whatever `--device` selects
/// and require the values to agree.
#[test]
fn blocks_agree_across_backends() {
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cpu = dev();
    let xs = Shape::new(1, 4, 5, 5);
    let spec = ConvTrSpec::new(6, 3, 2, 1).with_act(Act::GeluErf);
    let x = Lcg::new(111).vec(xs.numel() as usize);

    let run_ct = |g: &Gpu| {
        let ctx = Ctx::new(g, ids());
        let ct = ConvTranspose::torch(&ctx, "d", xs, spec);
        let ps = store(g, ct.param_list(), 112);
        let xb = g.storage_init("x", &x);
        ct.forward(&ctx, &ps, &xb);
        let out = g.read(ct.out(), ct.out_shape.numel() as usize);
        let dob = g.storage_init("dy", &Lcg::new(113).vec(ct.out_shape.numel() as usize));
        let dxb = g.storage(xs.numel() as u64);
        ps.zero_grads(g);
        ct.backward(&ctx, &ps, &xb, &dob, &dxb);
        (out, g.read(&dxb, xs.numel() as usize), ps.read_grad(g, "d.bias"))
    };
    let (o_g, dx_g, db_g) = run_ct(&gpu);
    let (o_c, dx_c, db_c) = run_ct(&cpu);
    close(&o_g, &o_c, 2e-4, "ConvTranspose forward across backends");
    close(&dx_g, &dx_c, 2e-4, "ConvTranspose d_in across backends");
    close(&db_g, &db_c, 2e-4, "ConvTranspose bias grad across backends");

    let s = Shape::new(1, 8, 6, 6);
    let cxs = CxSpec { k: 3, pad: 1, mlp_ratio: 2, ..CxSpec::new() };
    let xc = Lcg::new(114).vec(s.numel() as usize);
    let run_cx = |g: &Gpu| {
        let ctx = Ctx::new(g, ids());
        let cx = CXBlock::new(&ctx, "b", s, cxs, true);
        let ps = store(g, cx.param_list(), 115);
        let xb = g.storage_init("x", &xc);
        cx.forward(&ctx, &ps, &xb);
        let out = g.read(cx.out(), s.numel() as usize);
        let dob = g.storage_init("dy", &Lcg::new(116).vec(s.numel() as usize));
        let dxb = g.storage(s.numel() as u64);
        ps.zero_grads(g);
        cx.backward(&ctx, &ps, &xb, &dob, &dxb);
        (out, g.read(&dxb, s.numel() as usize), ps.read_grad(g, "b.gamma"))
    };
    let (o_g, dx_g, dg_g) = run_cx(&gpu);
    let (o_c, dx_c, dg_c) = run_cx(&cpu);
    close(&o_g, &o_c, 2e-4, "CXBlock forward across backends");
    close(&dx_g, &dx_c, 2e-4, "CXBlock d_in across backends");
    close(&dg_g, &dg_c, 2e-4, "CXBlock LayerScale grad across backends");

    // MaxPool + LayerNorm2d, same treatment.
    let run_mp = |g: &Gpu| {
        let ctx = Ctx::new(g, ids());
        let mp = MaxPool::new(&ctx, s, PoolSpec::new(3, 2, 1));
        let xb = g.storage_init("x", &xc);
        mp.forward(&ctx, &xb);
        let dob = g.storage_init("dy", &Lcg::new(117).vec(mp.out_shape.numel() as usize));
        mp.backward(&ctx, &dob);
        (g.read(mp.out(), mp.out_shape.numel() as usize), g.read(mp.d_in(), s.numel() as usize))
    };
    let (a, b) = run_mp(&gpu);
    let (c, d) = run_mp(&cpu);
    close(&a, &c, 1e-5, "MaxPool forward across backends");
    close(&b, &d, 1e-5, "MaxPool d_in across backends");

    let run_ln = |g: &Gpu| {
        let ctx = Ctx::new(g, ids());
        let ln = LayerNorm2d::new(&ctx, Ln2dNames::torch("n"), s, 1e-6);
        let ps = store(g, ln.param_list(), 118);
        let xb = g.storage_init("x", &xc);
        ln.forward(&ctx, &ps, &xb);
        let out = g.read(ln.out(), s.numel() as usize);
        let dob = g.storage_init("dy", &Lcg::new(119).vec(s.numel() as usize));
        let dxb = g.storage(s.numel() as u64);
        ps.zero_grads(g);
        ln.backward(&ctx, &ps, &dob, &dxb);
        (out, g.read(&dxb, s.numel() as usize), ps.read_grad(g, "n.weight"))
    };
    let (a, b, e) = run_ln(&gpu);
    let (c, d, f) = run_ln(&cpu);
    close(&a, &c, 2e-4, "LayerNorm2d forward across backends");
    close(&b, &d, 2e-4, "LayerNorm2d d_in across backends");
    close(&e, &f, 2e-4, "LayerNorm2d dgamma across backends");
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

/// What the composed LayerNorm2d costs, split into the two NCHW<->NLC permutes
/// and the LayerNorm itself.
///
/// This is the measurement `docs/imaging/plan.md` §3.1 asks for BEFORE anyone
/// adds a fused `layernorm2d` kernel. A fused channels-first kernel replaces the
/// permutes but must then walk each position's channels with ONE thread — the
/// documented coalescing trap — so it only wins if the permutes dominate by more
/// than the ~8x sector amplification it would take on. Run with `--nocapture`.
///
/// **`Gpu::submit` is not a synchronisation point.** On the wgpu backend it
/// appends the dispatches to a pending list; nothing reaches the queue until a
/// `read`/`write`/`flush`/`poll_wait`. A timing loop of bare `submit`s therefore
/// measures bind-group construction on the HOST and reports it as device time —
/// which is how an earlier version of this test produced "377 GB/s" on a card
/// whose peak is ~346 GB/s, a self-refuting number. Every timed region below is
/// bracketed by [`Gpu::poll_wait`], which flushes the pending pass and blocks
/// until the device has finished it.
#[test]
fn layernorm2d_composition_cost() {
    // The pooled test device, so the number reflects whatever `--device` /
    // `BRAIN_DEVICE` selects — a CPU-backend timing would say nothing about the
    // coalescing argument this measurement exists to settle.
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let ctx = Ctx::new(&gpu, ids());
    println!("backend: {} / {:?}", gpu_core::backend_name(), gpu_core::adapter_info());
    // The card's advertised peak, so the table can be read against the roof
    // rather than against intuition. A measured figure ABOVE this is a broken
    // measurement, never a fast kernel.
    println!("(a Tesla P40's peak is ~346 GB/s; anything above the roof means the timing is wrong)");
    // SAM 2 Hiera-B+ at 1024x1024 (stage 1-4) plus one tiny shape, so the table
    // spans the dispatch-latency-bound and the bandwidth-bound regimes.
    for s in [
        Shape::new(1, 96, 64, 64),
        Shape::new(1, 112, 256, 256),
        Shape::new(1, 224, 128, 128),
        Shape::new(1, 448, 64, 64),
        Shape::new(1, 896, 32, 32),
    ] {
        let ln = LayerNorm2d::new(&ctx, Ln2dNames::torch("n"), s, 1e-6);
        let ps = store(&gpu, ln.param_list(), 101);
        let n = s.numel() as usize;
        let xb = gpu.storage_init("x", &Lcg::new(102).vec(n));
        let tmp = gpu.storage(n as u64);
        let perm = [s.numel(), s.c, s.h * s.w];
        let reps = 20;

        // warm-up (pipeline JIT + first touch), drained and never measured.
        for _ in 0..2 {
            ln.forward(&ctx, &ps, &xb);
        }
        gpu.poll_wait();

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ln.forward(&ctx, &ps, &xb);
        }
        gpu.poll_wait();
        let whole = t0.elapsed().as_secs_f64() / reps as f64;

        // The two permutes alone. `nlc_nchw(nchw_nlc(x)) == x` bitwise, so
        // writing back into `xb` leaves the input unchanged between reps.
        for _ in 0..2 {
            let a = ctx.step(ids().nchw_nlc, &[&xb, &tmp], &perm, s.numel());
            let b = ctx.step(ids().nlc_nchw, &[&tmp, &xb], &perm, s.numel());
            gpu.submit(&[], &[a, b]);
        }
        gpu.poll_wait();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            let a = ctx.step(ids().nchw_nlc, &[&xb, &tmp], &perm, s.numel());
            let b = ctx.step(ids().nlc_nchw, &[&tmp, &xb], &perm, s.numel());
            gpu.submit(&[], &[a, b]);
        }
        gpu.poll_wait();
        let permutes = t1.elapsed().as_secs_f64() / reps as f64;

        // Each permute reads n and writes n floats; two of them move 4n floats.
        let perm_gbs = 4.0 * n as f64 * 4.0 / permutes / 1e9;
        println!(
            "LayerNorm2d {:?} ({:5.1} MiB): total {:7.3} ms | 2 permutes {:7.3} ms ({:4.1}%, {:5.1} GB/s) | norm {:7.3} ms",
            (s.n, s.c, s.h, s.w),
            n as f64 * 4.0 / (1024.0 * 1024.0),
            whole * 1e3,
            permutes * 1e3,
            100.0 * permutes / whole,
            perm_gbs,
            (whole - permutes) * 1e3
        );
        // The roof check, asserted rather than left to the reader: a permute
        // that "runs" faster than the card can move bytes did not run.
        assert!(
            perm_gbs < 700.0,
            "permute measured at {perm_gbs:.0} GB/s — above any plausible roof, so the timing is not \
             measuring the device (is every timed region bracketed by poll_wait?)"
        );
    }
}
