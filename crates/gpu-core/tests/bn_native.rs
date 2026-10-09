// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native BatchNorm reductions (`kernels_cuda`,
//! `cu/bn_f32.cu`) that `gpu_core::native_upgrade` substitutes for `bn_stats`,
//! `bn_dstats`, `bn_dgamma` and `bn_dbeta` on a CUDA device.
//!
//! Swedish Embedded AB implements training kernels for convolutional networks.
//! If your team needs expertise in fast and verifiably correct normalisation
//! layers, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # The bar
//!
//! The native kernels re-associate each channel's sum (per-thread partials,
//! then a fixed tree), so they are held to the fp32 summation bound against an
//! f64 oracle, not to the reference's bits. For a sum of `L` terms `t_i`, each
//! itself computed with `r` roundings, any evaluation order obeys
//! `|computed - exact| <= gamma_{L+r} * sum |t_i|`, `gamma_n = n u / (1 - n u)`,
//! `u = 2^-24` (Higham, 2nd ed., §3.1, §4.2). Per output:
//!
//! - `mean`: `L = M = N*H*W` plain terms, then one division: `gamma_{M+1} sum|x| / M`;
//! - `var` (two passes): each term `(x - m)^2` carries three roundings and the
//!   computed mean's own error `d` enters only squared (the cross term sums to
//!   zero about the exact mean), so `gamma_{M+4} sum (x - m)^2 / M + d^2`;
//! - `sum dy`: `gamma_M sum|dy|`, plus one rounding for the accumulation into
//!   an existing gradient;
//! - `sum dy*xhat`, `xhat = (x - mean) * inv`: four roundings a term with
//!   `inv` itself one rounding off, `gamma_{M+5} sum|dy * xhat|`.
//!
//! Both the native kernels and the WGSL reference are held to it - the
//! reference passing is what shows the oracle and the bound are right.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[
    ("bn_stats", kernels::BN_STATS),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("bn_stats_ref", kernels::BN_STATS),
    ("bn_dstats_ref", kernels::BN_DSTATS),
    ("bn_dgamma_ref", kernels::BN_DGAMMA),
    ("bn_dbeta_ref", kernels::BN_DBETA),
];
/// Slot of each kernel, native-redirected first and the reference copies
/// `REF` slots later.
const STATS: usize = 0;
const DSTATS: usize = 1;
const DGAMMA: usize = 2;
const DBETA: usize = 3;
const REF: usize = 4;

const EPS: f64 = 1e-5;
const U: f64 = 1.0 / (1u64 << 24) as f64;

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && !std::env::var("BRAIN_NO_NATIVE_KERNELS").is_ok_and(|v| v != "0")
}

#[derive(Clone, Copy, Debug)]
struct Bn {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
}

impl Bn {
    fn len(&self) -> usize {
        (self.n * self.c * self.h * self.w) as usize
    }
    fn per_channel(&self) -> usize {
        (self.n * self.h * self.w) as usize
    }
    fn params(&self) -> [u32; 4] {
        [self.n, self.c, self.h, self.w]
    }
    /// Element `i` of channel `c` (i over `N*H*W`, image-major).
    fn at(&self, c: usize, i: usize) -> usize {
        let hw = (self.h * self.w) as usize;
        let (n, j) = (i / hw, i % hw);
        (n * self.c as usize + c) * hw + j
    }
}

/// Every BatchNorm map of a YOLOv8n training step (batch 8, 512 input) -
/// `(C, side)` - plus maps whose `H*W` is not a multiple of four (the scalar
/// path), a single pixel, and channel counts that are no tile's multiple.
fn shapes() -> Vec<Bn> {
    let mut v: Vec<Bn> = [(16, 256), (32, 128), (16, 128), (64, 64), (32, 64), (128, 32), (64, 32), (256, 16), (128, 16), (64, 16)]
        .iter()
        .map(|&(c, s)| Bn { n: 8, c, h: s, w: s })
        .collect();
    v.extend([Bn { n: 3, c: 3, h: 7, w: 5 }, Bn { n: 1, c: 5, h: 1, w: 1 }, Bn { n: 2, c: 33, h: 13, w: 11 }, Bn { n: 1, c: 17, h: 6, w: 6 }]);
    v
}

struct Inputs {
    x: Vec<f32>,
    dy: Vec<f32>,
    /// `[mean, var, gamma]` per channel, the `mvg` packing.
    mvg: Vec<f32>,
    /// `[mean, var]` per channel, the `mv` packing.
    mv: Vec<f32>,
    dgamma0: Vec<f32>,
    dbeta0: Vec<f32>,
}

fn inputs(b: &Bn, seed: u64) -> Inputs {
    let mut r = data::rng::Lcg::new(seed);
    let c = b.c as usize;
    // A per-channel offset and scale, so the mean is not near zero and the
    // two-pass variance is actually exercised against cancellation.
    let off: Vec<f32> = (0..c).map(|_| r.scaled(3.0)).collect();
    let mut x = r.vec(b.len());
    for ch in 0..c {
        for i in 0..b.per_channel() {
            x[b.at(ch, i)] = off[ch] + 0.5 * x[b.at(ch, i)];
        }
    }
    let dy = r.vec(b.len());
    let mut mvg = Vec::with_capacity(3 * c);
    let mut mv = Vec::with_capacity(2 * c);
    for ch in 0..c {
        let (mean, var, g) = (off[ch] + r.scaled(0.1), 0.05 + r.unit(), r.signed());
        mvg.extend([mean, var, g]);
        mv.extend([mean, var]);
    }
    Inputs { x, dy, mvg, mv, dgamma0: r.vec(c), dbeta0: r.vec(c) }
}

/// What the four kernels produce: `mean`, `var`, `bp` and the accumulated
/// `dgamma`, `dbeta`.
struct Out {
    mean: Vec<f32>,
    var: Vec<f32>,
    bp: Vec<f32>,
    dgamma: Vec<f32>,
    dbeta: Vec<f32>,
}

fn run(gpu: &Gpu, b: &Bn, inp: &Inputs, base: usize) -> Out {
    let c = b.c as u64;
    let p = b.params();
    let x = gpu.storage_init("x", &inp.x);
    let dy = gpu.storage_init("dy", &inp.dy);
    let mvg = gpu.storage_init("mvg", &inp.mvg);
    let mv = gpu.storage_init("mv", &inp.mv);
    let dg = gpu.storage_init("dgamma", &inp.dgamma0);
    let db = gpu.storage_init("dbeta", &inp.dbeta0);
    let mean = gpu.storage(c);
    let var = gpu.storage(c);
    let bp = gpu.storage(5 * c);
    let steps = [
        gpu.step(base + STATS, &[&x, &mean, &var], &p, b.c),
        gpu.step(base + DSTATS, &[&x, &dy, &mvg, &bp], &p, b.c),
        gpu.step(base + DGAMMA, &[&x, &dy, &mv, &dg], &p, b.c),
        gpu.step(base + DBETA, &[&dy, &db], &p, b.c),
    ];
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let c = b.c as usize;
    Out {
        mean: gpu.read(&mean, c),
        var: gpu.read(&var, c),
        bp: gpu.read(&bp, 5 * c),
        dgamma: gpu.read(&dg, c),
        dbeta: gpu.read(&db, c),
    }
}

/// `|got - exact| <= bound`, reported against the bound.
fn within(got: f32, exact: f64, bound: f64, what: &str) -> f64 {
    let err = (f64::from(got) - exact).abs();
    assert!(got.is_finite() && err <= bound, "{what}: {got} vs exact {exact}, |err| {err:e} > bound {bound:e}");
    if bound > 0.0 {
        err / bound
    } else {
        0.0
    }
}

fn check(b: &Bn, inp: &Inputs, o: &Out, who: &str) -> f64 {
    let m = b.per_channel();
    let mut worst = 0.0f64;
    for ch in 0..b.c as usize {
        let tag = format!("{who} {b:?} channel {ch}");
        let xs: Vec<f64> = (0..m).map(|i| f64::from(inp.x[b.at(ch, i)])).collect();
        let ds: Vec<f64> = (0..m).map(|i| f64::from(inp.dy[b.at(ch, i)])).collect();

        // Statistics.
        let mean = xs.iter().sum::<f64>() / m as f64;
        let abs_sum: f64 = xs.iter().map(|v| v.abs()).sum();
        let mean_bound = gamma(m + 1) * abs_sum / m as f64;
        worst = worst.max(within(o.mean[ch], mean, mean_bound, &format!("{tag} mean")));
        let sq: f64 = xs.iter().map(|v| (v - mean) * (v - mean)).sum();
        let var = sq / m as f64;
        worst = worst.max(within(o.var[ch], var, gamma(m + 4) * var + mean_bound * mean_bound, &format!("{tag} var")));

        // Backward sums against the given (mean, var).
        let (mm, vv) = (f64::from(inp.mvg[3 * ch]), f64::from(inp.mvg[3 * ch + 1]));
        let inv = 1.0 / (vv + EPS).sqrt();
        let terms: Vec<f64> = xs.iter().zip(&ds).map(|(x, d)| d * (x - mm) * inv).collect();
        let dxhat: f64 = terms.iter().sum();
        let dxhat_abs: f64 = terms.iter().map(|t| t.abs()).sum();
        let dsum: f64 = ds.iter().sum();
        let dsum_abs: f64 = ds.iter().map(|d| d.abs()).sum();
        assert_eq!(&o.bp[5 * ch..5 * ch + 3], &inp.mvg[3 * ch..3 * ch + 3], "{tag}: bp copies mean|var|gamma through");
        worst = worst.max(within(o.bp[5 * ch + 3], dsum, gamma(m) * dsum_abs, &format!("{tag} dsum")));
        worst = worst.max(within(o.bp[5 * ch + 4], dxhat, gamma(m + 5) * dxhat_abs, &format!("{tag} dxhat_sum")));
        let g0 = f64::from(inp.dgamma0[ch]);
        let b0 = f64::from(inp.dbeta0[ch]);
        worst = worst.max(within(o.dgamma[ch], g0 + dxhat, gamma(m + 6) * (dxhat_abs + g0.abs()), &format!("{tag} dgamma")));
        worst = worst.max(within(o.dbeta[ch], b0 + dsum, gamma(m + 1) * (dsum_abs + b0.abs()), &format!("{tag} dbeta")));
    }
    worst
}

/// Data on which every output is EXACT in fp32 whatever the summation order:
/// per channel, small integers in `[-2, 2]` whose sum is exactly zero (each
/// value paired with its negation), so the mean is exactly 0 and the sum of
/// squares is an exact integer; integer `dy`; and a variance `va` for which
/// `va + eps` rounds to exactly 1, so `xhat = x` and every product is an
/// integer. A kernel that drops, repeats or misplaces one term of a
/// half-million-term channel then differs from the reference in its bits,
/// which no rounding bound that long could show.
fn integer_inputs(b: &Bn, seed: u64) -> Inputs {
    let mut r = data::rng::Lcg::new(seed);
    let c = b.c as usize;
    let m = b.per_channel();
    let mut ints = |n: usize| (0..n).map(|_| (r.next_u32() % 5) as f32 - 2.0).collect::<Vec<f32>>();
    let mut x = vec![0.0f32; b.len()];
    for ch in 0..c {
        let half = ints(m / 2);
        for (k, v) in half.iter().enumerate() {
            x[b.at(ch, 2 * k)] = *v;
            x[b.at(ch, 2 * k + 1)] = -*v;
        }
    }
    let dy = ints(b.len());
    // The largest f32 below 1 whose sum with eps rounds to exactly 1.
    let mut va = 1.0f32 - 1e-5;
    while va + 1e-5f32 != 1.0 {
        va = f32::from_bits(va.to_bits() + if va + 1e-5f32 < 1.0 { 1 } else { u32::MAX });
    }
    let gam = ints(c);
    let mvg = (0..c).flat_map(|ch| [0.0, va, gam[ch]]).collect();
    let mv = (0..c).flat_map(|_| [0.0, va]).collect();
    Inputs { x, dy, mvg, mv, dgamma0: ints(c), dbeta0: ints(c) }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn the_real_yolov8n_batchnorms_are_exact_on_integer_data() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (i, b) in shapes().iter().enumerate() {
        let inp = integer_inputs(b, 900 + i as u64);
        let native = run(&gpu, b, &inp, 0);
        let reference = run(&gpu, b, &inp, REF);
        // The exact answers, independent of both kernels.
        let m = b.per_channel();
        for ch in 0..b.c as usize {
            let xs = (0..m).map(|i| f64::from(inp.x[b.at(ch, i)]));
            let sq: f64 = xs.map(|v| v * v).sum();
            let dsum: f64 = (0..m).map(|i| f64::from(inp.dy[b.at(ch, i)])).sum();
            let dx: f64 = (0..m).map(|i| f64::from(inp.dy[b.at(ch, i)]) * f64::from(inp.x[b.at(ch, i)])).sum();
            let tag = format!("{b:?} channel {ch}");
            assert_eq!(native.mean[ch], 0.0, "{tag}: mean");
            assert_eq!(native.var[ch], (sq as f32) / (m as f32), "{tag}: var");
            assert_eq!(native.bp[5 * ch + 3], dsum as f32, "{tag}: dsum");
            assert_eq!(native.bp[5 * ch + 4], dx as f32, "{tag}: dxhat_sum");
            assert_eq!(native.dgamma[ch], inp.dgamma0[ch] + dx as f32, "{tag}: dgamma");
            assert_eq!(native.dbeta[ch], inp.dbeta0[ch] + dsum as f32, "{tag}: dbeta");
        }
        assert!(bits(&native.mean) == bits(&reference.mean) && bits(&native.var) == bits(&reference.var), "{b:?}: statistics differ");
        assert!(bits(&native.bp) == bits(&reference.bp), "{b:?}: bp differs");
        assert!(bits(&native.dgamma) == bits(&reference.dgamma) && bits(&native.dbeta) == bits(&reference.dbeta), "{b:?}: gradients differ");
    }
}

#[test]
fn the_native_bn_reductions_are_redirected_to_only_on_cuda() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let p = [2, 16, 8, 8];
    if !is_cuda(&gpu) {
        for k in [STATS, DSTATS, DGAMMA, DBETA] {
            assert_eq!(gpu.native_kernel_for(k, &p), None, "only a CUDA device takes a native kernel");
        }
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(STATS, &p), Some("bn_stats_f32"));
    assert_eq!(gpu.native_kernel_for(DSTATS, &p), Some("bn_dstats_f32"));
    assert_eq!(gpu.native_kernel_for(DGAMMA, &p), Some("bn_dgamma_f32"));
    assert_eq!(gpu.native_kernel_for(DBETA, &p), Some("bn_dbeta_f32"));
    assert_eq!(gpu.native_kernel_for(REF + STATS, &p), None, "the reference slots keep the WGSL tier");
    assert_eq!(gpu.native_kernel_for(STATS, &[2, 0, 8, 8]), None, "no channels is not served");
}

#[test]
fn every_yolov8n_batchnorm_is_within_the_fp32_summation_bound_of_an_f64_oracle() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (i, b) in shapes().iter().enumerate() {
        let inp = inputs(b, 300 + i as u64);
        let native = run(&gpu, b, &inp, 0);
        let reference = run(&gpu, b, &inp, REF);
        let wr = check(b, &inp, &reference, "reference");
        let wn = check(b, &inp, &native, "native");
        eprintln!("{b:?}: worst error at {wn:.3} of the bound native, {wr:.3} reference");
    }
}

const ELEM_KERNELS: &[(&str, &str)] = &[
    ("bn_train", kernels::BN_TRAIN),
    ("bn_dx", kernels::BN_DX),
    ("bn_train_ref", kernels::BN_TRAIN),
    ("bn_dx_ref", kernels::BN_DX),
];

/// `bn_train` and `bn_dx` are elementwise: no reduction, so the native kernels
/// do the reference's arithmetic in the reference's order and must reproduce
/// its RAW BITS, at every YOLOv8n BatchNorm map and on the scalar path
/// (`H*W` not a multiple of four), from random data with non-trivial
/// statistics and sums.
#[test]
fn the_native_bn_elementwise_passes_are_bit_identical_to_the_reference() {
    let gpu = gpu_core::testgpu::dev(ELEM_KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(0, &[2, 16, 8, 8]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(0, &[2, 16, 8, 8]), Some("bn_train_f32"));
    assert_eq!(gpu.native_kernel_for(1, &[2, 16, 8, 8]), Some("bn_dx_f32"));
    for (i, b) in shapes().iter().enumerate() {
        let inp = inputs(b, 600 + i as u64);
        let c = b.c as usize;
        let mut r = data::rng::Lcg::new(700 + i as u64);
        let gb: Vec<f32> = (0..c).flat_map(|_| [r.signed(), r.signed()]).collect();
        let bp: Vec<f32> =
            (0..c).flat_map(|ch| [inp.mvg[3 * ch], inp.mvg[3 * ch + 1], inp.mvg[3 * ch + 2], r.scaled(50.0), r.scaled(50.0)]).collect();
        let x = gpu.storage_init("x", &inp.x);
        let dy = gpu.storage_init("dy", &inp.dy);
        let mv = gpu.storage_init("mv", &inp.mv);
        let gbb = gpu.storage_init("gb", &gb);
        let bpb = gpu.storage_init("bp", &bp);
        let outs: Vec<_> = (0..4).map(|_| gpu.storage(b.len() as u64)).collect();
        let p = b.params();
        let n = b.len() as u32;
        let steps = [
            gpu.step(0, &[&x, &mv, &gbb, &outs[0]], &p, n),
            gpu.step(1, &[&x, &dy, &bpb, &outs[1]], &p, n),
            gpu.step(2, &[&x, &mv, &gbb, &outs[2]], &p, n),
            gpu.step(3, &[&x, &dy, &bpb, &outs[3]], &p, n),
        ];
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let rd = |k: usize| bits(&gpu.read(&outs[k], b.len()));
        assert!(rd(0) == rd(2), "{b:?}: bn_train differs from the reference");
        assert!(rd(1) == rd(3), "{b:?}: bn_dx differs from the reference");
    }
}
