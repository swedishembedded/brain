// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) block backward vs the gradchecked host reference.
//!
//! `grad.rs` is finite-difference-gradchecked (`tests/block_grad.rs`,
//! `gradcheck::check_wan`); this test confirms the device path (`devgrad`) -
//! real training kernels: `matmul_{dx,dw}`, `layernorm_{dgamma,dbeta,dx}`,
//! `rms_inv_eps`/`rmsnorm_dw`/`rmsnorm_dx_eps`, the `attn_bwd_*_cross` family
//! on BOTH attentions, `gate_row_{dg,dh}`, `gelu_bwd` and
//! interleaved-RoPE-via-negated-sine - reproduces those gradients. Matching the
//! gradchecked host transitively validates the device ones.
//!
//! Runs on the CPU backend by default (so it is always exercised) and on the
//! real GPU with `BRAIN_DEV_GPU=1`.

use wan::devgrad::block_backward_device;
use wan::grad::{block_backward, block_forward, BlockGrads, BlockW, Dims, Lin};
use wan::modelgrad::Cfg;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

fn vof(n: usize, r: &mut impl FnMut() -> f64, s: f64) -> Vec<f64> {
    (0..n).map(|_| r() * s).collect()
}
fn gain(n: usize, r: &mut impl FnMut() -> f64, s: f64) -> Vec<f64> {
    (0..n).map(|_| 1.0 + r() * s).collect()
}
fn lin(out: usize, inn: usize, r: &mut impl FnMut() -> f64) -> Lin<f64> {
    Lin { w: vof(out * inn, r, 0.2), b: vof(out, r, 0.05) }
}

pub fn block_w(d: Dims, r: &mut impl FnMut() -> f64) -> BlockW<f64> {
    let (dim, ffn) = (d.dim, d.ffn);
    BlockW {
        modulation: vof(6 * dim, r, 0.05),
        sq: lin(dim, dim, r),
        sk: lin(dim, dim, r),
        sv: lin(dim, dim, r),
        so: lin(dim, dim, r),
        snq: gain(dim, r, 0.1),
        snk: gain(dim, r, 0.1),
        cq: lin(dim, dim, r),
        ck: lin(dim, dim, r),
        cv: lin(dim, dim, r),
        co: lin(dim, dim, r),
        cnq: gain(dim, r, 0.1),
        cnk: gain(dim, r, 0.1),
        norm3_w: gain(dim, r, 0.1),
        norm3_b: vof(dim, r, 0.05),
        ff1: lin(ffn, dim, r),
        ff2: lin(dim, ffn, r),
    }
}

fn f32w(w: &BlockW<f64>) -> BlockW<f32> {
    let v = |x: &Vec<f64>| -> Vec<f32> { x.iter().map(|&a| a as f32).collect() };
    let l = |x: &Lin<f64>| Lin { w: v(&x.w), b: v(&x.b) };
    BlockW {
        modulation: v(&w.modulation),
        sq: l(&w.sq),
        sk: l(&w.sk),
        sv: l(&w.sv),
        so: l(&w.so),
        snq: v(&w.snq),
        snk: v(&w.snk),
        cq: l(&w.cq),
        ck: l(&w.ck),
        cv: l(&w.cv),
        co: l(&w.co),
        cnq: v(&w.cnq),
        cnk: v(&w.cnk),
        norm3_w: v(&w.norm3_w),
        norm3_b: v(&w.norm3_b),
        ff1: l(&w.ff1),
        ff2: l(&w.ff2),
    }
}

/// Every grad tensor in one fixed order, so a mismatch names the tensor.
fn views(g: &BlockGrads<f64>) -> Vec<(&'static str, &Vec<f64>)> {
    vec![
        ("modulation", &g.modulation),
        ("self_attn.q.weight", &g.sq.w), ("self_attn.q.bias", &g.sq.b),
        ("self_attn.k.weight", &g.sk.w), ("self_attn.k.bias", &g.sk.b),
        ("self_attn.v.weight", &g.sv.w), ("self_attn.v.bias", &g.sv.b),
        ("self_attn.o.weight", &g.so.w), ("self_attn.o.bias", &g.so.b),
        ("self_attn.norm_q", &g.snq), ("self_attn.norm_k", &g.snk),
        ("cross_attn.q.weight", &g.cq.w), ("cross_attn.q.bias", &g.cq.b),
        ("cross_attn.k.weight", &g.ck.w), ("cross_attn.k.bias", &g.ck.b),
        ("cross_attn.v.weight", &g.cv.w), ("cross_attn.v.bias", &g.cv.b),
        ("cross_attn.o.weight", &g.co.w), ("cross_attn.o.bias", &g.co.b),
        ("cross_attn.norm_q", &g.cnq), ("cross_attn.norm_k", &g.cnk),
        ("norm3.weight", &g.norm3_w), ("norm3.bias", &g.norm3_b),
        ("ffn.0.weight", &g.ff1.w), ("ffn.0.bias", &g.ff1.b),
        ("ffn.2.weight", &g.ff2.w), ("ffn.2.bias", &g.ff2.b),
        ("dx", &g.dx), ("dctx", &g.dctx),
    ]
}

fn to64(g: &BlockGrads<f32>) -> BlockGrads<f64> {
    let v = |x: &Vec<f32>| -> Vec<f64> { x.iter().map(|&a| a as f64).collect() };
    let l = |x: &Lin<f32>| Lin { w: v(&x.w), b: v(&x.b) };
    BlockGrads {
        modulation: v(&g.modulation),
        sq: l(&g.sq),
        sk: l(&g.sk),
        sv: l(&g.sv),
        so: l(&g.so),
        snq: v(&g.snq),
        snk: v(&g.snk),
        cq: l(&g.cq),
        ck: l(&g.ck),
        cv: l(&g.cv),
        co: l(&g.co),
        cnq: v(&g.cnq),
        cnk: v(&g.cnk),
        norm3_w: v(&g.norm3_w),
        norm3_b: v(&g.norm3_b),
        ff1: l(&g.ff1),
        ff2: l(&g.ff2),
        dx: v(&g.dx),
        dctx: v(&g.dctx),
    }
}

/// Relative L2, well-defined for an all-zero reference.
fn rel_l2(host: &[f64], dev: &[f64]) -> f64 {
    let n = host.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = host.iter().zip(dev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    diff / n.max(1e-9)
}

/// `"gpu"` under `BRAIN_DEV_GPU=1`, else the CPU JIT backend.
fn device() -> &'static str {
    if std::env::var("BRAIN_DEV_GPU").as_deref() == Ok("1") {
        "gpu"
    } else {
        "cpu"
    }
}

struct Fixture {
    d: Dims,
    w: BlockW<f64>,
    x: Vec<f64>,
    e0: Vec<f64>,
    ctx: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    dout: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let cfg = Cfg::tiny();
    let d = cfg.dims();
    let mut r = rng(seed);
    let w = block_w(d, &mut r);
    let half = d.hd() / 2;
    Fixture {
        x: vof(d.t * d.dim, &mut r, 1.0),
        e0: vof(6 * d.dim, &mut r, 0.2),
        ctx: vof(d.te * d.dim, &mut r, 1.0),
        cos: (0..d.t * half).map(|i| (i as f64 * 0.11).cos()).collect(),
        sin: (0..d.t * half).map(|i| (i as f64 * 0.11).sin()).collect(),
        dout: vof(d.t * d.dim, &mut r, 1.0),
        d,
        w,
    }
}

fn f32v(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn the_device_block_forward_matches_the_host_reference() {
    let f = fixture(0x5EED_1234);
    let (want, _) = block_forward(f.d, &f.w, &f.x, &f.e0, &f.ctx, &f.cos, &f.sin);
    let eng = wan::devgrad::BlockDev::on_device(f.d, f.d.t, Some(device()));
    let got = eng.forward(f.d, &f32w(&f.w), &f32v(&f.x), &f32v(&f.e0), &f32v(&f.ctx), &f32v(&f.cos), &f32v(&f.sin));
    let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
    let rel = rel_l2(&want, &got64);
    eprintln!("device block forward ({}) vs host f64: rel_l2 = {rel:.3e}", device());
    assert!(rel < 1e-5, "device forward rel_l2 {rel:.3e}");
}

#[test]
fn the_device_block_backward_matches_the_gradchecked_host_reference() {
    let f = fixture(0x5EED_1234);
    let (_out, cache) = block_forward(f.d, &f.w, &f.x, &f.e0, &f.ctx, &f.cos, &f.sin);
    let host = block_backward(f.d, &f.w, &cache, &f.dout);
    let dev = to64(&block_backward_device(
        f.d,
        &f32w(&f.w),
        &f32v(&f.x),
        &f32v(&f.e0),
        &f32v(&f.ctx),
        &f32v(&f.cos),
        &f32v(&f.sin),
        &f32v(&f.dout),
        Some(device()),
    ));

    let (mut worst, mut worst_name) = (0.0f64, "");
    for ((name, h), (_, g)) in views(&host).into_iter().zip(views(&dev)) {
        let rel = rel_l2(h, g);
        assert!(rel.is_finite(), "{name}: non-finite grad");
        if rel > worst {
            worst = rel;
            worst_name = name;
        }
    }
    eprintln!("device block backward ({}) vs host f64: worst rel_l2 = {worst:.3e} ({worst_name})", device());
    assert!(worst < 5e-5, "device grad rel_l2 {worst:.3e} too high ({worst_name})");
}

/// The device backward must reproduce the HOST f32 backward, not merely land
/// inside a gradcheck tolerance: both are the same math at the same precision,
/// so a real difference means a different function, not rounding.
#[test]
fn the_device_block_backward_matches_the_host_f32_backward() {
    let f = fixture(0xABCD_0007);
    let w32 = f32w(&f.w);
    let (x, e0, ctx, cos, sin, dout) = (f32v(&f.x), f32v(&f.e0), f32v(&f.ctx), f32v(&f.cos), f32v(&f.sin), f32v(&f.dout));
    let (_o, cache) = block_forward(f.d, &w32, &x, &e0, &ctx, &cos, &sin);
    let host = to64(&block_backward(f.d, &w32, &cache, &dout));
    let dev = to64(&block_backward_device(f.d, &w32, &x, &e0, &ctx, &cos, &sin, &dout, Some(device())));

    let (mut worst, mut worst_name) = (0.0f64, "");
    for ((name, h), (_, g)) in views(&host).into_iter().zip(views(&dev)) {
        let rel = rel_l2(h, g);
        if rel > worst {
            worst = rel;
            worst_name = name;
        }
    }
    eprintln!("device block backward ({}) vs host f32: worst rel_l2 = {worst:.3e} ({worst_name})", device());
    assert!(worst < 2e-5, "device/host f32 divergence {worst:.3e} ({worst_name})");
}
