// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) block backward vs the gradchecked host reference.
//!
//! `grad.rs` is finite-difference-gradchecked (tests/block_grad.rs); this test
//! confirms the GPU path (`devgrad`) — real training kernels: matmul_dx_reg/
//! matmul_dw_reg, rms_inv_eps/rmsnorm_dw/rmsnorm_dx_eps, attn_bwd_*_bidir,
//! interleaved-RoPE-via-negated-sin, silu_bwd_* — reproduces those gradients.
//! Matching the gradchecked host to fp32 tolerance transitively validates the
//! device gradients. Needs a GPU: `BRAIN_DEV_GPU=1`.

use s3dit::grad::{backward, forward, Dims, Grads, Weights};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

fn vec_of(n: usize, r: &mut impl FnMut() -> f64, scale: f64) -> Vec<f64> {
    (0..n).map(|_| r() * scale).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut n, mut den) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        n += (x - y) * (x - y);
        den += y * y;
    }
    (n / den.max(1e-30)).sqrt()
}

#[test]
fn device_block_backward_matches_host() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 (needs a GPU) for the device block-backward parity test");
        return;
    }
    // A tile-friendly small config: dim 128 (1 tile), 4 heads (hd 32), 16 tokens.
    let d = Dims::new(16, 128, 4);
    let (t, dim, hd, half) = (d.t, d.dim, d.hd, d.half());
    let mut r = rng(0x5EED_1234);
    let w = Weights {
        wq: vec_of(dim * dim, &mut r, 0.05),
        wk: vec_of(dim * dim, &mut r, 0.05),
        wv: vec_of(dim * dim, &mut r, 0.05),
        wo: vec_of(dim * dim, &mut r, 0.05),
        w1: vec_of(d.hidden * dim, &mut r, 0.05),
        w2: vec_of(dim * d.hidden, &mut r, 0.05),
        w3: vec_of(d.hidden * dim, &mut r, 0.05),
        nq: vec_of(hd, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        nk: vec_of(hd, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        an1: vec_of(dim, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        an2: vec_of(dim, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        fn1: vec_of(dim, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        fn2: vec_of(dim, &mut r, 0.1).iter().map(|v| 1.0 + v).collect(),
        adaln_w: vec_of(4 * dim * d.cdim, &mut r, 0.05),
        adaln_b: vec_of(4 * dim, &mut r, 0.05),
    };
    let x = vec_of(t * dim, &mut r, 1.0);
    let c = vec_of(d.cdim, &mut r, 1.0);
    let cos: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.11).cos()).collect();
    let sin: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.11).sin()).collect();
    let dout = vec_of(t * dim, &mut r, 1.0);

    // host (f64, gradchecked)
    let (_o, cache) = forward(d, &w, &x, &c, &cos, &sin);
    let host = backward(d, &w, &cache, &dout);

    // device (f32 kernels)
    let to32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let dev = s3dit::devgrad::block_backward_device(d, &w.to_f32(), &to32(&x), &to32(&c), &to32(&cos), &to32(&sin), &to32(&dout));

    let check = |name: &str, h: &[f64], g: &[f64]| {
        let cos = cosine(h, g);
        let rel = rel_l2(g, h);
        eprintln!("  {name:8} cosine={cos:.6}  rel_l2={rel:.2e}");
        assert!(cos > 0.999, "{name}: device cosine {cos:.6} < 0.999 vs host");
        assert!(rel < 2e-2, "{name}: device rel_l2 {rel:.3e} too high vs host");
    };
    macro_rules! ck {
        ($f:ident) => {
            check(stringify!($f), &host.$f, &dev.$f.iter().map(|&x| x as f64).collect::<Vec<f64>>());
        };
    }
    let _ = &host as &Grads;
    ck!(wq);
    ck!(wk);
    ck!(wv);
    ck!(wo);
    ck!(w1);
    ck!(w2);
    ck!(w3);
    ck!(nq);
    ck!(nk);
    ck!(an1);
    ck!(an2);
    ck!(fn1);
    ck!(fn2);
    ck!(adaln_w);
    ck!(adaln_b);
    ck!(dx);
    ck!(dc);
    eprintln!("Device S³-DiT block backward matches the gradchecked host reference (cosine > 0.999).");
}
