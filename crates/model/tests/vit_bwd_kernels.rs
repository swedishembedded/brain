// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kernel-level gradcheck for the new ViT-trunk backward kernels
//! (rope2d sign=-1, ln_head_dx/dgb, scale_chan_dg): analytic grads vs central
//! finite differences of the forward kernels — all smooth ops, so plain
//! finite differences are exact here. CPU backend (no GPU needed).

use gpu_core::{f, DeviceBuffer, Gpu};

const PIPES: &[(&str, &str)] = &[
    ("rope2d", kernels::ROPE2D),
    ("ln_head", kernels::LN_HEAD),
    ("ln_head_dx", kernels::LN_HEAD_DX),
    ("ln_head_dgb", kernels::LN_HEAD_DGB),
    ("scale_chan", kernels::SCALE_CHAN),
    ("scale_chan_dg", kernels::SCALE_CHAN_DG),
];
const ROPE2D: usize = 0;
const LN_HEAD: usize = 1;
const LN_HEAD_DX: usize = 2;
const LN_HEAD_DGB: usize = 3;
const SCALE_CHAN: usize = 4;
const SCALE_CHAN_DG: usize = 5;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn buf(g: &Gpu, d: &[f32]) -> DeviceBuffer {
    g.storage_init("t", d)
}

/// dL/dx via central differences of a scalar loss L = <fwd(x), w>.
fn numeric_grad(fwd: &dyn Fn(&[f32]) -> Vec<f32>, x: &[f32], w: &[f32], i: usize) -> f32 {
    let eps = 1e-3;
    let mut xp = x.to_vec();
    xp[i] += eps;
    let mut xm = x.to_vec();
    xm[i] -= eps;
    let lp: f64 = fwd(&xp).iter().zip(w).map(|(a, b)| (a * b) as f64).sum();
    let lm: f64 = fwd(&xm).iter().zip(w).map(|(a, b)| (a * b) as f64).sum();
    ((lp - lm) / (2.0 * eps as f64)) as f32
}

fn assert_close(name: &str, a: f32, n: f32) {
    let denom = a.abs().max(n.abs()).max(1e-3);
    assert!((a - n).abs() / denom < 2e-2, "{name}: analytic {a} vs numeric {n}");
}

#[test]
fn rope2d_bwd_is_exact_inverse_transpose() {
    // rope2d is orthogonal per pair; its VJP is the sign=-1 rotation applied
    // to the upstream grad. Check dL/dx numerically.
    let g = Gpu::new_cpu(PIPES);
    let (rows, heads, half, stride, off, tmod) = (6u32, 2u32, 4u32, 24u32, 8u32, 3u32);
    let n = (rows * stride) as usize;
    let mut r = Lcg(7);
    let x = r.vec(n);
    let cos = r.vec((tmod * half) as usize).iter().map(|v| v.cos()).collect::<Vec<_>>();
    let sin: Vec<f32> = cos.iter().map(|c| (1.0 - c * c).max(0.0).sqrt()).collect();
    let w = r.vec(n);

    let run = |input: &[f32], sign: f32| -> Vec<f32> {
        let b = buf(&g, input);
        let cb = buf(&g, &cos);
        let sb = buf(&g, &sin);
        let s = g.step(ROPE2D, &[&b, &cb, &sb], &[rows, heads, half, stride, off, tmod, f(sign)], rows * heads * half);
        g.submit(&[], &[s]);
        g.read(&b, n)
    };
    // analytic: dx = rope2d(sign=-1)(dL/dy) on the rotated region
    let analytic = run(&w, -1.0);
    let fwd = |input: &[f32]| run(input, 1.0);
    for i in [8usize, 9, 12, 15, 33, 56, 100] {
        assert_close(&format!("rope2d dx[{i}]"), analytic[i], numeric_grad(&fwd, &x, &w, i));
    }
}

#[test]
fn ln_head_dx_dgb() {
    let g = Gpu::new_cpu(PIPES);
    let (rows, heads, hd, stride, off) = (5u32, 2u32, 8u32, 24u32, 4u32);
    let eps = 1e-5f32;
    let n = (rows * stride) as usize;
    let mut r = Lcg(11);
    let x = r.vec(n);
    let gamma = r.vec(hd as usize);
    let beta = r.vec(hd as usize);
    let w = r.vec(n);

    let fwd_full = |xi: &[f32], ga: &[f32], be: &[f32]| -> Vec<f32> {
        let b = buf(&g, xi);
        let gb = buf(&g, ga);
        let bb = buf(&g, be);
        let s = g.step(LN_HEAD, &[&b, &gb, &bb], &[rows, heads, hd, stride, off, f(eps)], rows * heads);
        g.submit(&[], &[s]);
        g.read(&b, n)
    };

    // analytic dx
    let (xb, gb, wb) = (buf(&g, &x), buf(&g, &gamma), buf(&g, &w));
    let dx = g.storage_init("dx", &vec![0.0; n]);
    let s = g.step(LN_HEAD_DX, &[&xb, &gb, &wb, &dx], &[rows, heads, hd, stride, off, f(eps)], rows * heads);
    g.submit(&[], &[s]);
    let dx_a = g.read(&dx, n);
    // analytic dgamma/dbeta
    let dgm = g.storage_init("dg", &vec![0.0; hd as usize]);
    let dbt = g.storage_init("db", &vec![0.0; hd as usize]);
    let wb2 = buf(&g, &w);
    let s = g.step(LN_HEAD_DGB, &[&xb, &wb2, &dgm, &dbt], &[rows, heads, hd, stride, off, f(eps)], hd);
    g.submit(&[], &[s]);
    let dg_a = g.read(&dgm, hd as usize);
    let db_a = g.read(&dbt, hd as usize);

    let fwd_x = |xi: &[f32]| fwd_full(xi, &gamma, &beta);
    // grads only exist inside normalized regions; off=4, stride=24, heads*hd=16
    for i in [4usize, 7, 12, 19, 28, 43, 100] {
        let region = i % 24;
        if !(4..20).contains(&region) {
            continue;
        }
        assert_close(&format!("ln_head dx[{i}]"), dx_a[i], numeric_grad(&fwd_x, &x, &w, i));
    }
    for c in 0..hd as usize {
        let fwd_g = |ga: &[f32]| fwd_full(&x, ga, &beta);
        assert_close(&format!("ln_head dgamma[{c}]"), dg_a[c], numeric_grad(&fwd_g, &gamma, &w, c));
        let fwd_b = |be: &[f32]| fwd_full(&x, &gamma, be);
        assert_close(&format!("ln_head dbeta[{c}]"), db_a[c], numeric_grad(&fwd_b, &beta, &w, c));
    }
}

#[test]
fn scale_chan_dg_matches() {
    let g = Gpu::new_cpu(PIPES);
    let (rows, c, inner) = (7usize, 5usize, 3usize);
    let total = rows * c * inner;
    let mut r = Lcg(13);
    let x = r.vec(total);
    let scale = r.vec(c);
    let w = r.vec(total);

    let fwd_s = |sc: &[f32]| -> Vec<f32> {
        let xb = buf(&g, &x);
        let sb = buf(&g, sc);
        let ob = g.storage(total as u64);
        let s = g.step(SCALE_CHAN, &[&xb, &sb, &ob], &[total as u32, c as u32, inner as u32], total as u32);
        g.submit(&[], &[s]);
        g.read(&ob, total)
    };
    let (xb, wb) = (buf(&g, &x), buf(&g, &w));
    let dg = g.storage_init("dg", &vec![0.0; c]);
    let s = g.step(SCALE_CHAN_DG, &[&xb, &wb, &dg], &[total as u32, c as u32, inner as u32], c as u32);
    g.submit(&[], &[s]);
    let dg_a = g.read(&dg, c);
    for ci in 0..c {
        assert_close(&format!("scale_chan dg[{ci}]"), dg_a[ci], numeric_grad(&fwd_s, &scale, &w, ci));
    }
}
