// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the new 1D-conv WGSL kernels, run on the CPU (Cranelift
//! JIT) backend so it needs no GPU. For each shape we check:
//!   1. forward parity: GPU `conv1d`/`convtr1d` == the CPU reference oracle;
//!   2. input grad: `conv1d_dx`/`convtr1d_dx` == central finite differences of
//!      a scalar loss `L = <y, dy>` w.r.t. x;
//!   3. weight grad: `conv1d_dw`/`convtr1d_dw` == finite differences w.r.t. w.

use data::rng::Lcg;
use audio::conv::{conv1d_ref, convtr1d_ref, Conv1d};
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

const PIPES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("convtr1d", kernels::CONVTR1D),
    ("convtr1d_dx", kernels::CONVTR1D_DX),
    ("convtr1d_dw", kernels::CONVTR1D_DW),
];
const C_FWD: usize = 0;
const C_DX: usize = 1;
const C_DW: usize = 2;
const T_FWD: usize = 3;
const T_DX: usize = 4;
const T_DW: usize = 5;
fn buf(g: &Gpu, data: &[f32]) -> DeviceBuffer {
    let b = g.buffer("b", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    g.write(&b, bytemuck::cast_slice(data));
    b
}
fn zeros(g: &Gpu, n: usize) -> DeviceBuffer {
    let b = g.buffer("z", (n * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    g.write(&b, &vec![0u32; n]);
    b
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Generic check for one conv (transposed = whether to use the transposed
/// kernels / weight layout / reference).
fn check(c: Conv1d, transposed: bool, seed: u64) {
    let g = Gpu::new_cpu(PIPES);
    let mut r = Lcg::new(seed);
    let xn = (c.n * c.cin * c.l) as usize;
    let yn = (c.n * c.cout * c.lo) as usize;
    let wn = if transposed { c.weight_numel_transposed() } else { c.weight_numel() };
    let x = r.vec(xn);
    let w = r.vec(wn);
    let dy = r.vec(yn); // random cotangent

    let reff = |x: &[f32], w: &[f32]| if transposed { convtr1d_ref(&c, x, w) } else { conv1d_ref(&c, x, w) };
    let (kf, kdx, kdw) = if transposed { (T_FWD, T_DX, T_DW) } else { (C_FWD, C_DX, C_DW) };

    // 1. forward parity vs CPU reference.
    let xb = buf(&g, &x);
    let wb = buf(&g, &w);
    let yb = zeros(&g, yn);
    g.submit(&[], &[g.step(kf, &[&xb, &wb, &yb], &params(&c), c.n * c.cout * c.lo)]);
    let y_gpu = g.read(&yb, yn);
    let y_ref = reff(&x, &w);
    assert!(max_abs(&y_gpu, &y_ref) < 1e-3, "fwd mismatch (transposed={transposed}): {}", max_abs(&y_gpu, &y_ref));

    // 2 & 3. analytic grads from the kernels.
    let dyb = buf(&g, &dy);
    let dxb = zeros(&g, xn);
    let dwb = zeros(&g, wn);
    g.submit(&[], &[g.step(kdx, &[&dyb, &wb, &dxb], &params(&c), c.n * c.cin * c.l)]);
    let dw_threads = wn as u32;
    g.submit(&[&dwb], &[g.step(kdw, &[&dyb, &xb, &dwb], &params(&c), dw_threads)]);
    let dx_a = g.read(&dxb, xn);
    let dw_a = g.read(&dwb, wn);

    // finite differences of L(x,w) = <reff(x,w), dy>.
    let loss = |x: &[f32], w: &[f32]| -> f32 { reff(x, w).iter().zip(&dy).map(|(a, b)| a * b).sum() };
    let eps = 1e-3f32;
    let fd = |base: &[f32], i: usize, f: &dyn Fn(&[f32]) -> f32| {
        let mut p = base.to_vec();
        p[i] = base[i] + eps;
        let lp = f(&p);
        p[i] = base[i] - eps;
        let lm = f(&p);
        (lp - lm) / (2.0 * eps)
    };
    for i in (0..xn).step_by((xn / 17).max(1)) {
        let num = fd(&x, i, &|xx| loss(xx, &w));
        assert!((num - dx_a[i]).abs() < 1e-2 + 1e-2 * num.abs().max(dx_a[i].abs()), "dx[{i}] num={num} ana={} (transposed={transposed})", dx_a[i]);
    }
    for i in (0..wn).step_by((wn / 13).max(1)) {
        let num = fd(&w, i, &|ww| loss(&x, ww));
        assert!((num - dw_a[i]).abs() < 1e-2 + 1e-2 * num.abs().max(dw_a[i].abs()), "dw[{i}] num={num} ana={} (transposed={transposed})", dw_a[i]);
    }
}

fn params(c: &Conv1d) -> [u32; 10] {
    [c.n, c.cin, c.l, c.cout, c.k, c.stride, c.pad, c.dilation, c.groups, c.lo]
}

#[test]
fn conv1d_plain() {
    let l = 9;
    let c = Conv1d { n: 2, cin: 3, l, cout: 4, k: 3, stride: 1, pad: 0, dilation: 1, groups: 1, lo: Conv1d::out_len(l, 3, 1, 0, 0, 1) };
    check(c, false, 1);
}
#[test]
fn conv1d_causal() {
    let (l, k) = (10u32, 4u32);
    let c = Conv1d { n: 1, cin: 2, l, cout: 5, k, stride: 1, pad: k - 1, dilation: 1, groups: 1, lo: l };
    check(c, false, 2);
}
#[test]
fn conv1d_dilated_causal() {
    let (l, k, dil) = (16u32, 3u32, 2u32);
    let c = Conv1d { n: 1, cin: 4, l, cout: 4, k, stride: 1, pad: dil * (k - 1), dilation: dil, groups: 1, lo: l };
    check(c, false, 3);
}
#[test]
fn conv1d_grouped() {
    let l = 8;
    let c = Conv1d { n: 2, cin: 6, l, cout: 6, k: 3, stride: 1, pad: 1, dilation: 1, groups: 3, lo: Conv1d::out_len(l, 3, 1, 1, 1, 1) };
    check(c, false, 4);
}
#[test]
fn conv1d_strided() {
    let l = 12;
    let c = Conv1d { n: 1, cin: 2, l, cout: 3, k: 4, stride: 2, pad: 1, dilation: 1, groups: 1, lo: Conv1d::out_len(l, 4, 2, 1, 1, 1) };
    check(c, false, 5);
}
#[test]
fn convtr1d_upsample() {
    let (l, k, s) = (5u32, 4u32, 2u32);
    let c = Conv1d { n: 2, cin: 3, l, cout: 4, k, stride: s, pad: 1, dilation: 1, groups: 1, lo: Conv1d::out_len_transposed(l, k, s, 1, 0, 1) };
    check(c, true, 6);
}
#[test]
fn convtr1d_grouped() {
    let (l, k, s) = (6u32, 4u32, 2u32);
    let c = Conv1d { n: 1, cin: 4, l, cout: 4, k, stride: s, pad: 1, dilation: 1, groups: 2, lo: Conv1d::out_len_transposed(l, k, s, 1, 0, 1) };
    check(c, true, 7);
}
