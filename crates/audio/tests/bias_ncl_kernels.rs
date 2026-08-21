// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for `add_chan_inplace` (NCL per-channel bias forward)
//! paired with the new `bias_grad_ncl` (its backward): the bias gradient
//! must match central finite differences of `L = <y, dy>` w.r.t. `bias`,
//! run on the CPU (Cranelift JIT) backend so it needs no GPU.

use data::rng::Lcg;
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

const PIPES: &[(&str, &str)] = &[("add_chan_inplace", kernels::ADD_CHAN_INPLACE), ("bias_grad_ncl", kernels::BIAS_GRAD_NCL)];
const FWD: usize = 0;
const BWD: usize = 1;

fn buf(g: &Gpu, data: &[f32]) -> DeviceBuffer {
    let b = g.buffer("b", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    g.write(&b, bytemuck::cast_slice(data));
    b
}

fn fwd_ref(x: &[f32], bias: &[f32], rows: usize, c: usize, inner: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for row in 0..rows {
        for (ch, &b) in bias.iter().enumerate().take(c) {
            let base = row * c * inner + ch * inner;
            for l in 0..inner {
                y[base + l] += b;
            }
        }
    }
    y
}

#[test]
fn bias_grad_ncl_matches_finite_differences() {
    let g = Gpu::new_cpu(PIPES);
    let (rows, c, inner) = (2usize, 3usize, 5usize);
    let n = rows * c * inner;
    let mut r = Lcg::new(7);
    let x = r.vec(n);
    let bias = r.vec(c);
    let dy = r.vec(n);

    let xb = buf(&g, &x);
    let bb = buf(&g, &bias);
    g.submit(&[], &[g.step(FWD, &[&xb, &bb], &[n as u32, c as u32, inner as u32], n as u32)]);
    let y_gpu = g.read(&xb, n);
    let y_ref = fwd_ref(&x, &bias, rows, c, inner);
    let max_abs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    assert!(max_abs(&y_gpu, &y_ref) < 1e-5, "fwd mismatch: {}", max_abs(&y_gpu, &y_ref));

    let dyb = buf(&g, &dy);
    let dbb = buf(&g, &vec![0.0f32; c]);
    g.submit(&[], &[g.step(BWD, &[&dyb, &dbb], &[rows as u32, c as u32, inner as u32], c as u32)]);
    let db_a = g.read(&dbb, c);

    let loss = |bias: &[f32]| -> f32 { fwd_ref(&x, bias, rows, c, inner).iter().zip(&dy).map(|(a, b)| a * b).sum() };
    let eps = 1e-3f32;
    for i in 0..c {
        let mut p = bias.clone();
        p[i] = bias[i] + eps;
        let lp = loss(&p);
        p[i] = bias[i] - eps;
        let lm = loss(&p);
        let num = (lp - lm) / (2.0 * eps);
        assert!((num - db_a[i]).abs() < 1e-3, "dbias[{i}] num={num} ana={}", db_a[i]);
    }
}
