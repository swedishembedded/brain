// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Step-builders for the Nemotron-specific WGSL kernels (GLU for the Conformer
//! convolution module, fused LSTM gates for the RNN-T prediction network), each
//! gradient-checked against central finite differences in the tests below.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Kernel pipeline exposing the Nemotron primitives. Indices are the constants
/// below.
pub fn nemotron_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("glu", kernels::GLU),                       // 0
        ("glu_bwd", kernels::GLU_BWD),               // 1
        ("lstm_gates", kernels::LSTM_GATES),         // 2
        ("lstm_gates_bwd", kernels::LSTM_GATES_BWD), // 3
    ]
}
pub const K_GLU: usize = 0;
pub const K_GLU_BWD: usize = 1;
pub const K_LSTM_GATES: usize = 2;
pub const K_LSTM_GATES_BWD: usize = 3;

/// `out[outer,d,inner] = glu(x[outer,2d,inner], dim=middle)`.
pub fn glu_fwd(g: &Gpu, glu: usize, x: &DeviceBuffer, out: &DeviceBuffer, outer: u32, d: u32, inner: u32, steps: &mut Vec<Step>) {
    steps.push(g.step(glu, &[x, out], &[outer, d, inner], outer * d * inner));
}

/// GLU backward: `dx[outer,2d,inner]` from `dy[outer,d,inner]` and `x`.
pub fn glu_bwd(g: &Gpu, glu_bwd: usize, dy: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, outer: u32, d: u32, inner: u32, steps: &mut Vec<Step>) {
    steps.push(g.step(glu_bwd, &[dy, x, dx], &[outer, d, inner], outer * d * inner));
}

/// LSTM cell gate activation: `(c_out, h_out)` from `pre[rows,4H]`, `c_prev[rows,H]`.
#[allow(clippy::too_many_arguments)]
pub fn lstm_gates_fwd(g: &Gpu, k: usize, pre: &DeviceBuffer, c_prev: &DeviceBuffer, c_out: &DeviceBuffer, h_out: &DeviceBuffer, rows: u32, h: u32, steps: &mut Vec<Step>) {
    steps.push(g.step(k, &[pre, c_prev, c_out, h_out], &[rows, h], rows * h));
}

/// LSTM gate backward: `(d_pre[rows,4H], d_cprev[rows,H])`.
#[allow(clippy::too_many_arguments)]
pub fn lstm_gates_bwd(
    g: &Gpu,
    k: usize,
    dh: &DeviceBuffer,
    dc_next: &DeviceBuffer,
    pre: &DeviceBuffer,
    c_prev: &DeviceBuffer,
    c_out: &DeviceBuffer,
    d_pre: &DeviceBuffer,
    d_cprev: &DeviceBuffer,
    rows: u32,
    h: u32,
    steps: &mut Vec<Step>,
) {
    steps.push(g.step(k, &[dh, dc_next, pre, c_prev, c_out, d_pre, d_cprev], &[rows, h], rows * h));
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    fn maxrel(a: f32, n: f32) -> bool {
        (a - n).abs() <= 3e-3 + 6e-2 * n.abs()
    }

    #[test]
    fn glu_forward_and_backward_match_finite_diff() {
        let g = Gpu::new_cpu(nemotron_pipelines());
        let (outer, d, inner) = (2u32, 3u32, 4u32);
        let nx = (outer * 2 * d * inner) as usize;
        let ny = (outer * d * inner) as usize;
        let mut rng = Rng::new(7);
        let xh: Vec<f32> = (0..nx).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();

        let fwd = |xin: &[f32]| -> Vec<f32> {
            let x = g.storage_init("x", xin);
            let out = g.storage(ny as u64);
            let mut s = Vec::new();
            glu_fwd(&g, K_GLU, &x, &out, outer, d, inner, &mut s);
            g.submit(&[], &s);
            g.read(&out, ny)
        };
        // analytic dx (loss = sum(out) -> dy = 1)
        let x = g.storage_init("x", &xh);
        let dy = g.storage_init("dy", &vec![1.0f32; ny]);
        let dx = g.storage(nx as u64);
        let mut s = Vec::new();
        glu_bwd(&g, K_GLU_BWD, &dy, &x, &dx, outer, d, inner, &mut s);
        g.submit(&[], &s);
        let dxh = g.read(&dx, nx);

        let eps = 1e-3f32;
        for i in 0..nx {
            let (mut xp, mut xm) = (xh.clone(), xh.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let num = (fwd(&xp).iter().sum::<f32>() - fwd(&xm).iter().sum::<f32>()) / (2.0 * eps);
            assert!(maxrel(dxh[i], num), "dx[{i}] analytic {} vs numeric {}", dxh[i], num);
        }
    }

    #[test]
    fn lstm_gates_forward_and_backward_match_finite_diff() {
        let g = Gpu::new_cpu(nemotron_pipelines());
        let (rows, h) = (2u32, 3u32);
        let npre = (rows * 4 * h) as usize;
        let nc = (rows * h) as usize;
        let mut rng = Rng::new(11);
        let pre_h: Vec<f32> = (0..npre).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();
        let cprev_h: Vec<f32> = (0..nc).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();

        // loss = sum(h_out) + sum(c_out)
        let loss = |pre_in: &[f32], c_in: &[f32]| -> f32 {
            let pre = g.storage_init("pre", pre_in);
            let cp = g.storage_init("cp", c_in);
            let co = g.storage(nc as u64);
            let ho = g.storage(nc as u64);
            let mut s = Vec::new();
            lstm_gates_fwd(&g, K_LSTM_GATES, &pre, &cp, &co, &ho, rows, h, &mut s);
            g.submit(&[], &s);
            g.read(&co, nc).iter().sum::<f32>() + g.read(&ho, nc).iter().sum::<f32>()
        };

        // analytic grads (dh = 1, dc_next = 1 matching the loss)
        let pre = g.storage_init("pre", &pre_h);
        let cp = g.storage_init("cp", &cprev_h);
        let co = g.storage(nc as u64);
        let ho = g.storage(nc as u64);
        let mut sf = Vec::new();
        lstm_gates_fwd(&g, K_LSTM_GATES, &pre, &cp, &co, &ho, rows, h, &mut sf);
        g.submit(&[], &sf);
        let dh = g.storage_init("dh", &vec![1.0f32; nc]);
        let dcn = g.storage_init("dcn", &vec![1.0f32; nc]);
        let d_pre = g.storage(npre as u64);
        let d_cprev = g.storage(nc as u64);
        let mut sb = Vec::new();
        lstm_gates_bwd(&g, K_LSTM_GATES_BWD, &dh, &dcn, &pre, &cp, &co, &d_pre, &d_cprev, rows, h, &mut sb);
        g.submit(&[], &sb);
        let d_pre_h = g.read(&d_pre, npre);
        let d_cprev_h = g.read(&d_cprev, nc);

        let eps = 1e-3f32;
        for i in 0..npre {
            let (mut pp, mut pm) = (pre_h.clone(), pre_h.clone());
            pp[i] += eps;
            pm[i] -= eps;
            let num = (loss(&pp, &cprev_h) - loss(&pm, &cprev_h)) / (2.0 * eps);
            assert!(maxrel(d_pre_h[i], num), "d_pre[{i}] analytic {} vs numeric {}", d_pre_h[i], num);
        }
        for i in 0..nc {
            let (mut cp2, mut cm2) = (cprev_h.clone(), cprev_h.clone());
            cp2[i] += eps;
            cm2[i] -= eps;
            let num = (loss(&pre_h, &cp2) - loss(&pre_h, &cm2)) / (2.0 * eps);
            assert!(maxrel(d_cprev_h[i], num), "d_cprev[{i}] analytic {} vs numeric {}", d_cprev_h[i], num);
        }
    }
}
