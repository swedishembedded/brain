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
        ("rel_shift", kernels::REL_SHIFT),           // 4
        ("rel_shift_bwd", kernels::REL_SHIFT_BWD),   // 5
    ]
}
pub const K_GLU: usize = 0;
pub const K_GLU_BWD: usize = 1;
pub const K_LSTM_GATES: usize = 2;
pub const K_LSTM_GATES_BWD: usize = 3;
pub const K_REL_SHIFT: usize = 4;
pub const K_REL_SHIFT_BWD: usize = 5;

/// Transformer-XL relative shift: `out[rows,q,p]` from `x[rows,q,p]`.
pub fn rel_shift_fwd(g: &Gpu, k: usize, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, q: u32, p: u32, steps: &mut Vec<Step>) {
    steps.push(g.step(k, &[x, out], &[rows, q, p], rows * q * p));
}

/// rel_shift backward (caller must zero `dx` first): scatters `dy` into `dx`.
pub fn rel_shift_bwd(g: &Gpu, k: usize, dy: &DeviceBuffer, dx: &DeviceBuffer, rows: u32, q: u32, p: u32, steps: &mut Vec<Step>) {
    steps.push(g.step(k, &[dy, dx], &[rows, q, p], rows * q * p));
}

/// CPU oracle for `rel_shift` — the exact torch pad → reshape → drop-row op.
pub fn rel_shift_ref(x: &[f32], rows: usize, q: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * q * p];
    for r in 0..rows {
        // padded [q, p+1] flattened; then viewed as [p+1, q], drop first row (q elems).
        let mut xp = vec![0.0f32; q * (p + 1)];
        for i in 0..q {
            for k in 1..=p {
                xp[i * (p + 1) + k] = x[r * q * p + i * p + (k - 1)];
            }
        }
        for idx2 in 0..q * p {
            out[r * q * p + idx2] = xp[q + idx2];
        }
    }
    out
}

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
    use gpu_core::f;

    /// Pipeline for the on-device FF fwd/bwd gradcheck (proves the device training
    /// path with the real backward kernels).
    fn ff_bwd_pipelines() -> &'static [(&'static str, &'static str)] {
        &[
            ("matmul", kernels::MATMUL),       // 0
            ("silu", kernels::SILU),           // 1
            ("matmul_dx", kernels::MATMUL_DX), // 2
            ("silu_bwd", kernels::SILU_BWD),   // 3
            ("matmul_dw", kernels::MATMUL_DW), // 4
        ]
    }

    /// On-device Conformer feed-forward (Linear→SiLU→Linear) forward AND backward,
    /// gradchecked against central finite differences of the device forward — the
    /// device training path using the real gradient kernels (matmul_dx/dw, silu_bwd).
    #[test]
    fn device_ff_backward_matches_finite_diff() {
        let g = gpu_core::testgpu::dev(ff_bwd_pipelines());
        let (t, c, ffn) = (4u32, 6u32, 10u32);
        let mut rng = Rng::new(13);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.6).collect::<Vec<f32>>();
        let (xh, w1h, w2h) = (r((t * c) as usize), r((ffn * c) as usize), r((c * ffn) as usize));
        let w1 = g.storage_init("w1", &w1h);
        let w2 = g.storage_init("w2", &w2h);

        // forward: h1 = x·w1ᵀ [t,ffn]; s = silu(h1); out = s·w2ᵀ [t,c]
        let fwd = |xin: &[f32]| -> Vec<f32> {
            let x = g.storage_init("x", xin);
            let h1 = g.storage((t * ffn) as u64);
            let s = g.storage((t * ffn) as u64);
            let out = g.storage((t * c) as u64);
            g.submit(&[], &[
                g.step(0, &[&x, &w1, &h1], &[t, c, ffn], t * ffn),
                g.step(1, &[&h1, &s], &[t * ffn], t * ffn),
                g.step(0, &[&s, &w2, &out], &[t, ffn, c], t * c),
            ]);
            g.read(&out, (t * c) as usize)
        };

        // analytic d_x (loss = Σ out → d_out = 1) via device backward kernels
        let x = g.storage_init("x", &xh);
        let h1 = g.storage((t * ffn) as u64);
        let s = g.storage((t * ffn) as u64);
        let out = g.storage((t * c) as u64);
        g.submit(&[], &[
            g.step(0, &[&x, &w1, &h1], &[t, c, ffn], t * ffn),
            g.step(1, &[&h1, &s], &[t * ffn], t * ffn),
            g.step(0, &[&s, &w2, &out], &[t, ffn, c], t * c),
        ]);
        let d_out = g.storage_init("dout", &vec![1.0f32; (t * c) as usize]);
        let d_s = g.storage((t * ffn) as u64);
        let d_h1 = g.storage((t * ffn) as u64);
        let d_x = g.storage((t * c) as u64);
        g.submit(&[], &[
            // d_s = d_out · w2   (matmul_dx: dx[m,k]=dy[m,n]·w[n,k]; m=t,n=c,k=ffn)
            g.step(2, &[&d_out, &w2, &d_s], &[t, ffn, c, 0], t * ffn),
            // d_h1 = silu'(h1) ⊙ d_s
            g.step(3, &[&h1, &d_s, &d_h1], &[t * ffn], t * ffn),
            // d_x = d_h1 · w1   (m=t,n=ffn,k=c)
            g.step(2, &[&d_h1, &w1, &d_x], &[t, c, ffn, 0], t * c),
        ]);
        let _ = f(0.0);
        let dxh = g.read(&d_x, (t * c) as usize);

        let eps = 1e-3f32;
        for i in 0..(t * c) as usize {
            let (mut xp, mut xm) = (xh.clone(), xh.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let num = (fwd(&xp).iter().sum::<f32>() - fwd(&xm).iter().sum::<f32>()) / (2.0 * eps);
            assert!((dxh[i] - num).abs() <= 3e-3 + 6e-2 * num.abs(), "d_x[{i}] {} vs {}", dxh[i], num);
        }
    }

    fn maxrel(a: f32, n: f32) -> bool {
        (a - n).abs() <= 3e-3 + 6e-2 * n.abs()
    }

    #[test]
    fn glu_forward_and_backward_match_finite_diff() {
        let g = gpu_core::testgpu::dev(nemotron_pipelines());
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
    fn rel_shift_matches_oracle_and_backward_is_transpose() {
        let g = gpu_core::testgpu::dev(nemotron_pipelines());
        let (rows, q, p) = (2u32, 4u32, 7u32); // p = 2*L-1 style
        let n = (rows * q * p) as usize;
        let mut rng = Rng::new(5);
        let xh: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

        // forward vs oracle
        let x = g.storage_init("x", &xh);
        let out = g.storage(n as u64);
        let mut s = Vec::new();
        rel_shift_fwd(&g, K_REL_SHIFT, &x, &out, rows, q, p, &mut s);
        g.submit(&[], &s);
        let outh = g.read(&out, n);
        let refh = rel_shift_ref(&xh, rows as usize, q as usize, p as usize);
        let d = outh.iter().zip(&refh).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(d < 1e-6, "rel_shift forward vs oracle maxdiff {d}");

        // backward = transpose: dx from dy (loss = sum(w·out)) must match FD
        let mut rng2 = Rng::new(9);
        let dyh: Vec<f32> = (0..n).map(|_| rng2.next_f32() - 0.5).collect();
        let dy = g.storage_init("dy", &dyh);
        let dx = g.storage_init("dx", &vec![0.0f32; n]);
        let mut sb = Vec::new();
        rel_shift_bwd(&g, K_REL_SHIFT_BWD, &dy, &dx, rows, q, p, &mut sb);
        g.submit(&[], &sb);
        let dxh = g.read(&dx, n);
        // numeric: d/dx_i sum(dy·rel_shift(x)) via the oracle
        let eps = 1e-3f32;
        for i in 0..n {
            let (mut xp, mut xm) = (xh.clone(), xh.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let lp: f32 = rel_shift_ref(&xp, rows as usize, q as usize, p as usize).iter().zip(&dyh).map(|(a, b)| a * b).sum();
            let lm: f32 = rel_shift_ref(&xm, rows as usize, q as usize, p as usize).iter().zip(&dyh).map(|(a, b)| a * b).sum();
            let num = (lp - lm) / (2.0 * eps);
            assert!((dxh[i] - num).abs() <= 1e-3 + 1e-2 * num.abs(), "dx[{i}] {} vs {}", dxh[i], num);
        }
    }

    #[test]
    fn lstm_gates_forward_and_backward_match_finite_diff() {
        let g = gpu_core::testgpu::dev(nemotron_pipelines());
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
