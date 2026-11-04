// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Depthwise 3D conv (PEG) forward vs a host reference, plus finite-difference
//! checks of the input and weight gradients. Tiny [N,C,T,H,W] with K=3, pad 1.
use gpu_core::Gpu;

const K: [(&str, &str); 3] = [
    ("dwconv3d", kernels::DWCONV3D),
    ("dwconv3d_dx", kernels::DWCONV3D_DX),
    ("dwconv3d_dw", kernels::DWCONV3D_DW),
];

fn rnd(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

const N: usize = 1;
const C: usize = 3;
const T: usize = 3;
const H: usize = 4;
const W: usize = 4;
const KS: usize = 3;
const P: usize = 1;

fn host_fwd(x: &[f32], wt: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N * C * T * H * W];
    let idx = |c: usize, t: usize, h: usize, w: usize| ((c * T + t) * H + h) * W + w;
    for c in 0..C {
        for t in 0..T {
            for h in 0..H {
                for w in 0..W {
                    let mut acc = bias[c];
                    for kt in 0..KS {
                        for kh in 0..KS {
                            for kw in 0..KS {
                                let it = t as isize + kt as isize - P as isize;
                                let ih = h as isize + kh as isize - P as isize;
                                let iw = w as isize + kw as isize - P as isize;
                                if it >= 0 && (it as usize) < T && ih >= 0 && (ih as usize) < H && iw >= 0 && (iw as usize) < W {
                                    let wti = ((c * KS + kt) * KS + kh) * KS + kw;
                                    acc += x[idx(c, it as usize, ih as usize, iw as usize)] * wt[wti];
                                }
                            }
                        }
                    }
                    y[idx(c, t, h, w)] = acc;
                }
            }
        }
    }
    y
}

#[test]
fn dwconv3d_forward_matches_host_reference() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let gpu = Gpu::new_cpu(&K);
    let x = rnd(1, N * C * T * H * W);
    let wt = rnd(2, C * KS * KS * KS);
    let bias = rnd(3, C);
    let xb = gpu.storage_init("x", &x);
    let wtb = gpu.storage_init("wt", &wt);
    let bb = gpu.storage_init("b", &bias);
    let yb = gpu.storage((N * C * T * H * W) as u64);
    let params = [N as u32, C as u32, T as u32, H as u32, W as u32, KS as u32, P as u32, P as u32];
    gpu.submit(&[], &[gpu.step(0, &[&xb, &wtb, &bb, &yb], &params, (N * C * T * H * W) as u32)]);
    let y = gpu.read(&yb, N * C * T * H * W);
    let want = host_fwd(&x, &wt, &bias);
    let max = y.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-4, "dwconv3d fwd max abs {max}");
}

#[test]
fn dwconv3d_backward_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let gpu = Gpu::new_cpu(&K);
    let x = rnd(4, N * C * T * H * W);
    let wt = rnd(5, C * KS * KS * KS);
    let bias = vec![0.0f32; C];
    let dy = rnd(6, N * C * T * H * W); // upstream grad
    let params = [N as u32, C as u32, T as u32, H as u32, W as u32, KS as u32, P as u32, P as u32];
    let wf = |b: &gpu_core::DeviceBuffer, d: &[f32]| gpu.write(b, &d.iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    let xb = gpu.storage_init("x", &x);
    let wtb = gpu.storage_init("wt", &wt);
    let bb = gpu.storage_init("b", &bias);
    let yb = gpu.storage((N * C * T * H * W) as u64);
    let dyb = gpu.storage_init("dy", &dy);
    let dxb = gpu.storage((N * C * T * H * W) as u64);
    let dwb = gpu.storage((C * KS * KS * KS) as u64);

    let fwd = |gpu: &Gpu| gpu.submit(&[], &[gpu.step(0, &[&xb, &wtb, &bb, &yb], &params, (N * C * T * H * W) as u32)]);
    fwd(&gpu);
    gpu.submit(&[], &[
        gpu.step(1, &[&dyb, &wtb, &dxb], &params, (N * C * T * H * W) as u32),
        gpu.step(2, &[&dyb, &xb, &dwb], &params, (C * KS * KS * KS) as u32),
    ]);
    let dxg = gpu.read(&dxb, N * C * T * H * W);
    let dwg = gpu.read(&dwb, C * KS * KS * KS);

    let loss = |gpu: &Gpu| { fwd(gpu); gpu.read(&yb, N * C * T * H * W).iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>() };
    let eps = 1e-3f32;

    // dx directional
    let dir = rnd(7, N * C * T * H * W);
    let analytic_x: f32 = dxg.iter().zip(&dir).map(|(a, b)| a * b).sum();
    let xp: Vec<f32> = x.iter().zip(&dir).map(|(v, d)| v + eps * d).collect(); wf(&xb, &xp); let lp = loss(&gpu);
    let xm: Vec<f32> = x.iter().zip(&dir).map(|(v, d)| v - eps * d).collect(); wf(&xb, &xm); let lm = loss(&gpu);
    wf(&xb, &x);
    let numeric_x = (lp - lm) / (2.0 * eps);
    assert!((analytic_x - numeric_x).abs() < 4e-3 + 8e-2 * analytic_x.abs().max(numeric_x.abs()), "dx: {analytic_x} vs {numeric_x}");

    // dw directional
    let dirw = rnd(8, C * KS * KS * KS);
    let analytic_w: f32 = dwg.iter().zip(&dirw).map(|(a, b)| a * b).sum();
    let wp: Vec<f32> = wt.iter().zip(&dirw).map(|(v, d)| v + eps * d).collect(); wf(&wtb, &wp); let lpw = loss(&gpu);
    let wm: Vec<f32> = wt.iter().zip(&dirw).map(|(v, d)| v - eps * d).collect(); wf(&wtb, &wm); let lmw = loss(&gpu);
    wf(&wtb, &wt);
    let numeric_w = (lpw - lmw) / (2.0 * eps);
    assert!((analytic_w - numeric_w).abs() < 4e-3 + 8e-2 * analytic_w.abs().max(numeric_w.abs()), "dw: {analytic_w} vs {numeric_w}");
}
