// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Batched conv2d fast-path parity vs a scalar reference — written for a real
//! bug: the WorldMirror S=3 patch conv (N=3, Cin=3, 56x56, Cout=64, K=14 s14)
//! produced garbage for every frame after the first on the CPU backend.

use gpu_core::Gpu;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
}

// Conv dims + the two input tensors: the reference this test exists to
// compare the JIT against, so it takes the shape explicitly.
#[allow(clippy::too_many_arguments)]
fn scalar_conv(
    n: usize, cin: usize, h: usize, w: usize, cout: usize, k: usize, stride: usize, pad: usize,
    x: &[f32], wt: &[f32],
) -> Vec<f32> {
    let ho = (h + 2 * pad - k) / stride + 1;
    let wo = (w + 2 * pad - k) / stride + 1;
    let mut y = vec![0.0f32; n * cout * ho * wo];
    for ni in 0..n {
        for co in 0..cout {
            for oy in 0..ho {
                for ox in 0..wo {
                    let mut acc = 0.0f32;
                    for ci in 0..cin {
                        for ky in 0..k {
                            let iy = (oy * stride + ky).wrapping_sub(pad);
                            if iy >= h {
                                continue;
                            }
                            for kx in 0..k {
                                let ix = (ox * stride + kx).wrapping_sub(pad);
                                if ix >= w {
                                    continue;
                                }
                                acc += x[((ni * cin + ci) * h + iy) * w + ix]
                                    * wt[((co * cin + ci) * k + ky) * k + kx];
                            }
                        }
                    }
                    y[((ni * cout + co) * ho + oy) * wo + ox] = acc;
                }
            }
        }
    }
    y
}

#[test]
fn batched_conv_matches_scalar() {
    let gpu = Gpu::new_cpu(&[("conv2d", kernels::CONV2D)]);
    let mut seed = 0xABCD;
    for (n, cin, h, w, cout, k, stride, pad) in [
        (3usize, 3usize, 56usize, 56usize, 64usize, 14usize, 14usize, 0usize),
        (2, 4, 9, 9, 8, 3, 1, 1),
        (4, 1, 8, 8, 2, 2, 2, 0),
        (1, 8, 56, 56, 32, 3, 1, 1),
        (1, 32, 56, 56, 3, 1, 1, 0),
        (1, 3, 56, 56, 8, 7, 1, 3),
    ] {
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let x: Vec<f32> = (0..n * cin * h * w).map(|_| lcg(&mut seed)).collect();
        let wt: Vec<f32> = (0..cout * cin * k * k).map(|_| lcg(&mut seed)).collect();
        let want = scalar_conv(n, cin, h, w, cout, k, stride, pad, &x, &wt);
        let xb = gpu.storage_init("x", &x);
        let wb = gpu.storage_init("w", &wt);
        let yb = gpu.storage((n * cout * ho * wo) as u64);
        let params = [
            n as u32, cin as u32, h as u32, w as u32, cout as u32, k as u32, stride as u32,
            pad as u32, ho as u32, wo as u32,
        ];
        let steps = vec![gpu.step(0, &[&xb, &wb, &yb], &params, (n * cout * ho * wo) as u32)];
        gpu.submit(&[], &steps);
        let got = gpu.read(&yb, n * cout * ho * wo);
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        assert!(
            worst < 1e-4,
            "n={n} cin={cin} {h}x{w} cout={cout} k={k} s{stride} p{pad}: worst {worst} at {worst_i} (batch {})",
            worst_i / (cout * ho * wo)
        );
    }
}
