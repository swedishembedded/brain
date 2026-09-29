// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The interleaved (adjacent-pair) RoPE kernels rotate at the caller's base
//! `theta`, not a compiled-in 10000.
//!
//! `rope_train`, `rope_train_bwd` and `rope_sub` are GLM-DSA's rotations
//! (forward, backward, and the indexer's partial-head one). GLM-5 declares
//! `rope_theta: 8e6`; at the positions and channels below, a kernel still
//! rotating at 10000 is off by whole radians, not rounding.

const PIPELINES: &[(&str, &str)] = &[("rope_train", kernels::ROPE_TRAIN), ("rope_train_bwd", kernels::ROPE_TRAIN_BWD), ("rope_sub", kernels::ROPE_SUB)];
const ROPE_TRAIN: usize = 0;
const ROPE_TRAIN_BWD: usize = 1;
const ROPE_SUB: usize = 2;

/// Rotate pairs `(2j, 2j+1)` of the first `rot` channels of every head by
/// `sign * pos * theta^(-2j/rot)`, `pos = row % tcols`.
#[allow(clippy::too_many_arguments)]
fn host_rotate(buf: &mut [f32], rows: usize, heads: usize, head_dim: usize, rot: usize, stride: usize, off: usize, tcols: usize, theta: f32, sign: f32) {
    for row in 0..rows {
        let pos = (row % tcols) as f64;
        for h in 0..heads {
            for j in 0..rot / 2 {
                let angle = sign as f64 * pos * (theta as f64).powf(-((2 * j) as f64) / rot as f64);
                let (s, c) = angle.sin_cos();
                let b = row * stride + off + h * head_dim + 2 * j;
                let (e, o) = (buf[b] as f64, buf[b + 1] as f64);
                buf[b] = (e * c - o * s) as f32;
                buf[b + 1] = (e * s + o * c) as f32;
            }
        }
    }
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn every_interleaved_rope_kernel_rotates_at_the_callers_theta() {
    let theta = 8.0e6f32;
    let (rows, heads, head_dim, tcols) = (12usize, 2usize, 16usize, 6usize);
    let (stride, off) = (3 * heads * head_dim, heads * head_dim); // the k region of a fused qkv row
    let x: Vec<f32> = (0..rows * stride).map(|i| (i as f32 * 0.37 + 0.1).sin()).collect();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let run = |kind: usize, params: &[u32], threads: u32| {
        let b = gpu.storage_init("x", &x);
        gpu.submit(&[], &[gpu.step(kind, &[&b], params, threads)]);
        gpu.read(&b, x.len())
    };
    let t = theta.to_bits();
    let (r, h, hd, tc, s, o) = (rows as u32, heads as u32, head_dim as u32, tcols as u32, stride as u32, off as u32);

    for (kind, sign, name) in [(ROPE_TRAIN, 1.0, "rope_train"), (ROPE_TRAIN_BWD, -1.0, "rope_train_bwd")] {
        let mut want = x.clone();
        host_rotate(&mut want, rows, heads, head_dim, head_dim, stride, off, tcols, theta, sign);
        let got = run(kind, &[r, h, hd, s, o, tc, t], (rows * heads * head_dim / 2) as u32);
        let err = worst(&got, &want);
        assert!(err < 1e-5, "{name}: max abs error {err:e} at theta {theta}");
    }

    // The indexer layout: rotate the first `rope_dim` channels of each
    // `head_dim`-wide head, rows packed at `heads * head_dim`.
    let rope_dim = 8usize;
    let packed = heads * head_dim;
    let mut want = x.clone();
    host_rotate(&mut want, rows, heads, head_dim, rope_dim, packed, 0, tcols, theta, 1.0);
    let got = run(ROPE_SUB, &[r, h, hd, rope_dim as u32, packed as u32, tc, t], (rows * heads * rope_dim / 2) as u32);
    let err = worst(&got, &want);
    assert!(err < 1e-5, "rope_sub: max abs error {err:e} at theta {theta}");
}
