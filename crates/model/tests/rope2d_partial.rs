// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::{rope2d_partial_fwd, rope2d_partial_bwd}` vs. a host-computed
//! oracle. Validates the Qwen3.5 `partial_rotary_factor` case that plain
//! `rope2d` cannot express: a head wider than the rotated sub-space, so the
//! per-head buffer stride (`head_dim`) must differ from the table width
//! (`half = rot_dim/2`). Two things must hold: (1) the rotated prefix matches
//! `rope2d`'s own half-split rotation math exactly, and (2) the untouched tail
//! of each head is bit-identical to the input (never addressed by the
//! kernel).

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{rope2d_partial_bwd, rope2d_partial_fwd};

const PIPES: &[(&str, &str)] = &[("rope2d_partial", kernels::ROPE2D_PARTIAL)];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Host oracle: rotate only the first `2*half` channels of each `head_dim`
/// head, in place, at `sign` (1 = forward, -1 = exact inverse).
fn host_rope2d_partial(buf: &mut [f32], cos: &[f32], sin: &[f32], rows: usize, heads: usize, half: usize, row_stride: usize, off: usize, head_dim: usize, sign: f32) {
    for row in 0..rows {
        for h in 0..heads {
            let base = row * row_stride + off + h * head_dim;
            for d in 0..half {
                let c = cos[row * half + d];
                let s = sin[row * half + d] * sign;
                let x1 = buf[base + d];
                let x2 = buf[base + d + half];
                buf[base + d] = x1 * c - x2 * s;
                buf[base + d + half] = x2 * c + x1 * s;
            }
        }
    }
}

#[test]
fn rotated_prefix_matches_host_oracle_and_tail_is_untouched() {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "rope2d_partial");

    // Deliberately pairwise-distinct dims: head_dim (10) >> rot_dim (4, half=2)
    // so the untouched-tail region is nonempty and easy to spot-check.
    let (rows, heads, half, head_dim) = (3usize, 2usize, 2usize, 10usize);
    let row_stride = heads * head_dim;
    let off = 0usize;

    let mut rng = Lcg::new(4242);
    let buf_h = rng.vec_scaled(rows * row_stride, 1.0);
    let cos_h = rng.vec_scaled(rows * half, 1.0).iter().map(|x| x.cos()).collect::<Vec<_>>();
    let sin_h = rng.vec_scaled(rows * half, 1.0).iter().map(|x| x.sin()).collect::<Vec<_>>();

    let buf = g.storage_init("buf", &buf_h);
    let cos = g.storage_init("cos", &cos_h);
    let sin = g.storage_init("sin", &sin_h);

    let step = rope2d_partial_fwd(&g, kernel, &buf, &cos, &sin, rows as u32, heads as u32, half as u32, row_stride as u32, off as u32, head_dim as u32);
    g.submit(&[], &[step]);
    let got = g.read(&buf, rows * row_stride);

    let mut want = buf_h.clone();
    host_rope2d_partial(&mut want, &cos_h, &sin_h, rows, heads, half, row_stride, off, head_dim, 1.0);

    for i in 0..want.len() {
        assert!((got[i] - want[i]).abs() < 1e-5, "mismatch at {i}: got {} want {}", got[i], want[i]);
    }

    // The tail [2*half, head_dim) of every head must be bit-identical to the
    // original input — proof the kernel never addresses it.
    for row in 0..rows {
        for h in 0..heads {
            let base = row * row_stride + off + h * head_dim;
            for d in (2 * half)..head_dim {
                assert_eq!(got[base + d], buf_h[base + d], "untouched tail must be bit-identical at row {row} head {h} d {d}");
            }
        }
    }
}

#[test]
fn backward_is_the_exact_inverse_of_forward() {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "rope2d_partial");

    let (rows, heads, half, head_dim) = (2usize, 3usize, 3usize, 8usize);
    let row_stride = heads * head_dim;
    let off = 0usize;

    let mut rng = Lcg::new(777);
    let buf_h = rng.vec_scaled(rows * row_stride, 1.0);
    let angle = rng.vec_scaled(rows * half, 1.0);
    let cos_h: Vec<f32> = angle.iter().map(|x| x.cos()).collect();
    let sin_h: Vec<f32> = angle.iter().map(|x| x.sin()).collect();

    let buf = g.storage_init("buf", &buf_h);
    let cos = g.storage_init("cos", &cos_h);
    let sin = g.storage_init("sin", &sin_h);

    let fwd = rope2d_partial_fwd(&g, kernel, &buf, &cos, &sin, rows as u32, heads as u32, half as u32, row_stride as u32, off as u32, head_dim as u32);
    let bwd = rope2d_partial_bwd(&g, kernel, &buf, &cos, &sin, rows as u32, heads as u32, half as u32, row_stride as u32, off as u32, head_dim as u32);
    g.submit(&[], &[fwd, bwd]);
    let got = g.read(&buf, rows * row_stride);

    for i in 0..got.len() {
        assert!((got[i] - buf_h[i]).abs() < 1e-4, "fwd then bwd must round-trip at {i}: got {} want {}", got[i], buf_h[i]);
    }
}
