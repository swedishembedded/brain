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

/// Start index of every rotated `(2j, 2j+1)` pair in a `rows x stride` buffer.
fn rotated_pairs(rows: usize, stride: usize, off: usize, heads: usize, head_dim: usize, rot: usize) -> impl Iterator<Item = usize> {
    (0..rows).flat_map(move |r| (0..heads).flat_map(move |h| (0..rot / 2).map(move |j| r * stride + off + h * head_dim + 2 * j)))
}

/// Largest position `row % tcols` the test uses. The WGSL accuracy table bounds `sin`/`cos` only on `[-pi, pi]`, so
/// the largest angle (`pos * theta^0 = pos` at channel pair 0) is kept inside it.
const MAX_POS: usize = 2;
const _: () = assert!(MAX_POS < 3, "pos must stay below pi for the sin/cos bound to apply");

/// Bound on `|device - f64 oracle|` for the pair `(e, o)` rotated by an angle in `[-pi, pi]`, from the WGSL accuracy
/// table (W3C WGSL, "Floating Point Accuracy"), not chosen:
///
/// * `sin(x)`, `cos(x)`: absolute error at most `2^-11` on `[-pi, pi]`. Each of `e*c - o*s` and `e*s + o*c` is
///   therefore off by at most `(|e| + |o|) * 2^-11`. This dominates: an Intel Arc measures `~3e-5` here, a 1e-5
///   fixed tolerance was below what the specification allows a conformant device.
/// * The angle `pos * pow(theta, -2j/rot)`: `pow` inherits `exp2(y * log2(x))`. With `|y log2 x| <= 20.1` (theta =
///   8e6, `2j/rot <= 7/8`): `log2` is within 3 ULP of `22.93` (`3 * 2^-19`, scaled by `|y| <= 7/8`) plus half an ULP of
///   the product (`2^-20`), `<= 6e-6` absolute in the exponent, which is `ln 2 * 6e-6 = 4.2e-6` relative in the result;
///   `exp2` adds `(3 + 2 * 20.1) ULP <= 43 * 2^-23 = 5.1e-6`. So the relative error of the weight is `<= 1e-5`, and
///   the angle (`<= MAX_POS`) is off by `<= MAX_POS * 1e-5`, which moves `c` and `s` by at most that much.
/// * The two products and the sum of each output: three rounded operations on terms of absolute sum `S`,
///   `rounding::accumulation_bound(3, S)`.
fn rotation_bound(e: f32, o: f32) -> f64 {
    const SIN_COS_ABS_ERR: f64 = 1.0 / 2048.0;
    const ANGLE_REL_ERR: f64 = 1e-5;
    let (e, o) = (f64::from(e).abs(), f64::from(o).abs());
    let trig = SIN_COS_ABS_ERR + MAX_POS as f64 * ANGLE_REL_ERR;
    (e + o) * trig + brain_testutil::rounding::accumulation_bound(3, e + o)
}

/// Assert every rotated pair of `got` is within [`rotation_bound`] of `want` (`input` holds the unrotated values) and
/// that every element outside the pairs is untouched, bit for bit.
fn assert_within_bound(name: &str, input: &[f32], got: &[f32], want: &[f32], pairs: impl Iterator<Item = usize>) {
    let mut rotated = vec![false; input.len()];
    for b in pairs {
        rotated[b] = true;
        rotated[b + 1] = true;
        let bound = rotation_bound(input[b], input[b + 1]);
        for i in [b, b + 1] {
            let err = (f64::from(got[i]) - f64::from(want[i])).abs();
            assert!(err <= bound, "{name}: element {i} error {err:e} exceeds the WGSL-derived bound {bound:e}");
        }
    }
    for (i, _) in rotated.iter().enumerate().filter(|(_, r)| !**r) {
        assert_eq!(got[i].to_bits(), input[i].to_bits(), "{name}: element {i} outside the rotated region changed");
    }
}

#[test]
fn every_interleaved_rope_kernel_rotates_at_the_callers_theta() {
    let theta = 8.0e6f32;
    let (rows, heads, head_dim, tcols) = (12usize, 2usize, 16usize, MAX_POS + 1);
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
        assert_within_bound(name, &x, &got, &want, rotated_pairs(rows, stride, off, heads, head_dim, head_dim));
    }

    // The indexer layout: rotate the first `rope_dim` channels of each
    // `head_dim`-wide head, rows packed at `heads * head_dim`.
    let rope_dim = 8usize;
    let packed = heads * head_dim;
    let mut want = x.clone();
    host_rotate(&mut want, rows, heads, head_dim, rope_dim, packed, 0, tcols, theta, 1.0);
    let got = run(ROPE_SUB, &[r, h, hd, rope_dim as u32, packed as u32, tc, t], (rows * heads * rope_dim / 2) as u32);
    assert_within_bound("rope_sub", &x, &got, &want, rotated_pairs(rows, packed, 0, heads, head_dim, rope_dim));
}
