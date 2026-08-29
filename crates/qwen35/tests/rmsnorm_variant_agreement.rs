// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decode tape's RMSNorm seam (`qwen35::model::rms_step`, which routes
//! through `model::block::rms_variant`) must compute the same normalization as
//! the per-element reference kernel it replaced, at every shape the real
//! Qwen3.8-27B decode step actually dispatches.
//!
//! This gate exists because the swap is NOT bit-identical and was adopted for
//! speed: `rmsnorm_rows` folds 64 partial sums in a different order than
//! `rmsnorm`'s single-threaded loop, so the two agree to floating-point
//! rounding, not to the bit. Adopting a kernel on a throughput argument without
//! pinning what it computes is precisely how a fast-but-wrong kernel gets into
//! a decode path, so the agreement is asserted here rather than assumed from
//! the kernel's own header.
//!
//! The shapes are the REAL ones, read off `Qwen35Config::qwen38_27b()` rather
//! than written down: the three the decode tape uses are a `[1, d_model]`
//! residual norm (ln1/ln2/final/head), a `[n_heads, head_dim]` QK-norm, and a
//! `[linear_num_value_heads, linear_value_head_dim]` gated GDN norm. The
//! narrow-row cases are the ones worth having - that is where the cooperative
//! kernel's reduction tree differs most from a serial loop, and where the
//! per-element kernel's uncoalesced reads made it the top row of the decode
//! profile in the first place.

use data::rng::Lcg;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::pipelines;

/// Relative agreement floor. The kernels differ only in reduction ORDER over
/// the same `d` squares, so the error is `O(sqrt(d) * eps)` on a sum whose
/// scale is `d`; `rmsnorm_rows`'s own header records `3.3e-6` max_abs measured
/// over a wide sweep. This is that, with room for a wider row, and it is a
/// TIGHT bound on purpose: a real defect (wrong eps, a missed tail element, a
/// mis-strided row) moves the answer by orders of magnitude more, not by a
/// few ulp.
const TOL: f32 = 2e-5;

/// The reference: RMSNorm exactly as `rmsnorm.wgsl` computes it (serial inner
/// reduction, hardcoded 1e-6), on the host. Comparing the two DEVICE kernels
/// against each other would pass if both were wrong the same way; a host
/// reference cannot be.
fn rmsnorm_host(x: &[f32], w: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let base = r * d;
        let mut ss = 0.0f32;
        for c in 0..d {
            let v = x[base + c];
            ss += v * v;
        }
        let inv = 1.0 / (ss / d as f32 + 1e-6).sqrt();
        for c in 0..d {
            out[base + c] = w[c] * x[base + c] * inv;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

#[test]
fn the_decode_rmsnorm_seam_matches_the_reference_at_every_real_shape() {
    let gpu = Gpu::new(pipelines());
    let cfg = Qwen35Config::qwen38_27b();

    // Every `(rows, d)` the real decode tape dispatches an RMSNorm at - see
    // `Qwen35::run_decode_step`, `layer_gqa_decode_step` and
    // `layer_gdn_decode_step`.
    let shapes: [(u32, u32, &str); 4] = [
        (1, cfg.d_model, "ln1/ln2/final-norm/head: one residual row"),
        (cfg.n_heads, cfg.head_dim, "GQA q_norm"),
        (cfg.n_kv_heads, cfg.head_dim, "GQA k_norm"),
        (cfg.linear_num_value_heads, cfg.linear_value_head_dim, "GDN gated norm"),
    ];

    let mut rng = Lcg::new(20260829);
    for (rows, d, what) in shapes {
        let (rows_u, d_u) = (rows as usize, d as usize);
        // Scaled well away from 1.0 so a dropped `eps` or a wrong element count
        // cannot hide inside a coincidentally-unit normalization.
        let x = rng.vec_scaled(rows_u * d_u, 3.0);
        let w = rng.vec_scaled(d_u, 0.5);

        let xb = gpu.storage_init("x", &x);
        let wb = gpu.storage_init("w", &w);
        let ob = gpu.storage((rows_u * d_u) as u64);
        gpu.submit(&[], &[qwen35::model::rms_step(&gpu, &xb, &wb, &ob, d, rows)]);
        let got = gpu.read(&ob, rows_u * d_u);

        let want = rmsnorm_host(&x, &w, rows_u, d_u);
        let e = rel_err(&got, &want);
        assert!(e <= TOL, "{what} ({rows}x{d}): relative error {e:e} exceeds {TOL:e}");
        assert!(got.iter().all(|v| v.is_finite()), "{what} ({rows}x{d}): produced a non-finite value");
    }
}
