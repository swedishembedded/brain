// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BatchNorm folding + weight quantization helpers.
//!
//! Brain's eval-mode `Conv` already collapses conv+BN into a per-channel
//! scale/shift (`yolo::blocks::Conv::pack_sb`). The exporter reproduces the exact
//! same fold here so the ONNX `Conv` (weight + bias) is numerically identical.

/// Eps used by brain's BN-eval collapse (`Conv::pack_sb` / `bn_eval`). Must match.
pub const BN_EPS: f32 = 1e-5;

/// Fold BN(eval) into a bias-free conv weight, returning `(w', bias)`:
/// `scale[o] = gamma[o]/sqrt(run_var[o]+eps)`, `w'[o,…] = w[o,…]·scale[o]`,
/// `bias[o] = beta[o] − run_mean[o]·scale[o]`. `cout` = output channels; `w` is
/// row-major `[cout, cin*k*k]`.
pub fn fold_bn(
    w: &[f32],
    gamma: &[f32],
    beta: &[f32],
    run_mean: &[f32],
    run_var: &[f32],
    cout: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(w.len() % cout, 0, "weight length {} not divisible by cout {cout}", w.len());
    let per = w.len() / cout;
    let mut wp = vec![0.0f32; w.len()];
    let mut bias = vec![0.0f32; cout];
    for o in 0..cout {
        let scale = gamma[o] / (run_var[o] + BN_EPS).sqrt();
        for i in 0..per {
            wp[o * per + i] = w[o * per + i] * scale;
        }
        bias[o] = beta[o] - run_mean[o] * scale;
    }
    (wp, bias)
}

/// Symmetric per-output-channel INT8 quantization of a folded conv weight
/// `[cout, per]`: returns the INT8 weights (same layout) and the per-channel
/// scales `[cout]` (`scale[o] = max_i|w'[o,i]| / 127`). Data-independent, so it
/// lives here (no calibration needed for weights).
pub fn quantize_weight_per_channel(wp: &[f32], cout: usize, per: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; wp.len()];
    let mut scales = vec![0.0f32; cout];
    for o in 0..cout {
        let mut mx = 0.0f32;
        for i in 0..per {
            mx = mx.max(wp[o * per + i].abs());
        }
        let scale = (mx / 127.0).max(1e-12);
        scales[o] = scale;
        for i in 0..per {
            let v = (wp[o * per + i] / scale).round().clamp(-127.0, 127.0);
            q[o * per + i] = v as i8;
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_bn_matches_pack_sb_formula() {
        // 1 out-channel, 2x1x1 weight. gamma=2, beta=1, mean=0.5, var=3.
        let (w, gamma, beta, rm, rv) = (vec![1.0f32, -2.0], vec![2.0], vec![1.0], vec![0.5], vec![3.0]);
        let (wp, bias) = fold_bn(&w, &gamma, &beta, &rm, &rv, 1);
        let scale = 2.0f32 / (3.0 + BN_EPS).sqrt();
        assert!((wp[0] - 1.0 * scale).abs() < 1e-6);
        assert!((wp[1] - (-2.0) * scale).abs() < 1e-6);
        assert!((bias[0] - (1.0 - 0.5 * scale)).abs() < 1e-6);
    }

    #[test]
    fn weight_quant_roundtrip_is_close() {
        let wp = vec![0.5f32, -0.25, 0.1, -0.9];
        let (q, s) = quantize_weight_per_channel(&wp, 2, 2);
        // dequantize and check error is within a quantization step.
        for o in 0..2 {
            for i in 0..2 {
                let deq = q[o * 2 + i] as f32 * s[o];
                assert!((deq - wp[o * 2 + i]).abs() <= s[o] + 1e-9);
            }
        }
    }
}
