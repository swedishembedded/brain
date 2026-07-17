// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight quantization helpers, and a re-export of the BN fold.
//!
//! `fold_bn`/`BN_EPS` moved to `crates/vision`: the fold is a property of the conv
//! block (it is what `Conv`'s eval path already does), not of quantization, and
//! ZipDepth's RepVGG reparameterization needs it without taking a dependency on
//! this crate's OpenVINO runtime. Re-exported here so `topology.rs`/`sim.rs` and
//! any out-of-tree caller keep working unchanged.

pub use vision::fold::{fold_bn, BN_EPS};

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
