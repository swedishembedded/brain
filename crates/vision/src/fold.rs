// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BatchNorm folding: collapse conv+BN(eval) into a single biased conv.
//!
//! Lives here, not in `crates/npu`, because it is a property of the CONV BLOCK,
//! not of quantization or of OpenVINO. Three separate consumers need it and only
//! one of them is the NPU:
//!   * `vision::blocks::Conv`'s eval path collapses BN into a per-channel
//!     scale/shift (`pack_sb`);
//!   * `brain-npu`'s exporter reproduces the identical fold so the emitted ONNX
//!     `Conv` (weight + bias) is numerically the same graph;
//!   * ZipDepth's RepVGG reparameterization folds each of its three branches
//!     before merging them into one 3x3.
//!
//! The third is why it moved: `crates/zipdepth` must not depend on `brain-npu`,
//! which carries the OpenVINO runtime. A pure function over flat slices with no
//! imports has no business gating a model crate on a hardware backend.

/// Eps used by brain's BN-eval collapse (`Conv::pack_sb` / `bn_eval.wgsl`), and
/// PyTorch's `nn.BatchNorm2d` default. All three must agree or the folded and
/// unfolded paths diverge.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The fold must be exact: BN(conv(x)) == conv'(x) + b for every x.
    /// Checked on a 1x1x1 "conv" so the arithmetic is inspectable by hand.
    #[test]
    fn folded_conv_reproduces_conv_then_bn() {
        let (w, gamma, beta, rm, rv) = (vec![2.0f32, -3.0], vec![0.5f32, 2.0], vec![1.0f32, -1.0], vec![0.25f32, 0.5], vec![4.0f32, 1.0]);
        let (wp, b) = fold_bn(&w, &gamma, &beta, &rm, &rv, 2);
        for (o, x) in [(0usize, 1.5f32), (1, -0.75)] {
            let conv = w[o] * x;
            let want = gamma[o] * (conv - rm[o]) / (rv[o] + BN_EPS).sqrt() + beta[o];
            let got = wp[o] * x + b[o];
            assert!((got - want).abs() < 1e-6, "channel {o}: folded {got} != BN(conv) {want}");
        }
    }
}
