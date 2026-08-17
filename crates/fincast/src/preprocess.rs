// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side preprocessing — faithful to the reference
//! `PatchedTimeSeriesDecoder_MOE._preprocess_input` / `_forward_transform`.
//!
//! The context is padded/truncated to `context_len`, split into `patch_len`
//! windows, standardized by the **first patch with >3 valid values** (not the
//! whole series), padded positions zeroed, and the per-patch padding flag
//! computed. Returns the `[n_patches, 2*patch_len]` `[values, mask]` features,
//! the per-patch padding mask, and the `(mu, sigma)` used to reverse it.

use crate::config::FincastConfig;

/// `(mu, sigma)` of the standardization (reversed on the head output).
#[derive(Clone, Copy, Debug)]
pub struct LocScale {
    pub mu: f32,
    pub sigma: f32,
}

/// Result of patch preprocessing.
pub struct Patched {
    /// `[n_patches, 2*patch_len]` — standardized `[values, mask]` per patch.
    pub features: Vec<f32>,
    /// `[n_patches]` — 1.0 if the whole patch is padded, else 0.0 (min over the
    /// patch's per-element padding, per the reference `torch.min(patched_pads)`).
    pub patch_padding: Vec<f32>,
    pub n_patches: usize,
    pub loc_scale: LocScale,
}

/// Pad (front) / truncate a raw context to exactly `context_len`, returning the
/// padded values and a parallel padding flag (1.0 = padded).
fn pad_to_context(context: &[f32], context_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut vals = vec![0.0f32; context_len];
    let mut pad = vec![0.0f32; context_len];
    let l = context.len();
    if l >= context_len {
        vals.copy_from_slice(&context[l - context_len..]);
    } else {
        let front = context_len - l;
        pad[..front].fill(1.0);
        vals[front..].copy_from_slice(context);
    }
    (vals, pad)
}

/// `_masked_mean_std`: statistics of the first patch with >3 valid (unpadded)
/// values; falls back to the last patch when none qualify.
fn masked_mean_std(patch_vals: &[f32], patch_pad: &[f32], n: usize, patch_len: usize, tol: f32) -> LocScale {
    // pick the patch index
    let mut idx = n - 1;
    for p in 0..n {
        let valid: f32 = (0..patch_len).map(|i| 1.0 - patch_pad[p * patch_len + i]).sum();
        if valid >= 3.0 {
            idx = p;
            break;
        }
    }
    let base = idx * patch_len;
    let mut num_valid = 0.0f32;
    let mut sum = 0.0f32;
    for i in 0..patch_len {
        let m = 1.0 - patch_pad[base + i];
        num_valid += m;
        sum += patch_vals[base + i] * m;
    }
    num_valid = num_valid.max(1.0);
    let mu = sum / num_valid;
    let mut var = 0.0f32;
    for i in 0..patch_len {
        let m = 1.0 - patch_pad[base + i];
        let c = (patch_vals[base + i] - mu) * m;
        var += c * c;
    }
    var = (var / num_valid).max(0.0);
    let sigma = var.sqrt().max(tol);
    LocScale { mu, sigma }
}

/// Full patch preprocessing for one context series.
pub fn preprocess(cfg: &FincastConfig, context: &[f32]) -> Patched {
    let patch = cfg.patch_len;
    let (vals, pad) = pad_to_context(context, cfg.context_len);
    let n = cfg.context_len / patch;
    let ls = masked_mean_std(&vals, &pad, n, patch, cfg.tolerance);

    let mut features = vec![0.0f32; n * 2 * patch];
    let mut patch_padding = vec![0.0f32; n];
    for p in 0..n {
        let mut all_pad = 1.0f32;
        for i in 0..patch {
            let padded = pad[p * patch + i];
            all_pad = all_pad.min(padded);
            // standardize, zero padded positions
            let sv = if padded > 0.5 { 0.0 } else { (vals[p * patch + i] - ls.mu) / ls.sigma };
            features[p * 2 * patch + i] = sv;
            features[p * 2 * patch + patch + i] = padded;
        }
        patch_padding[p] = all_pad;
    }
    Patched { features, patch_padding, n_patches: n, loc_scale: ls }
}

/// Reverse the standardization on a head output value.
#[inline]
pub fn denorm(v: f32, ls: LocScale) -> f32 {
    v * ls.sigma + ls.mu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_context_standardizes_by_first_patch() {
        let cfg = FincastConfig::tiny(); // patch 4, context 32
        let ctx: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let pp = preprocess(&cfg, &ctx);
        assert_eq!(pp.n_patches, 8);
        // no padding
        assert!(pp.patch_padding.iter().all(|&v| v == 0.0));
        // first patch = [0,1,2,3], mean 1.5
        assert!((pp.loc_scale.mu - 1.5).abs() < 1e-6);
        // features: first value standardized = (0-1.5)/sigma
        let expect0 = (0.0 - pp.loc_scale.mu) / pp.loc_scale.sigma;
        assert!((pp.features[0] - expect0).abs() < 1e-5);
        // mask half is zero (unpadded)
        assert_eq!(pp.features[cfg.patch_len], 0.0);
    }

    #[test]
    fn short_context_front_pads() {
        let cfg = FincastConfig::tiny();
        let ctx: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();
        let pp = preprocess(&cfg, &ctx);
        // 32 - 10 = 22 front-padded -> first 5 patches (20) fully padded, patch 5 partial
        assert_eq!(pp.patch_padding[0], 1.0);
        assert_eq!(pp.patch_padding[7], 0.0);
        // padded value positions are zeroed and flagged
        assert_eq!(pp.features[0], 0.0);
        assert_eq!(pp.features[cfg.patch_len], 1.0);
    }
}
