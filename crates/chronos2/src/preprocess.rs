// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side preprocessing — the numerically load-bearing input contract, done
//! in Rust (not on the GPU) because it is per-row, cheap, and runs once.
//!
//! Two pieces, both parity-critical (the spec flags them as the traps):
//! - [`instance_norm`] / [`instance_norm_inverse`] — Chronos-2's `InstanceNorm`:
//!   per-row **standardization** (NaN-aware mean/std; a scale of exactly 0 becomes 1e-5)
//!   followed by **arcsinh** squashing. The inverse applies `sinh` then the
//!   affine. This is standardization, *not* median/MAD, despite loose
//!   descriptions of it as "robust scaling".
//! - [`context_features`] — the non-overlapping patcher: **left-pad** the series
//!   with NaN to a multiple of the patch size, unfold, derive the observed mask,
//!   zero unobserved values, and build the 48-wide per-patch feature
//!   `[time_enc(P), values(P), mask(P)]` plus the per-patch attention mask.
//!
//! `arcsinh(x) = ln(x + sqrt(x^2 + 1))`, `sinh(x) = (e^x - e^-x)/2`.

use crate::config::Chronos2Config;

/// Per-row standardization statistics, carried for future covariates and for
/// denormalizing the forecast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocScale {
    pub loc: f32,
    pub scale: f32,
}

fn arcsinh(x: f32) -> f32 {
    // ln(x + sqrt(x^2 + 1)); numerically fine for the magnitudes here.
    (x + (x * x + 1.0).sqrt()).ln()
}

fn sinh(x: f32) -> f32 {
    (x.exp() - (-x).exp()) * 0.5
}

/// Standardize one series (NaN = missing) and optionally apply arcsinh.
/// Returns the `(loc, scale)` and the transformed values (NaN preserved where
/// the input was NaN — the patcher zeroes those via the mask).
pub fn instance_norm(x: &[f32], use_arcsinh: bool) -> (LocScale, Vec<f32>) {
    // NaN-aware mean.
    let (mut sum, mut cnt) = (0.0f32, 0usize);
    for &v in x {
        if v.is_finite() {
            sum += v;
            cnt += 1;
        }
    }
    let loc = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
    // NaN-aware population variance about loc.
    let (mut vsum, mut vcnt) = (0.0f32, 0usize);
    for &v in x {
        if v.is_finite() {
            let d = v - loc;
            vsum += d * d;
            vcnt += 1;
        }
    }
    let var = if vcnt > 0 { vsum / vcnt as f32 } else { f32::NAN };
    let mut scale = var.sqrt();
    if !scale.is_finite() {
        scale = 1.0; // nan_to_num(..., 1.0): all-NaN row
    }
    if scale == 0.0 {
        scale = 1e-5; // constant row
    }
    let out: Vec<f32> = x
        .iter()
        .map(|&v| {
            let s = (v - loc) / scale;
            if use_arcsinh {
                arcsinh(s)
            } else {
                s
            }
        })
        .collect();
    (LocScale { loc, scale }, out)
}

/// Apply an *existing* `LocScale` to new values (standardize + optional arcsinh),
/// the forward transform without recomputing statistics — used to normalize a
/// series' *future* covariate values by the scale learned from its context
/// (matching the reference's `instance_norm(future, loc_scale)`).
pub fn instance_norm_apply(x: &[f32], ls: LocScale, use_arcsinh: bool) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let s = (v - ls.loc) / ls.scale;
            if use_arcsinh {
                arcsinh(s)
            } else {
                s
            }
        })
        .collect()
}

/// Invert [`instance_norm`] for a forecast: `sinh` (if arcsinh) then the affine
/// `x*scale + loc`.
pub fn instance_norm_inverse(x: &[f32], ls: LocScale, use_arcsinh: bool) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let u = if use_arcsinh { sinh(v) } else { v };
            u * ls.scale + ls.loc
        })
        .collect()
}

/// The patched context, ready to feed the input embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Patched {
    /// `[n_patches, 3*patch]` row-major features `[time_enc, values, mask]`.
    pub features: Vec<f32>,
    /// `[n_patches]` — 1.0 if the patch has any observed value, else 0.0.
    pub attn_mask: Vec<f32>,
    /// Number of patches.
    pub n_patches: usize,
    /// Feature width `3*patch`.
    pub feat_dim: usize,
}

/// Build the per-patch context features from an already-standardized series.
///
/// Steps (matching the reference `Patch` + `_prepare_patched_context`):
/// 1. **Left-pad** with NaN to a multiple of `patch`.
/// 2. Unfold into non-overlapping patches.
/// 3. `mask = is_finite`; zero unobserved values.
/// 4. `attn_mask[p] = any observed in patch p`.
/// 5. Time encoding: raw indices `-(n*patch) .. -1` divided by `time_scale`,
///    laid out per patch.
/// 6. Concatenate `[time_enc(P), values(P), mask(P)]` per patch → `3*patch`.
pub fn context_features(scaled: &[f32], patch: usize, time_scale: f32) -> Patched {
    assert!(patch > 0);
    let len = scaled.len();
    let pad = (patch - (len % patch)) % patch;
    let n_total = len + pad;
    let n_patches = n_total / patch;
    let feat_dim = 3 * patch;

    // padded series (left NaN pad)
    let mut padded = vec![f32::NAN; pad];
    padded.extend_from_slice(scaled);

    let mut features = vec![0.0f32; n_patches * feat_dim];
    let mut attn_mask = vec![0.0f32; n_patches];

    for (p, am) in attn_mask.iter_mut().enumerate() {
        let base = p * feat_dim;
        let mut any = false;
        for c in 0..patch {
            let gi = p * patch + c; // index into padded
            let v = padded[gi];
            let observed = v.is_finite();
            if observed {
                any = true;
            }
            // time index runs from -(n_total) .. -1 over the padded stream
            let t_idx = gi as f32 - n_total as f32;
            let value = if observed { v } else { 0.0 };
            let m = if observed { 1.0 } else { 0.0 };
            features[base + c] = t_idx / time_scale; // time_enc channel
            features[base + patch + c] = value; // values channel
            features[base + 2 * patch + c] = m; // mask channel
        }
        *am = if any { 1.0 } else { 0.0 };
    }

    Patched { features, attn_mask, n_patches, feat_dim }
}

/// Build per-patch features for the forecast horizon of a **pure target** (no
/// known-future covariate): values and mask are all zero, and the time encoding
/// runs forward `0 .. n_out*patch-1`. Right-padded to a whole number of patches.
pub fn future_features(horizon: usize, patch: usize, time_scale: f32) -> Patched {
    future_features_with_values(horizon, patch, time_scale, None)
}

/// As [`future_features`], but if `future_scaled` (an already-standardized future
/// series of length `horizon`) is given, its values fill the `values` channel and
/// the `mask` channel is set to 1 — the reference's known-future-covariate path.
/// Padding positions past `horizon` (within the final patch) stay 0/masked.
pub fn future_features_with_values(
    horizon: usize,
    patch: usize,
    time_scale: f32,
    future_scaled: Option<&[f32]>,
) -> Patched {
    assert!(patch > 0);
    let n_patches = horizon.div_ceil(patch);
    let feat_dim = 3 * patch;
    let mut features = vec![0.0f32; n_patches * feat_dim];
    for p in 0..n_patches {
        let base = p * feat_dim;
        for c in 0..patch {
            let idx = p * patch + c;
            features[base + c] = idx as f32 / time_scale; // time_enc
            if let Some(fv) = future_scaled {
                if idx < horizon {
                    features[base + patch + c] = fv[idx]; // values
                    features[base + 2 * patch + c] = 1.0; // mask (known)
                }
            }
        }
    }
    Patched { features, attn_mask: vec![1.0; n_patches], n_patches, feat_dim }
}

/// One univariate input laid out for the embedding: the per-patch features of
/// the context patches followed by those of the future patches, and the additive
/// attention mask over the full token sequence (context patches, `[REG]`,
/// future patches).
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// `[n_ctx + n_out, 3 * patch]`: context rows then future rows.
    pub features: Vec<f32>,
    /// `[n_ctx + 1 + n_out]`, `0` to attend and `-3.4e38` for a context patch
    /// with no observed value.
    pub mask: Vec<f32>,
    pub n_ctx: usize,
    pub n_out: usize,
    /// The standardization the forecast is mapped back through.
    pub ls: LocScale,
}

/// Standardize `context`, patch it, and append the future patches for
/// `horizon` steps (steps 1-3 of a forecast, shared by inference and training).
pub fn layout(cfg: &Chronos2Config, context: &[f32], horizon: usize) -> Layout {
    let patch = cfg.input_patch_size;
    let (ls, scaled) = instance_norm(context, cfg.use_arcsinh);
    let ctx = context_features(&scaled, patch, cfg.time_encoding_scale);
    let fut = future_features(horizon, patch, cfg.time_encoding_scale);
    let (n_ctx, n_out) = (ctx.n_patches, fut.n_patches);

    let mut features = ctx.features;
    features.extend_from_slice(&fut.features);
    // context patches per observed-mask, REG + future attendable.
    let mut mask = vec![0.0f32; n_ctx + 1 + n_out];
    for (m, &attend) in mask.iter_mut().zip(&ctx.attn_mask) {
        if attend == 0.0 {
            *m = -3.4e38;
        }
    }
    Layout { features, mask, n_ctx, n_out, ls }
}

/// Splice the embedded patches `[n_ctx + n_out, d]` and the `[REG]` row into
/// the token sequence `[n_ctx + 1 + n_out, d]`: context tokens, REG, future tokens.
pub fn assemble_tokens(patch_emb: &[f32], reg: &[f32], n_ctx: usize, d: usize) -> Vec<f32> {
    let mut emb = Vec::with_capacity(patch_emb.len() + d);
    emb.extend_from_slice(&patch_emb[..n_ctx * d]);
    emb.extend_from_slice(reg);
    emb.extend_from_slice(&patch_emb[n_ctx * d..]);
    emb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_places_context_then_future_and_masks_only_unobserved_patches() {
        let cfg = Chronos2Config::tiny(); // patch 4
        // 6 observed values: left-padded to 8 = 2 patches, the first half observed.
        let ctx: Vec<f32> = (0..6).map(|i| 10.0 + i as f32).collect();
        let l = layout(&cfg, &ctx, 5); // 5 steps -> 2 future patches
        assert_eq!((l.n_ctx, l.n_out), (2, 2));
        assert_eq!(l.features.len(), 4 * cfg.patch_feat_dim());
        assert_eq!(l.mask.len(), 2 + 1 + 2);
        assert!(l.mask.iter().all(|&m| m == 0.0), "every patch has an observed value");
        // an all-NaN leading patch is masked out
        let mut gap = vec![f32::NAN; 4];
        gap.extend((0..4).map(|i| i as f32));
        let l = layout(&cfg, &gap, 4);
        assert_eq!(l.mask[0], -3.4e38);
        assert_eq!(l.mask[1], 0.0);
    }

    #[test]
    fn tokens_are_context_then_reg_then_future() {
        let patch_emb = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]; // d = 2: two context rows, one future row
        assert_eq!(assemble_tokens(&patch_emb, &[9.0, 9.0], 2, 2), vec![1.0, 1.0, 2.0, 2.0, 9.0, 9.0, 3.0, 3.0]);
    }

    #[test]
    fn standardize_matches_the_closed_form_without_arcsinh() {
        // [1,2,3,4,5]: loc=3, var=2, scale=sqrt(2)
        let (ls, out) = instance_norm(&[1.0, 2.0, 3.0, 4.0, 5.0], false);
        assert!((ls.loc - 3.0).abs() < 1e-6);
        assert!((ls.scale - 2.0f32.sqrt()).abs() < 1e-6);
        let s = 2.0f32.sqrt();
        assert!((out[0] - (-2.0 / s)).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[4] - (2.0 / s)).abs() < 1e-6);
    }

    #[test]
    fn arcsinh_is_applied_and_inverts() {
        let x = [10.0, -4.0, 0.5, 3.0, 7.0, 2.0];
        let (ls, fwd) = instance_norm(&x, true);
        // arcsinh(0) == 0, so the mean-valued point maps near 0 after squashing
        let back = instance_norm_inverse(&fwd, ls, true);
        for i in 0..x.len() {
            assert!((back[i] - x[i]).abs() < 1e-4, "roundtrip {i}: {} vs {}", back[i], x[i]);
        }
    }

    #[test]
    fn constant_row_floors_the_scale() {
        let (ls, out) = instance_norm(&[5.0, 5.0, 5.0], false);
        assert!((ls.loc - 5.0).abs() < 1e-6);
        assert!((ls.scale - 1e-5).abs() < 1e-9, "scale {}", ls.scale);
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn all_nan_row_is_neutral() {
        let (ls, _) = instance_norm(&[f32::NAN, f32::NAN], false);
        assert_eq!(ls.loc, 0.0);
        assert_eq!(ls.scale, 1.0);
    }

    #[test]
    fn nan_aware_mean_and_std_skip_missing() {
        // [1, NaN, 3]: loc=2, var=mean(1,1)=1, scale=1
        let (ls, out) = instance_norm(&[1.0, f32::NAN, 3.0], false);
        assert!((ls.loc - 2.0).abs() < 1e-6);
        assert!((ls.scale - 1.0).abs() < 1e-6);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!(out[1].is_nan(), "missing stays NaN pre-patch");
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn patcher_left_pads_and_zeroes_missing() {
        // 5 values, patch 4 -> pad 3 on the LEFT -> padded=[N,N,N,1,2,3,4,5].
        let scaled = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = context_features(&scaled, 4, 8.0);
        assert_eq!(p.n_patches, 2);
        assert_eq!(p.feat_dim, 12);
        // both patches contain a real value (padding is < patch, so the first
        // patch always keeps at least one real cell) -> both attendable.
        assert_eq!(p.attn_mask, vec![1.0, 1.0]);
        // patch 0 = [N,N,N,1]: three padded cells zeroed, last cell is the real 1.
        assert_eq!(&p.features[4..8], &[0.0, 0.0, 0.0, 1.0], "values channel");
        assert_eq!(&p.features[8..12], &[0.0, 0.0, 0.0, 1.0], "mask channel");
        // patch 1 = [2,3,4,5]
        let base = p.feat_dim;
        assert_eq!(&p.features[base + 4..base + 8], &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&p.features[base + 8..base + 12], &[1.0, 1.0, 1.0, 1.0]);
        // time encoding of the very last cell is -1/time_scale
        assert!((p.features[base + 3] - (-1.0 / 8.0)).abs() < 1e-6);
    }

    #[test]
    fn a_fully_missing_patch_is_not_attendable() {
        // patch 4; a whole interior patch of NaN (missing data, not padding)
        // must set attn_mask 0 for that patch and zero its values.
        let scaled = vec![
            1.0, 2.0, 3.0, 4.0, // patch 0: observed
            f32::NAN, f32::NAN, f32::NAN, f32::NAN, // patch 1: all missing
            5.0, 6.0, 7.0, 8.0, // patch 2: observed
        ];
        let p = context_features(&scaled, 4, 100.0);
        assert_eq!(p.n_patches, 3);
        assert_eq!(p.attn_mask, vec![1.0, 0.0, 1.0]);
        // the missing patch's values + mask channels are all zero
        let base = p.feat_dim; // patch 1
        assert_eq!(&p.features[base + 4..base + 8], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&p.features[base + 8..base + 12], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn future_features_are_zero_valued_with_forward_time() {
        let f = future_features(6, 4, 8.0);
        assert_eq!(f.n_patches, 2); // ceil(6/4)
        assert!(f.attn_mask.iter().all(|&m| m == 1.0));
        // values + mask channels are zero; time runs forward from 0
        assert!((f.features[0] - 0.0).abs() < 1e-6); // t=0
        assert!((f.features[1] - (1.0 / 8.0)).abs() < 1e-6); // t=1
        // values channel (index 4) is zero
        assert_eq!(f.features[4], 0.0);
    }
}
