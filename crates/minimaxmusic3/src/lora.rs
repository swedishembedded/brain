// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA fine-tuning for the vocoder's conv weights.
//!
//! Every conv weight here is `[Cout, Cin_or_1, K]` (`Conv1d`) or
//! `[Cin, Cout, K]` (`ConvTranspose1d`) - in both cases dim0 is a row
//! count and the rest is one flat feature axis, so LoRA's usual
//! `ΔW = (alpha/r)·B·A` low-rank decomposition applies unchanged: treat the
//! flattened weight as `[rows, cols]`, `A: [r, cols]`, `B: [rows, r]`.
//!
//! Fold-then-run: the adapter is never a separate device path. Each step,
//! `apply` composes `W_eff = W_base + delta` on the host (small - `rows` is
//! at most 1536, `cols` at most `1536*16`, and this runs once per step, not
//! per element of the forward/backward tape) and hands `W_eff` to the
//! ordinary `train::Trainer`, so every existing conv/backward kernel runs
//! completely unaware a LoRA adapter exists. `backward` converts the
//! resulting `dW_eff` (from `Trainer::read_grad`, exactly as if the whole
//! tensor were trainable) into `(dA, dB)`; `W_base` itself is never
//! written back, so it stays frozen across steps.

use crate::config::VocoderConfig;
use crate::vocoder::VocoderWeights;

/// `(rows, cols)` of one conv weight's own storage layout - `Cout` (or,
/// for `blocks.{i}.conv_t1`'s native `ConvTranspose1d` `[Cin,Cout,K]`
/// layout, `Cin`) as `rows`, everything else flattened as `cols`. Matches
/// `crate::train::run_forward`'s per-layer dimension tracking exactly - the
/// dims a LoRA adapter's shape is meaningless without.
pub fn conv_weight_shape(cfg: &VocoderConfig, name: &str) -> (usize, usize) {
    let half = cfg.latent_channels as usize / 2;
    match name.strip_suffix(".weight").expect("conv_weight_shape: name must end in .weight") {
        "dec_in_proj" => (cfg.decoder_input_dim as usize, half),
        "conv_in" => (cfg.decoder_hidden_dim as usize, cfg.decoder_input_dim as usize * 7),
        "conv_out" => {
            let dim_final = cfg.decoder_hidden_dim as usize / (1 << cfg.upsampling_ratios.len());
            (1, dim_final * 7)
        }
        other => {
            let rest = other.strip_prefix("blocks.").expect("conv_weight_shape: unknown conv name");
            let (i, rest) = rest.split_once('.').unwrap();
            let i: usize = i.parse().unwrap();
            let mut dim = cfg.decoder_hidden_dim as usize;
            for _ in 0..i {
                dim /= 2;
            }
            if rest == "conv_t1" {
                let stride = cfg.upsampling_ratios[i] as usize;
                return (dim, (dim / 2) * (2 * stride));
            }
            let out_dim = dim / 2;
            let conv = rest.strip_prefix("res_unit").unwrap().split_once('.').unwrap().1;
            match conv {
                "conv1" => (out_dim, out_dim * 7),
                "conv2" => (out_dim, out_dim),
                _ => panic!("conv_weight_shape: unknown conv name {name:?}"),
            }
        }
    }
}

/// One conv's LoRA adapter. `b` is `[rows, rank]`; `a` is `[rank, cols]`.
/// Standard LoRA init: `b` starts at all zeros, so [`apply`] is an exact
/// no-op until the first update actually changes it.
#[derive(Clone)]
pub struct LoraW {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl LoraW {
    pub fn zero_init(rows: usize, cols: usize, rank: usize, a_init: impl FnMut(usize) -> f32) -> LoraW {
        LoraW { rows, cols, rank, a: (0..rank * cols).map(a_init).collect(), b: vec![0.0; rows * rank] }
    }
}

/// `delta[r,c] = scale * sum_k b[r,k]*a[k,c]` - `ΔW` flattened to `[rows,
/// cols]` row-major, matching every conv weight's own storage order.
pub fn delta(w: &LoraW, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; w.rows * w.cols];
    for r in 0..w.rows {
        for k in 0..w.rank {
            let bv = w.b[r * w.rank + k] * scale;
            if bv == 0.0 {
                continue;
            }
            let a_row = &w.a[k * w.cols..(k + 1) * w.cols];
            let out_row = &mut out[r * w.cols..(r + 1) * w.cols];
            for (o, &av) in out_row.iter_mut().zip(a_row) {
                *o += bv * av;
            }
        }
    }
    out
}

/// `W_eff = base + delta(w, scale)`, in place.
pub fn apply(base: &mut [f32], w: &LoraW, scale: f32) {
    let d = delta(w, scale);
    for (o, di) in base.iter_mut().zip(&d) {
        *o += di;
    }
}

/// `(dA, dB)` from `dW_eff` (the gradient the base conv's own backward
/// already computed, as if the whole flattened `[rows, cols]` weight were
/// directly trainable): `dB = scale * dW_eff @ A^T`, `dA = scale * B^T @
/// dW_eff`.
pub fn backward(w: &LoraW, d_w_eff: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
    let mut db = vec![0.0f32; w.rows * w.rank];
    for r in 0..w.rows {
        for k in 0..w.rank {
            let mut acc = 0.0f32;
            let a_row = &w.a[k * w.cols..(k + 1) * w.cols];
            let dw_row = &d_w_eff[r * w.cols..(r + 1) * w.cols];
            for (dwv, av) in dw_row.iter().zip(a_row) {
                acc += dwv * av;
            }
            db[r * w.rank + k] = acc * scale;
        }
    }
    let mut da = vec![0.0f32; w.rank * w.cols];
    for k in 0..w.rank {
        for c in 0..w.cols {
            let mut acc = 0.0f32;
            for r in 0..w.rows {
                acc += w.b[r * w.rank + k] * d_w_eff[r * w.cols + c];
            }
            da[k * w.cols + c] = acc * scale;
        }
    }
    (da, db)
}

/// Every `train::flatten`-style name that names a conv `.weight` leaf (the
/// only tensors LoRA ever adapts - never a bias, never a Snake alpha).
pub fn conv_weight_names(w: &VocoderWeights) -> Vec<String> {
    crate::train::flatten(w).into_iter().map(|(n, _)| n).filter(|n| n.ends_with(".weight")).collect()
}

/// `base` with every adapter in `adapters` applied (`W_eff = base + delta`
/// for each named conv), for a single forward/backward step.
pub fn effective_weights(base: &VocoderWeights, adapters: &std::collections::HashMap<String, LoraW>, scale: f32) -> VocoderWeights {
    let mut eff = base.clone();
    for (name, w) in adapters {
        let slot = eff.conv_weight_mut(name).unwrap_or_else(|| panic!("lora: {name:?} is not a conv weight"));
        apply(slot, w, scale);
    }
    eff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{random_weights, Trainer};
    use data::rng::Lcg;

    fn rank2_adapters(cfg: &VocoderConfig, base: &VocoderWeights, seed: u64) -> std::collections::HashMap<String, LoraW> {
        let mut r = Lcg::new(seed);
        conv_weight_names(base)
            .into_iter()
            .map(|name| {
                let (rows, cols) = conv_weight_shape(cfg, &name);
                let w = LoraW::zero_init(rows, cols, 2, |_| r.signed() * 0.1);
                (name, w)
            })
            .collect()
    }

    #[test]
    fn zero_b_is_an_exact_no_op() {
        let cfg = VocoderConfig::tiny();
        let base = random_weights(&cfg, 41);
        let adapters = rank2_adapters(&cfg, &base, 42);
        let eff = effective_weights(&base, &adapters, 1.0);

        for name in conv_weight_names(&base) {
            let want = base.clone().conv_weight_mut(&name).unwrap().clone();
            let got = eff.clone().conv_weight_mut(&name).unwrap().clone();
            assert_eq!(want, got, "{name}: B=0 must leave the weight untouched");
        }
    }

    #[test]
    fn fold_matches_apply_bit_for_bit() {
        // Composing W_eff via `effective_weights` (apply, in place) and via a
        // fresh `base.clone() + delta` (fold, a separate add) must agree
        // exactly - both are the same fp32 arithmetic in the same order.
        let cfg = VocoderConfig::tiny();
        let base = random_weights(&cfg, 51);
        let adapters = rank2_adapters(&cfg, &base, 52);
        let applied = effective_weights(&base, &adapters, 0.5);

        let mut folded = base.clone();
        for (name, w) in &adapters {
            let d = delta(w, 0.5);
            let slot = folded.conv_weight_mut(name).unwrap();
            for (s, di) in slot.iter_mut().zip(&d) {
                *s += di;
            }
        }
        for name in conv_weight_names(&base) {
            let a = applied.clone().conv_weight_mut(&name).unwrap().clone();
            let f = folded.clone().conv_weight_mut(&name).unwrap().clone();
            assert_eq!(a, f, "{name}: apply and fold must produce bit-identical weights");
        }
    }

    #[test]
    fn lora_grads_match_finite_differences() {
        let cfg = VocoderConfig::tiny();
        let base = random_weights(&cfg, 61);
        let mut adapters = rank2_adapters(&cfg, &base, 62);
        // Non-zero B, or every gradient here would trivially be zero too.
        for w in adapters.values_mut() {
            for b in w.b.iter_mut() {
                *b = 0.05;
            }
        }
        let scale = 0.7f32;
        let (batch, length) = (1, 4);
        let mut r = Lcg::new(63);
        let latents = r.vec_scaled(batch * cfg.latent_channels as usize * length, 0.5);
        let out_len = length * cfg.upsampling_ratios.iter().product::<u32>() as usize;
        let target = r.vec_scaled(batch * 2 * out_len, 0.5);

        let loss_at = |adapters: &std::collections::HashMap<String, LoraW>| -> f32 {
            let eff = effective_weights(&base, adapters, scale);
            let trainer = Trainer::new(cfg.clone(), &eff, latents.clone(), batch, length, target.clone());
            trainer.loss()
        };

        // Analytic: one forward+backward at the current adapters, converting
        // every conv's dW_eff to (dA, dB).
        let eff = effective_weights(&base, &adapters, scale);
        let trainer = Trainer::new(cfg.clone(), &eff, latents.clone(), batch, length, target.clone());
        trainer.zero_grads();
        let _ = trainer.loss();
        trainer.backward();

        let eps = 5e-3f32;
        let mut checked = 0;
        for (name, w) in &adapters {
            let d_w_eff = trainer.read_grad(name);
            let (da, db) = backward(w, &d_w_eff, scale);

            // One representative index each for A and B is enough here - the
            // conv backward itself (dW_eff) is already gradchecked in
            // train.rs; this proves only the (dA, dB) CONVERSION is correct.
            let mut pa = adapters.clone();
            let base_a0 = pa[name].a[0];
            pa.get_mut(name).unwrap().a[0] = base_a0 + eps;
            let lp = loss_at(&pa);
            pa.get_mut(name).unwrap().a[0] = base_a0 - eps;
            let lm = loss_at(&pa);
            let num_a = (lp - lm) / (2.0 * eps);
            assert!(
                (num_a - da[0]).abs() < 2e-2 + 2e-2 * num_a.abs().max(da[0].abs()),
                "{name}.a[0]: numeric={num_a} analytic={}",
                da[0]
            );

            let mut pb = adapters.clone();
            let base_b0 = pb[name].b[0];
            pb.get_mut(name).unwrap().b[0] = base_b0 + eps;
            let lp = loss_at(&pb);
            pb.get_mut(name).unwrap().b[0] = base_b0 - eps;
            let lm = loss_at(&pb);
            let num_b = (lp - lm) / (2.0 * eps);
            assert!(
                (num_b - db[0]).abs() < 2e-2 + 2e-2 * num_b.abs().max(db[0].abs()),
                "{name}.b[0]: numeric={num_b} analytic={}",
                db[0]
            );
            checked += 1;
        }
        // ::tiny() has 2 upsample stages (1 dec_in_proj + 1 conv_in + 2*(1
        // conv_t1 + 3*2 res-unit convs) + 1 conv_out = 17), not the real
        // checkpoint's 4.
        assert_eq!(checked, 17, "expected every one of ::tiny()'s 17 conv weights to be checked, got {checked}");
    }

    /// The third leg (with `zero_b_is_an_exact_no_op` and
    /// `fold_matches_apply_bit_for_bit`): LoRA-only training - base weights
    /// untouched, only (A, B) updated - must still drive the loss down,
    /// proving the adapter path is trainable end to end, not just locally
    /// gradient-correct.
    #[test]
    fn lora_only_overfits_with_base_frozen() {
        let cfg = VocoderConfig::tiny();
        let base = random_weights(&cfg, 71);
        let base_snapshot: Vec<(String, Vec<f32>)> = conv_weight_names(&base)
            .into_iter()
            .map(|n| (n.clone(), base.clone().conv_weight_mut(&n).unwrap().clone()))
            .collect();
        let mut adapters = rank2_adapters(&cfg, &base, 72);
        let scale = 1.0f32;
        let (batch, length) = (1, 4);
        let mut r = Lcg::new(73);
        let latents = r.vec_scaled(batch * cfg.latent_channels as usize * length, 0.5);
        let out_len = length * cfg.upsampling_ratios.iter().product::<u32>() as usize;
        let target = r.vec_scaled(batch * 2 * out_len, 0.5);
        let lr = 0.1f32;

        let loss_at = |adapters: &std::collections::HashMap<String, LoraW>| -> f32 {
            let eff = effective_weights(&base, adapters, scale);
            let trainer = Trainer::new(cfg.clone(), &eff, latents.clone(), batch, length, target.clone());
            trainer.loss()
        };

        let loss0 = loss_at(&adapters);
        let mut loss = loss0;
        for _ in 0..1500 {
            let eff = effective_weights(&base, &adapters, scale);
            let trainer = Trainer::new(cfg.clone(), &eff, latents.clone(), batch, length, target.clone());
            trainer.zero_grads();
            loss = trainer.loss();
            trainer.backward();
            for (name, w) in adapters.iter_mut() {
                let d_w_eff = trainer.read_grad(name);
                let (da, db) = backward(w, &d_w_eff, scale);
                for (ai, dai) in w.a.iter_mut().zip(&da) {
                    *ai -= lr * dai;
                }
                for (bi, dbi) in w.b.iter_mut().zip(&db) {
                    *bi -= lr * dbi;
                }
            }
        }
        // A rank-2 adapter has far fewer trainable parameters than a full
        // fine-tune, so this is a much looser bar than
        // `Trainer::overfits_a_single_batch`'s - a real, honest limitation
        // of low-rank adapters, not a bug.
        assert!(loss < loss0 * 0.6, "LoRA-only training did not reduce loss enough: start={loss0} end={loss} (1500 steps, lr={lr})");

        for (name, snapshot) in &base_snapshot {
            let now = base.clone().conv_weight_mut(name).unwrap().clone();
            assert_eq!(snapshot, &now, "{name}: base weight must stay frozen during LoRA-only training");
        }
    }
}
