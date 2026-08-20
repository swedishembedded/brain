// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA fine-tuning for the flow-matching DiT's 6 per-block linear
//! projections (`attn.{to_q,to_k,to_v,to_out}`, `ff_in.weight`,
//! `ff_out.weight` - the fused gated-FFN's up/gate-in and down-out
//! projections). Reuses `crate::lora`'s adapter math (`LoraW`/`delta`/
//! `apply`/`backward`) unchanged - that math only ever needed a flat
//! `[rows, cols]` weight and never referenced any component's own types, so
//! it is exactly as applicable here as it already is to the vocoder's convs
//! and the depth decoder's linears.
//!
//! Fold-then-run, same as the vocoder (`crate::lora`) and unlike the depth
//! decoder (`crate::depth_lora`, which wraps pure host functions): the DiT's
//! own `dit_train::Trainer` is device-resident, so each step composes
//! `W_eff = W_base + (alpha/r)*B@A` on the host and hands the folded
//! `DitWeights` to a FRESH `dit_train::Trainer`, which runs its ordinary
//! device-dispatched forward/backward completely unaware an adapter exists.
//! `W_base` itself is never written back, so it stays frozen across steps.

use crate::config::DitConfig;
use crate::dit::DitWeights;
use crate::lora::LoraW;
use std::collections::HashMap;

/// `(rows, cols)` of one of the DiT's 6 LoRA-eligible per-block linear
/// weights, matching `DitWeights::linear_mut`'s own naming and
/// `model::block`'s `matmul` convention (`out = x @ W^T`, `W: [out, in]`).
pub fn linear_shape(cfg: &DitConfig, name: &str) -> (usize, usize) {
    let inner = cfg.inner_dim() as usize;
    let ff_inner = cfg.ff_inner_dim as usize;
    let rest = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).map(|(_, rest)| rest).unwrap_or(name);
    match rest {
        "attn.to_q" | "attn.to_k" | "attn.to_v" | "attn.to_out" => (inner, inner),
        "ff_in.weight" => (2 * ff_inner, inner),
        "ff_out.weight" => (inner, ff_inner),
        _ => panic!("dit_lora::linear_shape: unknown linear name {name:?}"),
    }
}

/// `base` with every adapter in `adapters` applied (`W_eff = base + delta`
/// for each named linear weight), for a single forward/backward step.
pub fn effective_weights(base: &DitWeights, adapters: &HashMap<String, LoraW>, scale: f32) -> DitWeights {
    let mut eff = base.clone();
    for (name, w) in adapters {
        let slot = eff.linear_mut(name).unwrap_or_else(|| panic!("dit_lora: {name:?} is not a linear weight"));
        crate::lora::apply(slot, w, scale);
    }
    eff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit_train;
    use data::rng::Lcg;

    fn rank2_adapters(cfg: &DitConfig, base: &DitWeights, seed: u64) -> HashMap<String, LoraW> {
        let mut r = Lcg::new(seed);
        base.linear_names()
            .into_iter()
            .map(|name| {
                let (rows, cols) = linear_shape(cfg, &name);
                let w = LoraW::zero_init(rows, cols, 2, |_| r.signed() * 0.1);
                (name, w)
            })
            .collect()
    }

    #[test]
    fn zero_b_is_an_exact_no_op() {
        let cfg = DitConfig::tiny();
        let base = dit_train::random_weights(&cfg, 41);
        let adapters = rank2_adapters(&cfg, &base, 42);
        let eff = effective_weights(&base, &adapters, 1.0);

        for name in base.linear_names() {
            let want = base.clone().linear_mut(&name).unwrap().clone();
            let got = eff.clone().linear_mut(&name).unwrap().clone();
            assert_eq!(want, got, "{name}: B=0 must leave the weight untouched");
        }
    }

    #[test]
    fn fold_matches_apply_bit_for_bit() {
        // Composing W_eff via `effective_weights` (apply, in place) and via a
        // fresh `base.clone() + delta` (fold, a separate add) must agree
        // exactly - both are the same fp32 arithmetic in the same order.
        let cfg = DitConfig::tiny();
        let base = dit_train::random_weights(&cfg, 51);
        let adapters = rank2_adapters(&cfg, &base, 52);
        let applied = effective_weights(&base, &adapters, 0.5);

        let mut folded = base.clone();
        for (name, w) in &adapters {
            let d = crate::lora::delta(w, 0.5);
            let slot = folded.linear_mut(name).unwrap();
            for (s, di) in slot.iter_mut().zip(&d) {
                *s += di;
            }
        }
        for name in base.linear_names() {
            let a = applied.clone().linear_mut(&name).unwrap().clone();
            let f = folded.clone().linear_mut(&name).unwrap().clone();
            assert_eq!(a, f, "{name}: apply and fold must produce bit-identical weights");
        }
    }

    #[test]
    fn lora_grads_match_finite_differences() {
        let cfg = DitConfig::tiny();
        let base = dit_train::random_weights(&cfg, 61);
        let mut adapters = rank2_adapters(&cfg, &base, 62);
        // Non-zero B, or every gradient here would trivially be zero too.
        for w in adapters.values_mut() {
            for b in w.b.iter_mut() {
                *b = 0.05;
            }
        }
        let scale = 0.7f32;
        let length = 3usize;
        let mut r = Lcg::new(63);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
        let timestep = 0.4f32;
        let target = r.vec_scaled(cfg.in_channels as usize * length, 0.3);

        let loss_at = |adapters: &HashMap<String, LoraW>| -> f32 {
            let eff = effective_weights(&base, adapters, scale);
            let trainer = dit_train::Trainer::new(cfg, &eff, latents.clone(), condition.clone(), timestep, length, target.clone());
            trainer.loss()
        };

        // Analytic: one forward+backward at the current adapters, converting
        // every linear's dW_eff to (dA, dB).
        let eff = effective_weights(&base, &adapters, scale);
        let trainer = dit_train::Trainer::new(cfg, &eff, latents.clone(), condition.clone(), timestep, length, target.clone());
        trainer.zero_grads();
        let _ = trainer.loss();
        trainer.backward();

        let eps = 5e-3f32;
        let mut checked = 0;
        for (name, w) in &adapters {
            let d_w_eff = trainer.read_grad(name);
            let (da, db) = crate::lora::backward(w, &d_w_eff, scale);

            // One representative index each for A and B is enough here - the
            // DiT's own backward (dW_eff) is already gradchecked in
            // dit_train.rs; this proves only the (dA, dB) CONVERSION is correct.
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
        // ::tiny() has 2 layers * 6 LoRA-eligible linears per block = 12.
        assert_eq!(checked, DitConfig::tiny().num_layers as usize * 6, "expected every one of ::tiny()'s linear weights to be checked, got {checked}");
    }

    /// The third leg (with `zero_b_is_an_exact_no_op` and
    /// `fold_matches_apply_bit_for_bit`): LoRA-only training - base weights
    /// untouched, only (A, B) updated - must still drive the loss down,
    /// proving the adapter path is trainable end to end, not just locally
    /// gradient-correct.
    #[test]
    fn lora_only_overfits_with_base_frozen() {
        let cfg = DitConfig::tiny();
        let base = dit_train::random_weights(&cfg, 71);
        let base_snapshot: Vec<(String, Vec<f32>)> = base.linear_names().into_iter().map(|n| (n.clone(), base.clone().linear_mut(&n).unwrap().clone())).collect();
        let mut adapters = rank2_adapters(&cfg, &base, 72);
        let scale = 1.0f32;
        let length = 3usize;
        let mut r = Lcg::new(73);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
        let timestep = 0.4f32;
        let target = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let lr = 0.3f32;

        let loss_at = |adapters: &HashMap<String, LoraW>| -> f32 {
            let eff = effective_weights(&base, adapters, scale);
            let trainer = dit_train::Trainer::new(cfg, &eff, latents.clone(), condition.clone(), timestep, length, target.clone());
            trainer.loss()
        };

        let loss0 = loss_at(&adapters);
        let mut loss = loss0;
        for _ in 0..1500 {
            let eff = effective_weights(&base, &adapters, scale);
            let trainer = dit_train::Trainer::new(cfg, &eff, latents.clone(), condition.clone(), timestep, length, target.clone());
            trainer.zero_grads();
            loss = trainer.loss();
            trainer.backward();
            for (name, w) in adapters.iter_mut() {
                let d_w_eff = trainer.read_grad(name);
                let (da, db) = crate::lora::backward(w, &d_w_eff, scale);
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
        // `dit_train::Trainer`'s own `overfits_a_single_batch` - a real,
        // honest limitation of low-rank adapters, not a bug.
        assert!(loss < loss0 * 0.6, "LoRA-only training did not reduce loss enough: start={loss0} end={loss} (1500 steps, lr={lr})");

        for (name, snapshot) in &base_snapshot {
            let now = base.clone().linear_mut(name).unwrap().clone();
            assert_eq!(snapshot, &now, "{name}: base weight must stay frozen during LoRA-only training");
        }
    }
}
