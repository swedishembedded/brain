// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the Z-Image DiT, over the generic
//! `model::adapter` substrate.
//!
//! Each targeted linear `W [out×in]` (the per-block `wq/wk/wv/wo/w1/w2/w3`) gets
//! `W_eff = W + (α/r)·B·A` with `A [r×in]`, `B [out×r]`. The **base is frozen**;
//! only `A,B` train. We reuse the gradchecked fp32 trainer unchanged: rebuild the
//! effective weights, run its forward+backward to get `dL/dW_eff` for each linear,
//! then *project* to the adapter grads
//!   `dA = (α/r)·Bᵀ·dW`,   `dB = (α/r)·dW·Aᵀ`.
//! Only `A,B` get Adam state, so a rank-16 adapter is tiny (~MBs) next to the 6B
//! base — the efficient personalisation path. Validated by `tests/lora_train.rs`
//! (base frozen, LoRA-only overfit drives the loss down; adapter save/load
//! round-trips).
//!
//! [`ModelWeightsF32`]/[`ModelGradsF32`] are TYPED structs (`main: Vec<{Weights,
//! Grads}F32>` with named `wq`/`wk`/.../`w3` fields), not name-keyed maps, so
//! [`LoraAdapter::apply`]/[`LoraAdapter::step`] match each
//! [`model::adapter::LinearSite::leaf`] against the base struct's field by
//! hand. [`LoraAdapter::fold_into_comfy`] targets a THIRD, unrelated naming
//! (the diffusers-style `layers.{l}.attention.to_q.weight` a released comfy
//! checkpoint uses) via its own small leaf table - kept separate rather than
//! forced through [`model::adapter::AdapterSet::fold_into`], because the two
//! namespaces (`blocks.{l}.{leaf}` for this adapter's own serialization vs.
//! `layers.{l}.{comfy_leaf}.weight` for the inference fold target) are
//! genuinely different keys for the same tensor, not a naming-convention
//! difference [`model::adapter::KeyStyle`] is meant to cover.

use crate::grad::{GradsF32, WeightsF32};
use crate::modelgrad::{Cfg, ModelGradsF32, ModelWeightsF32};
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
// The generic pair machinery (A/B init, ΔW apply, dW→(dA,dB) projection, Adam
// moments) is model-agnostic and lives ONCE in `model::lora` — this module
// keeps only the Z-Image-specific block walk and serialization naming.
// `LoraCfg` is re-exported for existing callers.
pub use model::lora::LoraCfg;
use model::lora::{LoraGrads, LoraPair};

/// The seven leaves targeted per block, `(leaf, out, inn)` given `dim`/`hidden`.
fn leaf_shapes(dim: usize, hidden: usize) -> [(&'static str, usize, usize); 7] {
    [("wq", dim, dim), ("wk", dim, dim), ("wv", dim, dim), ("wo", dim, dim), ("w1", hidden, dim), ("w2", dim, hidden), ("w3", hidden, dim)]
}

/// The comfy-layout inference key for one block's leaf - a DIFFERENT
/// namespace from this adapter's own `blocks.{l}.{leaf}` serialization.
fn comfy_key(layer: usize, leaf: &str) -> String {
    let comfy_leaf = match leaf {
        "wq" => "attention.to_q.weight",
        "wk" => "attention.to_k.weight",
        "wv" => "attention.to_v.weight",
        "wo" => "attention.to_out.0.weight",
        "w1" => "feed_forward.w1.weight",
        "w2" => "feed_forward.w2.weight",
        "w3" => "feed_forward.w3.weight",
        other => panic!("s3dit lora: unknown leaf {other:?}"),
    };
    format!("layers.{layer}.{comfy_leaf}")
}

/// Every linear this crate offers to a PEFT adapter: the seven leaves of
/// every `main` block, in that order - the pre-migration `BlockLora` field
/// order (and its random-init draw order).
pub fn linear_sites(cfg: &Cfg) -> Vec<LinearSite> {
    let hidden = cfg.dim * 8 / 3;
    let mut sites = Vec::with_capacity(cfg.n_layers * 7);
    for l in 0..cfg.n_layers {
        for (leaf, out, inn) in leaf_shapes(cfg.dim, hidden) {
            sites.push(LinearSite { name: format!("blocks.{l}.{leaf}"), leaf, layer: Some(l), spec: TargetSpec::whole(out, inn), save_name: None });
        }
    }
    sites
}

fn field_mut<'a>(w: &'a mut WeightsF32, leaf: &str) -> &'a mut Vec<f32> {
    match leaf {
        "wq" => &mut w.wq,
        "wk" => &mut w.wk,
        "wv" => &mut w.wv,
        "wo" => &mut w.wo,
        "w1" => &mut w.w1,
        "w2" => &mut w.w2,
        "w3" => &mut w.w3,
        other => panic!("s3dit lora: unknown leaf {other:?}"),
    }
}

fn field<'a>(g: &'a GradsF32, leaf: &str) -> &'a Vec<f32> {
    match leaf {
        "wq" => &g.wq,
        "wk" => &g.wk,
        "wv" => &g.wv,
        "wo" => &g.wo,
        "w1" => &g.w1,
        "w2" => &g.w2,
        "w3" => &g.w3,
        other => panic!("s3dit lora: unknown leaf {other:?}"),
    }
}

/// A LoRA adapter over all `main` blocks of the DiT.
pub struct LoraAdapter {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
}

impl LoraAdapter {
    /// Fresh adapter (B=0 → initial no-op) sized for `cfg`, over `cfg.n_layers`
    /// main blocks. Targets attention (`wq/wk/wv/wo`) and MLP (`w1/w2/w3`).
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Same init distribution as before the model::lora hoist (gaussian,
        // σ 0.02) so existing seeds reproduce bit-identical adapters.
        let mut init = move || (model::lora::randn(&mut rng) * 0.02) as f32;
        let set = AdapterSet::build(sites, hp, KeyStyle::Brain, &mut init);
        LoraAdapter { set, hp }
    }

    /// Build the effective weights `W_eff = W + scale·B·A` (base cloned, adapters
    /// added onto each targeted `main` linear).
    pub fn apply(&self, base: &ModelWeightsF32) -> ModelWeightsF32 {
        let mut w = base.clone();
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("s3dit lora: every site has a layer index");
            k.delta_into(1.0, field_mut(&mut w.main[l], site.leaf));
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads to adapter
    /// grads and Adam-update `A,B`. `grads` is `dL/dW_eff` from the frozen-base
    /// forward on the current `apply()`ed weights.
    pub fn step(&mut self, grads: &ModelGradsF32, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let l = site.layer.expect("s3dit lora: every site has a layer index");
                k.project(field(&grads.main[l], site.leaf))
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }

    /// Serialise to `(name, shape, data)` tensors — `blocks.{l}.{lin}.lora_{a,b}`.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }

    pub fn alpha(&self) -> f32 {
        self.hp.alpha
    }

    /// Fold this adapter's deltas into an **inference** tensor map (the
    /// `import_comfy` layout the generation path builds from), so a plain
    /// `text2image` produces adapter-conditioned images with no model change.
    /// Each main block `l`'s `W += (α/r)·B·A` is added onto the matching
    /// `layers.{l}.{…}.weight`. Refiner blocks are not adapted (the adapter only
    /// targets the main layers). Errors if a targeted tensor is absent.
    pub fn fold_into_comfy(&self, t: &mut crate::block::Tensors) -> Result<(), String> {
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("s3dit lora: every site has a layer index");
            let key = comfy_key(l, site.leaf);
            let w = t.get_mut(&key).ok_or_else(|| format!("lora: base tensor {key} missing"))?;
            let spec = k.spec();
            if w.1.len() != spec.out * spec.inn {
                return Err(format!("lora: {key} is {} elems, adapter expects {}", w.1.len(), spec.out * spec.inn));
            }
            k.delta_into(1.0, &mut w.1);
        }
        Ok(())
    }

    /// Reload an adapter (weights only; Adam state reset) from `to_tensors`
    /// output — a fresh adapter of the right shape with `A,B` overwritten.
    pub fn from_tensors(cfg: &Cfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let set = AdapterSet::from_tensors(sites, hp, KeyStyle::Brain, tensors)?;
        Ok(LoraAdapter { set, hp })
    }
}

/// Convenience: are the block linears of `w` the expected shapes for `cfg`?
/// (Guards `apply`/`step` against a mismatched base.)
pub fn check_shapes(cfg: &Cfg, w: &WeightsF32) -> Result<(), String> {
    let (dim, hidden) = (cfg.dim, cfg.dim * 8 / 3);
    let want = [
        ("wq", dim * dim, w.wq.len()), ("w1", hidden * dim, w.w1.len()), ("w2", dim * hidden, w.w2.len()),
    ];
    for (n, e, g) in want {
        if e != g {
            return Err(format!("lora: base linear {n} is {g}, expected {e}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        Cfg { dim: 8, nh: 2, n_layers: 2, n_refiner: 1, cap_feat_dim: 4, in_channels: 4, patch: 2, h: 4, w: 4, ncap: 3, t_scale: 1000.0 }
    }

    /// `linear_sites` must enumerate the same seven leaves per block, in the
    /// same order, as the pre-migration `BlockLora` struct did - the order
    /// every existing seed's random-init draw sequence depends on.
    #[test]
    fn linear_sites_enumerates_seven_leaves_per_block_in_order() {
        let cfg = cfg();
        let sites = linear_sites(&cfg);
        assert_eq!(sites.len(), cfg.n_layers * 7);
        for (l, chunk) in sites.chunks(7).enumerate() {
            let leaves: Vec<&str> = chunk.iter().map(|s| s.leaf).collect();
            assert_eq!(leaves, ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]);
            assert!(chunk.iter().all(|s| s.layer == Some(l)));
            assert!(chunk.iter().all(|s| s.name == format!("blocks.{l}.{}", s.leaf)));
        }
    }

    /// `to_tensors`/`from_tensors` and `fold_into_comfy` must use two
    /// genuinely different namespaces for the same tensor, per the module
    /// doc - assert both concretely rather than just describing it.
    #[test]
    fn serialization_and_comfy_fold_use_different_namespaces() {
        assert_eq!(comfy_key(3, "wq"), "layers.3.attention.to_q.weight");
        let ad = LoraAdapter::new(&cfg(), LoraCfg::new(2));
        let tensors = ad.to_tensors();
        assert!(tensors.iter().any(|(name, _, _)| name == "blocks.0.wq.lora_a"));
        assert!(!tensors.iter().any(|(name, _, _)| name.starts_with("layers.")));
    }

    /// `B = 0` at construction: `apply` must reproduce the base bit-exactly.
    #[test]
    fn fresh_adapter_is_a_bit_exact_no_op() {
        let cfg = cfg();
        let hidden = cfg.dim * 8 / 3;
        let block = |v: f32| crate::grad::WeightsF32 {
            wq: vec![v; cfg.dim * cfg.dim],
            wk: vec![v; cfg.dim * cfg.dim],
            wv: vec![v; cfg.dim * cfg.dim],
            wo: vec![v; cfg.dim * cfg.dim],
            w1: vec![v; hidden * cfg.dim],
            w2: vec![v; cfg.dim * hidden],
            w3: vec![v; hidden * cfg.dim],
            nq: vec![1.0; cfg.dim / cfg.nh],
            nk: vec![1.0; cfg.dim / cfg.nh],
            an1: vec![1.0; cfg.dim],
            an2: vec![1.0; cfg.dim],
            fn1: vec![1.0; cfg.dim],
            fn2: vec![1.0; cfg.dim],
            adaln_w: vec![0.0; 4 * cfg.dim * cfg.dim.min(256)],
            adaln_b: vec![0.0; 4 * cfg.dim],
        };
        let base = ModelWeightsF32 {
            t0_w: vec![],
            t0_b: vec![],
            t2_w: vec![],
            t2_b: vec![],
            xemb_w: vec![],
            xemb_b: vec![],
            capn_w: vec![],
            cap1_w: vec![],
            cap1_b: vec![],
            noise_ref: vec![],
            ctx_ref: vec![],
            main: (0..cfg.n_layers).map(|i| block(i as f32)).collect(),
            fadaln_w: vec![],
            fadaln_b: vec![],
            flin_w: vec![],
            flin_b: vec![],
        };
        let ad = LoraAdapter::new(&cfg, LoraCfg::new(2));
        let applied = ad.apply(&base);
        for (a, b) in base.main.iter().zip(applied.main.iter()) {
            for (x, y) in a.wq.iter().zip(b.wq.iter()) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}
