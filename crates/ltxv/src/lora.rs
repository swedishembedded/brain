// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the video-only LTX DiT, over the generic
//! `model::adapter` substrate.
//!
//! Each targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A` with
//! `A [r×in]`, `B [out×r]`. The **base is frozen**; only `A,B` train. Same
//! scheme as `wan::lora` (which this module mirrors closely): rebuild the
//! effective weights, run the gradchecked host trainer
//! ([`crate::modelgrad::grads`]) to get `dL/dW_eff`, then *project* onto the
//! adapter grads (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) and Adam-step
//! `A,B`. The generic pair machinery lives once in `model::lora`, dispatched
//! through `model::adapter::AdapterKind`; this module keeps only the
//! LTX-specific block walk and serialization naming.
//!
//! ## LTX fuses nothing at this milestone, so there are no fused offsets
//!
//! Like Wan (and unlike FLUX.2/Z-Image's fused `qkv`/`mlp.0`), an LTX block's
//! `attn1.{to_q,to_k,to_v,to_out.0}`, `attn2.{to_q,to_k,to_v,to_out.0}` and
//! `ff.net.{0.proj,2}` are ten independently-named `[out, in]` tensors
//! (`crate::dit::dit_tensor_manifest`), so each pair maps onto a whole
//! tensor at offset 0 ([`model::adapter::TargetSpec::whole`]) and
//! [`LoraAdapter::fold_into_tensors`] reaches inference by name, and
//! `tests/lora_train.rs` asserts fold-vs-apply is **bit-equal** rather than
//! close, the same bar `wan::lora`'s own doc explains.
//!
//! ## Key layout: ComfyUI, not this crate's own bare manifest names
//!
//! The adapter's OWN saved/loaded representation
//! ([`LoraAdapter::to_tensors`]/[`LoraAdapter::from_tensors`]) uses the
//! ComfyUI convention: `diffusion_model.<module path>.lora_A.weight` /
//! `.lora_B.weight` (capital `A`/`B` - the diffusers/ComfyUI spelling, NOT
//! `wan::lora`'s own lowercase `.lora_a`/`.lora_b`, a genuinely different
//! ecosystem convention this port is asked to match) - expressed here as
//! [`model::adapter::KeyStyle::Peft`] with prefix `"diffusion_model."`.
//! `<module path>` is `crate::dit::dit_tensor_manifest`'s own tensor path
//! MINUS the trailing `.weight` (e.g. `transformer_blocks.0.attn1.to_q`).
//! This is purely about how the ADAPTER file names its own tensors;
//! [`LoraAdapter::fold_into_tensors`] still targets the base model's OWN
//! bare tensor keys (no `diffusion_model.` prefix) when folding into
//! `crate::dit::LtxDit`'s inference tensor map - exactly how a real ComfyUI
//! loader matches an adapter key to a base key by stripping the
//! `diffusion_model.` prefix and the `.lora_{A,B}.weight` suffix.
//!
//! Biases, the QK-norm gains, and the whole conditioning path
//! (`scale_shift_table`, `prompt_scale_shift_table`, `adaln_single.*`) are
//! deliberately NOT adapted: LoRA's premise is a low-rank correction to a
//! big matrix, and those are vectors (or, for `prompt_scale_shift_table`,
//! too small - `2*dim` - for a rank decomposition to make sense).

use crate::grad::{BlockGrads, BlockW};
use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
pub use model::lora::LoraCfg;
use model::lora::{LoraGrads, LoraPair};

/// The checkpoint leaf each pair adapts, in a fixed order - one table so the
/// walk, the serializer and the fold cannot disagree about which tensor is
/// which. `s` = `attn1`/self, `c` = `attn2`/cross, matching `wan::lora`'s own
/// short names for the same shape of block.
const LEAVES: [&str; 10] =
    ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"];

fn field_mut<'a>(b: &'a mut BlockW<f32>, leaf: &str) -> &'a mut Vec<f32> {
    match leaf {
        "attn1.to_q" => &mut b.attn1.q.w,
        "attn1.to_k" => &mut b.attn1.k.w,
        "attn1.to_v" => &mut b.attn1.v.w,
        "attn1.to_out.0" => &mut b.attn1.o.w,
        "attn2.to_q" => &mut b.attn2.q.w,
        "attn2.to_k" => &mut b.attn2.k.w,
        "attn2.to_v" => &mut b.attn2.v.w,
        "attn2.to_out.0" => &mut b.attn2.o.w,
        "ff.net.0.proj" => &mut b.ff1.w,
        "ff.net.2" => &mut b.ff2.w,
        other => panic!("ltxv lora: unknown leaf {other:?}"),
    }
}

fn field<'a>(g: &'a BlockGrads<f32>, leaf: &str) -> &'a Vec<f32> {
    match leaf {
        "attn1.to_q" => &g.attn1.q.w,
        "attn1.to_k" => &g.attn1.k.w,
        "attn1.to_v" => &g.attn1.v.w,
        "attn1.to_out.0" => &g.attn1.o.w,
        "attn2.to_q" => &g.attn2.q.w,
        "attn2.to_k" => &g.attn2.k.w,
        "attn2.to_v" => &g.attn2.v.w,
        "attn2.to_out.0" => &g.attn2.o.w,
        "ff.net.0.proj" => &g.ff1.w,
        "ff.net.2" => &g.ff2.w,
        other => panic!("ltxv lora: unknown leaf {other:?}"),
    }
}

/// Every linear this crate offers to a PEFT adapter: the ten leaves of every
/// transformer block, in [`LEAVES`] order - the pre-migration `BlockLora`
/// field order (and its random-init draw order).
pub fn linear_sites(cfg: &Cfg) -> Vec<LinearSite> {
    let dim = cfg.dim;
    let mut sites = Vec::with_capacity(cfg.num_layers * LEAVES.len());
    for l in 0..cfg.num_layers {
        for leaf in LEAVES {
            let (out, inn) = match leaf {
                "ff.net.0.proj" => (4 * dim, dim),
                "ff.net.2" => (dim, 4 * dim),
                _ => (dim, dim),
            };
            sites.push(LinearSite { name: format!("transformer_blocks.{l}.{leaf}.weight"), leaf, layer: Some(l), spec: TargetSpec::whole(out, inn), save_name: None });
        }
    }
    sites
}

/// A LoRA adapter over every block of the DiT.
pub struct LoraAdapter {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
}

const KEY_STYLE: KeyStyle = KeyStyle::Peft { prefix: "diffusion_model." };

impl LoraAdapter {
    /// Fresh adapter sized for `cfg`. `B = 0`, so it is an **exact no-op at
    /// init** - `apply` returns weights bit-identical to the base, which
    /// `tests/lora_train.rs` asserts rather than assumes.
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution `wan::lora`/`s3dit::lora`
        // use, so a seed means the same thing across models.
        let mut init = move || (model::lora::randn(&mut rng) * 0.02) as f32;
        let set = AdapterSet::build(sites, hp, KEY_STYLE, &mut init);
        LoraAdapter { set, hp }
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }

    pub fn alpha(&self) -> f32 {
        self.hp.alpha
    }

    /// Effective weights `W_eff = W + scale·B·A` (base cloned; every other
    /// tensor - biases, QK-norm gains, the conditioning path - passes
    /// through frozen).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("ltxv lora: every site has a layer index");
            k.delta_into(1.0, field_mut(&mut w.blocks[l], site.leaf));
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads onto
    /// the adapter grads and Adam-update `A,B`. `grads` must be `dL/dW_eff`
    /// from a forward on this adapter's own [`LoraAdapter::apply`] output.
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let l = site.layer.expect("ltxv lora: every site has a layer index");
                k.project(field(&grads.blocks[l], site.leaf))
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }

    /// Serialise to `(name, shape, data)` in the ComfyUI key layout -
    /// `diffusion_model.transformer_blocks.{l}.{leaf}.lora_A/B.weight` - see
    /// this module's doc.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
    }

    /// Reload an adapter (weights only; Adam state resets by design).
    pub fn from_tensors(cfg: &Cfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let set = AdapterSet::from_tensors(sites, hp, KEY_STYLE, tensors)?;
        Ok(LoraAdapter { set, hp })
    }

    /// Fold this adapter into an **inference** tensor map
    /// (`crate::dit::dit_tensor_manifest`'s own bare naming - what
    /// `crate::dit::LtxDit` reads from), so the unchanged generation path
    /// produces adapter-conditioned output. Errors by name if a targeted
    /// tensor is absent or the wrong size.
    pub fn fold_into_tensors(&self, ts: &mut vae::blocks::Tensors) -> Result<(), String> {
        self.set.fold_into(ts, 1.0)
    }
}

/// Save an adapter to brain's checkpoint container, header
/// `{"model":"ltxv-lora","rank":R,"alpha":A}`.
pub fn save_adapter(path: &str, ad: &LoraAdapter) {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> = ad.to_tensors().into_iter().map(|(n, s, d)| (n, s.iter().map(|&x| x as u64).collect(), d)).collect();
    checkpoint::save(path, serde_json::json!({"model": "ltxv-lora", "rank": ad.rank(), "alpha": ad.alpha()}), &t);
}

/// Load an adapter written by [`save_adapter`].
pub fn load_adapter(path: &str, cfg: &Cfg) -> Result<LoraAdapter, String> {
    let c = checkpoint::load(path);
    let rank = c.header["config"]["rank"].as_u64().ok_or("adapter: missing rank in header")? as usize;
    let alpha = c.header["config"]["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let map: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> = c.tensors.into_iter().map(|t| (t.name, (Vec::new(), t.data))).collect();
    LoraAdapter::from_tensors(cfg, LoraCfg { rank, alpha, seed: 0 }, &map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The leaf table must name real checkpoint tensors - a typo here folds
    /// into nothing and trains an adapter that never reaches inference.
    #[test]
    fn every_targeted_leaf_exists_in_the_checkpoint_manifest() {
        let cfg = crate::LtxDitConfig::tiny();
        let names: std::collections::HashSet<String> = crate::dit::dit_tensor_manifest(&cfg).into_iter().map(|(n, _)| n).collect();
        for leaf in LEAVES {
            let key = format!("transformer_blocks.0.{leaf}.weight");
            assert!(names.contains(&key), "adapter targets {key}, which the manifest does not define");
        }
    }

    /// `linear_sites` targets exactly the manifest keys `fold_into_tensors`
    /// will look up, and the adapter's own serialization key style
    /// reproduces the ComfyUI spelling this port is asked to match.
    #[test]
    fn linear_sites_names_match_the_fold_target_and_the_comfy_key_style() {
        let cfg = Cfg::tiny();
        let sites = linear_sites(&cfg);
        assert_eq!(sites.len(), cfg.num_layers * LEAVES.len());
        assert_eq!(sites[0].name, "transformer_blocks.0.attn1.to_q.weight");
        assert_eq!(KEY_STYLE.format(&sites[0].name, ".lora_a"), "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight");
        assert_eq!(KEY_STYLE.format(&sites[0].name, ".lora_b"), "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight");
    }
}
