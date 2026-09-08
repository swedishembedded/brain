// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for [`crate::lmgrad`]'s host Qwen2-style LM
//! reference, over the generic `model::adapter` substrate.
//!
//! ## Why `model::lora::Pair`, not `qwen3::lora`/`qwen3::LoraCfg`
//!
//! The plan for this workstream was to wire `crate::llm::CosyVoiceLm` to
//! `qwen3::lora` directly, the way `qwen3`'s own device-resident `Model`
//! (`Qwen::lora_fwd`/`Qwen::lora_for`) applies it during a batched forward.
//! That is not reachable here: `crate::lmgrad` is a fresh host reference (see
//! its module doc for why), not `qwen3::Qwen`'s own training graph, so there
//! is no `qwen3::Qwen` instance in this crate's training path for
//! `qwen3::lora`'s device-adapter machinery (`.lora_a`/`.lora_b` tensors in a
//! live `ParamStore`) to attach to.
//!
//! What this module reuses instead is the OTHER LoRA family already shared
//! across this workspace's host training references: `model::lora::Pair`,
//! the same `W_eff = W + (α/r)·B·A` host adapter `wan::lora`/`flux2::lora`/
//! `s3dit::lora`/`supir::lora` build on, wearing the generic
//! `model::adapter::AdapterKind` seam via `model::lora::LoraPair`.
//!
//! Targets: `wq`/`wk`/`wv`/`wo` per layer - the same four projections
//! `qwen3::LoraCfg::attn` targets by default, so a rank/alpha choice means the
//! same thing here as it does for `qwen3`-hosted models in this workspace.
//! The MLP (`gate`/`up`/`down`) and the embedding/decoder tables are left to
//! full fine-tune (see `crate::lmgrad`'s own full set of trainable tensors) -
//! LoRA's premise is a low-rank correction to a big square-ish attention
//! projection, and CosyVoice's speech vocabulary is exactly the kind of
//! architecture-specific table a low-rank update does not suit well (a
//! handful of tokens would need to move by a large amount each, which a
//! shared rank-`r` factor does not represent efficiently).
//!
//! [`crate::lmgrad::LmWeights`] is a TYPED struct (`layers: Vec<LayerW<T>>`
//! with named `wq`/`wk`/`wv`/`wo` fields), not a name-keyed map, so unlike
//! `supir::lora`'s `fold_into` over a `HashMap`, [`LmLora::apply`]/
//! [`LmLora::step`] match each [`model::adapter::LinearSite::leaf`] against
//! the base struct's field by hand - the same name-to-field seam
//! `minimaxmusic3::depth_lora::read_layer_grad` already uses for its own
//! typed grads struct.

use crate::lmgrad::{Fp, LmDims, LmGrads, LmWeights};
pub use model::lora::LoraCfg;
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
use model::lora::{LoraGrads, LoraPair};

/// Every linear this crate offers to a PEFT adapter: `wq`/`wk`/`wv`/`wo` per
/// layer, in that order, matching the pre-migration `LayerLora` field order
/// (and its random-init draw order) exactly.
pub fn linear_sites(d: &LmDims) -> Vec<LinearSite> {
    let (dm, hq, hkv) = (d.d_model, d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim);
    let mut sites = Vec::with_capacity(d.n_layers * 4);
    for l in 0..d.n_layers {
        sites.push(LinearSite { name: format!("layers.{l}.wq"), leaf: "wq", layer: Some(l), spec: TargetSpec::whole(hq, dm) });
        sites.push(LinearSite { name: format!("layers.{l}.wk"), leaf: "wk", layer: Some(l), spec: TargetSpec::whole(hkv, dm) });
        sites.push(LinearSite { name: format!("layers.{l}.wv"), leaf: "wv", layer: Some(l), spec: TargetSpec::whole(hkv, dm) });
        sites.push(LinearSite { name: format!("layers.{l}.wo"), leaf: "wo", layer: Some(l), spec: TargetSpec::whole(dm, hq) });
    }
    sites
}

fn field_mut<'a, T>(w: &'a mut LmWeights<T>, layer: usize, leaf: &str) -> &'a mut Vec<T> {
    let l = &mut w.layers[layer];
    match leaf {
        "wq" => &mut l.wq,
        "wk" => &mut l.wk,
        "wv" => &mut l.wv,
        "wo" => &mut l.wo,
        other => panic!("cosyvoice lmlora: unknown leaf {other:?}"),
    }
}

fn field<'a, T>(w: &'a LmWeights<T>, layer: usize, leaf: &str) -> &'a Vec<T> {
    let l = &w.layers[layer];
    match leaf {
        "wq" => &l.wq,
        "wk" => &l.wk,
        "wv" => &l.wv,
        "wo" => &l.wo,
        other => panic!("cosyvoice lmlora: unknown leaf {other:?}"),
    }
}

/// A LoRA adapter over every layer of [`crate::lmgrad`]'s LM.
pub struct LmLora {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
    n_layers: usize,
}

impl LmLora {
    /// Fresh adapter sized for `d`. `B = 0`, so [`Self::apply`] returns
    /// weights bit-identical to the base - this module's own
    /// `lora_is_an_exact_no_op_at_init` test asserts this rather than
    /// assumes it (the top-level `tests/lm_overfit.rs` in this crate does
    /// not reference this module at all).
    pub fn new(d: &LmDims, lc: LoraCfg) -> LmLora {
        let sites = linear_sites(d);
        let hp = TargetHp::from(lc);
        let mut seed = lc.seed ^ 0x434F_5359_564F_4943; // "COSYVOIC"
        let mut init = move || (model::lora::randn(&mut seed) * 0.02) as f32;
        let set = AdapterSet::build(sites, hp, KeyStyle::Brain, &mut init);
        LmLora { set, hp, n_layers: d.n_layers }
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }
    pub fn scale(&self) -> f32 {
        self.hp.scale()
    }
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Effective weights `W_eff = W + scale·B·A` on `wq/wk/wv/wo`; every other
    /// tensor (embeddings, norms, MLP, decoder head) passes through frozen -
    /// base is cloned, never mutated.
    pub fn apply(&self, base: &LmWeights<f32>) -> LmWeights<f32> {
        let mut w = base.clone();
        for (site, k) in self.set.iter() {
            let layer = site.layer.expect("cosyvoice lmlora: every site has a layer index");
            k.delta_into(1.0, field_mut(&mut w, layer, site.leaf));
        }
        w
    }

    /// One optimisation step: project the trainer's `dL/dW_eff` (from a
    /// forward on [`Self::apply`]'s own output) onto `(dA, dB)` per targeted
    /// linear and Adam-step them. The base itself never moves.
    pub fn step(&mut self, base_grads: &LmGrads<f32>, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let layer = site.layer.expect("cosyvoice lmlora: every site has a layer index");
                k.project(field(base_grads, layer, site.leaf))
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }
}

/// Sanity helper for tests: every targeted tensor's `(A, B)` shapes match the
/// base layer's own projection shapes.
pub fn shapes_match<T: Fp>(d: &LmDims, lora: &LmLora, base: &LmWeights<T>) -> bool {
    let (dm, hq, hkv) = (d.d_model, d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim);
    lora.set.iter().all(|(site, k)| {
        let layer = site.layer.expect("layer index");
        let spec = k.spec();
        let (want_out, want_inn) = match site.leaf {
            "wq" => (hq, dm),
            "wk" => (hkv, dm),
            "wv" => (hkv, dm),
            "wo" => (dm, hq),
            other => panic!("unknown leaf {other:?}"),
        };
        spec.out == want_out && spec.inn == want_inn && field(base, layer, site.leaf).len() == want_out * want_inn
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmgrad::{grads, init_weights, Example};

    fn tiny_example(d: &LmDims) -> Example {
        Example { text_ids: vec![2, 4, 1], special_sos: 0, special_task: if d.special_vocab > 0 { 1 } else { d.speech_vocab - 2 }, speech_tokens: vec![1, 3, 5, 2] }
    }

    /// `linear_sites` must enumerate exactly `wq,wk,wv,wo` per layer, in that
    /// order - the pre-migration draw order every existing seed depends on.
    #[test]
    fn linear_sites_enumerates_four_leaves_per_layer_in_order() {
        let d = LmDims::tiny();
        let sites = linear_sites(&d);
        assert_eq!(sites.len(), d.n_layers * 4);
        for (l, chunk) in sites.chunks(4).enumerate() {
            let leaves: Vec<&str> = chunk.iter().map(|s| s.leaf).collect();
            assert_eq!(leaves, ["wq", "wk", "wv", "wo"]);
            assert!(chunk.iter().all(|s| s.layer == Some(l)));
        }
    }

    #[test]
    fn lora_is_an_exact_no_op_at_init() {
        let d = LmDims::tiny();
        let base = init_weights::<f32>(&d, 7);
        let lora = LmLora::new(&d, LoraCfg::new(4));
        assert!(shapes_match(&d, &lora, &base));
        let applied = lora.apply(&base);
        assert!(applied == base, "a fresh LoRA adapter (B=0) must not change a single weight");
    }

    #[test]
    fn lora_training_descends_with_the_base_frozen() {
        let d = LmDims::tiny();
        let base = init_weights::<f32>(&d, 11);
        let mut lora = LmLora::new(&d, LoraCfg::new(4));
        let ex = tiny_example(&d);

        let (l0, _) = grads(&d, &lora.apply(&base), &ex);
        let mut last = l0;
        for _ in 0..120 {
            let w_eff = lora.apply(&base);
            let (l, g) = grads(&d, &w_eff, &ex);
            lora.step(&g, 5e-3);
            last = l;
        }
        assert!(last < l0 * 0.9, "LoRA training must descend: {l0} -> {last}");

        let base_again = init_weights::<f32>(&d, 11);
        assert!(base == base_again, "the base weights must never move during LoRA training");
    }
}
