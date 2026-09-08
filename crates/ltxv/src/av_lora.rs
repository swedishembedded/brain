// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the audio+video LTX DiT - [`crate::lora`]'s
//! AV twin, same scheme (`W_eff = W + (α/r)·B·A`, base frozen, `B = 0` at
//! init so [`LoraAdapter::apply`] is an exact no-op), same generic
//! `model::adapter::AdapterKind` seam over `model::lora::Pair`, same ComfyUI
//! key layout (`diffusion_model.<module path>.lora_A/B.weight`).
//!
//! ## What this adapter targets, and why
//!
//! **28 leaves per block**, three groups:
//!
//! * Video stream's `attn1`/`attn2` q/k/v/o (8) + `ff.net.{0.proj,2}` (2) -
//!   IDENTICAL to [`crate::lora`]'s own 10, same reasoning.
//! * Audio stream's `audio_attn1`/`audio_attn2` q/k/v/o (8) +
//!   `audio_ff.net.{0.proj,2}` (2) - the audio stream's own structural
//!   twin, same 10 leaves at audio's dims.
//! * **The audio<->video cross-attention's q/k/v/o, both directions**
//!   (`audio_to_video_attn`, `video_to_audio_attn`, 4 each = 8) - included
//!   deliberately, not merely for symmetry: this coupling is what makes an
//!   AV LoRA genuinely different from training two independent video-only
//!   and audio-only adapters, and a concept that manifests as a
//!   video/audio correlation (e.g. a sound cued by an on-screen event) can
//!   only be learned through these four attention modules. Their own
//!   `to_gate_logits`/QK-norm/biases are excluded on the same grounds as
//!   every other attention module here (see below).
//!
//! **Excluded, deliberately**: biases, QK-norm gains, every adaLN-single
//! table (model-level AND per-block, video's own, audio's own, and all four
//! AV cross-modal tables) - [`crate::lora`]'s own doc explains why (vectors,
//! or too small for a rank decomposition to make sense); the AV cross
//! block's own `[5,dim]` static tables (`scale_shift_table_a2v_ca_
//! {video,audio}`) fall under the same reasoning. `to_gate_logits` and both
//! embeddings connectors are untouched because they are outside this
//! milestone's TRAINING scope entirely (`crate::av_grad`'s own doc - gated
//! attention has no backward here, and neither connector runs in this
//! forward at `use_embeddings_connector: false`), not a LoRA-specific
//! decision.
//!
//! `LEAVES`' 28 entries have HETEROGENEOUS rectangular shapes across the two
//! modalities (`audio_to_video_attn.to_q` is `[adim, vdim]`, not square) -
//! [`model::adapter::LinearSite`] carries `out`/`inn` as data per site
//! (inside its `spec`), so [`linear_sites`] is the single source of that
//! shape table rather than the two independent hand-maintained 28-element
//! arrays (`apply`'s `targets`, `step`'s `dw`) this module had before.

use crate::av_grad::{AvBlockGrads, AvBlockW};
use crate::av_modelgrad::{AvCfg, AvModelGrads, AvModelWeights};
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
pub use model::lora::LoraCfg;
use model::lora::{LoraGrads, LoraPair};

/// The checkpoint leaf each pair adapts, in a fixed order - one table so the
/// walk, the serializer and the fold cannot disagree about which tensor is
/// which (`crate::lora`'s own doc).
const LEAVES: [&str; 28] = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
];

/// `(out, inn)` for one leaf, given the video/audio widths - the single
/// source of the shape table every one of `LEAVES`' 28 entries used to be
/// hand-copied into (`new`'s `mk(...)` calls, `apply`'s/`step`'s parallel
/// arrays).
fn leaf_shape(leaf: &str, vdim: usize, adim: usize) -> (usize, usize) {
    match leaf {
        "attn1.to_q" | "attn1.to_k" | "attn1.to_v" | "attn1.to_out.0" | "attn2.to_q" | "attn2.to_k" | "attn2.to_v" | "attn2.to_out.0" => (vdim, vdim),
        "ff.net.0.proj" => (4 * vdim, vdim),
        "ff.net.2" => (vdim, 4 * vdim),
        "audio_attn1.to_q" | "audio_attn1.to_k" | "audio_attn1.to_v" | "audio_attn1.to_out.0" | "audio_attn2.to_q" | "audio_attn2.to_k" | "audio_attn2.to_v" | "audio_attn2.to_out.0" => (adim, adim),
        "audio_ff.net.0.proj" => (4 * adim, adim),
        "audio_ff.net.2" => (adim, 4 * adim),
        // audio_to_video_attn: q_dim=vdim, kv_dim=adim, inner=adim - to_q is
        // [adim,vdim], to_out.0 is [vdim,adim]; to_k/to_v are [adim,adim].
        "audio_to_video_attn.to_q" => (adim, vdim),
        "audio_to_video_attn.to_k" => (adim, adim),
        "audio_to_video_attn.to_v" => (adim, adim),
        "audio_to_video_attn.to_out.0" => (vdim, adim),
        // video_to_audio_attn: q_dim=adim, kv_dim=vdim, inner=adim - to_k/
        // to_v are [adim,vdim], everything else [adim,adim].
        "video_to_audio_attn.to_q" => (adim, adim),
        "video_to_audio_attn.to_k" => (adim, vdim),
        "video_to_audio_attn.to_v" => (adim, vdim),
        "video_to_audio_attn.to_out.0" => (adim, adim),
        other => panic!("ltxv av_lora: unknown leaf {other:?}"),
    }
}

fn field_mut<'a>(b: &'a mut AvBlockW<f32>, leaf: &str) -> &'a mut Vec<f32> {
    match leaf {
        "attn1.to_q" => &mut b.v_attn1.q.w,
        "attn1.to_k" => &mut b.v_attn1.k.w,
        "attn1.to_v" => &mut b.v_attn1.v.w,
        "attn1.to_out.0" => &mut b.v_attn1.o.w,
        "attn2.to_q" => &mut b.v_attn2.q.w,
        "attn2.to_k" => &mut b.v_attn2.k.w,
        "attn2.to_v" => &mut b.v_attn2.v.w,
        "attn2.to_out.0" => &mut b.v_attn2.o.w,
        "ff.net.0.proj" => &mut b.v_ff1.w,
        "ff.net.2" => &mut b.v_ff2.w,
        "audio_attn1.to_q" => &mut b.a_attn1.q.w,
        "audio_attn1.to_k" => &mut b.a_attn1.k.w,
        "audio_attn1.to_v" => &mut b.a_attn1.v.w,
        "audio_attn1.to_out.0" => &mut b.a_attn1.o.w,
        "audio_attn2.to_q" => &mut b.a_attn2.q.w,
        "audio_attn2.to_k" => &mut b.a_attn2.k.w,
        "audio_attn2.to_v" => &mut b.a_attn2.v.w,
        "audio_attn2.to_out.0" => &mut b.a_attn2.o.w,
        "audio_ff.net.0.proj" => &mut b.a_ff1.w,
        "audio_ff.net.2" => &mut b.a_ff2.w,
        "audio_to_video_attn.to_q" => &mut b.av.a2v.q.w,
        "audio_to_video_attn.to_k" => &mut b.av.a2v.k.w,
        "audio_to_video_attn.to_v" => &mut b.av.a2v.v.w,
        "audio_to_video_attn.to_out.0" => &mut b.av.a2v.o.w,
        "video_to_audio_attn.to_q" => &mut b.av.v2a.q.w,
        "video_to_audio_attn.to_k" => &mut b.av.v2a.k.w,
        "video_to_audio_attn.to_v" => &mut b.av.v2a.v.w,
        "video_to_audio_attn.to_out.0" => &mut b.av.v2a.o.w,
        other => panic!("ltxv av_lora: unknown leaf {other:?}"),
    }
}

fn field<'a>(g: &'a AvBlockGrads<f32>, leaf: &str) -> &'a Vec<f32> {
    match leaf {
        "attn1.to_q" => &g.v_attn1.q.w,
        "attn1.to_k" => &g.v_attn1.k.w,
        "attn1.to_v" => &g.v_attn1.v.w,
        "attn1.to_out.0" => &g.v_attn1.o.w,
        "attn2.to_q" => &g.v_attn2.q.w,
        "attn2.to_k" => &g.v_attn2.k.w,
        "attn2.to_v" => &g.v_attn2.v.w,
        "attn2.to_out.0" => &g.v_attn2.o.w,
        "ff.net.0.proj" => &g.v_ff1.w,
        "ff.net.2" => &g.v_ff2.w,
        "audio_attn1.to_q" => &g.a_attn1.q.w,
        "audio_attn1.to_k" => &g.a_attn1.k.w,
        "audio_attn1.to_v" => &g.a_attn1.v.w,
        "audio_attn1.to_out.0" => &g.a_attn1.o.w,
        "audio_attn2.to_q" => &g.a_attn2.q.w,
        "audio_attn2.to_k" => &g.a_attn2.k.w,
        "audio_attn2.to_v" => &g.a_attn2.v.w,
        "audio_attn2.to_out.0" => &g.a_attn2.o.w,
        "audio_ff.net.0.proj" => &g.a_ff1.w,
        "audio_ff.net.2" => &g.a_ff2.w,
        "audio_to_video_attn.to_q" => &g.av.a2v.q.w,
        "audio_to_video_attn.to_k" => &g.av.a2v.k.w,
        "audio_to_video_attn.to_v" => &g.av.a2v.v.w,
        "audio_to_video_attn.to_out.0" => &g.av.a2v.o.w,
        "video_to_audio_attn.to_q" => &g.av.v2a.q.w,
        "video_to_audio_attn.to_k" => &g.av.v2a.k.w,
        "video_to_audio_attn.to_v" => &g.av.v2a.v.w,
        "video_to_audio_attn.to_out.0" => &g.av.v2a.o.w,
        other => panic!("ltxv av_lora: unknown leaf {other:?}"),
    }
}

/// Every linear this crate offers to a PEFT adapter: the 28 leaves of every
/// AV block, in [`LEAVES`] order - the pre-migration `AvBlockLora` field
/// order (and its random-init draw order).
pub fn linear_sites(cfg: &AvCfg) -> Vec<LinearSite> {
    let (vdim, adim) = (cfg.vdim, cfg.adim);
    let mut sites = Vec::with_capacity(cfg.num_layers * LEAVES.len());
    for l in 0..cfg.num_layers {
        for leaf in LEAVES {
            let (out, inn) = leaf_shape(leaf, vdim, adim);
            sites.push(LinearSite { name: format!("transformer_blocks.{l}.{leaf}.weight"), leaf, layer: Some(l), spec: TargetSpec::whole(out, inn) });
        }
    }
    sites
}

const KEY_STYLE: KeyStyle = KeyStyle::Peft { prefix: "diffusion_model." };

/// A LoRA adapter over every block of the AV DiT.
pub struct LoraAdapter {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
}

impl LoraAdapter {
    /// Fresh adapter sized for `cfg`. `B = 0`, so it is an **exact no-op at
    /// init** - `apply` returns weights bit-identical to the base, the same
    /// bar `crates/ltxv/tests/av_lora_train.rs` asserts.
    pub fn new(cfg: &AvCfg, lc: LoraCfg) -> LoraAdapter {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution `crate::lora`/
        // `wan::lora`/`s3dit::lora` use, so a seed means the same thing
        // across models.
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
    /// tensor - biases, QK-norm gains, every adaLN table - passes through
    /// frozen).
    pub fn apply(&self, base: &AvModelWeights<f32>) -> AvModelWeights<f32> {
        let mut w = base.clone();
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("ltxv av_lora: every site has a layer index");
            k.delta_into(1.0, field_mut(&mut w.blocks[l], site.leaf));
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads onto
    /// the adapter grads and Adam-update `A,B`. `grads` must be `dL/dW_eff`
    /// from a forward on this adapter's own [`LoraAdapter::apply`] output.
    pub fn step(&mut self, grads: &AvModelGrads<f32>, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let l = site.layer.expect("ltxv av_lora: every site has a layer index");
                k.project(field(&grads.blocks[l], site.leaf))
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }

    /// Serialise to `(name, shape, data)` in the ComfyUI key layout -
    /// `diffusion_model.transformer_blocks.{l}.{leaf}.lora_A/B.weight` -
    /// same convention `crate::lora`'s own doc explains.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
    }

    /// Reload an adapter (weights only; Adam state resets by design).
    pub fn from_tensors(cfg: &AvCfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let set = AdapterSet::from_tensors(sites, hp, KEY_STYLE, tensors)?;
        Ok(LoraAdapter { set, hp })
    }

    /// Fold this adapter into an **inference** tensor map (`crate::dit::
    /// av_dit_tensor_manifest`'s own bare naming), so the unchanged
    /// generation path produces adapter-conditioned output.
    pub fn fold_into_tensors(&self, ts: &mut vae::blocks::Tensors) -> Result<(), String> {
        self.set.fold_into(ts, 1.0)
    }
}

/// Save an adapter to brain's checkpoint container, header
/// `{"model":"ltxv-av-lora","rank":R,"alpha":A}`.
pub fn save_adapter(path: &str, ad: &LoraAdapter) {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> = ad.to_tensors().into_iter().map(|(n, s, d)| (n, s.iter().map(|&x| x as u64).collect(), d)).collect();
    checkpoint::save(path, serde_json::json!({"model": "ltxv-av-lora", "rank": ad.rank(), "alpha": ad.alpha()}), &t);
}

/// Load an adapter written by [`save_adapter`].
pub fn load_adapter(path: &str, cfg: &AvCfg) -> Result<LoraAdapter, String> {
    let c = checkpoint::load(path);
    let rank = c.header["config"]["rank"].as_u64().ok_or("av adapter: missing rank in header")? as usize;
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
        let cfg = crate::LtxAvDitConfig::tiny();
        let names: std::collections::HashSet<String> = crate::dit::av_dit_tensor_manifest(&cfg).into_iter().map(|(n, _)| n).collect();
        for leaf in LEAVES {
            let key = format!("transformer_blocks.0.{leaf}.weight");
            assert!(names.contains(&key), "AV adapter targets {key}, which the manifest does not define");
        }
    }

    /// `linear_sites`' per-leaf shapes must match every manifest tensor's
    /// REAL shape exactly - the single source of truth this module's doc
    /// claims replaces the two hand-maintained parallel arrays `apply`/
    /// `step` used to carry.
    #[test]
    fn linear_sites_shapes_match_the_checkpoint_manifest_exactly() {
        let cfg = crate::LtxAvDitConfig::tiny();
        let manifest: std::collections::HashMap<String, Vec<usize>> = crate::dit::av_dit_tensor_manifest(&cfg).into_iter().collect();
        let avcfg = AvCfg::tiny();
        for site in linear_sites(&avcfg) {
            let want = manifest.get(&site.name).unwrap_or_else(|| panic!("{} not in manifest", site.name));
            assert_eq!(want, &vec![site.spec.out, site.spec.inn], "{}", site.name);
        }
    }
}
