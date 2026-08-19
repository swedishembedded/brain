// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the audio+video LTX DiT - [`crate::lora`]'s
//! AV twin, same scheme (`W_eff = W + (α/r)·B·A`, base frozen, `B = 0` at
//! init so [`LoraAdapter::apply`] is an exact no-op), same generic pair
//! machinery from `model::lora`, same ComfyUI key layout
//! (`diffusion_model.<module path>.lora_A/B.weight`).
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

use crate::av_modelgrad::{AvCfg, AvModelGrads, AvModelWeights};
pub use model::lora::LoraCfg;
use model::lora::{proj_step, randn, Pair};

/// One block's 28 pairs, grouped the same way [`AvBlockW`](crate::av_grad::AvBlockW)
/// is: video stream (10), audio stream (10), AV cross both directions (8).
#[derive(Clone)]
struct AvBlockLora {
    v_sq: Pair,
    v_sk: Pair,
    v_sv: Pair,
    v_so: Pair,
    v_cq: Pair,
    v_ck: Pair,
    v_cv: Pair,
    v_co: Pair,
    v_ff1: Pair,
    v_ff2: Pair,
    a_sq: Pair,
    a_sk: Pair,
    a_sv: Pair,
    a_so: Pair,
    a_cq: Pair,
    a_ck: Pair,
    a_cv: Pair,
    a_co: Pair,
    a_ff1: Pair,
    a_ff2: Pair,
    a2v_q: Pair,
    a2v_k: Pair,
    a2v_v: Pair,
    a2v_o: Pair,
    v2a_q: Pair,
    v2a_k: Pair,
    v2a_v: Pair,
    v2a_o: Pair,
}

/// The checkpoint leaf each pair adapts, in the SAME fixed order
/// [`pairs`]/[`pairs_mut`] walk - one table so the walk, the serializer and
/// the fold cannot disagree about which tensor is which (`crate::lora`'s
/// own doc).
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

fn pairs(b: &AvBlockLora) -> [&Pair; 28] {
    [
        &b.v_sq, &b.v_sk, &b.v_sv, &b.v_so, &b.v_cq, &b.v_ck, &b.v_cv, &b.v_co, &b.v_ff1, &b.v_ff2, &b.a_sq, &b.a_sk, &b.a_sv, &b.a_so, &b.a_cq, &b.a_ck, &b.a_cv, &b.a_co, &b.a_ff1, &b.a_ff2,
        &b.a2v_q, &b.a2v_k, &b.a2v_v, &b.a2v_o, &b.v2a_q, &b.v2a_k, &b.v2a_v, &b.v2a_o,
    ]
}

fn pairs_mut(b: &mut AvBlockLora) -> [&mut Pair; 28] {
    [
        &mut b.v_sq, &mut b.v_sk, &mut b.v_sv, &mut b.v_so, &mut b.v_cq, &mut b.v_ck, &mut b.v_cv, &mut b.v_co, &mut b.v_ff1, &mut b.v_ff2, &mut b.a_sq, &mut b.a_sk, &mut b.a_sv, &mut b.a_so,
        &mut b.a_cq, &mut b.a_ck, &mut b.a_cv, &mut b.a_co, &mut b.a_ff1, &mut b.a_ff2, &mut b.a2v_q, &mut b.a2v_k, &mut b.a2v_v, &mut b.a2v_o, &mut b.v2a_q, &mut b.v2a_k, &mut b.v2a_v, &mut b.v2a_o,
    ]
}

/// A LoRA adapter over every block of the AV DiT.
pub struct LoraAdapter {
    scale: f32,
    rank: usize,
    blocks: Vec<AvBlockLora>,
    t: u64, // Adam step counter
}

impl LoraAdapter {
    /// Fresh adapter sized for `cfg`. `B = 0`, so it is an **exact no-op at
    /// init** - `apply` returns weights bit-identical to the base, the same
    /// bar `crates/ltxv/tests/av_lora_train.rs` asserts.
    pub fn new(cfg: &AvCfg, lc: LoraCfg) -> LoraAdapter {
        let (vdim, adim, r) = (cfg.vdim, cfg.adim, lc.rank);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution `crate::lora`/
        // `wan::lora`/`s3dit::lora` use, so a seed means the same thing
        // across models.
        let mut mk = |out: usize, inn: usize| Pair::new(out, inn, r, || (randn(&mut rng) * 0.02) as f32);
        let blocks = (0..cfg.num_layers)
            .map(|_| AvBlockLora {
                v_sq: mk(vdim, vdim),
                v_sk: mk(vdim, vdim),
                v_sv: mk(vdim, vdim),
                v_so: mk(vdim, vdim),
                v_cq: mk(vdim, vdim),
                v_ck: mk(vdim, vdim),
                v_cv: mk(vdim, vdim),
                v_co: mk(vdim, vdim),
                v_ff1: mk(4 * vdim, vdim),
                v_ff2: mk(vdim, 4 * vdim),
                a_sq: mk(adim, adim),
                a_sk: mk(adim, adim),
                a_sv: mk(adim, adim),
                a_so: mk(adim, adim),
                a_cq: mk(adim, adim),
                a_ck: mk(adim, adim),
                a_cv: mk(adim, adim),
                a_co: mk(adim, adim),
                a_ff1: mk(4 * adim, adim),
                a_ff2: mk(adim, 4 * adim),
                // audio_to_video_attn: q_dim=vdim, kv_dim=adim, inner=adim
                // (crate::av_grad's doc) - to_q is [adim,vdim], to_out.0 is
                // [vdim,adim]; to_k/to_v/to_out.0's OTHER operand are all
                // [adim,adim].
                a2v_q: mk(adim, vdim),
                a2v_k: mk(adim, adim),
                a2v_v: mk(adim, adim),
                a2v_o: mk(vdim, adim),
                // video_to_audio_attn: q_dim=adim, kv_dim=vdim, inner=adim -
                // to_k/to_v are [adim,vdim], everything else [adim,adim].
                v2a_q: mk(adim, adim),
                v2a_k: mk(adim, vdim),
                v2a_v: mk(adim, vdim),
                v2a_o: mk(adim, adim),
            })
            .collect();
        LoraAdapter { scale: lc.scale(), rank: r, blocks, t: 0 }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn alpha(&self) -> f32 {
        self.scale * self.rank as f32
    }

    /// Effective weights `W_eff = W + scale·B·A` (base cloned; every other
    /// tensor - biases, QK-norm gains, every adaLN table - passes through
    /// frozen).
    pub fn apply(&self, base: &AvModelWeights<f32>) -> AvModelWeights<f32> {
        let mut w = base.clone();
        for (bl, wb) in self.blocks.iter().zip(w.blocks.iter_mut()) {
            let targets: [&mut Vec<f32>; 28] = [
                &mut wb.v_attn1.q.w, &mut wb.v_attn1.k.w, &mut wb.v_attn1.v.w, &mut wb.v_attn1.o.w,
                &mut wb.v_attn2.q.w, &mut wb.v_attn2.k.w, &mut wb.v_attn2.v.w, &mut wb.v_attn2.o.w,
                &mut wb.v_ff1.w, &mut wb.v_ff2.w,
                &mut wb.a_attn1.q.w, &mut wb.a_attn1.k.w, &mut wb.a_attn1.v.w, &mut wb.a_attn1.o.w,
                &mut wb.a_attn2.q.w, &mut wb.a_attn2.k.w, &mut wb.a_attn2.v.w, &mut wb.a_attn2.o.w,
                &mut wb.a_ff1.w, &mut wb.a_ff2.w,
                &mut wb.av.a2v.q.w, &mut wb.av.a2v.k.w, &mut wb.av.a2v.v.w, &mut wb.av.a2v.o.w,
                &mut wb.av.v2a.q.w, &mut wb.av.v2a.k.w, &mut wb.av.v2a.v.w, &mut wb.av.v2a.o.w,
            ];
            for (p, t) in pairs(bl).into_iter().zip(targets) {
                p.delta(self.scale, t);
            }
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads onto
    /// the adapter grads and Adam-update `A,B`. `grads` must be `dL/dW_eff`
    /// from a forward on this adapter's own [`LoraAdapter::apply`] output.
    pub fn step(&mut self, grads: &AvModelGrads<f32>, lr: f32) {
        self.t += 1;
        let (scale, t) = (self.scale, self.t);
        for (bl, g) in self.blocks.iter_mut().zip(grads.blocks.iter()) {
            let dw: [&Vec<f32>; 28] = [
                &g.v_attn1.q.w, &g.v_attn1.k.w, &g.v_attn1.v.w, &g.v_attn1.o.w,
                &g.v_attn2.q.w, &g.v_attn2.k.w, &g.v_attn2.v.w, &g.v_attn2.o.w,
                &g.v_ff1.w, &g.v_ff2.w,
                &g.a_attn1.q.w, &g.a_attn1.k.w, &g.a_attn1.v.w, &g.a_attn1.o.w,
                &g.a_attn2.q.w, &g.a_attn2.k.w, &g.a_attn2.v.w, &g.a_attn2.o.w,
                &g.a_ff1.w, &g.a_ff2.w,
                &g.av.a2v.q.w, &g.av.a2v.k.w, &g.av.a2v.v.w, &g.av.a2v.o.w,
                &g.av.v2a.q.w, &g.av.v2a.k.w, &g.av.v2a.v.w, &g.av.v2a.o.w,
            ];
            for (p, d) in pairs_mut(bl).into_iter().zip(dw) {
                proj_step(p, d, scale, lr, t);
            }
        }
    }

    /// Serialise to `(name, shape, data)` in the ComfyUI key layout -
    /// `diffusion_model.transformer_blocks.{l}.{leaf}.lora_A/B.weight` -
    /// same convention `crate::lora`'s own doc explains.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs(bl)) {
                out.push((format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_A.weight"), vec![p.r, p.inn], p.a.clone()));
                out.push((format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_B.weight"), vec![p.out, p.r], p.b.clone()));
            }
        }
        out
    }

    /// Reload an adapter (weights only; Adam state resets by design).
    pub fn from_tensors(cfg: &AvCfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let mut ad = LoraAdapter::new(cfg, lc);
        for (l, bl) in ad.blocks.iter_mut().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs_mut(bl)) {
                let (ka, kb) = (format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_A.weight"), format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_B.weight"));
                let a = &tensors.get(&ka).ok_or_else(|| format!("av lora: missing {ka}"))?.1;
                let b = &tensors.get(&kb).ok_or_else(|| format!("av lora: missing {kb}"))?.1;
                if a.len() != p.r * p.inn || b.len() != p.out * p.r {
                    return Err(format!("av lora: {ka}/{kb} are {}/{} elems, expected {}/{}", a.len(), b.len(), p.r * p.inn, p.out * p.r));
                }
                p.a = a.clone();
                p.b = b.clone();
            }
        }
        Ok(ad)
    }

    /// Fold this adapter into an **inference** tensor map (`crate::dit::
    /// av_dit_tensor_manifest`'s own bare naming), so the unchanged
    /// generation path produces adapter-conditioned output.
    pub fn fold_into_tensors(&self, ts: &mut vae::blocks::Tensors) -> Result<(), String> {
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs(bl)) {
                let key = format!("transformer_blocks.{l}.{leaf}.weight");
                let w = ts.get_mut(&key).ok_or_else(|| format!("av lora: base tensor {key} missing"))?;
                if w.1.len() != p.out * p.inn {
                    return Err(format!("av lora: {key} is {} elems, adapter expects {}", w.1.len(), p.out * p.inn));
                }
                p.delta(self.scale, &mut w.1);
            }
        }
        Ok(())
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
}
