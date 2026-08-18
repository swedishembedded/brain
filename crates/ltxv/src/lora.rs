// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the video-only LTX DiT.
//!
//! Each targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A` with
//! `A [r×in]`, `B [out×r]`. The **base is frozen**; only `A,B` train. Same
//! scheme as `wan::lora` (which this module mirrors closely): rebuild the
//! effective weights, run the gradchecked host trainer
//! ([`crate::modelgrad::grads`]) to get `dL/dW_eff`, then *project* onto the
//! adapter grads (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) and Adam-step
//! `A,B`. The generic pair machinery lives once in `model::lora`; this
//! module keeps only the LTX-specific block walk and serialization naming.
//!
//! ## LTX fuses nothing at this milestone, so there are no fused offsets
//!
//! Like Wan (and unlike FLUX.2/Z-Image's fused `qkv`/`mlp.0`), an LTX block's
//! `attn1.{to_q,to_k,to_v,to_out.0}`, `attn2.{to_q,to_k,to_v,to_out.0}` and
//! `ff.net.{0.proj,2}` are ten independently-named `[out, in]` tensors
//! (`crate::dit::dit_tensor_manifest`), so each pair maps onto a whole
//! tensor at offset 0 and [`model::lora::Pair::delta`] is the exact fold -
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
//! ecosystem convention this port is asked to match). `<module path>` is
//! `crate::dit::dit_tensor_manifest`'s own tensor path MINUS the trailing
//! `.weight` (e.g. `transformer_blocks.0.attn1.to_q`). This is purely about
//! how the ADAPTER file names its own tensors; [`LoraAdapter::fold_into_tensors`]
//! still targets the base model's OWN bare tensor keys (no `diffusion_model.`
//! prefix) when folding into `crate::dit::LtxDit`'s inference tensor map -
//! exactly how a real ComfyUI loader matches an adapter key to a base key by
//! stripping the `diffusion_model.` prefix and the `.lora_{A,B}.weight` suffix.
//!
//! Biases, the QK-norm gains, and the whole conditioning path
//! (`scale_shift_table`, `prompt_scale_shift_table`, `adaln_single.*`) are
//! deliberately NOT adapted: LoRA's premise is a low-rank correction to a
//! big matrix, and those are vectors (or, for `prompt_scale_shift_table`,
//! too small - `2*dim` - for a rank decomposition to make sense).

use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
pub use model::lora::LoraCfg;
use model::lora::{proj_step, randn, Pair};

/// The ten pairs of one block's TWO attention modules plus its FFN, named as
/// the checkpoint names the tensors they adapt (`s` = `attn1`/self, `c` =
/// `attn2`/cross - the same short names `wan::lora::BlockLora` uses).
#[derive(Clone)]
struct BlockLora {
    sq: Pair,
    sk: Pair,
    sv: Pair,
    so: Pair,
    cq: Pair,
    ck: Pair,
    cv: Pair,
    co: Pair,
    ff1: Pair,
    ff2: Pair,
}

/// The checkpoint leaf each pair adapts, in a fixed order - one table so the
/// walk, the serializer and the fold cannot disagree about which tensor is
/// which.
const LEAVES: [&str; 10] =
    ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"];

fn pairs(b: &BlockLora) -> [&Pair; 10] {
    [&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.ff1, &b.ff2]
}

fn pairs_mut(b: &mut BlockLora) -> [&mut Pair; 10] {
    [&mut b.sq, &mut b.sk, &mut b.sv, &mut b.so, &mut b.cq, &mut b.ck, &mut b.cv, &mut b.co, &mut b.ff1, &mut b.ff2]
}

/// A LoRA adapter over every block of the DiT.
pub struct LoraAdapter {
    scale: f32,
    rank: usize,
    blocks: Vec<BlockLora>,
    t: u64, // Adam step counter
}

impl LoraAdapter {
    /// Fresh adapter sized for `cfg`. `B = 0`, so it is an **exact no-op at
    /// init** - `apply` returns weights bit-identical to the base, which
    /// `tests/lora_train.rs` asserts rather than assumes.
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let (dim, r) = (cfg.dim, lc.rank);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution `wan::lora`/`s3dit::lora`
        // use, so a seed means the same thing across models.
        let mut mk = |out: usize, inn: usize| Pair::new(out, inn, r, || (randn(&mut rng) * 0.02) as f32);
        let blocks = (0..cfg.num_layers)
            .map(|_| BlockLora {
                sq: mk(dim, dim),
                sk: mk(dim, dim),
                sv: mk(dim, dim),
                so: mk(dim, dim),
                cq: mk(dim, dim),
                ck: mk(dim, dim),
                cv: mk(dim, dim),
                co: mk(dim, dim),
                ff1: mk(4 * dim, dim),
                ff2: mk(dim, 4 * dim),
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
    /// tensor - biases, QK-norm gains, the conditioning path - passes
    /// through frozen).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        for (bl, wb) in self.blocks.iter().zip(w.blocks.iter_mut()) {
            let targets: [&mut Vec<f32>; 10] = [
                &mut wb.attn1.q.w, &mut wb.attn1.k.w, &mut wb.attn1.v.w, &mut wb.attn1.o.w,
                &mut wb.attn2.q.w, &mut wb.attn2.k.w, &mut wb.attn2.v.w, &mut wb.attn2.o.w,
                &mut wb.ff1.w, &mut wb.ff2.w,
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
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        self.t += 1;
        let (scale, t) = (self.scale, self.t);
        for (bl, g) in self.blocks.iter_mut().zip(grads.blocks.iter()) {
            let dw: [&Vec<f32>; 10] = [
                &g.attn1.q.w, &g.attn1.k.w, &g.attn1.v.w, &g.attn1.o.w,
                &g.attn2.q.w, &g.attn2.k.w, &g.attn2.v.w, &g.attn2.o.w,
                &g.ff1.w, &g.ff2.w,
            ];
            for (p, d) in pairs_mut(bl).into_iter().zip(dw) {
                proj_step(p, d, scale, lr, t);
            }
        }
    }

    /// Serialise to `(name, shape, data)` in the ComfyUI key layout -
    /// `diffusion_model.transformer_blocks.{l}.{leaf}.lora_A/B.weight` - see
    /// this module's doc.
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
    pub fn from_tensors(cfg: &Cfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let mut ad = LoraAdapter::new(cfg, lc);
        for (l, bl) in ad.blocks.iter_mut().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs_mut(bl)) {
                let (ka, kb) = (format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_A.weight"), format!("diffusion_model.transformer_blocks.{l}.{leaf}.lora_B.weight"));
                let a = &tensors.get(&ka).ok_or_else(|| format!("lora: missing {ka}"))?.1;
                let b = &tensors.get(&kb).ok_or_else(|| format!("lora: missing {kb}"))?.1;
                if a.len() != p.r * p.inn || b.len() != p.out * p.r {
                    return Err(format!("lora: {ka}/{kb} are {}/{} elems, expected {}/{}", a.len(), b.len(), p.r * p.inn, p.out * p.r));
                }
                p.a = a.clone();
                p.b = b.clone();
            }
        }
        Ok(ad)
    }

    /// Fold this adapter into an **inference** tensor map
    /// (`crate::dit::dit_tensor_manifest`'s own bare naming - what
    /// `crate::dit::LtxDit` reads from), so the unchanged generation path
    /// produces adapter-conditioned output. Errors by name if a targeted
    /// tensor is absent or the wrong size.
    pub fn fold_into_tensors(&self, ts: &mut vae::blocks::Tensors) -> Result<(), String> {
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs(bl)) {
                let key = format!("transformer_blocks.{l}.{leaf}.weight");
                let w = ts.get_mut(&key).ok_or_else(|| format!("lora: base tensor {key} missing"))?;
                if w.1.len() != p.out * p.inn {
                    return Err(format!("lora: {key} is {} elems, adapter expects {}", w.1.len(), p.out * p.inn));
                }
                p.delta(self.scale, &mut w.1);
            }
        }
        Ok(())
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
}
