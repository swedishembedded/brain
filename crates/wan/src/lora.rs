// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the Wan DiT.
//!
//! Each targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A` with
//! `A [r×in]`, `B [out×r]`. The **base is frozen**; only `A,B` train. Same
//! scheme as `s3dit::lora` / `flux2::lora`: rebuild the effective weights, run
//! the gradchecked host trainer ([`crate::modelgrad::grads`]) to get
//! `dL/dW_eff`, then *project* onto the adapter grads
//! (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) and Adam-step `A,B`. The generic
//! pair machinery lives once in `model::lora`; this module keeps only the
//! Wan-specific block walk and serialization naming.
//!
//! That host route materialises a full `W_eff` per step and reads a full `dW`
//! back per block, which on a discrete card is gigabytes each way for a value
//! only the rank-sized `(A, B)` ever consumes.
//! [`crate::train::DeviceTrainer::lora_grads`] runs the same two operations
//! on-device against a resident frozen base and hands back
//! [`LoraGrads`] - the same `(dA, dB)` [`LoraAdapter::project`] produces, which
//! [`LoraAdapter::step_projected`] Adam-steps identically.
//!
//! ## Wan fuses nothing, so there are no fused offsets to fold at
//!
//! The workspace rule that a LoRA over a fused checkpoint needs one adapter
//! pair per **slice**, folded back at the exact fused offsets, exists because
//! FLUX.2 and Z-Image ship `qkv` and `mlp.0` as single fused tensors. A Wan
//! checkpoint does not: `self_attn.{q,k,v,o}`,
//! `cross_attn.{q,k,v,o}`, `ffn.0` and `ffn.2` are ten independently-named
//! `[out, in]` tensors (`crate::import::dit_manifest`), so each pair maps onto a
//! whole tensor at offset 0 and [`model::lora::Pair::delta`] is the exact fold.
//! [`LoraAdapter::fold_into_tensors`] therefore reaches inference by name, and
//! `tests/lora_train.rs` asserts fold-vs-apply is **bit-equal** rather than
//! close - with no offsets in play, anything but bit-equality is a bug.
//!
//! Biases, norms and the whole conditioning path (`modulation`,
//! `time_projection`, `head.modulation`) are deliberately NOT adapted: LoRA's
//! premise is a low-rank correction to a big matrix, and those are vectors.

use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
pub use model::lora::{LoraCfg, Pair};
use model::lora::{proj_step, randn};

/// The ten pairs of one Wan block, named as the checkpoint names the tensors
/// they adapt.
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

/// The checkpoint leaf each pair adapts, in a fixed order. One table so the
/// walk, the serializer and the fold cannot disagree about which tensor is
/// which - the failure mode that silently trains `k` into `q`.
const LEAVES: [&str; 10] =
    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"];

fn pairs(b: &BlockLora) -> [&Pair; 10] {
    [&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.ff1, &b.ff2]
}

fn pairs_mut(b: &mut BlockLora) -> [&mut Pair; 10] {
    [&mut b.sq, &mut b.sk, &mut b.sv, &mut b.so, &mut b.cq, &mut b.ck, &mut b.cv, &mut b.co, &mut b.ff1, &mut b.ff2]
}

/// Adapter gradients: `(dA [r·in], dB [out·r])` per targeted linear, in
/// [`LEAVES`] order, per block.
///
/// The full-`dW` projection ([`LoraAdapter::step`]) and the device one
/// ([`crate::devgrad::BlockDev::backward_lora_loaded`]) both produce this, and
/// [`LoraAdapter::step_projected`] consumes either.
pub struct LoraGrads {
    pub blocks: Vec<crate::devgrad::AdapterGrads>,
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
        let (dim, ffn, r) = (cfg.dim, cfg.ffn_dim, lc.rank);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution the other two adapters
        // use, so a seed means the same thing across models.
        let mut mk = |out: usize, inn: usize| Pair::new(out, inn, r, || (randn(&mut rng) * 0.02) as f32);
        let blocks = (0..cfg.n_layers)
            .map(|_| BlockLora {
                sq: mk(dim, dim),
                sk: mk(dim, dim),
                sv: mk(dim, dim),
                so: mk(dim, dim),
                cq: mk(dim, dim),
                ck: mk(dim, dim),
                cv: mk(dim, dim),
                co: mk(dim, dim),
                ff1: mk(ffn, dim),
                ff2: mk(dim, ffn),
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

    /// The delta scale `α/r`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block `l`'s ten `(A, B)` pairs in [`LEAVES`] order - the operands a
    /// device-side fold and projection upload.
    pub fn block_ab(&self, l: usize) -> Vec<(&[f32], &[f32])> {
        pairs(&self.blocks[l]).into_iter().map(|p| (p.a.as_slice(), p.b.as_slice())).collect()
    }

    /// Effective weights `W_eff = W + scale·B·A` (base cloned; every other
    /// tensor - biases, norms, the conditioning path - passes through frozen).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        for (bl, wb) in self.blocks.iter().zip(w.blocks.iter_mut()) {
            let targets = [
                &mut wb.sq.w, &mut wb.sk.w, &mut wb.sv.w, &mut wb.so.w, &mut wb.cq.w, &mut wb.ck.w, &mut wb.cv.w, &mut wb.co.w,
                &mut wb.ff1.w, &mut wb.ff2.w,
            ];
            for (p, t) in pairs(bl).into_iter().zip(targets) {
                p.delta(self.scale, t);
            }
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads onto the
    /// adapter grads and Adam-update `A,B`. `grads` must be `dL/dW_eff` from a
    /// forward on this adapter's own [`LoraAdapter::apply`] output.
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        self.t += 1;
        let (scale, t) = (self.scale, self.t);
        for (bl, g) in self.blocks.iter_mut().zip(grads.blocks.iter()) {
            let dw = [&g.sq.w, &g.sk.w, &g.sv.w, &g.so.w, &g.cq.w, &g.ck.w, &g.cv.w, &g.co.w, &g.ff1.w, &g.ff2.w];
            for (p, d) in pairs_mut(bl).into_iter().zip(dw) {
                proj_step(p, d, scale, lr, t);
            }
        }
    }

    /// The projection half of [`LoraAdapter::step`] on its own: `dL/dW_eff` for
    /// every block onto `(dA, dB)`, adapter unchanged. What a device projection
    /// is checked against.
    pub fn project(&self, grads: &ModelGrads<f32>) -> LoraGrads {
        let scale = self.scale;
        let blocks = self
            .blocks
            .iter()
            .zip(grads.blocks.iter())
            .map(|(bl, g)| {
                let dw = [&g.sq.w, &g.sk.w, &g.sv.w, &g.so.w, &g.cq.w, &g.ck.w, &g.cv.w, &g.co.w, &g.ff1.w, &g.ff2.w];
                pairs(bl).into_iter().zip(dw).map(|(p, d)| p.project(d, scale)).collect()
            })
            .collect();
        LoraGrads { blocks }
    }

    /// The Adam half of [`LoraAdapter::step`] on its own, over adapter grads a
    /// caller already has - what the device trainer's on-device projection
    /// feeds.
    pub fn step_projected(&mut self, g: &LoraGrads, lr: f32) {
        self.t += 1;
        let t = self.t;
        assert_eq!(g.blocks.len(), self.blocks.len(), "step_projected: one grad set per block");
        for (bl, gb) in self.blocks.iter_mut().zip(g.blocks.iter()) {
            assert_eq!(gb.len(), LEAVES.len(), "step_projected: one (dA, dB) per targeted linear");
            for (p, (da, db)) in pairs_mut(bl).into_iter().zip(gb) {
                p.adam_step(da, db, lr, t);
            }
        }
    }

    /// Serialise to `(name, shape, data)` - `blocks.{l}.{leaf}.lora_{a,b}`,
    /// where `{leaf}` is the checkpoint's own tensor path.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs(bl)) {
                out.push((format!("blocks.{l}.{leaf}.lora_a"), vec![p.r, p.inn], p.a.clone()));
                out.push((format!("blocks.{l}.{leaf}.lora_b"), vec![p.out, p.r], p.b.clone()));
            }
        }
        out
    }

    /// Reload an adapter (weights only; Adam state resets by design).
    ///
    /// Validates the FULL shape of every `lora_a`/`lora_b` tensor, not just its
    /// element count: `A [r,in]` and `B [out,r]` can have equal length for
    /// square-ish targets, so a length-only check would silently accept an
    /// A/B swap.
    pub fn from_tensors(
        cfg: &Cfg,
        lc: LoraCfg,
        tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    ) -> Result<LoraAdapter, String> {
        let mut ad = LoraAdapter::new(cfg, lc);
        for (l, bl) in ad.blocks.iter_mut().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs_mut(bl)) {
                let (ka, kb) = (format!("blocks.{l}.{leaf}.lora_a"), format!("blocks.{l}.{leaf}.lora_b"));
                let (sa, a) = tensors.get(&ka).map(|(s, d)| (s.clone(), d)).ok_or_else(|| format!("lora: missing {ka}"))?;
                let (sb, b) = tensors.get(&kb).map(|(s, d)| (s.clone(), d)).ok_or_else(|| format!("lora: missing {kb}"))?;
                let (want_a, want_b) = (vec![p.r, p.inn], vec![p.out, p.r]);
                if !sa.is_empty() && sa != want_a {
                    return Err(format!("lora: {ka} has shape {sa:?}, expected {want_a:?}"));
                }
                if !sb.is_empty() && sb != want_b {
                    return Err(format!("lora: {kb} has shape {sb:?}, expected {want_b:?}"));
                }
                if a.len() != p.r * p.inn || b.len() != p.out * p.r {
                    return Err(format!("lora: {ka}/{kb} are {}/{} elems, expected {}/{}", a.len(), b.len(), p.r * p.inn, p.out * p.r));
                }
                p.a = a.clone();
                p.b = b.clone();
            }
        }
        Ok(ad)
    }

    /// Fold this adapter into an **inference** tensor map (what
    /// [`crate::WanDit`] / [`crate::WanDitDev`] build from), so the unchanged
    /// generation path produces adapter-conditioned video. Errors by name if a
    /// targeted tensor is absent or the wrong size.
    pub fn fold_into_tensors(&self, ts: &mut crate::model::Tensors) -> Result<(), String> {
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in LEAVES.iter().zip(pairs(bl)) {
                let key = format!("blocks.{l}.{leaf}.weight");
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
/// `{"model":"wan-lora","rank":R,"alpha":A}`.
///
/// Returns the write error instead of swallowing it: a failed periodic
/// checkpoint (disk full, permissions) must be visible to the caller, not
/// discovered only when the run finishes and the adapter is missing.
pub fn save_adapter(path: &str, ad: &LoraAdapter) -> Result<(), String> {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> =
        ad.to_tensors().into_iter().map(|(n, s, d)| (n, s.iter().map(|&x| x as u64).collect(), d)).collect();
    let config = serde_json::json!({"model": "wan-lora", "rank": ad.rank(), "alpha": ad.alpha()});
    checkpoint::st::save_safetensors(path, &t, &config, None).map_err(|e| format!("wan lora: cannot write {path}: {e}"))
}

/// Load an adapter written by [`save_adapter`]. Reads the header (for
/// `rank`/`alpha`) and the tensors (for their real shapes, per D3) from two
/// views of the same file - `checkpoint::load` never carried shapes, and
/// carrying them is what lets [`LoraAdapter::from_tensors`] catch an A/B swap.
///
/// A missing file is a `Result::Err` naming the path, never a panic:
/// `checkpoint::load` panics on a read failure, which is fine for a one-shot
/// tool but would take down a resident server on a typo'd `--adapter` path.
pub fn load_adapter(path: &str, cfg: &Cfg) -> Result<LoraAdapter, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("wan lora: adapter file not found: {path}"));
    }
    let c = checkpoint::load(path);
    let rank = c.header["config"]["rank"].as_u64().ok_or("adapter: missing rank in header")? as usize;
    let alpha = c.header["config"]["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let shaped = checkpoint::safetensors::read(path)?;
    let map: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> =
        shaped.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    LoraAdapter::from_tensors(cfg, LoraCfg { rank, alpha, seed: 0 }, &map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The leaf table must name real checkpoint tensors - a typo here folds
    /// into nothing and trains an adapter that never reaches inference.
    #[test]
    fn every_targeted_leaf_exists_in_the_checkpoint_manifest() {
        let wc = crate::WanConfig::t2v_1_3b();
        let names: std::collections::HashSet<String> = crate::import::dit_manifest(&wc).into_iter().map(|(n, _)| n).collect();
        for leaf in LEAVES {
            let key = format!("blocks.0.{leaf}.weight");
            assert!(names.contains(&key), "adapter targets {key}, which the manifest does not define");
        }
    }
}
