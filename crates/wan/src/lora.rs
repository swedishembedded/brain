// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the Wan DiT, over the generic
//! `model::adapter` substrate.
//!
//! Each targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A` with
//! `A [r×in]`, `B [out×r]`. The **base is frozen**; only `A,B` train. Same
//! scheme as `s3dit::lora` / `flux2::lora`: rebuild the effective weights, run
//! the gradchecked host trainer ([`crate::modelgrad::grads`]) to get
//! `dL/dW_eff`, then *project* onto the adapter grads
//! (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) and Adam-step `A,B`. The generic
//! pair machinery lives once in `model::lora`, dispatched through
//! `model::adapter::AdapterKind`; this module keeps only the Wan-specific
//! block walk and serialization naming. `Pair` itself is untouched by that
//! seam and stays re-exported here (`pub use model::lora::{LoraCfg, Pair}`)
//! for existing callers.
//!
//! That host route materialises a full `W_eff` per step and reads a full `dW`
//! back per block, which on a discrete card is gigabytes each way for a value
//! only the rank-sized `(A, B)` ever consumes.
//! [`crate::train::DeviceTrainer::lora_grads`] runs the same two operations
//! on-device against a resident frozen base and hands back
//! [`LoraGrads`] - the same `(dA, dB)` [`LoraAdapter::project`] produces, which
//! [`LoraAdapter::step_projected`] Adam-steps identically. [`LoraGrads`] is
//! this crate's own model-wide grad bundle (`Vec<blocks> of
//! Vec<crate::devgrad::AdapterGrads>`, i.e. plain `(Vec<f32>, Vec<f32>)`
//! tuples with no `model::adapter` dependency) - a DIFFERENT type from
//! [`model::lora::LoraGrads`] (one target's `(dA, dB)`, the associated
//! `AdapterKind::Grads` for a single [`model::lora::LoraPair`]);
//! [`LoraAdapter::step_projected`] is the seam between the two.
//!
//! ## Wan fuses nothing, so there are no fused offsets to fold at
//!
//! The workspace rule that a LoRA over a fused checkpoint needs one adapter
//! pair per **slice**, folded back at the exact fused offsets, exists because
//! FLUX.2 and Z-Image ship `qkv` and `mlp.0` as single fused tensors. A Wan
//! checkpoint does not: `self_attn.{q,k,v,o}`,
//! `cross_attn.{q,k,v,o}`, `ffn.0` and `ffn.2` are ten independently-named
//! `[out, in]` tensors (`crate::import::dit_manifest`), so each pair maps onto a
//! whole tensor at offset 0 ([`model::adapter::TargetSpec::whole`]).
//! [`LoraAdapter::fold_into_tensors`] therefore reaches inference by name, and
//! `tests/lora_train.rs` asserts fold-vs-apply is **bit-equal** rather than
//! close - with no offsets in play, anything but bit-equality is a bug.
//!
//! Biases, norms and the whole conditioning path (`modulation`,
//! `time_projection`, `head.modulation`) are deliberately NOT adapted: LoRA's
//! premise is a low-rank correction to a big matrix, and those are vectors.

use crate::grad::{BlockGrads, BlockW};
use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
pub use model::lora::{LoraCfg, Pair};
use model::lora::LoraPair;

/// The checkpoint leaf each pair adapts, in a fixed order. One table so the
/// walk, the serializer and the fold cannot disagree about which tensor is
/// which - the failure mode that silently trains `k` into `q`.
const LEAVES: [&str; 10] =
    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"];

fn field_mut<'a>(b: &'a mut BlockW<f32>, leaf: &str) -> &'a mut Vec<f32> {
    match leaf {
        "self_attn.q" => &mut b.sq.w,
        "self_attn.k" => &mut b.sk.w,
        "self_attn.v" => &mut b.sv.w,
        "self_attn.o" => &mut b.so.w,
        "cross_attn.q" => &mut b.cq.w,
        "cross_attn.k" => &mut b.ck.w,
        "cross_attn.v" => &mut b.cv.w,
        "cross_attn.o" => &mut b.co.w,
        "ffn.0" => &mut b.ff1.w,
        "ffn.2" => &mut b.ff2.w,
        other => panic!("wan lora: unknown leaf {other:?}"),
    }
}

fn field<'a>(g: &'a BlockGrads<f32>, leaf: &str) -> &'a Vec<f32> {
    match leaf {
        "self_attn.q" => &g.sq.w,
        "self_attn.k" => &g.sk.w,
        "self_attn.v" => &g.sv.w,
        "self_attn.o" => &g.so.w,
        "cross_attn.q" => &g.cq.w,
        "cross_attn.k" => &g.ck.w,
        "cross_attn.v" => &g.cv.w,
        "cross_attn.o" => &g.co.w,
        "ffn.0" => &g.ff1.w,
        "ffn.2" => &g.ff2.w,
        other => panic!("wan lora: unknown leaf {other:?}"),
    }
}

/// Every linear this crate offers to a PEFT adapter: the ten leaves of every
/// block, in [`LEAVES`] order - the pre-migration `BlockLora` field order
/// (and its random-init draw order).
pub fn linear_sites(cfg: &Cfg) -> Vec<LinearSite> {
    let (dim, ffn) = (cfg.dim, cfg.ffn_dim);
    let mut sites = Vec::with_capacity(cfg.n_layers * LEAVES.len());
    for l in 0..cfg.n_layers {
        for leaf in LEAVES {
            let (out, inn) = match leaf {
                "ffn.0" => (ffn, dim),
                "ffn.2" => (dim, ffn),
                _ => (dim, dim),
            };
            sites.push(LinearSite { name: format!("blocks.{l}.{leaf}.weight"), leaf, layer: Some(l), spec: TargetSpec::whole(out, inn), save_name: None });
        }
    }
    sites
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
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
    n_blocks: usize,
}

impl LoraAdapter {
    /// Fresh adapter sized for `cfg`. `B = 0`, so it is an **exact no-op at
    /// init** - `apply` returns weights bit-identical to the base, which
    /// `tests/lora_train.rs` asserts rather than assumes.
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Gaussian σ 0.02, the same init distribution the other two adapters
        // use, so a seed means the same thing across models.
        let mut init = move || (model::lora::randn(&mut rng) * 0.02) as f32;
        let set = AdapterSet::build(sites, hp, KeyStyle::Brain, &mut init);
        LoraAdapter { set, hp, n_blocks: cfg.n_layers }
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }

    pub fn alpha(&self) -> f32 {
        self.hp.alpha
    }

    /// The delta scale `α/r`.
    pub fn scale(&self) -> f32 {
        self.hp.scale()
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Block `l`'s ten `(A, B)` pairs in [`LEAVES`] order - the operands a
    /// device-side fold and projection upload.
    pub fn block_ab(&self, l: usize) -> Vec<(&[f32], &[f32])> {
        self.set
            .iter()
            .filter(|(site, _)| site.layer == Some(l))
            .map(|(_, k)| {
                let p = k.pair();
                (p.a.as_slice(), p.b.as_slice())
            })
            .collect()
    }

    /// Effective weights `W_eff = W + scale·B·A` (base cloned; every other
    /// tensor - biases, norms, the conditioning path - passes through frozen).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("wan lora: every site has a layer index");
            k.delta_into(1.0, field_mut(&mut w.blocks[l], site.leaf));
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads onto the
    /// adapter grads and Adam-update `A,B`. `grads` must be `dL/dW_eff` from a
    /// forward on this adapter's own [`LoraAdapter::apply`] output.
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        let projected: Vec<model::lora::LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let l = site.layer.expect("wan lora: every site has a layer index");
                k.project(field(&grads.blocks[l], site.leaf))
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }

    /// The projection half of [`LoraAdapter::step`] on its own: `dL/dW_eff` for
    /// every block onto `(dA, dB)`, adapter unchanged. What a device projection
    /// is checked against.
    pub fn project(&self, grads: &ModelGrads<f32>) -> LoraGrads {
        let mut blocks: Vec<crate::devgrad::AdapterGrads> = (0..self.n_blocks).map(|_| Vec::with_capacity(LEAVES.len())).collect();
        for (site, k) in self.set.iter() {
            let l = site.layer.expect("wan lora: every site has a layer index");
            let g = k.project(field(&grads.blocks[l], site.leaf));
            blocks[l].push((g.da, g.db));
        }
        LoraGrads { blocks }
    }

    /// The Adam half of [`LoraAdapter::step`] on its own, over adapter grads a
    /// caller already has - what the device trainer's on-device projection
    /// feeds.
    pub fn step_projected(&mut self, g: &LoraGrads, lr: f32) {
        assert_eq!(g.blocks.len(), self.n_blocks, "step_projected: one grad set per block");
        let flat: Vec<model::lora::LoraGrads> = g
            .blocks
            .iter()
            .flat_map(|block_grads| {
                assert_eq!(block_grads.len(), LEAVES.len(), "step_projected: one (dA, dB) per targeted linear");
                block_grads.iter()
            })
            .map(|(da, db)| model::lora::LoraGrads { da: da.clone(), db: db.clone() })
            .collect();
        self.set.step_projected(&flat, lr);
    }

    /// Serialise to `(name, shape, data)` - `blocks.{l}.{leaf}.lora_{a,b}`,
    /// where `{leaf}` is the checkpoint's own tensor path.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
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
        let sites = linear_sites(cfg);
        let hp = TargetHp::from(lc);
        let set = AdapterSet::from_tensors(sites, hp, KeyStyle::Brain, tensors)?;
        Ok(LoraAdapter { set, hp, n_blocks: cfg.n_layers })
    }

    /// Fold this adapter into an **inference** tensor map (what
    /// [`crate::WanDit`] / [`crate::WanDitDev`] build from), so the unchanged
    /// generation path produces adapter-conditioned video. Errors by name if a
    /// targeted tensor is absent or the wrong size.
    pub fn fold_into_tensors(&self, ts: &mut crate::model::Tensors) -> Result<(), String> {
        self.set.fold_into(ts, 1.0)
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

    /// `block_ab`/`project`/`step_projected` reshape between this crate's own
    /// per-block `Vec<AdapterGrads>` layout and the generic engine's flat
    /// entry order - assert the reshape is a lossless round trip: project
    /// then step_projected must reproduce plain `step`'s effect exactly, on
    /// real gradients from a real (tiny) forward+backward.
    #[test]
    fn project_then_step_projected_matches_step() {
        use crate::config::WanConfig;
        use crate::import::dit_manifest;
        use crate::modelgrad::{grads, make_flow_batch, ModelWeights};
        use std::collections::HashMap;

        let cfg = Cfg::tiny();
        let wc = WanConfig {
            name: "tiny-lora",
            dim: cfg.dim,
            ffn_dim: cfg.ffn_dim,
            num_heads: cfg.n_heads,
            num_layers: cfg.n_layers,
            in_channels: cfg.in_channels,
            out_channels: cfg.out_channels,
            text_dim: cfg.text_dim,
            text_len: cfg.text_len,
            freq_dim: cfg.freq_dim,
            ..WanConfig::t2v_1_3b()
        };
        let mut ts: crate::model::Tensors = HashMap::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for (name, shape) in dit_manifest(&wc) {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5)
                })
                .collect();
            ts.insert(name, (shape, v));
        }
        let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
        let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
        let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
        let rows = cfg.text_len - 1;
        let text_ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
        let batch = make_flow_batch(&cfg, &x0, &text_ctx, rows, 0.5, &noise);

        let mut direct = LoraAdapter::new(&cfg, LoraCfg::new(2));
        let mut via_projection = LoraAdapter::new(&cfg, LoraCfg::new(2));

        let (_l, g) = grads(&cfg, &direct.apply(&base), &batch);
        direct.step(&g, 0.01);
        let (_l2, g2) = grads(&cfg, &via_projection.apply(&base), &batch);
        let projected = via_projection.project(&g2);
        via_projection.step_projected(&projected, 0.01);

        let (a, b) = (direct.to_tensors(), via_projection.to_tensors());
        for ((na, sa, da), (nb, sb, db)) in a.iter().zip(b.iter()) {
            assert_eq!(na, nb);
            assert_eq!(sa, sb);
            for (x, y) in da.iter().zip(db.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "{na}");
            }
        }
    }
}
