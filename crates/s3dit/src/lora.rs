// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the Z-Image DiT.
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

use crate::grad::WeightsF32;
use crate::modelgrad::{Cfg, ModelGradsF32, ModelWeightsF32};
// The generic pair machinery (A/B init, ΔW apply, dW→(dA,dB) projection, Adam
// moments) is model-agnostic and lives ONCE in `model::lora` — this module
// keeps only the Z-Image-specific block walk and serialization naming.
// `LoraCfg` is re-exported for existing callers.
pub use model::lora::LoraCfg;
use model::lora::{proj_step, randn, Pair};

/// The seven low-rank pairs for one transformer block.
#[derive(Clone)]
struct BlockLora {
    wq: Pair,
    wk: Pair,
    wv: Pair,
    wo: Pair,
    w1: Pair,
    w2: Pair,
    w3: Pair,
}

/// A LoRA adapter over all `main` blocks of the DiT.
pub struct LoraAdapter {
    scale: f32,
    rank: usize,
    blocks: Vec<BlockLora>,
    t: u64, // Adam step counter
}

impl LoraAdapter {
    /// Fresh adapter (B=0 → initial no-op) sized for `cfg`, over `cfg.n_layers`
    /// main blocks. Targets attention (`wq/wk/wv/wo`) and MLP (`w1/w2/w3`).
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let (dim, r) = (cfg.dim, lc.rank);
        let hidden = dim * 8 / 3;
        let mut rng = lc.seed ^ 0x1234_5678_9abc_def0;
        // Same init distribution as before the model::lora hoist (gaussian,
        // σ 0.02) so existing seeds reproduce bit-identical adapters.
        let mk = |out, inn, rng: &mut u64| Pair::new(out, inn, r, || (randn(rng) * 0.02) as f32);
        let blocks = (0..cfg.n_layers)
            .map(|_| BlockLora {
                wq: mk(dim, dim, &mut rng),
                wk: mk(dim, dim, &mut rng),
                wv: mk(dim, dim, &mut rng),
                wo: mk(dim, dim, &mut rng),
                w1: mk(hidden, dim, &mut rng),
                w2: mk(dim, hidden, &mut rng),
                w3: mk(hidden, dim, &mut rng),
            })
            .collect();
        LoraAdapter { scale: lc.scale(), rank: r, blocks, t: 0 }
    }

    /// Build the effective weights `W_eff = W + scale·B·A` (base cloned, adapters
    /// added onto each targeted `main` linear).
    pub fn apply(&self, base: &ModelWeightsF32) -> ModelWeightsF32 {
        let mut w = base.clone();
        for (bl, wb) in self.blocks.iter().zip(w.main.iter_mut()) {
            bl.wq.delta(self.scale, &mut wb.wq);
            bl.wk.delta(self.scale, &mut wb.wk);
            bl.wv.delta(self.scale, &mut wb.wv);
            bl.wo.delta(self.scale, &mut wb.wo);
            bl.w1.delta(self.scale, &mut wb.w1);
            bl.w2.delta(self.scale, &mut wb.w2);
            bl.w3.delta(self.scale, &mut wb.w3);
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads to adapter
    /// grads and Adam-update `A,B`. `grads` is `dL/dW_eff` from the frozen-base
    /// forward on the current `apply()`ed weights.
    pub fn step(&mut self, grads: &ModelGradsF32, lr: f32) {
        self.t += 1;
        let (scale, t) = (self.scale, self.t);
        for (bl, g) in self.blocks.iter_mut().zip(grads.main.iter()) {
            proj_step(&mut bl.wq, &g.wq, scale, lr, t);
            proj_step(&mut bl.wk, &g.wk, scale, lr, t);
            proj_step(&mut bl.wv, &g.wv, scale, lr, t);
            proj_step(&mut bl.wo, &g.wo, scale, lr, t);
            proj_step(&mut bl.w1, &g.w1, scale, lr, t);
            proj_step(&mut bl.w2, &g.w2, scale, lr, t);
            proj_step(&mut bl.w3, &g.w3, scale, lr, t);
        }
    }

    /// Serialise to `(name, shape, data)` tensors — `blocks.{l}.{lin}.lora_{a,b}`.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        for (l, bl) in self.blocks.iter().enumerate() {
            for (name, p) in [
                ("wq", &bl.wq), ("wk", &bl.wk), ("wv", &bl.wv), ("wo", &bl.wo),
                ("w1", &bl.w1), ("w2", &bl.w2), ("w3", &bl.w3),
            ] {
                out.push((format!("blocks.{l}.{name}.lora_a"), vec![p.r, p.inn], p.a.clone()));
                out.push((format!("blocks.{l}.{name}.lora_b"), vec![p.out, p.r], p.b.clone()));
            }
        }
        out
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn alpha(&self) -> f32 {
        self.scale * self.rank as f32
    }

    /// Fold this adapter's deltas into an **inference** tensor map (the
    /// `import_comfy` layout the generation path builds from), so a plain
    /// `text2image` produces adapter-conditioned images with no model change.
    /// Each main block `l`'s `W += (α/r)·B·A` is added onto the matching
    /// `layers.{l}.{…}.weight`. Refiner blocks are not adapted (the adapter only
    /// targets the main layers). Errors if a targeted tensor is absent.
    pub fn fold_into_comfy(&self, t: &mut crate::block::Tensors) -> Result<(), String> {
        for (l, bl) in self.blocks.iter().enumerate() {
            for (leaf, p) in [
                ("attention.to_q.weight", &bl.wq), ("attention.to_k.weight", &bl.wk),
                ("attention.to_v.weight", &bl.wv), ("attention.to_out.0.weight", &bl.wo),
                ("feed_forward.w1.weight", &bl.w1), ("feed_forward.w2.weight", &bl.w2),
                ("feed_forward.w3.weight", &bl.w3),
            ] {
                let key = format!("layers.{l}.{leaf}");
                let w = t.get_mut(&key).ok_or_else(|| format!("lora: base tensor {key} missing"))?;
                if w.1.len() != p.out * p.inn {
                    return Err(format!("lora: {key} is {} elems, adapter expects {}", w.1.len(), p.out * p.inn));
                }
                p.delta(self.scale, &mut w.1);
            }
        }
        Ok(())
    }

    /// Reload an adapter (weights only; Adam state reset) from `to_tensors`
    /// output — a fresh adapter of the right shape with `A,B` overwritten.
    pub fn from_tensors(cfg: &Cfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<LoraAdapter, String> {
        let mut ad = LoraAdapter::new(cfg, lc);
        for (l, bl) in ad.blocks.iter_mut().enumerate() {
            for (name, p) in [
                ("wq", &mut bl.wq), ("wk", &mut bl.wk), ("wv", &mut bl.wv), ("wo", &mut bl.wo),
                ("w1", &mut bl.w1), ("w2", &mut bl.w2), ("w3", &mut bl.w3),
            ] {
                let ka = format!("blocks.{l}.{name}.lora_a");
                let kb = format!("blocks.{l}.{name}.lora_b");
                p.a = tensors.get(&ka).ok_or_else(|| format!("missing {ka}"))?.1.clone();
                p.b = tensors.get(&kb).ok_or_else(|| format!("missing {kb}"))?.1.clone();
            }
        }
        Ok(ad)
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
