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

use crate::grad::{GradsF32, WeightsF32};
use crate::modelgrad::{Cfg, ModelGradsF32, ModelWeightsF32};

/// LoRA hyper-parameters.
#[derive(Clone, Copy)]
pub struct LoraCfg {
    pub rank: usize,
    pub alpha: f32,
    pub seed: u64,
}

impl LoraCfg {
    pub fn new(rank: usize) -> LoraCfg {
        LoraCfg { rank, alpha: rank as f32, seed: 0 }
    }
    fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

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

/// A single linear's adapter: `A [r×in]`, `B [out×r]`, plus Adam moments.
#[derive(Clone)]
struct Pair {
    out: usize,
    inn: usize,
    r: usize,
    a: Vec<f32>, // [r × in], row-major
    b: Vec<f32>, // [out × r], row-major
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Pair {
    fn new(out: usize, inn: usize, r: usize, rng: &mut u64) -> Pair {
        // Standard LoRA init: A ~ small gaussian, B = 0 (so the initial adapter is
        // a no-op and W_eff == W).
        let a: Vec<f32> = (0..r * inn).map(|_| (randn(rng) * 0.02) as f32).collect();
        Pair {
            out,
            inn,
            r,
            a,
            b: vec![0.0; out * r],
            ma: vec![0.0; r * inn],
            va: vec![0.0; r * inn],
            mb: vec![0.0; out * r],
            vb: vec![0.0; out * r],
        }
    }

    /// `Δ = scale·B·A` in `W`'s `[out×in]` row-major layout — added onto the base.
    fn delta(&self, scale: f32, w: &mut [f32]) {
        // w[o,i] += scale · Σ_k B[o,k]·A[k,i]
        for o in 0..self.out {
            let brow = &self.b[o * self.r..o * self.r + self.r];
            let wrow = &mut w[o * self.inn..o * self.inn + self.inn];
            for k in 0..self.r {
                let bok = brow[k] * scale;
                if bok == 0.0 {
                    continue;
                }
                let arow = &self.a[k * self.inn..k * self.inn + self.inn];
                for i in 0..self.inn {
                    wrow[i] += bok * arow[i];
                }
            }
        }
    }

    /// Project the base-weight grad `dW [out×in]` to `(dA [r×in], dB [out×r])`:
    /// `dA = scale·Bᵀ·dW`, `dB = scale·dW·Aᵀ`.
    fn project(&self, dw: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
        let mut da = vec![0.0f32; self.r * self.inn];
        let mut db = vec![0.0f32; self.out * self.r];
        for o in 0..self.out {
            let dwrow = &dw[o * self.inn..o * self.inn + self.inn];
            let brow = &self.b[o * self.r..o * self.r + self.r];
            for k in 0..self.r {
                // dB[o,k] = scale·Σ_i dW[o,i]·A[k,i]
                let arow = &self.a[k * self.inn..k * self.inn + self.inn];
                let mut acc = 0.0f32;
                for i in 0..self.inn {
                    acc += dwrow[i] * arow[i];
                }
                db[o * self.r + k] = acc * scale;
                // dA[k,i] += scale·B[o,k]·dW[o,i]
                let bok = brow[k] * scale;
                let darow = &mut da[k * self.inn..k * self.inn + self.inn];
                for i in 0..self.inn {
                    darow[i] += bok * dwrow[i];
                }
            }
        }
        (da, db)
    }

    /// One AdamW-less Adam step on `A` and `B` from their projected grads.
    fn adam_step(&mut self, da: &[f32], db: &[f32], lr: f32, t: u64) {
        adam(&mut self.a, &mut self.ma, &mut self.va, da, lr, t);
        adam(&mut self.b, &mut self.mb, &mut self.vb, db, lr, t);
    }
}

fn adam(p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32], lr: f32, t: u64) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        p[i] -= lr * mh / (vh.sqrt() + eps);
    }
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
        let mk = |out, inn, rng: &mut u64| Pair::new(out, inn, r, rng);
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

fn proj_step(p: &mut Pair, dw: &[f32], scale: f32, lr: f32, t: u64) {
    let (da, db) = p.project(dw, scale);
    p.adam_step(&da, &db, lr, t);
}

/// A cheap deterministic standard-normal (xorshift + Box–Muller half).
fn randn(s: &mut u64) -> f64 {
    let mut nx = || {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    };
    let (u1, u2) = (nx(), nx());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
