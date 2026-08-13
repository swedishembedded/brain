// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the FLUX.2 Klein DiT.
//!
//! Each targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A` with
//! `A [r×in]`, `B [out×r]`. The **base is frozen**; only `A,B` train. Design:
//! s3dit::lora's `Pair { a, b, project }` scheme - rebuild the effective
//! weights, run the gradchecked host trainer ([`crate::modelgrad::grads`]) to
//! get `dL/dW_eff`, then *project* onto the adapter grads
//! (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) and Adam-step `A,B`. Chosen over
//! qwen's param-list design because the host trainer already returns dense
//! per-slice grads for every split projection, so projection reuses the
//! gradchecked backward with zero new backward code — and the checkpoint's
//! fused tensors are handled by giving each fused **slice** its own pair.
//!
//! Targets, per the fused-checkpoint layout:
//! * double block, per stream: `qkv` → three row-slice pairs (q/k/v), `proj`,
//!   `mlp.0` → two row-slice pairs (w1/w3), `mlp.2`;
//! * single block: `linear1` → five row-slice pairs (q/k/v/w1/w3), `linear2` →
//!   two column-split pairs (`wo_a`/`wo_b`).
//!
//! [`LoraAdapter::fold_into_tensors`] adds each pair's `(α/r)·B·A` back into
//! the fused inference tensors (row/column offsets matching `model.rs`'s
//! build-time split) so the unchanged generation path picks a trained adapter
//! up. Serialization: brain's `checkpoint` container, header
//! `{"model":"flux2-lora","rank":R,"alpha":A}`.

use crate::grad::StreamW;
use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
// The generic pair machinery (A/B init, ΔW apply, dW→(dA,dB) projection, Adam
// moments) is model-agnostic and lives ONCE in `model::lora` — this module
// keeps only the FLUX.2-specific block walk, fused-tensor offsets and
// serialization naming. `LoraCfg` is re-exported for existing callers.
pub use model::lora::LoraCfg;
use model::lora::{proj_step, Pair};

/// The seven pairs of one double-block stream (qkv slices + proj + mlp slices).
#[derive(Clone)]
struct StreamLora {
    wq: Pair,
    wk: Pair,
    wv: Pair,
    wo: Pair,
    w1: Pair,
    w3: Pair,
    w2: Pair,
}

/// The seven pairs of one single block (linear1 slices + linear2 column split).
#[derive(Clone)]
struct SingleLora {
    wq: Pair,
    wk: Pair,
    wv: Pair,
    w1: Pair,
    w3: Pair,
    wo_a: Pair,
    wo_b: Pair,
}

/// A LoRA adapter over every double- and single-block linear of the DiT
/// (qk-norm scales and the global embed/modulation/final linears stay frozen).
pub struct LoraAdapter {
    scale: f32,
    rank: usize,
    dbl: Vec<(StreamLora, StreamLora)>, // (img, txt)
    sgl: Vec<SingleLora>,
    t: u64, // Adam step counter
}

const STREAMS: [&str; 2] = ["img", "txt"];
const STREAM_LEAVES: [&str; 7] = ["wq", "wk", "wv", "wo", "w1", "w3", "w2"];
const SINGLE_LEAVES: [&str; 7] = ["wq", "wk", "wv", "w1", "w3", "wo_a", "wo_b"];

impl LoraAdapter {
    /// Fresh adapter (B=0 → initial no-op) sized for `cfg`.
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        let (d, mlp, r) = (cfg.hidden, cfg.mlp, lc.rank);
        let mut rng = data::rng::Rng::new(lc.seed ^ 0xf1a2_b3c4_d5e6_0789);
        // Same init distribution as before the model::lora hoist (uniform,
        // ±0.02) so existing seeds reproduce bit-identical adapters.
        let mut mk = |out, inn| Pair::new(out, inn, r, || (rng.next_f64() - 0.5) as f32 * 0.04);
        let stream = |mk: &mut dyn FnMut(usize, usize) -> Pair| StreamLora {
            wq: mk(d, d),
            wk: mk(d, d),
            wv: mk(d, d),
            wo: mk(d, d),
            w1: mk(mlp, d),
            w3: mk(mlp, d),
            w2: mk(d, mlp),
        };
        let dbl = (0..cfg.depth_double).map(|_| (stream(&mut mk), stream(&mut mk))).collect();
        let sgl = (0..cfg.depth_single)
            .map(|_| SingleLora {
                wq: mk(d, d),
                wk: mk(d, d),
                wv: mk(d, d),
                w1: mk(mlp, d),
                w3: mk(mlp, d),
                wo_a: mk(d, d),
                wo_b: mk(d, mlp),
            })
            .collect();
        LoraAdapter { scale: lc.scale(), rank: r, dbl, sgl, t: 0 }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn alpha(&self) -> f32 {
        self.scale * self.rank as f32
    }

    /// Build the effective weights `W_eff = W + scale·B·A` (base cloned).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        for ((li, lt), bw) in self.dbl.iter().zip(w.dbl.iter_mut()) {
            apply_stream(li, &mut bw.img, self.scale);
            apply_stream(lt, &mut bw.txt, self.scale);
        }
        for (l, bw) in self.sgl.iter().zip(w.sgl.iter_mut()) {
            l.wq.delta(self.scale, &mut bw.wq);
            l.wk.delta(self.scale, &mut bw.wk);
            l.wv.delta(self.scale, &mut bw.wv);
            l.w1.delta(self.scale, &mut bw.w1);
            l.w3.delta(self.scale, &mut bw.w3);
            l.wo_a.delta(self.scale, &mut bw.wo_a);
            l.wo_b.delta(self.scale, &mut bw.wo_b);
        }
        w
    }

    /// One optimisation step: project the trainer's base-weight grads
    /// (`dL/dW_eff` from the frozen-base forward on `apply()`ed weights) to
    /// adapter grads and Adam-update `A,B`.
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        self.t += 1;
        let (scale, t) = (self.scale, self.t);
        for ((li, lt), g) in self.dbl.iter_mut().zip(grads.dbl.iter()) {
            step_stream(li, &g.img, scale, lr, t);
            step_stream(lt, &g.txt, scale, lr, t);
        }
        for (l, g) in self.sgl.iter_mut().zip(grads.sgl.iter()) {
            proj_step(&mut l.wq, &g.wq, scale, lr, t);
            proj_step(&mut l.wk, &g.wk, scale, lr, t);
            proj_step(&mut l.wv, &g.wv, scale, lr, t);
            proj_step(&mut l.w1, &g.w1, scale, lr, t);
            proj_step(&mut l.w3, &g.w3, scale, lr, t);
            proj_step(&mut l.wo_a, &g.wo_a, scale, lr, t);
            proj_step(&mut l.wo_b, &g.wo_b, scale, lr, t);
        }
    }

    /// Serialise to `(name, shape, data)` tensors —
    /// `double_blocks.{n}.{img|txt}.{leaf}.lora_{a,b}` /
    /// `single_blocks.{n}.{leaf}.lora_{a,b}`.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        let mut push = |name: String, p: &Pair| {
            out.push((format!("{name}.lora_a"), vec![p.r, p.inn], p.a.clone()));
            out.push((format!("{name}.lora_b"), vec![p.out, p.r], p.b.clone()));
        };
        for (n, (li, lt)) in self.dbl.iter().enumerate() {
            for (s, sl) in STREAMS.iter().zip([li, lt]) {
                for (leaf, p) in STREAM_LEAVES.iter().zip(stream_pairs(sl)) {
                    push(format!("double_blocks.{n}.{s}.{leaf}"), p);
                }
            }
        }
        for (n, sl) in self.sgl.iter().enumerate() {
            for (leaf, p) in SINGLE_LEAVES.iter().zip(single_pairs(sl)) {
                push(format!("single_blocks.{n}.{leaf}"), p);
            }
        }
        out
    }

    /// Reload an adapter (weights only; Adam state reset) from [`Self::to_tensors`]
    /// output — a fresh adapter of the right shape with `A,B` overwritten.
    pub fn from_tensors(
        cfg: &Cfg,
        lc: LoraCfg,
        tensors: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<LoraAdapter, String> {
        let mut ad = LoraAdapter::new(cfg, lc);
        let load = |name: String, p: &mut Pair| -> Result<(), String> {
            let a = tensors.get(&format!("{name}.lora_a")).ok_or_else(|| format!("missing {name}.lora_a"))?;
            let b = tensors.get(&format!("{name}.lora_b")).ok_or_else(|| format!("missing {name}.lora_b"))?;
            if a.len() != p.r * p.inn || b.len() != p.out * p.r {
                return Err(format!("{name}: adapter tensor size mismatch"));
            }
            p.a = a.clone();
            p.b = b.clone();
            Ok(())
        };
        for (n, (li, lt)) in ad.dbl.iter_mut().enumerate() {
            for (s, sl) in STREAMS.iter().zip([li, lt]) {
                for (leaf, p) in STREAM_LEAVES.iter().zip(stream_pairs_mut(sl)) {
                    load(format!("double_blocks.{n}.{s}.{leaf}"), p)?;
                }
            }
        }
        for (n, sl) in ad.sgl.iter_mut().enumerate() {
            for (leaf, p) in SINGLE_LEAVES.iter().zip(single_pairs_mut(sl)) {
                load(format!("single_blocks.{n}.{leaf}"), p)?;
            }
        }
        Ok(ad)
    }

    /// Fold this adapter's deltas into an **inference** tensor map (the
    /// BFL-named fused layout `Flux2Model::new` builds from), so a plain
    /// generation run produces adapter-conditioned images with no model
    /// change. Row/column offsets mirror the build-time fused → split slicing.
    pub fn fold_into_tensors(&self, ts: &mut crate::import::Tensors) -> Result<(), String> {
        let get = |ts: &mut crate::import::Tensors, key: &str, want: usize| -> Result<Vec<usize>, String> {
            match ts.get(key) {
                Some((shape, data)) if data.len() == want => Ok(shape.clone()),
                Some((_, data)) => Err(format!("lora: {key} has {} values, adapter expects {want}", data.len())),
                None => Err(format!("lora: base tensor {key} missing")),
            }
        };
        for (n, (li, lt)) in self.dbl.iter().enumerate() {
            for (s, sl) in STREAMS.iter().zip([li, lt]) {
                let d = sl.wq.inn;
                let mlp = sl.w1.out;
                let qkv = format!("double_blocks.{n}.{s}_attn.qkv.weight");
                get(ts, &qkv, 3 * d * d)?;
                let buf = &mut ts.get_mut(&qkv).unwrap().1;
                sl.wq.delta_strided(self.scale, buf, 0, d, 0);
                sl.wk.delta_strided(self.scale, buf, d, d, 0);
                sl.wv.delta_strided(self.scale, buf, 2 * d, d, 0);
                let proj = format!("double_blocks.{n}.{s}_attn.proj.weight");
                get(ts, &proj, d * d)?;
                sl.wo.delta(self.scale, &mut ts.get_mut(&proj).unwrap().1);
                let m0 = format!("double_blocks.{n}.{s}_mlp.0.weight");
                get(ts, &m0, 2 * mlp * d)?;
                let buf = &mut ts.get_mut(&m0).unwrap().1;
                sl.w1.delta_strided(self.scale, buf, 0, d, 0);
                sl.w3.delta_strided(self.scale, buf, mlp, d, 0);
                let m2 = format!("double_blocks.{n}.{s}_mlp.2.weight");
                get(ts, &m2, d * mlp)?;
                sl.w2.delta(self.scale, &mut ts.get_mut(&m2).unwrap().1);
            }
        }
        for (n, sl) in self.sgl.iter().enumerate() {
            let d = sl.wq.inn;
            let mlp = sl.w1.out;
            let l1 = format!("single_blocks.{n}.linear1.weight");
            get(ts, &l1, (3 * d + 2 * mlp) * d)?;
            let buf = &mut ts.get_mut(&l1).unwrap().1;
            sl.wq.delta_strided(self.scale, buf, 0, d, 0);
            sl.wk.delta_strided(self.scale, buf, d, d, 0);
            sl.wv.delta_strided(self.scale, buf, 2 * d, d, 0);
            sl.w1.delta_strided(self.scale, buf, 3 * d, d, 0);
            sl.w3.delta_strided(self.scale, buf, 3 * d + mlp, d, 0);
            let l2 = format!("single_blocks.{n}.linear2.weight");
            get(ts, &l2, d * (d + mlp))?;
            let buf = &mut ts.get_mut(&l2).unwrap().1;
            // linear2 [D, D+mlp]: wo_a occupies columns 0..D, wo_b columns D..D+mlp
            sl.wo_a.delta_strided(self.scale, buf, 0, d + mlp, 0);
            sl.wo_b.delta_strided(self.scale, buf, 0, d + mlp, d);
        }
        Ok(())
    }
}

fn apply_stream(l: &StreamLora, w: &mut StreamW<f32>, scale: f32) {
    l.wq.delta(scale, &mut w.wq);
    l.wk.delta(scale, &mut w.wk);
    l.wv.delta(scale, &mut w.wv);
    l.wo.delta(scale, &mut w.wo);
    l.w1.delta(scale, &mut w.w1);
    l.w3.delta(scale, &mut w.w3);
    l.w2.delta(scale, &mut w.w2);
}

fn step_stream(l: &mut StreamLora, g: &crate::grad::StreamG<f32>, scale: f32, lr: f32, t: u64) {
    proj_step(&mut l.wq, &g.wq, scale, lr, t);
    proj_step(&mut l.wk, &g.wk, scale, lr, t);
    proj_step(&mut l.wv, &g.wv, scale, lr, t);
    proj_step(&mut l.wo, &g.wo, scale, lr, t);
    proj_step(&mut l.w1, &g.w1, scale, lr, t);
    proj_step(&mut l.w3, &g.w3, scale, lr, t);
    proj_step(&mut l.w2, &g.w2, scale, lr, t);
}

fn stream_pairs(s: &StreamLora) -> [&Pair; 7] {
    [&s.wq, &s.wk, &s.wv, &s.wo, &s.w1, &s.w3, &s.w2]
}
fn stream_pairs_mut(s: &mut StreamLora) -> [&mut Pair; 7] {
    [&mut s.wq, &mut s.wk, &mut s.wv, &mut s.wo, &mut s.w1, &mut s.w3, &mut s.w2]
}
fn single_pairs(s: &SingleLora) -> [&Pair; 7] {
    [&s.wq, &s.wk, &s.wv, &s.w1, &s.w3, &s.wo_a, &s.wo_b]
}
fn single_pairs_mut(s: &mut SingleLora) -> [&mut Pair; 7] {
    [&mut s.wq, &mut s.wk, &mut s.wv, &mut s.w1, &mut s.w3, &mut s.wo_a, &mut s.wo_b]
}

/// Save an adapter to brain's checkpoint format (header
/// `{"model":"flux2-lora","rank":R,"alpha":A}`), reloadable by
/// [`load_adapter`].
pub fn save_adapter(path: &str, ad: &LoraAdapter) {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> = ad
        .to_tensors()
        .into_iter()
        .map(|(n, s, d)| (n, s.iter().map(|&x| x as u64).collect(), d))
        .collect();
    checkpoint::save(
        path,
        serde_json::json!({"model": "flux2-lora", "rank": ad.rank(), "alpha": ad.alpha()}),
        &t,
    );
}

/// Load an adapter saved by [`save_adapter`] (rank/alpha from the header;
/// Adam state reset).
pub fn load_adapter(path: &str, cfg: &Cfg) -> Result<LoraAdapter, String> {
    let c = checkpoint::load(path);
    if c.header["config"]["model"] != "flux2-lora" {
        return Err(format!("{path}: not a flux2-lora checkpoint"));
    }
    let rank = c.header["config"]["rank"].as_u64().ok_or("adapter: missing rank in header")? as usize;
    let alpha = c.header["config"]["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let map: std::collections::HashMap<String, Vec<f32>> =
        c.tensors.into_iter().map(|t| (t.name, t.data)).collect();
    LoraAdapter::from_tensors(cfg, LoraCfg { rank, alpha, seed: 0 }, &map)
}
