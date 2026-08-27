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
    /// Optimiser steps already folded into this adapter. Persisted in the
    /// checkpoint header so an interrupted run can pick up its schedule -
    /// Adam's bias correction, the sample cycle and the sigma draw are all
    /// functions of it, and restarting them at zero would silently retrain
    /// the same first steps rather than continue.
    pub fn steps_done(&self) -> u64 {
        self.t
    }
    /// Restore the step counter on reload. Separate from the tensors because
    /// it is metadata, not a parameter.
    pub fn set_steps_done(&mut self, t: u64) {
        self.t = t;
    }
    pub fn alpha(&self) -> f32 {
        self.scale * self.rank as f32
    }
    /// The delta scale `α/r` every pair's `B·A` is multiplied by.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Every adapter pair in ONE canonical order: for each double block the
    /// image stream's seven leaves then the text stream's, then each single
    /// block's seven - the same walk [`Self::to_tensors`] serialises in. The
    /// device trainer holds its own device-side pairs in this order and steps
    /// them in lockstep, so the two cannot drift.
    pub fn pairs(&self) -> Vec<&Pair> {
        let mut v = Vec::new();
        for (li, lt) in &self.dbl {
            v.extend(stream_pairs(li));
            v.extend(stream_pairs(lt));
        }
        for sl in &self.sgl {
            v.extend(single_pairs(sl));
        }
        v
    }

    /// [`Self::pairs`], mutably - what an optimiser step writes through.
    pub fn pairs_mut(&mut self) -> Vec<&mut Pair> {
        let mut v = Vec::new();
        for (li, lt) in &mut self.dbl {
            v.extend(stream_pairs_mut(li));
            v.extend(stream_pairs_mut(lt));
        }
        for sl in &mut self.sgl {
            v.extend(single_pairs_mut(sl));
        }
        v
    }

    /// Adam-step every pair from gradients already in `(dA, dB)` form -
    /// what the device trainer produces directly, without ever materialising
    /// the dense `dW` [`Self::step`] projects.
    pub fn step_projected(&mut self, grads: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        self.t += 1;
        let t = self.t;
        let mut pairs = self.pairs_mut();
        assert_eq!(pairs.len(), grads.len(), "adapter has {} pairs, got {} gradient pairs", pairs.len(), grads.len());
        // One task per pair. Adam is elementwise and pairs share nothing, so
        // this is bit-identical to the serial walk - which matters, because a
        // training trajectory that depended on the thread count would not be
        // reproducible. It is parallel because it is not small: at klein-4b
        // rank 16 the adapter is tens of millions of parameters and Adam
        // touches seven floats per parameter, which measured as the largest
        // single HOST cost of a step.
        backend_cpu::par::chunks_mut(&mut pairs, 1, |i, one| {
            let (da, db) = &grads[i];
            one[0].adam_step(da, db, lr, t);
        });
    }

    /// Build the effective weights `W_eff = W + scale·B·A` (base cloned).
    pub fn apply(&self, base: &ModelWeights<f32>) -> ModelWeights<f32> {
        let mut w = base.clone();
        self.apply_into(&mut w);
        w
    }

    /// [`Self::apply`] onto weights the caller already owns: add
    /// `scale·B·A` into `w` **in place**, no clone.
    ///
    /// The clone in `apply` is a whole fp32 copy of the model, so a caller
    /// that can produce the frozen base directly into its own buffer - by
    /// re-reading the checkpoint, say - holds one copy where `apply` holds
    /// two. Same deltas in the same order onto the same bytes, so the result
    /// is bit-identical to `apply`'s; `apply` is written in terms of it.
    ///
    /// `w` must be the **pristine** base: the deltas are additive and applying
    /// them twice is not the same model.
    pub fn apply_into(&self, w: &mut ModelWeights<f32>) {
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

    /// Open one optimisation step whose block gradients arrive **one block at
    /// a time** ([`crate::modelgrad::GradSink`]) rather than as a whole-model
    /// [`ModelGrads`]. Advances the Adam counter once, here, so every pair in
    /// the step sees the same `t` whatever order the blocks arrive in - each
    /// pair's moments are its own, so the result is identical to
    /// [`Self::step`]'s.
    ///
    /// Only the block linears are LoRA targets, so the global grads
    /// `backward_into` still returns need no step at all.
    ///
    /// The block walk below is written out rather than shared with
    /// [`Self::step`] **on purpose**: `tests/streamed_grads.rs` gates the two
    /// against each other to the bit, and a comparison whose two sides call
    /// the same projection code could not fail. Same reason `modelgrad`'s
    /// `timestep_embedding` is a deliberate second implementation of
    /// `hostmath`'s.
    pub fn stepper(&mut self, lr: f32) -> LoraStep<'_> {
        self.t += 1;
        LoraStep { scale: self.scale, t: self.t, lr, ad: self }
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

/// One in-flight optimisation step, opened by [`LoraAdapter::stepper`]: it
/// Adam-updates the pairs of each block as that block's dense gradients
/// arrive and lets the caller drop them immediately.
///
/// It is a [`crate::modelgrad::GradSink`], so `modelgrad::grads_into` drives
/// it directly and no whole-model `ModelGrads` is ever built.
pub struct LoraStep<'a> {
    ad: &'a mut LoraAdapter,
    scale: f32,
    lr: f32,
    t: u64,
}

impl LoraStep<'_> {
    /// Project + Adam-step double block `i`'s two streams.
    pub fn double_block(&mut self, i: usize, g: &crate::grad::DoubleGrads<f32>) {
        let (li, lt) = &mut self.ad.dbl[i];
        step_stream(li, &g.img, self.scale, self.lr, self.t);
        step_stream(lt, &g.txt, self.scale, self.lr, self.t);
    }
    /// Project + Adam-step single block `i`.
    pub fn single_block(&mut self, i: usize, g: &crate::grad::SingleGrads<f32>) {
        let (scale, lr, t) = (self.scale, self.lr, self.t);
        let l = &mut self.ad.sgl[i];
        proj_step(&mut l.wq, &g.wq, scale, lr, t);
        proj_step(&mut l.wk, &g.wk, scale, lr, t);
        proj_step(&mut l.wv, &g.wv, scale, lr, t);
        proj_step(&mut l.w1, &g.w1, scale, lr, t);
        proj_step(&mut l.w3, &g.w3, scale, lr, t);
        proj_step(&mut l.wo_a, &g.wo_a, scale, lr, t);
        proj_step(&mut l.wo_b, &g.wo_b, scale, lr, t);
    }
}

impl crate::modelgrad::GradSink<f32> for LoraStep<'_> {
    fn double(&mut self, i: usize, g: crate::grad::DoubleGrads<f32>) {
        self.double_block(i, &g);
    }
    fn single(&mut self, i: usize, g: crate::grad::SingleGrads<f32>) {
        self.single_block(i, &g);
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

/// What [`fold_external_adapter`] folded, for the caller to log. A run that
/// claims to be adapted should be able to say how much of the model it moved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalFold {
    /// Adapted linears (klein-9b's own full-coverage adapters have 112).
    pub pairs: usize,
    /// The file's rank, or the largest one if it is not uniform.
    pub rank: usize,
    /// The `strength` the delta was scaled by.
    pub scale: f32,
}

/// Fold a THIRD-PARTY (ai-toolkit / ComfyUI / diffusers) LoRA `.safetensors`
/// into the inference tensor map, so an unchanged generation run produces
/// adapter-conditioned images.
///
/// This is the OTHER direction from [`load_adapter`]: that one reloads an
/// adapter brain itself trained (brain's checkpoint container, per-slice pairs
/// over `q`/`k`/`v` separately). A third-party file instead adapts the FUSED
/// matrices - one shared `A` for the whole `qkv`, one for the whole
/// `linear1` - which is not a shape [`LoraAdapter`] can hold, but is a
/// strictly simpler fold: every target is a whole tensor at offset 0, so
/// [`model::lora::Pair::delta`] is the exact operation.
///
/// ## Semantics, taken from the reference implementations
///
/// `W += strength · (alpha/r) · B·A`, matching ComfyUI's weight adapter
/// (`comfy/weight_adapter/lora.py`: `weight += (strength * alpha) * mm(mat1,
/// mat2)` with `mat1` the up/`lora_B` and `mat2` the down/`lora_A`, and
/// `alpha = v[2]/rank` or `1.0` when no `.alpha` tensor is present) and
/// ai-toolkit's trainer (`toolkit/network_mixins.py`: `scale = alpha /
/// lora_dim`, alpha initialised to the rank and stripped from PEFT-format
/// saves). `B·A` needs no transpose: both store PyTorch `nn.Linear` weights
/// `[out, in]`, which is already brain's row-major manifest layout.
///
/// `scale` is ComfyUI's `strength_model` - a user dial, default 1.0, NOT a
/// value read from the file.
///
/// Every pair is validated against the base map BEFORE anything is written,
/// so a rejected adapter leaves the weights untouched rather than half folded.
pub fn fold_external_adapter(
    path: &str,
    ts: &mut crate::import::Tensors,
    scale: f32,
) -> Result<ExternalFold, String> {
    let pairs = model::lora::read_external_adapter(path)?;
    // Validate the WHOLE adapter first. A key that matches nothing is a hard
    // error naming the tensor: silently skipping it would return base-model
    // output from a run the user believes is adapted.
    for p in &pairs {
        match ts.get(&p.base_key) {
            None => {
                return Err(format!(
                    "lora {path}: adapter targets '{}' (from '{}'), which this FLUX.2 variant \
                     does not have - wrong base model for this adapter?",
                    p.base_key, p.stem
                ))
            }
            Some((shape, data)) => {
                if shape.as_slice() != [p.out, p.inn] {
                    return Err(format!(
                        "lora {path}: '{}' is {shape:?}, but the adapter for it is [{}, {}]",
                        p.base_key, p.out, p.inn
                    ));
                }
                if data.len() != p.out * p.inn {
                    return Err(format!(
                        "lora {path}: '{}' holds {} values, expected {}",
                        p.base_key,
                        data.len(),
                        p.out * p.inn
                    ));
                }
            }
        }
    }
    let rank = pairs.iter().map(|p| p.r).max().unwrap_or(0);
    for p in &pairs {
        let w = &mut ts.get_mut(&p.base_key).expect("validated above").1;
        p.as_pair().delta(scale * p.alpha_mult, w);
    }
    Ok(ExternalFold { pairs: pairs.len(), rank, scale })
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
        serde_json::json!({"model": "flux2-lora", "rank": ad.rank(), "alpha": ad.alpha(), "steps": ad.steps_done()}),
        &t,
    );
}

/// Load an adapter saved by [`save_adapter`] (rank/alpha from the header).
///
/// The step counter is restored from the header's `steps` when present - a
/// checkpoint written before that field existed simply resumes from 0. The
/// Adam MOMENTS are not stored and do reset: they are a few hundred MB of
/// state that no inference path reads, and a restarted moment estimate costs
/// a short warm-up (the bias correction divides a zero moment by
/// `1 - beta^t`, so the first resumed updates are small and grow back) rather
/// than a wrong answer.
pub fn load_adapter(path: &str, cfg: &Cfg) -> Result<LoraAdapter, String> {
    let c = checkpoint::load(path);
    if c.header["config"]["model"] != "flux2-lora" {
        return Err(format!("{path}: not a flux2-lora checkpoint"));
    }
    let rank = c.header["config"]["rank"].as_u64().ok_or("adapter: missing rank in header")? as usize;
    let alpha = c.header["config"]["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let map: std::collections::HashMap<String, Vec<f32>> =
        c.tensors.into_iter().map(|t| (t.name, t.data)).collect();
    let steps = c.header["config"]["steps"].as_u64().unwrap_or(0);
    let mut ad = LoraAdapter::from_tensors(cfg, LoraCfg { rank, alpha, seed: 0 }, &map)?;
    ad.set_steps_done(steps);
    Ok(ad)
}
