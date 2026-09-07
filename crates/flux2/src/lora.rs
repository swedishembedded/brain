// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for the FLUX.2 Klein DiT, over the generic
//! `model::adapter` substrate.
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
//! ## Two representations, on purpose
//!
//! [`apply`](LoraAdapter::apply)/[`step`](LoraAdapter::step)/[`to_tensors`]/
//! [`from_tensors`] all operate against the host trainer's SPLIT weights
//! ([`ModelWeights`]/[`ModelGrads`]: `wq`/`wk`/`wv` are already three
//! independent `[d,d]` tensors, not slices of one `[3d,d]` buffer) - every
//! [`model::adapter::LinearSite`] here therefore uses
//! [`model::adapter::TargetSpec::whole`], offset zero, because that IS the
//! whole tensor in this representation.
//!
//! [`LoraAdapter::fold_into_tensors_at`] instead targets the FUSED inference
//! layout (`qkv`/`mlp.0`/`linear1`/`linear2`), where the very same `A,B`
//! pairs land at real row/column offsets inside a shared buffer. That fold
//! is deliberately NOT routed through [`model::adapter::AdapterSet::fold_into`]
//! (which assumes one site owns one whole destination tensor) - it reaches
//! into each [`model::lora::LoraPair`]'s underlying [`Pair`] via
//! [`model::lora::LoraPair::pair`] and describes each one's fused offset as a
//! [`model::lora::Placement`], leaving the validate-then-write arithmetic to
//! [`model::lora::fold_placements`].
//!
//! Targets, per the fused-checkpoint layout:
//! * double block, per stream: `qkv` → three row-slice pairs (q/k/v), `proj`,
//!   `mlp.0` → two row-slice pairs (w1/w3), `mlp.2`;
//! * single block: `linear1` → five row-slice pairs (q/k/v/w1/w3), `linear2` →
//!   two column-split pairs (`wo_a`/`wo_b`).
//!
//! Serialization: brain's `checkpoint` container, header
//! `{"model":"flux2-lora","rank":R,"alpha":A}`.

use crate::grad::{SingleW, StreamG, StreamW};
use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
// The generic pair machinery (A/B init, ΔW apply, dW→(dA,dB) projection, Adam
// moments) is model-agnostic and lives ONCE in `model::lora` - this module
// keeps only the FLUX.2-specific block walk, fused-tensor offsets and
// serialization naming. `LoraCfg` is re-exported for existing callers.
pub use model::lora::{ExternalFold, LoraCfg};
use model::lora::{fold_placements, proj_step, LoraGrads, LoraPair, Pair, Placement};

/// How FLUX.2 names itself in a wrong-base-model adapter error. One spelling,
/// because that message is the one a user reads when an adapter trained for
/// another model is loaded, and it has to say which base was expected.
const ARCH: &str = "FLUX.2";

const STREAMS: [&str; 2] = ["img", "txt"];
/// Short leaf names, in the fixed order the walk/serializer/fold all share -
/// one table so they cannot disagree about which tensor is which.
const STREAM_LEAVES: [&str; 7] = ["wq", "wk", "wv", "wo", "w1", "w3", "w2"];
const SINGLE_LEAVES: [&str; 7] = ["wq", "wk", "wv", "w1", "w3", "wo_a", "wo_b"];

/// `site.leaf` for a double-block entry is `"{stream}.{short}"` (e.g.
/// `"img.wq"`) - a compound literal, not a parsed string, so [`apply_into`]/
/// [`step`](LoraAdapter::step) can dispatch on it directly the same way
/// `wan::lora`/`ltxv::lora` already match compound leaves like
/// `"cross_attn.q"`. All literals, so no allocation is needed to keep
/// [`LinearSite::leaf`]'s `&'static str` bound.
fn compound_leaf(stream: &str, short: &str) -> &'static str {
    match (stream, short) {
        ("img", "wq") => "img.wq",
        ("img", "wk") => "img.wk",
        ("img", "wv") => "img.wv",
        ("img", "wo") => "img.wo",
        ("img", "w1") => "img.w1",
        ("img", "w3") => "img.w3",
        ("img", "w2") => "img.w2",
        ("txt", "wq") => "txt.wq",
        ("txt", "wk") => "txt.wk",
        ("txt", "wv") => "txt.wv",
        ("txt", "wo") => "txt.wo",
        ("txt", "w1") => "txt.w1",
        ("txt", "w3") => "txt.w3",
        ("txt", "w2") => "txt.w2",
        other => panic!("flux2 lora: unknown stream leaf {other:?}"),
    }
}

fn stream_shape(short: &str, d: usize, mlp: usize) -> (usize, usize) {
    match short {
        "wq" | "wk" | "wv" | "wo" => (d, d),
        "w1" | "w3" => (mlp, d),
        "w2" => (d, mlp),
        other => panic!("flux2 lora: unknown stream leaf {other:?}"),
    }
}

fn single_shape(short: &str, d: usize, mlp: usize) -> (usize, usize) {
    match short {
        "wq" | "wk" | "wv" => (d, d),
        "w1" | "w3" => (mlp, d),
        "wo_a" => (d, d),
        "wo_b" => (d, mlp),
        other => panic!("flux2 lora: unknown single leaf {other:?}"),
    }
}

fn stream_field_mut<'a>(s: &'a mut StreamW<f32>, short: &str) -> &'a mut Vec<f32> {
    match short {
        "wq" => &mut s.wq,
        "wk" => &mut s.wk,
        "wv" => &mut s.wv,
        "wo" => &mut s.wo,
        "w1" => &mut s.w1,
        "w3" => &mut s.w3,
        "w2" => &mut s.w2,
        other => panic!("flux2 lora: unknown stream leaf {other:?}"),
    }
}

fn stream_field<'a>(g: &'a StreamG<f32>, short: &str) -> &'a Vec<f32> {
    match short {
        "wq" => &g.wq,
        "wk" => &g.wk,
        "wv" => &g.wv,
        "wo" => &g.wo,
        "w1" => &g.w1,
        "w3" => &g.w3,
        "w2" => &g.w2,
        other => panic!("flux2 lora: unknown stream leaf {other:?}"),
    }
}

fn single_field_mut<'a>(s: &'a mut SingleW<f32>, short: &str) -> &'a mut Vec<f32> {
    match short {
        "wq" => &mut s.wq,
        "wk" => &mut s.wk,
        "wv" => &mut s.wv,
        "w1" => &mut s.w1,
        "w3" => &mut s.w3,
        "wo_a" => &mut s.wo_a,
        "wo_b" => &mut s.wo_b,
        other => panic!("flux2 lora: unknown single leaf {other:?}"),
    }
}

fn single_field<'a>(g: &'a crate::grad::SingleGrads<f32>, short: &str) -> &'a Vec<f32> {
    match short {
        "wq" => &g.wq,
        "wk" => &g.wk,
        "wv" => &g.wv,
        "w1" => &g.w1,
        "w3" => &g.w3,
        "wo_a" => &g.wo_a,
        "wo_b" => &g.wo_b,
        other => panic!("flux2 lora: unknown single leaf {other:?}"),
    }
}

/// Every linear this crate offers to a PEFT adapter: for each double block,
/// the image stream's seven leaves then the text stream's; then each single
/// block's seven - the SAME canonical order [`LoraAdapter::new`] drew its
/// (now-superseded) per-field random init in, so every existing seed
/// reproduces a bit-identical adapter. All specs are
/// [`TargetSpec::whole`] - see the module doc for why (this walk targets the
/// SPLIT host representation, never the fused one).
pub fn linear_sites(cfg: &Cfg) -> Vec<LinearSite> {
    let (d, mlp) = (cfg.hidden, cfg.mlp);
    let mut sites = Vec::with_capacity(cfg.depth_double * 14 + cfg.depth_single * 7);
    for n in 0..cfg.depth_double {
        for s in STREAMS {
            for short in STREAM_LEAVES {
                let (out, inn) = stream_shape(short, d, mlp);
                sites.push(LinearSite {
                    name: format!("double_blocks.{n}.{s}.{short}"),
                    leaf: compound_leaf(s, short),
                    layer: Some(n),
                    spec: TargetSpec::whole(out, inn),
                    save_name: None,
                });
            }
        }
    }
    for n in 0..cfg.depth_single {
        for short in SINGLE_LEAVES {
            let (out, inn) = single_shape(short, d, mlp);
            sites.push(LinearSite { name: format!("single_blocks.{n}.{short}"), leaf: short, layer: Some(n), spec: TargetSpec::whole(out, inn), save_name: None });
        }
    }
    sites
}

/// A LoRA adapter over every double- and single-block linear of the DiT
/// (qk-norm scales and the global embed/modulation/final linears stay frozen).
pub struct LoraAdapter {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
    depth_double: usize,
}

impl LoraAdapter {
    /// Fresh adapter (B=0 → initial no-op) sized for `cfg`.
    pub fn new(cfg: &Cfg, lc: LoraCfg) -> LoraAdapter {
        Self::new_with_hp(cfg, TargetHp::from(lc), lc.seed)
    }

    /// [`Self::new`], with every [`TargetHp`] field a caller wants (rsLoRA,
    /// LoRA+'s `lr_ratio`, LoRA-FA's `freeze_a`) instead of only what
    /// [`LoraCfg`] carries - what `brain flux2 finetune --method/--lr-ratio/
    /// --freeze-a` builds from. `seed` is separate from `hp` because
    /// `TargetHp` has no seed field (it is a training-time draw, not an
    /// adapter hyperparameter).
    pub fn new_with_hp(cfg: &Cfg, hp: TargetHp, seed: u64) -> LoraAdapter {
        let sites = linear_sites(cfg);
        let mut rng = data::rng::Rng::new(seed ^ 0xf1a2_b3c4_d5e6_0789);
        // Same init distribution as before the model::lora hoist (uniform,
        // ±0.02) so existing seeds reproduce bit-identical adapters.
        let mut init = move || (rng.next_f64() - 0.5) as f32 * 0.04;
        let set = AdapterSet::build(sites, hp, KeyStyle::Brain, &mut init);
        LoraAdapter { set, hp, depth_double: cfg.depth_double }
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }
    /// The full hyperparameter set this adapter was built with - what a
    /// caller checks a resumed run's request against (see `finetune::run`).
    pub fn hp(&self) -> TargetHp {
        self.hp
    }
    /// Optimiser steps already folded into this adapter. Persisted in the
    /// checkpoint header so an interrupted run can pick up its schedule -
    /// Adam's bias correction, the sample cycle and the sigma draw are all
    /// functions of it, and restarting them at zero would silently retrain
    /// the same first steps rather than continue.
    pub fn steps_done(&self) -> u64 {
        self.set.steps()
    }
    /// Restore the step counter on reload. Separate from the tensors because
    /// it is metadata, not a parameter.
    pub fn set_steps_done(&mut self, t: u64) {
        self.set.set_steps(t);
    }
    pub fn alpha(&self) -> f32 {
        self.hp.alpha
    }
    /// The delta scale `α/r` every pair's `B·A` is multiplied by.
    pub fn scale(&self) -> f32 {
        self.hp.scale()
    }

    /// Every adapter pair in ONE canonical order: for each double block the
    /// image stream's seven leaves then the text stream's, then each single
    /// block's seven - the same walk [`Self::to_tensors`] serialises in. The
    /// device trainer holds its own device-side pairs in this order and steps
    /// them in lockstep, so the two cannot drift.
    pub fn pairs(&self) -> Vec<&Pair> {
        self.set.iter().map(|(_, k)| k.pair()).collect()
    }

    /// [`Self::pairs`], mutably - what an optimiser step writes through.
    pub fn pairs_mut(&mut self) -> Vec<&mut Pair> {
        self.set.iter_mut().map(|(_, k)| k.pair_mut()).collect()
    }

    /// Adam-step every pair from gradients already in `(dA, dB)` form -
    /// what the device trainer produces directly, without ever materialising
    /// the dense `dW` [`Self::step`] projects.
    pub fn step_projected(&mut self, grads: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        assert_eq!(self.set.len(), grads.len(), "adapter has {} pairs, got {} gradient pairs", self.set.len(), grads.len());
        let owned: Vec<LoraGrads> = grads.iter().map(|(da, db)| LoraGrads { da: da.clone(), db: db.clone() }).collect();
        // Chunk length 1 inside AdapterSet::step_projected - bit-identical to
        // a serial walk, which matters because a training trajectory that
        // depended on the thread count would not be reproducible. It is
        // parallel because it is not small: at klein-4b rank 16 the adapter
        // is tens of millions of parameters and Adam touches seven floats
        // per parameter, measured as the largest single HOST cost of a step.
        self.set.step_projected(&owned, lr);
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
        for (site, k) in self.set.iter() {
            let n = site.layer.expect("flux2 lora: every site has a layer index");
            if let Some(short) = site.leaf.strip_prefix("img.") {
                k.delta_into(1.0, stream_field_mut(&mut w.dbl[n].img, short));
            } else if let Some(short) = site.leaf.strip_prefix("txt.") {
                k.delta_into(1.0, stream_field_mut(&mut w.dbl[n].txt, short));
            } else {
                k.delta_into(1.0, single_field_mut(&mut w.sgl[n], site.leaf));
            }
        }
    }

    /// One optimisation step: project the trainer's base-weight grads
    /// (`dL/dW_eff` from the frozen-base forward on `apply()`ed weights) to
    /// adapter grads and Adam-update `A,B`.
    pub fn step(&mut self, grads: &ModelGrads<f32>, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let n = site.layer.expect("flux2 lora: every site has a layer index");
                if let Some(short) = site.leaf.strip_prefix("img.") {
                    k.project(stream_field(&grads.dbl[n].img, short))
                } else if let Some(short) = site.leaf.strip_prefix("txt.") {
                    k.project(stream_field(&grads.dbl[n].txt, short))
                } else {
                    k.project(single_field(&grads.sgl[n], site.leaf))
                }
            })
            .collect();
        self.set.step_projected(&projected, lr);
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
    /// against each other to the bit, and a comparison whose two sides
    /// called the same WALK could not fail (both still call into the same
    /// shared [`model::adapter::AdapterKind::project`]/`step` primitives -
    /// that sharing is the point of this crate; what stays independent here
    /// is which entries get grouped into one pass). Same reason
    /// `modelgrad`'s `timestep_embedding` is a deliberate second
    /// implementation of `hostmath`'s.
    pub fn stepper(&mut self, lr: f32) -> LoraStep<'_> {
        let t = self.set.steps() + 1;
        self.set.set_steps(t);
        LoraStep { t, lr, depth_double: self.depth_double, set: &mut self.set }
    }

    /// Serialise to `(name, shape, data)` tensors —
    /// `double_blocks.{n}.{img|txt}.{leaf}.lora_{a,b}` /
    /// `single_blocks.{n}.{leaf}.lora_{a,b}`.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
    }

    /// Reload an adapter (weights only; Adam state reset) from [`Self::to_tensors`]
    /// output — a fresh adapter of the right shape with `A,B` overwritten.
    pub fn from_tensors(cfg: &Cfg, lc: LoraCfg, tensors: &std::collections::HashMap<String, Vec<f32>>) -> Result<LoraAdapter, String> {
        Self::from_tensors_with_hp(cfg, TargetHp::from(lc), tensors)
    }

    /// [`Self::from_tensors`], with a full [`TargetHp`] - see [`Self::new_with_hp`].
    pub fn from_tensors_with_hp(cfg: &Cfg, hp: TargetHp, tensors: &std::collections::HashMap<String, Vec<f32>>) -> Result<LoraAdapter, String> {
        let sites = linear_sites(cfg);
        // flux2's own checkpoint round-trip never carried shapes (`load_adapter`
        // reads a plain name->data map) - wrap with an empty shape per tensor
        // so `LoraPair::load_tensors` falls back to its length check, the
        // same tolerance ltxv's adapter files already need.
        let shaped: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> = tensors.iter().map(|(k, v)| (k.clone(), (Vec::new(), v.clone()))).collect();
        let set = AdapterSet::from_tensors(sites, hp, KeyStyle::Brain, &shaped)?;
        Ok(LoraAdapter { set, hp, depth_double: cfg.depth_double })
    }

    /// Fold this adapter's deltas into an **inference** tensor map (the
    /// BFL-named fused layout `Flux2Model::new` builds from), so a plain
    /// generation run produces adapter-conditioned images with no model
    /// change. Row/column offsets mirror the build-time fused → split slicing.
    pub fn fold_into_tensors(&self, ts: &mut crate::import::Tensors) -> Result<(), String> {
        self.fold_into_tensors_at(ts, 1.0)
    }

    /// [`Self::fold_into_tensors`] with the caller's **strength** on top of the
    /// checkpoint's own `α/r` - the ComfyUI `strength_model` dial, and what
    /// `brain flux2 generate --lora-scale` means.
    ///
    /// It multiplies the header alpha rather than replacing it, so `1.0` is
    /// exactly the unscaled fold and an adapter keeps the strength it was
    /// trained to want as its default. `0.0` reproduces the base bit-for-bit,
    /// which is the honest way to see what an adapter actually contributes.
    ///
    /// Strength lives here, not on the adapter's own `scale` field, because
    /// that field is the *training* parameter the optimiser steps against:
    /// bending it for inference would silently change what a resumed run
    /// trains.
    pub fn fold_into_tensors_at(&self, ts: &mut crate::import::Tensors, strength: f32) -> Result<(), String> {
        fold_placements(ts, self.scale() * strength, &self.placements())
    }

    /// Where every pair's delta lands in the BFL-named fused layout - the
    /// FLUX.2-specific half of a fold, and the only half this crate owns
    /// ([`model::lora::fold_placements`] owns the validation contract and the
    /// arithmetic). Row/column offsets mirror `model.rs`'s build-time
    /// fused → split slicing, which is what makes a folded adapter and an
    /// `apply`ed one the same model.
    fn placements(&self) -> Vec<Placement<'_>> {
        let mut v = Vec::new();
        for n in 0..self.depth_double {
            let block = &self.set.as_slice()[n * 14..n * 14 + 14];
            for (j, s) in STREAMS.iter().enumerate() {
                // Positional, not a leaf search over the whole model: every
                // block shares the same compound leaf strings ("img.wq" etc),
                // so searching unscoped would always find block 0's pair.
                let stream = &block[j * 7..j * 7 + 7];
                let pair_for = |short: &str| {
                    let idx = STREAM_LEAVES.iter().position(|&l| l == short).expect("known leaf");
                    stream[idx].1.pair()
                };
                let (wq, wk, wv, wo, w1, w3, w2) = (pair_for("wq"), pair_for("wk"), pair_for("wv"), pair_for("wo"), pair_for("w1"), pair_for("w3"), pair_for("w2"));
                let d = wq.inn;
                let mlp = w1.out;
                let qkv = format!("double_blocks.{n}.{s}_attn.qkv.weight");
                let nq = 3 * d * d;
                v.push(Placement::fused(qkv.clone(), wq, nq, 0, d, 0));
                v.push(Placement::fused(qkv.clone(), wk, nq, d, d, 0));
                v.push(Placement::fused(qkv, wv, nq, 2 * d, d, 0));
                v.push(Placement::whole(format!("double_blocks.{n}.{s}_attn.proj.weight"), wo));
                let m0 = format!("double_blocks.{n}.{s}_mlp.0.weight");
                let nm = 2 * mlp * d;
                v.push(Placement::fused(m0.clone(), w1, nm, 0, d, 0));
                v.push(Placement::fused(m0, w3, nm, mlp, d, 0));
                v.push(Placement::whole(format!("double_blocks.{n}.{s}_mlp.2.weight"), w2));
            }
        }
        let single_start = self.depth_double * 14;
        for n in 0..(self.set.len() - single_start) / 7 {
            let base = single_start + n * 7;
            let entries = &self.set.as_slice()[base..base + 7];
            let pair_for = |short: &str| entries.iter().find(|(site, _)| site.leaf == short).map(|(_, k)| k.pair()).expect("site exists");
            let (wq, wk, wv, w1, w3, wo_a, wo_b) = (pair_for("wq"), pair_for("wk"), pair_for("wv"), pair_for("w1"), pair_for("w3"), pair_for("wo_a"), pair_for("wo_b"));
            let d = wq.inn;
            let mlp = w1.out;
            let l1 = format!("single_blocks.{n}.linear1.weight");
            let n1 = (3 * d + 2 * mlp) * d;
            v.push(Placement::fused(l1.clone(), wq, n1, 0, d, 0));
            v.push(Placement::fused(l1.clone(), wk, n1, d, d, 0));
            v.push(Placement::fused(l1.clone(), wv, n1, 2 * d, d, 0));
            v.push(Placement::fused(l1.clone(), w1, n1, 3 * d, d, 0));
            v.push(Placement::fused(l1, w3, n1, 3 * d + mlp, d, 0));
            // linear2 [D, D+mlp]: wo_a occupies columns 0..D, wo_b columns D..D+mlp
            let l2 = format!("single_blocks.{n}.linear2.weight");
            let n2 = d * (d + mlp);
            v.push(Placement::fused(l2.clone(), wo_a, n2, 0, d + mlp, 0));
            v.push(Placement::fused(l2, wo_b, n2, 0, d + mlp, d));
        }
        v
    }
}

/// Fold EVERY adapter in `adapters` into the DiT tensor map, in list order,
/// each at its own `AdapterSpec::scale` - the one seam every `Pipeline`
/// builder's LoRA handling goes through, so a stacked run's effect on the
/// weights is testable without a checkpoint.
///
/// ## Order, and what stacking actually does to a weight
///
/// Adapter *n+1* folds onto the map adapter *n* already changed. The deltas
/// are additive, so a linear adapted by several of them ends at
/// `W + Σᵢ sᵢ·(αᵢ/rᵢ)·Bᵢ·Aᵢ`: each `--lora-scale` multiplies only its own
/// adapter's delta, and adapters over the same linear **sum** there. They do
/// not average and the later one does not win - so a face adapter and a style
/// adapter both at 1.0 move their shared linears by the sum of two deltas,
/// each of which was trained (and validated) alone. That is the dial to turn
/// down when a stack over-cooks; the list order itself reaches the result only
/// through float rounding, and is fixed rather than left to iteration order so
/// the same command is reproducible.
///
/// An empty list returns immediately: no file is opened and `ts` is not
/// touched, so an unadapted build is exactly what it was before stacking
/// existed.
///
/// Each path picks its own family by extension - a `.safetensors` is a
/// third-party (ai-toolkit / ComfyUI / diffusers) adapter over the fused
/// matrices, anything else is brain's own trained container - and the two
/// families may be mixed freely in one list.
pub fn fold_adapters(
    cfg: &crate::Flux2Config,
    ts: &mut crate::import::Tensors,
    adapters: &[crate::AdapterSpec],
) -> Result<Vec<model::lora::FoldReport>, String> {
    let specs: Vec<(&str, f32)> = adapters.iter().map(|a| (a.path.as_str(), a.scale)).collect();
    // An adapter's tensor shapes depend only on the architecture, not the
    // latent grid, so any (lh, lw) loads one.
    let tcfg = crate::modelgrad::Cfg::from_flux2(cfg, 1, 1);
    model::lora::fold_adapter_files(ts, &specs, ARCH, |path, ts, strength| {
        let ad = load_adapter(path, &tcfg)?;
        // `strength` multiplies the checkpoint's own alpha, exactly as it does
        // on the third-party branch - a strength the model ignores is worse
        // than no strength.
        ad.fold_into_tensors_at(ts, strength)?;
        Ok((ad.pairs().len(), ad.rank()))
    })
}

/// One in-flight optimisation step, opened by [`LoraAdapter::stepper`]: it
/// Adam-updates the pairs of each block as that block's dense gradients
/// arrive and lets the caller drop them immediately.
///
/// It is a [`crate::modelgrad::GradSink`], so `modelgrad::grads_into` drives
/// it directly and no whole-model `ModelGrads` is ever built.
pub struct LoraStep<'a> {
    set: &'a mut AdapterSet<LoraPair>,
    depth_double: usize,
    lr: f32,
    t: u64,
}

impl LoraStep<'_> {
    /// Project + Adam-step double block `i`'s two streams.
    pub fn double_block(&mut self, i: usize, g: &crate::grad::DoubleGrads<f32>) {
        let base = i * 14;
        for (j, s) in STREAMS.iter().enumerate() {
            let gs = if *s == "img" { &g.img } else { &g.txt };
            for (k, short) in STREAM_LEAVES.iter().enumerate() {
                let (_, pair) = &mut self.set.as_mut_slice()[base + j * 7 + k];
                let dw = stream_field(gs, short);
                pair.proj_step(dw, self.lr, self.t);
            }
        }
    }
    /// Project + Adam-step single block `i`.
    pub fn single_block(&mut self, i: usize, g: &crate::grad::SingleGrads<f32>) {
        let base = self.depth_double * 14 + i * 7;
        for (k, short) in SINGLE_LEAVES.iter().enumerate() {
            let (_, pair) = &mut self.set.as_mut_slice()[base + k];
            let dw = single_field(g, short);
            pair.proj_step(dw, self.lr, self.t);
        }
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


/// Fold a THIRD-PARTY (ai-toolkit / ComfyUI / diffusers) LoRA `.safetensors`
/// into the inference tensor map, so an unchanged generation run produces
/// adapter-conditioned images.
///
/// This is the OTHER direction from [`load_adapter`]: that one reloads an
/// adapter brain itself trained (brain's checkpoint container, per-slice pairs
/// over `q`/`k`/`v` separately). A third-party file instead adapts the FUSED
/// matrices - one shared `A` for the whole `qkv`, one for the whole
/// `linear1` - which is not a shape [`LoraAdapter`] can hold, but is a
/// strictly simpler fold: every target is a whole tensor at offset 0.
///
/// A thin wrapper over [`model::lora::fold_external_into`] supplying this
/// architecture's own name for the wrong-base-model message; the reference
/// semantics (`W += strength·(alpha/r)·B·A`, per ComfyUI's weight adapter and
/// ai-toolkit's trainer) and the validate-before-writing contract are
/// documented there, once, for every model that reads such a file.
pub fn fold_external_adapter(
    path: &str,
    ts: &mut crate::import::Tensors,
    scale: f32,
) -> Result<ExternalFold, String> {
    model::lora::fold_external_into(path, ts, scale, ARCH)
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
    let hp = ad.hp();
    checkpoint::save(
        path,
        serde_json::json!({
            "model": "flux2-lora", "rank": ad.rank(), "alpha": ad.alpha(), "steps": ad.steps_done(),
            // Additive: a reader from before these existed uses their
            // documented defaults (see load_adapter) and gets plain LoRA -
            // exactly what every adapter saved before this change was.
            "rs": hp.rank_stabilized, "lr_ratio": hp.lr_ratio, "freeze_a": hp.freeze_a,
        }),
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
    // Additive fields, defaulting to plain LoRA for a file saved before they
    // existed: no rank-stabilization, no LoRA+ ratio, A trainable.
    let rank_stabilized = c.header["config"]["rs"].as_bool().unwrap_or(false);
    let lr_ratio = c.header["config"]["lr_ratio"].as_f64().unwrap_or(1.0) as f32;
    let freeze_a = c.header["config"]["freeze_a"].as_bool().unwrap_or(false);
    let hp = TargetHp { rank, alpha, rank_stabilized, dropout: 0.0, lr_ratio, freeze_a };
    let map: std::collections::HashMap<String, Vec<f32>> =
        c.tensors.into_iter().map(|t| (t.name, t.data)).collect();
    let steps = c.header["config"]["steps"].as_u64().unwrap_or(0);
    let mut ad = LoraAdapter::from_tensors_with_hp(cfg, hp, &map)?;
    ad.set_steps_done(steps);
    Ok(ad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `linear_sites` must reproduce the pre-migration draw order exactly:
    /// double blocks (each stream's 7 leaves in `STREAM_LEAVES` order),
    /// then single blocks (`SINGLE_LEAVES` order) - the order every existing
    /// seed's random init depends on.
    #[test]
    fn linear_sites_reproduce_the_legacy_draw_order() {
        let cfg = Cfg { depth_double: 2, depth_single: 3, hidden: 8, mlp: 16, ..Cfg::tiny() };
        let sites = linear_sites(&cfg);
        assert_eq!(sites.len(), 2 * 14 + 3 * 7);
        let mut i = 0;
        for n in 0..cfg.depth_double {
            for s in STREAMS {
                for short in STREAM_LEAVES {
                    assert_eq!(sites[i].name, format!("double_blocks.{n}.{s}.{short}"));
                    assert_eq!(sites[i].leaf, compound_leaf(s, short));
                    assert_eq!(sites[i].layer, Some(n));
                    i += 1;
                }
            }
        }
        for n in 0..cfg.depth_single {
            for short in SINGLE_LEAVES {
                assert_eq!(sites[i].name, format!("single_blocks.{n}.{short}"));
                assert_eq!(sites[i].leaf, short);
                assert_eq!(sites[i].layer, Some(n));
                i += 1;
            }
        }
    }
}
