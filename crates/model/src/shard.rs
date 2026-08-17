// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic layer sharding across devices - the ONE home for "split a layered
//! model's layers into contiguous per-device stages", for every model crate.
//!
//! # Layering
//!
//! This is infrastructure, not a model. `crates/model` depends only on
//! `gpu-core`/`checkpoint`/`kernels`/`paramstore`/`optim`/`data` - no model
//! family - so `qwen3`, `omni` and anything added later depend DOWN onto this
//! and supply a small adapter each. Nothing here knows what a Qwen or an Omni
//! is, how many cards the box has, or whether they are the same size.
//!
//! The neighbouring pieces, and why they are not this:
//! * `residency` decides POLICY over devices and bytes - budgets, placement,
//!   eviction, multi-device claims. It deliberately has no model concepts at
//!   all (no layers, no weights), which is what keeps it unit-testable
//!   without a GPU. It is the consumer of a plan, not the maker of one.
//! * `paramstore::upload` moves the bytes once a plan exists (bounded
//!   disk→VRAM streaming).
//! * A model crate's own `shard.rs` (e.g. `qwen3`) is an ADAPTER - a
//!   [`Shardable`] impl or a per-layer byte cost - never a second copy of the
//!   partitioning logic.
//!
//! # Two planners, two questions
//!
//! * [`plan_balanced`] over [`ShardCost`] - "which split is most even?", in
//!   an abstract unit, for the training [`Pipeline`] below, where the cards
//!   are assumed interchangeable.
//! * [`plan_by_capacity`]/[`plan_fewest_devices`] over [`LayerBytes`] -
//!   "does this fit, and where?", in real bytes against each device's real
//!   capacity. This is what a resident model loading real weights needs, and
//!   it handles any device count, unequal VRAM, and honest infeasibility.
//!
//! # Pipeline-parallel training (the rest of this file)
//!
//! A model's decoder layers are split into contiguous ranges — one **stage** per
//! GPU — so a model whose weights exceed a single card fits across several. The
//! only tensor crossing a stage boundary is the residual stream (one
//! `[b·t·d_model]` slab): forward carries it host-staged from stage *i* to *i+1*,
//! backward carries its gradient back. This traffic is tiny and **the same size
//! at every possible cut** (the residual width is uniform), so where to cut is
//! purely a *load-balancing* choice, not a transfer-minimising one — [`plan_balanced`]
//! places the cuts to balance per-stage memory (the embed and head stages carry
//! the big embedding / lm_head, so they get fewer layers), which both maximises
//! the model size that fits and evens out the pipeline.
//!
//! The seam a model must expose is small ([`Shardable`]); the orchestration,
//! optimiser (fused host AdamW over the stages' disjoint params, plus summed
//! gradients for any replicated/tied weight), and placement are all generic here.

use std::collections::HashMap;

use backend_cpu::par;

use crate::Model;

/// One pipeline stage: a contiguous layer range `[start, end)` plus whether it
/// owns the token embedding (stage 0) and/or the final-norm+lm_head+loss (last
/// stage), and which physical GPU it runs on. [`Shard::whole`] is the entire
/// model on GPU 0 — the single-device path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    pub start: usize,
    pub end: usize,
    pub embed: bool,
    pub head: bool,
    pub gpu_index: usize,
}

impl Shard {
    /// Placement wildcard: "wherever the ambient device selection lands".
    /// [`Shard::whole`] uses it so single-device construction keeps following
    /// `--device` / scoped placement; an explicit index pins that canonical
    /// physical card (see `gpu_core::devices`).
    pub const ANY_GPU: usize = usize::MAX;

    pub fn whole(n_layers: usize) -> Shard {
        Shard { start: 0, end: n_layers, embed: true, head: true, gpu_index: Shard::ANY_GPU }
    }
    pub fn owns(&self, l: usize) -> bool {
        l >= self.start && l < self.end
    }
    pub fn is_whole(&self, n_layers: usize) -> bool {
        self.start == 0 && self.end == n_layers && self.embed && self.head
    }
}

/// A model's per-stage cost model, in a single arbitrary unit (parameter count
/// works well — sharding balances *memory* to maximise the model size that fits).
pub struct ShardCost {
    pub n_layers: usize,
    /// Cost of one decoder layer.
    pub per_layer: f64,
    /// Extra cost carried by the embed stage (token/positional embedding weights).
    pub embed: f64,
    /// Extra cost carried by the head stage (final norm + lm_head).
    pub head: f64,
    /// f32 words transferred across each cut (`b·t·d_model`).
    pub boundary_words: usize,
}

/// The seam a model exposes so the generic [`Pipeline`] can shard it. Everything
/// else (forward/backward orchestration, the optimiser, cut placement) is generic.
pub trait Shardable: Model + Send {
    /// Cost model for placing the cuts (see [`plan_balanced`]).
    fn shard_cost(cfg: &Self::Config, b: u32, t: u32) -> ShardCost;
    /// Build one stage: only `shard`'s layers (and endpoint weights) are allocated.
    fn new_shard(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> Self
    where
        Self: Sized;
    /// Weights replicated across stages (a tied embedding used by both the embed
    /// and head stages); their gradients are summed. Empty when nothing is tied.
    fn replicated_params(&self) -> Vec<String> {
        Vec::new()
    }
    /// Run this stage's forward graph; returns the loss on the head stage, `None`
    /// otherwise. The batch must already be set (via [`Model::set_batch`]).
    fn run_forward_stage(&self) -> Option<f32>;
    /// Run this stage's backward graph. Consumes `dres[end]` (set by the next
    /// stage via [`Shardable::write_out_dres`]) on non-head stages.
    fn run_backward_stage(&self);
    /// Read this stage's OUTPUT residual `res[end]` (input to the next stage).
    fn read_out_res(&self) -> Vec<f32>;
    /// Write this stage's INPUT residual `res[start]` (from the previous stage).
    fn write_in_res(&self, data: &[f32]);
    /// Read this stage's INPUT-side residual grad `dres[start]` (to the previous stage).
    fn read_in_dres(&self) -> Vec<f32>;
    /// Write this stage's OUTPUT-side residual grad `dres[end]` (from the next stage).
    fn write_out_dres(&self, data: &[f32]);
}

/// Partition `cost.n_layers` layers into `gpus.len()` contiguous stages so the
/// **maximum** per-stage cost is minimised (exact DP). The embed cost lands on
/// stage 0 and the head cost on the last stage, so those stages are given fewer
/// layers — balancing memory across cards. Returns one [`Shard`] per GPU.
pub fn plan_balanced(cost: &ShardCost, gpus: &[usize]) -> Vec<Shard> {
    let l = cost.n_layers;
    let k = gpus.len();
    assert!(k >= 1, "pipeline needs >=1 stage");
    let stage_cost = |a: usize, b: usize, s: usize| -> f64 {
        let mut c = cost.per_layer * (b - a) as f64;
        if s == 0 {
            c += cost.embed;
        }
        if s == k - 1 {
            c += cost.head;
        }
        c
    };
    if k == 1 {
        return vec![Shard { start: 0, end: l, embed: true, head: true, gpu_index: gpus[0] }];
    }
    // best[s][i] = min achievable max-stage-cost covering layers [0..i) with s+1
    // stages (stage s ends at i, not yet counting the head unless s==k-1).
    let inf = f64::INFINITY;
    let mut best = vec![vec![inf; l + 1]; k];
    let mut cut = vec![vec![0usize; l + 1]; k];
    for (i, b) in best[0].iter_mut().enumerate() {
        *b = stage_cost(0, i, 0);
    }
    for s in 1..k {
        for i in 0..=l {
            for j in 0..=i {
                let cand = best[s - 1][j].max(stage_cost(j, i, s));
                if cand < best[s][i] {
                    best[s][i] = cand;
                    cut[s][i] = j;
                }
            }
        }
    }
    let mut bounds = vec![0usize; k + 1];
    bounds[k] = l;
    let mut i = l;
    for s in (1..k).rev() {
        bounds[s] = cut[s][i];
        i = bounds[s];
    }
    (0..k)
        .map(|s| Shard { start: bounds[s], end: bounds[s + 1], embed: s == 0, head: s == k - 1, gpu_index: gpus[s] })
        .collect()
}

/// A **byte-exact** per-stage cost model, for placing a model across devices
/// of DIFFERING capacity - the residency-side counterpart to [`ShardCost`].
///
/// [`ShardCost`]/[`plan_balanced`] balance an abstract unit across cards
/// *assumed to be interchangeable*: the DP minimises the maximum stage cost,
/// which is the right objective only when every device has the same capacity.
/// That is fine for the training pipeline it was written for (one box, matched
/// cards), and wrong the moment the cards differ or the question is "does this
/// even fit?" rather than "which split is most even?". A resident model
/// deciding placement from real, queried VRAM needs actual bytes and a
/// per-device ceiling, which is what this type carries.
pub struct LayerBytes {
    /// Device bytes for each layer, in layer order. Per-layer rather than one
    /// uniform figure because real checkpoints are not uniform (a quantized
    /// layer next to an unquantized one, a shared expert on some layers only),
    /// and because a cost derived from a checkpoint's declared shapes gets
    /// this for free.
    pub per_layer: Vec<u64>,
    /// Extra bytes carried by the FIRST stage (token embedding, if it is
    /// device-resident at all - pass 0 when it is held on the host).
    pub embed: u64,
    /// Extra bytes carried by the LAST stage (final norm + lm_head).
    pub head: u64,
}

impl LayerBytes {
    /// Bytes stage `s` of `k` would occupy holding layers `[a, b)`.
    pub fn stage_bytes(&self, a: usize, b: usize, s: usize, k: usize) -> u64 {
        let mut c: u64 = self.per_layer[a..b].iter().sum();
        if s == 0 {
            c += self.embed;
        }
        if s + 1 == k {
            c += self.head;
        }
        c
    }

    /// Total device bytes across every stage - what the model costs whatever
    /// the split (the per-layer sum plus both endpoint weights).
    pub fn total(&self) -> u64 {
        self.per_layer.iter().sum::<u64>() + self.embed + self.head
    }
}

/// One placed stage: which device, and the contiguous layer range plus
/// endpoint weights it holds, with the byte total it will actually occupy
/// there. Returned by [`plan_by_capacity`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub shard: Shard,
    /// Device bytes this stage occupies - exactly what a caller should
    /// reserve against that device's budget.
    pub bytes: u64,
}

/// Partition `cost`'s layers into contiguous stages across `devices`
/// (`(device index, USABLE bytes)`), such that **every stage fits its own
/// device**, minimising the worst per-device *utilisation fraction* (bytes
/// used / that device's capacity).
///
/// Two properties [`plan_balanced`] does not have, both required for placing a
/// model on real hardware rather than a matched training rig:
///
/// * **Capacity is respected per device.** A 24 GB card and an 8 GB card get
///   layer counts in roughly that ratio, instead of an even split that
///   overruns the smaller one. Uneven capacity is the normal case for a box
///   that grew a second GPU, and an even split there does not merely perform
///   badly - it OOMs.
/// * **Infeasibility is reported, not papered over.** `None` means the model
///   genuinely does not fit across these devices at this precision. The
///   caller can then say so, rather than allocating until the driver refuses
///   partway through a multi-minute load.
///
/// Ties break toward earlier cuts, so the result is deterministic. `devices`
/// order is respected (stage `i` goes to `devices[i]`); pass them in the
/// order weights should flow, since the residual stream is handed along that
/// chain. A single device is the ordinary un-sharded case and is handled
/// (it just has to fit).
pub fn plan_by_capacity(cost: &LayerBytes, devices: &[(usize, u64)]) -> Option<Vec<Placement>> {
    let l = cost.per_layer.len();
    let k = devices.len();
    if k == 0 {
        return None;
    }
    // Objective: minimise the maximum utilisation fraction. Compared in
    // rationals-as-f64 - the numbers are byte counts under 2^53, so the
    // division is exact enough that a tie never silently reorders (and ties
    // break deterministically toward the earlier cut below regardless).
    let util = |a: usize, b: usize, s: usize| -> Option<f64> {
        let need = cost.stage_bytes(a, b, s, k);
        let cap = devices[s].1;
        if need > cap {
            return None; // hard infeasible for this device
        }
        Some(if cap == 0 { 0.0 } else { need as f64 / cap as f64 })
    };

    // best[s][i]: min achievable worst-utilisation covering layers [0..i)
    // with stages 0..=s, stage s ending at i.
    let inf = f64::INFINITY;
    let mut best = vec![vec![inf; l + 1]; k];
    let mut cut = vec![vec![0usize; l + 1]; k];
    for (i, b) in best[0].iter_mut().enumerate() {
        // A one-stage model must also carry the head, so stage 0 is the last
        // stage exactly when k == 1 - `util`'s `s`/`k` handles that.
        if let Some(u) = util(0, i, 0) {
            *b = u;
        }
    }
    for s in 1..k {
        for i in 0..=l {
            for j in 0..=i {
                if best[s - 1][j].is_infinite() {
                    continue;
                }
                let Some(u) = util(j, i, s) else { continue };
                let cand = best[s - 1][j].max(u);
                if cand < best[s][i] {
                    best[s][i] = cand;
                    cut[s][i] = j;
                }
            }
        }
    }
    if best[k - 1][l].is_infinite() {
        return None; // does not fit across these devices, at all
    }

    let mut bounds = vec![0usize; k + 1];
    bounds[k] = l;
    let mut i = l;
    for s in (1..k).rev() {
        bounds[s] = cut[s][i];
        i = bounds[s];
    }
    Some(
        (0..k)
            .map(|s| Placement {
                shard: Shard { start: bounds[s], end: bounds[s + 1], embed: s == 0, head: s + 1 == k, gpu_index: devices[s].0 },
                bytes: cost.stage_bytes(bounds[s], bounds[s + 1], s, k),
            })
            .collect(),
    )
}

/// [`plan_by_capacity`] over the FEWEST leading devices that fit: try one
/// device, then two, and so on, returning the first plan that works.
///
/// This is what "shard automatically across however many GPUs are available"
/// should mean - a model that fits one card should not be spread over four
/// (every extra stage adds a cross-device residual hop and strands the other
/// cards' capacity for other models), and a model that fits none of the
/// prefixes genuinely does not fit the box. Returns `None` only when even
/// ALL the devices together cannot hold it.
pub fn plan_fewest_devices(cost: &LayerBytes, devices: &[(usize, u64)]) -> Option<Vec<Placement>> {
    (1..=devices.len()).find_map(|n| plan_by_capacity(cost, &devices[..n]))
}

/// Host-resident fused optimiser state (master weights + AdamW moments in RAM),
/// covering the union of all stages' parameters.
struct FusedAdam {
    state: Vec<AdamSlot>,
}

/// One parameter's host-resident optimiser state: `(name, master, m, v)`.
type AdamSlot = (String, Vec<f32>, Vec<f32>, Vec<f32>);

/// A pipeline of decoder stages across GPUs, for any [`Shardable`] model.
pub struct Pipeline<M: Shardable> {
    stages: Vec<M>,
    shards: Vec<Shard>,
    /// param name -> indices of the stages that hold it (>1 only for a replicated
    /// tied weight).
    holders: Vec<(String, Vec<usize>)>,
    fused: Option<FusedAdam>,
}

impl<M: Shardable> Pipeline<M> {
    /// Build a pipeline over `gpus` (one stage per entry; repeats allowed). The
    /// cuts are placed automatically by [`plan_balanced`]; `init` is the full
    /// model's weights (held once in host RAM), each stage uploads only its slice.
    pub fn new(cfg: M::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, gpus: &[usize]) -> Pipeline<M> {
        let cost = M::shard_cost(&cfg, b, t);
        let shards = plan_balanced(&cost, gpus);
        Pipeline::with_shards(cfg, b, t, init, shards)
    }

    /// Build with explicit shards (bypasses [`plan_balanced`]).
    pub fn with_shards(cfg: M::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shards: Vec<Shard>) -> Pipeline<M> {
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        std::env::set_var("BRAIN_OFFLOAD_ADAM", "1"); // stages keep weight+grad on GPU; moments in RAM
        let mut stages = Vec::with_capacity(shards.len());
        for sh in &shards {
            // Scoped (thread-local, race-free) placement on the shard's card;
            // an ANY_GPU shard keeps the ambient selection.
            let stage = if sh.gpu_index == Shard::ANY_GPU {
                M::new_shard(cfg.clone(), b, t, init, sh.clone())
            } else {
                gpu_core::devices::with_gpu(sh.gpu_index as u32, || {
                    M::new_shard(cfg.clone(), b, t, init, sh.clone())
                })
                .unwrap_or_else(|e| panic!("pipeline stage placement: {e}"))
            };
            stages.push(stage);
        }
        match prev_off {
            Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
            None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
        }
        // param name -> holder stage indices, in a stable (first-seen) order.
        let mut holders: Vec<(String, Vec<usize>)> = Vec::new();
        let mut idx: HashMap<String, usize> = HashMap::new();
        for (si, st) in stages.iter().enumerate() {
            for n in st.param_names() {
                match idx.get(&n) {
                    Some(&h) => holders[h].1.push(si),
                    None => {
                        idx.insert(n.clone(), holders.len());
                        holders.push((n, vec![si]));
                    }
                }
            }
        }
        Pipeline { stages, shards, holders, fused: None }
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }
    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }

    pub fn zero_grads(&self) {
        for st in &self.stages {
            st.zero_grads();
        }
    }

    /// Forward `batch` through every stage, returning the loss. The residual is
    /// carried host-staged from each stage to the next.
    pub fn forward(&self, batch: crate::Batch) -> f32 {
        for st in &self.stages {
            st.set_batch(clone_batch(&batch));
        }
        let last = self.stages.len() - 1;
        let mut carry: Option<Vec<f32>> = None;
        for (i, st) in self.stages.iter().enumerate() {
            if let Some(res) = &carry {
                st.write_in_res(res);
            }
            let loss = st.run_forward_stage();
            if i == last {
                return loss.expect("head stage must return a loss");
            }
            carry = Some(st.read_out_res());
        }
        unreachable!("pipeline has at least one stage")
    }

    /// Backward through every stage (reverse order), carrying the residual grad
    /// back. Requires a preceding [`Self::forward`] on the same batch.
    pub fn backward(&self) {
        let mut carry: Option<Vec<f32>> = None;
        for i in (0..self.stages.len()).rev() {
            let st = &self.stages[i];
            if let Some(d) = &carry {
                st.write_out_dres(d);
            }
            st.run_backward_stage();
            if i > 0 {
                carry = Some(st.read_in_dres());
            }
        }
    }

    /// Fused optimiser step: gather each parameter's gradient from its owning
    /// stage(s) — summed for a replicated tied weight — to the host, one AdamW
    /// update with a global grad-norm clip, write the new weights back to the
    /// owning stage(s). Mirrors the single-device `adamw_step(.., 1/K)`.
    pub fn adamw_step(&mut self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // Gather grads (summed across holders for replicated params).
        let grads: Vec<Vec<f32>> = self
            .holders
            .iter()
            .map(|(name, hs)| {
                let mut g = self.stages[hs[0]].read_grad(name);
                for &si in &hs[1..] {
                    for (a, b) in g.iter_mut().zip(self.stages[si].read_grad(name)) {
                        *a += b;
                    }
                }
                g
            })
            .collect();

        if self.fused.is_none() {
            let state = self
                .holders
                .iter()
                .map(|(name, hs)| {
                    let w = self.stages[hs[0]].read_weight(name);
                    let z = vec![0f32; w.len()];
                    (name.clone(), w, z.clone(), z)
                })
                .collect();
            self.fused = Some(FusedAdam { state });
        }

        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let sq: f64 = par::sum_sq_f64(&grads);
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let fused = self.fused.as_mut().unwrap();
        par::zip_each(&mut fused.state, &grads, |(_, w, m, v), gi| {
            for i in 0..w.len() {
                let gg = gi[i] * scale;
                let mi = b1 * m[i] + (1.0 - b1) * gg;
                let vi = b2 * v[i] + (1.0 - b2) * gg * gg;
                m[i] = mi;
                v[i] = vi;
                let mhat = mi / bc1;
                let vhat = vi / bc2;
                let mut wi = w[i];
                wi -= lr * wd * wi;
                wi -= lr * mhat / (vhat.sqrt() + eps);
                w[i] = wi;
            }
        });

        // Scatter updated weights back to every holder stage.
        let Pipeline { stages, holders, fused, .. } = self;
        let fused = fused.as_ref().unwrap();
        for ((name, w, _, _), (_, hs)) in fused.state.iter().zip(holders.iter()) {
            for &si in hs {
                stages[si].write_weight(name, w);
            }
        }
        for st in stages.iter() {
            st.poll_wait();
        }
    }

    /// Run `microbatches` through the pipeline with a **GPipe** schedule: stages
    /// run **concurrently** (one thread per GPU, connected by channels), so while
    /// stage *i* processes microbatch *k* stage *i+1* processes *k-1* — overlapping
    /// the cards instead of the plain sequential (one-batch) path's 1/p duty
    /// cycle. Activations are **re-materialised** in the backward (each stage
    /// keeps only per-microbatch *input residuals* and recomputes its forward
    /// before its backward), so activation memory is `O(p · b·t·d)` regardless of
    /// the microbatch count. Gradients accumulate across microbatches; returns the
    /// summed loss. Bit-exact to running the microbatches sequentially with
    /// grad-accum (validated in `shard_microbatch.rs`).
    ///
    /// Call [`Self::zero_grads`] first and an optimiser step (e.g.
    /// [`Self::adamw_step`] with `extra_scale = 1/m`) after.
    pub fn pipelined_fwd_bwd(&mut self, microbatches: &[crate::Batch]) -> f32 {
        use std::sync::mpsc::{channel, Receiver, Sender};
        let p = self.stages.len();
        let m = microbatches.len();
        if p == 1 {
            // Degenerate pipeline: just accumulate the microbatches on the one stage.
            let st = &self.stages[0];
            let mut total = 0.0;
            for mb in microbatches {
                st.set_batch(clone_batch(mb));
                total += st.run_forward_stage().unwrap_or(0.0);
                st.run_backward_stage();
            }
            return total;
        }
        // fwd[s]: stage s -> s+1 ; bwd[s]: stage s+1 -> s   (s in 0..p-1)
        let mut fwd_tx: Vec<Option<Sender<Vec<f32>>>> = Vec::new();
        let mut fwd_rx: Vec<Option<Receiver<Vec<f32>>>> = Vec::new();
        let mut bwd_tx: Vec<Option<Sender<Vec<f32>>>> = Vec::new();
        let mut bwd_rx: Vec<Option<Receiver<Vec<f32>>>> = Vec::new();
        for _ in 0..p - 1 {
            let (ft, fr) = channel();
            let (bt, br) = channel();
            fwd_tx.push(Some(ft));
            fwd_rx.push(Some(fr));
            bwd_tx.push(Some(bt));
            bwd_rx.push(Some(br));
        }
        let mbs = microbatches;
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for (s, stage) in self.stages.iter_mut().enumerate() {
                let fin = if s > 0 { fwd_rx[s - 1].take() } else { None };
                let fout = if s < p - 1 { fwd_tx[s].take() } else { None };
                let bin = if s < p - 1 { bwd_rx[s].take() } else { None };
                let bout = if s > 0 { bwd_tx[s - 1].take() } else { None };
                handles.push(sc.spawn(move || {
                    // Per-microbatch input residual (None on the embed stage, which
                    // recomputes straight from the tokens).
                    let mut stash: Vec<Option<Vec<f32>>> = (0..m).map(|_| None).collect();
                    let mut total = 0.0f32;
                    // Forward phase (microbatches in order).
                    for mb in 0..m {
                        stage.set_batch(clone_batch(&mbs[mb]));
                        if let Some(rx) = &fin {
                            // A closed channel here means the NEIGHBOUR stage
                            // died — a cascade, not the root cause. Say so,
                            // instead of a bare RecvError unwrap that buries
                            // the first stage's real panic among p-1 copies.
                            let input = rx.recv().unwrap_or_else(|_| panic!("stage {s}: upstream stage terminated mid-forward (cascade — see the first stage panic for the root cause)"));
                            stage.write_in_res(&input);
                            stash[mb] = Some(input);
                        }
                        let loss = stage.run_forward_stage();
                        match &fout {
                            Some(tx) => tx.send(stage.read_out_res()).unwrap_or_else(|_| panic!("stage {s}: downstream stage terminated mid-forward (cascade — see the first stage panic for the root cause)")),
                            None => total += loss.expect("head stage returns a loss"),
                        }
                    }
                    // Backward phase (reverse order): recompute forward, then backward.
                    for mb in (0..m).rev() {
                        stage.set_batch(clone_batch(&mbs[mb]));
                        if let Some(inp) = &stash[mb] {
                            stage.write_in_res(inp);
                        }
                        stage.run_forward_stage(); // re-materialise activations
                        if let Some(rx) = &bin {
                            let d = rx.recv().unwrap_or_else(|_| panic!("stage {s}: downstream stage terminated mid-backward (cascade — see the first stage panic for the root cause)"));
                            stage.write_out_dres(&d);
                        }
                        stage.run_backward_stage();
                        if let Some(tx) = &bout {
                            tx.send(stage.read_in_dres()).unwrap_or_else(|_| panic!("stage {s}: upstream stage terminated mid-backward (cascade — see the first stage panic for the root cause)"));
                        }
                    }
                    total
                }));
            }
            // resume_unwind the FIRST failed stage's own payload (stage order)
            // rather than unwrap's opaque re-panic; the cascade messages above
            // mark every secondary failure as such.
            let results: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
            let mut total = 0.0f32;
            for r in results {
                match r {
                    Ok(v) => total += v,
                    Err(p) => std::panic::resume_unwind(p),
                }
            }
            total
        })
    }

    /// A full micro-batched training step: [`Self::pipelined_fwd_bwd`] then the
    /// fused optimiser (grads averaged by `1/m`). Returns the mean loss.
    pub fn train_step(&mut self, microbatches: &[crate::Batch], t: u32, lr: f32, wd: f32, clip: Option<f32>) -> f32 {
        self.zero_grads();
        let total = self.pipelined_fwd_bwd(microbatches);
        let m = microbatches.len().max(1);
        self.adamw_step(t, lr, wd, clip, 1.0 / m as f32);
        total / m as f32
    }

    /// The true gradient for `name` (summed across replicas / read from its owner).
    pub fn reduced_grad(&self, name: &str) -> Vec<f32> {
        let (_, hs) = self.holders.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no stage holds {name}"));
        let mut g = self.stages[hs[0]].read_grad(name);
        for &si in &hs[1..] {
            for (a, b) in g.iter_mut().zip(self.stages[si].read_grad(name)) {
                *a += b;
            }
        }
        g
    }
}

fn clone_batch<'a>(b: &crate::Batch<'a>) -> crate::Batch<'a> {
    use crate::Batch::*;
    match *b {
        Lm { tokens, targets } => Lm { tokens, targets },
        Seq2Seq { src, tgt, labels } => Seq2Seq { src, tgt, labels },
        Tensor { tokens, inputs, targets } => Tensor { tokens, inputs, targets },
        Multimodal { tokens, targets, image_embeds, image_rows } => {
            Multimodal { tokens, targets, image_embeds, image_rows }
        }
        LmWeighted { tokens, targets, weights } => LmWeighted { tokens, targets, weights },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_counts(shards: &[Shard]) -> Vec<usize> {
        shards.iter().map(|s| s.end - s.start).collect()
    }

    #[test]
    fn balanced_gives_endpoint_stages_fewer_layers() {
        // Big embedding/head relative to a layer: the embed (stage 0) and head
        // (last) stages should get FEWER layers so no stage is the bottleneck.
        let cost = ShardCost { n_layers: 12, per_layer: 1.0, embed: 4.0, head: 4.0, boundary_words: 1 };
        let shards = plan_balanced(&cost, &[0, 1, 2]);
        let cnts = layer_counts(&shards);
        assert_eq!(cnts.iter().sum::<usize>(), 12, "covers all layers");
        assert!(shards[0].embed && shards[2].head, "endpoints carry embed/head");
        // middle stage (no endpoint weight) takes on the most layers
        assert!(cnts[1] > cnts[0] && cnts[1] > cnts[2], "middle stage balances by taking more: {cnts:?}");
        // the resulting max stage cost is optimal (7): 4+2 | 8? no -> DP finds <=7
        let sc = |a: usize, b: usize, s: usize| {
            cost.per_layer * (b - a) as f64 + if s == 0 { cost.embed } else { 0.0 } + if s == 2 { cost.head } else { 0.0 }
        };
        let worst = shards.iter().enumerate().map(|(s, sh)| sc(sh.start, sh.end, s)).fold(0.0, f64::max);
        assert!(worst <= 7.0 + 1e-9, "bottleneck stage cost not minimal: {worst}");
    }

    #[test]
    fn even_split_when_no_endpoint_weight() {
        let cost = ShardCost { n_layers: 8, per_layer: 1.0, embed: 0.0, head: 0.0, boundary_words: 1 };
        let shards = plan_balanced(&cost, &[0, 1]);
        assert_eq!(layer_counts(&shards), vec![4, 4]);
    }

    #[test]
    fn single_stage_is_whole() {
        let cost = ShardCost { n_layers: 6, per_layer: 1.0, embed: 2.0, head: 2.0, boundary_words: 1 };
        let shards = plan_balanced(&cost, &[0]);
        assert_eq!(shards.len(), 1);
        assert!(shards[0].is_whole(6));
    }

    // ---- capacity-aware placement (plan_by_capacity / plan_fewest_devices) ----

    const GB: u64 = 1 << 30;

    fn uniform(n_layers: usize, per_layer: u64, embed: u64, head: u64) -> LayerBytes {
        LayerBytes { per_layer: vec![per_layer; n_layers], embed, head }
    }

    fn counts(p: &[Placement]) -> Vec<usize> {
        p.iter().map(|x| x.shard.end - x.shard.start).collect()
    }

    /// Every stage must fit its OWN device - never "the total fits the sum".
    fn assert_fits(p: &[Placement], devices: &[(usize, u64)]) {
        for (pl, &(idx, cap)) in p.iter().zip(devices) {
            assert_eq!(pl.shard.gpu_index, idx, "stage placed on the wrong device");
            assert!(pl.bytes <= cap, "stage of {} bytes exceeds device {idx}'s {cap}", pl.bytes);
        }
    }

    /// Contiguous, complete, endpoints marked exactly once.
    fn assert_well_formed(p: &[Placement], n_layers: usize) {
        assert_eq!(p[0].shard.start, 0);
        assert_eq!(p[p.len() - 1].shard.end, n_layers);
        for w in p.windows(2) {
            assert_eq!(w[0].shard.end, w[1].shard.start, "layer ranges must be contiguous with no gap or overlap");
        }
        assert_eq!(p.iter().filter(|x| x.shard.embed).count(), 1);
        assert_eq!(p.iter().filter(|x| x.shard.head).count(), 1);
    }

    #[test]
    fn equal_capacity_two_gpus_splits_evenly() {
        let cost = uniform(48, GB, 0, 0);
        let devices = [(0usize, 32 * GB), (1usize, 32 * GB)];
        let p = plan_by_capacity(&cost, &devices).expect("48 GB across 2x32 GB fits");
        assert_well_formed(&p, 48);
        assert_fits(&p, &devices);
        assert_eq!(counts(&p), vec![24, 24]);
    }

    /// The property `plan_balanced` cannot express: a 3:1 capacity ratio must
    /// produce a ~3:1 layer split, not an even one that overruns the small card.
    #[test]
    fn uneven_capacity_splits_in_proportion_not_evenly() {
        let cost = uniform(40, GB, 0, 0);
        let devices = [(0usize, 30 * GB), (1usize, 10 * GB)];
        let p = plan_by_capacity(&cost, &devices).expect("40 GB across 30+10 GB fits exactly");
        assert_well_formed(&p, 40);
        assert_fits(&p, &devices);
        assert_eq!(counts(&p), vec![30, 10], "layers must follow capacity, not card count");
        // An even split would have put 20 GB on the 10 GB card.
        assert!(counts(&p) != vec![20, 20]);
    }

    /// Genericity over device COUNT: three cards, three different capacities.
    #[test]
    fn three_gpus_with_three_different_capacities() {
        let cost = uniform(36, GB, 0, 0);
        let devices = [(0usize, 8 * GB), (1usize, 16 * GB), (2usize, 24 * GB)];
        let p = plan_by_capacity(&cost, &devices).expect("36 GB across 8+16+24 fits");
        assert_eq!(p.len(), 3);
        assert_well_formed(&p, 36);
        assert_fits(&p, &devices);
        let c = counts(&p);
        assert!(c[0] < c[1] && c[1] < c[2], "layer counts must increase with capacity: {c:?}");
        assert_eq!(c.iter().sum::<usize>(), 36);
    }

    /// Four cards, and endpoint weights that make the two END stages heavier
    /// per layer - the endpoint stages must give up layers for them.
    #[test]
    fn four_gpus_account_for_the_endpoint_weights() {
        let cost = uniform(32, GB, 6 * GB, 6 * GB);
        let devices = [(0usize, 16 * GB), (1usize, 16 * GB), (2usize, 16 * GB), (3usize, 16 * GB)];
        let p = plan_by_capacity(&cost, &devices).expect("44 GB across 4x16 GB fits");
        assert_well_formed(&p, 32);
        assert_fits(&p, &devices);
        let c = counts(&p);
        assert!(c[0] < c[1] && c[3] < c[2], "the embed/head stages must carry fewer layers: {c:?}");
        // The embed stage really is charged the embedding.
        assert_eq!(p[0].bytes, c[0] as u64 * GB + 6 * GB);
        assert_eq!(p[3].bytes, c[3] as u64 * GB + 6 * GB);
    }

    /// Non-uniform layers (a real checkpoint: some layers quantized, some not)
    /// are placed by their ACTUAL bytes, not by layer count.
    #[test]
    fn non_uniform_layers_are_placed_by_real_bytes() {
        // Four fat layers then eight thin ones: an even 6/6 count split would
        // put 4 fat + 2 thin (18 GB) on card 0, which does not fit 12 GB.
        let mut per_layer = vec![4 * GB; 4];
        per_layer.extend(vec![GB; 8]);
        let cost = LayerBytes { per_layer, embed: 0, head: 0 };
        let devices = [(0usize, 12 * GB), (1usize, 12 * GB)];
        let p = plan_by_capacity(&cost, &devices).expect("24 GB across 2x12 GB fits");
        assert_well_formed(&p, 12);
        assert_fits(&p, &devices);
        assert_ne!(counts(&p), vec![6, 6], "an even COUNT split would overrun card 0");
        assert_eq!(p[0].bytes + p[1].bytes, cost.total());
    }

    /// Infeasible is `None`, never a plan that overruns a card. This is the
    /// difference between "we told you it does not fit" and an OOM ten
    /// minutes into a load.
    #[test]
    fn a_model_that_does_not_fit_reports_none() {
        let cost = uniform(10, 4 * GB, 0, 0); // 40 GB
        assert!(plan_by_capacity(&cost, &[(0, 8 * GB), (1, 8 * GB)]).is_none(), "40 GB must not 'fit' 16 GB");
        assert!(plan_fewest_devices(&cost, &[(0, 8 * GB), (1, 8 * GB)]).is_none());
        // One indivisible layer bigger than any single card is also infeasible
        // however many cards there are - layer-range sharding cannot split a layer.
        let big = LayerBytes { per_layer: vec![20 * GB, GB], embed: 0, head: 0 };
        assert!(plan_by_capacity(&big, &[(0, 8 * GB), (1, 8 * GB), (2, 8 * GB)]).is_none());
    }

    /// A stage may legitimately end up EMPTY when capacity is lopsided enough;
    /// the plan must stay well-formed rather than silently dropping a device.
    #[test]
    fn a_tiny_card_may_take_zero_layers_and_the_plan_stays_wellformed() {
        let cost = uniform(8, GB, 0, 0);
        let devices = [(0usize, 64 * GB), (1usize, GB)];
        let p = plan_by_capacity(&cost, &devices).expect("fits on card 0 alone");
        assert_eq!(p.len(), 2, "every named device still gets a stage");
        assert_well_formed(&p, 8);
        assert_fits(&p, &devices);
    }

    #[test]
    fn fewest_devices_prefers_one_card_when_the_model_fits_it() {
        let cost = uniform(24, GB, 0, 0);
        let devices = [(0usize, 32 * GB), (1usize, 32 * GB), (2usize, 32 * GB)];
        let p = plan_fewest_devices(&cost, &devices).expect("24 GB fits one 32 GB card");
        assert_eq!(p.len(), 1, "a model that fits one card must not be spread across three");
        assert!(p[0].shard.is_whole(24));
        assert_eq!(p[0].bytes, 24 * GB);
    }

    #[test]
    fn fewest_devices_grows_only_as_far_as_needed() {
        let cost = uniform(24, GB, 0, 0);
        // 24 GB needs two 16 GB cards, not all four.
        let devices = [(0usize, 16 * GB), (1usize, 16 * GB), (2usize, 16 * GB), (3usize, 16 * GB)];
        let p = plan_fewest_devices(&cost, &devices).expect("fits two cards");
        assert_eq!(p.len(), 2);
        assert_fits(&p, &devices[..2]);
    }

    /// Deterministic: the same inputs must always give the same plan (a
    /// placement that varied run to run would make a resident's `estimate`
    /// and its `activate` disagree about which device holds what).
    #[test]
    fn placement_is_deterministic() {
        let cost = uniform(31, GB, 3 * GB, 5 * GB);
        let devices = [(0usize, 20 * GB), (1usize, 13 * GB), (2usize, 17 * GB)];
        let first = plan_by_capacity(&cost, &devices).expect("fits");
        for _ in 0..8 {
            assert_eq!(plan_by_capacity(&cost, &devices).expect("fits"), first);
        }
    }

    #[test]
    fn zero_devices_is_not_placeable() {
        let cost = uniform(4, GB, 0, 0);
        assert!(plan_by_capacity(&cost, &[]).is_none());
        assert!(plan_fewest_devices(&cost, &[]).is_none());
    }
}
