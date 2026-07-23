// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic pipeline-parallel sharding for any [`Shardable`] model.
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

use rayon::prelude::*;

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
    pub fn whole(n_layers: usize) -> Shard {
        Shard { start: 0, end: n_layers, embed: true, head: true, gpu_index: 0 }
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
    for i in 0..=l {
        best[0][i] = stage_cost(0, i, 0);
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

/// Host-resident fused optimiser state (master weights + AdamW moments in RAM),
/// covering the union of all stages' parameters.
struct FusedAdam {
    state: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)>, // name, master, m, v
}

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
        let prev_gpu = std::env::var("BRAIN_GPU_INDEX").ok();
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        std::env::set_var("BRAIN_OFFLOAD_ADAM", "1"); // stages keep weight+grad on GPU; moments in RAM
        let mut stages = Vec::with_capacity(shards.len());
        for sh in &shards {
            std::env::set_var("BRAIN_GPU_INDEX", sh.gpu_index.to_string());
            stages.push(M::new_shard(cfg.clone(), b, t, init, sh.clone()));
        }
        match prev_gpu {
            Some(v) => std::env::set_var("BRAIN_GPU_INDEX", v),
            None => std::env::remove_var("BRAIN_GPU_INDEX"),
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
            let sq: f64 = grads.par_iter().map(|g| g.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sum();
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let fused = self.fused.as_mut().unwrap();
        fused.state.par_iter_mut().zip(grads.par_iter()).for_each(|((_, w, m, v), gi)| {
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
                            let input = rx.recv().unwrap();
                            stage.write_in_res(&input);
                            stash[mb] = Some(input);
                        }
                        let loss = stage.run_forward_stage();
                        match &fout {
                            Some(tx) => tx.send(stage.read_out_res()).unwrap(),
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
                            let d = rx.recv().unwrap();
                            stage.write_out_dres(&d);
                        }
                        stage.run_backward_stage();
                        if let Some(tx) = &bout {
                            tx.send(stage.read_in_dres()).unwrap();
                        }
                    }
                    total
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).sum::<f32>()
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
}
