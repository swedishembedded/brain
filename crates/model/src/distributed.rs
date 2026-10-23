// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SPMD distributed training on the [`Collective`] seam — the one path that
//! scales from threads on a box to processes across a cluster to federated nodes,
//! for **any** [`Model`], by swapping only the [`Collective`] impl.
//!
//! The design is canonical data-parallel SPMD: every rank holds a full replica,
//! runs forward/backward on its own shard of the batch, then **all-reduces the
//! gradient** through the collective and applies an **identical** optimiser step
//! to the summed gradient. Because the reduction is deterministic (fixed rank
//! order) and every rank runs the same update, the replicas stay bit-identical
//! with no separate broadcast — the sum every rank already holds is enough.
//!
//! * threads + [`HostCollective`](crate::HostCollective) → single-box multi-GPU;
//! * processes + [`NetworkCollective`](crate::NetworkCollective) → a cluster;
//! * a federated round → [`federated_average`], the same collective averaging
//!   *weights* (sample-count weighted, FedAvg) every K local steps instead of
//!   *gradients* every step.
//!
//! It rides the [`Model`] trait (`param_names` / `read_grad` / `read_weight` /
//! `write_weight`), so it is architecture-agnostic. The numeric core
//! ([`FlatAdam`], [`fed_avg_flat`]) is split out and unit-tested against a
//! collective directly; the `Model`-generic wrappers are thin flatten/scatter glue.

use crate::collective::{CollectiveError, CollectiveResult, Payload};
use crate::{Collective, Model};

// ---- testable numeric core (operates on flat parameter vectors) ----------------

/// Host-resident AdamW over one flattened parameter vector, whose step reduces the
/// gradient across ranks through a [`Collective`] first. Every rank keeps an
/// identical copy and applies the identical update, so they never diverge.
pub struct FlatAdam {
    master: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl FlatAdam {
    pub fn new(w0: Vec<f32>) -> FlatAdam {
        let n = w0.len();
        FlatAdam { master: w0, m: vec![0.0; n], v: vec![0.0; n] }
    }

    pub fn weights(&self) -> &[f32] {
        &self.master
    }

    /// All-reduce `local_grad` across the collective (sum), then one AdamW step on
    /// the **mean** gradient (`sum / world`), with an optional global-norm clip.
    /// Returns the updated weights (identical on every rank). `t` is the 1-based
    /// step (Adam bias correction).
    // (coll, rank) address the collective and (t, lr, wd, adam, clip) are AdamW's
    // hyperparameters. Boxing the latter into a struct would need the same
    // struct threaded through `DataParallel::step` and every caller in
    // `crates/{qwen,gpt,moe}`; the flat list matches `optim`'s existing API.
    ///
    /// A failed collective returns its error and leaves the weights untouched:
    /// the driver decides whether to checkpoint the last committed step and
    /// restart the job, and a half-updated replica would make that decision
    /// unsound.
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, coll: &dyn Collective, rank: usize, local_grad: Vec<f32>, t: u32, lr: f32, wd: f32, adam: crate::Adam, clip: Option<f32>) -> CollectiveResult<&[f32]> {
        assert_eq!(local_grad.len(), self.master.len(), "grad/param length mismatch");
        // `Collective::all_reduce` is async and `Result`-based; every caller
        // here is a plain OS thread, so this bridges with `pollster::block_on`
        // the same way this crate's other native async boundaries do.
        let summed = pollster::block_on(coll.all_reduce(rank, Payload::f32(local_grad)))?.data;
        let world = coll.world_size() as f32;
        let mean_scale = 1.0 / world;

        // global grad-norm clip on the mean gradient (deterministic on every rank).
        let raw_sq: f64 = if clip.is_some() { summed.iter().map(|&x| (x as f64).powi(2)).sum() } else { 0.0 };
        let eff = optim::grad_multiplier(raw_sq, clip, mean_scale);

        let n = self.master.len();
        adam.update_slice(t, lr, wd, eff, &mut self.master, &mut self.m, &mut self.v, &summed[..n]);
        Ok(&self.master)
    }
}

/// FedAvg over a flat weight vector: the sample-count-weighted average across the
/// collective, `Σ(wᵢ·nᵢ) / Σ nᵢ`. One numerator all-reduce + one scalar denominator
/// all-reduce. Every rank receives the same averaged weights.
pub fn fed_avg_flat(coll: &dyn Collective, rank: usize, weights: &[f32], local_samples: f32) -> CollectiveResult<Vec<f32>> {
    let scaled: Vec<f32> = weights.iter().map(|&w| w * local_samples).collect();
    let num = pollster::block_on(coll.all_reduce(rank, Payload::f32(scaled)))?.data;
    let den = pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![local_samples])))?.data[0];
    let inv = 1.0 / den.max(1e-12);
    Ok(num.iter().map(|&x| x * inv).collect())
}

/// Replace every parameter's gradient with the mean of the replicas' gradients,
/// which is the update a data-parallel step applies. Used before the model's own
/// optimiser step, so its schedule, clipping and resume state stay the model's.
///
/// Every rank must call this after its backward pass. The averaging is what
/// makes the replicas bit-identical: they start equal, reduce to the same
/// vector, and run the same optimiser.
pub fn reduce_grads<M: Model>(model: &M, coll: &dyn Collective, rank: usize) -> CollectiveResult<()> {
    let world = coll.world_size() as f32;
    for name in model.param_names() {
        let local = model.read_grad(&name);
        let summed = pollster::block_on(coll.all_reduce(rank, Payload::f32(local)))?.data;
        let mean: Vec<f32> = summed.iter().map(|value| value / world).collect();
        model.write_grad(&name, &mean).map_err(CollectiveError::Transport)?;
    }
    Ok(())
}

// ---- Model-generic glue (flatten from / scatter to the model) ------------------

/// Byte offsets of each named parameter within the flattened vector.
fn layout<M: Model>(model: &M, names: &[String]) -> Vec<(usize, usize)> {
    let mut offs = Vec::with_capacity(names.len());
    let mut o = 0;
    for n in names {
        let len = model.read_weight(n).len();
        offs.push((o, len));
        o += len;
    }
    offs
}

/// Data-parallel AdamW for any [`Model`], reducing gradients through a
/// [`Collective`]. Construct once per replica; call [`Self::step`] after each
/// forward/backward. Bit-identical across replicas.
pub struct DdpOptimizer {
    names: Vec<String>,
    offs: Vec<(usize, usize)>,
    adam: FlatAdam,
}

impl DdpOptimizer {
    pub fn new<M: Model>(model: &M) -> DdpOptimizer {
        let names = model.param_names();
        let offs = layout(model, &names);
        let mut w0 = Vec::with_capacity(offs.last().map(|(o, l)| o + l).unwrap_or(0));
        for n in &names {
            w0.extend(model.read_weight(n));
        }
        DdpOptimizer { names, offs, adam: FlatAdam::new(w0) }
    }

    /// One optimiser step: flatten this replica's grads, all-reduce + AdamW via
    /// [`FlatAdam::step`], scatter the new weights back to the model.
    // Same AdamW hyperparameter list as [`FlatAdam::step`], which it forwards to.
    /// A failed collective returns its error and leaves the model's weights
    /// untouched (neither the replica nor the optimiser advances).
    #[allow(clippy::too_many_arguments)]
    pub fn step<M: Model>(&mut self, model: &M, coll: &dyn Collective, rank: usize, t: u32, lr: f32, wd: f32, adam: crate::Adam, clip: Option<f32>) -> CollectiveResult<()> {
        let mut flat = Vec::with_capacity(self.adam.master.len());
        for n in &self.names {
            flat.extend(model.read_grad(n));
        }
        self.adam.step(coll, rank, flat, t, lr, wd, adam, clip)?;
        let w = self.adam.weights();
        for (n, &(o, l)) in self.names.iter().zip(&self.offs) {
            model.write_weight(n, &w[o..o + l]);
        }
        model.poll_wait();
        Ok(())
    }

    pub fn weights(&self) -> &[f32] {
        self.adam.weights()
    }
}

/// A federated round for any [`Model`]: replace this replica's weights with the
/// sample-count-weighted average across the collective (FedAvg). Call every K
/// local steps; `local_samples` is how many examples this node trained on in the
/// round. Works identically in-process or across machines — only the collective
/// differs. On a collective failure the model keeps the weights it had.
pub fn federated_average<M: Model>(model: &M, coll: &dyn Collective, rank: usize, local_samples: f32) -> CollectiveResult<()> {
    let names = model.param_names();
    let mut flat = Vec::new();
    let mut offs = Vec::with_capacity(names.len());
    for n in &names {
        let w = model.read_weight(n);
        offs.push((flat.len(), w.len()));
        flat.extend(w);
    }
    let avg = fed_avg_flat(coll, rank, &flat, local_samples)?;
    for (n, &(o, l)) in names.iter().zip(&offs) {
        model.write_weight(n, &avg[o..o + l]);
    }
    model.poll_wait();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HostCollective;
    use std::sync::Mutex;
    use std::thread;

    fn run<F>(world: usize, f: F) -> Vec<Vec<f32>>
    where
        F: Fn(&HostCollective, usize) -> Vec<f32> + Send + Sync,
    {
        let c = HostCollective::new(world);
        let out: Vec<Mutex<Vec<f32>>> = (0..world).map(|_| Mutex::new(Vec::new())).collect();
        thread::scope(|s| {
            for r in 0..world {
                let (c, f, out) = (&c, &f, &out);
                s.spawn(move || {
                    *out[r].lock().unwrap() = f(c, r);
                });
            }
        });
        out.into_iter().map(|m| m.into_inner().unwrap()).collect()
    }

    /// DDP over 2 ranks with different gradients must equal a single-rank optimiser
    /// fed the mean gradient — and both ranks must end bit-identical.
    #[test]
    fn ddp_step_equals_mean_grad_baseline() {
        // rank grads: g0 = [1,-2,3], g1 = [3, 0, 1]; mean = [2,-1,2].
        let g = [vec![1.0f32, -2.0, 3.0], vec![3.0, 0.0, 1.0]];
        let w0 = vec![0.5f32, 0.5, 0.5];
        let (lr, wd) = (0.1f32, 0.0);

        let out = {
            let g = g.clone();
            let w0 = w0.clone();
            run(2, move |c, r| {
                let mut opt = FlatAdam::new(w0.clone());
                for t in 1..=5 {
                    opt.step(c, r, g[r].clone(), t, lr, wd, crate::Adam::default(), None).expect("host collective");
                }
                opt.weights().to_vec()
            })
        };
        // baseline: one rank, gradient = mean, world_size 1 → step uses sum/1 = mean.
        let mut base = FlatAdam::new(w0);
        let solo = HostCollective::new(1);
        let mean: Vec<f32> = (0..3).map(|i| (g[0][i] + g[1][i]) / 2.0).collect();
        for t in 1..=5 {
            base.step(&*solo, 0, mean.clone(), t, lr, wd, crate::Adam::default(), None).expect("host collective");
        }
        assert_eq!(out[0], out[1], "replicas diverged");
        for (a, b) in out[0].iter().zip(base.weights()) {
            assert!((a - b).abs() < 1e-6, "DDP != mean-grad baseline: {a} vs {b}");
        }
    }

    /// FedAvg = sample-weighted mean of the ranks' weights.
    #[test]
    fn fed_avg_is_sample_weighted_mean() {
        // rank 0: w=[10,20], n=1 ; rank 1: w=[0,0], n=3 → (10·1+0·3)/4 = 2.5, 5.0
        let ws = [vec![10.0f32, 20.0], vec![0.0, 0.0]];
        let ns = [1.0f32, 3.0];
        let out = {
            let ws = ws.clone();
            run(2, move |c, r| fed_avg_flat(c, r, &ws[r], ns[r]).expect("host collective"))
        };
        for row in &out {
            assert!((row[0] - 2.5).abs() < 1e-6 && (row[1] - 5.0).abs() < 1e-6, "fedavg wrong: {row:?}");
        }
    }
}
