// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Data-parallel Qwen training across GPUs: a **full replica** of the model on
//! each card, each replica processing a different slice of the step's
//! micro-batches **concurrently** (one thread per GPU), then a gradient
//! all-reduce so every replica applies the identical update and they stay
//! bit-identical.
//!
//! This is the throughput path (a training **speedup** on the two P40s), for
//! models that fit on one card; it composes with [`crate::shard::Pipeline`] (the
//! capacity path, for models that do not). With the AdamW moments offloaded to
//! RAM, each replica's GPU footprint is only weight+grad, so both fit easily.
//!
//! ## Why the gradient math is exactly single-GPU accumulation
//!
//! A single-GPU step with grad-accum `K` does: `zero_grads`, then `K` times
//! `{forward; backward}` (backward **accumulates** `grad_mb / count_mb` into the
//! grad buffer), then one `adamw_step` scaled by `1/K`. Data-parallel splits the
//! `K` micro-batches across replicas; each accumulates its subset; the all-reduce
//! **sums** the replicas' grad buffers, giving exactly the same total as the
//! single-GPU run. Because after the all-reduce **every replica holds the full
//! gradient**, each independently computes the correct global grad-norm — so clip
//! needs no cross-replica reduction. Validated bit-exact in `tests/dp_parity.rs`.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::config::QwenConfig;
use crate::model::Qwen;

/// Host-resident fused all-reduce + AdamW for data-parallel training. The
/// optimiser moments (m/v) and a master copy of the weights live in RAM; one step
/// pulls each replica's grads to the host (both cards concurrently), **sums**
/// them, computes the global grad-norm (rayon — not the single-threaded on-GPU
/// `gradnorm_sq`, which is catastrophically slow over a 155M-row tied embedding),
/// runs one AdamW update, and broadcasts the new weights back to every replica.
/// Reading grads once and updating once — instead of a separate all-reduce plus a
/// per-replica optimiser — is what makes data-parallel actually faster here.
struct FusedAdam {
    state: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)>, // name, master, m, v
}

/// One full model replica per GPU, trained data-parallel.
pub struct DataParallel {
    replicas: Vec<Qwen>,
    cfg: QwenConfig,
    fused: Option<FusedAdam>,
}

impl DataParallel {
    /// Build one full replica per entry of `gpus` (the physical GPU index).
    /// Replicas train with the offload optimiser (moments in RAM). `init` is the
    /// full model's weights, uploaded to each GPU.
    pub fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, gpus: &[usize]) -> DataParallel {
        assert!(!gpus.is_empty(), "data-parallel needs at least one GPU");
        // Each replica holds only weight+grad on its GPU (Role::Offload); the
        // AdamW moments live once in host RAM, driven by the fused DP optimiser
        // (see FusedAdam / adamw_step). This halves per-replica VRAM vs on-GPU
        // AdamW AND sidesteps the single-threaded on-GPU `gradnorm_sq`, which is
        // catastrophically slow (~30 s/step) over the 155M-row tied embedding.
        let prev_gpu = std::env::var("BRAIN_GPU_INDEX").ok();
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
        let mut replicas = Vec::with_capacity(gpus.len());
        for &g in gpus {
            std::env::set_var("BRAIN_GPU_INDEX", g.to_string());
            replicas.push(Qwen::new(cfg.clone(), b, t, init));
        }
        match prev_gpu {
            Some(v) => std::env::set_var("BRAIN_GPU_INDEX", v),
            None => std::env::remove_var("BRAIN_GPU_INDEX"),
        }
        match prev_off {
            Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
            None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
        }
        DataParallel { replicas, cfg, fused: None }
    }

    /// The optimised (trainable) parameter names, in a stable order.
    fn opt_names(&self) -> Vec<String> {
        self.cfg
            .param_list()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| self.replicas[0].has_param(n))
            .collect()
    }

    pub fn n_replicas(&self) -> usize {
        self.replicas.len()
    }

    pub fn zero_grads(&self) {
        for r in &self.replicas {
            r.zero_grads();
        }
    }

    /// Run forward+backward over `mbs` micro-batches, split round-robin across the
    /// replicas and executed **concurrently** (one thread per GPU). Gradients
    /// accumulate in each replica's buffers (call [`Self::zero_grads`] first).
    /// Returns the summed loss over all micro-batches.
    pub fn forward_backward(&mut self, mbs: &[(Vec<u32>, Vec<u32>)]) -> f32 {
        let nr = self.replicas.len();
        // Round-robin assignment: replica r runs micro-batches r, r+nr, r+2nr, …
        let assign: Vec<Vec<usize>> =
            (0..nr).map(|r| (r..mbs.len()).step_by(nr).collect()).collect();
        let mut losses = vec![0f32; nr];
        std::thread::scope(|s| {
            for ((r, lo), my) in self.replicas.iter_mut().zip(losses.iter_mut()).zip(&assign) {
                let mbs = &mbs; // &[..] of Send data is Sync — shareable across threads
                s.spawn(move || {
                    let mut l = 0f32;
                    for &mi in my {
                        let (x, y) = &mbs[mi];
                        r.set_batch(x, y);
                        l += r.forward();
                        r.backward();
                    }
                    *lo = l;
                });
            }
        });
        losses.iter().sum()
    }

    /// Sum each parameter's gradient across all replicas and write the total back
    /// to every replica (host-staged all-reduce, the only cross-GPU traffic). No
    /// NVLink here, so this goes over PCIe via host RAM; the two cards' transfers
    /// are **overlapped** (one thread per replica) so the reads/writes run in
    /// parallel rather than serially. After this every replica holds the identical
    /// full gradient.
    pub fn all_reduce(&mut self) {
        let nr = self.replicas.len();
        if nr < 2 {
            return;
        }
        let names: Vec<String> = self
            .cfg
            .param_list()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| self.replicas[0].has_param(n))
            .collect();

        // Phase 1: pull every grad off each card, both cards concurrently.
        let mut grads: Vec<Vec<Vec<f32>>> = Vec::new();
        std::thread::scope(|s| {
            let names = &names;
            let handles: Vec<_> = self
                .replicas
                .iter_mut()
                .map(|r| s.spawn(move || names.iter().map(|n| r.read_grad(n)).collect::<Vec<_>>()))
                .collect();
            grads = handles.into_iter().map(|h| h.join().unwrap()).collect();
        });

        // Phase 2: sum on the host (parallel across params).
        let mut sums = std::mem::take(&mut grads[0]);
        for rg in &grads[1..] {
            sums.par_iter_mut().zip(rg.par_iter()).for_each(|(acc, g)| {
                for (a, b) in acc.iter_mut().zip(g) {
                    *a += b;
                }
            });
        }

        // Phase 3: write the summed grad back to every card, concurrently.
        std::thread::scope(|s| {
            let (names, sums) = (&names, &sums);
            for r in self.replicas.iter_mut() {
                s.spawn(move || {
                    for (n, g) in names.iter().zip(sums.iter()) {
                        r.write_grad(n, g);
                    }
                });
            }
        });
    }

    /// All-reduce the gradients, then AdamW-step every replica identically. Each
    /// replica computes the global grad-norm over its (now full) gradient, so the
    /// clip coefficient is correct and identical everywhere — the replicas stay
    /// bit-identical. Mirrors the single-GPU `adamw_step(.., 1/K)`.
    /// Fused all-reduce + AdamW step (see [`FusedAdam`]): pull grads off both
    /// cards concurrently, sum, one host AdamW update with a global clip, broadcast
    /// the new weights back. Mirrors the single-GPU `adamw_step(.., 1/K)`.
    pub fn adamw_step(&mut self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        let names = self.opt_names();

        // Phase 1: pull every grad off each card, both cards concurrently.
        let mut grads: Vec<Vec<Vec<f32>>> = Vec::new();
        std::thread::scope(|s| {
            let names = &names;
            let handles: Vec<_> = self
                .replicas
                .iter_mut()
                .map(|r| s.spawn(move || names.iter().map(|n| r.read_grad(n)).collect::<Vec<_>>()))
                .collect();
            grads = handles.into_iter().map(|h| h.join().unwrap()).collect();
        });

        // Phase 2: sum grads across replicas (host, parallel across params).
        let mut g = std::mem::take(&mut grads[0]);
        for rg in &grads[1..] {
            g.par_iter_mut().zip(rg.par_iter()).for_each(|(acc, gi)| {
                for (a, b) in acc.iter_mut().zip(gi) {
                    *a += b;
                }
            });
        }

        // Lazily build host optimiser state from replica 0's weights.
        if self.fused.is_none() {
            let state = names
                .iter()
                .map(|n| {
                    let w = self.replicas[0].read_weight(n);
                    let z = vec![0f32; w.len()];
                    (n.clone(), w, z.clone(), z)
                })
                .collect();
            self.fused = Some(FusedAdam { state });
        }

        // Phase 3: global grad-norm -> clip coefficient (rayon, not on-GPU).
        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let sq: f64 = g.par_iter().map(|gi| gi.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sum();
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        // Phase 4: one AdamW update on the host (parallel across params).
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let fused = self.fused.as_mut().unwrap();
        fused.state.par_iter_mut().zip(g.par_iter()).for_each(|((_, w, m, v), gi)| {
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

        // Phase 5: broadcast updated weights to every card, concurrently.
        let DataParallel { replicas, fused, .. } = self;
        let fused = fused.as_ref().unwrap();
        std::thread::scope(|s| {
            let fused = &*fused;
            for r in replicas.iter_mut() {
                s.spawn(move || {
                    for (n, w, _, _) in &fused.state {
                        r.write_weight(n, w);
                    }
                    r.poll_wait();
                });
            }
        });
    }

    /// Gradient for `name` (identical on every replica once [`Self::all_reduce`]
    /// has run) — read from replica 0.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.replicas[0].read_grad(name)
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.replicas[0].read_weight(name)
    }

    /// Save replica 0's weights (all replicas are identical after any step).
    pub fn save(&self, path: &str) {
        self.replicas[0].save(path);
    }
}
