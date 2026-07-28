// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic data-parallel training across GPUs — works for **any** [`Model`].
//!
//! A full replica of the model on each GPU, each processing a different slice of
//! the step's micro-batches **concurrently** (one thread per card), then a fused
//! gradient all-reduce + host AdamW so every replica applies the identical update
//! and they stay bit-identical.
//!
//! It rides entirely on the [`Model`] trait surface — `set_batch` / `forward` /
//! `backward` / `zero_grads` / `read_grad` / `read_weight` / `write_weight` — so
//! it is architecture-agnostic: gpt, glm, moe, qwen, seq2seq, … all get
//! multi-GPU data-parallel training for free. (Pipeline *sharding*, by contrast,
//! is woven into each model's forward/backward graph and stays per-architecture.)
//!
//! ## Why a fused optimiser rather than an all-reduce + per-replica optimiser
//!
//! On a box without NVLink the 2.4 GB gradient sync and the optimiser dominate a
//! step; done naively, data-parallel is *slower* than one GPU. The fix (measured
//! in `brain-qwen`): the optimiser lives on the **host** — it already has to pull
//! every gradient off the cards, so it **sums** the replicas there, runs **one**
//! AdamW update (shared state — all replicas are identical), and broadcasts the
//! new weights back. Reading grads once and updating once, with both cards'
//! transfers overlapped, is what turns a slowdown into a speedup. Computing the
//! grad-norm on the host (rayon) also sidesteps the single-threaded on-GPU
//! `gradnorm_sq`, which is catastrophic over a large tied embedding.

use std::collections::HashMap;

// Host-parallel reductions go through the CPU scheduler (`backend_cpu::par`);
// rayon lives only there, so `--device cpuN` pool policy governs these loops.
use backend_cpu::par;

use crate::{Batch, Model};

/// Host-resident fused all-reduce + AdamW state (master weights + moments in RAM).
struct FusedAdam {
    state: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)>, // name, master, m, v
}

/// One full model replica per GPU, trained data-parallel. Generic over the model.
pub struct DataParallel<M: Model> {
    replicas: Vec<M>,
    names: Vec<String>,
    fused: Option<FusedAdam>,
}

impl<M: Model + Send> DataParallel<M> {
    /// Build one full replica per entry of `gpus` (the physical GPU index). The
    /// replicas train with host-resident AdamW (this optimiser); `init` is the
    /// full model's weights, uploaded to each card.
    pub fn new(cfg: M::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, gpus: &[usize]) -> DataParallel<M> {
        assert!(!gpus.is_empty(), "data-parallel needs at least one GPU");
        let prev_gpu = std::env::var("BRAIN_GPU_INDEX").ok();
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        // Ask models that support it (qwen) to keep only weight+grad on the GPU —
        // the moments live here in host RAM. Models that ignore it simply keep
        // their (unused) on-GPU moment buffers; correctness is unaffected.
        std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
        let mut replicas = Vec::with_capacity(gpus.len());
        for &g in gpus {
            std::env::set_var("BRAIN_GPU_INDEX", g.to_string());
            replicas.push(M::new(cfg.clone(), b, t, init));
        }
        match prev_gpu {
            Some(v) => std::env::set_var("BRAIN_GPU_INDEX", v),
            None => std::env::remove_var("BRAIN_GPU_INDEX"),
        }
        match prev_off {
            Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
            None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
        }
        let names = replicas[0].param_names();
        DataParallel { replicas, names, fused: None }
    }

    pub fn n_replicas(&self) -> usize {
        self.replicas.len()
    }

    pub fn zero_grads(&self) {
        for r in &self.replicas {
            r.zero_grads();
        }
    }

    /// Run forward+backward over `batches`, split round-robin across the replicas
    /// and executed **concurrently** (one thread per GPU). Gradients accumulate in
    /// each replica (call [`Self::zero_grads`] first). Returns the summed loss.
    pub fn forward_backward(&mut self, batches: &[Batch]) -> f32 {
        let nr = self.replicas.len();
        let assign: Vec<Vec<usize>> = (0..nr).map(|r| (r..batches.len()).step_by(nr).collect()).collect();
        let mut losses = vec![0f32; nr];
        std::thread::scope(|s| {
            for ((r, lo), my) in self.replicas.iter_mut().zip(losses.iter_mut()).zip(&assign) {
                let batches = &batches;
                s.spawn(move || {
                    let mut l = 0f32;
                    for &mi in my {
                        r.set_batch(clone_batch(&batches[mi]));
                        l += r.forward();
                        r.backward();
                    }
                    r.poll_wait();
                    *lo = l;
                });
            }
        });
        losses.iter().sum()
    }

    /// The summed gradient for `name` across all replicas (read-only). After a
    /// [`Self::forward_backward`] this is the true accumulated gradient; used by
    /// the parity test.
    pub fn reduced_grad(&self, name: &str) -> Vec<f32> {
        let mut sum = self.replicas[0].read_grad(name);
        for r in &self.replicas[1..] {
            for (a, b) in sum.iter_mut().zip(r.read_grad(name)) {
                *a += b;
            }
        }
        sum
    }

    /// Fused all-reduce + AdamW step: pull grads off every card concurrently, sum,
    /// one host AdamW update with a global grad-norm clip, broadcast the new
    /// weights back. Mirrors the single-GPU `adamw_step(.., 1/K)`.
    pub fn adamw_step(&mut self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // Phase 1: pull every grad off each card, both cards concurrently.
        let mut grads: Vec<Vec<Vec<f32>>> = Vec::new();
        std::thread::scope(|s| {
            let names = &self.names;
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
            par::zip_each(&mut g, rg, |acc, gi| {
                for (a, b) in acc.iter_mut().zip(gi) {
                    *a += b;
                }
            });
        }

        // Lazily build host optimiser state from replica 0's weights.
        if self.fused.is_none() {
            let state = self
                .names
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
            let sq: f64 = par::sum_sq_f64(&g);
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
        par::zip_each(&mut fused.state, &g, |(_, w, m, v), gi| {
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

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.replicas[0].read_weight(name)
    }

    /// Save replica 0's weights (all replicas are identical after any step).
    pub fn save(&self, path: &str) {
        self.replicas[0].save(path);
    }
}

/// Re-borrow a `Batch` for another `set_batch` call (the enum only holds shared
/// slices, so this is a cheap field copy — not a data clone).
fn clone_batch<'a>(b: &Batch<'a>) -> Batch<'a> {
    match *b {
        Batch::Lm { tokens, targets } => Batch::Lm { tokens, targets },
        Batch::Seq2Seq { src, tgt, labels } => Batch::Seq2Seq { src, tgt, labels },
        Batch::Tensor { tokens, inputs, targets } => Batch::Tensor { tokens, inputs, targets },
    }
}
