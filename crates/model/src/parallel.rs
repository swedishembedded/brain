// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic data-parallel training across GPUs - works for **any** [`Model`].
//!
//! A full replica of the model on each GPU, each processing a different slice of
//! the step's micro-batches **concurrently** (one thread per card), then a fused
//! gradient all-reduce + host AdamW so every replica applies the identical update
//! and they stay bit-identical.
//!
//! It rides entirely on the [`Model`] trait surface - `set_batch` / `forward` /
//! `backward` / `zero_grads` / `read_grad` / `read_weight` / `write_weight` - so
//! it is architecture-agnostic: gpt, glm, moe, qwen, seq2seq, … all get
//! multi-GPU data-parallel training for free. (Pipeline *sharding*, by contrast,
//! is woven into each model's forward/backward graph and stays per-architecture.)
//!
//! ## Why a fused optimiser rather than an all-reduce + per-replica optimiser
//!
//! On a box without NVLink the 2.4 GB gradient sync and the optimiser dominate a
//! step; done naively, data-parallel is *slower* than one GPU. The fix (measured
//! in `brain-qwen`): the optimiser lives on the **host** - it already has to pull
//! every gradient off the cards, so it **sums** the replicas there, runs **one**
//! AdamW update (shared state - all replicas are identical), and broadcasts the
//! new weights back. Reading grads once and updating once, with both cards'
//! transfers overlapped, is what turns a slowdown into a speedup at all
//! (`qwen3::tests::integration_qwen3::qwen3_dataparallel_speedup` is the gate,
//! and the place to read this box's own ratio).
//!
//! ## Why the grad-norm is on the host too - and why the on-GPU one cannot replace it
//!
//! It is **not** a workaround for the old serial `gradnorm_sq` (the dominant
//! share of GPT's training GPU time until `gradnorm_part` + `clip_coef_wg`
//! made it orders of magnitude faster). That kernel was one reason
//! *this design was reachable*, but it is not what keeps the norm here:
//!
//! * The clip is over the **summed** gradient, and the sum exists only in host
//!   RAM. Phase 1 pulls each replica's grads off its card because the fused
//!   optimiser needs them here anyway; Phase 2 adds them. No single card holds
//!   `Σ_r g_r` at any point.
//! * The norm does not decompose over replicas - `‖Σ_r g_r‖ ≠ f(‖g_r‖)` - so
//!   "device norm per rank, reduce the scalars" computes a *different* clip
//!   coefficient and silently changes every data-parallel training run. There
//!   is no per-rank local norm here to swap out.
//! * Running the device kernels would therefore mean **uploading the summed
//!   gradient back** (2.4 GB for the 0.6B Qwen) onto the PCIe leg that is
//!   already the whole cost of a step, and a fixed one - to save a host
//!   reduction over buffers that are already in cache from Phase 2.
//!
//! `model::shard`'s fused optimiser and `model::distributed`'s `Adam` clip on a
//! host-resident gradient for the same reason. The on-device pair is what
//! `optim::Optim` uses, where the gradient lives on the card and never leaves it.

use std::collections::HashMap;

// Host-parallel reductions go through the CPU scheduler (`backend_cpu::par`);
// rayon lives only there, so `--device cpuN` pool policy governs these loops.
use backend_cpu::par;

use crate::{Batch, Model};

/// Host-resident fused all-reduce + AdamW state (master weights + moments in
/// RAM), laid out as ONE contiguous slab per category rather than one
/// `Vec<f32>` per tensor - see [`DataParallel::adamw_step`]'s "bucketed
/// transfers" note for why. `offs[i]` is `(offset, len)` of `names[i]`'s
/// region within `master`/`m`/`v` (all three share the same layout).
struct FusedAdam {
    offs: Vec<(usize, usize)>,
    master: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
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
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        // Ask models that support it (qwen) to keep only weight+grad on the GPU —
        // the moments live here in host RAM. Models that ignore it simply keep
        // their (unused) on-GPU moment buffers; correctness is unaffected.
        std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
        let mut replicas = Vec::with_capacity(gpus.len());
        for &g in gpus {
            // Scoped (thread-local, race-free) placement on canonical card `g`.
            let replica = gpu_core::devices::with_gpu(g as u32, || M::new(cfg.clone(), b, t, init))
                .unwrap_or_else(|e| panic!("data-parallel replica placement: {e}"));
            replicas.push(replica);
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
    ///
    /// ## Bucketed transfers, not one-per-tensor
    ///
    /// `Model::read_grad`/`write_weight` are (and stay) per-name - each names one
    /// tensor's own `DeviceBuffer`, so the CALL count here is `P` per replica per
    /// direction (`P` = trainable tensor count) no matter how this method is
    /// written; that floor is `paramstore::ParamStore`'s one-`DeviceBuffer`-per-
    /// tensor layout, not this method's. What WAS this method's own cost, and
    /// what this bucketing fixes: every one of those `P` reads used to land in
    /// its OWN freshly-allocated `Vec<f32>`, nested two deep
    /// (`Vec<replica><tensor><f32>>`), which phases 2-4 then walked tensor by
    /// tensor. `distributed.rs::DdpOptimizer` already established the fix for
    /// its own (SPMD, `Collective`-based) path: `flat.extend(model.read_grad(n))`
    /// into ONE contiguous host buffer per side. Phases 1 and 5 below do the
    /// same - flatten every replica's grads into a single `Vec<f32>` as they
    /// arrive (Phase 1), and scatter ONE flat AdamW result back out by
    /// name-range (Phase 5, mirroring `distributed.rs`'s `layout`/scatter) - so
    /// phases 2-4 run over `P` contiguous slabs' worth of ONE buffer instead of
    /// `P` separate allocations, and `FusedAdam`'s own state is the same flat
    /// shape. Bit-identical: each element's AdamW update depends only on its own
    /// `(g, m, v, w)`, never on a neighbour, so concatenation order does not
    /// change a single computed value.
    pub fn adamw_step(&mut self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // Phase 1: pull every grad off each card, both cards concurrently, each
        // flattened into ONE Vec<f32> as it arrives (not P nested per-tensor Vecs).
        let mut grads: Vec<Vec<f32>> = Vec::new();
        std::thread::scope(|s| {
            let names = &self.names;
            let handles: Vec<_> = self
                .replicas
                .iter_mut()
                .map(|r| {
                    s.spawn(move || {
                        let mut flat = Vec::new();
                        for n in names.iter() {
                            flat.extend(r.read_grad(n));
                        }
                        flat
                    })
                })
                .collect();
            // resume_unwind, not unwrap: unwrap on a JoinError re-panics with
            // an opaque "Any { .. }" that buries the replica thread's real
            // panic message; resuming the original payload preserves it.
            grads = handles.into_iter().map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))).collect();
        });

        // Phase 2: sum grads across replicas (host, parallel over the flat slab).
        let mut g = std::mem::take(&mut grads[0]);
        for rg in &grads[1..] {
            par::zip_each(&mut g, rg, |a, b| *a += b);
        }

        // Lazily build host optimiser state from replica 0's weights, flattened
        // into one slab with a name-range table (mirrors `distributed.rs::layout`).
        if self.fused.is_none() {
            let mut master = Vec::new();
            let mut offs = Vec::with_capacity(self.names.len());
            for n in &self.names {
                let w = self.replicas[0].read_weight(n);
                offs.push((master.len(), w.len()));
                master.extend(w);
            }
            let n = master.len();
            self.fused = Some(FusedAdam { offs, master, m: vec![0f32; n], v: vec![0f32; n] });
        }

        // Phase 3: global grad-norm -> clip coefficient, over the host-resident
        // SUM `g`. Not a `gradnorm_sq` workaround - see the module header: the
        // summed gradient is on no card, and ‖Σ_r g_r‖ does not decompose into
        // per-rank norms, so the device pair (`gradnorm_part` + `clip_coef_wg`)
        // cannot compute this number without a 2.4 GB upload first.
        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let sq: f64 = par::sum_sq_f64(std::slice::from_ref(&g));
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        // Phase 4: one AdamW update on the host, parallel over the flat slab
        // (master/m/v mutated together, driven by the read-only summed grad).
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let fused = self.fused.as_mut().unwrap();
        par::zip3_mut(&mut fused.master, &mut fused.m, &mut fused.v, &g, |wi, mi, vi, &gi| {
            let gg = gi * scale;
            *mi = b1 * *mi + (1.0 - b1) * gg;
            *vi = b2 * *vi + (1.0 - b2) * gg * gg;
            let mhat = *mi / bc1;
            let vhat = *vi / bc2;
            *wi -= lr * wd * *wi;
            *wi -= lr * mhat / (vhat.sqrt() + eps);
        });

        // Phase 5: broadcast updated weights to every card, concurrently - ONE
        // flat buffer read out by name-range, not `state`'s old per-tensor Vecs.
        let DataParallel { replicas, fused, names, .. } = self;
        let fused = fused.as_ref().unwrap();
        let names: &Vec<String> = &*names; // reborrow &mut as shared+Copy, so every spawned closure below can take its own copy
        std::thread::scope(|s| {
            for r in replicas.iter_mut() {
                s.spawn(move || {
                    for (n, &(o, l)) in names.iter().zip(&fused.offs) {
                        r.write_weight(n, &fused.master[o..o + l]);
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
/// slices, so this is a cheap field copy - not a data clone).
fn clone_batch<'a>(b: &Batch<'a>) -> Batch<'a> {
    match *b {
        Batch::Lm { tokens, targets } => Batch::Lm { tokens, targets },
        Batch::Seq2Seq { src, tgt, labels } => Batch::Seq2Seq { src, tgt, labels },
        Batch::Tensor { tokens, inputs, targets } => Batch::Tensor { tokens, inputs, targets },
        Batch::Multimodal { tokens, targets, image_embeds, image_rows } => {
            Batch::Multimodal { tokens, targets, image_embeds, image_rows }
        }
        Batch::LmWeighted { tokens, targets, weights } => Batch::LmWeighted { tokens, targets, weights },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct CountCfg;
    impl ModelConfig for CountCfg {
        fn param_list(&self) -> Vec<(String, usize)> {
            Vec::new()
        }
        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn from_json(_v: &serde_json::Value) -> Self {
            CountCfg
        }
        fn vocab(&self) -> u32 {
            0
        }
        fn block_size(&self) -> u32 {
            0
        }
        fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
            self
        }
    }

    /// A `Model` whose only job is counting device<->host calls, so
    /// `adamw_step`'s transfer shape (not just its numeric result) is directly
    /// observable - see [`DataParallel::adamw_step`]'s "bucketed transfers" doc
    /// for the architectural floor this pins and the buffer collapse it proves.
    struct CountingModel {
        w: RefCell<HashMap<String, Vec<f32>>>,
        grad: RefCell<HashMap<String, Vec<f32>>>,
        grad_reads: AtomicUsize,
        weight_writes: AtomicUsize,
    }
    impl CountingModel {
        fn new(sizes: &[(&str, usize)]) -> CountingModel {
            let w = sizes.iter().map(|&(n, len)| (n.to_string(), vec![0f32; len])).collect();
            let grad = sizes.iter().map(|&(n, len)| (n.to_string(), vec![0f32; len])).collect();
            CountingModel { w: RefCell::new(w), grad: RefCell::new(grad), grad_reads: AtomicUsize::new(0), weight_writes: AtomicUsize::new(0) }
        }
        fn seed_grad(&self, name: &str, g: Vec<f32>) {
            self.grad.borrow_mut().insert(name.to_string(), g);
        }
    }
    impl Model for CountingModel {
        type Config = CountCfg;
        fn new(_cfg: CountCfg, _b: u32, _t: u32, _init: &HashMap<String, Vec<f32>>) -> Self {
            unreachable!("this test builds replicas directly, not via DataParallel::new")
        }
        fn init_weights(_cfg: &CountCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
            HashMap::new()
        }
        fn config(&self) -> &CountCfg {
            unreachable!()
        }
        fn set_batch(&self, _b: Batch) {}
        fn forward(&self) -> f32 {
            0.0
        }
        fn backward(&self) {}
        fn zero_grads(&self) {
            for v in self.grad.borrow_mut().values_mut() {
                v.iter_mut().for_each(|x| *x = 0.0);
            }
        }
        fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {}
        fn poll_wait(&self) {}
        fn param_names(&self) -> Vec<String> {
            self.w.borrow().keys().cloned().collect()
        }
        fn read_weight(&self, name: &str) -> Vec<f32> {
            self.w.borrow().get(name).cloned().unwrap()
        }
        fn write_weight(&self, name: &str, data: &[f32]) {
            self.weight_writes.fetch_add(1, Ordering::SeqCst);
            self.w.borrow_mut().insert(name.to_string(), data.to_vec());
        }
        fn read_grad(&self, name: &str) -> Vec<f32> {
            self.grad_reads.fetch_add(1, Ordering::SeqCst);
            self.grad.borrow().get(name).cloned().unwrap()
        }
        fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
            None
        }
        fn save(&self, _path: &str) {}
        fn config_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /// `Model::read_grad`/`write_weight` are (and stay) one call per named
    /// tensor - `ParamStore` keeps one `DeviceBuffer` per name, so that floor
    /// cannot move without a `Model`/`ParamStore` API change (out of scope
    /// here, see the module doc). What THIS pins: exactly that floor, never
    /// more (a future regression re-reading/re-writing a tensor would trip
    /// it), AND that phases 2-4 now run over ONE flat buffer per replica per
    /// direction rather than `names.len()` separate ones - `FusedAdam`'s own
    /// fields prove the latter directly, since this test lives in the same
    /// module. Fails to COMPILE against the pre-bucketing tree (`FusedAdam`
    /// had no `offs`/`master`/`m`/`v` fields, only a `Vec<AdamSlot>`), which is
    /// this refactor's RED: a structural change with no numeric change has no
    /// meaningful runtime-red state to assert against.
    #[test]
    fn adamw_step_flattens_replica_transfers_into_one_buffer_per_direction() {
        let sizes = [("a", 3usize), ("b", 2), ("c", 4)];
        let names: Vec<String> = sizes.iter().map(|&(n, _)| n.to_string()).collect();
        let r0 = CountingModel::new(&sizes);
        let r1 = CountingModel::new(&sizes);
        r0.seed_grad("a", vec![1.0, 2.0, 3.0]);
        r0.seed_grad("b", vec![0.5, -0.5]);
        r0.seed_grad("c", vec![1.0, 1.0, 1.0, 1.0]);
        r1.seed_grad("a", vec![3.0, 0.0, -1.0]);
        r1.seed_grad("b", vec![1.5, 2.5]);
        r1.seed_grad("c", vec![0.0, 0.0, 0.0, 0.0]);
        let mut dp = DataParallel { replicas: vec![r0, r1], names: names.clone(), fused: None };

        dp.adamw_step(1, 0.1, 0.0, None, 1.0);

        for r in &dp.replicas {
            assert_eq!(r.grad_reads.load(Ordering::SeqCst), names.len(), "must read each tensor's grad exactly once per replica, no more");
            assert_eq!(r.weight_writes.load(Ordering::SeqCst), names.len(), "must write each tensor's weight exactly once per replica, no more");
        }

        let fused = dp.fused.as_ref().unwrap();
        let total: usize = sizes.iter().map(|&(_, n)| n).sum();
        assert_eq!(fused.offs.len(), names.len(), "one name-range per tensor, in ONE offset table");
        assert_eq!(fused.master.len(), total, "ONE flat master buffer spanning every tensor, not `names.len()` separate ones");
        assert_eq!(fused.m.len(), total);
        assert_eq!(fused.v.len(), total);

        // Bit-identical vs a hand-computed reference: sum the two replicas'
        // grads, one AdamW step (all weights start at 0, wd=0 so the decay
        // term drops out).
        let (b1, b2, eps, lr) = (0.9f32, 0.999f32, 1e-8f32, 0.1f32);
        let (bc1, bc2) = (1.0 - b1, 1.0 - b2);
        let expect = |g0: &[f32], g1: &[f32]| -> Vec<f32> {
            g0.iter()
                .zip(g1)
                .map(|(&a, &b)| {
                    let g = a + b;
                    let m = (1.0 - b1) * g;
                    let v = (1.0 - b2) * g * g;
                    let (mhat, vhat) = (m / bc1, v / bc2);
                    -lr * mhat / (vhat.sqrt() + eps)
                })
                .collect()
        };
        let expected = [
            ("a", expect(&[1.0, 2.0, 3.0], &[3.0, 0.0, -1.0])),
            ("b", expect(&[0.5, -0.5], &[1.5, 2.5])),
            ("c", expect(&[1.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 0.0])),
        ];
        for (name, exp) in expected {
            let got = dp.read_weight(name);
            for (g, e) in got.iter().zip(&exp) {
                assert!((g - e).abs() < 1e-6, "{name}: got {g}, expected {e}");
            }
        }
    }
}
