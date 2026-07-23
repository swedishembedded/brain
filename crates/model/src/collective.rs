// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transport-agnostic collective communication — the layer every parallelism
//! dimension (tensor / pipeline / data) moves data through.
//!
//! A [`Collective`] is a group of `world_size` **ranks** that exchange tensors.
//! The interface is **per-rank**: every rank calls the same op with *its own*
//! local data and gets its result back, the collective handling the exchange.
//! This is deliberately the NCCL / MPI shape, so the same call sites work whether
//! the ranks are:
//!
//! * threads in one process over several GPUs — [`HostCollective`], which stages
//!   through host RAM (today's 2×P40 box, no NVLink); or
//! * processes on different machines over a network — a future `NetworkCollective`
//!   implementing this same trait (ring/tree all-reduce over sockets). brain's
//!   sharding code never names the transport, so adding compute-node/cluster
//!   support is a new `Collective` impl, not a rewrite.
//!
//! Ops are the standard set: `all_reduce` (sum, everyone gets the total),
//! `all_gather` (concat in rank order), `reduce_scatter` (sum then each rank
//! keeps its slice), `broadcast` (root's data to all). Reductions are **sum** in
//! a fixed rank order so the result is deterministic / bit-reproducible.

/// A group of ranks that exchange tensors. `Send + Sync` so one instance is
/// shared across the per-rank threads (or, later, lives per process).
pub trait Collective: Send + Sync {
    fn world_size(&self) -> usize;

    /// Element-wise **sum** of every rank's `local`; all ranks receive the total.
    /// All `local` must have equal length. Blocks until every rank has called.
    fn all_reduce(&self, rank: usize, local: Vec<f32>) -> Vec<f32>;

    /// Concatenate every rank's `local` in rank order; all ranks receive it.
    /// Lengths may differ per rank.
    fn all_gather(&self, rank: usize, local: Vec<f32>) -> Vec<f32>;

    /// Sum every rank's `local` element-wise, then return to `rank` only its
    /// contiguous `1/world_size` slice of the sum. `local.len()` must be divisible
    /// by `world_size` and equal across ranks.
    fn reduce_scatter(&self, rank: usize, local: Vec<f32>) -> Vec<f32>;

    /// Every rank receives `root`'s `local`. Non-root `local` is ignored (pass an
    /// empty vec).
    fn broadcast(&self, rank: usize, local: Vec<f32>, root: usize) -> Vec<f32>;
}

// ---- in-process, host-staged implementation ------------------------------------

use std::sync::{Arc, Barrier, Mutex};

/// In-process [`Collective`]: ranks are threads, exchange is via a shared host
/// staging area guarded by a barrier. This is the local (single-box, multi-GPU)
/// transport — each rank reads its shard off its GPU to host, calls the op, and
/// writes the result back to its GPU. A networked transport would implement the
/// same trait by exchanging the staged bytes over the wire instead.
pub struct HostCollective {
    world: usize,
    slots: Mutex<Vec<Vec<f32>>>,
    barrier: Barrier,
}

impl HostCollective {
    pub fn new(world_size: usize) -> Arc<HostCollective> {
        assert!(world_size >= 1, "world_size must be >= 1");
        Arc::new(HostCollective {
            world: world_size,
            slots: Mutex::new(vec![Vec::new(); world_size]),
            barrier: Barrier::new(world_size),
        })
    }

    /// Publish `local` into this rank's slot, barrier, run `combine` over all
    /// slots (read-only, every rank computes the same thing), barrier again so no
    /// rank overwrites its slot before all have read. Returns `combine`'s output.
    fn exchange<T>(&self, rank: usize, local: Vec<f32>, combine: impl Fn(&[Vec<f32>]) -> T) -> T {
        {
            let mut slots = self.slots.lock().unwrap();
            slots[rank] = local;
        }
        self.barrier.wait();
        let out = {
            let slots = self.slots.lock().unwrap();
            combine(&slots)
        };
        self.barrier.wait();
        out
    }
}

impl Collective for HostCollective {
    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce(&self, rank: usize, local: Vec<f32>) -> Vec<f32> {
        self.exchange(rank, local, |slots| {
            let n = slots[0].len();
            let mut sum = vec![0f32; n];
            for s in slots {
                debug_assert_eq!(s.len(), n, "all_reduce: unequal lengths");
                for (a, b) in sum.iter_mut().zip(s) {
                    *a += b;
                }
            }
            sum
        })
    }

    fn all_gather(&self, rank: usize, local: Vec<f32>) -> Vec<f32> {
        self.exchange(rank, local, |slots| slots.iter().flatten().copied().collect())
    }

    fn reduce_scatter(&self, rank: usize, local: Vec<f32>) -> Vec<f32> {
        let world = self.world;
        self.exchange(rank, local, move |slots| {
            let n = slots[0].len();
            assert_eq!(n % world, 0, "reduce_scatter: length not divisible by world_size");
            let chunk = n / world;
            let lo = rank * chunk;
            let mut out = vec![0f32; chunk];
            for s in slots {
                for j in 0..chunk {
                    out[j] += s[lo + j];
                }
            }
            out
        })
    }

    fn broadcast(&self, rank: usize, local: Vec<f32>, root: usize) -> Vec<f32> {
        let _ = rank;
        self.exchange(rank, local, move |slots| slots[root].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Drive a collective op across `world` threads, returning each rank's result.
    fn run<F>(world: usize, f: F) -> Vec<Vec<f32>>
    where
        F: Fn(&HostCollective, usize) -> Vec<f32> + Send + Sync,
    {
        let c = HostCollective::new(world);
        let results: Vec<Mutex<Vec<f32>>> = (0..world).map(|_| Mutex::new(Vec::new())).collect();
        thread::scope(|s| {
            for r in 0..world {
                let (c, f, results) = (&c, &f, &results);
                s.spawn(move || {
                    let out = f(c, r);
                    *results[r].lock().unwrap() = out;
                });
            }
        });
        results.into_iter().map(|m| m.into_inner().unwrap()).collect()
    }

    #[test]
    fn all_reduce_sums_and_broadcasts() {
        // rank r contributes [r, 10+r, 20+r]; sum over 4 ranks = [6, 46, 86].
        let out = run(4, |c, r| c.all_reduce(r, vec![r as f32, 10.0 + r as f32, 20.0 + r as f32]));
        for row in &out {
            assert_eq!(row, &vec![6.0, 46.0, 86.0]);
        }
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        let out = run(3, |c, r| c.all_gather(r, vec![r as f32, r as f32 + 0.5]));
        for row in &out {
            assert_eq!(row, &vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        }
    }

    #[test]
    fn reduce_scatter_sums_then_slices() {
        // each rank contributes the same [1,2,3,4]; sum over 2 ranks = [2,4,6,8];
        // rank 0 gets [2,4], rank 1 gets [6,8].
        let out = run(2, |c, r| c.reduce_scatter(r, vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(out[0], vec![2.0, 4.0]);
        assert_eq!(out[1], vec![6.0, 8.0]);
    }

    #[test]
    fn broadcast_delivers_roots_data() {
        let out = run(3, |c, r| {
            let local = if r == 2 { vec![7.0, 8.0, 9.0] } else { Vec::new() };
            c.broadcast(r, local, 2)
        });
        for row in &out {
            assert_eq!(row, &vec![7.0, 8.0, 9.0]);
        }
    }

    #[test]
    fn reusable_across_multiple_ops() {
        // The same collective must work for a sequence of ops (barrier reuse).
        let c = HostCollective::new(2);
        let results: Vec<Mutex<Vec<f32>>> = (0..2).map(|_| Mutex::new(Vec::new())).collect();
        thread::scope(|s| {
            for r in 0..2usize {
                let (c, results) = (&c, &results);
                s.spawn(move || {
                    let a = c.all_reduce(r, vec![r as f32 + 1.0]); // [3]
                    let b = c.all_gather(r, a.clone()); // [3,3]
                    let d = c.all_reduce(r, b); // [6,6]
                    *results[r].lock().unwrap() = d;
                });
            }
        });
        for m in results {
            assert_eq!(m.into_inner().unwrap(), vec![6.0, 6.0]);
        }
    }
}
