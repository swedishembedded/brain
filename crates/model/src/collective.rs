// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transport-agnostic collective communication - the layer every parallelism
//! dimension (tensor / pipeline / data) moves data through.
//!
//! A [`Collective`] is a group of `world_size` **ranks** that exchange tensors.
//! The interface is **per-rank**: every rank calls the same op with *its own*
//! local data and gets its result back, the collective handling the exchange.
//! This is deliberately the NCCL / MPI shape, so the same call sites work whether
//! the ranks are:
//!
//! * threads in one process over several GPUs - [`HostCollective`], which stages
//!   through host RAM (today's 2×P40 box, no NVLink); or
//! * processes on different machines over a network -
//!   [`crate::netcollective::NetworkCollective`], the coordinator-star TCP
//!   transport behind this same trait. brain's sharding code never names the
//!   transport, so a future ring/tree all-reduce is a new `Collective` impl,
//!   not a rewrite.
//!
//! Ops are the standard set: `all_reduce` (sum, everyone gets the total),
//! `all_gather` (concat in rank order), `reduce_scatter` (sum then each rank
//! keeps its slice), `broadcast` (root's data to all). Reductions are **sum** in
//! a fixed rank order so the result is deterministic / bit-reproducible.
//!
//! ## Payload, error channel, async (M7.2 - closes audit F34's "no error channel"
//! gap the doc here used to carry)
//!
//! Every op moves a [`Payload`], data plus the [`Dtype`] tag it was staged at
//! (the SAME device-dtype enum `model::{fp8,int8,lut4}` already tag their
//! bf16/int8/fp8 storage tiers with, not a parallel one invented for this
//! trait), instead of a bare `Vec<f32>`, and every op returns
//! [`CollectiveResult`], not the value directly: a world-size mismatch (a
//! shape that doesn't decompose evenly across ranks), a dtype disagreement
//! between ranks, or an out-of-range rank/root is now a reportable
//! [`CollectiveError`], not a panic or a silently wrong answer.
//!
//! Every op is also `async` - returning a boxed [`CollectiveFuture`] rather than
//! blocking the caller directly - so a `Collective` composes into async call
//! sites instead of forcing one. `dyn Collective` is used pervasively
//! (`Arc<dyn Collective>` in `grid`/`distributed`/`tests`), and native `async fn`
//! in a trait is not dyn-compatible, so the future is boxed by hand (the
//! standard pre-RPITIT shape for an object-safe async trait) rather than adding
//! a proc-macro dependency for it. Plain-thread call sites (every rank here is
//! literally an OS thread, not an async task) drive the future the same way this
//! repo's own native async boundary already does -
//! `pollster::block_on` (see `backend_wgpu::WgpuBackend::new` wrapping
//! `new_async`) - rather than pulling in a second async runtime.
//!
//! `HostCollective`'s exchange itself is still `Barrier`-blocking internally
//! (each rank is a real OS thread; there is nothing to yield to), and
//! `NetworkCollective`'s I/O is still blocking `std::net::TcpStream` reads. The
//! `async` signature makes `Collective` *composable* with other async work and
//! *cancellable/awaitable* at the type level; it does not, on its own, make
//! either transport's internal wait non-blocking - that would need a rewrite of
//! the exchange primitive itself (an async-aware barrier / non-blocking socket
//! read), which is real follow-up work, not a trait-shape change.
//!
//! Widening `distributed`/`parallel`/`grid`'s own Model-facing APIs
//! (`DdpOptimizer::step`, `federated_average`, …) to `Result` is *also*
//! follow-up work this milestone does not take on - they stay synchronous and
//! infallible from a `Model`'s point of view, bridging to the new trait via
//! `pollster::block_on(..).expect(..)` internally (a collective failure mid
//! training step is exactly as fatal as the old panic was).

use gpu_core::select::Dtype;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

/// One rank's payload for a collective op: raw data plus the [`Dtype`] tag it
/// was staged at, so an op can validate that every participating rank agrees on
/// the data's type instead of silently mixing types through a bare `Vec<f32>`.
///
/// The exchange itself still moves plain `f32` host memory today (every
/// implementation here - same as before this redesign); `dtype` is metadata
/// carried through and validated, not yet a second on-the-wire encoding. A
/// caller staging bf16/int8/fp8 data already decodes to fp32 host-side before
/// it reaches this layer (`model::{fp8,int8,lut4}`'s existing host decoders) -
/// this just stops that fact from being invisible to the collective.
#[derive(Clone, Debug, PartialEq)]
pub struct Payload {
    pub dtype: Dtype,
    pub data: Vec<f32>,
}

impl Payload {
    pub fn new(dtype: Dtype, data: Vec<f32>) -> Payload {
        Payload { dtype, data }
    }

    /// The common case: plain fp32 data (also [`Payload`]'s `From<Vec<f32>>`).
    pub fn f32(data: Vec<f32>) -> Payload {
        Payload { dtype: Dtype::F32, data }
    }
}

impl From<Vec<f32>> for Payload {
    fn from(data: Vec<f32>) -> Payload {
        Payload::f32(data)
    }
}

/// Everything a [`Collective`] op can fail with (audit F34: this trait used to
/// have no error channel at all). Every variant carries enough to log or report
/// without a second round trip.
#[derive(Clone, Debug, PartialEq)]
pub enum CollectiveError {
    /// `rank` (or `broadcast`'s `root`) is not `< world_size`.
    RankOutOfRange { rank: usize, world_size: usize },
    /// A participating rank's [`Payload::dtype`] disagrees with the rest.
    DtypeMismatch { rank: usize, expected: Dtype, got: Dtype },
    /// A per-rank length doesn't decompose evenly across `world_size` (e.g.
    /// `reduce_scatter` requires `local.len() % world_size == 0`).
    WorldSizeMismatch { world_size: usize, len: usize },
    /// Participating ranks contributed different-length payloads where the op
    /// requires equal lengths (`all_reduce`'s elementwise sum,
    /// `reduce_scatter`'s pre-scatter sum - `all_gather` has no such
    /// requirement and never raises this).
    LengthMismatch { rank: usize, expected: usize, got: usize },
    /// The transport itself failed - today only
    /// [`crate::netcollective::NetworkCollective`]'s TCP path, wrapping the I/O
    /// error with which peer/op it happened on.
    Transport(String),
}

impl fmt::Display for CollectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CollectiveError::RankOutOfRange { rank, world_size } => {
                write!(f, "rank {rank} out of range for world_size {world_size}")
            }
            CollectiveError::DtypeMismatch { rank, expected, got } => {
                write!(f, "rank {rank} contributed dtype {got:?}, rest of the group agreed on {expected:?}")
            }
            CollectiveError::WorldSizeMismatch { world_size, len } => {
                write!(f, "length {len} does not divide evenly across world_size {world_size}")
            }
            CollectiveError::LengthMismatch { rank, expected, got } => {
                write!(f, "rank {rank} contributed length {got}, expected {expected} (equal to the other ranks)")
            }
            CollectiveError::Transport(msg) => write!(f, "collective transport error: {msg}"),
        }
    }
}

impl std::error::Error for CollectiveError {}

pub type CollectiveResult<T> = Result<T, CollectiveError>;

/// The object-safe shape of an async [`Collective`] op's return type - see the
/// module doc for why this is hand-boxed rather than a native `async fn`.
pub type CollectiveFuture<'a, T> = Pin<Box<dyn Future<Output = CollectiveResult<T>> + Send + 'a>>;

/// A group of ranks that exchange tensors. `Send + Sync` so one instance is
/// shared across the per-rank threads (or, later, lives per process).
pub trait Collective: Send + Sync {
    fn world_size(&self) -> usize;

    /// Element-wise **sum** of every rank's `local`; all ranks receive the total.
    /// All `local` must have equal length and the same [`Payload::dtype`].
    /// Blocks (its future does not resolve) until every rank has called.
    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload>;

    /// Concatenate every rank's `local` in rank order; all ranks receive it.
    /// Lengths may differ per rank; every rank's [`Payload::dtype`] must match.
    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload>;

    /// Sum every rank's `local` element-wise, then return to `rank` only its
    /// contiguous `1/world_size` slice of the sum. `local.len()` must be equal
    /// across ranks and divisible by `world_size`.
    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload>;

    /// Every rank receives `root`'s `local`. Non-root `local` (data AND dtype)
    /// is ignored (pass an empty vec).
    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload>;
}

pub(crate) fn check_rank(rank: usize, world_size: usize) -> CollectiveResult<()> {
    if rank >= world_size {
        Err(CollectiveError::RankOutOfRange { rank, world_size })
    } else {
        Ok(())
    }
}

/// Every slot must agree on [`Payload::dtype`] (rank 0 is the reference).
pub(crate) fn check_dtypes(slots: &[Payload]) -> CollectiveResult<Dtype> {
    let expected = slots[0].dtype;
    for (rank, s) in slots.iter().enumerate().skip(1) {
        if s.dtype != expected {
            return Err(CollectiveError::DtypeMismatch { rank, expected, got: s.dtype });
        }
    }
    Ok(expected)
}

/// Every slot's `data` must be `expected_len` (rank 0's own length).
pub(crate) fn check_equal_lengths(slots: &[Payload]) -> CollectiveResult<usize> {
    let expected = slots[0].data.len();
    for (rank, s) in slots.iter().enumerate().skip(1) {
        if s.data.len() != expected {
            return Err(CollectiveError::LengthMismatch { rank, expected, got: s.data.len() });
        }
    }
    Ok(expected)
}

// ---- in-process, host-staged implementation ------------------------------------

use std::sync::{Arc, Barrier, Mutex};

/// In-process [`Collective`]: ranks are threads, exchange is via a shared host
/// staging area guarded by a barrier. This is the local (single-box, multi-GPU)
/// transport - each rank reads its shard off its GPU to host, calls the op, and
/// writes the result back to its GPU. A networked transport would implement the
/// same trait by exchanging the staged bytes over the wire instead.
pub struct HostCollective {
    world: usize,
    slots: Mutex<Vec<Payload>>,
    barrier: Barrier,
}

impl HostCollective {
    pub fn new(world_size: usize) -> Arc<HostCollective> {
        assert!(world_size >= 1, "world_size must be >= 1");
        Arc::new(HostCollective {
            world: world_size,
            slots: Mutex::new(vec![Payload::f32(Vec::new()); world_size]),
            barrier: Barrier::new(world_size),
        })
    }

    /// Publish `local` into this rank's slot, barrier, run `combine` over all
    /// slots (read-only, every rank computes the same thing), barrier again so no
    /// rank overwrites its slot before all have read. Returns `combine`'s output
    /// (an `Err` included - every rank computes `combine` identically over the
    /// same published slots, so a validation failure is the same `Err` on every
    /// rank, not a divergence).
    ///
    /// An out-of-range `rank` is rejected up front, before either barrier wait -
    /// same as the old code's out-of-bounds index would have panicked before
    /// reaching the barrier. Neither behavior rescues *other* ranks that were
    /// genuinely waiting on this one to show up; that would need a
    /// poisonable/timeout-aware barrier, real follow-up work, not this
    /// signature's job.
    fn exchange<T>(&self, rank: usize, local: Payload, combine: impl Fn(&[Payload]) -> CollectiveResult<T>) -> CollectiveResult<T> {
        check_rank(rank, self.world)?;
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

    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            self.exchange(rank, local, |slots| {
                let dtype = check_dtypes(slots)?;
                let n = check_equal_lengths(slots)?;
                let mut sum = vec![0f32; n];
                for s in slots {
                    for (a, b) in sum.iter_mut().zip(&s.data) {
                        *a += b;
                    }
                }
                Ok(Payload::new(dtype, sum))
            })
        })
    }

    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            self.exchange(rank, local, |slots| {
                let dtype = check_dtypes(slots)?;
                Ok(Payload::new(dtype, slots.iter().flat_map(|s| s.data.iter().copied()).collect()))
            })
        })
    }

    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        let world = self.world;
        Box::pin(async move {
            self.exchange(rank, local, move |slots| {
                let dtype = check_dtypes(slots)?;
                let n = check_equal_lengths(slots)?;
                if n % world != 0 {
                    return Err(CollectiveError::WorldSizeMismatch { world_size: world, len: n });
                }
                let chunk = n / world;
                let lo = rank * chunk;
                let mut out = vec![0f32; chunk];
                for s in slots {
                    for (o, v) in out.iter_mut().zip(&s.data[lo..lo + chunk]) {
                        *o += v;
                    }
                }
                Ok(Payload::new(dtype, out))
            })
        })
    }

    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        let world = self.world;
        Box::pin(async move {
            check_rank(root, world)?;
            self.exchange(rank, local, move |slots| Ok(slots[root].clone()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Drive a collective op across `world` threads, returning each rank's
    /// `Ok` result. Panics (via `unwrap`) on an `Err` - the ok-path tests below
    /// don't expect one; the error-path tests drive ops directly instead.
    fn run<F>(world: usize, f: F) -> Vec<Payload>
    where
        F: Fn(&HostCollective, usize) -> CollectiveResult<Payload> + Send + Sync,
    {
        let c = HostCollective::new(world);
        let results: Vec<Mutex<Option<CollectiveResult<Payload>>>> = (0..world).map(|_| Mutex::new(None)).collect();
        thread::scope(|s| {
            for r in 0..world {
                let (c, f, results) = (&c, &f, &results);
                s.spawn(move || {
                    let out = f(c, r);
                    *results[r].lock().unwrap() = Some(out);
                });
            }
        });
        results.into_iter().map(|m| m.into_inner().unwrap().unwrap().unwrap()).collect()
    }

    #[test]
    fn all_reduce_sums_and_broadcasts() {
        // rank r contributes [r, 10+r, 20+r]; sum over 4 ranks = [6, 46, 86].
        let out = run(4, |c, r| pollster::block_on(c.all_reduce(r, Payload::f32(vec![r as f32, 10.0 + r as f32, 20.0 + r as f32]))));
        for row in &out {
            assert_eq!(row.dtype, Dtype::F32);
            assert_eq!(row.data, vec![6.0, 46.0, 86.0]);
        }
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        let out = run(3, |c, r| pollster::block_on(c.all_gather(r, Payload::f32(vec![r as f32, r as f32 + 0.5]))));
        for row in &out {
            assert_eq!(row.data, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        }
    }

    #[test]
    fn reduce_scatter_sums_then_slices() {
        // each rank contributes the same [1,2,3,4]; sum over 2 ranks = [2,4,6,8];
        // rank 0 gets [2,4], rank 1 gets [6,8].
        let out = run(2, |c, r| pollster::block_on(c.reduce_scatter(r, Payload::f32(vec![1.0, 2.0, 3.0, 4.0]))));
        assert_eq!(out[0].data, vec![2.0, 4.0]);
        assert_eq!(out[1].data, vec![6.0, 8.0]);
    }

    #[test]
    fn broadcast_delivers_roots_data() {
        let out = run(3, |c, r| {
            let local = if r == 2 { vec![7.0, 8.0, 9.0] } else { Vec::new() };
            pollster::block_on(c.broadcast(r, Payload::f32(local), 2))
        });
        for row in &out {
            assert_eq!(row.data, vec![7.0, 8.0, 9.0]);
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
                    let a = pollster::block_on(c.all_reduce(r, Payload::f32(vec![r as f32 + 1.0]))).unwrap(); // [3]
                    let b = pollster::block_on(c.all_gather(r, a)).unwrap(); // [3,3]
                    let d = pollster::block_on(c.all_reduce(r, b)).unwrap(); // [6,6]
                    *results[r].lock().unwrap() = d.data;
                });
            }
        });
        for m in results {
            assert_eq!(m.into_inner().unwrap(), vec![6.0, 6.0]);
        }
    }

    // ---- the new error channel (M7.2 / audit F34) ----

    #[test]
    fn reduce_scatter_reports_world_size_mismatch_not_panic() {
        // both ranks contribute length 5, which does not divide evenly across
        // world_size 2 - both ranks see the same slots post-barrier, so both
        // get the identical Err.
        let c = HostCollective::new(2);
        let results: Vec<Mutex<Option<CollectiveResult<Payload>>>> = (0..2).map(|_| Mutex::new(None)).collect();
        thread::scope(|s| {
            for r in 0..2usize {
                let (c, results) = (&c, &results);
                s.spawn(move || {
                    *results[r].lock().unwrap() = Some(pollster::block_on(c.reduce_scatter(r, Payload::f32(vec![1.0, 2.0, 3.0, 4.0, 5.0]))));
                });
            }
        });
        for m in results {
            assert_eq!(m.into_inner().unwrap(), Some(Err(CollectiveError::WorldSizeMismatch { world_size: 2, len: 5 })));
        }
    }

    #[test]
    fn all_reduce_reports_dtype_mismatch_not_panic() {
        // rank 0 contributes F32, rank 1 contributes BF16 to the same all_reduce
        // - both ranks see the same published slots, so both get the identical
        // Err naming rank 1 as the one that disagreed with rank 0's F32.
        let c = HostCollective::new(2);
        let results: Vec<Mutex<Option<CollectiveResult<Payload>>>> = (0..2).map(|_| Mutex::new(None)).collect();
        thread::scope(|s| {
            for r in 0..2usize {
                let (c, results) = (&c, &results);
                s.spawn(move || {
                    let p = if r == 0 { Payload::new(Dtype::F32, vec![1.0]) } else { Payload::new(Dtype::BF16, vec![2.0]) };
                    *results[r].lock().unwrap() = Some(pollster::block_on(c.all_reduce(r, p)));
                });
            }
        });
        for m in results {
            let got = m.into_inner().unwrap();
            assert_eq!(got, Some(Err(CollectiveError::DtypeMismatch { rank: 1, expected: Dtype::F32, got: Dtype::BF16 })));
        }
    }

    #[test]
    fn rank_out_of_range_is_reported_not_a_panic() {
        let c = HostCollective::new(1);
        let err = pollster::block_on(c.all_reduce(5, Payload::f32(vec![1.0])));
        assert_eq!(err, Err(CollectiveError::RankOutOfRange { rank: 5, world_size: 1 }));

        let err = pollster::block_on(c.broadcast(0, Payload::f32(vec![1.0]), 9));
        assert_eq!(err, Err(CollectiveError::RankOutOfRange { rank: 9, world_size: 1 }));
    }
}
