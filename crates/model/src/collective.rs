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
//! The Model-facing distributed optimiser APIs (`DdpOptimizer::step`,
//! `federated_average`, `FlatAdam::step`) are `Result`-based over the same
//! [`CollectiveError`]: a lost peer or a rejected payload stops the step and
//! leaves the replica's weights untouched, so a driver can checkpoint the last
//! committed step and restart the job instead of taking the process down.

use gpu_core::select::Dtype;
use std::fmt;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

// ---- measurement ------------------------------------------------------------------

/// Bytes one rank's [`Payload`] occupies at its staged dtype: elements times
/// [`Dtype::bits`], rounded up to a whole byte so the sub-byte tiers (`Q4`,
/// `Q4K`, `NF4`, `F4E2M1`) are counted as the bytes they need rather than
/// truncated to zero for a single element.
pub fn payload_bytes(payload: &Payload) -> u64 {
    (payload.data.len() as u64)
        .saturating_mul(payload.dtype.bits() as u64)
        .div_ceil(8)
}

/// The collective ops a [`TimedCollective`] accounts for. The discriminants
/// index [`Snapshot::ops`], so declaration order is part of the API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    AllReduce = 0,
    AllGather = 1,
    ReduceScatter = 2,
    Broadcast = 3,
}

impl OpKind {
    /// Every op, in [`Snapshot::ops`] order.
    pub const ALL: [OpKind; 4] = [OpKind::AllReduce, OpKind::AllGather, OpKind::ReduceScatter, OpKind::Broadcast];

    /// The stable lowercase name used in reports.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::AllReduce => "all_reduce",
            OpKind::AllGather => "all_gather",
            OpKind::ReduceScatter => "reduce_scatter",
            OpKind::Broadcast => "broadcast",
        }
    }
}

/// What one op kind cost on one rank. `total` is the sum of the raw per-op
/// samples the percentiles are drawn from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpSnapshot {
    /// Calls of this op kind.
    pub count: u64,
    /// Payload bytes this rank contributed, summed over those calls.
    pub bytes: u64,
    /// Wall time of those calls, summed.
    pub total: Duration,
    /// Nearest-rank percentiles of the per-call wall time.
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// [`TimedCollective`]'s accounting, as of the moment it was taken.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Per-op accounting, indexed by [`OpKind`]'s discriminant.
    pub ops: [OpSnapshot; 4],
    /// Calls across every op kind.
    pub count: u64,
    /// Payload bytes across every op kind.
    pub bytes: u64,
    /// Wall time across every op kind.
    pub total: Duration,
}

impl Snapshot {
    pub fn op(&self, kind: OpKind) -> &OpSnapshot {
        &self.ops[kind as usize]
    }

    /// This accounting as the JSON object a profile reports: call count,
    /// staged bytes and wall time across every op, plus one entry per op kind
    /// with its count, bytes, total and p50/p95/p99. Durations are
    /// milliseconds.
    ///
    /// The shape lives here rather than in a caller so the single-machine
    /// profile and a distributed run's per-rank document cannot drift into two
    /// spellings of the same measurement.
    pub fn to_json(&self) -> serde_json::Value {
        let ops: serde_json::Map<String, serde_json::Value> = OpKind::ALL
            .into_iter()
            .map(|kind| {
                let op = self.op(kind);
                (
                    kind.name().to_string(),
                    serde_json::json!({
                        "count": op.count,
                        "bytes": op.bytes,
                        "time_ms": millis(op.total),
                        "p50_ms": millis(op.p50),
                        "p95_ms": millis(op.p95),
                        "p99_ms": millis(op.p99),
                    }),
                )
            })
            .collect();
        serde_json::json!({
            "count": self.count,
            "bytes": self.bytes,
            "time_ms": millis(self.total),
            "ops": ops,
        })
    }
}

/// `d` in milliseconds, the unit every profile document reports.
fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// One rank's collective profile: the accounting over its [`TimedCollective`]
/// plus the identity of the run that produced it.
///
/// A distributed run writes one document per rank so the cost can be compared
/// across ranks; the [`plan`](Self::plan) field is the job plan's identity, so
/// documents from different jobs cannot be mistaken for one measurement. The
/// run's own wall time turns the accounting into a share, so a reader does not
/// need a second measurement of the same run.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveProfile {
    /// This document's rank.
    pub rank: usize,
    /// How many ranks the job had.
    pub world: usize,
    /// The identity bytes every rank of the job agreed on.
    pub plan: String,
    /// Wall time of the training call the snapshot covers, as its caller
    /// measured it. It includes whatever the call does besides the steps (a
    /// checkpoint load, a dataset render), so the reported share is of this
    /// call, not of one optimizer step.
    pub run: Duration,
    /// The rank's accounting, as of the end of its run.
    pub snapshot: Snapshot,
}

impl CollectiveProfile {
    pub fn new(rank: usize, world: usize, plan: impl Into<String>, run: Duration, snapshot: Snapshot) -> CollectiveProfile {
        CollectiveProfile { rank, world, plan: plan.into(), run, snapshot }
    }

    /// The file a rank's document goes to: `base` with `.rank<rank>.json`
    /// appended, so every rank of one run writes its own file.
    pub fn path_for(base: &Path, rank: usize) -> PathBuf {
        PathBuf::from(format!("{}.rank{rank}.json", base.display()))
    }

    /// This rank's document: the identity fields and the snapshot's
    /// accounting in one JSON object.
    pub fn to_json(&self) -> serde_json::Value {
        let serde_json::Value::Object(mut document) = self.snapshot.to_json() else {
            unreachable!("a snapshot's JSON is always an object");
        };
        let share = if self.run.is_zero() { 0.0 } else { self.snapshot.total.as_secs_f64() / self.run.as_secs_f64() };
        document.insert("rank".to_string(), serde_json::json!(self.rank));
        document.insert("world".to_string(), serde_json::json!(self.world));
        document.insert("plan".to_string(), serde_json::json!(self.plan));
        document.insert("run_ms".to_string(), serde_json::json!(millis(self.run)));
        document.insert("share_of_run".to_string(), serde_json::json!(share));
        serde_json::Value::Object(document)
    }

    /// Write this rank's document beside `base` (see [`Self::path_for`]),
    /// returning the path written. A temporary file and a rename mean a reader
    /// sees the whole document or none of it.
    pub fn write(&self, base: &Path) -> io::Result<PathBuf> {
        let path = Self::path_for(base, self.rank);
        let mut body = serde_json::to_vec_pretty(&self.to_json()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        body.push(b'\n');
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, &body)?;
        std::fs::rename(&temporary, &path)?;
        Ok(path)
    }
}

/// Nearest-rank percentile of an ascending sample slice: the smallest sample
/// at or above `p` percent of the distribution. An empty slice has no sample,
/// so it reports zero rather than an invented value.
fn percentile(sorted: &[Duration], p: u32) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let n = sorted.len() as u128;
    let rank = ((p as u128) * n).div_ceil(100).max(1).min(n) as usize;
    sorted[rank - 1]
}

#[derive(Clone, Debug, Default)]
struct OpSample {
    count: u64,
    bytes: u64,
    total: Duration,
    /// Raw per-call wall times; the snapshot's percentiles are computed from
    /// these, so nothing is pre-aggregated in a way that loses the tail.
    durations: Vec<Duration>,
}

impl OpSample {
    fn record(&mut self, bytes: u64, elapsed: Duration) {
        self.count += 1;
        self.bytes += bytes;
        self.total += elapsed;
        self.durations.push(elapsed);
    }

    fn snapshot(&self) -> OpSnapshot {
        let mut sorted = self.durations.clone();
        sorted.sort_unstable();
        OpSnapshot {
            count: self.count,
            bytes: self.bytes,
            total: self.total,
            p50: percentile(&sorted, 50),
            p95: percentile(&sorted, 95),
            p99: percentile(&sorted, 99),
        }
    }
}

/// A [`Collective`] that times every op and accounts for what each rank
/// staged. Wrap the transport you want measured; nothing global is touched, so
/// a run that does not construct one pays nothing for its existence.
///
/// The sample is taken *inside* the op's async block - from just before the
/// inner call to just after it resolves - so it is the op's own wall time
/// (including the transport's wait for its peers), not the time the caller
/// spent around it.
pub struct TimedCollective<'a> {
    inner: &'a dyn Collective,
    stats: Mutex<[OpSample; 4]>,
}

impl<'a> TimedCollective<'a> {
    pub fn new(inner: &'a dyn Collective) -> TimedCollective<'a> {
        TimedCollective { inner, stats: Mutex::new(Default::default()) }
    }

    /// The accounting so far. Percentiles are computed here, from the raw
    /// samples, so taking a snapshot never changes the measurement.
    pub fn snapshot(&self) -> Snapshot {
        let stats = self.stats.lock().unwrap();
        let mut out = Snapshot::default();
        for (slot, kind) in out.ops.iter_mut().zip(OpKind::ALL) {
            *slot = stats[kind as usize].snapshot();
            out.count += slot.count;
            out.bytes += slot.bytes;
            out.total += slot.total;
        }
        out
    }

    /// Forget every sample (a run-to-run reset that keeps the same wrapper).
    pub fn reset(&self) {
        *self.stats.lock().unwrap() = Default::default();
    }

    fn record(&self, kind: OpKind, bytes: u64, elapsed: Duration) {
        self.stats.lock().unwrap()[kind as usize].record(bytes, elapsed);
    }
}

impl Collective for TimedCollective<'_> {
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            let bytes = payload_bytes(&local);
            let started = Instant::now();
            let out = self.inner.all_reduce(rank, local).await;
            self.record(OpKind::AllReduce, bytes, started.elapsed());
            out
        })
    }

    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            let bytes = payload_bytes(&local);
            let started = Instant::now();
            let out = self.inner.all_gather(rank, local).await;
            self.record(OpKind::AllGather, bytes, started.elapsed());
            out
        })
    }

    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            let bytes = payload_bytes(&local);
            let started = Instant::now();
            let out = self.inner.reduce_scatter(rank, local).await;
            self.record(OpKind::ReduceScatter, bytes, started.elapsed());
            out
        })
    }

    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            let bytes = payload_bytes(&local);
            let started = Instant::now();
            let out = self.inner.broadcast(rank, local, root).await;
            self.record(OpKind::Broadcast, bytes, started.elapsed());
            out
        })
    }
}

// ---- link shaping -----------------------------------------------------------------

/// SplitMix64's finalizer. Used as a pure hash of `(seed, op index)`, so the
/// same run replays the same jitter without carrying an RNG.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A synthetic link a [`ShapedCollective`] charges each op for. Every field is
/// explicit and deterministic: nothing is read from the environment or from a
/// mutable RNG, so a run is reproducible from this value alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulatedLink {
    /// One-way propagation delay. An op pays it in both directions.
    pub one_way_latency: Duration,
    /// Sustained transfer rate, bytes per second. The transfer term is
    /// `payload / bandwidth`; `f64::INFINITY` models a link with no transfer
    /// cost.
    pub bandwidth_bytes_per_sec: f64,
    /// Upper bound of the deterministic jitter added per op.
    pub jitter: Duration,
    /// Seeds the jitter hash; two runs with the same seed see the same
    /// sequence of jitter values.
    pub seed: u64,
    /// Fail every op at or after this index with
    /// [`CollectiveError::Transport`], without touching the wrapped
    /// collective. `None` never fails.
    pub fail_after_ops: Option<u64>,
}

impl SimulatedLink {
    /// The delay charged for op `op_index` moving `payload_bytes`. Deterministic
    /// in `(seed, op_index)`.
    pub fn delay_for(&self, op_index: u64, payload_bytes: u64) -> Duration {
        let round_trip_ns = 2.0 * self.one_way_latency.as_nanos() as f64;
        let transfer_ns = if self.bandwidth_bytes_per_sec.is_finite() && self.bandwidth_bytes_per_sec > 0.0 {
            (payload_bytes as f64 / self.bandwidth_bytes_per_sec) * 1e9
        } else {
            0.0
        };
        let jitter_ns = self.jitter_ns(op_index);
        let total = (round_trip_ns + transfer_ns + jitter_ns).min(u64::MAX as f64);
        Duration::from_nanos(total.round() as u64)
    }

    /// The jitter draw in `[0, jitter)`, from the hash of `seed ^ op_index`.
    fn jitter_ns(&self, op_index: u64) -> f64 {
        if self.jitter.is_zero() {
            return 0.0;
        }
        let fraction = (splitmix64(self.seed ^ op_index) >> 11) as f64 / (1u64 << 53) as f64;
        fraction * self.jitter.as_nanos() as f64
    }
}

/// A [`Collective`] that charges the wrapped transport a simulated network
/// delay before every op, for measuring how a training step's cost decomposes
/// between computation and communication on a link other than the one under
/// the desk.
///
/// Each op sleeps [`SimulatedLink::delay_for`]'s delay and then runs the inner
/// op unchanged, so the exchange itself is still whatever the wrapped
/// collective does (a host-staged barrier, a TCP round trip). A
/// [`SimulatedLink::fail_after_ops`] returns a transport error at its firing
/// point *before* the inner op is touched, which lets a caller verify a
/// failure path without a real broken peer.
pub struct ShapedCollective<'a> {
    inner: &'a dyn Collective,
    link: SimulatedLink,
    op_index: AtomicU64,
}

impl<'a> ShapedCollective<'a> {
    pub fn new(inner: &'a dyn Collective, link: SimulatedLink) -> ShapedCollective<'a> {
        ShapedCollective { inner, link, op_index: AtomicU64::new(0) }
    }

    pub fn link(&self) -> SimulatedLink {
        self.link
    }

    /// Count this op and return the delay it pays. A configured failure point
    /// is checked here, before the inner collective sees the op.
    fn schedule(&self, bytes: u64) -> CollectiveResult<Duration> {
        // A one-rank group exchanges nothing: there is no link to shape and no
        // peer that can fail. Charging the simulated link here would make the
        // world-1 baseline of a profile slower than the world that actually
        // communicates, and every speedup derived from it too optimistic.
        if self.inner.world_size() <= 1 {
            return Ok(Duration::ZERO);
        }
        let index = self.op_index.fetch_add(1, Ordering::Relaxed);
        if let Some(limit) = self.link.fail_after_ops {
            if index >= limit {
                return Err(CollectiveError::Transport(format!(
                    "simulated link: op {index} is past the configured failure point of {limit} op(s)"
                )));
            }
        }
        Ok(self.link.delay_for(index, bytes))
    }
}

impl Collective for ShapedCollective<'_> {
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            std::thread::sleep(self.schedule(payload_bytes(&local))?);
            self.inner.all_reduce(rank, local).await
        })
    }

    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            std::thread::sleep(self.schedule(payload_bytes(&local))?);
            self.inner.all_gather(rank, local).await
        })
    }

    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            std::thread::sleep(self.schedule(payload_bytes(&local))?);
            self.inner.reduce_scatter(rank, local).await
        })
    }

    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            std::thread::sleep(self.schedule(payload_bytes(&local))?);
            self.inner.broadcast(rank, local, root).await
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

    // ---- measurement and link shaping ----

    /// A one-rank collective that only counts how many ops reached it, so a
    /// shape's failure point can be shown to fire before the inner op runs.
    struct Counting {
        ops: AtomicU64,
    }

    impl Counting {
        fn new() -> Counting {
            Counting { ops: AtomicU64::new(0) }
        }

        fn ops(&self) -> u64 {
            self.ops.load(Ordering::Relaxed)
        }

        fn count(&self) -> u64 {
            self.ops.fetch_add(1, Ordering::Relaxed) + 1
        }
    }

    impl Collective for Counting {
        /// A group with a peer, so the shaped-link schedule applies. The stub
        /// answers locally; nothing here waits on a second thread.
        fn world_size(&self) -> usize {
            2
        }

        fn all_reduce<'a>(&'a self, _rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
            Box::pin(async move {
                self.count();
                Ok(local)
            })
        }

        fn all_gather<'a>(&'a self, _rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
            Box::pin(async move {
                self.count();
                Ok(local)
            })
        }

        fn reduce_scatter<'a>(&'a self, _rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
            Box::pin(async move {
                self.count();
                Ok(local)
            })
        }

        fn broadcast<'a>(&'a self, _rank: usize, local: Payload, _root: usize) -> CollectiveFuture<'a, Payload> {
            Box::pin(async move {
                self.count();
                Ok(local)
            })
        }
    }

    fn link() -> SimulatedLink {
        SimulatedLink { one_way_latency: Duration::ZERO, bandwidth_bytes_per_sec: f64::INFINITY, jitter: Duration::ZERO, seed: data::rng::random_seed(), fail_after_ops: None }
    }

    #[test]
    fn payload_bytes_use_each_tier_s_width_and_round_up() {
        // 32-bit tier: 3 elements are 12 bytes.
        assert_eq!(payload_bytes(&Payload::new(Dtype::F32, vec![0.0; 3])), 12);
        // 16-bit tiers: 3 elements are 6 bytes.
        assert_eq!(payload_bytes(&Payload::new(Dtype::BF16, vec![0.0; 3])), 6);
        assert_eq!(payload_bytes(&Payload::new(Dtype::F16, vec![0.0; 3])), 6);
        // 8-bit tiers: bytes == elements.
        assert_eq!(payload_bytes(&Payload::new(Dtype::I8, vec![0.0; 5])), 5);
        assert_eq!(payload_bytes(&Payload::new(Dtype::F8E4M3, vec![0.0; 5])), 5);
        // 4-bit tiers: half a byte per element, rounded up for an odd count.
        assert_eq!(payload_bytes(&Payload::new(Dtype::Q4, vec![0.0; 1])), 1);
        assert_eq!(payload_bytes(&Payload::new(Dtype::Q4, vec![0.0; 3])), 2);
        assert_eq!(payload_bytes(&Payload::new(Dtype::Q4K, vec![0.0; 8])), 4);
        // An empty payload moves nothing.
        assert_eq!(payload_bytes(&Payload::f32(Vec::new())), 0);
    }

    #[test]
    fn timed_collective_accounts_per_op_count_bytes_and_time() {
        let inner = HostCollective::new(1);
        let timed = TimedCollective::new(&*inner);
        pollster::block_on(timed.all_reduce(0, Payload::f32(vec![1.0; 4]))).unwrap();
        pollster::block_on(timed.all_reduce(0, Payload::new(Dtype::BF16, vec![1.0; 3]))).unwrap();
        pollster::block_on(timed.all_gather(0, Payload::f32(vec![1.0; 2]))).unwrap();

        let snap = timed.snapshot();
        let reduce = snap.op(OpKind::AllReduce);
        assert_eq!(reduce.count, 2);
        assert_eq!(reduce.bytes, 16 + 6);
        assert!(reduce.total > Duration::ZERO);
        assert!(reduce.p50 <= reduce.p99, "the fastest sample is at or below the slowest");
        assert_eq!(snap.op(OpKind::AllGather).count, 1);
        assert_eq!(snap.op(OpKind::AllGather).bytes, 8);
        assert_eq!(snap.op(OpKind::ReduceScatter).count, 0);
        assert_eq!(snap.count, 3);
        assert_eq!(snap.bytes, 16 + 6 + 8);
        assert!(snap.total >= reduce.total);

        timed.reset();
        assert_eq!(timed.snapshot().count, 0);
    }

    #[test]
    fn a_snapshot_reports_count_bytes_and_milliseconds_per_op_and_in_total() {
        let inner = HostCollective::new(1);
        let timed = TimedCollective::new(&*inner);
        pollster::block_on(timed.all_reduce(0, Payload::f32(vec![1.0; 4]))).unwrap();

        let json = timed.snapshot().to_json();
        assert_eq!(json["count"], 1);
        assert_eq!(json["bytes"], 16);
        assert!(json["time_ms"].as_f64().expect("a total") >= 0.0);
        let reduce = &json["ops"]["all_reduce"];
        assert_eq!(reduce["count"], 1);
        assert_eq!(reduce["bytes"], 16);
        assert!(reduce["time_ms"].as_f64().expect("a total") >= 0.0);
        assert!(reduce["p99_ms"].as_f64().expect("a p99") >= reduce["p50_ms"].as_f64().expect("a p50"));
        assert_eq!(json["ops"]["broadcast"]["count"], 0);
    }

    #[test]
    fn a_rank_profile_names_its_own_file_and_carries_the_run_identity() {
        let profile = CollectiveProfile::new(1, 2, "plan-bytes", Duration::from_millis(100), Snapshot::default());
        assert_eq!(CollectiveProfile::path_for(Path::new("collectives.json"), 1), PathBuf::from("collectives.json.rank1.json"));

        let json = profile.to_json();
        assert_eq!(json["rank"], 1);
        assert_eq!(json["world"], 2);
        assert_eq!(json["plan"], "plan-bytes");
        assert_eq!(json["run_ms"], 100.0);
        assert_eq!(json["share_of_run"], 0.0);
        assert_eq!(json["count"], 0);
        assert_eq!(json["bytes"], 0);
    }

    #[test]
    fn a_rank_profile_writes_one_document_at_its_own_path() {
        let dir = std::env::temp_dir().join(format!("brain-collective-profile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("run.json");
        let profile = CollectiveProfile::new(0, 2, "plan", Duration::from_millis(5), Snapshot::default());
        let path = profile.write(&base).unwrap();
        assert_eq!(path, CollectiveProfile::path_for(&base, 0));
        let parsed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["rank"], 0);
        assert_eq!(parsed["world"], 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile(&samples, 50), Duration::from_millis(50));
        assert_eq!(percentile(&samples, 95), Duration::from_millis(95));
        assert_eq!(percentile(&samples, 99), Duration::from_millis(99));
        assert_eq!(percentile(&samples[..1], 99), Duration::from_millis(1));
        assert_eq!(percentile(&[], 50), Duration::ZERO);
    }

    #[test]
    fn shaped_link_pays_latency_in_both_directions() {
        let mut l = link();
        l.one_way_latency = Duration::from_millis(2);
        assert_eq!(l.delay_for(0, 1 << 20), Duration::from_millis(4));
    }

    #[test]
    fn shaped_link_caps_transfer_at_bandwidth() {
        let mut l = link();
        l.bandwidth_bytes_per_sec = 1e9;
        // The transfer term is payload divided by bandwidth: a payload of
        // 1_000_000 bytes on a link of 1e9 bytes per second costs 1e6 ns.
        let delay = l.delay_for(0, 1_000_000).as_nanos() as i128;
        assert!((delay - 1_000_000).abs() <= 1, "got {delay} ns");
        // Twice the bytes, twice the transfer time.
        let double = l.delay_for(0, 2_000_000).as_nanos() as i128;
        assert!((double - 2_000_000).abs() <= 1, "got {double} ns");
    }

    #[test]
    fn shaped_jitter_is_seeded_bounded_and_reproducible() {
        let mut l = link();
        l.jitter = Duration::from_millis(5);
        l.seed = data::rng::random_seed();

        // Two links with the same seed charge the same delay for the same op.
        let same = l;
        for index in [0u64, 1, 7, 1234] {
            assert_eq!(l.delay_for(index, 1000), same.delay_for(index, 1000));
        }

        // The draw stays within the configured bound and is not always the same.
        let mut seen = Vec::new();
        for index in 0..16u64 {
            let delay = l.delay_for(index, 0);
            assert!(delay <= l.jitter, "jitter {delay:?} over bound {:?}", l.jitter);
            seen.push(delay);
        }
        assert!(seen.windows(2).any(|w| w[0] != w[1]), "the jitter draw never changed across op indices");

        // A different seed is a different stream.
        l.seed ^= 1;
        assert_ne!(same.delay_for(0, 0), l.delay_for(0, 0));
    }

    #[test]
    fn shaped_collective_fails_before_the_inner_op_at_its_firing_point() {
        let inner = Counting::new();
        let mut l = link();
        l.fail_after_ops = Some(2);
        let shaped = ShapedCollective::new(&inner, l);

        assert_eq!(pollster::block_on(shaped.all_reduce(0, Payload::f32(vec![1.0]))), Ok(Payload::f32(vec![1.0])));
        assert_eq!(pollster::block_on(shaped.all_reduce(0, Payload::f32(vec![2.0]))), Ok(Payload::f32(vec![2.0])));
        let third = pollster::block_on(shaped.all_reduce(0, Payload::f32(vec![3.0])));
        assert!(matches!(third, Err(CollectiveError::Transport(_))), "expected a transport error, got {third:?}");
        assert_eq!(inner.ops(), 2, "the failure must fire before the inner op runs");
    }

    #[test]
    fn shaped_collective_charges_the_simulated_delay() {
        let inner = Counting::new();
        let mut l = link();
        l.one_way_latency = Duration::from_millis(1);
        let shaped = ShapedCollective::new(&inner, l);
        let started = Instant::now();
        pollster::block_on(shaped.all_reduce(0, Payload::f32(vec![1.0]))).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(2), "the op did not pay the round trip");
    }

    #[test]
    fn a_one_rank_group_pays_no_simulated_link_delay() {
        let inner = HostCollective::new(1);
        let link = SimulatedLink { one_way_latency: Duration::from_millis(200), bandwidth_bytes_per_sec: 1e6, jitter: Duration::ZERO, seed: data::rng::random_seed(), fail_after_ops: Some(0) };
        let shaped = ShapedCollective::new(&*inner, link);
        let started = std::time::Instant::now();
        let out = pollster::block_on(shaped.all_reduce(0, Payload::f32(vec![1.0, 2.0]))).expect("one rank cannot fail");
        assert_eq!(out.data, vec![1.0, 2.0]);
        assert!(started.elapsed() < Duration::from_millis(50), "a one-rank op slept for the simulated link: {:?}", started.elapsed());
    }
}
