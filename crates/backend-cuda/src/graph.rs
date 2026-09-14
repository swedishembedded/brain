// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Capture a repeated submission into a CUDA graph once, then replay it - one
//! driver call in place of one per dispatch.
//!
//! Swedish Embedded AB implements batched GPU submission for its clients. If
//! your team needs expertise in cutting host-side launch cost out of a compute
//! pipeline without loosening what the pipeline computes, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # What is being paid for
//!
//! A dispatch costs the host a driver call whether or not the device is busy,
//! and a decode step of a model of any size is thousands of them. Recording
//! them once into a graph and replaying it collapses that to one call, and
//! changes nothing about what the device does - the same kernels, the same
//! order, the same arguments.
//!
//! # What makes a submission replayable, and what does not
//!
//! A graph node holds device *addresses*, a function handle and a grid. So a
//! submission may be replayed exactly when every one of those is what it was
//! when the graph was recorded. Three things move between one submission and
//! the next in this engine, and they are handled differently on purpose:
//!
//! 1. **Parameter values** change every step of every token (a position, a
//!    scale, a length). They must NOT be allowed to change a device address,
//!    or nothing is ever replayable - so the uniform allocation is keyed on
//!    the dispatch's structure and not on its parameter *values*, and each
//!    step's first graph node is a copy of that step's parameters from pinned
//!    host staging into that stable allocation. Pinned, because a copy whose
//!    source is pageable memory is rejected during capture.
//! 2. **The grid** grows with the sequence length. That is re-pointed inside
//!    the instantiated graph
//!    (`cuGraphExecKernelNodeSetParams`) rather than answered with a second
//!    graph: one instantiation per position costs more than the batching
//!    saves, which is the measurement that decided this design.
//! 3. **Anything else** - a different kernel, a different buffer, a different
//!    number of steps, or any free at all in the allocator - discards the
//!    graph and starts over. The set of things treated as replayable is
//!    deliberately narrow; a wrong replay is silent, not loud.
//!
//! # The capture region is one submission and nothing more
//!
//! A capture forbids synchronising calls from the capturing thread, and brain
//! reads logits from the device every single token. Capture therefore begins
//! and ends inside one [`backend_api::Backend::submit`] call, so a read can
//! only ever land between two submissions. Kernel compilation and entry-point
//! resolution are hoisted out of the region for the same reason. The
//! confinement is not left as a comment: the backend refuses a `read` or a
//! `poll_wait` that arrives while a capture is open, and
//! `tests/cuda_graphs.rs` drives the read-per-submission loop a decoder
//! actually performs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::driver::{CuDevicePtr, CuFunction, CuGraphNode, CuKernelNodeParams};
use crate::exec;

/// One dispatch's *structure*: everything a graph node records, minus the two
/// things a replay is allowed to change.
///
/// Parameter values are absent because they are re-copied per replay, and the
/// thread count is absent because a changed grid is re-pointed rather than
/// re-recorded. Buffers are identified by the address of their `Arc`, not by
/// their device address: a device address may be recycled by the allocator
/// after a free, whereas an `Arc` that something still holds cannot be, and
/// every captured node holds one.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct NodeSig {
    /// The address of the `Arc<Compiled>` this dispatch resolved to - the
    /// kernel's identity on this handle, not its catalogue index, because two
    /// indices can name the same compiled module and a `kind` is meaningless
    /// across the catalogue/native boundary.
    pub kernel: usize,
    pub uniform: usize,
    pub words: usize,
    pub bufs: Vec<(usize, u64)>,
}

/// A whole submission's structure.
#[derive(Clone, PartialEq, Eq, Default)]
pub(crate) struct SubmitSig {
    pub clears: Vec<usize>,
    pub nodes: Vec<NodeSig>,
}

/// Counters the host-cost report reads back - see
/// [`crate::backend::LaunchStats`].
#[derive(Default)]
pub(crate) struct GraphCounters {
    pub captures: AtomicU64,
    pub replays: AtomicU64,
    pub grid_updates: AtomicU64,
    /// How often a replay had to wait for the device before it could change
    /// something the in-flight graph was still using. Counted rather than
    /// hidden: it is the one place this mechanism can give host time back,
    /// and a caller whose parameters change every submission and which never
    /// reads between them would see it climb.
    pub staging_waits: AtomicU64,
}

/// Everything a captured submission needs to be replayed, including strong
/// handles on every allocation its nodes name.
///
/// The strong handles are load-bearing, not defensive. A graph node holds a
/// bare device address; if the allocation behind it were freed, the allocator
/// could hand that address to the next caller and the graph would then read
/// and write a live tensor at full speed, with nothing to fault on. Holding
/// the allocation makes that impossible for as long as the graph exists.
pub(crate) struct LiveNode {
    node: CuGraphNode,
    func: CuFunction,
    /// Keeps the module loaded: `func` is only a valid handle while it is.
    _module: Arc<crate::backend::Compiled>,
    block_dim: u32,
    /// What one block covers for THIS kind of kernel - the work-group size for
    /// a catalogue kernel, and one for a native kernel whose `threads` already
    /// counts blocks.
    per_block: u32,
    threads: u32,
    args: Vec<CuDevicePtr>,
    /// Page-locked host staging this node's first graph node copies from.
    /// `None` for a step with no parameters of its own to supply.
    staging: Option<exec::PinnedMem>,
    /// What is currently in `staging`. A replay that would write the same
    /// words writes nothing, which is not a micro-optimisation: writing is
    /// what forces the wait below, so a submission that repeats unchanged
    /// costs one `cuGraphLaunch` and no synchronisation at all.
    staged: Vec<u32>,
    _uniform: Option<Arc<exec::DeviceMem>>,
    _bufs: Vec<(Arc<exec::DeviceMem>, u64)>,
}

// A `CUgraphNode` and a `CUfunction` are opaque driver handles, not pointers
// into this process's memory, and neither is thread-affine: the context is
// made current before every call that uses them. Everything else in these two
// types is already `Send + Sync`, and the whole cache lives behind a `Mutex`,
// so the handles are the only reason the auto-traits do not apply.
unsafe impl Send for LiveNode {}
unsafe impl Sync for LiveNode {}

pub(crate) struct LiveGraph {
    sig: SubmitSig,
    /// The allocator's free count when this was captured.
    epoch: u64,
    /// Kept only to own it: an instantiated graph outlives the graph it was
    /// instantiated from, but destroying the source early would serve no
    /// purpose and a re-capture is what replaces both.
    _graph: exec::Graph,
    exec: exec::GraphExec,
    _clears: Vec<Arc<exec::DeviceMem>>,
    nodes: Vec<LiveNode>,
}

/// What the backend must do with a submission it just described.
pub(crate) enum Plan {
    /// Issue every dispatch one at a time. The submission's shape has been
    /// remembered, so an identical one next time is a capture candidate.
    Eager,
    /// Record this submission into a graph, then replay it.
    Capture,
    /// This exact shape is already captured; replay the graph at this index.
    Replay(usize),
}

/// One dispatch, resolved down to what both the eager path and a graph node
/// need: a function, a grid and a flat argument list.
#[derive(Clone)]
pub(crate) struct Resolved {
    pub compiled: Arc<crate::backend::Compiled>,
    pub func: CuFunction,
    pub args: Vec<CuDevicePtr>,
    pub per_block: u32,
    pub threads: u32,
    /// The step's own parameter words, or empty when its uniform is supplied
    /// by the caller and this backend has nothing to copy into it.
    pub params: Vec<u32>,
    pub uniform: Option<Arc<exec::DeviceMem>>,
    /// `(allocation, byte offset)` per storage binding. The offsets are
    /// already folded into `args`; these are kept so a captured node can hold
    /// every allocation it names alive.
    pub bufs: Vec<(Arc<exec::DeviceMem>, u64)>,
}

/// How many captured graphs a handle keeps at once.
///
/// One is not enough, and the reason is a measurement rather than a guess:
/// instantiating a graph costs milliseconds, far more than the launch overhead
/// a replay saves, so it only pays when it is amortised over many replays. A
/// caller that alternates between a small number of shapes - a forward and a
/// backward, a prefill and a decode, or a benchmark comparing two providers -
/// would re-instantiate on every switch with a single slot, and be slower than
/// launching each dispatch. A handful of slots makes each shape's
/// instantiation a once-per-shape cost, which is what the design assumed.
///
/// Small on purpose: every captured graph holds every allocation it names
/// alive, so the bound is also a bound on how long a freed-by-the-model buffer
/// can linger.
const MAX_LIVE: usize = 4;

/// How many capture or replay failures a handle tolerates before it stops
/// trying - see [`GraphCache::forget`].
const MAX_FAILURES: u32 = 3;

/// The per-handle capture state machine.
#[derive(Default)]
pub(crate) struct GraphCache {
    /// Captured graphs, most recently captured first. Bounded by [`MAX_LIVE`].
    live: Vec<LiveGraph>,
    /// The last shape submitted eagerly. A second consecutive submission of it
    /// is what triggers a capture: capturing the first would pay an
    /// instantiation for a shape that may never recur.
    seen: Option<(SubmitSig, u64)>,
    /// Captures or replays that returned an error - see [`Self::forget`].
    failures: u32,
    /// Set while a replayed graph may still be executing.
    ///
    /// A replay that has to CHANGE something - new parameter words, or a new
    /// grid - is changing state the in-flight graph is still using, so it
    /// waits first. Neither of the two loops this mechanism exists for pays
    /// that wait: a decoder's parameters change every token but it reads its
    /// logits between submissions, which already drains the device, and a loop
    /// that resubmits an unchanged shape changes nothing to wait for.
    in_flight: bool,
}

impl GraphCache {
    /// Wait for the device if a replay may still be running.
    ///
    /// Required before **discarding** a captured graph, not only before
    /// changing one. Dropping a `LiveGraph` frees its page-locked staging
    /// immediately, and an in-flight replay's copy nodes may not have read it
    /// yet - freeing host memory a DMA is reading is a fault or a corruption
    /// rather than an error code. (The instantiated graph itself is safe: the
    /// driver defers `cuGraphExecDestroy` until pending launches complete. The
    /// staging has no such protection.)
    fn settle(&mut self, ctx: &exec::Context) {
        if self.in_flight {
            if let Err(e) = ctx.sync() {
                tracing::warn!(reason = %e, "backend-cuda: synchronising before discarding a captured graph failed");
            }
            self.in_flight = false;
        }
    }

    /// Decide what to do with the submission `sig` describes, and update the
    /// state to match the decision.
    pub fn plan(&mut self, ctx: &exec::Context, sig: &SubmitSig, epoch: u64) -> Plan {
        if self.failures >= MAX_FAILURES {
            return Plan::Eager;
        }
        // A free may have handed any recorded device address to someone else,
        // so a graph captured before one describes memory it no longer owns.
        if self.live.iter().any(|l| l.epoch != epoch) {
            self.settle(ctx);
            self.live.retain(|l| l.epoch == epoch);
        }
        if let Some(i) = self.live.iter().position(|l| l.sig == *sig) {
            return Plan::Replay(i);
        }
        if self.seen.as_ref().is_some_and(|(s, e)| *e == epoch && s == sig) {
            return Plan::Capture;
        }
        self.seen = Some((sig.clone(), epoch));
        Plan::Eager
    }

    /// Called when the host has synchronised with the device.
    pub fn drained(&mut self) {
        self.in_flight = false;
    }

    /// Record `steps` into a graph, instantiate it and adopt it.
    ///
    /// Every fallible step returns `Err` rather than panicking: a device that
    /// cannot capture must keep computing, one dispatch at a time. The caller
    /// falls back to the eager path on `Err`, having issued nothing, because
    /// the capture swallowed rather than executed whatever was recorded before
    /// the failure.
    pub fn capture(
        &mut self,
        ctx: &exec::Context,
        sig: &SubmitSig,
        epoch: u64,
        clears: &[Arc<exec::DeviceMem>],
        steps: &[Resolved],
        counters: &GraphCounters,
    ) -> Result<(), String> {
        // The staging blocks are allocated BEFORE the capture opens:
        // `cuMemAllocHost` is not a stream operation and has no business
        // inside a captured region.
        let mut staging = Vec::with_capacity(steps.len());
        for s in steps {
            staging.push(match (s.uniform.as_ref(), s.params.is_empty()) {
                (Some(_), false) => Some(ctx.pinned(s.params.len())?),
                _ => None,
            });
        }

        let capture = ctx.begin_capture()?;
        // The submission's clears are nodes of the graph like everything else.
        // Leaving them out would make a replay accumulate into buffers the
        // unbatched path had zeroed first, which is a wrong number rather than
        // a failure.
        for c in clears {
            ctx.zero_async(c)?;
        }
        let mut nodes = Vec::with_capacity(steps.len());
        for (s, stage) in steps.iter().cloned().zip(staging) {
            if let (Some(u), Some(st)) = (s.uniform.as_ref(), stage.as_ref()) {
                // The step's FIRST node: its parameters land in the stable
                // uniform allocation before its kernel reads them, in stream
                // order, every replay.
                ctx.upload_async(u, st)?;
            }
            let (gx, gy) = backend_api::grid_ws(s.threads, s.per_block);
            // SAFETY: `s.func` was resolved from `s.compiled`'s module, which
            // the node below keeps loaded, and `s.args` is the argument list
            // that module's entry point takes.
            unsafe { ctx.launch_raw(s.func, (gx, gy, 1), (s.compiled.block_dim, 1, 1), &s.args)? };
            nodes.push(LiveNode {
                node: capture.last_node()?,
                func: s.func,
                _module: s.compiled.clone(),
                block_dim: s.compiled.block_dim,
                per_block: s.per_block,
                threads: s.threads,
                args: s.args,
                staging: stage,
                staged: Vec::new(),
                _uniform: s.uniform,
                _bufs: s.bufs,
            });
        }
        let graph = capture.finish()?;
        let exec = ctx.instantiate(&graph)?;
        counters.captures.fetch_add(1, Ordering::Relaxed);
        self.live.insert(
            0,
            LiveGraph { sig: sig.clone(), epoch, _graph: graph, exec, _clears: clears.to_vec(), nodes },
        );
        // Oldest capture out. Dropping it releases its instantiated graph, its
        // pinned staging and its hold on every buffer it named.
        if self.live.len() > MAX_LIVE {
            self.settle(ctx);
            self.live.truncate(MAX_LIVE);
        }
        Ok(())
    }

    /// Supply this submission's parameters and grids to the captured graph and
    /// launch it.
    ///
    /// `steps` must be the submission the caller just matched against the
    /// captured signature, in order; the signature is what proves they
    /// correspond node for node.
    pub fn replay(
        &mut self,
        ctx: &exec::Context,
        at: usize,
        steps: &[Resolved],
        counters: &GraphCounters,
    ) -> Result<(), String> {
        let Some(live) = self.live.get_mut(at) else {
            return Err(format!("replay of graph {at}, of which there are {}", self.live.len()));
        };
        if live.nodes.len() != steps.len() {
            return Err(format!(
                "a submission of {} steps was matched to a graph of {} nodes",
                steps.len(),
                live.nodes.len()
            ));
        }
        // Both of the things a replay may change - the words in a staging
        // block, and a node's grid inside the instantiated graph - are illegal
        // to perform while that graph is still running: the copy nodes may not
        // have read the staging yet, and an executable graph may not be
        // updated with a launch pending. A replay that changes NEITHER is
        // therefore free of any wait, which is the whole reason the comparison
        // is made before the writes rather than after them.
        let changed = live
            .nodes
            .iter()
            .zip(steps)
            .any(|(n, s)| n.threads != s.threads || (n.staging.is_some() && n.staged != s.params));
        if changed && self.in_flight {
            counters.staging_waits.fetch_add(1, Ordering::Relaxed);
            ctx.sync()?;
            self.in_flight = false;
        }
        for (node, step) in live.nodes.iter_mut().zip(steps) {
            if let Some(st) = &node.staging {
                if node.staged != step.params {
                    st.fill(&step.params);
                    node.staged.clear();
                    node.staged.extend_from_slice(&step.params);
                }
            }
            if node.threads != step.threads {
                let (gx, gy) = backend_api::grid_ws(step.threads, node.per_block);
                let mut params: Vec<*mut std::ffi::c_void> = node
                    .args
                    .iter_mut()
                    .map(|a| a as *mut CuDevicePtr as *mut std::ffi::c_void)
                    .collect();
                let p = CuKernelNodeParams {
                    func: node.func,
                    grid_dim_x: gx,
                    grid_dim_y: gy,
                    grid_dim_z: 1,
                    block_dim_x: node.block_dim,
                    block_dim_y: 1,
                    block_dim_z: 1,
                    shared_mem_bytes: 0,
                    kernel_params: params.as_mut_ptr(),
                    extra: std::ptr::null_mut(),
                };
                // SAFETY: `node.node` is a node of the graph `live.exec` was
                // instantiated from, and `params` names exactly the arguments
                // the node was recorded with - the same `args` vector, whose
                // addresses cannot have moved because the allocations behind
                // them are held by this node.
                unsafe { ctx.set_kernel_node_grid(&live.exec, node.node, &p)? };
                node.threads = step.threads;
                counters.grid_updates.fetch_add(1, Ordering::Relaxed);
            }
        }
        ctx.launch_graph(&live.exec)?;
        self.in_flight = true;
        counters.replays.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Forget every captured graph after a capture or a replay failed, and say
    /// whether this was the failure that gave up.
    ///
    /// A capture that fails is worth another attempt - a transient device
    /// condition is a real thing - but a mechanism that keeps failing would
    /// otherwise retry forever, warning and re-issuing every submission twice
    /// for the rest of the process. After [`MAX_FAILURES`] this handle
    /// launches every dispatch on its own, like one on a driver with no graph
    /// entry points at all. `true` comes back exactly once, so the caller logs
    /// that decision once.
    pub fn forget(&mut self, ctx: &exec::Context) -> bool {
        self.settle(ctx);
        self.live.clear();
        self.seen = None;
        self.failures += 1;
        self.failures == MAX_FAILURES
    }
}
