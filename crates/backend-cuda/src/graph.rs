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
//!    the dispatch's structure and not on its parameter *values*. A captured
//!    graph gives every step its own slice of ONE graph-private parameter
//!    block, and the graph's first node copies the whole block from pinned host
//!    staging at once. One node, not one per step: a copy node costs the device
//!    ~12 us between kernels, which at 2500 steps per token made a replayed
//!    decode slower than launching every dispatch on its own. The block is
//!    private to the graph so two steps that share a uniform when issued one at
//!    a time (same kernel, same buffers, different parameters) cannot share
//!    storage here. Pinned, because a copy whose source is pageable memory is
//!    rejected during capture.
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
use std::sync::{Arc, Weak};

use crate::driver::{CuDevicePtr, CuFunction, CuGraphNode, CuKernelNodeParams};
use crate::exec;

/// Alignment, in words, of every parameter block carved out of a pinned
/// staging allocation. Four words keeps each block on a 16-byte boundary, which
/// is what the widest scalar a kernel reads from its uniform needs.
pub(crate) const STAGING_ALIGN_WORDS: usize = 4;

/// One dispatch's *structure*: everything a graph node records, minus the two
/// things a replay is allowed to change.
///
/// Parameter values are absent because they are re-copied per replay, and the
/// thread count is absent because a changed grid is re-pointed rather than
/// re-recorded. Buffers are identified by the address of their `Arc`, not by
/// their device address: a device address may be recycled by the allocator
/// after a free, whereas an `Arc` that something still holds cannot be, and
/// every captured node holds one.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct NodeSig {
    /// The address of the `Arc<Compiled>` this dispatch resolved to - the
    /// kernel's identity on this handle, not its catalogue index, because two
    /// indices can name the same compiled module and a `kind` is meaningless
    /// across the catalogue/native boundary.
    pub kernel: usize,
    pub uniform: usize,
    pub words: usize,
    pub bufs: Vec<(usize, u64)>,
    /// Bound range length in words per binding. Part of the structure because
    /// it is baked into the node's argument list, which a replay never
    /// rewrites: two submissions differing only in a slice length must not
    /// share a graph.
    pub lens: Vec<u64>,
}

/// A whole submission's structure.
#[derive(Clone, PartialEq, Eq, Default, Hash)]
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
    /// Nodes given new by-value scalar arguments - see
    /// [`crate::backend::LaunchStats::graph_scalar_updates`].
    pub scalar_updates: AtomicU64,
    /// How often a replay had to wait for the device before it could change
    /// something the in-flight graph was still using. Counted rather than
    /// hidden: it is the one place this mechanism can give host time back,
    /// and a caller whose parameters change every submission and which never
    /// reads between them would see it climb.
    pub staging_waits: AtomicU64,
    /// Parameter copy nodes recorded into graphs - see
    /// [`crate::backend::LaunchStats::graph_param_copies`].
    pub param_copies: AtomicU64,
}

/// Everything a captured submission needs to be replayed, including WEAK
/// handles on every allocation its nodes name.
///
/// A graph node holds a bare device address; if the allocation behind it were
/// freed, the allocator could hand that address to the next caller and the
/// graph would then read and write a live tensor at full speed, with nothing
/// to fault on. What prevents that is [`LiveGraph::epoch`]: freeing a block a
/// graph names bumps the context's graph epoch, and a graph captured under an
/// older one is discarded before it can be matched, so it is never replayed
/// against a recycled address. Blocks no graph names do not move it.
///
/// The handles are weak because strong ones made a graph an owner of the
/// model's tensors: a model that dropped its buffers while the device handle
/// lived on (a handle serves the next model) found every weight still resident
/// behind the graph, until some later submission happened to evict it. A weak
/// handle keeps no memory alive, and still reserves the `Arc`'s own address, so
/// the identity [`NodeSig`] compares by cannot be recycled while the graph
/// exists.
pub(crate) struct LiveNode {
    node: CuGraphNode,
    func: CuFunction,
    /// Keeps the module loaded - `func` is only a valid handle while it is -
    /// and carries the launch shape a re-pointed node must repeat.
    compiled: Arc<crate::backend::Compiled>,
    block_dim: u32,
    /// What one block covers for THIS kind of kernel - the work-group size for
    /// a catalogue kernel, and one for a native kernel whose `threads` already
    /// counts blocks.
    per_block: u32,
    threads: u32,
    /// The kernel's argument slots: pointers, then by-value scalars from
    /// [`LiveNode::scalar_at`]. A replay overwrites the scalar tail in place.
    args: Vec<CuDevicePtr>,
    scalar_at: usize,
    /// Where in the graph's pinned staging ([`LiveGraph::staging`]), and so in
    /// its device parameter block ([`LiveGraph::_params`], laid out
    /// identically), this node's parameters live, as `(word offset, words)`.
    /// `None` for a step with no parameters of its own to supply.
    staging: Option<(usize, usize)>,
    /// What is currently in `staging`. A replay that would write the same
    /// words writes nothing, which is not a micro-optimisation: writing is
    /// what forces the wait below, so a submission that repeats unchanged
    /// costs one `cuGraphLaunch` and no synchronisation at all.
    staged: Vec<u32>,
    _uniform: Option<Weak<exec::DeviceMem>>,
    _bufs: Vec<(Weak<exec::DeviceMem>, u64)>,
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
    /// The context's graph epoch when this was captured.
    epoch: u64,
    /// Kept only to own it: an instantiated graph outlives the graph it was
    /// instantiated from, but destroying the source early would serve no
    /// purpose and a re-capture is what replaces both.
    _graph: exec::Graph,
    exec: exec::GraphExec,
    /// One page-locked block holding every node's parameters, each at the
    /// offset its [`LiveNode::staging`] names. One driver allocation per graph
    /// rather than one per node: a decode graph is thousands of nodes.
    staging: Option<exec::PinnedMem>,
    /// The device block the graph's one parameter copy fills and every node's
    /// uniform pointer points into. Declared after `exec` so the instantiated
    /// graph is released first.
    _params: Option<exec::DeviceMem>,
    _clears: Vec<Weak<exec::DeviceMem>>,
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
    /// Index in `args` of the first by-value scalar (`args.len()` when the
    /// kernel takes none). Everything before it is a pointer the signature
    /// pins; everything from it on may change between replays.
    pub scalar_at: usize,
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
    /// Bound range length in words per binding; already appended to `args`
    /// for a kernel that takes them, kept for the submission signature.
    pub lens: Vec<u64>,
}

// `func` is an opaque driver handle, valid for as long as `compiled`'s module
// is loaded (which `compiled` guarantees) and not thread-affine - the context is
// made current before every call that uses it. Everything else is `Send`. A
// pass holds these between `begin_pass` and `end_pass` behind a `Mutex`.
unsafe impl Send for Resolved {}

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
/// Eight, because a decode token is itself several graphs: its layer stack is
/// issued in chunks (see `crate::backend::PASS_FLUSH_STEPS`) so the card can
/// start on the first while the host builds the rest, and the head after its
/// readback is one more. A bound below that would evict a chunk every token.
///
/// Small on purpose: each one holds an instantiated graph and page-locked
/// staging until it is evicted, replaced or invalidated by a free.
const MAX_LIVE: usize = 8;

/// How many distinct recent shapes [`GraphCache::seen`] remembers.
const MAX_SEEN: usize = 8;

/// How many capture or replay failures a handle tolerates before it stops
/// trying - see [`GraphCache::forget`].
const MAX_FAILURES: u32 = 3;

/// The per-handle capture state machine.
#[derive(Default)]
pub(crate) struct GraphCache {
    /// Captured graphs, most recently captured first. Bounded by [`MAX_LIVE`].
    live: Vec<LiveGraph>,
    /// The CHUNKS (see [`Self::plan`]) most recently submitted eagerly, as
    /// `(fingerprint, epoch)`, newest last and bounded by [`MAX_SEEN`]. A second
    /// sighting of one is what triggers a capture: capturing the first would pay
    /// an instantiation for a shape that may never recur.
    ///
    /// More than the last one, because a decode token is several chunks in a
    /// fixed order (the layer stack's, then the head's after a readback) and
    /// none ever repeats back to back: remembering only the previous shape meant
    /// none was ever captured. A fingerprint rather than the shape itself because
    /// a chunk is hundreds of nodes, and a collision costs one needless capture,
    /// never a wrong replay - a replay is matched on the full [`SubmitSig`].
    seen: Vec<(u64, u64)>,
    /// The last submission that was NOT a chunk, issued eagerly: the original
    /// trigger, for submissions made one at a time. Only an immediate repeat
    /// captures. The wider [`Self::seen`] window must not apply here: a prefill
    /// round is hundreds of small submissions of which some recur every few
    /// layers, and capturing each of those paid an instantiation (milliseconds)
    /// per submission - a one-row chunk round went from 143 ms to 1473 ms.
    previous: Option<(u64, u64)>,
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
    ///
    /// `chunk` says the submission is a pass chunk - a large piece of a step the
    /// caller repeats - rather than one the caller made on its own. A chunk is
    /// captured when it has been seen before at all (within the last
    /// [`MAX_SEEN`] chunks); any other submission only when it repeats the one
    /// before it.
    pub fn plan(&mut self, ctx: &exec::Context, sig: &SubmitSig, epoch: u64, chunk: bool) -> Plan {
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
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            sig.hash(&mut h);
            h.finish()
        };
        if !chunk {
            if self.previous == Some((fingerprint, epoch)) {
                self.previous = None;
                return Plan::Capture;
            }
            self.previous = Some((fingerprint, epoch));
            return Plan::Eager;
        }
        if let Some(i) = self.seen.iter().position(|&(f, e)| f == fingerprint && e == epoch) {
            self.seen.remove(i);
            return Plan::Capture;
        }
        if self.seen.len() == MAX_SEEN {
            self.seen.remove(0);
        }
        self.seen.push((fingerprint, epoch));
        Plan::Eager
    }

    /// Called when the host has synchronised with the device, at allocator
    /// epoch `epoch`.
    ///
    /// Graphs captured under an older epoch can never be replayed again, so
    /// they are released here rather than at the next submission: a handle
    /// whose model is gone may never submit again, and each of them holds an
    /// instantiated graph and page-locked staging.
    pub fn drained(&mut self, epoch: u64) {
        self.in_flight = false;
        self.live.retain(|l| l.epoch == epoch);
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
        // The staging is allocated BEFORE the capture opens: `cuMemAllocHost`
        // is not a stream operation and has no business inside a captured
        // region.
        let mut words = 0usize;
        let mut slots = Vec::with_capacity(steps.len());
        for s in steps {
            slots.push(match (s.uniform.as_ref(), s.params.is_empty()) {
                (Some(_), false) => {
                    let at = words;
                    words += s.params.len().next_multiple_of(STAGING_ALIGN_WORDS);
                    Some((at, s.params.len()))
                }
                _ => None,
            });
        }
        let staging = if words > 0 { Some(ctx.pinned(words)?) } else { None };
        // The device block every node's uniform pointer will point into. Also
        // allocated before the capture opens, and zeroed along with its host
        // image: the alignment padding between two steps' parameters is part
        // of the one copy and must not be whatever the allocator left there.
        let params = if words > 0 {
            let block = ctx.alloc(words * 4)?;
            ctx.zero(&block)?;
            if let Some(st) = staging.as_ref() {
                st.fill_at(0, &vec![0u32; words]);
            }
            Some(block)
        } else {
            None
        };

        let capture = ctx.begin_capture()?;
        // The graph's ONE parameter copy: first, so every kernel after it reads
        // this replay's values. See the module doc for why it is one node.
        if let (Some(block), Some(st)) = (params.as_ref(), staging.as_ref()) {
            ctx.upload_async_range(block, st, 0, words)?;
            counters.param_copies.fetch_add(1, Ordering::Relaxed);
        }
        // The submission's clears are nodes of the graph like everything else.
        // Leaving them out would make a replay accumulate into buffers the
        // unbatched path had zeroed first, which is a wrong number rather than
        // a failure.
        for c in clears {
            ctx.zero(c)?;
        }
        let mut nodes = Vec::with_capacity(steps.len());
        for (s, slot) in steps.iter().cloned().zip(slots) {
            let mut args = s.args;
            if let (Some((at, _)), Some(block)) = (slot, params.as_ref()) {
                // A step with parameters of its own takes its uniform from the
                // graph's private block - always the first argument, because
                // `Backend::resolve` puts the uniform before the bindings.
                args[0] = block.device_ptr() + (at * 4) as u64;
            }
            let (gx, gy) = backend_api::grid_ws(s.threads, s.per_block);
            // SAFETY: `s.func` was resolved from `s.compiled`'s module, which
            // the node below keeps loaded, and `s.args` is the argument list
            // that module's entry point takes.
            unsafe { ctx.launch_shaped(s.func, (gx, gy, 1), (s.compiled.block_dim, 1, 1), &s.compiled.shape, &args)? };
            nodes.push(LiveNode {
                node: capture.last_node()?,
                func: s.func,
                compiled: s.compiled.clone(),
                block_dim: s.compiled.block_dim,
                per_block: s.per_block,
                threads: s.threads,
                args,
                scalar_at: s.scalar_at,
                staging: slot,
                staged: Vec::new(),
                _uniform: s.uniform.as_ref().map(Arc::downgrade),
                _bufs: s.bufs.iter().map(|(m, off)| (Arc::downgrade(m), *off)).collect(),
            });
        }
        let graph = capture.finish()?;
        let exec = ctx.instantiate(&graph)?;
        // Every block a node names, so that freeing one of them - and only
        // those - invalidates graphs (`Context::graph_epoch`).
        for s in steps {
            s.uniform.iter().for_each(|u| u.mark_graph_referenced());
            s.bufs.iter().for_each(|(m, _)| m.mark_graph_referenced());
        }
        clears.iter().for_each(|c| c.mark_graph_referenced());
        // The graph's own parameter block is not marked: only the graph holds
        // its address, so releasing it with the graph names nobody else.
        counters.captures.fetch_add(1, Ordering::Relaxed);
        self.live.insert(
            0,
            LiveGraph {
                sig: sig.clone(),
                epoch,
                _graph: graph,
                exec,
                staging,
                _params: params,
                _clears: clears.iter().map(Arc::downgrade).collect(),
                nodes,
            },
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
        let changed = live.nodes.iter().zip(steps).any(|(n, s)| {
            n.threads != s.threads
                || n.args[n.scalar_at..] != s.args[s.scalar_at..]
                || (n.staging.is_some() && n.staged != s.params)
        });
        if changed && self.in_flight {
            counters.staging_waits.fetch_add(1, Ordering::Relaxed);
            ctx.sync()?;
            self.in_flight = false;
        }
        for (node, step) in live.nodes.iter_mut().zip(steps) {
            if let (Some((at, _)), Some(st)) = (node.staging, live.staging.as_ref()) {
                if node.staged != step.params {
                    st.fill_at(at, &step.params);
                    node.staged.clear();
                    node.staged.extend_from_slice(&step.params);
                }
            }
            let scalars_changed = node.args[node.scalar_at..] != step.args[step.scalar_at..];
            if node.threads != step.threads || scalars_changed {
                if scalars_changed {
                    // Scalars are baked into the node's argument list; the
                    // pointers before them are what the signature pinned.
                    let at = node.scalar_at;
                    node.args[at..].copy_from_slice(&step.args[at..]);
                }
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
                    // Dynamic shared memory is part of the node and survives
                    // a re-point only if it is passed again here.
                    shared_mem_bytes: node.compiled.shape.shared_bytes,
                    kernel_params: params.as_mut_ptr(),
                    extra: std::ptr::null_mut(),
                };
                // SAFETY: `node.node` is a node of the graph `live.exec` was
                // instantiated from, and `params` names exactly the arguments
                // the node was recorded with - the same `args` vector, whose
                // addresses cannot have moved because the allocations behind
                // them are held by this node.
                unsafe { ctx.set_kernel_node_grid(&live.exec, node.node, &p)? };
                if node.threads != step.threads {
                    node.threads = step.threads;
                    counters.grid_updates.fetch_add(1, Ordering::Relaxed);
                }
                if scalars_changed {
                    counters.scalar_updates.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        ctx.launch_graph(&live.exec)?;
        self.in_flight = true;
        counters.replays.fetch_add(1, Ordering::Relaxed);
        crate::live::graph_replayed();
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
        self.seen.clear();
        self.previous = None;
        self.failures += 1;
        self.failures == MAX_FAILURES
    }
}

/// A run of submissions frozen at its current parameters: captured into a graph
/// once, launched with one driver call, nothing matched or staged per launch.
///
/// This is the other half of what a [`GraphCache`] does. The cache decides, per
/// submission, whether a graph it holds describes it - which costs a resolve and
/// a structural comparison of every dispatch, tens of milliseconds for a decode
/// step - and lets parameters and grids change between replays. A frozen program
/// is for the caller that KNOWS nothing changes (a recorded tape, `gpu_core::
/// tape`): the dispatches are the same and what varies reaches them through
/// buffers, so a launch needs none of that.
///
/// Parameters are frozen by giving every step its own PRIVATE uniform block,
/// filled once before the capture. The shared uniform blocks the eager path and a
/// [`GraphCache`] use are keyed on a dispatch's structure with its parameter
/// values excluded, so two steps of one shape in one submission share a block and
/// are told apart only by the copy node that rewrites it between them - which is
/// the per-replay staging this avoids, and the reason a frozen graph is kernel
/// nodes only and needs no pinned host memory.
///
/// Holds every allocation its nodes name by address - the bound buffers, the
/// private uniforms, the buffers cleared first - and the modules the kernels
/// live in, so no address can be freed and recycled behind it.
pub(crate) struct Frozen {
    _graph: exec::Graph,
    pub(crate) exec: exec::GraphExec,
    /// Buffers (and their allocator-visible owners) the nodes read or write.
    pub(crate) touched: Vec<Arc<exec::DeviceMem>>,
    _uniforms: Vec<exec::DeviceMem>,
    _modules: Vec<Arc<crate::backend::Compiled>>,
    /// Dispatches in the program, for the host-cost counters.
    pub(crate) dispatches: usize,
}

// Same reasoning as `LiveNode`: the contents are opaque driver handles and
// reference-counted allocations, and a launch makes the context current first.
unsafe impl Send for Frozen {}
unsafe impl Sync for Frozen {}

impl Frozen {
    /// Capture `segments` - `(buffers to zero first, resolved dispatches)`, in
    /// order - as one program. Fails, having issued nothing, if the driver
    /// refuses any part of it.
    pub fn capture(ctx: &exec::Context, segments: Vec<(Vec<Arc<exec::DeviceMem>>, Vec<Resolved>)>) -> Result<Frozen, String> {
        Self::capture_inner(ctx, segments, None).map(|(f, _)| f)
    }

    /// [`Self::capture`] with an event recorded after every dispatch (and one
    /// before the first), returned with each dispatch's kernel name: the program
    /// then carries its own device-side timeline.
    pub fn capture_profiled(
        ctx: &exec::Context,
        segments: Vec<(Vec<Arc<exec::DeviceMem>>, Vec<Resolved>)>,
    ) -> Result<(Frozen, Vec<(String, exec::Event)>, exec::Event), String> {
        let mut stamps = Vec::new();
        let start = ctx.event()?;
        let (frozen, _) = Self::capture_inner(ctx, segments, Some((&start, &mut stamps)))?;
        Ok((frozen, stamps, start))
    }

    fn capture_inner(
        ctx: &exec::Context,
        segments: Vec<(Vec<Arc<exec::DeviceMem>>, Vec<Resolved>)>,
        mut stamps: Option<(&exec::Event, &mut Vec<(String, exec::Event)>)>,
    ) -> Result<(Frozen, ()), String> {
        // Private uniforms first, before any capture opens: an allocation and a
        // copy are not stream operations a capture may contain.
        let mut uniforms: Vec<exec::DeviceMem> = Vec::new();
        let mut private: Vec<Vec<Option<usize>>> = Vec::with_capacity(segments.len());
        for (_, steps) in &segments {
            let mut idx = Vec::with_capacity(steps.len());
            for s in steps {
                if s.uniform.is_some() && !s.params.is_empty() {
                    let mem = ctx.alloc((s.params.len() * 4).max(4))?;
                    // SAFETY: `u32` has no padding, so the words are valid bytes.
                    let bytes = unsafe { std::slice::from_raw_parts(s.params.as_ptr() as *const u8, s.params.len() * 4) };
                    ctx.upload(&mem, bytes)?;
                    idx.push(Some(uniforms.len()));
                    uniforms.push(mem);
                } else {
                    idx.push(None);
                }
            }
            private.push(idx);
        }

        let capture = ctx.begin_capture()?;
        if let Some((start, _)) = &stamps {
            ctx.record(start)?;
        }
        let mut touched: Vec<Arc<exec::DeviceMem>> = Vec::new();
        let mut modules: Vec<Arc<crate::backend::Compiled>> = Vec::new();
        let mut dispatches = 0usize;
        for ((clears, steps), idx) in segments.iter().zip(&private) {
            for c in clears {
                ctx.zero(c)?;
                touched.push(c.clone());
            }
            for (s, own) in steps.iter().zip(idx) {
                let mut args = s.args.clone();
                if let Some(u) = own {
                    // `Resolved::args[0]` is the uniform pointer of a kernel that takes one.
                    args[0] = uniforms[*u].device_ptr();
                }
                let (gx, gy) = backend_api::grid_ws(s.threads, s.per_block);
                // SAFETY: `s.func` was resolved from `s.compiled`'s module, which
                // this program keeps loaded, and `args` is the argument list that
                // module's entry point takes.
                unsafe { ctx.launch_raw(s.func, (gx, gy, 1), (s.compiled.block_dim, 1, 1), &args)? };
                touched.extend(s.bufs.iter().map(|(m, _)| m.clone()));
                touched.extend(s.uniform.iter().cloned());
                modules.push(s.compiled.clone());
                dispatches += 1;
                if let Some((_, out)) = stamps.as_mut() {
                    let ev = ctx.event()?;
                    ctx.record(&ev)?;
                    out.push((s.compiled.name.clone(), ev));
                }
            }
        }
        let graph = capture.finish()?;
        let exec = ctx.instantiate(&graph)?;
        modules.sort_by_key(|m| Arc::as_ptr(m) as usize);
        modules.dedup_by_key(|m| Arc::as_ptr(m) as usize);
        touched.sort_by_key(|m| Arc::as_ptr(m) as usize);
        touched.dedup_by_key(|m| Arc::as_ptr(m) as usize);
        Ok((Frozen { _graph: graph, exec, touched, _uniforms: uniforms, _modules: modules, dispatches }, ()))
    }
}
