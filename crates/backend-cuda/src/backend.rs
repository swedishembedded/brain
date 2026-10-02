// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`backend_api::Backend`] on the CUDA Driver API: buffers, host transfers,
//! lazily compiled dispatches and a capability report built from real device
//! queries.
//!
//! Swedish Embedded AB implements compute backends for its clients - the
//! allocation, transfer, compilation and launch layer a model runs on. If your
//! team needs expertise in bringing up an accelerator backend behind an
//! existing engine's kernel contract, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! # The kernel a `kind` names
//!
//! `Backend::step(kind, ..)` indexes the `(name, wgsl)` list the handle was
//! built with, so this backend's job is to turn WGSL index `kind` into a
//! launchable CUDA entry point. It does that through [`wgsl_cuda`] (naga IR ->
//! CUDA C++) and NVRTC, and it does it **the first time that `kind` is
//! actually dispatched** - never for the catalogue at construction. The
//! difference is not a micro-optimisation: a model registers hundreds of
//! kernels and dispatches a few dozen of them, and compiling the whole list up
//! front would put minutes of NVRTC in front of the first token. (This is the
//! one place this backend deliberately diverges from `backend-vulkan`, which
//! builds every pipeline at `Factory` time.)
//!
//! A kernel outside the generator's supported subset is a **panic naming the
//! kernel and the construct that was refused**, at the dispatch that needed
//! it. It is not a fallback to another device and not an approximation: an
//! explicitly requested backend is a hard contract, and a silently diverted
//! dispatch is the exact failure this whole effort exists to make impossible.
//!
//! # Ordering
//!
//! Every launch and every clear goes on one stream, which serialises them
//! against each other, so a dispatch sees the writes of every dispatch
//! recorded before it - the same guarantee the wgpu backend gets from the
//! barriers it inserts between passes. `submit` returns without waiting for
//! the device; `read`/`poll_wait` are where the host actually synchronises.
//!
//! That stream is created rather than being the legacy default one, because
//! the legacy stream cannot be captured into a graph. It is created
//! *blocking*, so the synchronous host transfers that still use the legacy
//! stream (`write_at`, `read`, `storage_init`) stay ordered against it exactly
//! as they were when everything shared one stream.
//!
//! # Batched submission
//!
//! A submission whose shape repeats is recorded into a CUDA graph and
//! thereafter replayed with a single driver call - see [`crate::graph`], which
//! states what "the same shape" means and what a replay is allowed to change.
//! It changes nothing about what the device computes; `tests/cuda_graphs.rs`
//! is what holds it to that.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use backend_api::arch::{ArchDesc, TierLevel, TierSupport};
use backend_api::{
    grid_ws, NumericSupport, workgroup_size_of, BufUsage, DType, DeviceBuffer, DeviceCaps, DeviceClass, DeviceStats,
    GpuIdentity, Step,
};

use crate::driver::{CuDevicePtr, CuFunction};
use crate::exec;
use crate::graph::{GraphCache, GraphCounters, NodeSig, Plan, Resolved, SubmitSig, STAGING_ALIGN_WORDS};

/// One device allocation, behind an `Arc` so a [`DeviceBuffer`] clone and a
/// recorded [`Step`] both keep it alive - `DeviceBuffer` aliasing is by design
/// in this engine, and a step must outlive the handle the caller dropped.
#[derive(Clone)]
pub struct CudaBuf {
    mem: Arc<exec::DeviceMem>,
}

impl CudaBuf {
    fn wrap(mem: exec::DeviceMem) -> DeviceBuffer {
        DeviceBuffer::new(CudaBuf { mem: Arc::new(mem) })
    }
    fn of(b: &DeviceBuffer) -> &CudaBuf {
        b.downcast_ref::<CudaBuf>()
    }
}

/// A recorded dispatch. Holds strong references to every allocation it names,
/// so the dispatch is still launchable after the caller drops its handles.
pub struct CudaStep {
    kind: usize,
    threads: u32,
    /// The uniform stream's device allocation - **shared** between every step
    /// of the same shape, not private to this one.
    ///
    /// `None` only for a kernel that declares no uniform block. See
    /// [`CudaBackend::uniform_for`] for why this is keyed on the dispatch's
    /// structure rather than allocated per step, and why writing the
    /// parameters into it is therefore deferred to `submit`.
    uniform: Option<Arc<exec::DeviceMem>>,
    /// This step's own parameter words, carried rather than uploaded at record
    /// time. Empty when the caller supplied the uniform buffer itself
    /// (`step_buf`), in which case this backend has nothing to copy into it.
    params: Vec<u32>,
    /// `(allocation, byte offset)` per storage binding, in binding order. The
    /// offset is how a sliced step expresses its sub-range: a kernel argument
    /// is a bare pointer, so the slice IS the address.
    bufs: Vec<(Arc<exec::DeviceMem>, u64)>,
    /// Length in 32-bit words of the range each storage binding exposes, in
    /// binding order: the sub-range for a sliced step, otherwise everything
    /// from the offset to the end of the allocation. A generated kernel is
    /// passed these after its pointers and clamps every array access to
    /// them, as WGSL requires; a native kernel is not passed them.
    lens: Vec<u64>,
}

/// Words of `mem` that a binding at `off_bytes` exposes: `len_words` of them,
/// or - when that is 0, the "to the end" spelling of a sliced step - all that
/// follow the offset. Never more than the allocation holds, because a length
/// that outruns it would let the kernel's clamp admit an out-of-bounds access.
fn bound_words(mem: &exec::DeviceMem, off_bytes: u64, len_words: u64) -> u64 {
    let avail = (mem.len() as u64).saturating_sub(off_bytes) / 4;
    if len_words == 0 { avail } else { len_words.min(avail) }
}

/// The 64-bit argument slots for a native kernel's by-value scalars, from the
/// step's parameter words. A 32-bit scalar is its value zero-extended - the
/// kernel reads the low four bytes of its slot - and a 64-bit one is its two
/// words, low first. `register`/`step_native` guarantee `words` holds exactly
/// the words the kinds consume.
fn encode_scalars(kinds: &[backend_api::ScalarKind], words: &[u32]) -> Vec<u64> {
    let mut at = 0;
    kinds
        .iter()
        .map(|k| {
            let v = if k.words() == 1 { u64::from(words[at]) } else { u64::from(words[at]) | (u64::from(words[at + 1]) << 32) };
            at += k.words();
            v
        })
        .collect()
}

/// One compiled kernel, cached under its `kind`.
pub(crate) struct Compiled {
    /// The entry point, resolved once when the module was loaded rather than
    /// at every dispatch. `cuModuleGetFunction` is a driver call, and a
    /// submission of a few thousand steps was paying one per step for an
    /// answer that cannot change while the module is loaded - which this
    /// struct guarantees, because it owns the module.
    pub(crate) func: CuFunction,
    /// Owned, never read: `func` is a handle INTO this module and is valid
    /// only while it stays loaded, so the module's lifetime is the whole
    /// reason it is here.
    _module: exec::Module,
    /// Threads per block. For a catalogue kernel this is the WGSL
    /// `@workgroup_size` and for a native one the block size the source's own
    /// index arithmetic is written against - never a tuning knob either way.
    pub(crate) block_dim: u32,
    n_bindings: usize,
    takes_uniform: bool,
    /// Whether the entry point takes the bound range lengths after its
    /// pointers: true for a generated kernel (see `wgsl_cuda::Kernel`), false
    /// for a native one, whose signature is the provider's own.
    takes_lengths: bool,
    /// Dynamic shared memory and cluster dimensions this kernel is launched
    /// with - fixed at registration, and carried into every captured node.
    pub(crate) shape: exec::Shape,
    /// By-value scalar arguments after the pointers; empty for a kernel that
    /// takes a uniform block (or nothing).
    pub(crate) scalars: &'static [backend_api::ScalarKind],
    /// Diagnostic name, so a launch failure says which kernel failed whether
    /// it came from the WGSL catalogue or from a provider's own registry.
    name: String,
    /// For a native kernel, what it was registered from - so registering the
    /// same spec again finds this module instead of loading another. `None`
    /// for a catalogue kernel.
    native_key: Option<NativeKey>,
}

/// The identity of a registered native kernel: everything
/// [`backend_api::NativeSpec::Cuda`] says about it. Two registrations with
/// equal keys are the same kernel and share one loaded module.
#[derive(Clone, Copy, PartialEq, Eq)]
struct NativeKey {
    src: &'static str,
    entry: &'static str,
    block_dim: u32,
    bindings: &'static [backend_api::BindKind],
    shared_bytes: u32,
    launch: backend_api::CudaLaunch,
}

/// What a uniform allocation is shared by: the dispatch's **structure**, with
/// its parameter VALUES and its thread count deliberately left out.
///
/// Excluding the values is the whole basis of graph replay - see
/// [`CudaBackend::uniform_for`]. Excluding the thread count is the less
/// obvious half and matters just as much: the grid is not a property of the
/// storage, and keying on it would hand every advancing sequence position a
/// fresh uniform address, which is precisely the event
/// `cuGraphExecKernelNodeSetParams` exists to survive. The parameter *length*
/// is in the key because it is the allocation's size, which is structure.
#[derive(Clone, PartialEq, Eq, Hash)]
struct UniformKey {
    kind: usize,
    words: usize,
    bufs: Vec<(usize, u64)>,
}

/// A cached uniform allocation, plus proof that the key still means what it
/// meant.
///
/// The key identifies buffers by the address of their `Arc`. A raw address is
/// only an identity while the thing at it is alive, so the weak handles are
/// what make the key safe to trust: a `Weak` reserves the allocation's slot,
/// so no later `Arc` can occupy that address, and a dead one says this entry
/// describes buffers that no longer exist.
struct UniformSlot {
    mem: Arc<exec::DeviceMem>,
    keep: Vec<Weak<exec::DeviceMem>>,
}

/// Page-locked host memory the unbatched path stages its parameters through.
///
/// It exists because the obvious alternative is a trap. A *synchronous*
/// host-to-device copy runs on the legacy default stream, and the legacy
/// stream is ordered against every blocking stream in the context - so one
/// synchronous uniform upload per dispatch does not merely cost a driver call,
/// it drains the device between every pair of dispatches. That was measured:
/// eight back-to-back dispatches of one kernel took an order of magnitude
/// longer that way than as eight launches with the uploads already done.
///
/// So the upload is enqueued on the dispatch stream instead, which requires
/// page-locked source memory that stays untouched until the copy runs.
///
/// # Bounded by construction
///
/// A step's parameters are a handful of words, so the pool is a few
/// [`STAGING_SLAB_WORDS`] slabs handed out by bump allocation rather than one
/// driver allocation per step: `cuMemAllocHost` is a driver call with its own
/// bookkeeping, and a decode token is thousands of steps. Regions stay valid
/// until the device has drained (`read`/`poll_wait`, which a decode step
/// performs every token), at which point the whole pool is reused from the
/// start. The pool never holds more than [`STAGING_MAX_WORDS`]; a caller that
/// never synchronises hits that bound and pays one synchronise to reuse it -
/// slow, never wrong, and never unbounded pinned memory. Everything is freed
/// when the handle drops.
#[derive(Default)]
struct StagingPool {
    slabs: Vec<exec::PinnedMem>,
    /// The slab regions are currently being carved from, and how many of its
    /// words are taken.
    cursor: usize,
    used: usize,
    /// Words held across all slabs, against [`STAGING_MAX_WORDS`].
    total_words: usize,
}

/// Words per staging slab (256 KiB).
const STAGING_SLAB_WORDS: usize = 1 << 16;
/// Most page-locked words a handle's staging may hold (4 MiB).
const STAGING_MAX_WORDS: usize = 1 << 20;

impl StagingPool {
    /// A region of `words` that no other call is handed until [`Self::reset`],
    /// as `(slab, word offset)`. `Ok(None)` when [`STAGING_MAX_WORDS`] is
    /// reached: the caller must drain the device and reset.
    fn reserve(&mut self, ctx: &exec::Context, words: usize) -> Result<Option<(usize, usize)>, String> {
        let take = words.next_multiple_of(STAGING_ALIGN_WORDS).max(STAGING_ALIGN_WORDS);
        loop {
            match self.slabs.get(self.cursor) {
                Some(slab) if slab.words() - self.used >= take => {
                    let at = (self.cursor, self.used);
                    self.used += take;
                    return Ok(Some(at));
                }
                Some(_) => {
                    self.cursor += 1;
                    self.used = 0;
                }
                None => {
                    let cap = take.max(STAGING_SLAB_WORDS);
                    if self.total_words + cap > STAGING_MAX_WORDS {
                        return Ok(None);
                    }
                    self.slabs.push(ctx.pinned(cap)?);
                    self.total_words += cap;
                }
            }
        }
    }

    /// The device has drained, so every region handed out has been read.
    fn reset(&mut self) {
        self.cursor = 0;
        self.used = 0;
    }
}

/// What the handle knows about one registered kernel before anything compiles
/// it.
struct KernelSrc {
    name: String,
    wgsl: String,
    /// `@workgroup_size(x)`, read the way every other backend reads it, so all
    /// four lay out the same dispatch grid for the same `threads`.
    wg: u32,
}

/// Device-side kernel timing: events recorded around each launch while timing
/// is on, resolved into per-kernel totals once the device has drained.
#[derive(Default)]
struct KernelTimes {
    /// kernel name -> (accumulated milliseconds, calls).
    totals: HashMap<String, (f64, u64)>,
    /// Launches whose events have been recorded but not yet read.
    pending: Vec<(String, exec::Event, exec::Event)>,
    /// Events ready for reuse, so profiling a long run does not create and
    /// destroy two driver objects per launch.
    pool: Vec<exec::Event>,
}

/// Events kept for reuse; the rest are destroyed.
const EVENT_POOL_CAP: usize = 256;

#[derive(Default)]
struct Counters {
    submits: AtomicU64,
    dispatches: AtomicU64,
    readbacks: AtomicU64,
    uniform_allocs: AtomicU64,
    writes: AtomicU64,
    host_launches: AtomicU64,
    host_nanos: AtomicU64,
    graph: GraphCounters,
}

/// What submitting cost the **host**, as distinct from what the host asked the
/// device to do.
///
/// [`DeviceStats`] answers the second question only: `dispatches` counts
/// recorded steps, which is a property of the model's graph and does not move
/// however the backend chooses to issue them. The quantity this engine's CUDA
/// work is actually chasing is the first one - per-launch driver cost, which on
/// a decode step of a few thousand dispatches is a large fraction of the step -
/// and no counter in this tree measured it.
///
/// Two fields therefore exist that [`DeviceStats`] has no room for:
///
/// - `host_launches` - how many individual launch calls the host made into the
///   driver. It equals `dispatches` when every step is issued one at a time,
///   and is the number that must *stop growing* for a batched submission
///   mechanism to be doing anything at all.
/// - `host_nanos` - wall-clock spent inside
///   [`backend_api::Backend::submit`], which is host time by construction:
///   `submit` is asynchronous, so it returns without waiting for the device.
///
/// Both are cumulative over the handle's life, so a caller measures an interval
/// by differencing two reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaunchStats {
    /// `submit` calls.
    pub submits: u64,
    /// Recorded steps launched - the same quantity [`DeviceStats::dispatches`]
    /// reports.
    pub dispatches: u64,
    /// Individual launch calls made into the driver.
    pub host_launches: u64,
    /// Cumulative wall-clock nanoseconds spent inside `submit`.
    pub host_nanos: u64,
    /// Submissions recorded into a CUDA graph. One per distinct repeated
    /// shape; a number that keeps climbing says the shape is not actually
    /// repeating, or that something keeps invalidating it.
    pub graph_captures: u64,
    /// Submissions answered by replaying an already captured graph. This is
    /// the count `host_launches` stops growing in exchange for.
    pub graph_replays: u64,
    /// Kernel nodes re-pointed at a new grid inside an already instantiated
    /// graph - a sequence length advancing, in the loop this exists for.
    pub grid_updates: u64,
    /// Kernel nodes given new by-value scalar arguments inside an already
    /// instantiated graph - a native kernel whose scalars (a position, a
    /// scale) change between replays.
    pub graph_scalar_updates: u64,
    /// Replays that had to wait for the device before overwriting their
    /// parameter staging. Zero whenever the caller reads between submissions,
    /// which a decoder does every token.
    pub staging_waits: u64,
    /// Host-to-device parameter copies recorded into captured graphs - nodes
    /// the device runs on every replay. A captured submission carries one for
    /// all of its steps together, however many there are.
    pub graph_param_copies: u64,
}

/// What [`backend_api::Backend::begin_pass`] has collected so far.
///
/// A pass exists because a decode step submits per layer and every layer's
/// submissions name different weights: sixty-four different submissions per
/// token, each too small to be worth a graph of its own and none repeating until
/// the next token (one capture in a whole run, measured, while the trigger
/// still wanted a shape twice in a row). Holding them until the pass ends makes
/// the step ONE submission, which does repeat from token to token.
///
/// The held steps are already [`Resolved`]: each keeps its buffers alive, so a
/// temporary the caller drops mid-pass is not freed (and its address not
/// recycled into a later step of the same pass) before the work has been
/// issued.
#[derive(Default)]
struct PassState {
    /// Nesting depth; the outermost `end_pass` issues.
    depth: u32,
    held: Vec<Resolved>,
    /// Whether a chunk of the open pass has been issued yet - see
    /// [`PASS_FIRST_FLUSH_STEPS`].
    issued: bool,
}

/// A brain compute device driven by the CUDA Driver API.
pub struct CudaBackend {
    /// Shared by every handle made with [`backend_api::Backend::share`]: the
    /// catalogue is immutable after construction.
    kernels: Arc<Vec<KernelSrc>>,
    /// `kind` -> compiled kernel, populated on first dispatch. The lock is
    /// held across the NVRTC compile: a duplicate compile of the same kernel
    /// is pure waste, and first dispatch of a given `kind` happens once.
    ///
    /// Shared with every sibling handle: a module belongs to the primary
    /// context all of them retain, so one compile serves them all.
    compiled: Arc<Mutex<HashMap<usize, Arc<Compiled>>>>,
    /// Kernels a provider registered with
    /// [`backend_api::Backend::register_native`]: its OWN source, not a
    /// translation of anything in the WGSL catalogue.
    ///
    /// They occupy `kind` values from [`Self::native_base`] upward, which is
    /// past the end of `kernels` and therefore can never collide with a
    /// catalogue index (the catalogue's length is fixed at construction, long
    /// before a provider can register anything).
    ///
    /// Unlike the catalogue these compile EAGERLY, inside `register_native`:
    /// a provider registers exactly the kernel it is about to dispatch, and
    /// a compile failure has to be answerable with `None` - "this device
    /// declined it, use the portable path" - rather than surfacing as a
    /// panic three dispatches later, by which time the provider has already
    /// promised to serve the request.
    native: Arc<Mutex<Vec<Arc<Compiled>>>>,
    /// Uniform allocations shared by every step of the same shape - see
    /// [`UniformKey`].
    uniforms: Mutex<HashMap<UniformKey, UniformSlot>>,
    /// The allocator epoch at which [`Self::release_dead_uniforms`] last swept
    /// `uniforms`. A buffer can only die by being freed, which moves the
    /// epoch, so an unchanged epoch means there is nothing new to find.
    uniforms_swept_at: AtomicU64,
    /// Page-locked staging for the unbatched path - see [`StagingPool`].
    staging: Mutex<StagingPool>,
    /// The capture state machine. `None` when this handle will never capture:
    /// either the caller turned it off, or this driver has no graph entry
    /// points, which costs launch batching and nothing else.
    graph: Option<Mutex<GraphCache>>,
    /// Set only while a stream capture is open. `read`/`poll_wait` refuse
    /// while it is: a synchronising call from a capturing thread aborts the
    /// capture, and a capture is open for a few microseconds inside one
    /// `submit`, so anything that sees this flag is on another thread and is
    /// about to corrupt a graph rather than read a buffer.
    capturing: AtomicBool,
    /// The open pass, if any - see [`PassState`].
    pass: Mutex<PassState>,
    caps: DeviceCaps,
    identity: GpuIdentity,
    /// The CUDA ordinal this handle opened, so [`Backend::share`] can open the
    /// same device again.
    ordinal: u32,
    counters: Counters,
    /// Whether launches are bracketed with events (see [`KernelTimes`]). While
    /// it is on, submissions are issued one launch at a time, because a
    /// replayed graph reports one duration for the whole graph.
    timing: AtomicBool,
    kernel_times: Mutex<KernelTimes>,
    /// **Last field on purpose.** Rust drops a struct's fields in declaration
    /// order, and every other field above either holds device memory, a loaded
    /// module or an instantiated graph - all of which are resources OF this
    /// context and must be released before it is. With the context first, its
    /// release ran while those were still live, and tearing down a context out
    /// from under its own resources faults inside the driver rather than
    /// returning an error anything could report.
    ctx: exec::Context,
}

/// Steps an open pass holds before a [`backend_api::Backend::flush`] issues them
/// to the card.
///
/// A pass is one submission, issued when it ends - which left the card idle for
/// the whole host-side build of a decode token (about 2.5 ms of a 16 ms step).
/// A `flush` inside the pass hands what is held to the card without waiting, so
/// the host builds the rest while the first part runs; but each chunk is a graph
/// of its own, and a launch costs the host and the card something, so only once
/// this many steps are held. The cut depends on nothing but how many steps have
/// been held when `flush` is called, so a repeating pass is cut at the same
/// steps every iteration and each chunk repeats and replays.
pub const PASS_FLUSH_STEPS: usize = 256;

/// The same threshold for the FIRST chunk of a pass: the host time spent before
/// the card has anything to do is the part of the build that is not hidden
/// behind device work, so it is the one worth keeping short.
pub const PASS_FIRST_FLUSH_STEPS: usize = 64;

/// Turns submission capture off when set to `0`, `off` or `false`.
const ENV_GRAPHS: &str = "BRAIN_CUDA_GRAPHS";

/// Whether submission capture is on, given the value of [`ENV_GRAPHS`].
fn graphs_enabled(value: Option<&str>) -> bool {
    !matches!(value.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("0" | "off" | "false"))
}

impl CudaBackend {
    /// Open CUDA ordinal `ordinal` and register `kernels`.
    ///
    /// Nothing is compiled here - see the module doc. What this DOES do is
    /// read every kernel's `@workgroup_size` and query the device, so a
    /// capability report exists before the first dispatch.
    pub fn try_new_on_ordinal(kernels: &[(&str, &str)], ordinal: u32) -> Result<CudaBackend, String> {
        let ctx = exec::Context::open(ordinal)?;
        let caps = query_caps(&ctx)?;
        let identity = ctx.device_info().identity();
        let graph = match ctx.graphs() {
            Ok(_) if !graphs_enabled(std::env::var(ENV_GRAPHS).ok().as_deref()) => {
                tracing::info!("backend-cuda: graph capture is switched off by {ENV_GRAPHS}; every dispatch will be launched on its own");
                None
            }
            Ok(_) => Some(Mutex::new(GraphCache::default())),
            Err(e) => {
                tracing::info!(reason = %e, "backend-cuda: this driver cannot capture graphs; every dispatch will be launched on its own");
                None
            }
        };
        let catalogue = kernels
            .iter()
            .map(|(name, src)| KernelSrc {
                name: (*name).to_string(),
                wgsl: (*src).to_string(),
                wg: workgroup_size_of(src),
            })
            .collect();
        Ok(CudaBackend::assemble(ctx, ordinal, caps, identity, graph, Arc::new(catalogue), Arc::default(), Arc::default()))
    }

    /// Everything a handle is made of. A fresh handle gets its own stream,
    /// uniform pool, staging and graph cache; a [`share`](backend_api::Backend::share)d
    /// one passes the parent's catalogue and compiled-kernel caches instead of
    /// new ones.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        ctx: exec::Context,
        ordinal: u32,
        caps: DeviceCaps,
        identity: GpuIdentity,
        graph: Option<Mutex<GraphCache>>,
        kernels: Arc<Vec<KernelSrc>>,
        compiled: Arc<Mutex<HashMap<usize, Arc<Compiled>>>>,
        native: Arc<Mutex<Vec<Arc<Compiled>>>>,
    ) -> CudaBackend {
        CudaBackend {
            kernels,
            ctx,
            compiled,
            native,
            uniforms: Mutex::new(HashMap::new()),
            uniforms_swept_at: AtomicU64::new(0),
            staging: Mutex::new(StagingPool::default()),
            graph,
            capturing: AtomicBool::new(false),
            pass: Mutex::new(PassState::default()),
            caps,
            identity,
            ordinal,
            counters: Counters::default(),
            timing: AtomicBool::new(false),
            kernel_times: Mutex::new(KernelTimes::default()),
        }
    }

    /// Turn submission capture off (or back on) for this handle.
    ///
    /// On by default where the driver supports it. The knob exists because the
    /// two paths must be comparable: the claim "replaying costs the host less"
    /// is only measurable against the same handle type doing the same work
    /// without it, and a mechanism whose benefit cannot be measured cannot be
    /// shown to have regressed either.
    pub fn with_graph_capture(mut self, on: bool) -> CudaBackend {
        // Turning it off discards whatever was captured, which is why the
        // teardown wait belongs here as well as in `Drop`: a captured graph's
        // pinned staging must not be freed while the device might still be
        // reading it.
        if !on && self.graph.is_some() {
            let _ = self.ctx.sync();
        }
        if on {
            if self.graph.is_none() && self.ctx.graphs().is_ok() {
                self.graph = Some(Mutex::new(GraphCache::default()));
            }
        } else {
            self.graph = None;
        }
        self
    }

    /// Open the first CUDA device this driver exposes.
    pub fn try_new(kernels: &[(&str, &str)]) -> Result<CudaBackend, String> {
        CudaBackend::try_new_on_ordinal(kernels, 0)
    }

    /// Open the physical card `want` names, resolved by identity rather than
    /// by either enumeration's position - `CUDA_VISIBLE_DEVICES` renumbers
    /// CUDA's ordinals per process, so a canonical index is not a CUDA
    /// ordinal.
    pub fn try_new_on(kernels: &[(&str, &str)], want: &GpuIdentity) -> Result<CudaBackend, String> {
        let d = crate::driver::driver().map_err(str::to_string)?;
        let ordinal = d
            .devices()?
            .iter()
            .find(|dev| dev.identity().same_device(want))
            .map(|dev| dev.ordinal)
            .ok_or_else(|| format!("no CUDA device matches {} ({:?})", want.name, want.uuid))?;
        CudaBackend::try_new_on_ordinal(kernels, ordinal)
    }

    /// The compute capability this handle's device reported. Queried, never
    /// written down - every compilation and every tier decision keys on it.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.ctx.compute_capability()
    }

    /// How many distinct kernels have actually been compiled so far - the
    /// observable form of "compilation is lazy". A handle that has dispatched
    /// nothing reports 0 however many kernels it registered.
    pub fn compiled_kernel_count(&self) -> usize {
        self.compiled.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    fn kernel(&self, kind: usize) -> &KernelSrc {
        self.kernels.get(kind).unwrap_or_else(|| {
            panic!("backend-cuda: kernel index {kind} is out of range ({} registered)", self.kernels.len())
        })
    }

    /// How many native kernels are loaded on this device - shared by every
    /// sibling handle, so it counts distinct kernels, not registrations.
    pub fn native_kernel_count(&self) -> usize {
        self.native.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// The registered native kernel `id` names on THIS handle, if any.
    fn native_kernel_of(&self, id: backend_api::NativeId) -> Option<Arc<Compiled>> {
        let idx = (id.0 as usize).checked_sub(self.native_base())?;
        self.native.lock().unwrap_or_else(|e| e.into_inner()).get(idx).cloned()
    }

    /// The most shared memory per block a kernel may opt in to on this device,
    /// bytes; 0 where the driver does not report it.
    pub fn native_shared_optin_bytes(&self) -> u32 {
        self.ctx.device_info().shared_mem_optin
    }

    /// The first `kind` value that names a [`backend_api::Backend::register_native`]d
    /// kernel rather than a WGSL catalogue entry.
    fn native_base(&self) -> usize {
        self.kernels.len()
    }

    /// The compiled form of `kind`, compiling it on first use.
    fn compiled_for(&self, kind: usize) -> Arc<Compiled> {
        if kind >= self.native_base() {
            let native = self.native.lock().unwrap_or_else(|e| e.into_inner());
            return native
                .get(kind - self.native_base())
                .unwrap_or_else(|| {
                    panic!(
                        "backend-cuda: native kernel index {kind} is out of range ({} registered on this handle)",
                        native.len()
                    )
                })
                .clone();
        }
        let mut cache = self.compiled.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = cache.get(&kind) {
            return c.clone();
        }
        let k = self.kernel(kind);
        let gen = wgsl_cuda::generate(&k.name, &k.wgsl).unwrap_or_else(|e| {
            panic!(
                "backend-cuda: kernel '{}' has no generated CUDA form: {e}. The CUDA backend \
                 refuses a kernel it cannot translate rather than approximating it or running \
                 it somewhere else.",
                k.name
            )
        });
        // The dispatch grid is laid out from `workgroup_size_of`, the same
        // dumb scan every other backend uses; the launch uses naga's answer.
        // They describe the same attribute, so a disagreement means one of the
        // two parses is wrong and the grid would silently cover the wrong
        // number of invocations.
        assert_eq!(
            gen.block_dim, k.wg,
            "backend-cuda: kernel '{}' has @workgroup_size {} by naga and {} by the text scan \
             every backend lays its grid out with",
            k.name, gen.block_dim, k.wg
        );
        assert!(
            gen.block_dim <= self.caps.max_workgroup_size,
            "backend-cuda: kernel '{}' declares @workgroup_size({}) but {} accepts at most {} \
             threads per block",
            k.name,
            gen.block_dim,
            self.ctx.name(),
            self.caps.max_workgroup_size
        );
        let module = self.ctx.compile(&gen.source, &gen.entry).unwrap_or_else(|e| {
            panic!("backend-cuda: NVRTC rejected the generated source for kernel '{}': {e}", k.name)
        });
        let func = module.function(&gen.entry).unwrap_or_else(|e| {
            panic!("backend-cuda: generated kernel '{}' has no entry point '{}': {e}", k.name, gen.entry)
        });
        let c = Arc::new(Compiled {
            func: func.raw(),
            block_dim: gen.block_dim,
            n_bindings: gen.bindings.len(),
            takes_uniform: gen.uniform_bytes > 0,
            takes_lengths: true,
            shape: exec::Shape::default(),
            scalars: &[],
            name: k.name.clone(),
            native_key: None,
            _module: module,
        });
        tracing::debug!(kernel = %k.name, block_dim = c.block_dim, "backend-cuda compiled a kernel");
        cache.insert(kind, c.clone());
        c
    }

    fn alloc(&self, bytes: u64, what: &str) -> exec::DeviceMem {
        self.ctx
            .alloc(bytes as usize)
            .unwrap_or_else(|e| panic!("backend-cuda: allocating {bytes} bytes for {what} failed: {e}"))
    }

    /// The uniform allocation every step of this shape shares.
    ///
    /// Not one per step, and the reason is the whole basis of graph capture: a
    /// graph node holds a device *address*, so a uniform allocated afresh per
    /// dispatch would give every token a new address and no submission would
    /// ever be replayable. The key is therefore the dispatch's structure - the
    /// kernel, the buffers it binds and the size of its parameter block - with
    /// the parameter VALUES excluded, so that a position advancing by one
    /// reuses the same storage rather than invalidating everything.
    ///
    /// What that buys is paid for in `submit` rather than here: two steps of
    /// the same shape in one submission now share storage, so the parameters
    /// cannot be uploaded when the step is recorded (the second would
    /// overwrite the first before either had run). They are uploaded in stream
    /// order between the two dispatches instead, which is correct for the same
    /// reason the dispatches themselves are ordered.
    fn uniform_for(&self, kind: usize, bufs: &[(Arc<exec::DeviceMem>, u64)], words: usize) -> Arc<exec::DeviceMem> {
        let key = UniformKey {
            kind,
            words,
            bufs: bufs.iter().map(|(m, o)| (Arc::as_ptr(m) as usize, *o)).collect(),
        };
        let mut cache = self.uniforms.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = cache.get(&key) {
            if slot.keep.iter().all(|w| w.strong_count() > 0) {
                return slot.mem.clone();
            }
            // The key's addresses no longer name the buffers they were formed
            // from, so the entry is meaningless rather than stale.
            cache.remove(&key);
        }
        // Entries whose buffers are gone are the only thing that can make this
        // map grow without bound, and they can only be found by looking, so
        // the sweep is here rather than on a timer.
        if cache.len() >= 4096 {
            cache.retain(|_, s| s.keep.iter().all(|w| w.strong_count() > 0));
        }
        self.counters.uniform_allocs.fetch_add(1, Ordering::Relaxed);
        let mem = Arc::new(self.alloc((words * 4).max(4) as u64, "a step's uniform stream"));
        // Zeroed: a kernel that declares a uniform block wider than the
        // parameters a caller supplied would otherwise read whatever the last
        // tenant of this address left, and `cuMemAlloc` promises nothing.
        self.ctx.zero(&mem).unwrap_or_else(|e| panic!("backend-cuda: zeroing a uniform failed: {e}"));
        // A step records the uniform now and may be submitted on another
        // handle, whose stream must not run it ahead of this zero.
        self.fence_out([&mem]);
        cache.insert(
            key,
            UniformSlot { mem: mem.clone(), keep: bufs.iter().map(|(m, _)| Arc::downgrade(m)).collect() },
        );
        mem
    }

    /// Free the uniform blocks of steps whose buffers no longer exist.
    ///
    /// Such an entry can never be matched again (its key names addresses of
    /// freed allocations), but it keeps a device allocation resident. They used
    /// to be found only when the table reached its size cap or the same key was
    /// asked for again, so a handle that outlived its model kept one block per
    /// distinct step shape until it was dropped. Called where the host has just
    /// synchronised, so nothing in flight can still be reading a block.
    fn release_dead_uniforms(&self) {
        let epoch = self.ctx.alloc_epoch();
        if self.uniforms_swept_at.swap(epoch, Ordering::AcqRel) == epoch {
            return;
        }
        let dead: Vec<UniformSlot> = {
            let mut cache = self.uniforms.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<UniformKey> = cache
                .iter()
                .filter(|(_, s)| !s.keep.iter().all(|w| w.strong_count() > 0))
                .map(|(k, _)| k.clone())
                .collect();
            keys.iter().filter_map(|k| cache.remove(k)).collect()
        };
        // Freed outside the lock: `cuMemFree` waits for the device.
        drop(dead);
    }

    fn record(
        &self,
        kind: usize,
        bufs: &[&DeviceBuffer],
        offsets: Option<&[(u64, u64)]>,
        params: &[u32],
        threads: u32,
    ) -> Step {
        let mut lens = Vec::with_capacity(bufs.len());
        let bufs: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (off_words, len_words) = offsets.map_or((0, 0), |o| o[i]);
                let mem = CudaBuf::of(b).mem.clone();
                lens.push(bound_words(&mem, off_words * 4, len_words));
                (mem, off_words * 4)
            })
            .collect();
        let uniform = Some(self.uniform_for(kind, &bufs, params.len()));
        Step::new(CudaStep { kind, threads, uniform, params: params.to_vec(), bufs, lens })
    }

    /// Everything a dispatch needs, resolved before anything is issued.
    ///
    /// Compilation and entry-point lookup happen HERE rather than at launch,
    /// because both may run NVRTC and load a module, and neither is a stream
    /// operation - performing either inside an open capture is the class of
    /// call a capture rejects.
    fn resolve(&self, st: &CudaStep) -> Resolved {
        let c = self.compiled_for(st.kind);
        let name = &c.name;
        assert_eq!(
            c.n_bindings,
            st.bufs.len(),
            "backend-cuda: kernel '{name}' declares {} storage bindings, the dispatch supplies {}",
            c.n_bindings,
            st.bufs.len()
        );
        let mut args: Vec<CuDevicePtr> = Vec::with_capacity(2 * st.bufs.len() + 1);
        let uniform = if c.takes_uniform {
            let u = st.uniform.as_ref().unwrap_or_else(|| {
                panic!("backend-cuda: kernel '{name}' declares a uniform block but the dispatch bound none")
            });
            args.push(u.device_ptr());
            Some(u.clone())
        } else {
            None
        };
        for (mem, off) in &st.bufs {
            args.push(mem.device_ptr() + off);
        }
        if c.takes_lengths {
            // The kernel reads the low 64 bits as `unsigned long long`.
            args.extend(st.lens.iter().copied());
        }
        let scalar_at = args.len();
        args.extend(encode_scalars(c.scalars, &st.params));
        let func = c.func;
        // A catalogue dispatch counts INVOCATIONS and the grid is laid out by
        // dividing them across the kernel's declared work-group size; a
        // native dispatch counts BLOCKS directly (see
        // `backend_api::Backend::step_native`'s own contract - there is no
        // `@workgroup_size` in CUDA C++ for a caller to have divided by).
        // Dividing a block count by `block_dim` a second time would launch
        // `block_dim` times too few blocks and silently leave most of the
        // output unwritten.
        let per_block = if st.kind >= self.native_base() { 1 } else { c.block_dim };
        Resolved {
            func,
            args,
            scalar_at,
            per_block,
            threads: st.threads,
            params: st.params.clone(),
            uniform,
            bufs: st.bufs.clone(),
            lens: st.lens.clone(),
            compiled: c,
        }
    }

    /// Enqueue a copy of `words` into `dst` on the dispatch stream, staged
    /// through the pinned pool.
    ///
    /// Reserving, filling and enqueuing happen under the pool's lock, and
    /// [`Self::drain_staging`] resets under the same lock around its
    /// synchronise. That pairing is what makes reuse safe when two threads
    /// share a handle: a reset can never recycle a region whose copy was
    /// enqueued after the synchronise that was supposed to cover it.
    fn stage_upload(&self, dst: &exec::DeviceMem, words: &[u32]) {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        let mut at = pool
            .reserve(&self.ctx, words.len())
            .unwrap_or_else(|e| panic!("backend-cuda: page-locked staging of {} words: {e}", words.len()));
        if at.is_none() {
            self.ctx.sync().unwrap_or_else(|e| panic!("backend-cuda: device synchronise failed: {e}"));
            pool.reset();
            at = pool
                .reserve(&self.ctx, words.len())
                .unwrap_or_else(|e| panic!("backend-cuda: page-locked staging of {} words: {e}", words.len()));
        }
        let (slab, offset) = at.unwrap_or_else(|| {
            panic!(
                "backend-cuda: a step's {} parameter words exceed the {STAGING_MAX_WORDS}-word staging bound",
                words.len()
            )
        });
        let block = &pool.slabs[slab];
        block.fill_at(offset, words);
        self.ctx
            .upload_async_range(dst, block, offset, words.len())
            .unwrap_or_else(|e| panic!("backend-cuda: uploading a step's uniform failed: {e}"));
    }

    /// Wait for the device, then make the whole staging pool reusable. See
    /// [`Self::stage_upload`] for why the wait is inside the lock.
    fn drain_staging(&self) {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        self.ctx.sync().unwrap_or_else(|e| panic!("backend-cuda: device synchronise failed: {e}"));
        pool.reset();
    }

    /// Issue one resolved dispatch on its own - the unbatched path, and the
    /// answer whenever a submission's shape is new.
    fn issue(&self, r: &Resolved) {
        if let (Some(u), false) = (r.uniform.as_ref(), r.params.is_empty()) {
            // Enqueued on the dispatch stream, NOT performed synchronously.
            // A synchronous copy would run on the legacy stream and therefore
            // drain the device between every pair of dispatches - see
            // `StagingPool`. On the stream it is simply ordered: the previous
            // step's kernel has read this shared uniform before the next step
            // overwrites it.
            self.stage_upload(u, &r.params);
        }
        let (gx, gy) = grid_ws(r.threads, r.per_block);
        let start = self.timing.load(Ordering::Acquire).then(|| self.stamp()).flatten();
        // SAFETY: `r.func` was resolved from `r.compiled`'s module, which `r`
        // holds an `Arc` to, and `r.args` is that entry point's argument list.
        unsafe { self.ctx.launch_shaped(r.func, (gx, gy, 1), (r.compiled.block_dim, 1, 1), &r.compiled.shape, &r.args) }
            .unwrap_or_else(|e| panic!("backend-cuda: launching kernel '{}' failed: {e}", r.compiled.name));
        self.counters.host_launches.fetch_add(1, Ordering::Relaxed);
        crate::live::host_launched(1);
        if let Some(start) = start {
            if let Some(end) = self.stamp() {
                self.kernel_times.lock().unwrap_or_else(|e| e.into_inner()).pending.push((r.compiled.name.clone(), start, end));
            }
        }
    }

    /// Make this handle's stream wait for any OTHER handle's pending work on
    /// `bufs`. Two handles run on two streams and a GPU runs two streams
    /// concurrently, so a buffer one handle is still writing may be read by the
    /// other early unless it is ordered here. Free while only one handle is
    /// open: a stream orders its own work.
    fn fence_in<'a>(&self, bufs: impl IntoIterator<Item = &'a Arc<exec::DeviceMem>>) {
        if !exec::several_contexts_open() {
            return;
        }
        let mut waited: Vec<*const exec::Event> = Vec::new();
        for m in bufs {
            let guard = m.fence.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(f) = guard.as_ref() {
                if f.stream_id != self.ctx.stream_id() && !waited.contains(&Arc::as_ptr(&f.event)) {
                    waited.push(Arc::as_ptr(&f.event));
                    if let Err(e) = self.ctx.wait_event(&f.event) {
                        tracing::warn!(reason = %e, "backend-cuda: cannot order this stream after another handle's work");
                    }
                }
            }
        }
    }

    /// Wrap a freshly allocated and zeroed allocation as a buffer, fenced so
    /// that another handle using it waits for the zeroing, which is queued on
    /// this handle's stream and otherwise lands after that handle's own writes.
    fn wrap_initialised(&self, mem: exec::DeviceMem) -> DeviceBuffer {
        let buf = CudaBuf::wrap(mem);
        self.fence_out([&CudaBuf::of(&buf).mem]);
        buf
    }

    /// Record that this handle's stream has just done work on `bufs`, so any
    /// other handle that uses one of them first waits for it. See
    /// [`Self::fence_in`].
    fn fence_out<'a>(&self, bufs: impl IntoIterator<Item = &'a Arc<exec::DeviceMem>>) {
        if !exec::several_contexts_open() {
            return;
        }
        let event = match self.ctx.ordering_event().and_then(|e| self.ctx.record(&e).map(|()| e)) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                tracing::warn!(reason = %e, "backend-cuda: cannot record an ordering event");
                return;
            }
        };
        for m in bufs {
            *m.fence.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(exec::Fence { stream_id: self.ctx.stream_id(), event: Arc::clone(&event) });
        }
    }

    /// An event recorded on this handle's stream now, from the pool when one is
    /// free. `None` (and a log line) if the driver refuses: timing is a
    /// diagnostic and must never fail a dispatch.
    fn stamp(&self) -> Option<exec::Event> {
        let reused = self.kernel_times.lock().unwrap_or_else(|e| e.into_inner()).pool.pop();
        let ev = match reused {
            Some(e) => e,
            None => self.ctx.event().map_err(|e| tracing::warn!(reason = %e, "backend-cuda: cannot create a timing event")).ok()?,
        };
        self.ctx.record(&ev).map_err(|e| tracing::warn!(reason = %e, "backend-cuda: cannot record a timing event")).ok()?;
        Some(ev)
    }

    /// Fold every recorded launch into the per-kernel totals. The device must
    /// have drained (every caller has just synchronised), so every event is
    /// complete and reading it does not wait.
    fn collect_timings(&self) {
        let mut kt = self.kernel_times.lock().unwrap_or_else(|e| e.into_inner());
        for (name, start, end) in std::mem::take(&mut kt.pending) {
            match self.ctx.elapsed_ms(&start, &end) {
                Ok(ms) => {
                    let entry = kt.totals.entry(name).or_insert((0.0, 0));
                    entry.0 += ms as f64;
                    entry.1 += 1;
                }
                Err(e) => tracing::warn!(reason = %e, "backend-cuda: a kernel timing could not be read"),
            }
            for ev in [start, end] {
                if kt.pool.len() < EVENT_POOL_CAP {
                    kt.pool.push(ev);
                }
            }
        }
    }

    /// The structure of a submission, as [`SubmitSig`] defines it.
    fn signature(&self, clears: &[Arc<exec::DeviceMem>], steps: &[Resolved]) -> SubmitSig {
        SubmitSig {
            clears: clears.iter().map(|m| Arc::as_ptr(m) as usize).collect(),
            nodes: steps
                .iter()
                .map(|r| NodeSig {
                    kernel: Arc::as_ptr(&r.compiled) as usize,
                    uniform: r.uniform.as_ref().map(|u| Arc::as_ptr(u) as usize).unwrap_or(0),
                    words: r.params.len(),
                    bufs: r.bufs.iter().map(|(m, off)| (Arc::as_ptr(m) as usize, *off)).collect(),
                    lens: r.lens.clone(),
                })
                .collect(),
        }
    }

    /// What this handle's submissions have cost the host so far - see
    /// [`LaunchStats`].
    pub fn launch_stats(&self) -> LaunchStats {
        let g = &self.counters.graph;
        LaunchStats {
            submits: self.counters.submits.load(Ordering::Relaxed),
            dispatches: self.counters.dispatches.load(Ordering::Relaxed),
            host_launches: self.counters.host_launches.load(Ordering::Relaxed),
            host_nanos: self.counters.host_nanos.load(Ordering::Relaxed),
            graph_captures: g.captures.load(Ordering::Relaxed),
            graph_replays: g.replays.load(Ordering::Relaxed),
            grid_updates: g.grid_updates.load(Ordering::Relaxed),
            graph_scalar_updates: g.scalar_updates.load(Ordering::Relaxed),
            staging_waits: g.staging_waits.load(Ordering::Relaxed),
            graph_param_copies: g.param_copies.load(Ordering::Relaxed),
        }
    }

    /// Issue whatever an open pass is holding, as one submission, in the order
    /// it was submitted. A no-op outside a pass or when nothing is held.
    fn issue_held(&self) {
        let held = {
            let mut pass = self.pass.lock().unwrap_or_else(|e| e.into_inner());
            let held = std::mem::take(&mut pass.held);
            pass.issued |= !held.is_empty();
            held
        };
        if !held.is_empty() {
            self.issue_all(Vec::new(), held, true);
        }
    }

    /// Issue one whole submission: replay it from a captured graph, capture it
    /// first when its shape has just repeated, or launch each step on its own.
    /// `chunk` is whether it is a piece of a pass (see
    /// [`crate::graph::GraphCache::plan`]).
    fn issue_all(&self, clears: Vec<Arc<exec::DeviceMem>>, resolved: Vec<Resolved>, chunk: bool) {
        // Order this submission after other handles' work on the buffers it
        // names. Before the plan, so a capture does not record a wait on an
        // event from outside it, which a capture cannot represent.
        // The uniform blocks too: a step recorded on another handle brings its
        // zeroed uniform, and the parameters this submission uploads into it
        // must not overtake that handle's kernel still reading the old ones.
        let touched = || clears.iter().chain(resolved.iter().flat_map(|r| r.bufs.iter().map(|(m, _)| m).chain(r.uniform.as_ref())));
        self.fence_in(touched());

        let mut handled = false;
        let mut give_up = false;
        let timing = self.timing.load(Ordering::Acquire);
        if let (Some(cache), false) = (&self.graph, timing) {
            let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
            let sig = self.signature(&clears, &resolved);
            // Read AFTER the signature is built: every allocation the
            // signature names is held by a `Resolved`, so nothing it describes
            // can be freed between the two.
            let epoch = self.ctx.alloc_epoch();
            let plan = cache.plan(&self.ctx, &sig, epoch, chunk);
            let outcome = match plan {
                Plan::Eager => Ok(false),
                Plan::Replay(at) => cache.replay(&self.ctx, at, &resolved, &self.counters.graph).map(|()| true),
                Plan::Capture => {
                    self.capturing.store(true, Ordering::Release);
                    let mut r = cache.capture(
                        &self.ctx,
                        &sig,
                        epoch,
                        &clears,
                        &resolved,
                        &self.counters.graph,
                    );
                    self.capturing.store(false, Ordering::Release);
                    if r.is_ok() {
                        // Recording a launch IS a launch call into the driver;
                        // it just does not execute. Counting it keeps
                        // `host_launches` the honest total of what the host
                        // paid, so a capture reads as a one-off cost rather
                        // than as free.
                        self.counters.host_launches.fetch_add(resolved.len() as u64, Ordering::Relaxed);
                        crate::live::host_launched(resolved.len() as u64);
                        // Index 0 is the capture that just happened - a
                        // capture records the work but does not run it, so the
                        // submission is still owed its execution.
                        r = cache.replay(&self.ctx, 0, &resolved, &self.counters.graph);
                    }
                    r.map(|()| true)
                }
            };
            match outcome {
                Ok(done) => handled = done,
                Err(e) => {
                    // Capture is an optimisation and nothing else, so a
                    // failure costs batching rather than the submission: the
                    // recorded work was swallowed by the capture rather than
                    // executed, so falling through re-issues all of it.
                    tracing::warn!(reason = %e, "backend-cuda: submission capture failed; launching each dispatch");
                    give_up = cache.forget(&self.ctx);
                }
            }
        }

        if give_up {
            tracing::warn!(
                "backend-cuda: submission capture has failed repeatedly on this handle and will not be \
                 attempted again; every dispatch will be launched on its own"
            );
        }
        if !handled {
            for c in &clears {
                self.ctx
                    .zero(c)
                    .unwrap_or_else(|e| panic!("backend-cuda: clearing a buffer failed: {e}"));
            }
            for r in &resolved {
                self.issue(r);
            }
        }
        self.fence_out(touched());
    }

    /// Refuse a synchronising call that arrived while a capture is open.
    ///
    /// A capture is open for a few microseconds inside one `submit`, so an
    /// in-thread violation is structurally impossible - which is exactly why
    /// this check earns its place: what it catches is another thread reaching
    /// the same handle, where the consequence is not a slow read but a capture
    /// aborted mid-recording and a graph that silently never forms.
    fn refuse_during_capture(&self, what: &str) {
        assert!(
            !self.capturing.load(Ordering::Acquire),
            "backend-cuda: {what} was called while this handle was capturing a submission. A \
             synchronising call from inside a capture invalidates it; capture is confined to one \
             `submit`, so this came from another thread sharing the handle."
        );
    }

}

// `Compiled` owns a loaded module and the entry-point handle resolved out of
// it. Both are opaque driver handles that are used under `cuCtxSetCurrent`,
// and the struct is immutable once built.
unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Drop for CudaBackend {
    /// Wait for the device before anything this handle owns is released.
    ///
    /// `submit` does not wait, so a handle can be dropped with work still
    /// running, and among the things dropped are page-locked staging blocks a
    /// captured graph's copy nodes may not have read yet. Freeing host memory
    /// that a transfer is reading is a fault or a corruption, not an error
    /// code - and unlike an instantiated graph (which the driver holds until
    /// its pending launches finish) nothing defers it.
    fn drop(&mut self) {
        if let Err(e) = self.ctx.sync() {
            tracing::warn!(reason = %e, "backend-cuda: synchronising before tearing a handle down failed");
        }
    }
}

/// `params` as bytes. Native endianness on both sides of the copy - host and
/// device are the same byte order on every platform this driver runs on, and
/// a swap here would corrupt every uniform.
fn bytemuck_words(params: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and no invalid bit patterns, so any `[u32]`
    // is a valid `[u8]` of four times the length; the lifetime is tied to the
    // input and the result is read-only.
    unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * 4) }
}

/// Build this device's capability report from what the driver answered.
///
/// Nothing here is a constant about a card. The one judgement call is
/// [`DeviceCaps::max_storage_binding_bytes`], which is deliberately NOT
/// reported - see [`CudaBackend::max_storage_binding_bytes`].
fn query_caps(ctx: &exec::Context) -> Result<DeviceCaps, String> {
    let i = ctx.device_info();
    let mut arch = ArchDesc::default();
    // fp32 is what this device's ALUs are; there is no emulation involved.
    arch.set_tier(DType::F32, TierSupport { level: TierLevel::Native, ..TierSupport::default() });
    // Packed int8 EXECUTES - `wgsl-cuda` emits a written-out four-lane
    // multiply-accumulate for `dot4I8Packed`, valid on every capability -
    // but it is a polyfill, not the card's DP4A instruction, so the honest
    // level is `Emulated`. Reporting `Native` here would tell a selector this
    // backend has dedicated int8 hardware behind that kernel when what it
    // actually has is a loop, which is precisely the conflation `TierLevel`
    // exists to prevent. A hand-written tuned kernel that really does issue
    // the instruction is what raises this, on the capability that was queried.
    for dt in [DType::I8, DType::Q4, DType::Q4K, DType::Q8K] {
        arch.set_tier(dt, TierSupport { level: TierLevel::Emulated, ..TierSupport::default() });
    }
    // bf16 and f16 are STORAGE tiers: the `#w=bf16`/`#w=f16` kernel variants
    // decode packed words to f32 with integer and bitcast arithmetic, which
    // the generator translates, and the arithmetic stays fp32. That is not a
    // claim of fast half-precision compute (`is_fast` needs a measurement), and
    // the generator still refuses `enable f16;`, so no f16 ALU is used.
    for dt in [DType::BF16, DType::F16] {
        arch.set_tier(dt, TierSupport { level: TierLevel::Storage, ..TierSupport::default() });
    }
    // The instruction-set version, straight from the driver. This is the one
    // fact a native kernel's capability floor is resolved against
    // (`kernels_cuda::best_for`), and it is published here rather than being
    // asked for again at each dispatch so that every consumer sees the SAME
    // answer the compiler was given for this device.
    arch.compute_capability = Some(ctx.compute_capability());
    // Native f16 arithmetic is not reported: the generator refuses `enable
    // f16;` outright rather than widening it to fp32, so no f16 ALU runs on
    // this backend - whatever the attached card could do.
    Ok(DeviceCaps {
        class: if i.integrated { DeviceClass::IntegratedGpu } else { DeviceClass::DiscreteGpu },
        compute_units: Some(i.multiprocessors),
        max_workgroup_size: i.max_threads_per_block,
        workgroup_mem_bytes: i.shared_mem_per_block,
        subgroup_size: Some(i.warp_size),
        unified_memory: i.integrated,
        // One work-group is one block and `__syncthreads()` is emitted for
        // `workgroupBarrier()`, so the cooperative kernels execute correctly.
        workgroup_reductions: true,
        // Measured, never queried and never guessed - `gpu_core::roof` fills
        // these in if and when something measures them.
        peak_bandwidth_gbs: None,
        peak_gflops: None,
        // The portable fp8 decode kernels are plain select/bitcast WGSL, the
        // same reasoning as bf16/f16 storage; `numeric_view` does not model it.
        numeric: NumericSupport { fp8_storage: true, ..arch.numeric_view() },
        arch,
    })
}

impl backend_api::Backend for CudaBackend {
    fn storage(&self, n: u64) -> DeviceBuffer {
        let mem = self.alloc(n * 4, "storage");
        // Zeroed, because every other backend hands back zeroed storage and
        // model code adds into freshly allocated accumulators.
        self.ctx.zero(&mem).unwrap_or_else(|e| panic!("backend-cuda: zeroing new storage failed: {e}"));
        self.wrap_initialised(mem)
    }

    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
        let buf = CudaBuf::wrap(self.alloc((data.len() * 4).max(4) as u64, name));
        let mem = &CudaBuf::of(&buf).mem;
        if !data.is_empty() {
            // SAFETY: `f32` has no padding, so any `[f32]` is a valid `[u8]`
            // of four times the length, read-only and for this scope only.
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            self.ctx
                .upload(mem, bytes)
                .unwrap_or_else(|e| panic!("backend-cuda: uploading '{name}' failed: {e}"));
            // Another handle that uses this buffer waits for the copy.
            self.fence_out([mem]);
        } else {
            self.ctx.zero(mem).unwrap_or_else(|e| panic!("backend-cuda: zeroing '{name}' failed: {e}"));
        }
        buf
    }

    fn buffer(&self, label: &str, size: u64, _usage: BufUsage) -> DeviceBuffer {
        // `BufUsage` has no counterpart here: a CUDA allocation is a plain
        // device address usable as any kind of kernel argument, so there is
        // nothing to declare and nothing the driver would validate.
        let mem = self.alloc(size.max(4), label);
        self.ctx.zero(&mem).unwrap_or_else(|e| panic!("backend-cuda: zeroing '{label}' failed: {e}"));
        self.wrap_initialised(mem)
    }

    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
        let mem = self.alloc((len * 4).max(4) as u64, "a dynamic uniform");
        self.ctx.zero(&mem).unwrap_or_else(|e| panic!("backend-cuda: zeroing a uniform failed: {e}"));
        self.wrap_initialised(mem)
    }

    fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
        self.write_at(buf, 0, data);
    }

    fn write_at(&self, buf: &DeviceBuffer, offset_words: u64, data: &[u32]) {
        // A write is ordered after every submission before it, including the
        // ones an open pass is still holding.
        self.issue_held();
        self.counters.writes.fetch_add(1, Ordering::Relaxed);
        let mem = &CudaBuf::of(buf).mem;
        self.fence_in([mem]);
        self.ctx
            .upload_at(mem, (offset_words * 4) as usize, bytemuck_words(data))
            .unwrap_or_else(|e| panic!("backend-cuda: write failed: {e}"));
        self.fence_out([mem]);
    }

    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
        self.record(kind, bufs, None, params, threads)
    }

    fn step_sliced(
        &self,
        kind: usize,
        bufs: &[&DeviceBuffer],
        offsets: &[(u64, u64)],
        params: &[u32],
        threads: u32,
    ) -> Step {
        assert_eq!(
            offsets.len(),
            bufs.len(),
            "backend-cuda: step_sliced got {} offsets for {} buffers",
            offsets.len(),
            bufs.len()
        );
        self.record(kind, bufs, Some(offsets), params, threads)
    }

    /// The caller owns the uniform buffer here, so this backend has no
    /// parameters of its own to copy into it: `params` stays empty and no copy
    /// node is recorded for this step. Whatever the caller writes into that
    /// buffer is what the kernel reads, replay or not.
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
        let uniform = Some(CudaBuf::of(ubuf).mem.clone());
        let bufs: Vec<_> = bufs.iter().map(|b| (CudaBuf::of(b).mem.clone(), 0)).collect();
        let lens = bufs.iter().map(|(m, _)| bound_words(m, 0, 0)).collect();
        Step::new(CudaStep { kind, threads, uniform, params: Vec::new(), bufs, lens })
    }

    /// Compile and keep a provider's OWN CUDA C++ kernel, for the compute
    /// capability THIS handle's device reported.
    ///
    /// `None` rather than a panic on every refusal - a wrong-shaped spec, a
    /// kernel this card cannot host, no NVRTC on the box, or source NVRTC
    /// rejects. Each of those is an ordinary answer to "can you run this?",
    /// and the provider that asked has a portable path to fall back to; a
    /// panic here would turn a capability question into a crash. It is the
    /// one place in this backend where a compile failure is NOT fatal, and
    /// that is precisely because nothing has promised to run it yet.
    ///
    /// The two capability checks are read off [`Self::caps`], which was built
    /// from device queries at construction - a kernel is refused because THIS
    /// card answered that it has fewer threads per block or less shared
    /// memory than the kernel declares, never because of anything written
    /// down about any architecture.
    fn register_native(&self, spec: &backend_api::NativeSpec) -> Option<backend_api::NativeId> {
        let backend_api::NativeSpec::Cuda { src, entry, block_dim, bindings, shared_bytes, launch } = spec else {
            // SPIR-V is a Vulkan pipeline image and `HostFn` is a CPU
            // provider's own function; neither is source this driver can
            // compile. See `NativeSpec`'s own doc.
            tracing::info!("backend-cuda: register_native accepts only NativeSpec::Cuda");
            return None;
        };
        if *block_dim > self.caps.max_workgroup_size {
            tracing::info!(
                block_dim = *block_dim,
                limit = self.caps.max_workgroup_size,
                "backend-cuda: declining a native kernel that wants more threads per block than this device reported"
            );
            return None;
        }
        // Static shared memory is bounded by what every launch may use; the
        // dynamic part only by what the kernel has OPTED in to, up to the
        // device's own queried ceiling. Without the opt-in the driver would
        // refuse the launch, so the kernel is declined here, by name, instead.
        let total_shared = u64::from(*shared_bytes) + u64::from(launch.dynamic_shared_bytes);
        let shared_limit = if launch.shared_opt_in {
            u64::from(self.ctx.device_info().shared_mem_optin.max(self.caps.workgroup_mem_bytes))
        } else {
            u64::from(self.caps.workgroup_mem_bytes)
        };
        if total_shared > shared_limit {
            tracing::info!(
                entry = %entry,
                shared_bytes = total_shared,
                limit = shared_limit,
                opted_in = launch.shared_opt_in,
                "backend-cuda: declining a native kernel that wants more shared memory than it may use on this device"
            );
            return None;
        }
        if let Some(dims) = launch.cluster {
            // 8 is the portable cluster-size limit; a larger one is a
            // non-portable opt-in this registration does not offer.
            let size = dims.iter().map(|&d| u64::from(d)).product::<u64>();
            if dims.contains(&0) || size > 8 || !self.ctx.can_launch_clusters() {
                tracing::info!(entry = %entry, ?dims, "backend-cuda: declining a native kernel whose cluster this device cannot launch");
                return None;
            }
        }
        if !launch.scalars.is_empty() && bindings.contains(&backend_api::BindKind::Uniform) {
            tracing::info!(entry = %entry, "backend-cuda: declining a native kernel that declares both scalar arguments and a uniform block");
            return None;
        }
        let key = NativeKey { src, entry, block_dim: *block_dim, bindings, shared_bytes: *shared_bytes, launch: *launch };
        let id_of = |idx: usize| backend_api::NativeId((self.native_base() + idx) as u32);
        // The registry is shared by every sibling handle, so a registration
        // that always appended would load one more module per handle ever
        // built over a model carrying this kernel, for the life of the device.
        let existing = |native: &[Arc<Compiled>]| native.iter().position(|c| c.native_key == Some(key));
        if let Some(idx) = existing(&self.native.lock().unwrap_or_else(|e| e.into_inner())) {
            return Some(id_of(idx));
        }
        let options = exec::CompileOptions { features: launch.arch, defines: Vec::new() };
        let module = match self.ctx.compile_with(src, entry, &options) {
            Ok(m) => m,
            Err(e) => {
                tracing::info!(entry = %entry, reason = %e, "backend-cuda: this device declined a native kernel");
                return None;
            }
        };
        // Resolve the entry point NOW. A name that is not in the module is a
        // defect in the kernel's own registry metadata, and finding it at
        // registration means the provider gets a `None` it can fall back
        // from instead of a panic at the first dispatch. The handle is kept:
        // it cannot change while the module is loaded, and this struct owns
        // the module.
        let func = match module.function(entry) {
            Ok(f) => f.raw(),
            Err(e) => {
                tracing::info!(entry = %entry, reason = %e, "backend-cuda: a native kernel has no such entry point");
                return None;
            }
        };
        if launch.shared_opt_in && launch.dynamic_shared_bytes > 0 {
            // SAFETY: `func` was just resolved from `module`, still loaded.
            if let Err(e) = unsafe { self.ctx.allow_dynamic_shared(func, launch.dynamic_shared_bytes) } {
                tracing::info!(entry = %entry, reason = %e, "backend-cuda: the driver refused the shared-memory opt-in");
                return None;
            }
        }
        let c = Arc::new(Compiled {
            func,
            block_dim: *block_dim,
            shape: exec::Shape { shared_bytes: launch.dynamic_shared_bytes, cluster: launch.cluster },
            scalars: launch.scalars,
            // STORAGE bindings only - the uniform is counted separately
            // below, exactly as it is for a generated kernel (whose
            // `bindings` list never includes it). Counting it here would
            // make every dispatch look one buffer short.
            n_bindings: bindings.iter().filter(|b| **b != backend_api::BindKind::Uniform).count(),
            // The uniform (if any) is the kernel's first argument, exactly as
            // it is for a generated one.
            takes_uniform: bindings.contains(&backend_api::BindKind::Uniform),
            takes_lengths: false,
            name: format!("native:{entry}"),
            native_key: Some(key),
            _module: module,
        });
        let mut native = self.native.lock().unwrap_or_else(|e| e.into_inner());
        // Another handle may have registered the same kernel while this one
        // compiled; keep theirs and let this module unload with `c`.
        if let Some(idx) = existing(&native) {
            return Some(id_of(idx));
        }
        native.push(c);
        Some(id_of(native.len() - 1))
    }

    /// Record a dispatch of a [`Self::register_native`]d kernel. `threads` is
    /// the BLOCK count (see the trait's own contract).
    fn step_native(
        &self,
        id: backend_api::NativeId,
        bufs: &[&DeviceBuffer],
        params: &[u32],
        threads: u32,
    ) -> Option<Step> {
        self.step_native_sliced(id, bufs, &vec![(0, 0); bufs.len()], params, threads)
    }

    /// [`Self::step_native`] binding a sub-range of each buffer.
    ///
    /// Implemented for real rather than left at the trait's decline-on-any-
    /// offset default: a kernel argument here IS a bare device address, so a
    /// range is expressed by the address handed in and costs nothing - the
    /// same mechanism `step_sliced` already uses for a catalogue kernel.
    fn step_native_sliced(
        &self,
        id: backend_api::NativeId,
        bufs: &[&DeviceBuffer],
        offsets: &[(u64, u64)],
        params: &[u32],
        threads: u32,
    ) -> Option<Step> {
        let kind = id.0 as usize;
        let c = self.native_kernel_of(id);
        if c.is_none() || offsets.len() != bufs.len() {
            // An id from a different handle (or a different backend) names
            // nothing here. Answering `None` lets the provider fall back;
            // launching whatever happened to be at that index would run the
            // wrong kernel.
            return None;
        }
        let c = c?;
        // The parameter words are the kernel's scalar arguments, so the count
        // is part of its signature. A wrong count would read past the slice or
        // hand the kernel a half-formed 64-bit value.
        let want_words: usize = c.scalars.iter().map(|k| k.words()).sum();
        if !c.scalars.is_empty() && params.len() != want_words {
            tracing::info!(kernel = %c.name, got = params.len(), want = want_words, "backend-cuda: declining a native step with the wrong number of scalar words");
            return None;
        }
        // A cluster launch needs a grid of whole clusters; padding it would run
        // blocks the caller never asked for.
        if let Some(dims) = c.shape.cluster {
            let size: u32 = dims.iter().product();
            if threads % size != 0 {
                tracing::info!(kernel = %c.name, blocks = threads, cluster = size, "backend-cuda: declining a native step whose grid is not a whole number of clusters");
                return None;
            }
        }
        Some(self.record(kind, bufs, Some(offsets), params, threads))
    }

    fn native_max_active_blocks(&self, id: backend_api::NativeId) -> Option<u32> {
        let c = self.native_kernel_of(id)?;
        // SAFETY: `c.func` is an entry point of the module `c` keeps loaded.
        match unsafe { self.ctx.max_active_blocks(c.func, c.block_dim, c.shape.shared_bytes) } {
            Ok(n) => Some(n),
            Err(e) => {
                tracing::info!(kernel = %c.name, reason = %e, "backend-cuda: the driver would not answer an occupancy query");
                None
            }
        }
    }

    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
        // Wall-clock across the whole call, because that is what the host pays.
        // `submit` never waits for the device (see the module doc on
        // ordering), so this is host time by construction, not a device
        // measurement smuggled in under a host name.
        let t0 = std::time::Instant::now();
        self.counters.submits.fetch_add(1, Ordering::Relaxed);
        let clears: Vec<Arc<exec::DeviceMem>> = clears.iter().map(|c| CudaBuf::of(c).mem.clone()).collect();
        // Resolved up front, before anything is issued and before any capture
        // opens - compiling a kernel is not a stream operation.
        let resolved: Vec<Resolved> = steps.iter().map(|s| self.resolve(s.downcast_ref::<CudaStep>())).collect();
        self.counters.dispatches.fetch_add(resolved.len() as u64, Ordering::Relaxed);
        {
            let mut pass = self.pass.lock().unwrap_or_else(|e| e.into_inner());
            if pass.depth > 0 && clears.is_empty() {
                pass.held.extend(resolved);
                self.counters.host_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return;
            }
        }
        // A clear is a node ordered before this submission's steps and after
        // everything submitted earlier; hoisting it above held steps would
        // zero a buffer they still have to write, so the held run goes first.
        self.issue_held();
        self.issue_all(clears, resolved, false);
        self.counters.host_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Inside a pass, issue what is held so far once [`PASS_FLUSH_STEPS`] steps
    /// are; otherwise a no-op, as it is for every backend that issues each
    /// submission as it is made.
    fn flush(&self) {
        let due = {
            let pass = self.pass.lock().unwrap_or_else(|e| e.into_inner());
            let at = if pass.issued { PASS_FLUSH_STEPS } else { PASS_FIRST_FLUSH_STEPS };
            pass.depth > 0 && pass.held.len() >= at
        };
        if due {
            let t0 = std::time::Instant::now();
            self.issue_held();
            self.counters.host_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    fn begin_pass(&self) {
        let mut pass = self.pass.lock().unwrap_or_else(|e| e.into_inner());
        if pass.depth == 0 {
            pass.issued = false;
        }
        pass.depth += 1;
    }

    fn end_pass(&self) {
        let outermost = {
            let mut pass = self.pass.lock().unwrap_or_else(|e| e.into_inner());
            assert!(pass.depth > 0, "backend-cuda: end_pass without a matching begin_pass");
            pass.depth -= 1;
            pass.depth == 0
        };
        if outermost {
            let t0 = std::time::Instant::now();
            self.issue_held();
            self.counters.host_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.issue_held();
        self.refuse_during_capture("read");
        self.counters.readbacks.fetch_add(1, Ordering::Relaxed);
        // A launch reports only the errors it can see BEFORE running, so this
        // synchronise is where a faulting kernel is actually reported - and it
        // must happen before the copy, not as part of it.
        let mem = &CudaBuf::of(buf).mem;
        self.fence_in([mem]);
        self.poll_wait();
        let mut out = vec![0f32; n];
        // SAFETY: `out` owns `n * 4` writable bytes with no padding, and the
        // download bounds-checks against the allocation's own length.
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
        self.ctx
            .download(mem, bytes)
            .unwrap_or_else(|e| panic!("backend-cuda: read-back of {n} f32 failed: {e}"));
        out
    }

    /// Block until the device has finished everything submitted so far.
    ///
    /// This is also the point the capture machinery learns the device has
    /// drained, which is what lets a replay overwrite its parameter staging
    /// without waiting: a decoder reads its logits between submissions, so the
    /// wait has already happened by the time the next one is built.
    fn poll_wait(&self) {
        self.issue_held();
        self.refuse_during_capture("poll_wait");
        self.drain_staging();
        self.release_dead_uniforms();
        if self.timing.load(Ordering::Acquire) {
            self.collect_timings();
        }
        if let Some(cache) = &self.graph {
            cache.lock().unwrap_or_else(|e| e.into_inner()).drained(self.ctx.alloc_epoch());
        }
    }

    /// Device-side per-kernel timing, from events around each launch. See
    /// [`KernelTimes`]. While on, submissions are issued launch by launch.
    fn set_kernel_timing(&self, on: bool) -> bool {
        // Held steps were resolved under the old setting; issue them under it.
        self.issue_held();
        if !on && self.timing.swap(false, Ordering::AcqRel) {
            // Launches already stamped still hold events: drain and read them
            // so nothing is lost or left pending behind a switched-off timer.
            let _ = self.ctx.sync();
            self.collect_timings();
        }
        if on {
            self.timing.store(true, Ordering::Release);
        }
        on
    }

    /// `(kernel name, milliseconds of device time, calls)` since the last
    /// reset, sorted by name. Waits for the device, because a launch's events
    /// only mean something once it has run.
    fn kernel_times(&self) -> Option<Vec<(String, f64, u64)>> {
        self.issue_held();
        let _ = self.ctx.sync();
        self.collect_timings();
        let kt = self.kernel_times.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<(String, f64, u64)> = kt.totals.iter().map(|(n, (ms, calls))| (n.clone(), *ms, *calls)).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Some(rows)
    }

    fn reset_kernel_times(&self) {
        let _ = self.ctx.sync();
        self.collect_timings();
        self.kernel_times.lock().unwrap_or_else(|e| e.into_inner()).totals.clear();
    }

    fn hold_freed_blocks(&self, on: bool) {
        self.ctx.hold_freed_blocks(on);
    }

    fn kind(&self) -> &'static str {
        "cuda"
    }

    /// Deliberately the portable ~2 GiB, NOT this card's real VRAM.
    ///
    /// This number is read as a *tile-budget divisor* by chunked model code
    /// (`model::block::tile_budget_words_for` and its siblings in `wan` and
    /// `s3dit`), which sizes a working slab as a fraction of it. Answering
    /// honestly with a large card's memory therefore does not unlock a bigger
    /// binding - it makes those pipelines size slabs the device cannot
    /// allocate. The honest ceiling this backend CAN report without that
    /// consequence is [`Self::max_buffer_bytes`], which is a different
    /// question (the largest single allocation) and has no divisor semantics
    /// attached to it.
    fn max_storage_binding_bytes(&self) -> u64 {
        2 * 1024 * 1024 * 1024 - 1
    }

    /// The largest single allocation, from `cuMemGetInfo` at construction.
    ///
    /// Free memory rather than total: an allocation larger than what is free
    /// fails, whoever else is resident. It is a snapshot - the trait documents
    /// `caps`-adjacent reads as cached, and re-querying per call would make a
    /// sharding decision depend on another process's timing.
    fn max_buffer_bytes(&self) -> u64 {
        self.ctx.mem_info().map(|(free, _)| free).unwrap_or(0)
    }

    /// A second handle onto this device: the same primary context, so the same
    /// address space and the same loaded modules, with its own stream, uniform
    /// pool, staging and graph cache. Cheap by design: nothing is recompiled
    /// and no device memory moves.
    fn share(&self) -> Option<Box<dyn backend_api::Backend>> {
        let ctx = exec::Context::open(self.ordinal)
            .map_err(|e| tracing::warn!(reason = %e, "backend-cuda: cannot open a second handle on this device"))
            .ok()?;
        let graph = self.graph.as_ref().map(|_| Mutex::new(GraphCache::default()));
        Some(Box::new(CudaBackend::assemble(
            ctx,
            self.ordinal,
            self.caps.clone(),
            self.identity.clone(),
            graph,
            Arc::clone(&self.kernels),
            Arc::clone(&self.compiled),
            Arc::clone(&self.native),
        )))
    }

    fn caps(&self) -> DeviceCaps {
        self.caps.clone()
    }

    fn identity(&self) -> Option<GpuIdentity> {
        Some(self.identity.clone())
    }

    fn stats(&self) -> Option<DeviceStats> {
        Some(DeviceStats {
            submits: self.counters.submits.load(Ordering::Relaxed),
            dispatches: self.counters.dispatches.load(Ordering::Relaxed),
            readbacks: self.counters.readbacks.load(Ordering::Relaxed),
            // Nothing on this API corresponds to a bind group: a launch takes
            // its pointers directly. 0 is the honest answer, not a gap.
            bind_groups: 0,
            uniform_allocs: self.counters.uniform_allocs.load(Ordering::Relaxed),
            writes: self.counters.writes.load(Ordering::Relaxed),
        })
    }

    fn queue_submits(&self) -> u64 {
        self.counters.submits.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile-budget divisor rule, pinned where it is easy to "fix" by
    /// reporting the real number: `max_storage_binding_bytes` must stay the
    /// portable ceiling, and it must be strictly below `max_buffer_bytes` on
    /// any device with more than 2 GiB free, because they answer different
    /// questions. Skipped where there is no device.
    #[test]
    fn the_binding_ceiling_is_the_portable_one_and_not_the_card_s_memory() {
        use backend_api::Backend as _;
        let Ok(b) = CudaBackend::try_new(&[]) else { return };
        assert_eq!(b.max_storage_binding_bytes(), 2 * 1024 * 1024 * 1024 - 1);
        let (free, total) = b.ctx.mem_info().expect("cuMemGetInfo");
        assert!(total > 0, "a device reporting no memory at all");
        // Free memory moves between the two queries whenever anything else on
        // the card allocates (another test, another process), so equality
        // would be a race; what the number must be is a free-memory reading.
        let _ = free;
        assert!(b.max_buffer_bytes() > 0 && b.max_buffer_bytes() <= total, "max_buffer_bytes must be a free-memory reading of this device");
    }

    /// While one handle is capturing a graph, another handle on the same card
    /// (on another thread) must still be able to transfer, clear and
    /// synchronise. Waiting on the whole shared primary context, or issuing
    /// legacy-stream copies from a blocking stream's domain, made every handle
    /// in the process fail whenever any one of them was mid-capture.
    #[test]
    fn a_handle_is_not_blocked_by_another_handle_capturing_a_graph() {
        let (Ok(a), Ok(b)) = (CudaBackend::try_new(&[]), CudaBackend::try_new(&[])) else {
            brain_testutil::skip_unavailable("no usable CUDA backend");
            return;
        };
        // Capture is thread-local, so the capturing thread may not synchronise
        // anything; the other handle's thread is the one that must still work.
        let capture = a.ctx.begin_capture().expect("begin capture");
        // The same goes for every other way a handle talks to its card: a
        // transfer or a clear from a thread that is not capturing.
        let result = std::thread::scope(|s| {
            s.spawn(|| -> Result<(), String> {
                let mem = b.ctx.alloc(64)?;
                b.ctx.zero(&mem)?;
                b.ctx.upload(&mem, &[7u8; 64])?;
                let mut back = [0u8; 64];
                b.ctx.download(&mem, &mut back)?;
                assert_eq!(back, [7u8; 64]);
                b.ctx.sync()
            })
            .join()
            .expect("transferring thread")
        });
        drop(capture);
        result.expect("transfers and synchronise on a handle that is not capturing");
    }

    /// The tiers CUDA reports are what weights land on: `Weight::upload` picks
    /// the bf16/f16 storage layout, the packed-int layouts and the fp8 decode
    /// kernels only on a device that says it can run them. All of those kernels
    /// are plain integer and bitcast WGSL that the generator translates, so
    /// this backend reports the same tiers the other GPU backends do for the
    /// same kernels. Reporting them `Absent` made every weight fall back to f32.
    #[test]
    fn storage_and_packed_tiers_match_what_the_translated_kernels_can_run() {
        use backend_api::arch::TierLevel;
        let Ok(b) = CudaBackend::try_new(&[]) else {
            brain_testutil::skip_unavailable("no usable CUDA backend");
            return;
        };
        let caps = backend_api::Backend::caps(&b);
        for dt in [DType::BF16, DType::F16] {
            assert_eq!(caps.arch.tier(dt).level, TierLevel::Storage, "{dt:?}");
        }
        for dt in [DType::I8, DType::Q4, DType::Q4K, DType::Q8K] {
            assert_eq!(caps.arch.tier(dt).level, TierLevel::Emulated, "{dt:?}");
        }
        assert!(caps.numeric.bf16_storage && caps.numeric.f16_storage && caps.numeric.fp8_storage);
        assert!(!caps.numeric.f16 && !caps.numeric.bf16, "storage is not fast arithmetic");
    }

    /// `BRAIN_CUDA_GRAPHS` is the way to bisect a suspected capture bug on a
    /// running system without a rebuild: `0`, `off` or `false` turns submission
    /// capture off; anything else, or nothing, leaves it on.
    #[test]
    fn graph_capture_can_be_switched_off_from_the_environment() {
        for off in ["0", "off", "false", "OFF", " False "] {
            assert!(!graphs_enabled(Some(off)), "{off:?}");
        }
        for on in [None, Some(""), Some("1"), Some("on"), Some("yes")] {
            assert!(graphs_enabled(on), "{on:?}");
        }
    }

    /// Registering a catalogue compiles nothing. The whole point of the lazy
    /// path is that a handle is cheap to build; a regression to eager
    /// compilation is invisible except as a slow start-up.
    #[test]
    fn construction_compiles_no_kernel() {
        let Ok(b) = CudaBackend::try_new(&[("add2", kernels::ADD2), ("mul", kernels::MUL)]) else {
            return;
        };
        assert_eq!(b.compiled_kernel_count(), 0, "a kernel was compiled before any dispatch asked for one");
    }

    /// Capabilities come from the driver, not from this file. Asserted as
    /// *relations* (positive, a power of two, consistent with each other)
    /// rather than as values, so this test is correct on any card - including
    /// ones with tensor cores that nothing here can be run against.
    #[test]
    fn capabilities_are_queried_and_internally_consistent() {
        use backend_api::Backend as _;
        let Ok(b) = CudaBackend::try_new(&[]) else { return };
        let c = b.caps();
        assert!(c.max_workgroup_size >= 64, "max threads per block {}", c.max_workgroup_size);
        assert!(c.workgroup_mem_bytes >= 16 * 1024, "shared memory per block {}", c.workgroup_mem_bytes);
        let sg = c.subgroup_size.expect("the warp size is queryable on this API");
        assert!(sg.is_power_of_two() && sg > 0, "warp size {sg}");
        assert!(c.compute_units.unwrap_or(0) > 0, "SM count");
        assert!(c.workgroup_reductions, "one work-group is one block; barriers execute");
        assert_eq!(c.peak_gflops, None, "a roofline is measured, never reported from a query");
        let (major, _) = b.compute_capability();
        assert!(major > 0, "the driver reported no compute capability");
    }

    /// Whatever a handle creates and initialises on its own stream (zeroed
    /// storage, a step's zeroed uniform block) is used by other handles on the
    /// same card - a model holds two, and steps recorded on one are submitted
    /// on the other. The initialising work is queued on the creator's stream,
    /// so unless it is fenced the other stream runs ahead of it and the zero
    /// lands afterwards, over data the other handle has already written.
    #[test]
    fn what_a_handle_creates_carries_a_fence_for_other_handles() {
        use backend_api::Backend as _;
        let (Ok(a), Ok(_b)) = (CudaBackend::try_new(&[]), CudaBackend::try_new(&[])) else { return };
        let fenced_by_a = |buf: &DeviceBuffer| {
            let fence = CudaBuf::of(buf).mem.fence.lock().unwrap();
            fence.as_ref().is_some_and(|f| f.stream_id == a.ctx.stream_id())
        };
        assert!(fenced_by_a(&a.storage(4)), "zeroed storage");
        assert!(fenced_by_a(&a.buffer("b", 16, BufUsage::STORAGE)), "zeroed buffer");
        assert!(fenced_by_a(&a.uniform_dynamic(4)), "zeroed dynamic uniform");

        let uniform = a.uniform_for(0, &[], 2);
        assert!(
            uniform.fence.lock().unwrap().as_ref().is_some_and(|f| f.stream_id == a.ctx.stream_id()),
            "a step's zeroed uniform block"
        );
    }
}
