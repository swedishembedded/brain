// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`Executor`] — brain's general model-execution layer. Every path (the CLI,
//! the event runtime, the D-Bus surface) submits [`Job`]s here instead of calling
//! models directly, so scheduling, residency, and batching are shared and uniform.
//!
//! Design: a **dispatcher** thread owns the [`ResidencyManager`] and the pending
//! queue (so the manager needs no lock), and **per-device lanes** run the actual
//! inference. Each round the dispatcher drains new jobs + completions, then — for
//! every device NOT currently busy — asks the [`crate::scheduler`] policy for the
//! best runnable group (balancing batch size against queue age), claims it (promote
//! / evict via the manager, pinned so it can't be swapped mid-run), and hands the
//! hot instance to that device's lane. Lanes run concurrently, so models on
//! different GPUs (or the CPU) execute in parallel; jobs contending for one device
//! serialize on its lane (the GPU is the bottleneck anyway). Replies and progress go
//! back through each job's callbacks — no async runtime dependency here.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::manager::{ClaimError, Claimed, ClaimedMulti, InstanceHandle};
use crate::scheduler::{choose_next, Group, Policy};
use crate::{Device, InstanceKey, MultiDeviceResidentModel, ResidencyManager, ResidentModel};

/// One unit of work. `on_progress`/`reply`/`on_admit` are callbacks (no async
/// dependency here). Build with [`Job::new`] + the `.on_*`/`.reply` setters so new
/// optional signals don't churn every construction site.
pub struct Job {
    pub model: String,
    pub action: String,
    pub inv: Invocation,
    pub on_progress: Box<dyn FnMut(Progress) + Send>,
    pub reply: Box<dyn FnOnce(ActionResult) + Send>,
    /// Fired exactly once, at the moment the dispatcher CLAIMS this job onto a lane
    /// (work is about to start) — the admission signal an HTTP layer gates on.
    pub on_admit: Option<Box<dyn FnOnce() + Send>>,
}

impl Job {
    /// A job with no-op callbacks; attach them with `.on_progress`/`.reply`/`.on_admit`.
    pub fn new(model: impl Into<String>, action: impl Into<String>, inv: Invocation) -> Job {
        Job { model: model.into(), action: action.into(), inv, on_progress: Box::new(|_| {}), reply: Box::new(|_| {}), on_admit: None }
    }
    pub fn on_progress(mut self, f: impl FnMut(Progress) + Send + 'static) -> Job {
        self.on_progress = Box::new(f);
        self
    }
    pub fn reply(mut self, f: impl FnOnce(ActionResult) + Send + 'static) -> Job {
        self.reply = Box::new(f);
        self
    }
    pub fn on_admit(mut self, f: impl FnOnce() + Send + 'static) -> Job {
        self.on_admit = Some(Box::new(f));
        self
    }
}

struct Pending {
    /// Monotonic id assigned at submit, stable for the job's whole life (queued →
    /// running). Surfaced by [`Executor::in_flight`] so a live view can track a job.
    id: u64,
    model: String,
    action: String,
    inv: Invocation,
    key: InstanceKey,
    enqueued: Instant,
    on_progress: Box<dyn FnMut(Progress) + Send>,
    reply: Box<dyn FnOnce(ActionResult) + Send>,
    on_admit: Option<Box<dyn FnOnce() + Send>>,
}

/// One job the executor currently holds — either still in the pending `queue`
/// (`phase == "queued"`), claimed onto a device lane but still deferred-
/// activating/promoting (`phase == "building"`), or claimed AND running on an
/// already-adopted instance (`phase == "running"`). A live snapshot from
/// [`Executor::in_flight`]; `id` is the stable submit-order id and `since_ms`
/// is the elapsed time since the job was enqueued.
#[derive(Clone, Debug)]
pub struct InFlightJob {
    pub id: u64,
    pub model: String,
    pub action: String,
    /// `"queued"`, `"building"`, or `"running"` — see this struct's doc.
    /// Group-granular — a whole same-key group is admitted together, so all
    /// of its jobs flip together; `"building"` -> `"running"` flips the
    /// instant [`Msg::Built`] arrives (activate/promote finished), which can
    /// be seconds to well over a minute after admission for a real cold model.
    pub phase: String,
    pub since_ms: u64,
}

/// A job the dispatcher has handed to a lane and is tracking as running. Retained
/// until the lane's `Done` for its group arrives. Mirrors the fields
/// [`InFlightJob`] needs so the pending `Pending` can be dropped into the lane.
struct RunningJob {
    id: u64,
    model: String,
    action: String,
    key: InstanceKey,
    enqueued: Instant,
    /// True from the moment this group is claimed until its [`Msg::Built`]
    /// arrives (deferred activate/promote still in progress on the lane) —
    /// see [`InFlightJob::phase`].
    building: bool,
}

/// Live scheduler counters — proof that batching and eviction happen, and the
/// numbers to profile with. Cumulative unless noted.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub builds: u64,
    pub evictions: u64,
    pub batches: u64,
    pub jobs: u64,
    pub max_batch: usize,
    pub resident: usize,
    /// Resident MULTI-DEVICE instances (parallel to `resident`, which only
    /// counts single-device ones — an instance is never counted in both).
    pub resident_multi: usize,
    pub queue_peak: usize,
    /// Deepest observed number of DEVICES occupied at once — for a pure
    /// single-device workload this is identical to "lanes running at once"
    /// (the reading this field used to have exclusively); a multi-device
    /// group occupies every device it spans, so it can raise this count while
    /// only one lane is actually executing.
    pub max_parallel: usize,
    /// Cumulative count of jobs admitted onto a lane (claimed a device, `on_admit`
    /// fired) — as distinct from `jobs`, which only counts a job once ITS group's
    /// `Done` arrives. `admitted` moves the instant a job starts running;
    /// `jobs`/`batches` are the batching-shape counters `Done` already needed.
    pub admitted: u64,
    /// LIVE queued-job count (unlike `queue_peak`, a high-water mark that never
    /// resets) — the number a dashboard/braintop panel actually wants to watch
    /// change in real time. `InFlightJob::since_ms` is per-request; this is the
    /// aggregate depth.
    pub queue_depth: usize,
    /// Model-specific observability metrics (`Instance::metrics`), refreshed by
    /// the dispatcher after every `assign()` pass. Empty for any instance that
    /// doesn't override `metrics()` — most models.
    pub metrics: HashMap<InstanceKey, Vec<(String, serde_json::Value)>>,
}

/// A message to the dispatcher: a new job, a lane adopting a freshly built
/// instance, or a lane finishing a group (`failed` = the deferred activate
/// errored OR the group panicked mid-run; the lane already replied to the
/// jobs, the manager must unwind the claim).
enum Msg {
    Submit(Box<Job>),
    /// Register a newly-available model with the manager, so a `Submit`
    /// enqueued after this one (FIFO on this same channel) can find it.
    /// [`Executor::register`] updates the public manifest snapshot
    /// synchronously before sending this, so `manifests()` reflects the
    /// registration immediately even though scheduling eligibility lands
    /// here, one hop later, on the dispatcher thread.
    Register(Arc<dyn ResidentModel>),
    /// [`Msg::Register`]'s multi-device sibling — see
    /// [`Executor::register_multi`].
    RegisterMulti(Arc<dyn MultiDeviceResidentModel>),
    Built { key: InstanceKey, handle: InstanceHandle },
    /// [`Msg::Built`]'s multi-device sibling. A SEPARATE variant (rather than
    /// reusing `Built`) purely so `ResidencyManager::adopt_multi` runs instead
    /// of `adopt` — the audit log then says "built ... (multi-device)", not a
    /// false single-device event.
    BuiltMulti { key: InstanceKey, handle: InstanceHandle },
    Done { key: InstanceKey, device: Device, batch: usize, failed: bool },
    /// [`Msg::Done`]'s multi-device sibling — `devices` is every device the
    /// group occupied (in the same order `estimate_multi` named them), all of
    /// which must be freed from `busy` here.
    DoneMulti { key: InstanceKey, devices: Vec<Device>, batch: usize, failed: bool },
    /// A stats query: the dispatcher (the sole owner of the [`ResidencyManager`])
    /// replies with a residency + budget snapshot. Mirrors how [`Stats`] is
    /// exposed, but read straight from the manager rather than the counters.
    Report(Sender<crate::ResidencyReport>),
    /// An in-flight query: the dispatcher replies with one [`InFlightJob`] per job
    /// currently queued OR running. Handled like [`Msg::Report`] — read from the
    /// dispatcher's own live queue + running set, so it is scheduling-consistent.
    InFlight(Sender<Vec<InFlightJob>>),
    /// Demote a resident multi-device instance — see [`Executor::evict_multi`].
    EvictMulti { key: InstanceKey, reply: Sender<bool> },
    /// Stop the dispatcher - see [`Executor::shutdown`]'s doc for why this
    /// has to be an explicit message rather than relying on `rx.recv()`
    /// erroring once every sender is dropped: every lane thread holds its
    /// OWN clone of this same channel's sender (`done`, used to send
    /// [`Msg::Done`]/[`Msg::DoneMulti`] back), so dropping only the
    /// [`Executor`] handle's clone never actually closes the channel - the
    /// dispatcher and every lane would wait on each other forever.
    Shutdown,
}

/// What a [`RunReq`] runs — either a single-device claim (today's shape,
/// unchanged) or a multi-device one spanning several devices at once. Kept as
/// an enum (not two `Option` fields) so a `RunReq` can never claim to be both
/// or neither.
enum RunTarget {
    Single { work: Claimed, device: Device },
    Multi { work: ClaimedMulti, devices: Vec<Device> },
}

/// A group of same-key jobs handed to a device lane to run. `target` is
/// either a hot handle or a deferred build the LANE performs — activation
/// (weight load, NPU graph compile) can take seconds or hang, and on the
/// dispatcher thread that froze every model on the server; on the lane it can
/// only stall its own device (or, for a multi-device target, its own HOME
/// lane — see `assign`'s doc on how that is chosen).
struct RunReq {
    target: RunTarget,
    action: String,
    jobs: Vec<Pending>,
    key: InstanceKey,
}

/// Cheap-to-clone submission handle (many front-ends can submit concurrently).
#[derive(Clone)]
pub struct Executor {
    tx: Sender<Msg>,
    manifests: Arc<RwLock<Arc<Vec<Manifest>>>>,
    stats: Arc<Mutex<Stats>>,
    /// The dispatcher thread's handle plus one per lane, spawned in
    /// [`Self::start`] - see [`Self::shutdown`]'s doc for why a real,
    /// deterministic join (not a fixed sleep) needs these kept.
    join_handles: Arc<Mutex<Option<Vec<std::thread::JoinHandle<()>>>>>,
}

impl Executor {
    /// Build over a set of resident models + a policy, and start the dispatcher +
    /// one lane thread per device (GPUs + CPU).
    pub fn start(models: Vec<Arc<dyn ResidentModel>>, budgets: crate::budget::Budgets, policy: Policy) -> Executor {
        let manifests: Vec<Manifest> = models.iter().map(|m| m.manifest()).collect();
        let devices: Vec<Device> = budgets.devices().collect();
        let mut mgr = ResidencyManager::new(budgets);
        for m in models {
            crate::log::info(&format!("model registered: {}", m.manifest().model));
            mgr.register(m);
        }
        let stats = Arc::new(Mutex::new(Stats::default()));
        let (tx, rx) = channel::<Msg>();

        // One lane per device; each returns completions to the dispatcher via `tx`.
        let mut lanes: HashMap<Device, Sender<RunReq>> = HashMap::new();
        let mut join_handles = Vec::new();
        for d in devices {
            let (ltx, lrx) = channel::<RunReq>();
            let done = tx.clone();
            let h = std::thread::Builder::new()
                .name(format!("brain-lane-{d:?}"))
                .spawn(move || lane_loop(lrx, done))
                .expect("spawn lane");
            join_handles.push(h);
            lanes.insert(d, ltx);
        }

        let disp_stats = stats.clone();
        let h = std::thread::Builder::new()
            .name("brain-dispatcher".into())
            .spawn(move || dispatch_loop(rx, mgr, policy, lanes, disp_stats))
            .expect("spawn dispatcher");
        join_handles.push(h);

        Executor {
            tx,
            manifests: Arc::new(RwLock::new(Arc::new(manifests))),
            stats,
            join_handles: Arc::new(Mutex::new(Some(join_handles))),
        }
    }

    /// Block until every background thread this `Executor` started (the
    /// dispatcher plus one lane per device) has ACTUALLY finished - not a
    /// fixed sleep, a real join.
    ///
    /// This exists because plain `drop(exec)` starts a teardown cascade
    /// (dispatch thread exits -> its `lanes` map drops -> each lane's
    /// channel closes -> lane thread notices and exits, dropping whatever
    /// `Gpu`/`Instance` it was holding) with no built-in signal for "now
    /// it's actually done" - a caller that tears down a process (or a test)
    /// right after `drop` without waiting can race a lane thread still
    /// mid-teardown on a live Vulkan device, observed as an exit-time
    /// SIGSEGV in this sandbox (`crates/omni/tests/int8_thinker_executor.rs`,
    /// whose own doc has the full investigation).
    ///
    /// Sends [`Msg::Shutdown`] rather than just dropping `self` and waiting
    /// for the dispatcher's `rx.recv()` to error: every lane thread holds
    /// its OWN clone of that same `Msg` channel's sender (`done`, used to
    /// report [`Msg::Done`]/[`Msg::DoneMulti`] back), so dropping only this
    /// `Executor` handle's clone never actually closes the channel on its
    /// own - the dispatcher would then wait forever for a `rx.recv()` error
    /// that can only happen after the lanes exit, and the lanes only exit
    /// once the dispatcher drops their channel on its way out, a real
    /// deadlock (caught in this sandbox: the first version of this method
    /// tried the drop-and-wait shape and hung the whole test binary).
    /// `Shutdown` breaks that cycle by telling the dispatcher to exit
    /// directly, which is what then drops `lanes` and unblocks every lane.
    ///
    /// `join`ing every spawned thread's handle afterward is a real guarantee
    /// (Rust: every one of that thread's stack-local `Drop` calls has
    /// already run by the time `JoinHandle::join` returns), not a heuristic
    /// delay - replaces `int8_thinker_executor.rs`'s old `settle_teardown`
    /// sleep. Idempotent: a second call (or a call on a clone after another
    /// clone already shut down) finds the handles already taken and returns
    /// immediately.
    pub fn shutdown(&self) {
        let _ = self.tx.send(Msg::Shutdown);
        let taken = self.join_handles.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(handles) = taken {
            for h in handles {
                let _ = h.join();
            }
        }
    }

    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(Msg::Submit(Box::new(job)));
    }

    /// Register a model discovered after `start` -- the seam a supplier
    /// (e.g. a just-completed model-store fetch) uses to make a new model
    /// servable without restarting the process. Updates the public
    /// [`manifests`](Self::manifests) snapshot synchronously (so a caller
    /// that just registered a model sees it immediately in a listing), then
    /// hands the model to the dispatcher over the SAME channel `submit`
    /// uses -- FIFO ordering guarantees a `Submit` enqueued after this call
    /// returns sees the registration, so a caller may register-then-submit
    /// with no extra synchronization.
    pub fn register(&self, model: Arc<dyn ResidentModel>) {
        let manifest = model.manifest();
        {
            // Copy-on-write: readers hold cheap Arc snapshots, so a (rare)
            // registration rebuilds the Vec once instead of every reader
            // deep-cloning the whole catalog per call.
            let mut g = self.manifests.write().unwrap();
            let mut v = (**g).clone();
            v.push(manifest);
            *g = Arc::new(v);
        }
        let _ = self.tx.send(Msg::Register(model));
    }

    /// [`Self::register`], but a no-op if `model.manifest().model`'s name is
    /// already registered — atomically: the presence check and the insert
    /// happen under the SAME `manifests` write-lock acquisition, unlike a
    /// caller doing `!exec.manifests().iter().any(...)` then `exec.register()`
    /// as two separate steps (a check-then-act race a supplier's own
    /// single-flight gate does not close: that gate only serializes callers
    /// that overlap IN TIME — a straggler that lands after the leader already
    /// tore its gate down starts a fresh, unguarded episode). Returns `true`
    /// if this call actually registered the model, `false` if it was already
    /// present (in which case `model` is dropped, unused).
    pub fn register_if_absent(&self, model: Arc<dyn ResidentModel>) -> bool {
        let manifest = model.manifest();
        let name = manifest.model.clone();
        {
            let mut g = self.manifests.write().unwrap();
            if g.iter().any(|m| m.model == name) {
                return false;
            }
            let mut v = (**g).clone();
            v.push(manifest);
            *g = Arc::new(v);
        }
        let _ = self.tx.send(Msg::Register(model));
        true
    }

    /// [`Self::register`]'s multi-device sibling — registers a
    /// [`MultiDeviceResidentModel`] (e.g. a checkpoint layer-sharded across
    /// several GPUs, too large for any one of them alone). Register a model
    /// ONLY this way, never also via [`Self::register`]: a model whose plain
    /// [`ResidentModel::estimate`] reports a zero/placeholder cost (as a
    /// multi-device-only model's does, by necessity — it has no meaningful
    /// single-device footprint) would otherwise be reachable through the
    /// ordinary single-device claim path too, where a zero cost is placed on
    /// the CPU lane and its `activate` — which such a model correctly hard-
    /// errors on, since it can only run via `activate_multi` — fails every
    /// time. Keeping it out of the single-device registry makes that path
    /// structurally unreachable rather than relying on every caller to know
    /// not to take it.
    pub fn register_multi(&self, model: Arc<dyn MultiDeviceResidentModel>) {
        let manifest = model.manifest();
        {
            // Copy-on-write -- see `register`'s identical dance and its doc.
            let mut g = self.manifests.write().unwrap();
            let mut v = (**g).clone();
            v.push(manifest);
            *g = Arc::new(v);
        }
        let _ = self.tx.send(Msg::RegisterMulti(model));
    }

    /// Demote (drop) a resident multi-device instance, freeing its memory on
    /// every device it occupies. Returns `false` (refuses, evicts nothing)
    /// while a job is actively running against it, or if it isn't resident at
    /// all — see [`crate::manager::ResidencyManager::evict_multi`]'s doc for
    /// why pinned refuses rather than evicting out from under a running lane.
    /// Round-trips through the dispatcher like [`Self::residency`]/
    /// [`Self::in_flight`]; returns `false` if the dispatcher is gone.
    pub fn evict_multi(&self, key: InstanceKey) -> bool {
        let (tx, rx) = channel();
        if self.tx.send(Msg::EvictMulti { key, reply: tx }).is_err() {
            return false;
        }
        rx.recv().unwrap_or(false)
    }

    /// Submit and block for the result (the CLI's synchronous path).
    pub fn run_blocking(&self, model: &str, action: &str, inv: Invocation, on_progress: impl FnMut(Progress) + Send + 'static) -> ActionResult {
        let (rtx, rrx) = channel();
        self.submit(Job::new(model, action, inv).on_progress(on_progress).reply(move |r| {
            let _ = rtx.send(r);
        }));
        rrx.recv().unwrap_or_else(|_| Err("executor worker gone".into()))
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }

    /// A live residency + budget snapshot (which model is Hot on which device, at
    /// what memory cost, plus every device's budget). Round-trips a query through
    /// the dispatcher — the only thread that owns the [`ResidencyManager`] — so it
    /// is always consistent with scheduling. Returns an empty report if the
    /// dispatcher is gone. Mirrors [`stats`](Self::stats).
    pub fn residency(&self) -> crate::ResidencyReport {
        let (tx, rx) = channel();
        if self.tx.send(Msg::Report(tx)).is_err() {
            return crate::ResidencyReport::default();
        }
        rx.recv().unwrap_or_default()
    }

    /// A live list of every job the executor currently holds — queued or running —
    /// each with a stable submit-order `id`, model/action, coarse phase, and elapsed
    /// time since it was enqueued. Round-trips a query through the dispatcher (the
    /// only thread that owns the queue + running set) so it is consistent with
    /// scheduling. Returns an empty vec if the dispatcher is gone. Mirrors
    /// [`residency`](Self::residency).
    pub fn in_flight(&self) -> Vec<InFlightJob> {
        let (tx, rx) = channel();
        if self.tx.send(Msg::InFlight(tx)).is_err() {
            return Vec::new();
        }
        rx.recv().unwrap_or_default()
    }

    /// A snapshot of every currently-registered manifest, in registration
    /// order. An `Arc` snapshot (not a reference) because the underlying set
    /// can grow after `start` via [`register`](Self::register) -- and an Arc
    /// (not an owned Vec) because this used to DEEP-clone the whole catalog
    /// (every `ActionSpec`, param and help string) on every call, on every
    /// HTTP request path that resolves a model id. Registration is
    /// copy-on-write, so this clone is one refcount bump.
    pub fn manifests(&self) -> Arc<Vec<Manifest>> {
        self.manifests.read().unwrap().clone()
    }
}

// ---------------------------------------------------------------- dispatcher

fn dispatch_loop(rx: Receiver<Msg>, mut mgr: ResidencyManager, policy: Policy, lanes: HashMap<Device, Sender<RunReq>>, stats: Arc<Mutex<Stats>>) {
    let mut queue: Vec<Pending> = Vec::new();
    let mut running: HashSet<InstanceKey> = HashSet::new();
    let mut busy: HashSet<Device> = HashSet::new();
    // Jobs handed to a lane and running now — tracked here (not in `queue`) so the
    // in-flight query can report them, keyed by the group's instance key.
    let mut running_jobs: Vec<RunningJob> = Vec::new();
    // Monotonic job-id source; each submitted job gets the next value.
    let mut next_id: u64 = 0;
    // Best-effort observability refreshes on a TICK, not per message: rebuilding
    // a fresh HashMap of cloned keys/metrics under the stats lock after EVERY
    // dispatcher round was pure overhead on the scheduling thread (the D-Bus
    // stats stream itself only samples at ~2 Hz).
    const METRICS_REFRESH: std::time::Duration = std::time::Duration::from_millis(250);
    let mut last_metrics: Option<Instant> = None;

    loop {
        // Block for at least one message; a closed channel (every sender gone,
        // i.e. every Executor handle dropped) is real shutdown, not a panic --
        // stays outside the catch_unwind below so `return` here works normally.
        let first = match rx.recv() {
            Ok(msg) => msg,
            Err(_) => return,
        };
        // Drain everything else pending before scheduling, same as before.
        let mut msgs = vec![first];
        while let Ok(msg) = rx.try_recv() {
            msgs.push(msg);
        }
        // Shutdown takes priority over anything else drained alongside it -
        // an executor going away has nothing left to usefully do with a
        // trailing Submit/Report/etc. `return` here drops `lanes` (closing
        // every lane's channel, which is what makes each lane thread notice
        // and exit) - see `Msg::Shutdown`'s doc for why this explicit
        // message, not relying on `rx.recv()` erroring, is the only thing
        // that reliably unblocks this.
        if msgs.iter().any(|m| matches!(m, Msg::Shutdown)) {
            return;
        }
        // EACH message gets its OWN catch_unwind, for the SAME reason
        // `lane_loop` isolates each lane: a dropped `InstanceHandle` (e.g.
        // `ResidencyManager::build_failed` unwinding a failed claim) runs that
        // model's real Drop impl right here, on the dispatcher thread — and a
        // GPU backend whose device was already lost (a real, observed wgpu
        // `Device is lost` fault) can panic again from INSIDE that Drop, deep
        // in a third-party crate this code never calls directly. Without
        // this, that second panic kills the ONE thread that owns the entire
        // `ResidencyManager` and lane routing — every other model on the
        // server stops being scheduled forever, not just the one that hit the
        // fault. `build_failed`/`evict` release the budget and resident-slot
        // bookkeeping BEFORE the risky drop (see their own bodies), so a panic
        // here still leaves `queue`/`mgr`'s accounting consistent even though
        // the triggering message's own tail (e.g. its audit-log line) may not
        // have run. Per-message, not one catch_unwind around the whole batch:
        // an EARLIER message's panic must never stop a LATER already-drained
        // message (e.g. an unrelated model's own Submit) from still being
        // queued this round.
        for msg in msgs {
            if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                on_msg(msg, &mut queue, &mut mgr, &mut running, &mut busy, &stats, &mut running_jobs, &mut next_id);
            })) {
                eprintln!("[residency] dispatcher: panic processing a message: {} -- continuing", panic_message(p.as_ref()));
            }
        }
        // MUST run even if a message above panicked -- otherwise anything
        // that message's on_msg call had already queued (e.g. a DIFFERENT
        // model's Submit, drained into `msgs` before the panicking one) would
        // never get claimed: nothing else wakes this thread once `rx.recv()`
        // goes back to blocking, since no MORE messages are coming just
        // because scheduling stalled. Own catch_unwind too: `assign`'s
        // eviction fallback can ALSO drop a Hot instance.
        if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assign(&mut queue, &mut mgr, &policy, &lanes, &mut running, &mut busy, &stats, &mut running_jobs);
        })) {
            eprintln!("[residency] dispatcher: panic during assign: {} -- continuing", panic_message(p.as_ref()));
        }
        if last_metrics.is_none_or(|t| t.elapsed() >= METRICS_REFRESH) {
            if let Ok(mut s) = stats.lock() {
                s.metrics = mgr.all_metrics();
            }
            last_metrics = Some(Instant::now());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn on_msg(msg: Msg, queue: &mut Vec<Pending>, mgr: &mut ResidencyManager, running: &mut HashSet<InstanceKey>, busy: &mut HashSet<Device>, stats: &Arc<Mutex<Stats>>, running_jobs: &mut Vec<RunningJob>, next_id: &mut u64) {
    match msg {
        Msg::Submit(job) => match mgr.instance_key_for(&job.model, &job.action, &job.inv) {
            Some(key) => {
                let id = *next_id;
                *next_id += 1;
                queue.push(Pending {
                    id,
                    model: job.model,
                    action: job.action,
                    inv: job.inv,
                    key,
                    enqueued: Instant::now(),
                    on_progress: job.on_progress,
                    reply: job.reply,
                    on_admit: job.on_admit,
                });
                let mut s = stats.lock().unwrap();
                s.queue_peak = s.queue_peak.max(queue.len());
                s.queue_depth = queue.len();
            }
            None => (job.reply)(Err(format!("no model '{}'", job.model))),
        },
        Msg::Register(model) => {
            crate::log::info(&format!("model registered: {}", model.manifest().model));
            mgr.register(model);
        }
        Msg::RegisterMulti(model) => {
            crate::log::info(&format!("model registered (multi-device): {}", model.manifest().model));
            mgr.register_multi(model);
        }
        Msg::Built { key, handle } => {
            // `mgr.adopt` itself logs "built {key}" (via `ResidencyManager::event`).
            mgr.adopt(&key, handle);
            // Deferred activate()/promote() finished -- this group is no longer
            // "building" (see InFlightJob::phase's doc), even though it may keep
            // running for a while yet.
            for r in running_jobs.iter_mut() {
                if r.key == key {
                    r.building = false;
                }
            }
        }
        Msg::BuiltMulti { key, handle } => {
            // `mgr.adopt_multi` itself logs "built {key} (multi-device)".
            mgr.adopt_multi(&key, handle);
            // See Msg::Built's identical building-flip, above.
            for r in running_jobs.iter_mut() {
                if r.key == key {
                    r.building = false;
                }
            }
        }
        Msg::Report(tx) => {
            let _ = tx.send(mgr.report());
        }
        Msg::InFlight(tx) => {
            let now = Instant::now();
            let mut jobs: Vec<InFlightJob> = Vec::with_capacity(queue.len() + running_jobs.len());
            for p in queue.iter() {
                jobs.push(InFlightJob {
                    id: p.id,
                    model: p.model.clone(),
                    action: p.action.clone(),
                    phase: "queued".to_string(),
                    since_ms: now.saturating_duration_since(p.enqueued).as_millis() as u64,
                });
            }
            for r in running_jobs.iter() {
                jobs.push(InFlightJob {
                    id: r.id,
                    model: r.model.clone(),
                    action: r.action.clone(),
                    phase: if r.building { "building" } else { "running" }.to_string(),
                    since_ms: now.saturating_duration_since(r.enqueued).as_millis() as u64,
                });
            }
            jobs.sort_by_key(|j| j.id); // stable, submit-order
            let _ = tx.send(jobs);
        }
        Msg::Done { key, device, batch, failed } => {
            // Free the dispatcher's OWN bookkeeping (`busy`/`running`) BEFORE
            // the risky `mgr.build_failed`/`release` call below, which drops
            // the claim's `InstanceHandle` and can panic a SECOND time (a lost
            // GPU device's backend panicking again from inside its own Drop —
            // see dispatch_loop's per-message catch_unwind). If that panic cut
            // this handler off here instead, `device` would stay marked busy
            // FOREVER (nothing else ever clears it), silently wedging every
            // future claim on it — not a hang in this one model, a dead
            // device. `mgr`'s own state is a separate concern: `build_failed`/
            // `release` still run next, and their own edits to `residents`/
            // `budgets`/`instances` happen before THEIR risky drop too (see
            // their own bodies).
            running.remove(&key);
            running_jobs.retain(|r| r.key != key); // group finished — drop its jobs
            busy.remove(&device);
            if failed {
                crate::log::warn(&format!("model activation/run failed: {key} on {device:?}"));
                mgr.build_failed(&key); // unwind budget + slot; jobs already failed
            } else {
                mgr.release(&key);
            }
            let mut s = stats.lock().unwrap();
            s.batches += 1;
            s.jobs += batch as u64;
            s.max_batch = s.max_batch.max(batch);
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
        }
        Msg::DoneMulti { key, devices, batch, failed } => {
            // See `Msg::Done`'s identical reordering + doc, above.
            running.remove(&key);
            running_jobs.retain(|r| r.key != key);
            for &d in &devices {
                busy.remove(&d);
            }
            if failed {
                crate::log::warn(&format!("model activation/run failed (multi-device): {key} on {devices:?}"));
                mgr.build_failed_multi(&key); // unwind budget on EVERY device; jobs already failed
            } else {
                mgr.release_multi(&key);
            }
            let mut s = stats.lock().unwrap();
            s.batches += 1;
            s.jobs += batch as u64;
            s.max_batch = s.max_batch.max(batch);
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
            s.resident_multi = mgr.resident_multi_count();
        }
        Msg::EvictMulti { key, reply } => {
            let ok = mgr.evict_multi(&key);
            if let Ok(mut s) = stats.lock() {
                s.evictions = mgr.evictions;
                s.resident_multi = mgr.resident_multi_count();
            }
            let _ = reply.send(ok);
        }
        // Unreachable: dispatch_loop checks the drained batch for this
        // variant and returns before ever calling on_msg - see
        // Msg::Shutdown's doc. Still matched (not `_`) so a future Msg
        // variant added here fails exhaustiveness instead of silently
        // falling through the wildcard.
        Msg::Shutdown => {}
    }
}

/// Assign as many runnable groups as there are free device lanes.
#[allow(clippy::too_many_arguments)]
fn assign(queue: &mut Vec<Pending>, mgr: &mut ResidencyManager, policy: &Policy, lanes: &HashMap<Device, Sender<RunReq>>, running: &mut HashSet<InstanceKey>, busy: &mut HashSet<Device>, stats: &Arc<Mutex<Stats>>, running_jobs: &mut Vec<RunningJob>) {
    loop {
        // Groups whose key is not already running AND that can be placed on some
        // non-busy device (the scheduler policy then picks among them).
        let now = Instant::now();
        let rows = group_rows(queue, now, running);
        // Multi-device groups need `placeable_multi` (checks EVERY named device),
        // never plain `placeable` — a model registered ONLY via `register_multi`
        // is never in `self.models` at all, so `placeable` would always return
        // `false` for it and its jobs would sit in the queue forever, silently.
        let placeable: Vec<Group> = rows
            .iter()
            .filter(|r| if mgr.is_multi(&r.model) { mgr.placeable_multi(&r.key, &r.model, busy) } else { mgr.placeable(&r.key, &r.model, busy) })
            .map(|r| r.summary.clone())
            .collect();
        let (gid, batch) = match choose_next(&placeable, policy) {
            Some(x) => x,
            None => break, // nothing runnable right now
        };
        // choose_next returns the group's `id` FIELD — which group_rows numbered
        // as its index into `rows` — never an index into the filtered
        // `placeable` slice. (Indexing `placeable[gid]` here panicked the
        // dispatcher with index-out-of-bounds the moment any group was filtered
        // out as unplaceable, killing scheduling for the whole server.)
        let chosen = gid; // == the row index in `rows`
        let (key, model, action) = (rows[chosen].key.clone(), rows[chosen].model.clone(), rows[chosen].action.clone());

        // Claim (promote/evict, pin) on non-busy device(s). Multi-device models
        // (registered via `register_multi`, never also via `register` — see
        // `Executor::register_multi`'s doc) go through `claim_multi` instead of
        // `claim`; both `ClaimError` variants mean the same thing either way, so
        // the handling below is shared.
        let first = queue.iter().find(|p| p.key == key && p.action == action).map(|p| p.inv.clone());
        let inv = match first {
            Some(i) => i,
            None => break,
        };
        let is_multi = mgr.is_multi(&model);
        let claim_result = if is_multi {
            mgr.claim_multi(&model, &action, &inv, busy).map(|(w, d, k)| (ClaimOutcome::Multi(w, d), k))
        } else {
            mgr.claim(&model, &action, &inv, busy).map(|(w, d, k)| (ClaimOutcome::Single(w, d), k))
        };
        let (outcome, ckey) = match claim_result {
            Ok(x) => x,
            Err(ClaimError::NoCapacity(_)) => break, // wait for a lane to free a device
            Err(ClaimError::TooLarge(e)) | Err(ClaimError::Activate(e)) => {
                // Both permanent for this group: FAIL its queued jobs now and
                // keep scheduling the others. (The old code broke out of the
                // whole round here — the jobs waited forever and every other
                // group starved behind them.) TooLarge specifically will
                // never resolve itself by waiting (no eviction, however
                // aggressive, could ever make room), so it must not be
                // treated as `NoCapacity`'s "wait for a lane to free a device".
                let msg = format!("{model}/{action}: {e}");
                let mut i = 0;
                while i < queue.len() {
                    if queue[i].key == key && queue[i].action == action {
                        let p = queue.remove(i);
                        (p.reply)(Err(msg.clone()));
                    } else {
                        i += 1;
                    }
                }
                continue;
            }
        };

        // The HOME lane a multi-device group runs its ONE lane loop on: the
        // first device `estimate_multi` named (`pick_devices` preserves
        // declaration order). `busy` still gets every device the group
        // occupies below — the home lane only decides which thread runs
        // `run_group`, not which devices are considered occupied.
        let home: Device = match &outcome {
            ClaimOutcome::Single(_, device) => *device,
            ClaimOutcome::Multi(_, devices) => devices[0],
        };
        // Structurally shouldn't happen (`pick_device`/`pick_devices` only ever
        // name budgeted devices, and `Executor::start` spawns one lane per
        // budgeted device) — but a missing lane must fail this group's jobs
        // cleanly, never panic the one thread every model depends on.
        if !lanes.contains_key(&home) {
            match outcome {
                // Promote reuses the existing (Warm) instance, same as Hot --
                // just unpin it, don't unwind it out of existence.
                ClaimOutcome::Single(Claimed::Hot(_), _) | ClaimOutcome::Single(Claimed::Promote(_), _) => mgr.release(&ckey),
                ClaimOutcome::Single(Claimed::Build(_), _) => mgr.build_failed(&ckey),
                ClaimOutcome::Multi(ClaimedMulti::Hot(_), _) => mgr.release_multi(&ckey),
                ClaimOutcome::Multi(ClaimedMulti::Build(_), _) => mgr.build_failed_multi(&ckey),
            }
            let msg = format!("{model}/{action}: no lane for device {home:?}");
            let mut i = 0;
            while i < queue.len() {
                if queue[i].key == key && queue[i].action == action {
                    let p = queue.remove(i);
                    (p.reply)(Err(msg.clone()));
                } else {
                    i += 1;
                }
            }
            continue;
        }

        running.insert(ckey.clone());
        let target = match outcome {
            ClaimOutcome::Single(work, device) => {
                busy.insert(device);
                RunTarget::Single { work, device }
            }
            ClaimOutcome::Multi(work, devices) => {
                for &d in &devices {
                    busy.insert(d);
                }
                RunTarget::Multi { work, devices }
            }
        };

        // Pull the group's oldest `batch` jobs.
        let mut idxs: Vec<usize> = queue.iter().enumerate().filter(|(_, p)| p.key == ckey && p.action == action).map(|(i, _)| i).collect();
        idxs.sort_by_key(|&i| queue[i].enqueued);
        idxs.truncate(batch);
        idxs.sort_unstable_by(|a, b| b.cmp(a));
        let mut jobs: Vec<Pending> = idxs.into_iter().map(|i| queue.remove(i)).collect();
        jobs.reverse();

        // The group is now CLAIMED onto a lane — work is about to start. Fire each
        // job's admission signal exactly once (an HTTP layer gates streaming on it).
        for j in jobs.iter_mut() {
            if let Some(f) = j.on_admit.take() {
                f();
            }
        }

        // Track these jobs as running so the in-flight query keeps reporting them
        // (they've left `queue`); cleared on this group's `Done`. `building` starts
        // true for anything but an already-hot handle -- a deferred Build/Promote
        // (single-device) or Build (multi-device) still has to run
        // activate()/promote()/activate_multi() on the lane before Msg::Built/
        // BuiltMulti flips it to false (see on_msg's Msg::Built/BuiltMulti arms).
        let building = !matches!(target, RunTarget::Single { work: Claimed::Hot(_), .. } | RunTarget::Multi { work: ClaimedMulti::Hot(_), .. });
        if building {
            crate::log::info(&format!("model activating: {ckey}"));
        }
        for j in jobs.iter() {
            running_jobs.push(RunningJob { id: j.id, model: j.model.clone(), action: j.action.clone(), key: ckey.clone(), enqueued: j.enqueued, building });
        }

        // Sync residency counters immediately — a claim may have built/evicted, and
        // that must be visible before the lane's Done (which lags the actual run).
        if let Ok(mut s) = stats.lock() {
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
            s.resident_multi = mgr.resident_multi_count();
            s.max_parallel = s.max_parallel.max(busy.len());
            s.admitted += jobs.len() as u64;
            s.queue_depth = queue.len();
        }
        // Hand to the home lane (which builds the instance first when cold).
        let _ = lanes[&home].send(RunReq { target, action, jobs, key: ckey });
    }
}

/// The two shapes [`ResidencyManager::claim`]/[`ResidencyManager::claim_multi`]
/// can hand back, unified so `assign`'s post-claim tail (pull jobs, fire
/// `on_admit`, track `running_jobs`, sync stats, pick the lane) runs ONCE
/// instead of being duplicated per shape.
enum ClaimOutcome {
    Single(Claimed, Device),
    Multi(ClaimedMulti, Vec<Device>),
}

struct GroupRow {
    key: InstanceKey,
    model: String,
    action: String,
    summary: Group,
}

fn group_rows(queue: &[Pending], now: Instant, running: &HashSet<InstanceKey>) -> Vec<GroupRow> {
    let mut rows: Vec<GroupRow> = Vec::new();
    for p in queue {
        if running.contains(&p.key) {
            continue; // its instance is busy in a lane; wait for it to free
        }
        let age = now.saturating_duration_since(p.enqueued).as_millis() as u64;
        match rows.iter_mut().find(|r| r.key == p.key && r.action == p.action) {
            Some(r) => {
                r.summary.size += 1;
                r.summary.oldest_age_ms = r.summary.oldest_age_ms.max(age);
            }
            None => {
                let id = rows.len();
                rows.push(GroupRow { key: p.key.clone(), model: p.model.clone(), action: p.action.clone(), summary: Group { id, oldest_age_ms: age, size: 1 } });
            }
        }
    }
    // Re-number summary ids to be dense indices into `rows` (choose_next returns id).
    for (i, r) in rows.iter_mut().enumerate() {
        r.summary.id = i;
    }
    rows
}

// ---------------------------------------------------------------- lane

/// What a [`lane_loop`] iteration ran, carried alongside the handle out of
/// the big match below purely so the FINAL `Done`/`DoneMulti` send (after
/// `run_group`, or after a panic — see [`park_reply`]) knows which shape to
/// send without re-deriving it from anything guessable (e.g.
/// `devices.len() == 1`, which a genuine single-device multi-device instance
/// would make ambiguous).
enum Ran {
    Single(Device),
    Multi(Vec<Device>),
}

/// A job's reply callback, parked where BOTH the normal completion path and the
/// lane's panic handler can reach it — whichever runs first takes it, so the
/// reply fires exactly once either way.
type ReplySlot = Arc<Mutex<Option<Box<dyn FnOnce(ActionResult) + Send>>>>;

/// Move `p`'s reply into a shared slot and leave behind a wrapper that takes
/// from the slot. The normal path is unchanged (the wrapper delivers the reply);
/// after a panic the lane drains whatever slots are still full with an error.
fn park_reply(p: &mut Pending) -> ReplySlot {
    let slot: ReplySlot = Arc::new(Mutex::new(Some(std::mem::replace(&mut p.reply, Box::new(|_| {})))));
    let s = slot.clone();
    p.reply = Box::new(move |r| {
        if let Some(f) = s.lock().unwrap_or_else(|e| e.into_inner()).take() {
            f(r);
        }
    });
    slot
}

/// Best-effort human-readable panic payload.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "non-string panic payload".to_string())
}

fn lane_loop(rx: Receiver<RunReq>, done: Sender<Msg>) {
    while let Ok(mut req) = rx.recv() {
        let batch = req.jobs.len();
        let key = req.key.clone();
        // Salvage which `Done`/`DoneMulti` shape this group needs BEFORE `req`
        // moves into the `catch_unwind` closure below, so a panic can still
        // report on the right device(s) — matches `run_req`'s own `Ran` return
        // on the non-panic path.
        let ran_shape = match &req.target {
            RunTarget::Single { device, .. } => Ran::Single(*device),
            RunTarget::Multi { devices, .. } => Ran::Multi(devices.clone()),
        };
        // A lane runs model code — activate(), promote(), run_batch() — that
        // CAN panic in practice (a bug reachable only with real weights, a
        // driver fault). Without isolation a panic killed this thread with no
        // `Msg::Done`: the dispatcher never cleared `busy`/`running`, so the
        // device was never scheduled again and the model never ran again,
        // silently. Park each job's reply first so the unwind path can still
        // deliver an error to every waiter (never silence), then treat the
        // panic exactly like a failed activate: `failed: true` makes the
        // dispatcher unwind the claim (budget + slot + instance), so the key
        // is rebuilt fresh on its next claim instead of reusing an instance
        // whose internal state (and mutex) the panic may have poisoned.
        let slots: Vec<ReplySlot> = req.jobs.iter_mut().map(park_reply).collect();
        let (failed, ran) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_req(req, &done))) {
            Ok(outcome) => outcome,
            Err(p) => {
                let what = panic_message(p.as_ref());
                eprintln!("[residency] lane: panic while running {key}: {what}");
                let msg = format!("{key}: panicked while running: {what}");
                for slot in &slots {
                    if let Some(reply) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        reply(Err(msg.clone()));
                    }
                }
                (true, ran_shape)
            }
        };
        let done_msg = match ran {
            Ran::Single(device) => Msg::Done { key, device, batch, failed },
            Ran::Multi(devices) => Msg::DoneMulti { key, devices, batch, failed },
        };
        let _ = done.send(done_msg);
    }
}

/// One [`RunReq`]'s body: activate/promote as needed (single- or multi-device),
/// then run the group. Returns whether the claim failed (the dispatcher must
/// unwind it) alongside the [`Ran`] shape (needed for the `Done`/`DoneMulti`
/// choice on EVERY path, including a failed activate — `lane_loop`'s
/// `ran_shape` fallback exists only for the panic path, which never reaches
/// here). Runs under `lane_loop`'s `catch_unwind`; the `Done`/`DoneMulti` send
/// stays OUTSIDE, in `lane_loop` itself, so it is delivered on every path
/// including a panic.
fn run_req(req: RunReq, done: &Sender<Msg>) -> (bool, Ran) {
    let RunReq { target, action, jobs, key } = req;
    let (handle, ran): (InstanceHandle, Ran) = match target {
        RunTarget::Single { work: Claimed::Hot(h), device } => (h, Ran::Single(device)),
        // Deferred activation happens HERE, on the device's own lane: a slow
        // or wedged build stalls this device only, never the dispatcher.
        RunTarget::Single { work: Claimed::Build(m), device } => match m.activate(&key, device) {
            Ok(inst) => {
                let h: InstanceHandle = Arc::new(Mutex::new(inst));
                let _ = done.send(Msg::Built { key: key.clone(), handle: h.clone() });
                (h, Ran::Single(device))
            }
            Err(e) => {
                // Activation failed: every job in the group gets the error
                // (never silence), and the dispatcher unwinds the claim.
                let msg = format!("activate {key}: {e}");
                for p in jobs {
                    (p.reply)(Err(msg.clone()));
                }
                return (true, Ran::Single(device));
            }
        },
        // Deferred promotion, same reasoning as `Build`'s deferred
        // activate: rebuilding a demoted instance's device buffers can
        // be slow, so it happens on this device's own lane, never the
        // dispatcher thread. The existing `Instance` is reused in place.
        RunTarget::Single { work: Claimed::Promote(h), device } => {
            let result = h.lock().unwrap().promote(device);
            match result {
                Ok(()) => {
                    let _ = done.send(Msg::Built { key: key.clone(), handle: h.clone() });
                    (h, Ran::Single(device))
                }
                Err(e) => {
                    let msg = format!("promote {key}: {e}");
                    for p in jobs {
                        (p.reply)(Err(msg.clone()));
                    }
                    return (true, Ran::Single(device));
                }
            }
        }
        RunTarget::Multi { work: ClaimedMulti::Hot(h), devices } => (h, Ran::Multi(devices)),
        // Same deferred-to-the-lane discipline as the single-device case
        // above, and more load-bearing here: a multi-device weight stream
        // (tens of GiB across several GPUs) is exactly the "can take
        // seconds" case that rule exists for.
        RunTarget::Multi { work: ClaimedMulti::Build(m), devices } => match m.activate_multi(&key, &devices) {
            Ok(inst) => {
                let h: InstanceHandle = Arc::new(Mutex::new(inst));
                let _ = done.send(Msg::BuiltMulti { key: key.clone(), handle: h.clone() });
                (h, Ran::Multi(devices))
            }
            Err(e) => {
                let msg = format!("activate_multi {key}: {e}");
                for p in jobs {
                    (p.reply)(Err(msg.clone()));
                }
                return (true, Ran::Multi(devices));
            }
        },
    };
    // `run_group` is one implementation shared by both shapes above — a
    // multi-device instance is just an `Instance` whose forward happens to
    // span devices internally; nothing here needs to know that.
    run_group(&handle, &action, jobs);
    (false, ran)
}

fn run_group(handle: &InstanceHandle, action: &str, jobs: Vec<Pending>) {
    let invs: Vec<Invocation> = jobs.iter().map(|p| p.inv.clone()).collect();
    let mut sinks: Vec<Box<dyn FnMut(Progress) + Send>> = Vec::with_capacity(jobs.len());
    let mut replies: Vec<Box<dyn FnOnce(ActionResult) + Send>> = Vec::with_capacity(jobs.len());
    for p in jobs {
        sinks.push(p.on_progress);
        replies.push(p.reply);
    }
    let results = {
        let mut inst = handle.lock().unwrap();
        // Route each progress update to ITS job's sink only (by batch index), so
        // per-sequence token streams don't cross.
        let mut route = |i: usize, pr: Progress| {
            if let Some(s) = sinks.get_mut(i) {
                s(pr);
            }
        };
        inst.run_batch(action, &invs, &mut route)
    };
    if results.len() == replies.len() {
        for (r, o) in replies.into_iter().zip(results) {
            r(o);
        }
    } else {
        let err = format!("model returned {} results for {} jobs", results.len(), replies.len());
        for r in replies {
            r(Err(err.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budgets;
    use crate::{Instance, MemCost, MultiDeviceCost};
    use capability::{ActionResult, ActionSpec, Blob, Manifest, Media, Outcome};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    const GB: u64 = 1 << 30;

    struct Slow {
        name: String,
        vram: u64,
        ms: u64,
        builds: Arc<AtomicU32>,
    }
    struct SlowInst {
        ms: u64,
    }
    impl ResidentModel for Slow {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "slow", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SlowInst { ms: self.ms }))
        }
    }
    impl Instance for SlowInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            std::thread::sleep(Duration::from_millis(self.ms));
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    fn submit_wait(exec: &Executor, models: &[&str]) -> Duration {
        let (tx, rx) = channel();
        for m in models {
            let tx = tx.clone();
            exec.submit(Job::new(*m, "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        }
        drop(tx);
        let t = Instant::now();
        for _ in models {
            rx.recv().unwrap().unwrap();
        }
        t.elapsed()
    }

    #[test]
    fn two_models_on_two_gpus_run_in_parallel() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(Slow { name: "a".into(), vram: 10 * GB, ms: 120, builds: builds.clone() }),
            Arc::new(Slow { name: "b".into(), vram: 10 * GB, ms: 120, builds: builds.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());
        // a and b live on different GPUs → they run concurrently: wall ~120ms, not 240.
        let elapsed = submit_wait(&exec, &["a", "b"]);
        assert!(elapsed < Duration::from_millis(210), "expected parallel (<210ms), took {elapsed:?}");
        assert!(exec.stats().max_parallel >= 2, "expected 2 lanes at once, stats={:?}", exec.stats());
    }

    /// Regression test for a real deadlock in an earlier version of
    /// `Executor::shutdown`: it dropped the `Executor` handle's own `tx`
    /// clone and then waited for the dispatcher's `rx.recv()` to error out
    /// -- but every lane thread ALSO holds a clone of that same sender
    /// (`done`, used to report `Msg::Done` back), so dropping only the
    /// `Executor`'s clone never actually closed the channel: the dispatcher
    /// waited forever for lanes to exit, and the lanes waited forever for
    /// the dispatcher to drop their own channel first. `shutdown` now sends
    /// an explicit `Msg::Shutdown` instead of relying on that never-firing
    /// auto-close -- this test would hang (and eventually be killed by the
    /// test harness's timeout) if that regressed.
    #[test]
    fn shutdown_returns_promptly_and_is_idempotent() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Slow { name: "a".into(), vram: GB, ms: 10, builds })];
        let exec = Executor::start(models, budgets, Policy::default());
        submit_wait(&exec, &["a"]); // real GPU-lane work before teardown, like a_second_generate_reuses_the_hot_sharded_instance

        let t = Instant::now();
        exec.shutdown();
        assert!(t.elapsed() < Duration::from_secs(5), "shutdown took {:?} -- looks hung, not just slow", t.elapsed());

        // Idempotent: a second call must not hang or panic either (the
        // handles are already taken, so this is a no-op join of nothing).
        exec.shutdown();

        // The dispatcher is genuinely gone: a submit after shutdown must not
        // panic (Executor::submit already swallows a closed-channel send
        // error), and must never receive a reply.
        let (tx, rx) = channel();
        exec.submit(Job::new("a", "run", Invocation::new()).reply(move |r| {
            let _ = tx.send(r);
        }));
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "a submit after shutdown must never be scheduled");
    }

    #[test]
    fn same_model_batches_and_evicts() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB); // 22 usable
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(Slow { name: "a".into(), vram: 20 * GB, ms: 30, builds: builds.clone() }),
            Arc::new(Slow { name: "b".into(), vram: 20 * GB, ms: 30, builds: builds.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());
        // 12 jobs to `a`: while the first runs the rest queue and batch on one build.
        submit_wait(&exec, &["a"; 12]);
        let s = exec.stats();
        assert!(s.max_batch >= 2, "expected batching, max_batch={}", s.max_batch);
        // `b` (20 GB) can't fit beside `a` (20 GB) on the 22 GB card → `a` evicted.
        submit_wait(&exec, &["b"]);
        let s = exec.stats();
        assert!(s.evictions >= 1, "expected an eviction, stats={s:?}");
        assert_eq!(s.resident, 1);
    }

    #[test]
    fn stateless_zero_cost_model_is_schedulable_on_a_gpu_only_budget() {
        // Regression: a zero-cost instance (MemCost::default(), e.g. a stateless
        // ProviderResident like `demo`) must be placeable even when only GPU
        // budgets exist — this is exactly how the D-Bus roundtrip test wires up.
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Slow { name: "free".into(), vram: 0, ms: 1, builds })];
        let exec = Executor::start(models, budgets, Policy::default());
        assert!(exec.run_blocking("free", "run", Invocation::new(), |_| {}).is_ok());
    }

    #[test]
    fn unknown_model_replies_error() {
        let exec = Executor::start(vec![], Budgets::new(), Policy::default());
        assert!(exec.run_blocking("nope", "x", Invocation::new(), |_| {}).is_err());
    }

    /// A model whose activation always fails.
    struct BadActivate;
    impl ResidentModel for BadActivate {
        fn manifest(&self) -> Manifest {
            Manifest::new("bad", "always fails to activate", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("bad", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(GB, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("checkpoint not found: /nope.safetensors".into())
        }
    }

    /// REGRESSION (2026-07-30 wedge): an activation failure must (1) reply the
    /// error to every queued job of the group — never silence — and (2) leave the
    /// scheduler fully alive for other models. The old code broke out of the
    /// assign round on any claim error: the jobs waited forever and every later
    /// Run on ANY model queued behind them.
    #[test]
    fn activation_failure_replies_errors_and_does_not_wedge_the_executor() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(BadActivate),
            Arc::new(Slow { name: "good".into(), vram: GB, ms: 5, builds: builds.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());

        // several queued jobs on the failing model: ALL get the error
        let (tx, rx) = channel();
        for _ in 0..3 {
            let tx = tx.clone();
            exec.submit(Job::new("bad", "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        }
        for _ in 0..3 {
            let r = rx.recv_timeout(Duration::from_secs(5)).expect("reply must arrive, not hang");
            let e = r.expect_err("activation failure must surface");
            assert!(e.contains("checkpoint not found"), "err: {e}");
        }
        // the executor is still alive: a good model runs fine afterwards
        let ok = exec.run_blocking("good", "run", Invocation::new(), |_| {});
        assert!(ok.is_ok(), "executor wedged after activation failure: {ok:?}");
        // and the failed claim's budget was unwound (nothing resident from `bad`)
        assert_eq!(exec.stats().resident, 1, "only the good instance is resident");
        // budget really freed: `bad` can be retried and fails cleanly again
        assert!(exec.run_blocking("bad", "run", Invocation::new(), |_| {}).is_err());
    }

    /// A model whose activate() or run() PANICS (not Err) — the shape a truncated
    /// checkpoint used to produce inside a lane before mmap validation existed,
    /// and the shape any latent model bug still can.
    struct Panicky {
        name: String,
        panic_in_activate: bool,
    }
    struct PanickyInst;
    impl ResidentModel for Panicky {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "panics", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(GB, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            if self.panic_in_activate {
                panic!("simulated activation panic (e.g. corrupt checkpoint)");
            }
            Ok(Box::new(PanickyInst))
        }
    }
    impl Instance for PanickyInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            panic!("simulated run_batch panic");
        }
    }

    /// SPEC (audit F1): a panic in a lane — during activate() or run_batch() —
    /// must (1) reply an error to every job of the group, never hang the
    /// waiters, and (2) deliver Msg::Done so the dispatcher clears busy/running
    /// and the DEVICE AND MODEL both stay schedulable. Before panic isolation
    /// the lane thread died silently and the device + model were wedged forever.
    #[test]
    fn a_panicking_lane_replies_errors_and_the_device_recovers() {
        for panic_in_activate in [true, false] {
            let builds = Arc::new(AtomicU32::new(0));
            let mut budgets = Budgets::new();
            budgets.set(Device::Gpu(0), 24 * GB, 0); // ONE device: a wedge would block everything
            let models: Vec<Arc<dyn ResidentModel>> = vec![
                Arc::new(Panicky { name: "boom".into(), panic_in_activate }),
                Arc::new(Slow { name: "good".into(), vram: GB, ms: 1, builds: builds.clone() }),
            ];
            let exec = Executor::start(models, budgets, Policy::default());

            // Every queued job on the panicking model gets an error reply, not a hang.
            let (tx, rx) = channel();
            for _ in 0..3 {
                let tx = tx.clone();
                exec.submit(Job::new("boom", "run", Invocation::new()).reply(move |r| {
                    let _ = tx.send(r);
                }));
            }
            for _ in 0..3 {
                let r = rx.recv_timeout(Duration::from_secs(5)).expect("reply must arrive, not hang (panic_in_activate={panic_in_activate})");
                let e = r.expect_err("a panicked group must surface an error");
                assert!(e.contains("panic"), "err: {e}");
            }

            // The ONLY device recovered: another model runs on it afterwards.
            let ok = exec.run_blocking("good", "run", Invocation::new(), |_| {});
            assert!(ok.is_ok(), "device wedged after a lane panic (panic_in_activate={panic_in_activate}): {ok:?}");
            // The claim was unwound (nothing from `boom` left resident or charged).
            assert_eq!(exec.stats().resident, 1, "only the good instance may be resident");
            // And the panicking model itself stays schedulable — it fails again
            // cleanly (fresh build each time) instead of being silently dead.
            assert!(exec.run_blocking("boom", "run", Invocation::new(), |_| {}).is_err());
            assert!(exec.run_blocking("good", "run", Invocation::new(), |_| {}).is_ok());
        }
    }

    /// An `Instance` whose `run` panics (e.g. "device lost" mid-run — the real
    /// fault this regression is named for) AND whose `Drop` ALSO panics —
    /// modeling a third-party GPU backend's own internal teardown panicking a
    /// SECOND time when it discovers the device is already lost (observed
    /// verbatim: `wgpu-core`'s device-poll-on-drop hits the same fault its
    /// caller already panicked on). This second panic fires while
    /// `ResidencyManager::build_failed` drops the removed `InstanceHandle` --
    /// on the DISPATCHER thread, not a lane.
    struct DropPanics;
    impl Instance for DropPanics {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            panic!("simulated run panic (e.g. device lost mid-run)");
        }
    }
    impl Drop for DropPanics {
        fn drop(&mut self) {
            panic!("simulated drop panic (e.g. wgpu-core's own poll-on-drop hitting an already-lost device)");
        }
    }
    struct DropPanicsModel;
    impl ResidentModel for DropPanicsModel {
        fn manifest(&self) -> Manifest {
            Manifest::new("boom", "drop-panics", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("boom", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(GB, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Ok(Box::new(DropPanics))
        }
    }

    /// SPEC: a panic while DROPPING a failed claim's instance -- runs on the
    /// DISPATCHER thread (inside `on_msg`'s `Msg::Done` handling, via
    /// `ResidencyManager::build_failed`'s `instances.remove`), not a lane --
    /// must not kill the dispatcher. Before this fix it did: the dispatcher
    /// thread owns the whole `ResidencyManager` and every lane's routing, so
    /// its death silently wedges EVERY OTHER model on the server, not just the
    /// one that hit the fault (the exact failure mode `a_panicking_lane_
    /// replies_errors_and_the_device_recovers` already covers for a LANE
    /// panic; this is its dispatcher-thread sibling).
    #[test]
    fn a_panic_while_dropping_a_failed_instance_does_not_wedge_the_dispatcher() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0); // ONE device, same as the lane-panic test
        let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(DropPanicsModel), Arc::new(Slow { name: "good".into(), vram: GB, ms: 1, builds })];
        let exec = Executor::start(models, budgets, Policy::default());

        // "boom" fails (its run() panics, then its Drop panics AGAIN on the
        // dispatcher thread while the claim unwinds) -- the caller must still
        // get a reply, not hang. `run_blocking`'s own recv is unbounded, so
        // this call itself would hang forever pre-fix -- exactly the failure
        // this regression exists to catch. Bounded via a plain submit +
        // recv_timeout (matching `a_panicking_lane_...`'s own pattern) so a
        // real wedge fails the test instead of hanging the whole suite.
        let (tx, rx) = channel();
        exec.submit(Job::new("boom", "run", Invocation::new()).reply(move |r| {
            let _ = tx.send(r);
        }));
        let err = rx.recv_timeout(Duration::from_secs(5)).expect("boom must reply, not hang");
        assert!(err.is_err(), "a run()-panicking instance must reply an error");

        // The real assertion: the DISPATCHER survived dropping it. A
        // completely unrelated model, submitted after, must still get
        // scheduled and run -- bounded the same way, for the same reason.
        let (tx2, rx2) = channel();
        exec.submit(Job::new("good", "run", Invocation::new()).reply(move |r| {
            let _ = tx2.send(r);
        }));
        let ok = rx2.recv_timeout(Duration::from_secs(5)).expect("dispatcher wedged after a panic while dropping a failed instance -- 'good' never replied");
        assert!(ok.is_ok(), "an unrelated model must still run after the dispatcher survives a drop panic: {ok:?}");
    }

    /// The residency accessor (`Executor::residency`) round-trips a query through
    /// the dispatcher and reports every placed instance + every device budget. After
    /// a run the instance stays resident (released, not evicted), so it must appear
    /// in the report with its device, tier, and memory cost, and its bytes must be
    /// accounted against the placed device's budget.
    #[test]
    fn residency_accessor_reports_placed_instances_and_budgets() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Cpu, 8 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Slow { name: "a".into(), vram: 10 * GB, ms: 3, builds })];
        let exec = Executor::start(models, budgets, Policy::default());
        exec.run_blocking("a", "run", Invocation::new(), |_| {}).unwrap();

        let report = exec.residency();
        // The placed instance is reported on GPU 0 at its 10 GB cost, Hot.
        let p = report.placements.iter().find(|p| p.key.model == "a").expect("instance 'a' resident");
        assert_eq!(p.device, Device::Gpu(0));
        assert_eq!(p.tier, crate::Tier::Hot);
        assert_eq!(p.mem, 10 * GB);
        // Budgets cover every device, deterministically ordered (CPU first), with the
        // 10 GB accounted against GPU 0's `used`.
        assert_eq!(report.budgets.first().map(|b| b.device), Some(Device::Cpu));
        let gpu = report.budgets.iter().find(|b| b.device == Device::Gpu(0)).expect("gpu budget");
        assert_eq!(gpu.total, 24 * GB);
        assert_eq!(gpu.reserved, 2 * GB);
        assert!(gpu.used >= 10 * GB, "used={} should include the resident instance", gpu.used);
    }

    /// A model whose activation is SLOW (weight load / NPU graph compile).
    struct SlowActivate {
        ms: u64,
    }
    impl ResidentModel for SlowActivate {
        fn manifest(&self) -> Manifest {
            Manifest::new("slowboot", "slow activation", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("slowboot", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(GB, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            std::thread::sleep(Duration::from_millis(self.ms));
            Ok(Box::new(SlowInst { ms: 1 }))
        }
    }

    /// REGRESSION: a ZERO-cost (stateless) model must be schedulable. demo/
    /// imageops have MemCost::default() == 0/0/0; the old pick_device had no
    /// branch for that, so their jobs were unplaceable and hung forever.
    #[test]
    fn zero_cost_stateless_model_is_schedulable() {
        struct Free;
        struct FreeInst;
        impl ResidentModel for Free {
            fn manifest(&self) -> Manifest {
                Manifest::new("free", "stateless", vec![ActionSpec::new("run", "run")])
            }
            fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
                InstanceKey::new("free", "stateless")
            }
            fn estimate(&self, _k: &InstanceKey) -> MemCost {
                MemCost::default() // 0 vram / 0 ram / 0 npu — the demo case
            }
            fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
                Ok(Box::new(FreeInst))
            }
        }
        impl Instance for FreeInst {
            fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
                Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![7])))
            }
        }
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Cpu, 8 * GB, 0);
        let exec = Executor::start(vec![Arc::new(Free)], budgets, Policy::default());
        let r = exec.run_blocking("free", "run", Invocation::new(), |_| {});
        assert!(r.is_ok(), "stateless model must run, got {r:?}");
    }

    /// REGRESSION (dispatcher panic, 2026-07-30): `choose_next` returns a group
    /// **id** (an index into `rows`), not an index into the FILTERED `placeable`
    /// slice. With one group unplaceable (here: `b` only fits the busy GPU 0),
    /// the old `placeable[gid]` was out of bounds — the dispatcher died and every
    /// later call on any model got "executor worker gone".
    #[test]
    fn filtered_unplaceable_group_does_not_kill_the_dispatcher() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 8 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(Slow { name: "a".into(), vram: 10 * GB, ms: 250, builds: builds.clone() }),
            Arc::new(Slow { name: "b".into(), vram: 20 * GB, ms: 5, builds: builds.clone() }),
            Arc::new(Slow { name: "c".into(), vram: 4 * GB, ms: 5, builds: builds.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());
        let (tx, rx) = channel();
        // a occupies GPU 0; while it runs, b (only fits the busy GPU 0) is
        // unplaceable and filtered; c (fits GPU 1) must still be scheduled.
        for m in ["a", "b", "c"] {
            let tx = tx.clone();
            exec.submit(Job::new(m, "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        }
        for _ in 0..3 {
            rx.recv_timeout(Duration::from_secs(5))
                .expect("all three must complete (dispatcher alive)")
                .unwrap();
        }
    }

    /// REGRESSION: activation runs on the device's LANE, never the dispatcher.
    /// While one model's activation grinds (or hangs), models on other devices
    /// must keep dispatching. The old code activated inside the dispatcher's
    /// claim, freezing ALL scheduling for the duration.
    #[test]
    fn slow_activation_only_stalls_its_own_device() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(SlowActivate { ms: 600 }),
            Arc::new(Slow { name: "fast".into(), vram: GB, ms: 5, builds: builds.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());

        let (stx, srx) = channel();
        exec.submit(Job::new("slowboot", "run", Invocation::new()).reply(move |r| { let _ = stx.send(r); }));
        // While slowboot activates on its lane, `fast` must complete quickly.
        let t = Instant::now();
        let ok = exec.run_blocking("fast", "run", Invocation::new(), |_| {});
        let fast_elapsed = t.elapsed();
        assert!(ok.is_ok());
        assert!(fast_elapsed < Duration::from_millis(300),
                "fast model blocked behind another model's activation: {fast_elapsed:?}");
        // and slowboot itself still completes
        assert!(srx.recv_timeout(Duration::from_secs(5)).expect("slowboot reply").is_ok());
    }

    /// A model whose `run` blocks until a shared gate is released — so a job can be
    /// held mid-flight while the test inspects the executor. `entered` flips once the
    /// lane is actually executing the job (so the test can wait for the running
    /// state deterministically, with no sleeps-as-timing).
    struct Gated {
        name: String,
        vram: u64,
        entered: Arc<AtomicU32>,
        release: Arc<AtomicBool>,
    }
    struct GatedInst {
        entered: Arc<AtomicU32>,
        release: Arc<AtomicBool>,
    }
    impl ResidentModel for Gated {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "gated", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Ok(Box::new(GatedInst { entered: self.entered.clone(), release: self.release.clone() }))
        }
    }
    impl Instance for GatedInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            self.entered.fetch_add(1, Ordering::SeqCst);
            // Block until released — bounded so a bug can never hang the suite.
            let start = Instant::now();
            while !self.release.load(Ordering::SeqCst) {
                if start.elapsed() > Duration::from_secs(5) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    /// `Executor::in_flight` reports both a RUNNING job (claimed onto a lane) and a
    /// QUEUED one (still waiting), with stable, monotonic submit-order ids. Two
    /// 20 GB models on a 24 GB card: only one fits, so while `a` runs (gated) `b`
    /// cannot be placed and sits queued — a deterministic queued+running mix.
    #[test]
    fn in_flight_reports_queued_and_running_jobs_with_monotonic_ids() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let entered = Arc::new(AtomicU32::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(Gated { name: "a".into(), vram: 20 * GB, entered: entered.clone(), release: release.clone() }),
            Arc::new(Gated { name: "b".into(), vram: 20 * GB, entered: entered.clone(), release: release.clone() }),
        ];
        let exec = Executor::start(models, budgets, Policy::default());

        let (tx, rx) = channel();
        let (ta, tb) = (tx.clone(), tx.clone());
        exec.submit(Job::new("a", "run", Invocation::new()).reply(move |r| { let _ = ta.send(r); }));
        exec.submit(Job::new("b", "run", Invocation::new()).reply(move |r| { let _ = tb.send(r); }));

        // Wait until `a` is genuinely running on its lane (no timing guesswork).
        let start = Instant::now();
        while entered.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "gated run never started");
            std::thread::sleep(Duration::from_millis(2));
        }

        let jobs = exec.in_flight();
        // Both jobs are in flight: exactly one running, one queued (only one 20 GB
        // model fits the card). Which model the scheduler picked to run is not fixed,
        // so assert the split by phase, not by name.
        let running: Vec<_> = jobs.iter().filter(|j| j.phase == "running").collect();
        let queued: Vec<_> = jobs.iter().filter(|j| j.phase == "queued").collect();
        assert_eq!(running.len(), 1, "exactly one running, jobs={jobs:?}");
        assert_eq!(queued.len(), 1, "exactly one queued, jobs={jobs:?}");
        assert_eq!(running[0].action, "run");
        // Ids are monotonic in SUBMIT order (independent of which was scheduled): `a`
        // submitted first, so its id is the smaller one.
        let a = jobs.iter().find(|j| j.model == "a").expect("a in flight");
        let b = jobs.iter().find(|j| j.model == "b").expect("b in flight");
        assert!(a.id < b.id, "ids must be monotonic in submit order: a={} b={}", a.id, b.id);
        assert!(jobs.iter().all(|j| j.since_ms < 5_000), "since_ms should be a small elapsed, jobs={jobs:?}");

        // Release: both finish and the in-flight list drains.
        release.store(true, Ordering::SeqCst);
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        // The list drains once both groups' `Done` land (which the lane sends just
        // after each reply — so poll rather than assume it is already processed).
        let start = Instant::now();
        while !exec.in_flight().is_empty() {
            assert!(start.elapsed() < Duration::from_secs(5), "in-flight never drained: {:?}", exec.in_flight());
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn register_makes_a_model_immediately_listed_and_schedulable() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        // Starts with nothing registered -- a model discovered later (e.g. a
        // just-completed model-store fetch) is the case this seam exists for.
        let exec = Executor::start(vec![], budgets, Policy::default());
        assert!(exec.manifests().is_empty());

        let model: Arc<dyn ResidentModel> = Arc::new(Slow { name: "late".into(), vram: GB, ms: 5, builds: builds.clone() });
        exec.register(model);

        // The manifest snapshot reflects the registration synchronously --
        // register() returns only after updating it, before the dispatcher
        // even sees the message.
        let names: Vec<String> = exec.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["late".to_string()]);

        // And it is schedulable: a Submit enqueued right after register()
        // returns finds it, because both share one FIFO channel.
        let elapsed = submit_wait(&exec, &["late"]);
        assert!(elapsed < Duration::from_secs(2), "registered model never ran in time: {elapsed:?}");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn register_if_absent_is_atomically_idempotent_under_real_concurrency() {
        // A supplier's own single-flight gate only serializes callers that
        // overlap in time (see crates/cli/src/supply.rs's `ensure`) -- a
        // straggler landing after the gate tears down calls `register`-shaped
        // code completely unguarded. `register_if_absent` must still produce
        // exactly ONE manifest entry even when every caller races it with no
        // gate at all, which is the harder bar `register` alone does not clear
        // (a plain check-then-`register()` from separate calls is a TOCTOU).
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());

        let builds = Arc::new(AtomicU32::new(0));
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let exec = exec.clone();
                let builds = builds.clone();
                std::thread::spawn(move || {
                    let model: Arc<dyn ResidentModel> = Arc::new(Slow { name: "racy".into(), vram: 0, ms: 0, builds: builds.clone() });
                    exec.register_if_absent(model)
                })
            })
            .collect();
        let newly_registered = handles.into_iter().map(|h| h.join().unwrap()).filter(|&b| b).count();

        assert_eq!(newly_registered, 1, "exactly one of the 16 racing callers must see itself as the first to register");
        let names: Vec<String> = exec.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["racy".to_string()], "the manifest list must carry exactly one entry, not one per racing caller");
    }

    // ------------------------------------------------------------ multi-device

    /// A fake model spanning BOTH gpu0 and gpu1 at once — the `Executor`-level
    /// twin of `manager.rs`'s `MultiFake`, driven through the FULL dispatcher +
    /// lane machinery instead of `ResidencyManager` directly. Its plain
    /// `ResidentModel::estimate`/`activate` are deliberately unusable (zero
    /// cost, hard error) — exactly `Int8ThinkerResident`'s real shape — so any
    /// test that succeeds here proves the model ran via `claim_multi`, never
    /// `claim`.
    struct MultiFake {
        name: String,
        per_gpu: u64,
        live: Arc<AtomicU32>,
    }
    struct MultiFakeInst {
        live: Arc<AtomicU32>,
    }
    impl Drop for MultiFakeInst {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl ResidentModel for MultiFake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake multi", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0) // never consulted -- registered ONLY via register_multi
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("MultiFake: single-device activate is not this model's contract".to_string())
        }
    }
    impl MultiDeviceResidentModel for MultiFake {
        fn estimate_multi(&self, _k: &InstanceKey) -> MultiDeviceCost {
            MultiDeviceCost::new(vec![(Device::Gpu(0), self.per_gpu), (Device::Gpu(1), self.per_gpu)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
            assert_eq!(devices, [Device::Gpu(0), Device::Gpu(1)], "activate_multi must see exactly the devices estimate_multi named");
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(MultiFakeInst { live: self.live.clone() }))
        }
    }
    impl Instance for MultiFakeInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    #[test]
    fn multi_device_model_runs_through_the_executor_and_occupies_every_device() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 15 * GB, live: live.clone() }));

        let r = exec.run_blocking("int8thinker", "run", Invocation::new(), |_| {});
        assert!(r.is_ok(), "{r:?}");
        assert_eq!(live.load(Ordering::SeqCst), 1);

        let report = exec.residency();
        assert_eq!(report.multi_placements.len(), 1);
        assert!(report.placements.is_empty(), "a multi-device instance must not also appear in the single-device placements list");
        let mut devs = report.multi_placements[0].devices.clone();
        devs.sort_by_key(|&(d, _)| match d {
            Device::Gpu(i) => i,
            _ => u32::MAX,
        });
        assert_eq!(devs, vec![(Device::Gpu(0), 15 * GB), (Device::Gpu(1), 15 * GB)]);
        let gpu0 = report.budgets.iter().find(|b| b.device == Device::Gpu(0)).expect("gpu0 budget");
        let gpu1 = report.budgets.iter().find(|b| b.device == Device::Gpu(1)).expect("gpu1 budget");
        assert!(gpu0.used >= 15 * GB, "gpu0.used={}", gpu0.used);
        assert!(gpu1.used >= 15 * GB, "gpu1.used={}", gpu1.used);
    }

    /// A multi-only model (zero-cost `estimate`, hard-erroring `activate`) must
    /// run via `claim_multi` even when a CPU budget is ALSO present — the case
    /// that would silently break if `register_multi` ever also called
    /// `register` (a zero-cost model's PLAIN `estimate` places it on the CPU
    /// lane preferentially, per `place::pick_device`'s zero-cost branch, where
    /// `MultiFake::activate` deliberately errors).
    #[test]
    fn a_multi_only_model_is_never_claimed_on_the_single_device_path() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0).set(Device::Cpu, 128 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 1 * GB, live: live.clone() }));
        let r = exec.run_blocking("int8thinker", "run", Invocation::new(), |_| {});
        assert!(r.is_ok(), "must run via claim_multi, not land on the CPU lane's single-device activate: {r:?}");
        assert_eq!(live.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_hot_multi_device_instance_is_reused_not_rebuilt() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 1 * GB, live: live.clone() }));
        assert!(exec.run_blocking("int8thinker", "run", Invocation::new(), |_| {}).is_ok());
        assert!(exec.run_blocking("int8thinker", "run", Invocation::new(), |_| {}).is_ok());
        assert_eq!(exec.stats().builds, 1, "second run must reuse the hot instance, not rebuild");
        assert_eq!(exec.stats().resident_multi, 1);
        assert_eq!(live.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_unknown_model_name_still_replies_no_model_even_with_multi_models_registered() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 1 * GB, live }));
        assert!(exec.run_blocking("totally-unknown", "run", Invocation::new(), |_| {}).is_err());
    }

    /// A model whose `activate_multi` always fails.
    struct MultiBadActivate;
    impl ResidentModel for MultiBadActivate {
        fn manifest(&self) -> Manifest {
            Manifest::new("badmulti", "always fails multi activate", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("badmulti", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("not this model's contract".to_string())
        }
    }
    impl MultiDeviceResidentModel for MultiBadActivate {
        fn estimate_multi(&self, _k: &InstanceKey) -> MultiDeviceCost {
            MultiDeviceCost::new(vec![(Device::Gpu(0), 1 * GB), (Device::Gpu(1), 1 * GB)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn Instance>, String> {
            Err("checkpoint not found: /nope-multi.safetensors".to_string())
        }
    }

    /// Mirrors `activation_failure_replies_errors_and_does_not_wedge_the_executor`
    /// for the multi-device path: every queued job gets the error, BOTH
    /// devices' budgets are unwound (not just one — this is what distinguishes
    /// `build_failed_multi` from `build_failed`), and the executor stays alive.
    #[test]
    fn multi_device_activation_failure_replies_errors_and_unwinds_both_budgets() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiBadActivate));

        let (tx, rx) = channel();
        for _ in 0..3 {
            let tx = tx.clone();
            exec.submit(Job::new("badmulti", "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        }
        for _ in 0..3 {
            let r = rx.recv_timeout(Duration::from_secs(5)).expect("reply must arrive, not hang");
            let e = r.expect_err("activation failure must surface");
            assert!(e.contains("checkpoint not found"), "err: {e}");
        }

        let report = exec.residency();
        assert!(report.multi_placements.is_empty(), "failed multi-device claim must leave nothing resident");
        let gpu0 = report.budgets.iter().find(|b| b.device == Device::Gpu(0)).expect("gpu0 budget");
        let gpu1 = report.budgets.iter().find(|b| b.device == Device::Gpu(1)).expect("gpu1 budget");
        assert_eq!(gpu0.used, 0, "budget must be unwound on EVERY device, not just one");
        assert_eq!(gpu1.used, 0, "budget must be unwound on EVERY device, not just one");

        // the executor is still alive: an ordinary model runs fine afterwards
        exec.register(Arc::new(Slow { name: "good".into(), vram: GB, ms: 5, builds: Arc::new(AtomicU32::new(0)) }));
        assert!(exec.run_blocking("good", "run", Invocation::new(), |_| {}).is_ok());
    }

    /// A multi-device model whose `run` blocks until a shared gate is
    /// released — the multi-device twin of `Gated`/`GatedInst`, so a job can
    /// be held mid-flight while the test inspects `busy`/`in_flight`.
    struct MultiGated {
        name: String,
        per_gpu: u64,
        entered: Arc<AtomicU32>,
        release: Arc<AtomicBool>,
    }
    struct MultiGatedInst {
        entered: Arc<AtomicU32>,
        release: Arc<AtomicBool>,
    }
    impl ResidentModel for MultiGated {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "gated multi", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("MultiGated: single-device activate is not this model's contract".to_string())
        }
    }
    impl MultiDeviceResidentModel for MultiGated {
        fn estimate_multi(&self, _k: &InstanceKey) -> MultiDeviceCost {
            MultiDeviceCost::new(vec![(Device::Gpu(0), self.per_gpu), (Device::Gpu(1), self.per_gpu)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn Instance>, String> {
            Ok(Box::new(MultiGatedInst { entered: self.entered.clone(), release: self.release.clone() }))
        }
    }
    impl Instance for MultiGatedInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let start = Instant::now();
            while !self.release.load(Ordering::SeqCst) {
                if start.elapsed() > Duration::from_secs(5) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    /// THE busy-tracking test: while a multi-device group genuinely holds
    /// BOTH gpu0 and gpu1, a single-device model that would otherwise fit
    /// trivially (1 GB on a 24 GB card) must not even be CLAIMED, let alone
    /// run — proving `busy` really gained every device the group claimed, not
    /// just its home lane's device.
    ///
    /// Uses a SECOND `Gated` instance (not `Slow`) for the single-device job
    /// so the check is deterministic rather than a timing race: if `busy`
    /// only tracked the home device (the bug this pins), the single-device
    /// job would be claimed onto the OTHER, still-idle card and its own
    /// `entered2` counter would flip within microseconds — polling
    /// `exec.in_flight()` for a `"queued"` snapshot cannot reliably observe
    /// that (a fast wrongly-scheduled job can start AND finish inside one
    /// poll interval, so the racy version of this test silently passed
    /// against the very bug it was written to catch — confirmed once by
    /// deliberately reintroducing that bug and rerunning this exact test).
    /// Blocking `single` on its own gate makes a wrongly-early start
    /// observable no matter how fast the dispatcher reacts.
    #[test]
    fn a_running_multi_device_group_makes_every_device_it_uses_busy() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let entered = Arc::new(AtomicU32::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let entered2 = Arc::new(AtomicU32::new(0));
        let release2 = Arc::new(AtomicBool::new(false));
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiGated { name: "thinker".into(), per_gpu: 1 * GB, entered: entered.clone(), release: release.clone() }));
        exec.register(Arc::new(Gated { name: "single".into(), vram: 1 * GB, entered: entered2.clone(), release: release2.clone() }));

        let (tx, rx) = channel();
        exec.submit(Job::new("thinker", "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        let start = Instant::now();
        while entered.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "gated multi-device run never started");
            std::thread::sleep(Duration::from_millis(2));
        }

        let (stx, srx) = channel();
        exec.submit(Job::new("single", "run", Invocation::new()).reply(move |r| { let _ = stx.send(r); }));

        // Deterministic negative check over a generous window (in-process
        // scheduling reacts in microseconds either way, so 300ms is ample
        // margin, not a tight race): `single` must never enter while `thinker`
        // still holds both cards.
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            assert_eq!(entered2.load(Ordering::SeqCst), 0, "single-device job started while both cards are held by the multi-device group");
            std::thread::sleep(Duration::from_millis(5));
        }
        let single_job = exec.in_flight().into_iter().find(|j| j.model == "single").expect("single job still in flight");
        assert_eq!(single_job.phase, "queued");

        release.store(true, Ordering::SeqCst);
        assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap().is_ok());
        release2.store(true, Ordering::SeqCst);
        assert!(srx.recv_timeout(Duration::from_secs(5)).unwrap().is_ok());
    }

    #[test]
    fn evict_multi_frees_every_device_and_refuses_while_pinned() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let entered = Arc::new(AtomicU32::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let exec = Executor::start(vec![], budgets, Policy::default());
        exec.register_multi(Arc::new(MultiGated { name: "thinker".into(), per_gpu: 5 * GB, entered: entered.clone(), release: release.clone() }));

        let (tx, rx) = channel();
        exec.submit(Job::new("thinker", "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        let start = Instant::now();
        while entered.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "gated run never started");
            std::thread::sleep(Duration::from_millis(2));
        }

        let report = exec.residency();
        let key = report.multi_placements[0].key.clone();
        assert!(!exec.evict_multi(key.clone()), "must refuse to evict a PINNED (actively running) instance");

        release.store(true, Ordering::SeqCst);
        assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap().is_ok());

        // Poll until the group's Done has actually landed (unpinned) before
        // expecting a real eviction to succeed.
        let start = Instant::now();
        loop {
            if exec.evict_multi(key.clone()) {
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "evict_multi never succeeded once unpinned");
            std::thread::sleep(Duration::from_millis(2));
        }
        let report = exec.residency();
        assert!(report.multi_placements.is_empty());
        let gpu0 = report.budgets.iter().find(|b| b.device == Device::Gpu(0)).expect("gpu0 budget");
        assert_eq!(gpu0.used, 0);
    }
}
