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
/// (`phase == "queued"`) or handed to a device lane and running (`phase ==
/// "running"`). A live snapshot from [`Executor::in_flight`]; `id` is the stable
/// submit-order id and `since_ms` is the elapsed time since the job was enqueued.
#[derive(Clone, Debug)]
pub struct InFlightJob {
    pub id: u64,
    pub model: String,
    pub action: String,
    /// Coarse phase: `"queued"` (waiting in the pending queue) or `"running"`
    /// (claimed onto a device lane). Group-granular — a whole same-key group is
    /// admitted together, so all of its jobs flip to `running` at once.
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
/// errored; the lane already replied to the jobs, the manager must unwind).
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
    manifests: Arc<RwLock<Vec<Manifest>>>,
    stats: Arc<Mutex<Stats>>,
}

impl Executor {
    /// Build over a set of resident models + a policy, and start the dispatcher +
    /// one lane thread per device (GPUs + CPU).
    pub fn start(models: Vec<Arc<dyn ResidentModel>>, budgets: crate::budget::Budgets, policy: Policy) -> Executor {
        let manifests: Vec<Manifest> = models.iter().map(|m| m.manifest()).collect();
        let devices: Vec<Device> = budgets.devices().collect();
        let mut mgr = ResidencyManager::new(budgets);
        for m in models {
            mgr.register(m);
        }
        let stats = Arc::new(Mutex::new(Stats::default()));
        let (tx, rx) = channel::<Msg>();

        // One lane per device; each returns completions to the dispatcher via `tx`.
        let mut lanes: HashMap<Device, Sender<RunReq>> = HashMap::new();
        for d in devices {
            let (ltx, lrx) = channel::<RunReq>();
            let done = tx.clone();
            std::thread::Builder::new()
                .name(format!("brain-lane-{d:?}"))
                .spawn(move || lane_loop(lrx, done))
                .expect("spawn lane");
            lanes.insert(d, ltx);
        }

        let disp_stats = stats.clone();
        std::thread::Builder::new()
            .name("brain-dispatcher".into())
            .spawn(move || dispatch_loop(rx, mgr, policy, lanes, disp_stats))
            .expect("spawn dispatcher");

        Executor { tx, manifests: Arc::new(RwLock::new(manifests)), stats }
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
        self.manifests.write().unwrap().push(manifest);
        let _ = self.tx.send(Msg::Register(model));
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
        self.manifests.write().unwrap().push(manifest);
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
    /// order. Owned (not a reference) because the underlying set can grow
    /// after `start` via [`register`](Self::register) -- a snapshot is the
    /// only thing that can be handed out without holding the lock open
    /// across the caller's iteration.
    pub fn manifests(&self) -> Vec<Manifest> {
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

    loop {
        // Block for at least one message, then drain everything pending.
        match rx.recv() {
            Ok(msg) => on_msg(msg, &mut queue, &mut mgr, &mut running, &mut busy, &stats, &mut running_jobs, &mut next_id),
            Err(_) => return, // all senders gone
        }
        while let Ok(msg) = rx.try_recv() {
            on_msg(msg, &mut queue, &mut mgr, &mut running, &mut busy, &stats, &mut running_jobs, &mut next_id);
        }
        assign(&mut queue, &mut mgr, &policy, &lanes, &mut running, &mut busy, &stats, &mut running_jobs);
        if let Ok(mut s) = stats.lock() {
            s.metrics = mgr.all_metrics();
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
            mgr.register(model);
        }
        Msg::RegisterMulti(model) => {
            mgr.register_multi(model);
        }
        Msg::Built { key, handle } => {
            mgr.adopt(&key, handle);
        }
        Msg::BuiltMulti { key, handle } => {
            mgr.adopt_multi(&key, handle);
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
                    phase: "running".to_string(),
                    since_ms: now.saturating_duration_since(r.enqueued).as_millis() as u64,
                });
            }
            jobs.sort_by_key(|j| j.id); // stable, submit-order
            let _ = tx.send(jobs);
        }
        Msg::Done { key, device, batch, failed } => {
            if failed {
                mgr.build_failed(&key); // unwind budget + slot; jobs already failed
            } else {
                mgr.release(&key);
            }
            running.remove(&key);
            running_jobs.retain(|r| r.key != key); // group finished — drop its jobs
            busy.remove(&device);
            let mut s = stats.lock().unwrap();
            s.batches += 1;
            s.jobs += batch as u64;
            s.max_batch = s.max_batch.max(batch);
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
        }
        Msg::DoneMulti { key, devices, batch, failed } => {
            if failed {
                mgr.build_failed_multi(&key); // unwind budget on EVERY device; jobs already failed
            } else {
                mgr.release_multi(&key);
            }
            running.remove(&key);
            running_jobs.retain(|r| r.key != key);
            for d in devices {
                busy.remove(&d);
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
                ClaimOutcome::Single(Claimed::Hot(_), _) => mgr.release(&ckey),
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
        // (they've left `queue`); cleared on this group's `Done`.
        for j in jobs.iter() {
            running_jobs.push(RunningJob { id: j.id, model: j.model.clone(), action: j.action.clone(), key: ckey.clone(), enqueued: j.enqueued });
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
/// `run_group`) knows which shape to send without re-deriving it from
/// anything guessable (e.g. `devices.len() == 1`, which a genuine
/// single-device multi-device instance would make ambiguous).
enum Ran {
    Single(Device),
    Multi(Vec<Device>),
}

fn lane_loop(rx: Receiver<RunReq>, done: Sender<Msg>) {
    while let Ok(req) = rx.recv() {
        let batch = req.jobs.len();
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
                    let _ = done.send(Msg::Done { key, device, batch, failed: true });
                    continue;
                }
            },
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
                    let _ = done.send(Msg::DoneMulti { key, devices, batch, failed: true });
                    continue;
                }
            },
            // Deferred promotion, same reasoning as `Build`'s deferred
            // activate: rebuilding a demoted instance's device buffers can
            // be slow, so it happens on this device's own lane, never the
            // dispatcher thread. The existing `Instance` is reused in place.
            Claimed::Promote(h) => {
                let result = h.lock().unwrap().promote(req.device);
                match result {
                    Ok(()) => {
                        let _ = done.send(Msg::Built { key: req.key.clone(), handle: h.clone() });
                        h
                    }
                    Err(e) => {
                        let msg = format!("promote {}: {e}", req.key);
                        for p in req.jobs {
                            (p.reply)(Err(msg.clone()));
                        }
                        let _ = done.send(Msg::Done { key: req.key, device: req.device, batch, failed: true });
                        continue;
                    }
                }
            }
        };
        // `run_group` is one implementation shared by both shapes above — a
        // multi-device instance is just an `Instance` whose forward happens to
        // span devices internally; nothing here needs to know that.
        run_group(&handle, &action, jobs);
        let done_msg = match ran {
            Ran::Single(device) => Msg::Done { key, device, batch, failed: false },
            Ran::Multi(devices) => Msg::DoneMulti { key, devices, batch, failed: false },
        };
        let _ = done.send(done_msg);
    }
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
        let names: Vec<String> = exec.manifests().into_iter().map(|m| m.model).collect();
        assert_eq!(names, vec!["late".to_string()]);

        // And it is schedulable: a Submit enqueued right after register()
        // returns finds it, because both share one FIFO channel.
        let elapsed = submit_wait(&exec, &["late"]);
        assert!(elapsed < Duration::from_secs(2), "registered model never ran in time: {elapsed:?}");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
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
