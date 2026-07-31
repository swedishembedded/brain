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
use std::sync::{Arc, Mutex};
use std::time::Instant;

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::manager::{ClaimError, Claimed, InstanceHandle};
use crate::scheduler::{choose_next, Group, Policy};
use crate::{Device, InstanceKey, ResidencyManager, ResidentModel};

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
    model: String,
    action: String,
    inv: Invocation,
    key: InstanceKey,
    enqueued: Instant,
    on_progress: Box<dyn FnMut(Progress) + Send>,
    reply: Box<dyn FnOnce(ActionResult) + Send>,
    on_admit: Option<Box<dyn FnOnce() + Send>>,
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
    pub queue_peak: usize,
    /// Deepest observed number of lanes running at once (device-level parallelism).
    pub max_parallel: usize,
}

/// A message to the dispatcher: a new job, a lane adopting a freshly built
/// instance, or a lane finishing a group (`failed` = the deferred activate
/// errored; the lane already replied to the jobs, the manager must unwind).
enum Msg {
    Submit(Box<Job>),
    Built { key: InstanceKey, handle: InstanceHandle },
    Done { key: InstanceKey, device: Device, batch: usize, failed: bool },
    /// A stats query: the dispatcher (the sole owner of the [`ResidencyManager`])
    /// replies with a residency + budget snapshot. Mirrors how [`Stats`] is
    /// exposed, but read straight from the manager rather than the counters.
    Report(Sender<crate::ResidencyReport>),
}

/// A group of same-key jobs handed to a device lane to run. `work` is either a
/// hot handle or a deferred build the LANE performs — activation (weight load,
/// NPU graph compile) can take seconds or hang, and on the dispatcher thread
/// that froze every model on the server; on the lane it can only stall its own
/// device.
struct RunReq {
    work: Claimed,
    action: String,
    jobs: Vec<Pending>,
    key: InstanceKey,
    device: Device,
}

/// Cheap-to-clone submission handle (many front-ends can submit concurrently).
#[derive(Clone)]
pub struct Executor {
    tx: Sender<Msg>,
    manifests: Arc<Vec<Manifest>>,
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

        Executor { tx, manifests: Arc::new(manifests), stats }
    }

    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(Msg::Submit(Box::new(job)));
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

    pub fn manifests(&self) -> &[Manifest] {
        &self.manifests
    }
}

// ---------------------------------------------------------------- dispatcher

fn dispatch_loop(rx: Receiver<Msg>, mut mgr: ResidencyManager, policy: Policy, lanes: HashMap<Device, Sender<RunReq>>, stats: Arc<Mutex<Stats>>) {
    let mut queue: Vec<Pending> = Vec::new();
    let mut running: HashSet<InstanceKey> = HashSet::new();
    let mut busy: HashSet<Device> = HashSet::new();

    loop {
        // Block for at least one message, then drain everything pending.
        match rx.recv() {
            Ok(msg) => on_msg(msg, &mut queue, &mut mgr, &mut running, &mut busy, &stats),
            Err(_) => return, // all senders gone
        }
        while let Ok(msg) = rx.try_recv() {
            on_msg(msg, &mut queue, &mut mgr, &mut running, &mut busy, &stats);
        }
        assign(&mut queue, &mut mgr, &policy, &lanes, &mut running, &mut busy, &stats);
    }
}

fn on_msg(msg: Msg, queue: &mut Vec<Pending>, mgr: &mut ResidencyManager, running: &mut HashSet<InstanceKey>, busy: &mut HashSet<Device>, stats: &Arc<Mutex<Stats>>) {
    match msg {
        Msg::Submit(job) => match mgr.instance_key_for(&job.model, &job.action, &job.inv) {
            Some(key) => {
                queue.push(Pending {
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
            }
            None => (job.reply)(Err(format!("no model '{}'", job.model))),
        },
        Msg::Built { key, handle } => {
            mgr.adopt(&key, handle);
        }
        Msg::Report(tx) => {
            let _ = tx.send(mgr.report());
        }
        Msg::Done { key, device, batch, failed } => {
            if failed {
                mgr.build_failed(&key); // unwind budget + slot; jobs already failed
            } else {
                mgr.release(&key);
            }
            running.remove(&key);
            busy.remove(&device);
            let mut s = stats.lock().unwrap();
            s.batches += 1;
            s.jobs += batch as u64;
            s.max_batch = s.max_batch.max(batch);
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
        }
    }
}

/// Assign as many runnable groups as there are free device lanes.
fn assign(queue: &mut Vec<Pending>, mgr: &mut ResidencyManager, policy: &Policy, lanes: &HashMap<Device, Sender<RunReq>>, running: &mut HashSet<InstanceKey>, busy: &mut HashSet<Device>, stats: &Arc<Mutex<Stats>>) {
    loop {
        // Groups whose key is not already running AND that can be placed on some
        // non-busy device (the scheduler policy then picks among them).
        let now = Instant::now();
        let rows = group_rows(queue, now, running);
        let placeable: Vec<Group> = rows
            .iter()
            .filter(|r| mgr.placeable(&r.key, &r.model, busy))
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

        // Claim (promote/evict, pin) on a non-busy device.
        let first = queue.iter().find(|p| p.key == key && p.action == action).map(|p| p.inv.clone());
        let inv = match first {
            Some(i) => i,
            None => break,
        };
        let (work, device, ckey) = match mgr.claim(&model, &action, &inv, busy) {
            Ok(c) => c,
            Err(ClaimError::NoCapacity(_)) => break, // wait for a lane to free a device
            Err(ClaimError::Activate(e)) => {
                // Permanent for this group: FAIL its queued jobs now and keep
                // scheduling the others. (The old code broke out of the whole
                // round here — the jobs waited forever and every other group
                // starved behind them.)
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
        running.insert(ckey.clone());
        busy.insert(device);

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

        // Sync residency counters immediately — a claim may have built/evicted, and
        // that must be visible before the lane's Done (which lags the actual run).
        if let Ok(mut s) = stats.lock() {
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
            s.max_parallel = s.max_parallel.max(busy.len());
        }
        // Hand to the device's lane (which builds the instance first when cold).
        let _ = lanes[&device].send(RunReq { work, action, jobs, key: ckey, device });
    }
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

fn lane_loop(rx: Receiver<RunReq>, done: Sender<Msg>) {
    while let Ok(req) = rx.recv() {
        let batch = req.jobs.len();
        let handle = match req.work {
            Claimed::Hot(h) => h,
            // Deferred activation happens HERE, on the device's own lane: a slow
            // or wedged build stalls this device only, never the dispatcher.
            Claimed::Build(m) => match m.activate(&req.key, req.device) {
                Ok(inst) => {
                    let h: InstanceHandle = Arc::new(Mutex::new(inst));
                    let _ = done.send(Msg::Built { key: req.key.clone(), handle: h.clone() });
                    h
                }
                Err(e) => {
                    // Activation failed: every job in the group gets the error
                    // (never silence), and the dispatcher unwinds the claim.
                    let msg = format!("activate {}: {e}", req.key);
                    for p in req.jobs {
                        (p.reply)(Err(msg.clone()));
                    }
                    let _ = done.send(Msg::Done { key: req.key, device: req.device, batch, failed: true });
                    continue;
                }
            },
        };
        run_group(&handle, &req.action, req.jobs);
        let _ = done.send(Msg::Done { key: req.key, device: req.device, batch, failed: false });
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
    use crate::{Instance, MemCost};
    use capability::{ActionResult, ActionSpec, Blob, Manifest, Media, Outcome};
    use std::sync::atomic::{AtomicU32, Ordering};
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
}
