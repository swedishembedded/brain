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

use crate::manager::InstanceHandle;
use crate::scheduler::{choose_next, Group, Policy};
use crate::{Device, InstanceKey, ResidencyManager, ResidentModel};

/// One unit of work. `on_progress`/`reply` are callbacks (no async dependency here).
pub struct Job {
    pub model: String,
    pub action: String,
    pub inv: Invocation,
    pub on_progress: Box<dyn FnMut(Progress) + Send>,
    pub reply: Box<dyn FnOnce(ActionResult) + Send>,
}

struct Pending {
    model: String,
    action: String,
    inv: Invocation,
    key: InstanceKey,
    enqueued: Instant,
    on_progress: Box<dyn FnMut(Progress) + Send>,
    reply: Box<dyn FnOnce(ActionResult) + Send>,
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

/// A message to the dispatcher: a new job, or a lane finishing a group.
enum Msg {
    Submit(Box<Job>),
    Done { key: InstanceKey, device: Device, batch: usize },
}

/// A group of same-key jobs handed to a device lane to run.
struct RunReq {
    handle: InstanceHandle,
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
        self.submit(Job {
            model: model.into(),
            action: action.into(),
            inv,
            on_progress: Box::new(on_progress),
            reply: Box::new(move |r| {
                let _ = rtx.send(r);
            }),
        });
        rrx.recv().unwrap_or_else(|_| Err("executor worker gone".into()))
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
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
                });
                let mut s = stats.lock().unwrap();
                s.queue_peak = s.queue_peak.max(queue.len());
            }
            None => (job.reply)(Err(format!("no model '{}'", job.model))),
        },
        Msg::Done { key, device, batch } => {
            mgr.release(&key);
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
        // `placeable`'s ids index into itself; map back to the row it came from.
        let chosen = placeable[gid].id; // == the row index in `rows`
        let (key, model, action) = (rows[chosen].key.clone(), rows[chosen].model.clone(), rows[chosen].action.clone());

        // Claim (promote/evict, pin) on a non-busy device.
        let first = queue.iter().find(|p| p.key == key && p.action == action).map(|p| p.inv.clone());
        let inv = match first {
            Some(i) => i,
            None => break,
        };
        let (handle, device, ckey) = match mgr.claim(&model, &action, &inv, busy) {
            Ok(c) => c,
            Err(_) => break, // could not place on a free device this round
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

        // Sync residency counters immediately — a claim may have built/evicted, and
        // that must be visible before the lane's Done (which lags the actual run).
        if let Ok(mut s) = stats.lock() {
            s.builds = mgr.builds;
            s.evictions = mgr.evictions;
            s.resident = mgr.resident_count();
            s.max_parallel = s.max_parallel.max(busy.len());
        }
        // Hand to the device's lane.
        let _ = lanes[&device].send(RunReq { handle, action, jobs, key: ckey, device });
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
        run_group(&req.handle, &req.action, req.jobs);
        let _ = done.send(Msg::Done { key: req.key, device: req.device, batch });
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
        let mut fanout = |pr: Progress| {
            for s in sinks.iter_mut() {
                s(pr.clone());
            }
        };
        inst.run_batch(action, &invs, &mut fanout)
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
            exec.submit(Job { model: (*m).into(), action: "run".into(), inv: Invocation::new(), on_progress: Box::new(|_| {}), reply: Box::new(move |r| { let _ = tx.send(r); }) });
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
    fn unknown_model_replies_error() {
        let exec = Executor::start(vec![], Budgets::new(), Policy::default());
        assert!(exec.run_blocking("nope", "x", Invocation::new(), |_| {}).is_err());
    }
}
