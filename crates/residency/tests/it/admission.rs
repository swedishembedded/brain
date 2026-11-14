// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sharing accelerators between
// inference traffic and training runs without starving either for its
// clients. If your team needs expertise in admission control on shared GPUs
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The admission contract through a live executor (`residency::contract`):
//! every queued job is answered by its deadline, wherever it waits, and a
//! workload is given a device under continuous serving traffic.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::failure::{classify, FailureClass};
use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::contract::{Contract, QUEUE_DEADLINE_FLAG};
use residency::{Device, Executor, Instance, InstanceKey, Job, JobFeed, MemCost, Policy, ResidentModel};

const MB: u64 = 1 << 20;

/// A model whose single run blocks until the test releases it.
struct Holder {
    name: &'static str,
    gate: Arc<Mutex<Receiver<()>>>,
    vram: u64,
    batch_limit: Option<usize>,
}

struct HolderInst {
    gate: Arc<Mutex<Receiver<()>>>,
}

impl ResidentModel for Holder {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name, "holds its device", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.name, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(self.vram, MB)
    }
    fn batch_limit(&self) -> Option<usize> {
        self.batch_limit
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(HolderInst { gate: self.gate.clone() }))
    }
}

impl Instance for HolderInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let _ = self.gate.lock().unwrap().recv_timeout(Duration::from_secs(30));
        Ok(Outcome::new())
    }
    /// An open group that never takes a late job while its first one is held:
    /// a late job waits in the inbox for as long as the test keeps the gate.
    fn run_open(&mut self, _a: &str, invs: &[Invocation], _p: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        let _ = self.gate.lock().unwrap().recv_timeout(Duration::from_secs(30));
        for i in 0..invs.len() {
            feed.finish(i, Ok(Outcome::new()));
        }
        let mut next = invs.len();
        loop {
            let late = feed.take(1);
            if late.is_empty() && feed.close() {
                return true;
            }
            for _ in late {
                feed.finish(next, Ok(Outcome::new()));
                next += 1;
            }
        }
    }
}

fn on_cpu() -> Budgets {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    budgets
}

fn policy(contract: Contract) -> Policy {
    Policy { admission: contract, ..Policy::default() }
}

fn submit(exec: &Executor, job: Job) -> Receiver<ActionResult> {
    let (tx, rx) = channel();
    exec.submit(job.reply(move |r| {
        let _ = tx.send(r);
    }));
    rx
}

fn wait_running(exec: &Executor, model: &str) {
    let until = Instant::now() + Duration::from_secs(10);
    while exec.in_flight().iter().all(|j| j.model != model || j.phase != "running") {
        assert!(Instant::now() < until, "{model} never started");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A job that waits in the queue past its deadline is answered then, with an
/// error a caller can classify and that names the setting, while the device
/// it waited for is still busy. A caller's own earlier deadline applies too.
#[test]
fn a_queued_job_is_refused_at_its_deadline_naming_the_setting() {
    let (release, gate) = channel();
    let gate = Arc::new(Mutex::new(gate));
    let models: Vec<Arc<dyn ResidentModel>> = vec![
        Arc::new(Holder { name: "holder", gate: gate.clone(), vram: 0, batch_limit: None }),
        Arc::new(Holder { name: "late", gate, vram: 0, batch_limit: None }),
    ];
    let contract = Contract { queue_deadline: Duration::from_millis(300), ..Contract::default() };
    let exec = Executor::start(models, on_cpu(), policy(contract));
    let holding = submit(&exec, Job::new("holder", "run", Invocation::new()));
    wait_running(&exec, "holder");

    let t0 = Instant::now();
    let own = submit(&exec, Job::new("late", "run", Invocation::new()).deadline(t0 + Duration::from_millis(100)));
    let operators = submit(&exec, Job::new("late", "run", Invocation::new()));

    let own = own.recv_timeout(Duration::from_secs(5)).expect("the caller's deadline is honoured inside the executor").unwrap_err();
    assert!(own.contains("caller"), "{own}");
    let operators = operators.recv_timeout(Duration::from_secs(5)).expect("the operator's deadline is honoured inside the executor").unwrap_err();
    let waited = t0.elapsed();
    assert!(operators.contains(QUEUE_DEADLINE_FLAG), "the refusal names the setting: {operators}");
    assert_eq!(classify(&operators).map(|(class, _)| class), Some(FailureClass::Timeout), "{operators}");
    assert!(waited >= Duration::from_millis(300), "refused before its deadline: {waited:?}");
    assert!(waited < Duration::from_millis(1500), "answered at its deadline, not at a later idle tick: {waited:?}");
    assert!(holding.try_recv().is_err(), "the device was busy the whole time");

    release.send(()).unwrap();
    holding.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(exec.stats().queue_depth, 0);
    exec.shutdown();
}

/// A job a running group has accepted into its inbox but not taken has no
/// slot yet: it is answered at its deadline like a job in the queue, and the
/// group never runs it afterwards.
#[test]
fn a_job_waiting_in_a_running_groups_inbox_is_refused_at_its_deadline() {
    let (release, gate) = channel();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Holder { name: "open", gate: Arc::new(Mutex::new(gate)), vram: 0, batch_limit: Some(1) })];
    let contract = Contract { queue_deadline: Duration::from_millis(300), ..Contract::default() };
    let exec = Executor::start(models, on_cpu(), policy(contract));
    let first = submit(&exec, Job::new("open", "run", Invocation::new()));
    wait_running(&exec, "open");
    let admitted = Arc::new(AtomicBool::new(false));
    let seen = admitted.clone();
    let late = submit(&exec, Job::new("open", "run", Invocation::new()).on_admit(move || seen.store(true, Ordering::SeqCst)));

    let refused = late.recv_timeout(Duration::from_secs(5)).expect("a job in the inbox is answered at its deadline").unwrap_err();
    assert!(refused.contains(QUEUE_DEADLINE_FLAG), "{refused}");
    assert!(!admitted.load(Ordering::SeqCst), "it was never admitted");
    assert!(first.try_recv().is_err(), "the group still runs");

    release.send(()).unwrap();
    first.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while !exec.in_flight().is_empty() {
        assert!(Instant::now() < until, "the refused job is still held: {:?}", exec.in_flight());
        std::thread::sleep(Duration::from_millis(5));
    }
    let stats = exec.stats();
    assert_eq!((stats.admitted, stats.expired), (1, 1), "the expired job is counted expired, not admitted");
    exec.shutdown();
}

/// A serving model that decodes `slots` requests at once and finishes one
/// every `step`, taking late requests between steps.
struct Stream {
    step: Duration,
    slots: usize,
}

struct StreamInst {
    step: Duration,
    slots: usize,
}

impl ResidentModel for Stream {
    fn manifest(&self) -> Manifest {
        Manifest::new("serve", "continuous batching", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("serve", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(600 * MB, 0)
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.slots)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(StreamInst { step: self.step, slots: self.slots }))
    }
}

impl Instance for StreamInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        std::thread::sleep(self.step);
        Ok(Outcome::new())
    }
    fn run_open(&mut self, _a: &str, invs: &[Invocation], _p: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        let mut live: Vec<usize> = (0..invs.len()).collect();
        let mut next = invs.len();
        loop {
            for _ in feed.take(self.slots.saturating_sub(live.len())) {
                live.push(next);
                next += 1;
            }
            if live.is_empty() {
                if feed.close() {
                    return true;
                }
                continue;
            }
            std::thread::sleep(self.step);
            feed.finish(live.remove(0), Ok(Outcome::new()));
        }
    }
}

/// A fine-tune: it holds the whole card while it runs.
struct Train {
    runs: Arc<AtomicU32>,
}

struct TrainInst(Arc<AtomicU32>);

impl ResidentModel for Train {
    fn manifest(&self) -> Manifest {
        Manifest::new("train", "a workload", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("train", "run")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(600 * MB, 0)
    }
    fn transient(&self) -> bool {
        true
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainInst(self.runs.clone())))
    }
}

impl Instance for TrainInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(50));
        Ok(Outcome::new())
    }
}

/// Requests arrive faster than the card serves them, so the serving model
/// always has a full batch queued and a scheduler that favours batches never
/// picks the workload on its own. The contract still gives the workload the
/// card once it has waited its guaranteed time, and serving resumes after it.
#[test]
fn a_workload_gets_the_card_under_continuous_serving_traffic() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 1 << 30, 0);
    let runs = Arc::new(AtomicU32::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Stream { step: Duration::from_millis(4), slots: 2 }), Arc::new(Train { runs: runs.clone() })];
    // Age never outweighs a batch and nothing is ever overdue: without the
    // contract the serving batch wins every round.
    let scheduler = Policy { age_weight_per_ms: 0.0, max_wait_ms: u64::MAX, ..Policy::default() };
    let contract = Contract { workload_max_wait: Duration::from_millis(100), ..Contract::default() };
    let exec = Executor::start(models, budgets, Policy { admission: contract, ..scheduler });

    let traffic_for = Duration::from_millis(1500);
    let (served_tx, served_rx): (Sender<ActionResult>, Receiver<ActionResult>) = channel();
    let sent = Arc::new(AtomicU32::new(0));
    let traffic = {
        let (exec, sent) = (exec.clone(), sent.clone());
        std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < traffic_for {
                let tx = served_tx.clone();
                exec.submit(Job::new("serve", "run", Invocation::new()).reply(move |r| {
                    let _ = tx.send(r);
                }));
                sent.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };
    wait_running(&exec, "serve");

    let (admit_tx, admit_rx) = channel();
    let submitted = Instant::now();
    let trained = submit(&exec, Job::new("train", "run", Invocation::new()).on_admit(move || {
        let _ = admit_tx.send(Instant::now());
    }));
    let admitted_at = admit_rx.recv_timeout(Duration::from_secs(30)).expect("the workload is admitted");
    let waited = admitted_at.duration_since(submitted);
    assert!(waited < Duration::from_millis(1000), "the workload waited {waited:?} under serving traffic that lasted {traffic_for:?}");
    trained.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 1);

    traffic.join().unwrap();
    let total = sent.load(Ordering::SeqCst);
    for i in 0..total {
        served_rx.recv_timeout(Duration::from_secs(30)).unwrap_or_else(|_| panic!("serving request {i} of {total} never answered")).unwrap();
    }
    exec.shutdown();
}
