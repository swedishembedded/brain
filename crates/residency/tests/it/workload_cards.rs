// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sharing a server's accelerators
// between the models it serves and the training jobs it runs for its clients.
// If your team needs expertise in scheduling work across GPUs then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A workload (a job that holds cards only while it runs, such as a fine-tune) through the executor: what it is
//! charged while it runs and what it gives back on every way it ends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, Job, MemCost, Policy, ResidencyReport, ResidentModel};
use serde_json::json;

const GIB: u64 = 1 << 30;

/// The jobs a [`Workload`] or [`Served`] instance runs block on a gate, so a test sees what a running job holds.
#[derive(Clone)]
struct Gate {
    open: Arc<Mutex<Receiver<()>>>,
    running: Arc<AtomicBool>,
}

impl Gate {
    fn new() -> (Gate, Sender<()>) {
        let (tx, rx) = channel();
        (Gate { open: Arc::new(Mutex::new(rx)), running: Arc::new(AtomicBool::new(false)) }, tx)
    }

    /// What a job asked to `wait` does: hold its cards until the test lets it go.
    fn hold(&self, inv: &Invocation) {
        if inv.get_str("wait").as_deref() == Some("yes") {
            self.running.store(true, Ordering::SeqCst);
            let _ = self.open.lock().unwrap().recv_timeout(Duration::from_secs(60));
            self.running.store(false, Ordering::SeqCst);
        }
    }

    fn wait_running(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.running.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the gated job never started");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// A training-like workload: `per_card` bytes on each card it runs on, or host work when `host` is set. A job
/// asks for `"fail": "run"` to fail its run and `"fail": "build"` to fail its build.
struct Workload {
    per_card: u64,
    host: bool,
    gate: Gate,
    /// The cards each build was given, in order.
    built: Arc<Mutex<Vec<Vec<Device>>>>,
}

struct WorkloadRun {
    gate: Gate,
}

impl Instance for WorkloadRun {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.gate.hold(inv);
        match inv.get_str("fail").as_deref() {
            Some("run") => Err("the run failed".to_string()),
            _ => Ok(Outcome::new()),
        }
    }
}

impl ResidentModel for Workload {
    fn manifest(&self) -> Manifest {
        Manifest::new("work", "a workload", vec![ActionSpec::new("train", "train")])
    }
    fn instance_key(&self, _a: &str, inv: &Invocation) -> InstanceKey {
        let fail = inv.get_str("fail").unwrap_or_default();
        let cards = inv.params.get("cards").and_then(serde_json::Value::as_u64).unwrap_or(1);
        InstanceKey::new("work", format!("fail={fail};cards={cards}"))
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        match self.host {
            true => MemCost::host_work(self.per_card),
            false => MemCost::new(self.per_card, 0).with_activation(0),
        }
    }
    fn replicas(&self) -> usize {
        4
    }
    fn transient(&self) -> bool {
        true
    }
    fn cards(&self, key: &InstanceKey) -> usize {
        key.base().config.rsplit_once("cards=").and_then(|(_, n)| n.parse().ok()).unwrap_or(1)
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        self.activate_cards(key, &[device])
    }
    fn activate_cards(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        self.built.lock().unwrap().push(devices.to_vec());
        if key.base().config.starts_with("fail=build") {
            return Err("the build failed".to_string());
        }
        Ok(Box::new(WorkloadRun { gate: self.gate.clone() }))
    }
}

/// A served model holding `bytes` of one card while resident.
struct Served {
    name: &'static str,
    bytes: u64,
    gate: Gate,
}

impl ResidentModel for Served {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name, "a served model", vec![ActionSpec::new("generate", "generate")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.name, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(self.bytes, 0).with_activation(0)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(WorkloadRun { gate: self.gate.clone() }))
    }
}

fn gpus(n: u32, bytes: u64) -> Budgets {
    let mut budgets = Budgets::new();
    for i in 0..n {
        budgets.set(Device::Gpu(i), bytes, 0);
    }
    budgets
}

struct Rig {
    exec: Executor,
    built: Arc<Mutex<Vec<Vec<Device>>>>,
    gate: Gate,
    release: Sender<()>,
}

fn rig(per_card: u64, host: bool, budgets: Budgets, served: Vec<Arc<dyn ResidentModel>>) -> Rig {
    let (gate, release) = Gate::new();
    let built = Arc::new(Mutex::new(Vec::new()));
    let mut models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Workload { per_card, host, gate: gate.clone(), built: built.clone() })];
    models.extend(served);
    Rig { exec: Executor::start(models, budgets, Policy::default()), built, gate, release }
}

fn submit(exec: &Executor, model: &str, action: &str, params: serde_json::Value) -> Receiver<ActionResult> {
    let mut inv = Invocation::new();
    inv.params = params;
    let (tx, rx) = channel();
    exec.submit(Job::new(model, action, inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx
}

fn train(exec: &Executor, params: serde_json::Value) -> Receiver<ActionResult> {
    submit(exec, "work", "train", params)
}

fn answer(rx: &Receiver<ActionResult>) -> ActionResult {
    rx.recv_timeout(Duration::from_secs(30)).expect("the job was answered")
}

/// Bytes charged on each card, in card order.
fn card_use(report: &ResidencyReport) -> Vec<u64> {
    let mut cards: Vec<(u32, u64)> = report.budgets.iter().filter_map(|b| if let Device::Gpu(i) = b.device { Some((i, b.used)) } else { None }).collect();
    cards.sort_unstable();
    cards.into_iter().map(|(_, used)| used).collect()
}

/// Waits (bounded) until the residency report satisfies `settled`: a job's reply comes before the dispatcher has
/// processed its end.
fn wait_for(exec: &Executor, settled: impl Fn(&ResidencyReport) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let report = exec.residency();
        if settled(&report) {
            return;
        }
        assert!(Instant::now() < deadline, "the residency never settled: {report:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Waits (bounded) until no card or host byte is charged and nothing is placed: a job's reply comes before the
/// dispatcher has processed its end.
fn assert_all_returned(exec: &Executor) {
    wait_for(exec, |r| r.placements.is_empty() && r.multi_placements.is_empty() && r.budgets.iter().all(|b| b.used == 0));
}

#[test]
fn a_one_card_workload_holds_one_card_while_it_runs_and_gives_it_back_however_it_ends() {
    let rig = rig(20 * GIB, false, gpus(2, 24 * GIB), Vec::new());
    let done = train(&rig.exec, json!({"wait": "yes"}));
    rig.gate.wait_running();
    let mut used = card_use(&rig.exec.residency());
    used.sort_unstable();
    assert_eq!(used, vec![0, 20 * GIB], "one card is charged the run's footprint");
    rig.release.send(()).unwrap();
    answer(&done).expect("the run");
    assert_all_returned(&rig.exec);

    assert_eq!(answer(&train(&rig.exec, json!({"fail": "run"}))).unwrap_err(), "the run failed");
    assert_all_returned(&rig.exec);
    assert!(answer(&train(&rig.exec, json!({"fail": "build"}))).unwrap_err().contains("the build failed"));
    assert_all_returned(&rig.exec);
    assert!(rig.built.lock().unwrap().iter().all(|cards| cards.len() == 1), "every build was given one card");
}

#[test]
fn host_work_is_charged_to_the_host_and_never_to_a_card() {
    let mut budgets = gpus(1, 24 * GIB);
    budgets.set(Device::Cpu, 8 * GIB, 0);
    let rig = rig(2 * GIB, true, budgets, Vec::new());
    let done = train(&rig.exec, json!({"wait": "yes"}));
    rig.gate.wait_running();
    let report = rig.exec.residency();
    assert_eq!(card_use(&report), vec![0]);
    assert_eq!(report.budgets.iter().find(|b| b.device == Device::Cpu).unwrap().used, 2 * GIB);
    rig.release.send(()).unwrap();
    answer(&done).expect("the host work");
    assert_all_returned(&rig.exec);
    assert_eq!(*rig.built.lock().unwrap(), vec![vec![Device::Cpu]]);
}

#[test]
fn a_workload_no_card_can_hold_is_refused_at_once_and_evicts_nothing() {
    let (idle, _keep) = Gate::new();
    let served: Arc<dyn ResidentModel> = Arc::new(Served { name: "served", bytes: 10 * GIB, gate: idle });
    let rig = rig(30 * GIB, false, gpus(2, 24 * GIB), vec![served]);
    answer(&submit(&rig.exec, "served", "generate", json!({}))).expect("the served model runs");
    let refused = answer(&train(&rig.exec, json!({}))).unwrap_err();
    assert!(refused.contains("exceeds every device"), "{refused}");
    assert_eq!(rig.exec.residency().placements.len(), 1, "the idle served model was not evicted for a run that cannot fit");
    assert!(rig.built.lock().unwrap().is_empty());
}

#[test]
fn a_workload_on_several_cards_is_built_across_that_many_distinct_cards_charged_on_each_and_gives_all_back() {
    let (busy, release_served) = Gate::new();
    let served: Arc<dyn ResidentModel> = Arc::new(Served { name: "served", bytes: 20 * GIB, gate: busy.clone() });
    let rig = rig(20 * GIB, false, gpus(3, 24 * GIB), vec![served]);
    let done = train(&rig.exec, json!({"cards": 2, "wait": "yes"}));
    rig.gate.wait_running();
    let mut used = card_use(&rig.exec.residency());
    used.sort_unstable();
    assert_eq!(used, vec![0, 20 * GIB, 20 * GIB], "two cards are charged the run's footprint each");
    let cards = rig.built.lock().unwrap()[0].clone();
    assert_eq!(cards.len(), 2);
    assert_ne!(cards[0], cards[1], "two distinct cards");

    // The run holds both cards: a served model needing a whole card goes to the third.
    let serving = submit(&rig.exec, "served", "generate", json!({"wait": "yes"}));
    busy.wait_running();
    let placed = rig.exec.residency().placements;
    assert_eq!(placed.len(), 1);
    assert!(!cards.contains(&placed[0].device), "served on {:?} while the run holds {cards:?}", placed[0].device);
    release_served.send(()).unwrap();
    answer(&serving).expect("served");

    rig.release.send(()).unwrap();
    answer(&done).expect("the run");
    // Only the idle served model is left charged: the finished run is not kept resident.
    let only_served = |r: &ResidencyReport| r.multi_placements.is_empty() && card_use(r).iter().filter(|&&b| b > 0).count() == 1;
    wait_for(&rig.exec, only_served);

    assert_eq!(answer(&train(&rig.exec, json!({"cards": 2, "fail": "run"}))).unwrap_err(), "the run failed");
    wait_for(&rig.exec, only_served);
    assert!(answer(&train(&rig.exec, json!({"cards": 3, "fail": "build"}))).unwrap_err().contains("the build failed"));
    wait_for(&rig.exec, |r| r.multi_placements.is_empty() && r.budgets.iter().all(|b| b.used == 0));
    assert_eq!(rig.built.lock().unwrap().last().unwrap().len(), 3, "a three-card run evicted the idle served model for its third card");
}

#[test]
fn a_workload_needing_more_cards_than_can_ever_hold_it_is_refused_at_once() {
    let rig = rig(20 * GIB, false, gpus(2, 24 * GIB), Vec::new());
    let refused = answer(&train(&rig.exec, json!({"cards": 3}))).unwrap_err();
    assert!(refused.contains("3 cards") && refused.contains("2 card"), "{refused}");

    let mut uneven = gpus(1, 24 * GIB);
    uneven.set(Device::Gpu(1), 16 * GIB, 0);
    let rig = rig_with_budgets(20 * GIB, uneven);
    let refused = answer(&train(&rig.exec, json!({"cards": 2}))).unwrap_err();
    assert!(refused.contains("2 cards") && refused.contains("1 card"), "{refused}");
    assert!(rig.built.lock().unwrap().is_empty());
    assert_all_returned(&rig.exec);
}

fn rig_with_budgets(per_card: u64, budgets: Budgets) -> Rig {
    rig(per_card, false, budgets, Vec::new())
}

#[test]
fn only_idle_residents_give_way_and_none_is_evicted_while_another_card_is_still_busy() {
    let (idle_gate, _never) = Gate::new();
    let (busy_gate, release_busy) = Gate::new();
    let idle: Arc<dyn ResidentModel> = Arc::new(Served { name: "idle", bytes: 20 * GIB, gate: idle_gate });
    let busy: Arc<dyn ResidentModel> = Arc::new(Served { name: "busy", bytes: 20 * GIB, gate: busy_gate.clone() });
    let rig = rig(20 * GIB, false, gpus(2, 24 * GIB), vec![idle, busy]);
    answer(&submit(&rig.exec, "idle", "generate", json!({}))).expect("idle model built");
    let serving = submit(&rig.exec, "busy", "generate", json!({"wait": "yes"}));
    busy_gate.wait_running();
    assert_eq!(rig.exec.residency().placements.len(), 2, "one served model on each card");

    let done = train(&rig.exec, json!({"cards": 2}));
    std::thread::sleep(Duration::from_millis(500));
    assert!(done.try_recv().is_err(), "the run cannot start while a card is busy");
    assert_eq!(rig.exec.residency().placements.len(), 2, "the idle model was not evicted for a claim that cannot complete yet");
    assert!(rig.built.lock().unwrap().is_empty());

    release_busy.send(()).unwrap();
    answer(&serving).expect("served");
    answer(&done).expect("the run starts once both cards can be freed");
    assert_eq!(rig.built.lock().unwrap()[0].len(), 2);
    assert_all_returned(&rig.exec);
    assert!(rig.exec.stats().evictions >= 2, "both idle residents gave way");
}
