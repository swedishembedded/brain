// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sharing accelerators between
// serving and training so that every request and every job is eventually
// served for its clients. If your team needs expertise in GPU memory
// residency and recovery then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Residency conformance: a server serves every request and runs every job that its cards can hold, whatever it
//! holds at the moment. Idle serving residents give way to a training run that needs their memory, a request for a
//! model that gave way waits while it is rebuilt, and a run that fails gives its card back to serving.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, Job, MemCost, Policy, ResidentModel};

const GB: u64 = 1 << 30;

/// A serving model of `vram` bytes that counts its builds.
struct Served {
    name: &'static str,
    vram: u64,
    builds: Arc<AtomicU32>,
}

struct ServedInst;

impl ResidentModel for Served {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name, "a serving model", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.name, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(self.vram, 0)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(ServedInst))
    }
}

impl Instance for ServedInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        Ok(Outcome::new())
    }
}

/// A fine-tune reserving `vram` bytes: its run waits for the test's word, then succeeds or fails as told.
struct Train {
    vram: u64,
    gate: Arc<Mutex<Receiver<Result<(), String>>>>,
}

struct TrainInst(Arc<Mutex<Receiver<Result<(), String>>>>);

impl ResidentModel for Train {
    fn manifest(&self) -> Manifest {
        Manifest::new("train", "a workload", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("train", "run")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(self.vram, 0).with_activation(0)
    }
    fn transient(&self) -> bool {
        true
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainInst(self.gate.clone())))
    }
}

impl Instance for TrainInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let told = self.0.lock().unwrap().recv_timeout(Duration::from_secs(30)).unwrap_or(Ok(()));
        told.map(|()| Outcome::new())
    }
}

fn submit(exec: &Executor, model: &str) -> Receiver<ActionResult> {
    let (tx, rx) = channel();
    exec.submit(Job::new(model, "run", Invocation::new()).reply(move |r| {
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

fn resident(exec: &Executor) -> Vec<String> {
    let mut names: Vec<String> = exec.residency().placements.iter().map(|p| p.key.model.clone()).collect();
    names.sort();
    names
}

/// The production shape: a 96 GiB card holding three idle serving models (10 + 18 + 40 GiB) is asked for a 46 GiB
/// fine-tune. It was refused; the residents that stand in its way give way instead, the run starts, and a request
/// for a model that gave way while the run holds the card waits and is served by a rebuild once the run ends.
#[test]
fn a_training_run_is_admitted_by_evicting_idle_serving_residents_and_they_are_rebuilt_on_demand() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 96 * GB, 0);
    let builds: Vec<Arc<AtomicU32>> = (0..3).map(|_| Arc::new(AtomicU32::new(0))).collect();
    let (tell, gate) = channel();
    let models: Vec<Arc<dyn ResidentModel>> = vec![
        Arc::new(Served { name: "small", vram: 10 * GB, builds: builds[0].clone() }),
        Arc::new(Served { name: "mid", vram: 18 * GB, builds: builds[1].clone() }),
        Arc::new(Served { name: "large", vram: 40 * GB, builds: builds[2].clone() }),
        Arc::new(Train { vram: 46 * GB, gate: Arc::new(Mutex::new(gate)) }),
    ];
    let exec = Executor::start(models, budgets, Policy::default());
    for name in ["small", "mid", "large"] {
        submit(&exec, name).recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    }
    assert_eq!(resident(&exec), ["large", "mid", "small"], "every serving model is resident and idle");

    let trained = submit(&exec, "train");
    wait_running(&exec, "train");
    let used: u64 = exec.residency().placements.iter().map(|p| p.mem).sum();
    assert!(used <= 96 * GB, "the run's reservation and what stayed resident fit the card: {used}");
    let evicted: Vec<&str> = ["small", "mid", "large"].into_iter().filter(|m| !resident(&exec).iter().any(|r| r == m)).collect();
    assert!(!evicted.is_empty(), "the run took the place of idle residents");

    // A request for a model that gave way, while the run holds the card: it waits, it is not refused.
    let waiting = submit(&exec, evicted[0]);
    assert!(waiting.recv_timeout(Duration::from_millis(300)).is_err(), "the request waits while the run holds the card");
    tell.send(Ok(())).unwrap();
    trained.recv_timeout(Duration::from_secs(10)).unwrap().expect("the run completes");
    waiting.recv_timeout(Duration::from_secs(10)).expect("the waiting request is answered").expect("and served");
    let index = ["small", "mid", "large"].iter().position(|m| *m == evicted[0]).unwrap();
    assert_eq!(builds[index].load(Ordering::SeqCst), 2, "{} was rebuilt for it", evicted[0]);
    for name in ["small", "mid", "large"] {
        submit(&exec, name).recv_timeout(Duration::from_secs(10)).unwrap().expect("every model serves again after the run");
    }
    exec.shutdown();
}

/// A run that fails mid-way (a device allocation it could not get) gives its card back: its reservation is released
/// and serving resumes on the card at once. The server never stays in a state where it serves nothing while the
/// memory is there.
#[test]
fn a_failed_training_run_gives_its_card_back_to_serving() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 96 * GB, 0);
    let builds = Arc::new(AtomicU32::new(0));
    let (tell, gate) = channel();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Served { name: "large", vram: 60 * GB, builds: builds.clone() }), Arc::new(Train { vram: 46 * GB, gate: Arc::new(Mutex::new(gate)) })];
    let exec = Executor::start(models, budgets, Policy::default());
    submit(&exec, "large").recv_timeout(Duration::from_secs(10)).unwrap().unwrap();

    let trained = submit(&exec, "train");
    wait_running(&exec, "train");
    let waiting = submit(&exec, "large");
    tell.send(Err("train: the device ran out of memory allocating 4096 bytes".into())).unwrap();
    assert!(trained.recv_timeout(Duration::from_secs(10)).unwrap().is_err(), "the run's failure is its own answer");
    waiting.recv_timeout(Duration::from_secs(10)).expect("the waiting request is answered").expect("and served");
    assert_eq!(resident(&exec), ["large"], "nothing of the failed run stays charged");
    assert_eq!(exec.residency().placements.iter().map(|p| p.mem).sum::<u64>(), 60 * GB);
    exec.shutdown();
}
