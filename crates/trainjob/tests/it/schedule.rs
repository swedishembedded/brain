// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The scheduling policy, with real runs: an idle serving resident gives way to a run, a busy one is never
//! taken from, and a run that ends gives its memory back to serving.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::{Device, Instance, InstanceKey, Job, MemCost, ResidentModel};
use serde_json::json;

use crate::support::{data_digest, invocation, plain, rig_with, BASE};

const GIB: u64 = 1 << 30;

/// A served model that holds `bytes` of host memory while resident; its job blocks until it is let go.
struct Served {
    builds: Arc<AtomicU32>,
    gate: Arc<Mutex<Receiver<()>>>,
    running: Arc<AtomicBool>,
}
struct ServedInst {
    gate: Arc<Mutex<Receiver<()>>>,
    running: Arc<AtomicBool>,
}

const SERVED_BYTES: u64 = 3 * GIB;

impl ResidentModel for Served {
    fn manifest(&self) -> Manifest {
        Manifest::new("served", "a served model", vec![ActionSpec::new("hold", "hold until released")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("served", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(0, SERVED_BYTES)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(ServedInst { gate: Arc::clone(&self.gate), running: Arc::clone(&self.running) }))
    }
}

impl Instance for ServedInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.get_str("wait").as_deref() == Some("yes") {
            self.running.store(true, Ordering::SeqCst);
            let _ = self.gate.lock().unwrap().recv_timeout(Duration::from_secs(300));
            self.running.store(false, Ordering::SeqCst);
        }
        Ok(Outcome::new())
    }
}

struct Rig {
    inner: crate::support::Rig,
    builds: Arc<AtomicU32>,
    release: Sender<()>,
    running: Arc<AtomicBool>,
}

fn rig(tag: &str) -> Rig {
    let (release, gate) = channel();
    let builds = Arc::new(AtomicU32::new(0));
    let running = Arc::new(AtomicBool::new(false));
    let served = Served { builds: builds.clone(), gate: Arc::new(Mutex::new(gate)), running: running.clone() };
    // 4 GiB of host memory: the served model holds 3, a run needs a little over 2.
    Rig { inner: rig_with(tag, 4 * GIB, true, None, vec![Arc::new(served)]), builds, release, running }
}

fn train_params(rig: &Rig) -> serde_json::Value {
    let train = data_digest(&rig.inner, &plain());
    json!({"base": BASE, "arch": "qwen35", "steps": 1, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()})
}

fn submit_train(rig: &Rig) -> (Receiver<ActionResult>, Arc<AtomicBool>) {
    let inv = invocation(train_params(rig), &[]).unwrap();
    let admitted = Arc::new(AtomicBool::new(false));
    let seen = admitted.clone();
    let (tx, rx) = channel();
    rig.inner.exec.submit(
        Job::new(trainjob::MODEL, trainjob::ACTION, inv)
            .on_admit(move || seen.store(true, Ordering::SeqCst))
            .reply(move |r| {
                let _ = tx.send(r);
            }),
    );
    (rx, admitted)
}

#[test]
fn an_idle_serving_resident_gives_way_to_a_run_and_is_rebuilt_when_asked_for_again() {
    let rig = rig("schedule-idle");
    rig.inner.exec.run_blocking("served", "hold", Invocation::new(), |_| {}).unwrap();
    assert_eq!(rig.builds.load(Ordering::SeqCst), 1);
    assert_eq!(rig.inner.exec.residency().placements.len(), 1, "the served model is resident and idle");

    // The run does not fit beside it, so it takes its place.
    let (done, _) = submit_train(&rig);
    done.recv_timeout(Duration::from_secs(600)).expect("the run ended").expect("the run");
    let placed = rig.inner.exec.residency();
    assert!(placed.placements.is_empty(), "the run gave its memory back, and the evicted resident has not been rebuilt for nothing: {placed:?}");
    assert!(rig.inner.exec.stats().evictions >= 1);

    // The next request for the served model rebuilds it where there is room.
    rig.inner.exec.run_blocking("served", "hold", Invocation::new(), |_| {}).unwrap();
    assert_eq!(rig.builds.load(Ordering::SeqCst), 2);
}

#[test]
fn a_busy_serving_resident_is_never_taken_from_and_the_run_waits_for_it() {
    let rig = rig("schedule-busy");
    let busy = {
        let exec = rig.inner.exec.clone();
        std::thread::spawn(move || exec.run_blocking("served", "hold", Invocation::new().set("wait", json!("yes")), |_| {}))
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    while !rig.running.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the served job never started");
        std::thread::sleep(Duration::from_millis(10));
    }

    let (done, admitted) = submit_train(&rig);
    std::thread::sleep(Duration::from_secs(2));
    assert!(!admitted.load(Ordering::SeqCst), "the run took the memory serving is using");
    assert!(done.try_recv().is_err());
    assert_eq!(rig.inner.exec.residency().placements.len(), 1, "and the busy resident is untouched");

    // Serving finishes: the run is admitted and completes.
    rig.release.send(()).unwrap();
    busy.join().unwrap().unwrap();
    done.recv_timeout(Duration::from_secs(600)).expect("the run ended").expect("the run");
    assert!(admitted.load(Ordering::SeqCst));
}
