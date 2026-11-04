// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many variants of one
// resident model through a single scheduler for its clients. If your team
// needs expertise in multi-tenant model serving then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Several selectable model ids served by ONE built instance: a base and the
//! sibling ids that ride on it (the shape a LoRA adapter view takes).
//!
//! Pins the three seams that shape needs from the executor: a sibling id
//! resolves to the shared [`InstanceKey`] (so base and sibling jobs batch
//! together and activate once), [`ResidentModel::stamp`] tells the instance
//! which id the caller named, and [`Executor::unregister`] withdraws a sibling
//! without disturbing a job that was already accepted for it.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::{AdapterIdentity, Device, Executor, Instance, InstanceKey, Job, MemCost, Policy, ResidentModel};
use serde_json::json;

/// The reserved param a sibling stamps; the base removes it.
const ASKED: &str = "__asked";

/// What the shared instance saw: which id each invocation of a batch asked for.
#[derive(Default)]
struct Seen {
    batches: Mutex<Vec<Vec<Option<String>>>>,
    activations: AtomicU32,
    /// Batches that have started; a test waits on it to know a batch is running.
    entered: AtomicU32,
    /// Held by a test to keep a started batch from finishing.
    gate: Mutex<()>,
}

struct Base {
    seen: Arc<Seen>,
}

struct Sibling {
    name: String,
    base: Arc<Base>,
    dropped: Arc<AtomicU32>,
    identity: Option<AdapterIdentity>,
}

struct Shared {
    seen: Arc<Seen>,
}

fn manifest(name: &str) -> Manifest {
    Manifest::new(name, "sibling test", vec![ActionSpec::new("run", "run")])
}

fn shared_key() -> InstanceKey {
    InstanceKey::new("base", "default")
}

impl ResidentModel for Base {
    fn manifest(&self) -> Manifest {
        manifest("base")
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        shared_key()
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(0, 1 << 20)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        self.seen.activations.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Shared { seen: self.seen.clone() }))
    }
    fn stamp(&self, inv: &mut Invocation) {
        if let Some(params) = inv.params.as_object_mut() {
            params.remove(ASKED);
        }
    }
}

impl ResidentModel for Sibling {
    fn manifest(&self) -> Manifest {
        manifest(&self.name)
    }
    fn instance_key(&self, a: &str, i: &Invocation) -> InstanceKey {
        self.base.instance_key(a, i)
    }
    fn estimate(&self, k: &InstanceKey) -> MemCost {
        self.base.estimate(k)
    }
    fn activate(&self, k: &InstanceKey, d: Device) -> Result<Box<dyn Instance>, String> {
        self.base.activate(k, d)
    }
    fn stamp(&self, inv: &mut Invocation) {
        if let Some(params) = inv.params.as_object_mut() {
            params.insert(ASKED.to_string(), json!(self.name));
        }
    }
    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.identity.clone()
    }
}

impl Drop for Sibling {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl Instance for Shared {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_, p| progress(p)).pop().unwrap()
    }
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.seen.entered.fetch_add(1, Ordering::SeqCst);
        let _held = self.seen.gate.lock().unwrap();
        let asked: Vec<Option<String>> = invs.iter().map(|i| i.get_str(ASKED)).collect();
        self.seen.batches.lock().unwrap().push(asked.clone());
        asked.into_iter().map(|a| Ok(Outcome::new().set("asked", json!(a)))).collect()
    }
}

fn executor(seen: &Arc<Seen>, siblings: &[(&str, Option<AdapterIdentity>)], dropped: &Arc<AtomicU32>) -> Executor {
    let base = Arc::new(Base { seen: seen.clone() });
    let mut models: Vec<Arc<dyn ResidentModel>> = vec![base.clone()];
    for (name, identity) in siblings {
        models.push(Arc::new(Sibling { name: name.to_string(), base: base.clone(), dropped: dropped.clone(), identity: identity.clone() }));
    }
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

fn ask(exec: &Executor, model: &str, inv: Invocation) -> std::sync::mpsc::Receiver<ActionResult> {
    let (tx, rx) = channel();
    exec.submit(Job::new(model, "run", inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx
}

fn asked(r: ActionResult) -> Option<String> {
    r.expect("the job runs").outputs["asked"].as_str().map(str::to_string)
}

const WAIT: Duration = Duration::from_secs(20);

/// A sibling id reaches the shared instance stamped with its own name; the
/// base id arrives unstamped, and a caller-supplied value for the reserved
/// param never survives either path.
#[test]
fn the_instance_learns_which_sibling_id_was_asked_for() {
    let seen = Arc::new(Seen::default());
    let exec = executor(&seen, &[("base:team:tune:v1", None)], &Arc::new(AtomicU32::new(0)));

    assert_eq!(asked(ask(&exec, "base:team:tune:v1", Invocation::new()).recv_timeout(WAIT).unwrap()), Some("base:team:tune:v1".into()));
    assert_eq!(asked(ask(&exec, "base", Invocation::new()).recv_timeout(WAIT).unwrap()), None);
    // The base path strips a forged value; the sibling path overwrites it.
    assert_eq!(asked(ask(&exec, "base", Invocation::new().set(ASKED, json!("base:team:tune:v1"))).recv_timeout(WAIT).unwrap()), None);
    assert_eq!(
        asked(ask(&exec, "base:team:tune:v1", Invocation::new().set(ASKED, json!("somebody-else"))).recv_timeout(WAIT).unwrap()),
        Some("base:team:tune:v1".into())
    );
    assert_eq!(seen.activations.load(Ordering::SeqCst), 1, "siblings share the base's one instance");
    exec.shutdown();
}

/// A base job and two sibling ids queued behind a running batch are ONE batch
/// on the SAME activation afterwards, each invocation carrying its own id.
#[test]
fn a_base_job_and_sibling_jobs_run_as_one_batch() {
    let seen = Arc::new(Seen::default());
    let exec = executor(&seen, &[("base:t:a:v1", None), ("base:t:b:v1", None)], &Arc::new(AtomicU32::new(0)));

    let gate = seen.gate.lock().unwrap();
    let first = ask(&exec, "base", Invocation::new());
    let deadline = std::time::Instant::now() + WAIT;
    while seen.entered.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "the first batch never started");
        std::thread::sleep(Duration::from_millis(5));
    }
    let queued: Vec<_> = ["base:t:a:v1", "base", "base:t:b:v1"].into_iter().map(|m| ask(&exec, m, Invocation::new())).collect();
    // Let the dispatcher see all three before the first batch ends.
    std::thread::sleep(Duration::from_millis(200));
    drop(gate);

    assert_eq!(asked(first.recv_timeout(WAIT).unwrap()), None);
    let got: Vec<Option<String>> = queued.into_iter().map(|rx| asked(rx.recv_timeout(WAIT).unwrap())).collect();
    assert_eq!(got, vec![Some("base:t:a:v1".into()), None, Some("base:t:b:v1".into())], "each job was answered as the id it asked for");
    assert_eq!(seen.activations.load(Ordering::SeqCst), 1, "no id built its own instance");
    let batches = seen.batches.lock().unwrap();
    assert_eq!(batches.len(), 2, "one running batch, then one batch for everything that queued: {batches:?}");
    let mut second = batches[1].clone();
    second.sort();
    assert_eq!(second, vec![None, Some("base:t:a:v1".into()), Some("base:t:b:v1".into())], "base and both siblings batched together");
    exec.shutdown();
}

/// A registered sibling that names an adapter is listed with it; one that
/// does not leaves the table alone.
#[test]
fn a_registered_model_that_names_an_adapter_is_recorded_under_its_id() {
    let identity = AdapterIdentity { id: "base:t:a:v1".into(), digest: format!("sha256:{}", "a".repeat(64)) };
    let seen = Arc::new(Seen::default());
    let dropped = Arc::new(AtomicU32::new(0));
    let exec = executor(&seen, &[("base:t:a:v1", Some(identity.clone()))], &dropped);
    assert_eq!(exec.served_adapter("base:t:a:v1"), Some(identity));
    assert_eq!(exec.served_adapter("base"), None);

    let late = AdapterIdentity { id: "base:t:late:v1".into(), digest: format!("sha256:{}", "b".repeat(64)) };
    exec.register(Arc::new(Sibling { name: "base:t:late:v1".into(), base: Arc::new(Base { seen }), dropped, identity: Some(late.clone()) }));
    assert_eq!(exec.served_adapter("base:t:late:v1"), Some(late), "registration after start records it too");
    exec.shutdown();
}

/// An unregistered sibling leaves the listing and the adapter table at once,
/// refuses new work, and is dropped; the base is untouched.
#[test]
fn an_unregistered_sibling_is_unlisted_refused_and_released() {
    let identity = AdapterIdentity { id: "base:t:a:v1".into(), digest: format!("sha256:{}", "a".repeat(64)) };
    let seen = Arc::new(Seen::default());
    let dropped = Arc::new(AtomicU32::new(0));
    let exec = executor(&seen, &[("base:t:a:v1", Some(identity))], &dropped);
    assert!(exec.manifests().iter().any(|m| m.model == "base:t:a:v1"));

    exec.unregister("base:t:a:v1");
    assert!(!exec.manifests().iter().any(|m| m.model == "base:t:a:v1"), "unlisted immediately");
    assert_eq!(exec.served_adapter("base:t:a:v1"), None, "and no longer reports an adapter");
    let err = ask(&exec, "base:t:a:v1", Invocation::new()).recv_timeout(WAIT).unwrap().unwrap_err();
    assert!(err.contains("no model"), "a retired id is an unknown id: {err}");
    assert!(asked_ok(&exec, "base"), "the base keeps serving");

    let deadline = std::time::Instant::now() + WAIT;
    while dropped.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "the retired model was never released");
        std::thread::sleep(Duration::from_millis(10));
    }
    exec.shutdown();
}

fn asked_ok(exec: &Executor, model: &str) -> bool {
    ask(exec, model, Invocation::new()).recv_timeout(WAIT).unwrap().is_ok()
}

/// A job accepted for a sibling before it was unregistered still runs, and the
/// sibling is released only after that job is done.
#[test]
fn a_job_accepted_before_unregistration_still_runs_and_holds_the_sibling() {
    let seen = Arc::new(Seen::default());
    let dropped = Arc::new(AtomicU32::new(0));
    let exec = executor(&seen, &[("base:t:a:v1", None)], &dropped);
    // FIFO on the dispatcher channel guarantees the job is accepted before the
    // unregistration is processed.
    let rx = ask(&exec, "base:t:a:v1", Invocation::new());
    exec.unregister("base:t:a:v1");
    assert_eq!(asked(rx.recv_timeout(WAIT).unwrap()), Some("base:t:a:v1".into()), "the accepted job ran as the sibling");

    let deadline = std::time::Instant::now() + WAIT;
    while dropped.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "released after its last job");
        std::thread::sleep(Duration::from_millis(10));
    }
    exec.shutdown();
}

/// Registering a sibling's name again after unregistering it revives the id,
/// and the new registration is not swept away with the old one.
#[test]
fn re_registering_a_retired_name_revives_it() {
    let seen = Arc::new(Seen::default());
    let dropped = Arc::new(AtomicU32::new(0));
    let exec = executor(&seen, &[("base:t:a:v1", None)], &dropped);
    let base = Arc::new(Base { seen });

    exec.unregister("base:t:a:v1");
    exec.register(Arc::new(Sibling { name: "base:t:a:v1".into(), base, dropped, identity: None }));
    assert!(exec.manifests().iter().any(|m| m.model == "base:t:a:v1"));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(asked(ask(&exec, "base:t:a:v1", Invocation::new()).recv_timeout(WAIT).unwrap()), Some("base:t:a:v1".into()));
    exec.shutdown();
}
