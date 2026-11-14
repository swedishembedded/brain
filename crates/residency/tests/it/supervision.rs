// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping model servers
// available through wedged and crashing accelerators for its clients. If your
// team needs expertise in supervised device execution and scheduler recovery
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Supervised device work through the executor's lanes. A fake worker (a POSIX
//! shell speaking the worker protocol) hangs or aborts on request; each test
//! checks what a caller and the scheduler see: a definitive error within the
//! deadline, the worker gone from `/proc`, the claim unwound, the device
//! quarantined while the other one keeps serving, and the device readmitted
//! once a health probe passes.

use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use capability::{ActionResult, ActionSpec, Invocation, Manifest};
use gpu_supervisor::{BreakerPolicy, BreakerState, Health, WorkerCommand};
use residency::budget::Budgets;
use residency::supervised::{Deadlines, Probe, SupervisedModel, Supervision};
use residency::{Device, Executor, Instance, InstanceKey, Job, MemCost, MultiDeviceCost, MultiDeviceResidentModel, Policy, ResidentModel};
use serde_json::json;

const GB: u64 = 1 << 30;

/// `open` and `probe` answer unless the worker was started as `crash-open` or
/// `bad-probe`; a run hangs (in a grandchild, as a driver thread would) or
/// aborts when its parameters say so.
const FAKE: &str = r#"
role=$1
while IFS= read -r line; do
  id=${line%% *}
  req=${line#* }
  case "$req" in
    open*) if [ "$role" = crash-open ]; then kill -ABRT $$; fi; echo "$id ok {}" ;;
    probe*) if [ "$role" = bad-probe ]; then echo "$id err the device did not answer"; else echo "$id ok {}"; fi ;;
    *hang*) sleep 3600 ;;
    *crash*) kill -ABRT $$ ;;
    *) echo "$id ok {\"done\":true}" ;;
  esac
done
"#;

fn fake(role: &str) -> WorkerCommand {
    WorkerCommand::new("/bin/sh").arg("-c").arg(FAKE).arg("fake-worker").arg(role)
}

/// Long enough to outlast a crash's core dump on any host.
const REAP: Duration = Duration::from_secs(20);
const RUN_DEADLINE: Duration = Duration::from_millis(1500);

fn health() -> Arc<Health<Device>> {
    let policy = BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 };
    Arc::new(Health::new(policy, REAP))
}

fn model(name: &str, role: &str, health: &Arc<Health<Device>>) -> Arc<dyn ResidentModel> {
    let manifest = Manifest::new(name, "a fake supervised device model", vec![ActionSpec::new("run", "run")]);
    let deadlines = Deadlines { open: Duration::from_secs(5), run: RUN_DEADLINE, reap: REAP };
    Arc::new(SupervisedModel::new(manifest, fake(role), health.clone(), MemCost::new(20 * GB, 0), deadlines))
}

fn two_gpus() -> Budgets {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
    budgets
}

fn start(models: Vec<Arc<dyn ResidentModel>>, budgets: Budgets, health: &Arc<Health<Device>>, probe: &str) -> Executor {
    let supervision = Supervision { health: health.clone(), probe: Probe::Worker { command: fake(probe), deadline: Duration::from_secs(5) } };
    Executor::start_supervised(models, budgets, Policy::default(), None, Some(supervision))
}

/// Submit and return (admitted, replied) receivers.
fn submit(exec: &Executor, model: &str, mode: &str) -> (Receiver<()>, Receiver<(ActionResult, Instant)>) {
    let (atx, arx) = channel();
    let (rtx, rrx) = channel();
    let inv = Invocation::new().set("mode", json!(mode));
    exec.submit(Job::new(model, "run", inv).on_admit(move || { let _ = atx.send(()); }).reply(move |r| { let _ = rtx.send((r, Instant::now())); }));
    (arx, rrx)
}

fn wait_until(what: &str, within: Duration, mut done: impl FnMut() -> bool) {
    let until = Instant::now() + within;
    while !done() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The pid the fault message names ("worker <pid> was killed ...").
fn worker_pid(msg: &str) -> u32 {
    let after = msg.split("worker ").nth(1).unwrap_or_else(|| panic!("no worker pid in {msg}"));
    after.split_whitespace().next().and_then(|p| p.parse().ok()).unwrap_or_else(|| panic!("no worker pid in {msg}"))
}

fn assert_process_gone(pid: u32) {
    // A reaped process has no /proc entry; a zombie still would.
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "worker {pid} is still in /proc");
}

#[test]
fn a_hung_run_fails_by_its_deadline_and_quarantines_only_its_device() {
    let health = health();
    let exec = start(vec![model("a", "normal", &health), model("b", "normal", &health)], two_gpus(), &health, "normal");
    let (a_admitted, a_reply) = submit(&exec, "a", "hang");
    a_admitted.recv_timeout(Duration::from_secs(10)).expect("a is admitted");
    let started = Instant::now();

    // b lands on the other card and is served while a's device is stuck.
    let (_b_admitted, b_reply) = submit(&exec, "b", "ok");
    let (b, b_at) = b_reply.recv_timeout(Duration::from_secs(10)).expect("b is answered");
    b.expect("b runs on the healthy card");
    assert!(b_at.duration_since(started) < RUN_DEADLINE, "b waited for a's stuck device");

    let (a, a_at) = a_reply.recv_timeout(Duration::from_secs(10)).expect("a is answered");
    let err = a.expect_err("a hung run must fail");
    let took = a_at.duration_since(started);
    assert!(took < RUN_DEADLINE + Duration::from_secs(2), "a failed {took:?} after admission, deadline {RUN_DEADLINE:?}");
    for needle in ["deadline", "reaped", "quarantined"] {
        assert!(err.contains(needle), "the error must say {needle:?}: {err}");
    }
    assert_process_gone(worker_pid(&err));

    let quarantined = health.quarantined();
    assert_eq!(quarantined.len(), 1, "exactly a's device is quarantined: {quarantined:?}");
    let sick = quarantined[0];
    // The lane is free and the claim unwound: a is no longer resident, and the
    // restart probe readmits the device on its own.
    wait_until("the restart probe to readmit the device", Duration::from_secs(15), || health.status(&sick).state == BreakerState::Closed);
    wait_until("a's claim to be unwound", Duration::from_secs(5), || exec.residency().placements.iter().all(|r| r.key.model != "a"));
    let (_, again) = submit(&exec, "a", "ok");
    again.recv_timeout(Duration::from_secs(10)).expect("a answers").0.expect("a is rebuilt and runs once the device is readmitted");
    exec.shutdown();
}

#[test]
fn a_crash_during_activation_fails_the_job_and_the_device_stays_out_until_a_probe_passes() {
    let health = health();
    // A probe that fails keeps the device out: work goes to the other card.
    let exec = start(vec![model("c", "crash-open", &health), model("d", "normal", &health)], two_gpus(), &health, "bad-probe");
    let (_, c_reply) = submit(&exec, "c", "ok");
    let err = c_reply.recv_timeout(Duration::from_secs(30)).expect("c is answered").0.expect_err("an activation that crashes must fail");
    assert!(err.contains("signal 6"), "the error names the crash: {err}");
    assert_process_gone(worker_pid(&err));
    let sick = {
        let q = health.quarantined();
        assert_eq!(q.len(), 1, "{q:?}");
        q[0]
    };
    // Failing probes trip the breaker instead of spinning; the device stays out.
    wait_until("the failing probes to trip the breaker", Duration::from_secs(15), || health.status(&sick).tripped);
    assert_eq!(health.status(&sick).state, BreakerState::Open);
    for _ in 0..3 {
        let (_, r) = submit(&exec, "d", "ok");
        r.recv_timeout(Duration::from_secs(10)).expect("d answers").0.expect("d is placed on the healthy card");
    }
    let report = exec.residency();
    assert!(report.placements.iter().filter(|r| r.key.model == "d").all(|r| r.device != sick), "d must not be placed on the quarantined card: {:?}", report.placements);
    exec.shutdown();
}

#[test]
fn work_that_only_fits_a_quarantined_device_is_refused_not_left_waiting() {
    let health = health();
    let mut one = Budgets::new();
    one.set(Device::Gpu(0), 24 * GB, 0);
    let exec = start(vec![model("e", "normal", &health)], one, &health, "bad-probe");
    let (_, r) = submit(&exec, "e", "crash");
    r.recv_timeout(Duration::from_secs(30)).expect("answered").0.expect_err("the crash fails the run");
    wait_until("the breaker to trip", Duration::from_secs(15), || health.status(&Device::Gpu(0)).tripped);
    let started = Instant::now();
    let (_, r) = submit(&exec, "e", "ok");
    let err = r.recv_timeout(Duration::from_secs(10)).expect("a refusal, not a wait").0.expect_err("nothing may run on a quarantined device");
    assert!(err.contains("quarantined"), "{err}");
    assert!(started.elapsed() < Duration::from_secs(5), "the refusal took {:?}", started.elapsed());
    exec.shutdown();
}

/// An in-process model whose run fails the way a device wait past
/// `--gpu-wait-s` does when its parameters say `wedge`, without panicking.
struct InProcess;

struct InProcessInstance;

impl ResidentModel for InProcess {
    fn manifest(&self) -> Manifest {
        Manifest::new("in-process", "a fake in-process device model", vec![ActionSpec::new("run", "run")])
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("in-process", "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(GB, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(InProcessInstance))
    }
}

impl Instance for InProcessInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(capability::Progress)) -> ActionResult {
        match inv.params.get("mode").and_then(|m| m.as_str()) {
            Some("wedge") => Err(format!("poll_wait: GPU submit did not complete within 1s {} -- device likely wedged", backend_api::hardware::WAIT_TIMEOUT_MARK)),
            _ => Ok(capability::Outcome::new()),
        }
    }
}

#[test]
fn without_a_worker_an_in_process_wait_timeout_still_quarantines_the_card_until_the_breaker_readmits_it() {
    let policy = BreakerPolicy { window: Duration::from_secs(600), trip_after: 2, probe_interval: Duration::from_millis(300), probes_to_close: 2 };
    let health = Arc::new(Health::new(policy, REAP));
    let mut one = Budgets::new();
    one.set(Device::Gpu(0), 24 * GB, 0);
    let exec = Executor::start_supervised(vec![Arc::new(InProcess)], one, Policy::default(), None, Some(Supervision { health: health.clone(), probe: Probe::Elapsed }));
    let gpu0 = Device::Gpu(0);

    // A first wait timeout is a restart: the card is quarantined, and with no
    // worker to probe it the due probe readmits it without touching it.
    let (_, r) = submit(&exec, "in-process", "wedge");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect_err("the wedged run fails");
    assert_eq!(health.faults(&gpu0), 1, "the wait timeout is recorded against the card");
    wait_until("the card to be readmitted", Duration::from_secs(15), || health.admits(&gpu0));
    let (_, r) = submit(&exec, "in-process", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("the readmitted card serves");

    // A second inside the window trips the breaker: work is refused at once
    // until the spaced readmissions have passed, then served again.
    let (_, r) = submit(&exec, "in-process", "wedge");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect_err("the wedged run fails");
    assert!(health.status(&gpu0).tripped, "{:?}", health.status(&gpu0));
    let (_, r) = submit(&exec, "in-process", "ok");
    let err = r.recv_timeout(Duration::from_secs(5)).expect("a refusal, not a wait").0.expect_err("a tripped card takes no work");
    assert!(err.contains("quarantined"), "{err}");
    wait_until("the tripped card to be readmitted", Duration::from_secs(20), || health.admits(&gpu0));
    let (_, r) = submit(&exec, "in-process", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("the card serves again");
    exec.shutdown();
}

/// A model that spans both cards: its placement is exactly gpu0 and gpu1.
struct Spanning;

impl ResidentModel for Spanning {
    fn manifest(&self) -> Manifest {
        Manifest::new("spanning", "a fake model sharded over two cards", vec![ActionSpec::new("run", "run")])
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("spanning", "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(GB, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err("spanning runs only across both cards".into())
    }
}

impl MultiDeviceResidentModel for Spanning {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        MultiDeviceCost::new(vec![(Device::Gpu(0), GB), (Device::Gpu(1), GB)], 0)
    }

    fn activate_multi(&self, _key: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(InProcessInstance))
    }
}

#[test]
fn a_multi_device_model_whose_placement_has_a_tripped_card_is_refused_not_left_waiting() {
    let health = health();
    let exec = Executor::start_supervised(vec![Arc::new(InProcess)], two_gpus(), Policy::default(), None, Some(Supervision { health: health.clone(), probe: Probe::Elapsed }));
    exec.register_multi(Arc::new(Spanning));
    let (_, r) = submit(&exec, "spanning", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("both cards healthy: the model runs");

    let now = Instant::now();
    for _ in 0..health.policy().trip_after {
        health.record_fault(&Device::Gpu(1), now, "a device wait ran past (--gpu-wait-s)");
    }
    assert!(health.status(&Device::Gpu(1)).tripped);
    let started = Instant::now();
    let (_, r) = submit(&exec, "spanning", "ok");
    let err = r.recv_timeout(Duration::from_secs(10)).expect("a refusal, not a wait for the next probe").0.expect_err("gpu1 is out for a probe interval");
    assert!(err.contains("quarantined") && err.contains("gpu1"), "{err}");
    assert!(started.elapsed() < Duration::from_secs(5), "the refusal took {:?}", started.elapsed());
    // Single-device work still lands on the healthy card.
    let (_, r) = submit(&exec, "in-process", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("gpu0 serves");
    exec.shutdown();
}

/// An in-process model whose instance's drop blocks until the test opens a
/// gate, as a drop through a wedged driver would, and records the thread it
/// ran on.
#[derive(Default)]
struct Sticky {
    gate: Arc<(Mutex<bool>, Condvar)>,
    dropped_on: Arc<Mutex<Option<String>>>,
}

struct StickyInstance {
    gate: Arc<(Mutex<bool>, Condvar)>,
    dropped_on: Arc<Mutex<Option<String>>>,
}

impl Sticky {
    fn open_gate(&self) {
        let (open, cv) = &*self.gate;
        *open.lock().unwrap() = true;
        cv.notify_all();
    }
}

impl ResidentModel for Sticky {
    fn manifest(&self) -> Manifest {
        Manifest::new("sticky", "a fake model whose teardown wedges", vec![ActionSpec::new("run", "run")])
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("sticky", "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(20 * GB, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(StickyInstance { gate: self.gate.clone(), dropped_on: self.dropped_on.clone() }))
    }
}

impl Instance for StickyInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(capability::Progress)) -> ActionResult {
        InProcessInstance.run(action, inv, progress)
    }
}

impl Drop for StickyInstance {
    fn drop(&mut self) {
        *self.dropped_on.lock().unwrap() = std::thread::current().name().map(str::to_string);
        let (open, cv) = &*self.gate;
        let mut open = open.lock().unwrap();
        while !*open {
            open = cv.wait(open).unwrap();
        }
    }
}

fn used_on(exec: &Executor, device: Device) -> u64 {
    exec.residency().budgets.iter().find(|b| b.device == device).map_or(0, |b| b.used)
}

#[test]
fn a_resident_on_a_tripped_card_is_abandoned_off_the_dispatcher_and_rebuilt_on_a_healthy_one() {
    let health = health();
    let sticky = Arc::new(Sticky::default());
    let exec = Executor::start_supervised(vec![sticky.clone()], two_gpus(), Policy::default(), None, Some(Supervision { health: health.clone(), probe: Probe::Elapsed }));
    let (_, r) = submit(&exec, "sticky", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("the first job builds the instance");
    let home = exec.residency().placements.iter().find(|p| p.key.model == "sticky").map(|p| p.device).expect("sticky is resident");
    let other = if home == Device::Gpu(0) { Device::Gpu(1) } else { Device::Gpu(0) };

    let now = Instant::now();
    for _ in 0..health.policy().trip_after {
        health.record_fault(&home, now, "a device wait ran past (--gpu-wait-s)");
    }
    // The next job is served on the healthy card even though tearing the old
    // instance down never finishes: the teardown runs off the dispatcher.
    let (_, r) = submit(&exec, "sticky", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered, not refused or left waiting").0.expect("rebuilt on the healthy card");
    let placed: Vec<Device> = exec.residency().placements.iter().filter(|p| p.key.model == "sticky").map(|p| p.device).collect();
    assert_eq!(placed, vec![other], "the instance moved off the tripped card");
    wait_until("the abandoned instance's teardown to start", Duration::from_secs(10), || sticky.dropped_on.lock().unwrap().is_some());
    let dropped_on = sticky.dropped_on.lock().unwrap().clone();
    assert!(dropped_on.as_deref().is_some_and(|t| t != "brain-dispatcher" && !t.starts_with("brain-lane")), "the abandoned instance was dropped on {dropped_on:?}");
    // Its memory is still taken until the teardown finishes, so nothing is
    // placed on the card over it once the card is readmitted.
    assert_eq!(used_on(&exec, home), 20 * GB, "the abandoned instance stays charged while it is torn down");
    sticky.open_gate();
    wait_until("the abandoned charge to be released", Duration::from_secs(10), || used_on(&exec, home) == 0);
    exec.shutdown();
}

#[test]
fn a_faulted_instance_whose_teardown_wedges_does_not_stall_the_scheduler() {
    let health = health();
    let sticky = Arc::new(Sticky::default());
    let exec = Executor::start_supervised(vec![sticky.clone(), Arc::new(InProcess)], two_gpus(), Policy::default(), None, Some(Supervision { health: health.clone(), probe: Probe::Elapsed }));
    let (_, r) = submit(&exec, "sticky", "wedge");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect_err("the wedged run fails");
    // The faulted claim is unwound while its instance's teardown is stuck:
    // the scheduler still answers and still places work.
    wait_until("the stuck teardown to start", Duration::from_secs(10), || sticky.dropped_on.lock().unwrap().is_some());
    assert!(exec.residency().placements.iter().all(|p| p.key.model != "sticky"), "the faulted instance is no longer resident");
    let (_, r) = submit(&exec, "in-process", "ok");
    r.recv_timeout(Duration::from_secs(10)).expect("answered").0.expect("another model is served meanwhile");
    let dropped_on = sticky.dropped_on.lock().unwrap().clone();
    assert!(dropped_on.as_deref().is_some_and(|t| t != "brain-dispatcher"), "the faulted instance was dropped on {dropped_on:?}");
    sticky.open_gate();
    exec.shutdown();
}
