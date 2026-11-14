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
use std::sync::Arc;
use std::time::{Duration, Instant};

use capability::{ActionResult, ActionSpec, Invocation, Manifest};
use gpu_supervisor::{BreakerPolicy, BreakerState, Health, WorkerCommand};
use residency::budget::Budgets;
use residency::supervised::{Deadlines, SupervisedModel, Supervision};
use residency::{Device, Executor, Job, MemCost, Policy, ResidentModel};
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
    let supervision = Supervision { health: health.clone(), probe: fake(probe), probe_deadline: Duration::from_secs(5) };
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
