// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running model code in
// killable worker processes for its clients. If your team needs expertise in
// process isolation for accelerator workloads then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model hosted in a real worker process, through the executor.
//!
//! This binary is its own worker: started as `hosted worker` it runs
//! [`residency::supervised::host`] over a test family, exactly as `brain`'s
//! internal worker mode runs it over the served families; started any other
//! way it runs the tests below, which spawn it in that mode. No GPU is
//! touched: the test family only reports the device it was told to open.
//!
//! The harness is this file's own (`harness = false` in the manifest): a test
//! binary that re-executes itself must not print the test harness's lines to
//! the worker socket, and a worker mode is reached only by an argument.

use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::{failure, ActionResult, ActionSpec, Blob, CancelToken, Invocation, Manifest, Media, Outcome, Progress};
use gpu_supervisor::{BreakerPolicy, BreakerState, Health, Worker, WorkerCommand};
use residency::budget::Budgets;
use residency::supervised::{device_label, Deadlines, HostSpec, HostedFamily, Probe, SupervisedModel, Supervision};
use residency::{Device, Executor, Instance, InstanceKey, Job, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};

const GB: u64 = 1 << 30;
const REAP: Duration = Duration::from_secs(20);
const DEADLINES: Deadlines = Deadlines { open: Duration::from_secs(30), run: Duration::from_secs(30), reap: REAP };

const FAMILIES: &[HostedFamily] = &[HostedFamily { name: "test", open: open_test }];

fn open_test(args: &Value) -> Result<Arc<dyn ResidentModel>, String> {
    let id = args.get("id").and_then(Value::as_str).ok_or("the test family needs an id")?;
    Ok(Arc::new(TestModel(id.to_string())))
}

/// The test family's model. The serving process uses it for its manifest,
/// key and estimate; the worker activates it.
struct TestModel(String);

impl ResidentModel for TestModel {
    fn manifest(&self) -> Manifest {
        let actions = ["echo", "stream", "wait", "block", "panic"].map(|a| ActionSpec::new(a, a)).to_vec();
        Manifest::new(&self.0, "a model hosted in a test worker", actions)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.0, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(20 * GB, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TestInstance { device }))
    }
}

struct TestInstance {
    device: Device,
}

impl Instance for TestInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            // What the worker saw, and every blob back reversed.
            "echo" => {
                let mut out = Outcome::new().set("params", inv.params.clone()).set("scope", json!(inv.scope)).set("device", json!(device_label(self.device))).set("pid", json!(std::process::id()));
                for (name, blob) in &inv.blobs {
                    let reversed: Vec<u8> = blob.bytes.iter().rev().copied().collect();
                    out = out.blob(&format!("out_{name}"), Blob::new(blob.media, reversed).with_meta(blob.meta.clone()));
                }
                Ok(out)
            }
            "stream" => {
                let steps = inv.params.get("steps").and_then(Value::as_u64).unwrap_or(1) as u32;
                for i in 1..=steps {
                    progress(Progress::step(i, steps, format!("step {i}")));
                }
                Ok(Outcome::new().set("steps", json!(steps)))
            }
            // Waits for its cancellation, as a long generation observes one between steps.
            "wait" => {
                progress(Progress::step(0, 1, "waiting"));
                let until = Instant::now() + Duration::from_secs(30);
                while !inv.cancel.is_cancelled() && Instant::now() < until {
                    std::thread::sleep(Duration::from_millis(5));
                }
                inv.cancel.check().map(|()| Outcome::new())
            }
            // A driver call that never returns.
            "block" => {
                progress(Progress::step(0, 1, "blocked"));
                loop {
                    std::thread::sleep(Duration::from_secs(3600));
                }
            }
            "panic" => panic!("the model hit a bug"),
            other => Err(format!("no action {other}")),
        }
    }
}

fn worker_command() -> WorkerCommand {
    WorkerCommand::new(std::env::current_exe().expect("own path")).arg("worker")
}

fn health() -> Arc<Health<Device>> {
    let policy = BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 };
    Arc::new(Health::new(policy, REAP))
}

fn hosted(id: &str, health: &Arc<Health<Device>>) -> Arc<dyn ResidentModel> {
    let spec = HostSpec { family: "test".into(), args: json!({ "id": id }) };
    Arc::new(SupervisedModel::new(Arc::new(TestModel(id.to_string())), spec, worker_command(), health.clone(), DEADLINES))
}

fn gpus(n: u32) -> Budgets {
    let mut budgets = Budgets::new();
    for i in 0..n {
        budgets.set(Device::Gpu(i), 24 * GB, 0);
    }
    budgets
}

/// Probes that never pass keep a faulted card out, so where the next job
/// lands says which cards are healthy.
fn failing_probe() -> Probe {
    let script = r#"while IFS= read -r line; do echo "${line%% *} err the device did not answer"; done"#;
    Probe::Worker { command: WorkerCommand::new("/bin/sh").arg("-c").arg(script), deadline: Duration::from_secs(5) }
}

fn start(health: &Arc<Health<Device>>, cards: u32) -> Executor {
    let supervision = Supervision { health: health.clone(), probe: failing_probe() };
    Executor::start_supervised(vec![hosted("m", health)], gpus(cards), Policy::default(), None, Some(supervision))
}

/// Run one job and wait for its answer, handing every progress update to `on_progress`.
fn run(exec: &Executor, action: &str, inv: Invocation, on_progress: impl FnMut(Progress) + Send + 'static) -> ActionResult {
    let (tx, rx) = channel();
    exec.submit(Job::new("m", action, inv).on_progress(on_progress).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx.recv_timeout(Duration::from_secs(60)).expect("the job is answered")
}

fn echo(exec: &Executor) -> (String, u32) {
    let out = run(exec, "echo", Invocation::new(), |_| {}).expect("echo runs");
    (out.outputs["device"].as_str().unwrap().to_string(), out.outputs["pid"].as_u64().unwrap() as u32)
}

fn a_hosted_model_runs_in_its_worker_with_blobs_through_memfds() {
    let health = health();
    let exec = start(&health, 1);
    // Both are far larger than one line of the protocol (1 MiB).
    let image: Vec<u8> = (0..8u32 << 20).map(|i| (i % 253) as u8).collect();
    let prompt = "p".repeat(2 << 20);
    let mut inv = Invocation::new().set("prompt", json!(prompt)).blob("image", Blob::new(Media::Image, image.clone()).with_meta(json!({"w": 2, "h": 2})));
    inv.scope = "caller-7".into();
    let out = run(&exec, "echo", inv, |_| {}).expect("the hosted model answers");
    assert_eq!(out.outputs["params"]["prompt"].as_str().map(str::len), Some(prompt.len()));
    assert_eq!(out.outputs["scope"], json!("caller-7"), "the caller's scope reaches the worker");
    assert_eq!(out.outputs["device"], json!("gpu0"), "the worker opened the card it was assigned");
    assert_ne!(out.outputs["pid"].as_u64(), Some(u64::from(std::process::id())), "the model ran in another process");
    let back = &out.blobs["out_image"];
    assert_eq!((back.media, &back.meta), (Media::Image, &json!({"w": 2, "h": 2})));
    assert!(back.bytes.iter().eq(image.iter().rev()), "the blob came back whole");
    exec.shutdown();
}

fn progress_and_cancellation_cross_the_boundary() {
    let health = health();
    let exec = start(&health, 1);
    let (_, pid) = echo(&exec);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let out = run(&exec, "stream", Invocation::new().set("steps", json!(3)), move |p| sink.lock().unwrap().push((p.step, p.total, p.message))).expect("stream runs");
    assert_eq!(out.outputs["steps"], json!(3));
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, [(1, 3, "step 1".to_string()), (2, 3, "step 2".to_string()), (3, 3, "step 3".to_string())]);

    // The caller cancels once the job reports that it is waiting.
    let mut inv = Invocation::new();
    inv.cancel = CancelToken::armed();
    let token = inv.cancel.clone();
    let started = Instant::now();
    let err = run(&exec, "wait", inv, move |_| token.cancel()).expect_err("a cancelled job does not succeed");
    assert_eq!(err, failure::CANCELLED, "the model saw its token fire and said so");
    assert!(started.elapsed() < Duration::from_secs(10), "the cancellation took {:?}", started.elapsed());
    assert_eq!(health.status(&Device::Gpu(0)).state, BreakerState::Closed, "a cancellation says nothing about the card");
    assert_eq!(echo(&exec).1, pid, "the worker serves on after a cancelled job");
    exec.shutdown();
}

fn killing_the_worker_mid_job_fails_that_job_and_the_next_rebuilds_on_a_healthy_card() {
    let health = health();
    let exec = start(&health, 2);
    let (sick, pid) = echo(&exec);
    // Killed from outside while it is inside the job, as a crash would.
    let started = Instant::now();
    let err = run(&exec, "block", Invocation::new(), move |_| {
        let killed = std::process::Command::new("kill").arg("-KILL").arg(pid.to_string()).status();
        assert!(killed.is_ok_and(|s| s.success()), "kill {pid}");
    })
    .expect_err("a job whose worker died fails");
    assert!(started.elapsed() < Duration::from_secs(10), "the failure took {:?}", started.elapsed());
    for needle in ["lost its worker", "signal 9", &pid.to_string(), "quarantined"] {
        assert!(err.contains(needle), "the error must say {needle:?}: {err}");
    }
    let sick = residency::supervised::parse_device_label(&sick).unwrap();
    assert!(!health.admits(&sick), "the card the worker died on is quarantined");
    let (healthy, rebuilt) = echo(&exec);
    assert_ne!(healthy, device_label(sick), "the next job runs on a healthy card");
    assert_ne!(rebuilt, pid, "in a new worker");
    exec.shutdown();
}

fn a_panic_in_model_code_fails_the_job_without_quarantining_the_card() {
    let health = health();
    let exec = start(&health, 1);
    let (_, pid) = echo(&exec);
    let err = run(&exec, "panic", Invocation::new(), |_| {}).expect_err("a panic is not an answer");
    assert!(err.contains("the model hit a bug"), "the error carries the panic: {err}");
    let status = health.status(&Device::Gpu(0));
    assert_eq!((status.state, status.faults), (BreakerState::Closed, 0), "a bug in model code is not a device fault");
    let (device, rebuilt) = echo(&exec);
    assert_eq!(device, "gpu0");
    assert_ne!(rebuilt, pid, "the instance is rebuilt in a new worker, as an in-process panic rebuilds it");
    exec.shutdown();
}

fn the_worker_mode_serves_only_a_supervisor_that_greets_it() {
    // Not a socket: refused before anything is read.
    let out = std::process::Command::new(std::env::current_exe().unwrap()).arg("worker").stdin(std::process::Stdio::null()).output().expect("run the worker mode");
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("internal entry point"), "{out:?}");

    let wait = Duration::from_secs(10);
    let mut w = Worker::spawn_with(&worker_command(), REAP).unwrap();
    let err = w.request("open gpu0 {}", wait).expect("answered").expect_err("no greeting, no service");
    assert!(err.contains("hello"), "{err}");
    assert!(w.request("hello 1", wait).is_err(), "the worker ended after refusing");

    let mut w = Worker::spawn_with(&worker_command(), REAP).unwrap();
    let err = w.request("hello 999", wait).expect("answered").expect_err("another protocol version is refused");
    assert!(err.contains("hello 1"), "{err}");
    let mut w = Worker::spawn_with(&worker_command(), REAP).unwrap();
    w.request("hello 1", wait).expect("answered").expect("greeted");
    let err = w.request("open gpu0 {\"family\":\"other\",\"args\":{},\"key\":{\"model\":\"m\",\"config\":\"c\"}}", wait).expect("answered").expect_err("an unknown family");
    assert!(err.contains("\"other\""), "{err}");
}

const TESTS: &[(&str, fn())] = &[
    ("a_hosted_model_runs_in_its_worker_with_blobs_through_memfds", a_hosted_model_runs_in_its_worker_with_blobs_through_memfds),
    ("progress_and_cancellation_cross_the_boundary", progress_and_cancellation_cross_the_boundary),
    ("killing_the_worker_mid_job_fails_that_job_and_the_next_rebuilds_on_a_healthy_card", killing_the_worker_mid_job_fails_that_job_and_the_next_rebuilds_on_a_healthy_card),
    ("a_panic_in_model_code_fails_the_job_without_quarantining_the_card", a_panic_in_model_code_fails_the_job_without_quarantining_the_card),
    ("the_worker_mode_serves_only_a_supervisor_that_greets_it", the_worker_mode_serves_only_a_supervisor_that_greets_it),
];

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("worker") {
        std::process::exit(residency::supervised::host(FAMILIES));
    }
    // Plain words filter by name, as the standard harness does; its flags are ignored.
    let filters: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let chosen: Vec<_> = TESTS.iter().filter(|(name, _)| filters.is_empty() || filters.iter().any(|f| name.contains(f.as_str()))).collect();
    println!("running {} tests", chosen.len());
    let mut failed = Vec::new();
    for (name, test) in &chosen {
        match std::panic::catch_unwind(test) {
            Ok(()) => println!("test {name} ... ok"),
            Err(_) => {
                println!("test {name} ... FAILED");
                failed.push(*name);
            }
        }
    }
    println!("test result: {}. {} passed; {} failed", if failed.is_empty() { "ok" } else { "FAILED" }, chosen.len() - failed.len(), failed.len());
    if !failed.is_empty() {
        std::process::exit(101);
    }
}
