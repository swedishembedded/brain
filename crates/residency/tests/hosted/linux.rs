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
use residency::{Device, Executor, Instance, InstanceKey, Job, JobFeed, MemCost, Policy, ResidentModel};
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
        let actions = ["echo", "stream", "slow", "wait", "block", "panic", "batch", "open"].map(|a| ActionSpec::new(a, a)).to_vec();
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
            // A long job that keeps reporting: longer in all than a short run deadline.
            "slow" => {
                for i in 1..=8 {
                    std::thread::sleep(Duration::from_millis(150));
                    progress(Progress::step(i, 8, "working"));
                }
                Ok(Outcome::new().set("steps", json!(8)))
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

    /// `open` is a model that batches continuously: between steps it takes
    /// the jobs that joined, turns away the waiting ones asked to be `never`
    /// run, finishes each job as it is done, and ends once the feed closes. A job asked to `wait` waits for its cancellation; one
    /// asked to `block` never returns, as a wedged driver call.
    fn run_open(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        if action != "open" {
            return false;
        }
        let mut jobs: Vec<Invocation> = invs.to_vec();
        let mut finished = 0;
        loop {
            jobs.extend(feed.take(4));
            feed.screen(&mut |inv| inv.params.get("never").map(|_| format!("job {} can never run here", inv.params["n"])));
            for (i, inv) in jobs.iter().enumerate().skip(finished) {
                let mut step = Progress::step(1, 1, format!("job {i}"));
                step.event = Some(json!({ "pid": std::process::id() }));
                progress(i, step);
                if inv.params.get("block").is_some() {
                    loop {
                        std::thread::sleep(Duration::from_secs(3600));
                    }
                }
                if inv.params.get("wait").is_some() {
                    let until = Instant::now() + Duration::from_secs(30);
                    while !inv.cancel.is_cancelled() && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
                let result = inv.cancel.check().map(|()| Outcome::new().set("index", json!(i)).set("pid", json!(std::process::id())).set("params", inv.params.clone()));
                feed.finish(i, result);
            }
            finished = jobs.len();
            if feed.close() {
                return true;
            }
        }
    }

    /// `batch` is a model that batches: one call sees the whole group, each
    /// job reports under its own index, and a job asked to `wait` waits for
    /// its own cancellation.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action != "batch" {
            return invs.iter().enumerate().map(|(i, inv)| inv.cancel.check().and_then(|()| self.run(action, inv, &mut |p| progress(i, p)))).collect();
        }
        invs.iter()
            .enumerate()
            .map(|(i, inv)| {
                progress(i, Progress::step(1, 1, format!("job {i}")));
                if inv.params.get("wait").is_some() {
                    let until = Instant::now() + Duration::from_secs(30);
                    while !inv.cancel.is_cancelled() && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    inv.cancel.check()?;
                }
                let mut out = Outcome::new().set("index", json!(i)).set("group", json!(invs.len())).set("pid", json!(std::process::id())).set("params", inv.params.clone());
                for (name, blob) in &inv.blobs {
                    out = out.blob(&format!("out_{name}"), Blob::new(blob.media, blob.bytes.iter().rev().copied().collect()).with_meta(blob.meta.clone()));
                }
                Ok(out)
            })
            .collect()
    }
}

fn worker_command() -> WorkerCommand {
    WorkerCommand::new(std::env::current_exe().expect("own path")).arg("worker")
}

fn health() -> Arc<Health<Device>> {
    let policy = BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 };
    Arc::new(Health::new(policy, REAP))
}

fn hosted(id: &str, health: &Arc<Health<Device>>, deadlines: Deadlines) -> Arc<dyn ResidentModel> {
    let spec = HostSpec { family: "test".into(), args: json!({ "id": id }) };
    Arc::new(SupervisedModel::new(Arc::new(TestModel(id.to_string())), spec, worker_command(), health.clone(), deadlines))
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
    start_with(health, cards, DEADLINES)
}

fn start_with(health: &Arc<Health<Device>>, cards: u32, deadlines: Deadlines) -> Executor {
    let supervision = Supervision { health: health.clone(), probe: failing_probe() };
    Executor::start_supervised(vec![hosted("m", health, deadlines)], gpus(cards), Policy::default(), None, Some(supervision))
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

/// A group crosses as one request: the worker's model sees all of it in one
/// `run_batch` call, each job's progress, result and blobs come back under
/// its own index (more blobs than one frame may carry descriptors), and a
/// cancellation reaches only the job it was asked for.
fn a_group_crosses_as_one_batch_with_per_job_progress_results_and_cancellation() {
    let health = health();
    let mut instance = hosted("m", &health, DEADLINES).activate(&InstanceKey::new("m", "default"), Device::Gpu(0)).expect("the worker opens");
    const JOBS: usize = 40;
    let invs: Vec<Invocation> = (0..JOBS)
        .map(|i| {
            let mut inv = Invocation::new().set("n", json!(i)).blob("payload", Blob::new(Media::Bytes, vec![i as u8, 1, 2]));
            if i == 7 {
                inv = inv.set("wait", json!(true));
                inv.cancel = CancelToken::armed();
                inv.cancel.cancel();
            }
            inv
        })
        .collect();
    let mut seen: Vec<Vec<String>> = vec![Vec::new(); JOBS];
    let results = instance.run_batch("batch", &invs, &mut |i, p| seen[i].push(p.message));
    assert_eq!(results.len(), JOBS);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(seen[i], [format!("job {i}")], "job {i}'s progress went to its own caller");
        if i == 7 {
            assert_eq!(r.as_ref().err().map(String::as_str), Some(failure::CANCELLED), "the job asked to cancel was cancelled");
            continue;
        }
        let out = r.as_ref().unwrap_or_else(|e| panic!("job {i}: {e}"));
        assert_eq!((out.outputs["index"].clone(), out.outputs["group"].clone(), out.outputs["params"]["n"].clone()), (json!(i), json!(JOBS), json!(i)), "one run_batch call saw the whole group");
        assert_ne!(out.outputs["pid"].as_u64(), Some(u64::from(std::process::id())));
        assert_eq!(out.blobs["out_payload"].bytes, [2, 1, i as u8], "job {i}'s blob came back to job {i}");
    }
    assert_eq!(health.status(&Device::Gpu(0)).faults, 0);
}

/// The lane's side of an open group, as the executor's inbox would be:
/// rounds of late jobs handed out one round per `take`, each job's result as
/// it finished, and how often a close was refused because a job had slipped
/// in.
struct TestFeed {
    rounds: Vec<Vec<Invocation>>,
    /// A close is refused while this is above zero, once per count.
    refuse_closes: usize,
    finished: Vec<(usize, ActionResult)>,
    closes: usize,
    /// Jobs waiting behind the group, by ticket, handed out to be screened once.
    waiting: Vec<(u64, Invocation)>,
    refused: Vec<(u64, String)>,
}

impl TestFeed {
    fn new(rounds: Vec<Vec<Invocation>>, refuse_closes: usize) -> TestFeed {
        TestFeed { rounds, refuse_closes, finished: Vec::new(), closes: 0, waiting: Vec::new(), refused: Vec::new() }
    }
}

impl JobFeed for TestFeed {
    fn take(&mut self, max: usize) -> Vec<Invocation> {
        assert!(max > 0);
        if self.rounds.is_empty() {
            Vec::new()
        } else {
            self.rounds.remove(0)
        }
    }

    fn finish(&mut self, index: usize, result: ActionResult) {
        assert!(self.finished.iter().all(|(i, _)| *i != index), "job {index} was finished twice");
        self.finished.push((index, result));
    }

    fn unscreened(&mut self) -> Vec<(u64, Invocation)> {
        std::mem::take(&mut self.waiting)
    }

    fn refuse(&mut self, ticket: u64, why: String) {
        assert!(self.refused.iter().all(|(t, _)| *t != ticket), "ticket {ticket} was refused twice");
        self.refused.push((ticket, why));
    }

    fn close(&mut self) -> bool {
        self.closes += 1;
        if self.refuse_closes > 0 {
            self.refuse_closes -= 1;
            return false;
        }
        true
    }
}

fn result_of(feed: &TestFeed, index: usize) -> &ActionResult {
    &feed.finished.iter().find(|(i, _)| *i == index).unwrap_or_else(|| panic!("job {index} was never finished")).1
}

/// An open group runs in the worker's `run_open`: jobs that join while it
/// runs cross as they are taken, each is finished through the parent's feed
/// the moment it is done, a close the parent refuses (a job slipped in)
/// keeps the group going, and a late job's cancellation reaches it. A model
/// that only batches closed groups says so, having done nothing.
fn an_open_group_takes_late_joiners_and_finishes_each_job_through_the_feed() {
    let health = health();
    let mut instance = hosted("m", &health, DEADLINES).activate(&InstanceKey::new("m", "default"), Device::Gpu(0)).expect("the worker opens");
    let mut cancelled = Invocation::new().set("n", json!(3)).set("wait", json!(true));
    cancelled.cancel = CancelToken::armed();
    cancelled.cancel.cancel();
    let mut feed = TestFeed::new(vec![vec![Invocation::new().set("n", json!(2)), cancelled], vec![Invocation::new().set("n", json!(4))]], 1);
    let own = [Invocation::new().set("n", json!(0)), Invocation::new().set("n", json!(1))];
    let mut seen: Vec<(usize, String)> = Vec::new();
    assert!(instance.run_open("open", &own, &mut |i, p| seen.push((i, p.message)), &mut feed), "the worker ran the group");
    assert_eq!(feed.finished.len(), 5, "every job, own and late, was finished: {:?}", feed.finished.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    for (i, n) in [(0, 0), (1, 1), (2, 2), (4, 4)] {
        let out = result_of(&feed, i).as_ref().unwrap_or_else(|e| panic!("job {i}: {e}"));
        assert_eq!((out.outputs["index"].clone(), out.outputs["params"]["n"].clone()), (json!(i), json!(n)), "job {i} is the one that joined as {i}");
        assert_ne!(out.outputs["pid"].as_u64(), Some(u64::from(std::process::id())));
        assert!(seen.contains(&(i, format!("job {i}"))), "job {i}'s progress reached its caller");
    }
    assert_eq!(result_of(&feed, 3).as_ref().err().map(String::as_str), Some(failure::CANCELLED), "the late job asked to cancel was cancelled");
    assert_eq!(feed.closes, 2, "the refused close kept the group going");

    let mut feed = TestFeed::new(Vec::new(), 0);
    assert!(!instance.run_open("batch", &own, &mut |_, _| {}, &mut feed), "a model that only batches closed groups runs none open");
    assert!(feed.finished.is_empty() && feed.closes == 0, "and did nothing");
    assert_eq!(health.status(&Device::Gpu(0)).faults, 0);
}

/// The jobs waiting behind an open group cross to the worker to be screened
/// once: the one its model can never run is refused through the parent's
/// feed, by its ticket and with the model's reason, while the group runs;
/// the one it can run keeps waiting for a slot, and the group's own jobs are
/// unaffected.
fn an_open_group_screens_the_jobs_waiting_to_join_it_in_its_worker() {
    let health = health();
    let mut instance = hosted("m", &health, DEADLINES).activate(&InstanceKey::new("m", "default"), Device::Gpu(0)).expect("the worker opens");
    let mut feed = TestFeed::new(Vec::new(), 0);
    feed.waiting = vec![(17, Invocation::new().set("n", json!(5)).set("never", json!(true))), (18, Invocation::new().set("n", json!(6)))];
    let own = [Invocation::new().set("n", json!(0))];
    assert!(instance.run_open("open", &own, &mut |_, _| {}, &mut feed), "the worker ran the group");
    assert_eq!(feed.refused, [(17, "job 5 can never run here".to_string())], "only the job that can never run was refused, with the model's reason");
    assert!(result_of(&feed, 0).is_ok(), "the group's own job ran");
    assert_eq!(feed.finished.len(), 1, "a refused job is not one of the group's");
    assert_eq!(health.status(&Device::Gpu(0)).faults, 0);
}

/// A worker killed while an open group runs fails every job of the group
/// that was not finished yet, the late ones included, and only those, and
/// quarantines its card.
fn a_worker_killed_mid_group_fails_its_unfinished_jobs_once() {
    let health = health();
    let mut instance = hosted("m", &health, DEADLINES).activate(&InstanceKey::new("m", "default"), Device::Gpu(0)).expect("the worker opens");
    let mut feed = TestFeed::new(vec![vec![Invocation::new().set("block", json!(true))]], 0);
    let own = [Invocation::new().set("n", json!(0))];
    let handled = instance.run_open("open", &own, &mut |i, p| {
        if i == 1 {
            // The late job is inside its blocked step: the worker dies there.
            let worker = p.event.as_ref().and_then(|e| e["pid"].as_u64()).expect("the step names its worker");
            let killed = std::process::Command::new("kill").arg("-KILL").arg(worker.to_string()).status();
            assert!(killed.is_ok_and(|s| s.success()), "kill {worker}");
        }
    }, &mut feed);
    assert!(handled);
    assert!(result_of(&feed, 0).is_ok(), "the job finished before the crash keeps its answer");
    let err = result_of(&feed, 1).as_ref().expect_err("the job the worker died in fails");
    assert!(err.contains("lost its worker") && err.contains("quarantined"), "{err}");
    assert_eq!(feed.finished.len(), 2);
    assert!(!health.admits(&Device::Gpu(0)), "the card is quarantined");
}

/// The run deadline bounds a worker's silence: a job that keeps reporting
/// progress runs past it, and one that goes silent is killed at it and
/// quarantines its card.
fn a_job_that_keeps_reporting_outlives_the_run_deadline_and_a_silent_one_is_killed_at_it() {
    let health = health();
    let deadline = Duration::from_millis(600);
    let exec = start_with(&health, 1, Deadlines { run: deadline, ..DEADLINES });
    let (_, pid) = echo(&exec);
    let started = Instant::now();
    let out = run(&exec, "slow", Invocation::new(), |_| {}).expect("a job that reports progress is not wedged");
    assert_eq!(out.outputs["steps"], json!(8));
    assert!(started.elapsed() > deadline, "the job ran {:?}, not past one deadline", started.elapsed());
    assert_eq!(echo(&exec).1, pid, "the same worker serves on");

    let started = Instant::now();
    let err = run(&exec, "block", Invocation::new(), |_| {}).expect_err("a silent job is wedged");
    assert!(started.elapsed() < deadline + Duration::from_secs(5), "the failure took {:?}", started.elapsed());
    assert!(err.contains("deadline") && err.contains("quarantined"), "{err}");
    assert!(!health.admits(&Device::Gpu(0)), "the card is quarantined");
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
    assert!(w.request(&format!("hello {}", residency::supervised::PROTOCOL), wait).is_err(), "the worker ended after refusing");

    let mut w = Worker::spawn_with(&worker_command(), REAP).unwrap();
    let err = w.request("hello 999", wait).expect("answered").expect_err("another protocol version is refused");
    assert!(err.contains(&format!("hello {}", residency::supervised::PROTOCOL)), "{err}");
    let mut w = Worker::spawn_with(&worker_command(), REAP).unwrap();
    w.request(&format!("hello {}", residency::supervised::PROTOCOL), wait).expect("answered").expect("greeted");
    let err = w.request("open gpu0 {\"family\":\"other\",\"args\":{},\"key\":{\"model\":\"m\",\"config\":\"c\"}}", wait).expect("answered").expect_err("an unknown family");
    assert!(err.contains("\"other\""), "{err}");
}

const TESTS: &[(&str, fn())] = &[
    ("a_hosted_model_runs_in_its_worker_with_blobs_through_memfds", a_hosted_model_runs_in_its_worker_with_blobs_through_memfds),
    ("progress_and_cancellation_cross_the_boundary", progress_and_cancellation_cross_the_boundary),
    ("killing_the_worker_mid_job_fails_that_job_and_the_next_rebuilds_on_a_healthy_card", killing_the_worker_mid_job_fails_that_job_and_the_next_rebuilds_on_a_healthy_card),
    ("a_group_crosses_as_one_batch_with_per_job_progress_results_and_cancellation", a_group_crosses_as_one_batch_with_per_job_progress_results_and_cancellation),
    ("an_open_group_takes_late_joiners_and_finishes_each_job_through_the_feed", an_open_group_takes_late_joiners_and_finishes_each_job_through_the_feed),
    ("an_open_group_screens_the_jobs_waiting_to_join_it_in_its_worker", an_open_group_screens_the_jobs_waiting_to_join_it_in_its_worker),
    ("a_worker_killed_mid_group_fails_its_unfinished_jobs_once", a_worker_killed_mid_group_fails_its_unfinished_jobs_once),
    ("a_job_that_keeps_reporting_outlives_the_run_deadline_and_a_silent_one_is_killed_at_it", a_job_that_keeps_reporting_outlives_the_run_deadline_and_a_silent_one_is_killed_at_it),
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
