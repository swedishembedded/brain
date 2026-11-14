// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving models from killable
// worker processes without changing what they answer, for its clients. If
// your team needs expertise in fault-isolated model serving then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Decision models hosted in `brain`'s internal worker mode, against the
//! same models in process, on tiny random-weight checkpoints of both
//! architectures: the answers agree, a worker that dies fails the next job
//! definitively and quarantines its card while the job after it is built on
//! the healthy card, and what a worker costs per job is measured. Every test
//! but the first needs a GPU and skips, saying so, without one.

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

use capability::{ActionResult, Invocation};
use decisions::fixtures::{write_decide, write_laya, Scratch};
use decisions::resident::{serve_all, DecisionServeConfig};
use residency::budget::Budgets;
use residency::supervised::{Supervision, WorkerHosting};
use residency::{Device, Executor, Job, Policy, ResidentModel};
use serde_json::{json, Value};

const QUESTIONS: &str = r#"[
  {"id": "team", "type": "choice", "instruction": "who handles this",
   "options": [{"id": "billing", "description": "payments"}, "tech", "sales"]},
  {"id": "anger", "type": "score", "instruction": "how upset", "options": ["calm", "annoyed", "furious"]},
  {"id": "urgent", "type": "noul", "instruction": "is this urgent"}
]"#;

/// Both sides build the same graph on the same card.
const TOLERANCE: f64 = 1e-5;

fn brain() -> &'static str {
    env!("CARGO_BIN_EXE_brain")
}

#[test]
fn the_worker_mode_is_internal_and_refused_from_a_terminal() {
    let help = Command::new(brain()).arg("help").output().expect("run brain help");
    assert!(!String::from_utf8_lossy(&help.stdout).contains(serving::supervision::WORKER_MODE), "the internal worker mode is listed in `brain help`");
    let out = Command::new(brain()).arg(serving::supervision::WORKER_MODE).stdin(Stdio::null()).output().expect("run the worker mode");
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("internal entry point") && stderr.contains("brain serve"), "{stderr}");
}

fn store() -> Scratch {
    let s = Scratch::new("worker-hosted");
    write_laya(&s.join("acme/laya"), 48);
    write_decide(&s.join("acme/triage"), 3);
    s
}

fn cards(n: u32) -> Budgets {
    let mut budgets = Budgets::new();
    for i in 0..n {
        budgets.set(Device::Gpu(i), 8 << 30, 0);
    }
    budgets
}

fn supervision() -> Supervision {
    serving::ServeSupervision::new(Err("probes are paced by the breaker in this test".into())).supervision
}

fn hosting(supervision: &Supervision) -> WorkerHosting {
    serving::supervision::worker_hosting(Path::new(brain()), &[], &["decision".to_string()], Duration::from_secs(300), supervision.health.clone()).expect("decision models can be hosted")
}

/// The store's decision models in process, or each in a worker.
fn serve(store: &Scratch, cards_: u32, hosted: Option<&Supervision>) -> Executor {
    let hosting = hosted.map(hosting);
    let models: Vec<Arc<dyn ResidentModel>> = serve_all(&DecisionServeConfig::default(), Some(&**store), hosting.as_ref());
    Executor::start_supervised(models, cards(cards_), Policy::default(), None, hosted.cloned())
}

fn decide(exec: &Executor, model: &str) -> ActionResult {
    let (tx, rx) = channel();
    let inv = Invocation::new().set("state", json!("my payment failed again today")).set("questions", json!(QUESTIONS));
    exec.submit(Job::new(model, "decide", inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx.recv_timeout(Duration::from_secs(300)).expect("the job is answered")
}

/// Every number in `a` within the tolerance of the one at the same place in `b`, everything else equal.
fn agree(a: &Value, b: &Value, at: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() <= TOLERANCE, "{at}: in process {x}, in a worker {y}");
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{at}: lengths differ");
            for (i, (x, y)) in x.iter().zip(y).enumerate() {
                agree(x, y, &format!("{at}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.keys().collect::<Vec<_>>(), y.keys().collect::<Vec<_>>(), "{at}: fields differ");
            // How long the forward took is measured, not computed.
            for (k, x) in x.iter().filter(|(k, _)| k.as_str() != "latency_ms") {
                agree(x, &y[k], &format!("{at}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{at}"),
    }
}

/// The worker processes this process has started that are alive now.
fn live_workers() -> Vec<u32> {
    let mut out = Vec::new();
    for task in std::fs::read_dir("/proc/self/task").expect("procfs").flatten() {
        let Ok(children) = std::fs::read_to_string(task.path().join("children")) else { continue };
        for pid in children.split_whitespace().filter_map(|p| p.parse::<u32>().ok()) {
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            if cmdline.split(|&b| b == 0).any(|arg| arg == serving::supervision::WORKER_MODE.as_bytes()) {
                out.push(pid);
            }
        }
    }
    out
}

/// The GPU tests run one at a time: each counts the workers this process
/// started, and the cards are shared.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn no_gpu() -> bool {
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("worker-hosted decision models: no GPU (the decision models are GPU-only)");
        return true;
    }
    false
}

#[test]
fn a_worker_hosted_decision_model_answers_like_the_in_process_one() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    let store = store();
    for model in ["acme/laya", "acme/triage"] {
        let in_process = serve(&store, 1, None);
        let want = decide(&in_process, model).unwrap_or_else(|e| panic!("{model} in process: {e}"));
        in_process.shutdown();

        let supervision = supervision();
        let before = live_workers();
        let hosted = serve(&store, 1, Some(&supervision));
        let got = decide(&hosted, model).unwrap_or_else(|e| panic!("{model} in a worker: {e}"));
        assert!(live_workers().iter().any(|p| !before.contains(p)), "{model} ran in a worker this process started");
        agree(&want.outputs, &got.outputs, model);
        assert!(want.blobs.is_empty() && got.blobs.is_empty());
        hosted.shutdown();
    }
}

#[test]
fn a_dead_worker_fails_its_job_quarantines_its_card_and_the_next_job_is_built_on_the_healthy_one() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    if gpu_core::devices::gpus().len() < 2 {
        brain_testutil::skip_unavailable("worker-hosted decision models: one GPU (moving to a healthy card needs two)");
        return;
    }
    let store = store();
    let supervision = supervision();
    let before = live_workers();
    let exec = serve(&store, 2, Some(&supervision));
    decide(&exec, "acme/laya").expect("the first job runs");
    let workers: Vec<u32> = live_workers().into_iter().filter(|p| !before.contains(p)).collect();
    assert_eq!(workers.len(), 1, "{workers:?}");
    let sick = exec.residency().placements.iter().find(|p| p.key.model == "acme/laya").map(|p| p.device).expect("acme/laya is resident");
    let killed = Command::new("kill").arg("-KILL").arg(workers[0].to_string()).status().expect("kill");
    assert!(killed.success());

    let started = Instant::now();
    let err = decide(&exec, "acme/laya").expect_err("a job sent to a dead worker fails");
    assert!(started.elapsed() < Duration::from_secs(10), "the failure took {:?}", started.elapsed());
    assert!(err.contains("lost its worker") && err.contains("quarantined"), "{err}");
    assert_eq!(supervision.health.faults(&sick), 1, "the worker's death is recorded against its card, once");

    // Without a probe program the restart is readmitted at once; trip the
    // breaker so the card stays out and the rebuild must go elsewhere.
    for _ in 0..3 {
        supervision.health.record_fault(&sick, Instant::now(), "a test keeps the card out");
    }
    decide(&exec, "acme/laya").expect("the next job is built in a new worker on the healthy card");
    let moved = exec.residency().placements.iter().find(|p| p.key.model == "acme/laya").map(|p| p.device).expect("acme/laya is resident again");
    assert_ne!(moved, sick, "rebuilt on the healthy card");
    exec.shutdown();
}

/// The median of `n` decide jobs on `model`, after one that builds the instance.
fn median_job(exec: &Executor, model: &str, n: usize) -> Duration {
    decide(exec, model).expect("warm-up");
    let mut took: Vec<Duration> = (0..n)
        .map(|_| {
            let started = Instant::now();
            decide(exec, model).expect("decide");
            started.elapsed()
        })
        .collect();
    took.sort();
    took[n / 2]
}

/// What hosting in a worker costs per job: the same decide job through the
/// executor, in process and in a worker, on the same card.
#[test]
fn a_worker_hosted_job_costs_a_round_trip_more_than_in_process() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    let store = store();
    const JOBS: usize = 60;
    let in_process = serve(&store, 1, None);
    let local = median_job(&in_process, "acme/triage", JOBS);
    in_process.shutdown();
    let supervision = supervision();
    let hosted = serve(&store, 1, Some(&supervision));
    let worker = median_job(&hosted, "acme/triage", JOBS);
    hosted.shutdown();
    eprintln!("decide job (tiny decide checkpoint, {JOBS} jobs): median {local:?} in process, {worker:?} in a worker");
    assert!(worker < local + Duration::from_millis(50), "a worker added {:?} per job", worker.saturating_sub(local));
}
