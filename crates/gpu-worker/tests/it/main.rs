// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for recovering accelerator-backed
// services from wedged and crashing device drivers for its clients. If your
// team needs expertise in supervised GPU execution then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The real worker on a real card: the supervisor's probe, the executor
//! running a job through a supervised instance, and a quarantined card
//! readmitted by a real probe. Each uses a few kilobytes of one card for well
//! under a second, and skips, saying why, where no card can be opened.

use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

use capability::{ActionSpec, Invocation, Manifest};
use gpu_supervisor::{BreakerPolicy, BreakerState, CallError, Health, Worker, WorkerCommand};
use residency::budget::Budgets;
use residency::supervised::{Deadlines, Probe, SupervisedModel, Supervision};
use residency::{Device, Executor, Job, MemCost, Policy};

/// Device creation and pipeline compilation on a cold driver take seconds.
const DEADLINE: Duration = Duration::from_secs(120);
const REAP: Duration = Duration::from_secs(20);

fn worker() -> WorkerCommand {
    WorkerCommand::new(env!("CARGO_BIN_EXE_brain-gpu-worker"))
}

fn health() -> Arc<Health<Device>> {
    Arc::new(Health::new(BreakerPolicy::default(), REAP))
}

/// Probe gpu0 in a fresh worker; `false` (after printing why) when the host
/// has no card the worker can open.
fn gpu0_available() -> bool {
    let health: Health<Device> = Health::new(BreakerPolicy::default(), REAP);
    let mut w = Worker::spawn_with(&worker(), REAP).expect("start brain-gpu-worker");
    match health.call(&Device::Gpu(0), &mut w, "probe gpu0", DEADLINE) {
        Ok(_) => true,
        Err(CallError::Failed(why)) if why.starts_with("no-device") => {
            brain_testutil::skip_unavailable(&format!("no GPU the supervised worker can open: {why}"));
            false
        }
        Err(e) => panic!("the probe of gpu0 failed: {e}"),
    }
}

#[test]
fn a_job_runs_on_a_real_card_through_a_supervised_worker() {
    if !gpu0_available() {
        return;
    }
    let health = health();
    let manifest = Manifest::new("gpu-probe", "a real dispatch on a supervised card", vec![ActionSpec::new("probe", "dispatch one kernel")]);
    let deadlines = Deadlines { open: DEADLINE, run: DEADLINE, reap: REAP };
    let model = Arc::new(SupervisedModel::new(manifest, worker(), health.clone(), MemCost::new(64 << 20, 0), deadlines));
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 1 << 30, 0);
    let supervision = Supervision { health: health.clone(), probe: Probe::Worker { command: worker(), deadline: DEADLINE } };
    let exec = Executor::start_supervised(vec![model], budgets, Policy::default(), None, Some(supervision));
    let (tx, rx) = channel();
    exec.submit(Job::new("gpu-probe", "probe", Invocation::new()).reply(move |r| {
        let _ = tx.send(r);
    }));
    let out = rx.recv_timeout(DEADLINE * 2).expect("answered").expect("the supervised dispatch succeeds");
    assert_eq!(out.outputs["sum"], serde_json::json!([11.0, 22.0, 33.0, 44.0]), "{:?}", out.outputs);
    assert_eq!(health.status(&Device::Gpu(0)).state, BreakerState::Closed);
    exec.shutdown();
}

#[test]
fn a_quarantined_real_card_is_readmitted_by_a_real_probe() {
    if !gpu0_available() {
        return;
    }
    let health = health();
    health.record_fault(&Device::Gpu(0), Instant::now(), "an in-process wait ran past (--gpu-wait-s)");
    assert!(health.begin_probe(&Device::Gpu(0), Instant::now()));
    health.run_probe(&Device::Gpu(0), &worker(), "probe gpu0", DEADLINE).expect("a healthy card passes its probe");
    assert_eq!(health.status(&Device::Gpu(0)).state, BreakerState::Closed);
}
