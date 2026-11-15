// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving batching models from
// killable worker processes without changing what they answer, for its
// clients. If your team needs expertise in fault-isolated model serving then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! LFM2.5 encoders hosted in `brain`'s internal worker mode, against the
//! same encoder in process, on a tiny random-weight checkpoint: a group
//! batched in the worker answers as the in-process batch does, a worker
//! killed while a group streams fails that group and quarantines its card, a
//! dead worker's next job is built in a new worker on the healthy card, and
//! what a worker costs per job and per card is measured. Every test needs a
//! GPU and skips, saying so, without one.

use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

use capability::{ActionResult, Invocation, Progress};
use catalog::resident_lfm::LfmResident;
use checkpoint::st::ModelCard;
use decisions::fixtures::Scratch;
use residency::supervised::{Supervision, WorkerHosting};
use residency::{Device, Executor, Instance, Job, Policy, ResidentModel};
use serde_json::json;

use super::worker_hosted::{brain, cards, live_workers, serial, supervision};

const MODEL: &str = "acme/lfm-enc";

/// Both sides run the same graph on the same card.
const TOLERANCE: f32 = 1e-5;

/// A tiny LFM2.5 encoder with random weights and a byte-level tokenizer.
struct Checkpoint {
    _dir: Scratch,
    weights: PathBuf,
    tokenizer: PathBuf,
}

fn checkpoint() -> Checkpoint {
    let dir = Scratch::new("worker-hosted-lfm");
    // A byte-level vocabulary of 256 and room for prompts of a few dozen bytes.
    let cfg = lfm2::config::LfmConfig { vocab: 256, block_size: 64, d_model: 64, head_dim: 16, d_ff: 128, ..lfm2::config::LfmConfig::tiny() };
    let init = lfm2::init::init_weights(&cfg, data::rng::random_seed());
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init[&name].clone())).collect();
    let weights = dir.join("enc.safetensors");
    checkpoint::save(weights.to_str().unwrap(), cfg.to_json(), &tensors);
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let tokenizer = dir.join("tokenizer.json");
    std::fs::write(&tokenizer, json!({ "model": { "vocab": vocab, "merges": [] } }).to_string()).unwrap();
    Checkpoint { _dir: dir, weights, tokenizer }
}

fn resident(c: &Checkpoint, batch: u32) -> LfmResident {
    LfmResident::from_card(c.weights.to_str().unwrap(), &ModelCard::new(MODEL, "lfm"), c.tokenizer.to_str(), batch).expect("the tiny encoder registers")
}

fn hosting(supervision: &Supervision) -> WorkerHosting {
    serving::supervision::worker_hosting(Path::new(brain()), &[], &[catalog::resident_lfm::FAMILY.to_string()], Duration::from_secs(300), supervision.health.clone()).expect("LFM2.5 encoders can be hosted")
}

/// The encoder in process, or in a worker when `hosting` is given.
fn served(c: &Checkpoint, batch: u32, hosting: Option<&WorkerHosting>) -> Arc<dyn ResidentModel> {
    let r = resident(c, batch);
    let spec = r.host_spec();
    WorkerHosting::serve(hosting, Arc::new(r), spec)
}

/// `n` embed jobs of equal length, so they share one instance and batch.
fn texts(n: usize) -> Vec<Invocation> {
    (0..n).map(|i| Invocation::new().set("text", json!(format!("job {:02} says hi", i % 100)))).collect()
}

fn no_gpu() -> bool {
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("worker-hosted LFM2.5 encoders: no GPU (the encoder runs on a card)");
        return true;
    }
    false
}

/// Build `model`'s instance for `invs` on `device` and run them as one group,
/// with each job's progress messages.
fn run_group(model: &Arc<dyn ResidentModel>, invs: &[Invocation], device: Device) -> (Vec<ActionResult>, Vec<Vec<String>>, Box<dyn Instance>) {
    let key = model.instance_key("embed", &invs[0]);
    let mut instance = model.activate(&key, device).expect("the encoder builds");
    let mut seen = vec![Vec::new(); invs.len()];
    let results = instance.run_batch("embed", invs, &mut |i, p: Progress| seen[i].push(p.message));
    (results, seen, instance)
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= TOLERANCE, "{what}[{i}]: in process {x}, in a worker {y}");
    }
}

/// A group of six equal-length jobs at two per forward: the worker's own
/// `run_batch` runs all six in three batched forwards, as in process, and
/// every job's embedding and progress agree.
#[test]
fn a_worker_hosted_encoder_batches_a_group_and_answers_like_the_in_process_one() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    let c = checkpoint();
    let invs = texts(6);
    let (local, local_seen, local_instance) = run_group(&served(&c, 2, None), &invs, Device::Gpu(0));
    drop(local_instance);

    let supervision = supervision();
    let before = live_workers();
    let (hosted, hosted_seen, hosted_instance) = run_group(&served(&c, 2, Some(&hosting(&supervision))), &invs, Device::Gpu(0));
    assert!(live_workers().iter().any(|p| !before.contains(p)), "the group ran in a worker this process started");
    drop(hosted_instance);

    for (i, (want, got)) in local.iter().zip(&hosted).enumerate() {
        let (want, got) = (want.as_ref().unwrap_or_else(|e| panic!("job {i} in process: {e}")), got.as_ref().unwrap_or_else(|e| panic!("job {i} in a worker: {e}")));
        assert_eq!((&want.outputs["tokens"], &want.outputs["dim"]), (&got.outputs["tokens"], &got.outputs["dim"]), "job {i}");
        let mean = |o: &capability::Outcome| o.outputs["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<_>>();
        assert_close(&mean(want), &mean(got), &format!("job {i} mean"));
        assert_close(&floats(&want.blobs["embeddings"].bytes), &floats(&got.blobs["embeddings"].bytes), &format!("job {i} embeddings"));
        assert_eq!(want.blobs["embeddings"].meta, got.blobs["embeddings"].meta);
    }
    let batches = vec!["batch 1/3".to_string(), "batch 2/3".to_string(), "batch 3/3".to_string()];
    for (i, (want, got)) in local_seen.iter().zip(&hosted_seen).enumerate() {
        assert_eq!(want, &batches, "job {i} in process: one run_batch of three forwards");
        assert_eq!(got, &batches, "job {i} in a worker: one run_batch of three forwards");
    }
}

/// A worker killed while it streams a group's progress fails every job of
/// that group, naming the signal, and quarantines its card; the instance
/// answers later jobs at once with the same error.
#[test]
fn a_worker_killed_mid_stream_fails_its_group_and_quarantines_its_card() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    let c = checkpoint();
    let supervision = supervision();
    let model = served(&c, 2, Some(&hosting(&supervision)));
    // Sixty-four jobs at two per forward: the worker has 31 forwards left when
    // the first progress arrives, so the kill lands inside the group.
    let invs = texts(64);
    let key = model.instance_key("embed", &invs[0]);
    let before = live_workers();
    let mut instance = model.activate(&key, Device::Gpu(0)).expect("the encoder builds in a worker");
    let worker = live_workers().into_iter().find(|p| !before.contains(p)).expect("a worker this process started");
    let mut killed = false;
    let started = Instant::now();
    let results = instance.run_batch("embed", &invs, &mut |_, _| {
        if !killed {
            killed = true;
            // SAFETY: kill(2) on a process this test started.
            assert_eq!(unsafe { libc::kill(worker as libc::pid_t, libc::SIGKILL) }, 0, "kill {worker}");
        }
    });
    assert!(killed, "the group reported progress");
    assert!(started.elapsed() < Duration::from_secs(10), "the failure took {:?}", started.elapsed());
    assert_eq!(results.len(), invs.len());
    for (i, r) in results.iter().enumerate() {
        let err = r.as_ref().expect_err("a job whose worker died mid-group fails");
        for needle in ["lost its worker", "signal 9", "quarantined"] {
            assert!(err.contains(needle), "job {i}'s error must say {needle:?}: {err}");
        }
    }
    assert_eq!(supervision.health.faults(&Device::Gpu(0)), 1, "the death is recorded against its card, once");
    assert!(!supervision.health.admits(&Device::Gpu(0)), "the card is quarantined");
    let again = instance.run_batch("embed", &invs[..1], &mut |_, _| {});
    assert!(again[0].as_ref().is_err_and(|e| e.contains("lost its worker")), "the dead instance answers at once: {again:?}");
}

fn embed(exec: &Executor, inv: Invocation) -> ActionResult {
    let (tx, rx) = channel();
    exec.submit(Job::new(MODEL, "embed", inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx.recv_timeout(Duration::from_secs(300)).expect("the job is answered")
}

/// Through the executor: a worker killed between jobs fails the next job and
/// records one fault on its card, and once that card is tripped the job
/// after is built in a new worker on the other card.
#[test]
fn a_dead_encoder_worker_fails_its_job_and_the_next_is_built_on_the_healthy_card() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    if gpu_core::devices::gpus().len() < 2 {
        brain_testutil::skip_unavailable("worker-hosted LFM2.5 encoders: one GPU (moving to a healthy card needs two)");
        return;
    }
    let c = checkpoint();
    let supervision = supervision();
    let before = live_workers();
    let exec = Executor::start_supervised(vec![served(&c, 2, Some(&hosting(&supervision)))], cards(2), Policy::default(), None, Some(supervision.clone()));
    embed(&exec, texts(1).remove(0)).expect("the first job runs");
    let workers: Vec<u32> = live_workers().into_iter().filter(|p| !before.contains(p)).collect();
    assert_eq!(workers.len(), 1, "{workers:?}");
    let sick = exec.residency().placements.iter().find(|p| p.key.model == MODEL).map(|p| p.device).expect("the encoder is resident");
    assert!(std::process::Command::new("kill").arg("-KILL").arg(workers[0].to_string()).status().expect("kill").success());

    let err = embed(&exec, texts(1).remove(0)).expect_err("a job sent to a dead worker fails");
    assert!(err.contains("lost its worker") && err.contains("quarantined"), "{err}");
    assert_eq!(supervision.health.faults(&sick), 1);
    for _ in 0..3 {
        supervision.health.record_fault(&sick, Instant::now(), "a test keeps the card out");
    }
    embed(&exec, texts(1).remove(0)).expect("the next job is built in a new worker on the healthy card");
    let moved = exec.residency().placements.iter().find(|p| p.key.model == MODEL).map(|p| p.device).expect("resident again");
    assert_ne!(moved, sick, "rebuilt on the healthy card");
    exec.shutdown();
}

/// The median of `n` single embed jobs through `exec`, after a warm-up that builds the instance.
fn median_job(exec: &Executor, n: usize) -> Duration {
    embed(exec, texts(1).remove(0)).expect("warm-up");
    let mut took: Vec<Duration> = (0..n)
        .map(|_| {
            let started = Instant::now();
            embed(exec, texts(1).remove(0)).expect("embed");
            started.elapsed()
        })
        .collect();
    took.sort();
    took[n / 2]
}

/// The median of `n` runs of an eight-job group on `instance`.
fn median_group(instance: &mut Box<dyn Instance>, n: usize) -> Duration {
    let invs = texts(8);
    let mut took: Vec<Duration> = (0..n)
        .map(|_| {
            let started = Instant::now();
            let results = instance.run_batch("embed", &invs, &mut |_, _| {});
            assert!(results.iter().all(Result::is_ok));
            started.elapsed()
        })
        .collect();
    took.sort();
    took[n / 2]
}

/// Device memory in use across every card, in MiB, as the driver reports
/// it (`None` without `nvidia-smi`). The difference across a build is what
/// that build holds, while nothing else allocates on the cards.
fn cards_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().map(|l| l.trim().parse::<u64>().ok()).sum()
}

/// Resident host memory of `pid`, in MiB.
pub(super) fn resident_mib(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kib: u64 = status.lines().find_map(|l| l.strip_prefix("VmRSS:"))?.split_whitespace().next()?.parse().ok()?;
    Some(kib / 1024)
}

/// [`cards_used_mib`] once this process has no worker left and the driver's
/// count has had time to follow the last release.
fn settled_used_mib() -> Option<u64> {
    let until = Instant::now() + Duration::from_secs(10);
    while !live_workers().is_empty() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(500));
    cards_used_mib()
}

fn mib(m: Option<i64>) -> String {
    m.map_or("an unreported amount".to_string(), |m| format!("{m} MiB"))
}

/// What hosting an encoder in a worker costs: per job and per eight-job
/// group against the same work in process on the same card, and the device
/// and host memory a worker holds for its card against what the same
/// instance adds to a process that already has the card open.
#[test]
fn a_worker_hosted_encoder_costs_a_round_trip_per_group_and_a_context_per_card() {
    let _serial = serial();
    if no_gpu() {
        return;
    }
    let c = checkpoint();
    const JOBS: usize = 60;
    const GROUPS: usize = 30;
    let delta = |before: Option<u64>| -> Option<i64> {
        std::thread::sleep(Duration::from_millis(500));
        Some(cards_used_mib()? as i64 - before? as i64)
    };
    let in_process = Executor::start_supervised(vec![served(&c, 2, None)], cards(1), Policy::default(), None, None);
    let local_job = median_job(&in_process, JOBS);
    in_process.shutdown();
    let local_model = served(&c, 2, None);
    let before = settled_used_mib();
    let mut local = local_model.activate(&local_model.instance_key("embed", &texts(1)[0]), Device::Gpu(0)).expect("builds");
    warm_group(&mut local);
    let local_group = median_group(&mut local, GROUPS);
    let first_device = delta(before);
    // A second instance on the card this process already has open: what an
    // instance costs without a device context of its own.
    let before = settled_used_mib();
    let mut second = local_model.activate(&local_model.instance_key("embed", &texts(1)[0]), Device::Gpu(0)).expect("builds");
    warm_group(&mut second);
    let instance_device = delta(before);
    drop(second);
    drop(local);

    let supervision = supervision();
    let hosting = hosting(&supervision);
    let hosted_exec = Executor::start_supervised(vec![served(&c, 2, Some(&hosting))], cards(1), Policy::default(), None, Some(supervision.clone()));
    let worker_job = median_job(&hosted_exec, JOBS);
    hosted_exec.shutdown();
    let workers = live_workers();
    let hosted_model = served(&c, 2, Some(&hosting));
    let before = settled_used_mib();
    let mut hosted = hosted_model.activate(&hosted_model.instance_key("embed", &texts(1)[0]), Device::Gpu(0)).expect("builds in a worker");
    let worker = live_workers().into_iter().find(|p| !workers.contains(p)).expect("a worker this process started");
    warm_group(&mut hosted);
    let worker_group = median_group(&mut hosted, GROUPS);
    let (worker_device, worker_host) = (delta(before), resident_mib(worker));
    drop(hosted);

    eprintln!("embed job (tiny LFM2.5 encoder, {JOBS} jobs): median {local_job:?} in process, {worker_job:?} in a worker");
    eprintln!("eight-job embed group ({GROUPS} groups, four forwards each): median {local_group:?} in process, {worker_group:?} in a worker");
    eprintln!(
        "device memory the encoder adds: {} for the first instance in this process (the card opened with it), {} for another beside it, {} as a worker; the worker holds {} resident on the host",
        mib(first_device),
        mib(instance_device),
        mib(worker_device),
        mib(worker_host.map(|m| m as i64))
    );
    // What placement charges each hosted instance for its worker's context must cover what the driver reports.
    if let (Some(worker), Some(instance)) = (worker_device, instance_device) {
        let charged = (residency::supervised::WORKER_DEVICE_CONTEXT_BYTES >> 20) as i64;
        assert!(worker - instance <= charged, "a worker's device context holds {} MiB, placement charges {charged} MiB", worker - instance);
    }
    // And what it charges the host for the worker's own process must cover its resident set.
    if let Some(resident) = worker_host {
        let charged = residency::supervised::WORKER_HOST_BYTES >> 20;
        assert!(resident <= charged, "a worker holds {resident} MiB resident on the host, placement charges {charged} MiB");
    }
    assert!(worker_job < local_job + Duration::from_millis(50), "a worker added {:?} per job", worker_job.saturating_sub(local_job));
    assert!(worker_group < local_group + Duration::from_millis(50), "a worker added {:?} per group", worker_group.saturating_sub(local_group));
}


/// One group first, so the measured ones find the forwards built.
fn warm_group(instance: &mut Box<dyn Instance>) {
    let results = instance.run_batch("embed", &texts(8), &mut |_, _| {});
    assert!(results.iter().all(Result::is_ok), "{results:?}");
}
