// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing what a server that
// trains and serves on shared accelerators is doing, for its clients. If your
// team needs expertise in observable multi-model serving and training then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What an [`Executor`](crate::Executor) tells an observer ([`Executor::observe`](crate::Executor::observe)): every
//! card's memory as the residency manager accounts it, every serving instance's life, what each served model has
//! answered, and the workload jobs (fine-tunes) with their own reports.
//!
//! Every reading is cheap to take: counters bumped where a job is answered, records kept at the manager's own
//! transitions, one dispatcher round trip for the cards and lives.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use capability::report::JobReport;
use capability::ActionResult;

use crate::manager::InstanceLife;
use crate::Device;

/// A device's name in an observer's document: `gpu0`, `npu0`, `cpu` (the `--device` grammar's names).
pub fn device_name(device: Device) -> String {
    match device {
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
        Device::Cpu => "cpu".to_string(),
    }
}

/// Milliseconds since the Unix epoch now.
pub(crate) fn unix_ms_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// One card (or the host) as the residency manager accounts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardObservation {
    pub device: Device,
    /// The bytes the budget holds now: the device's capacity as last measured (or configured), less nothing.
    pub total_bytes: u64,
    /// Headroom kept free on purpose (`--reserve-gb` and the like).
    pub reserved_bytes: u64,
    /// Bytes charged to what is placed there: resident instances, builds in progress and workloads.
    pub used_bytes: u64,
    /// Bytes a new placement could take now: total less reserved less used.
    pub free_bytes: u64,
    /// The most the card could ever hold for one instance (its declared size less the reserve).
    pub ceiling_bytes: u64,
    /// Of `used_bytes`, what running workloads (fine-tunes) reserved: never given to serving while they run.
    pub workload_reservation_bytes: u64,
    /// A running workload holds the card and has not lent it: nothing serves on it until the workload ends.
    pub serving_held: bool,
}

/// Everything [`Executor::observe`](crate::Executor::observe) reports.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// Every device the manager budgets, accelerators by index first, then the host.
    pub cards: Vec<CardObservation>,
    /// Every serving instance the manager has claimed, by key.
    pub lives: Vec<InstanceLife>,
}

/// What one served model has answered since the server started.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelServed {
    /// Jobs answered with a result.
    pub completed: u64,
    /// Prompt and completion tokens of those results, as they report them (`prompt_tokens`, `completion_tokens`).
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Jobs whose caller cancelled them.
    pub cancelled: u64,
    /// Jobs answered with an error instead of a result, by failure class (`capability::failure`, its slug in snake
    /// case): `unavailable`, `timeout`, `invalid_request`, ...; `internal` for an error that names no class.
    pub refused: BTreeMap<String, u64>,
}

/// The per-model counters, keyed by the model a job named.
#[derive(Default)]
pub(crate) struct ServedCounters(Mutex<HashMap<String, ModelServed>>);

impl ServedCounters {
    /// Count the answer `result` to a job for `model` (`cancelled`: its caller had cancelled it).
    pub(crate) fn count(&self, model: &str, result: &ActionResult, cancelled: bool) {
        let mut all = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let served = match all.get_mut(model) {
            Some(s) => s,
            None => all.entry(model.to_string()).or_default(),
        };
        match result {
            Ok(outcome) => {
                let tokens = |name: &str| outcome.outputs.get(name).and_then(serde_json::Value::as_u64).unwrap_or(0);
                served.completed += 1;
                served.prompt_tokens += tokens("prompt_tokens");
                served.completion_tokens += tokens("completion_tokens");
            }
            Err(_) if cancelled => served.cancelled += 1,
            Err(e) => {
                let class = capability::failure::classify(e).map_or_else(|| "internal".to_string(), |(class, _)| class.slug().replace('-', "_"));
                *served.refused.entry(class).or_default() += 1;
            }
        }
    }

    pub(crate) fn snapshot(&self) -> HashMap<String, ModelServed> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

/// Where a workload job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadState {
    /// Waiting for its device(s).
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl WorkloadState {
    /// The state's stable name in a metrics document.
    pub fn slug(self) -> &'static str {
        match self {
            WorkloadState::Queued => "queued",
            WorkloadState::Running => "running",
            WorkloadState::Completed => "completed",
            WorkloadState::Failed => "failed",
            WorkloadState::Cancelled => "cancelled",
        }
    }
}

/// One workload job (a fine-tune, a merge, ...) as an observer reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkloadJob {
    /// The executor's job id (submit order).
    pub id: u64,
    pub model: String,
    pub action: String,
    pub state: WorkloadState,
    pub submitted_unix_ms: u64,
    pub started_unix_ms: Option<u64>,
    pub ended_unix_ms: Option<u64>,
    /// How long it waited for its device(s): until it started, or until now while queued.
    pub queued_for: Duration,
    /// From submission to its end, or to now while it has not ended.
    pub wall: Duration,
    /// What the job last published about itself ([`capability::report`]); `None` for a job that reports nothing.
    pub report: Option<serde_json::Value>,
}

/// A workload job while the executor knows it.
struct Record {
    id: u64,
    model: String,
    action: String,
    submitted: Instant,
    submitted_unix_ms: u64,
    started: Option<(Instant, u64)>,
    ended: Option<(Instant, u64, WorkloadState)>,
    report: JobReport,
}

/// The workload jobs: every one not ended, and the last [`Workloads::KEEP_ENDED`] that ended.
#[derive(Default)]
pub(crate) struct Workloads(Mutex<VecDeque<Record>>);

impl Workloads {
    /// Ended jobs kept for an observer that reads after a job ends (its final phase split).
    const KEEP_ENDED: usize = 16;

    fn records(&self) -> std::sync::MutexGuard<'_, VecDeque<Record>> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Job `id` of `model` was submitted; `report` is the report its invocation carries.
    pub(crate) fn submitted(&self, id: u64, model: &str, action: &str, report: JobReport) {
        self.records().push_back(Record { id, model: model.to_string(), action: action.to_string(), submitted: Instant::now(), submitted_unix_ms: unix_ms_now(), started: None, ended: None, report });
    }

    /// Job `id` got its device(s): it tells the job how long it waited.
    pub(crate) fn started(&self, id: u64) {
        if let Some(r) = self.records().iter_mut().find(|r| r.id == id) {
            let now = Instant::now();
            r.started = Some((now, unix_ms_now()));
            r.report.set_admitted_after(now.duration_since(r.submitted));
        }
    }

    /// Job `id` was answered with `result`.
    pub(crate) fn ended(&self, id: u64, result: &ActionResult, cancelled: bool) {
        let mut records = self.records();
        if let Some(r) = records.iter_mut().find(|r| r.id == id) {
            let state = match result {
                Ok(_) => WorkloadState::Completed,
                Err(_) if cancelled => WorkloadState::Cancelled,
                Err(_) => WorkloadState::Failed,
            };
            r.ended = Some((Instant::now(), unix_ms_now(), state));
        }
        // Keep the last few ended jobs; every job not ended stays.
        let ended = records.iter().filter(|r| r.ended.is_some()).count();
        let mut drop = ended.saturating_sub(Self::KEEP_ENDED);
        records.retain(|r| {
            let gone = drop > 0 && r.ended.is_some();
            drop -= usize::from(gone);
            !gone
        });
    }

    pub(crate) fn snapshot(&self) -> Vec<WorkloadJob> {
        let now = Instant::now();
        self.records()
            .iter()
            .map(|r| {
                let end = r.ended.map_or(now, |(at, _, _)| at);
                WorkloadJob {
                    id: r.id,
                    model: r.model.clone(),
                    action: r.action.clone(),
                    state: r.ended.map_or(if r.started.is_some() { WorkloadState::Running } else { WorkloadState::Queued }, |(_, _, s)| s),
                    submitted_unix_ms: r.submitted_unix_ms,
                    started_unix_ms: r.started.map(|(_, ms)| ms),
                    ended_unix_ms: r.ended.map(|(_, ms, _)| ms),
                    queued_for: r.started.map_or(end, |(at, _)| at).duration_since(r.submitted),
                    wall: end.duration_since(r.submitted),
                    report: r.report.latest(),
                }
            })
            .collect()
    }
}
