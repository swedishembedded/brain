// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping model servers
// available through wedged and crashing accelerators for its clients. If your
// team needs expertise in supervised device execution and scheduler recovery
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Device work that runs in a supervised worker process, and the executor's
//! side of device health.
//!
//! A [`SupervisedModel`] is a [`ResidentModel`] whose instance lives in a
//! child process ([`gpu_supervisor::Worker`]): activation starts the worker
//! and asks it to `open` the device, and every run is one request with a
//! deadline. A wedged or crashed worker is killed and reaped by the
//! supervisor, so a stuck activation or run returns to its lane with a
//! definitive error instead of holding the lane forever, and the executor
//! unwinds the claim (budget, slot, instance) as for any failed group.
//!
//! [`Supervision`] hands the executor the shared [`Health`] registry and how
//! a quarantined device is probed ([`Probe`]). With it the executor places no work on a quarantined device,
//! refuses work that nothing healthy can ever host, and probes each
//! quarantined device on its own lane until the breaker closes.
//!
//! The worker protocol a supervised model speaks (each a line; the answer is
//! `ok <payload>` or `err <message>`):
//!
//! | request | answer payload |
//! |---|---|
//! | `open <device>` | anything; the instance is ready |
//! | `run <action> <params JSON>` | the outputs, a JSON object |
//! | `probe <device>` | anything; the device works (sent to a fresh worker) |
//!
//! `<device>` is `gpu<i>`, `npu<i>` or `cpu` ([`device_label`]).

use std::sync::Arc;
use std::time::Duration;

use capability::{ActionResult, Invocation, Manifest, Outcome, Progress};
use gpu_supervisor::{CallError, Health, Worker, WorkerCommand};

use crate::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// What the executor needs to keep device health: the registry the supervised
/// models record faults in, and how a quarantined device is probed.
#[derive(Clone)]
pub struct Supervision {
    pub health: Arc<Health<Device>>,
    pub probe: Probe,
}

/// How a quarantined device is tested before it is readmitted.
#[derive(Clone, Debug)]
pub enum Probe {
    /// A fresh worker from `command` is sent `probe <device>` and must answer
    /// within `deadline`.
    Worker { command: WorkerCommand, deadline: Duration },
    /// No worker program is available: a due probe passes without touching
    /// the device. The breaker then only paces readmission (a restart is
    /// readmitted at once, a tripped device after its spaced probes), which
    /// still bounds what a wedging card costs.
    Elapsed,
}

/// The bounds a supervised model's requests run under.
#[derive(Clone, Copy, Debug)]
pub struct Deadlines {
    /// Starting the worker's device (`open`).
    pub open: Duration,
    /// One run.
    pub run: Duration,
    /// How long a killed worker is waited for before it is kept for later polling.
    pub reap: Duration,
}

/// The name a worker is told its device by.
pub fn device_label(device: Device) -> String {
    match device {
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
        Device::Cpu => "cpu".to_string(),
    }
}

/// A model served by a supervised worker process: one worker per instance
/// (one device and one kernel set), owned by that instance.
pub struct SupervisedModel {
    manifest: Manifest,
    command: WorkerCommand,
    health: Arc<Health<Device>>,
    cost: MemCost,
    deadlines: Deadlines,
}

impl SupervisedModel {
    pub fn new(manifest: Manifest, command: WorkerCommand, health: Arc<Health<Device>>, cost: MemCost, deadlines: Deadlines) -> SupervisedModel {
        SupervisedModel { manifest, command, health, cost, deadlines }
    }
}

impl ResidentModel for SupervisedModel {
    fn manifest(&self) -> Manifest {
        self.manifest.clone()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.manifest.model, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        self.cost
    }

    /// Runs on the device's lane, which outlives the worker it starts (the
    /// worker is killed when the thread that spawned it exits).
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let mut worker = Worker::spawn_with(&self.command, self.deadlines.reap).map_err(|e| format!("{key}: could not start worker {} ({e})", self.command.program().display()))?;
        self.health.call(&device, &mut worker, &format!("open {}", device_label(device)), self.deadlines.open).map_err(|e| format!("{key}: {e}"))?;
        Ok(Box::new(SupervisedInstance { worker, device, health: self.health.clone(), run: self.deadlines.run, lost: None }))
    }
}

/// An instance whose state lives in its worker process.
struct SupervisedInstance {
    worker: Worker,
    device: Device,
    health: Arc<Health<Device>>,
    run: Duration,
    /// Why the worker is gone. Every later run fails with it at once: a run
    /// is not idempotent, so nothing is retried on a replacement worker; the
    /// executor drops this instance and a later job builds a fresh one.
    lost: Option<String>,
}

impl Instance for SupervisedInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if let Some(lost) = &self.lost {
            return Err(lost.clone());
        }
        if !inv.blobs.is_empty() {
            return Err(format!("{action}: a supervised worker takes no blobs"));
        }
        if action.is_empty() || action.contains(char::is_whitespace) {
            return Err(format!("{action:?} is not an action name"));
        }
        // Compact JSON has no raw newline, so the request stays one frame.
        let params = serde_json::to_string(&inv.params).map_err(|e| format!("{action}: params: {e}"))?;
        match self.health.call(&self.device, &mut self.worker, &format!("run {action} {params}"), self.run) {
            Ok(payload) => {
                let outputs: serde_json::Value = serde_json::from_str(&payload).map_err(|e| format!("{action}: the worker's outputs are not JSON ({e})"))?;
                if !outputs.is_object() {
                    return Err(format!("{action}: the worker's outputs are not a JSON object"));
                }
                Ok(Outcome { outputs, ..Outcome::new() })
            }
            Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => {
                let msg = e.to_string();
                self.lost = Some(msg.clone());
                Err(msg)
            }
            Err(e) => Err(e.to_string()),
        }
    }
}
