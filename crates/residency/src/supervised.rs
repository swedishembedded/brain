// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping model servers
// available through wedged and crashing accelerators for its clients. If your
// team needs expertise in supervised device execution and scheduler recovery
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Models whose code runs in a supervised worker process, both ends of the
//! protocol they speak, and the executor's side of device health.
//!
//! A [`SupervisedModel`] is a [`ResidentModel`] whose instance lives in a
//! child process ([`gpu_supervisor::Worker`]). The serving process keeps the
//! model's own resident for everything the scheduler asks without a device
//! (manifest, instance key, estimate, admission checks); activation starts a
//! worker, which rebuilds that same resident from a [`HostSpec`] and
//! activates it on the assigned device ([`host`]), and every run is one
//! request with a deadline. A wedged or crashed worker is killed and reaped
//! by the supervisor, so a stuck activation or run returns to its lane with a
//! definitive error instead of holding the lane forever, and the executor
//! unwinds the claim (budget, slot, instance) as for any failed group.
//!
//! [`Supervision`] hands the executor the shared [`Health`] registry and how
//! a quarantined device is probed ([`Probe`]). With it the executor places no
//! work on a quarantined device, refuses work that nothing healthy can ever
//! host, and probes each quarantined device on its own lane until the breaker
//! closes.
//!
//! The protocol a hosted model speaks (each a line `<id> <request>`; the
//! answer is `<id> ok <payload>` or `<id> err <message>`):
//!
//! | request | answer payload |
//! |---|---|
//! | `hello <PROTOCOL>` | `{"protocol": n, "pid": p}`; must be the first request |
//! | `open <device> <spec>` | `{}`; the instance is built on `<device>` |
//! | `run[+<n>] <action> <envelope>` | `{"outputs": {...}, "blobs": [...]}`, with `ok+<n>` for `n` blobs |
//! | `probe <device>` | anything; the device works (sent to a fresh probe worker) |
//!
//! While a run is in flight the worker sends `<id> progress[+<n>] <json>`
//! frames ([`Progress`], a chunk blob in a descriptor) and the parent may send
//! one `<id> cancel`, which fires the run's [`capability::CancelToken`] in the
//! worker. Blobs travel in sealed memfds beside their frame, never on the
//! line; an envelope or answer too long for a line travels the same way, as
//! the first descriptor, with `-` in its place. `<device>` is `gpu<i>`,
//! `npu<i>` or `cpu` ([`device_label`]).
//!
//! Model code that panics in the worker answers `err panicked: <message>`;
//! the serving process raises that panic again on the lane that ran the job,
//! so the executor treats it exactly as an in-process panic (the group fails,
//! the instance is unwound and rebuilt for the next job, the card is not
//! quarantined), and the worker exits.

use std::sync::Arc;
use std::time::Duration;

use capability::{ActionResult, Invocation, Manifest, Progress};
use gpu_supervisor::{Health, Worker, WorkerCommand};
use serde_json::{json, Value};

use crate::{AdapterIdentity, Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// The protocol version a worker and its parent must agree on (`hello`).
pub const PROTOCOL: u32 = 1;

#[cfg(unix)]
mod codec;
#[cfg(unix)]
mod host;

#[cfg(target_os = "linux")]
pub use host::host;
#[cfg(unix)]
pub use host::host_on;

/// The first word of an answer whose model code panicked in the worker.
#[cfg(unix)]
const PANICKED: &str = "panicked:";

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
    /// Starting the worker and building the instance (`hello`, `open`).
    pub open: Duration,
    /// How long a run may go without a frame from its worker: each progress
    /// frame renews it, so a long job that keeps reporting runs on and one
    /// that goes silent (wedged in the driver) is killed this long after its
    /// last report.
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

/// The device a label from [`device_label`] names.
pub fn parse_device_label(label: &str) -> Result<Device, String> {
    let index = |rest: &str| rest.parse::<u32>().map_err(|_| format!("{label:?} is not a device (gpu<i>, npu<i> or cpu)"));
    match label {
        "cpu" => Ok(Device::Cpu),
        l if l.starts_with("gpu") => Ok(Device::Gpu(index(&l[3..])?)),
        l if l.starts_with("npu") => Ok(Device::Npu(index(&l[3..])?)),
        _ => Err(format!("{label:?} is not a device (gpu<i>, npu<i> or cpu)")),
    }
}

/// What a worker rebuilds to host a model: the family that knows how, and the
/// family's own description of the model (a checkpoint's id and directory,
/// say). The serving process and the worker build the same resident from it.
#[derive(Clone, Debug, PartialEq)]
pub struct HostSpec {
    pub family: String,
    pub args: Value,
}

/// A model family a worker can host: its name, and how it rebuilds a
/// resident from the arguments its [`HostSpec`] carries. Rebuilding reads no
/// weights; the worker then activates the resident on its device.
#[derive(Clone, Copy)]
pub struct HostedFamily {
    pub name: &'static str,
    pub open: fn(&Value) -> Result<Arc<dyn ResidentModel>, String>,
}

/// Worker hosting for the model families an operator chose: how to start a
/// worker, the health registry its faults go to, and its deadlines.
#[derive(Clone)]
pub struct WorkerHosting {
    command: WorkerCommand,
    health: Arc<Health<Device>>,
    deadlines: Deadlines,
    families: Vec<String>,
}

impl WorkerHosting {
    pub fn new(command: WorkerCommand, health: Arc<Health<Device>>, deadlines: Deadlines, families: Vec<String>) -> WorkerHosting {
        WorkerHosting { command, health, deadlines, families }
    }

    /// Whether models of `family` run in a worker.
    pub fn hosts(&self, family: &str) -> bool {
        self.families.iter().any(|f| f == family)
    }

    /// How a worker is started.
    pub fn command(&self) -> &WorkerCommand {
        &self.command
    }

    /// `resident`, served from a worker when `hosting` covers its family, else as is.
    pub fn serve(hosting: Option<&WorkerHosting>, resident: Arc<dyn ResidentModel>, spec: HostSpec) -> Arc<dyn ResidentModel> {
        match hosting {
            Some(h) if h.hosts(&spec.family) => Arc::new(SupervisedModel::new(resident, spec, h.command.clone(), h.health.clone(), h.deadlines)),
            _ => resident,
        }
    }
}

/// A model served by a supervised worker process: one worker per instance
/// (one device and one kernel set), owned by that instance.
pub struct SupervisedModel {
    inner: Arc<dyn ResidentModel>,
    spec: HostSpec,
    command: WorkerCommand,
    health: Arc<Health<Device>>,
    deadlines: Deadlines,
}

impl SupervisedModel {
    /// `inner` answers everything the scheduler asks without a device; a
    /// worker started from `command` rebuilds it from `spec` and runs it.
    pub fn new(inner: Arc<dyn ResidentModel>, spec: HostSpec, command: WorkerCommand, health: Arc<Health<Device>>, deadlines: Deadlines) -> SupervisedModel {
        SupervisedModel { inner, spec, command, health, deadlines }
    }
}

impl ResidentModel for SupervisedModel {
    fn manifest(&self) -> Manifest {
        self.inner.manifest()
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key)
    }

    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        self.inner.estimate_at(key, tier)
    }

    fn replicas(&self) -> usize {
        self.inner.replicas()
    }

    fn batch_limit(&self) -> Option<usize> {
        self.inner.batch_limit()
    }

    fn stamp(&self, inv: &mut Invocation) {
        self.inner.stamp(inv)
    }

    fn check_job(&self, action: &str, inv: &Invocation) -> Result<(), String> {
        self.inner.check_job(action, inv)
    }

    fn transient(&self) -> bool {
        self.inner.transient()
    }

    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.inner.adapter_identity()
    }

    /// Runs on the device's lane, which outlives the worker it starts (the
    /// worker is killed when the thread that spawned it exits).
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let mut worker = Worker::spawn_with(&self.command, self.deadlines.reap).map_err(|e| format!("{key}: could not start worker {} ({e})", self.command.program().display()))?;
        let open = json!({ "family": self.spec.family, "args": self.spec.args, "key": { "model": key.model, "config": key.config } });
        for request in [format!("hello {PROTOCOL}"), format!("open {} {open}", device_label(device))] {
            self.health.call(&device, &mut worker, &request, self.deadlines.open).map_err(|e| format!("{key}: {e}"))?;
        }
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
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if let Some(lost) = &self.lost {
            return Err(lost.clone());
        }
        if action.is_empty() || action.contains(char::is_whitespace) {
            return Err(format!("{action:?} is not an action name"));
        }
        self.run_in_worker(action, inv, progress)
    }
}

impl SupervisedInstance {
    /// One run as one exchange with the worker: the envelope (and the blobs
    /// in memfds) out, progress and a cancellation while it runs, the
    /// outcome back.
    #[cfg(unix)]
    fn run_in_worker(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        use gpu_supervisor::{CallError, Exchange};
        let mut fds = Vec::new();
        let blobs = codec::write_blobs(&inv.blobs, &mut fds).map_err(|e| format!("{action}: {e}"))?;
        let envelope = json!({ "params": inv.params, "scope": inv.scope, "blobs": blobs }).to_string();
        let envelope = codec::pack(envelope, &mut fds).map_err(|e| format!("{action}: {e}"))?;
        let verb = if fds.is_empty() { "run".to_string() } else { format!("run+{}", fds.len()) };
        let request = format!("{verb} {action} {envelope}");

        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        let mut talk = RunTalk { inv, progress, cancel_sent: false };
        let x = Exchange { fds: &borrowed, talk: Some(&mut talk), renew: true };
        let answer = self.health.exchange(&self.device, &mut self.worker, &request, self.run, x);
        match answer {
            Ok((payload, fds)) => codec::decode_outcome(&payload, fds).map_err(|e| self.lose(format!("{action}: the worker's answer breaks the protocol ({e})"))),
            Err(CallError::Failed(message)) => match message.strip_prefix(PANICKED) {
                // The same unwind an in-process panic takes: the lane fails
                // the group and the executor rebuilds the instance.
                Some(what) => {
                    let what = format!("{action}: the worker panicked:{what}");
                    self.lost = Some(what.clone());
                    panic!("{what}");
                }
                None => Err(message),
            },
            Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => Err(self.lose(e.to_string())),
            Err(e @ CallError::Invalid(_)) => Err(e.to_string()),
        }
    }

}

/// The lane's side of one run while it is in flight: progress to the job's
/// caller, and one `cancel` once the caller asks.
#[cfg(unix)]
struct RunTalk<'a> {
    inv: &'a Invocation,
    progress: &'a mut dyn FnMut(Progress),
    cancel_sent: bool,
}

#[cfg(unix)]
impl gpu_supervisor::Conversation for RunTalk<'_> {
    fn frame(&mut self, kind: &str, payload: &str, fds: Vec<std::os::fd::OwnedFd>) -> Result<Vec<gpu_supervisor::Outgoing>, String> {
        match kind {
            "progress" => {
                let p = codec::decode_progress(payload, fds).map_err(|e| format!("the worker's progress breaks the protocol ({e})"))?;
                (self.progress)(p);
                Ok(Vec::new())
            }
            other => Err(format!("a run takes progress frames, not {other:?}")),
        }
    }

    fn poll(&mut self) -> Vec<gpu_supervisor::Outgoing> {
        if self.cancel_sent || !self.inv.cancel.is_cancelled() {
            return Vec::new();
        }
        self.cancel_sent = true;
        vec![gpu_supervisor::Outgoing::line("cancel".to_string())]
    }

    fn polls(&self) -> bool {
        !self.cancel_sent
    }
}

impl SupervisedInstance {
    #[cfg(not(unix))]
    fn run_in_worker(&mut self, _action: &str, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        Err("supervised device workers need a Unix host".into())
    }

    /// Mark the worker unusable for every later run, and say why.
    fn lose(&mut self, why: String) -> String {
        self.lost = Some(why.clone());
        why
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_labels_round_trip_and_nothing_else_parses() {
        for d in [Device::Cpu, Device::Gpu(0), Device::Gpu(11), Device::Npu(1)] {
            assert_eq!(parse_device_label(&device_label(d)), Ok(d));
        }
        for bad in ["", "gpu", "gpu-1", "tpu0", "cpu0", "gpu 1"] {
            assert!(parse_device_label(bad).is_err(), "{bad:?} parsed");
        }
    }
}
