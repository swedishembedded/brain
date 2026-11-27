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
//! activates it on the assigned device ([`host`]), and every group of jobs is
//! one request with a deadline that the worker's progress renews. A wedged or crashed worker is killed and reaped
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
//! | `open <devices> <spec>` | `{"metrics"}`; the instance is built on `<devices>`: one, or every card of a multi-device instance, comma-separated |
//! | `batch[+<n>] <action> <k> <jobs>` | `{"results": [...], "metrics"}`, one result per job: `{"outputs", "blobs"}` or `{"error"}` |
//! | `group[+<n>] <action> <k> <jobs>` | `{"open": bool, "metrics"}`; the jobs were finished through `done` frames |
//! | `demote warm\|cold` | `{"metrics"}`; the instance released its device buffers, the worker keeps its device context |
//! | `promote <device>` | `{"metrics"}`; the instance is Hot again on `<device>`, which must be the card it was opened on |
//! | `attach {"id", "digest", "path", "version"}` | `{"metrics"}`; the model's registry offers the adapter the file holds under the version the serving process numbered it, once its digest checks ([`AdapterSink`]) |
//! | `detach <id>` | `{"metrics"}`; the registry no longer offers `<id>` |
//! | `probe <device>` | anything; the device works (sent to a fresh probe worker) |
//!
//! `"metrics"` is the instance's [`Instance::metrics`] after the request, as
//! `[[name, value], ...]`: the parent keeps the last one it read, so the
//! dispatcher's poll of a hosted instance's metrics never reaches its worker.
//! A demotion keeps the worker and so its device context, which is why a Warm
//! hosted instance stays charged that context on its card
//! ([`MemCost::context`]); a promotion onto another card kills the worker and
//! opens a new one there.
//!
//! While any request is in flight the worker may send `<id> fault <device>
//! <why>` before its next frame ([`report_device_fault`]: model code that saw
//! a driver call fail on one card of several). The parent quarantines that
//! card and loses the instance, whether the worker then answers or dies; a
//! worker that dies with no such frame has every card it held quarantined
//! ([`gpu_supervisor::Health::exchange_on`]). A multi-device model is hosted
//! whole by one worker ([`SupervisedMultiModel`], [`HostedResident::Multi`]).
//!
//! A base that serves adapters ([`SupervisedModel::with_adapters`]) tells its
//! worker every change of what the serving process offers
//! ([`AdapterSource`]): before a group as `attach`/`detach` requests, and while
//! one runs as `<id> attach ...`/`<id> detach ...` frames sent at a poll and
//! before the reply to each `take`, which the worker's reader applies at once.
//! A job that joins after a change therefore never reaches the worker before
//! the change does; the group refreshes its routes as it does in process.
//!
//! `<jobs>` is `{"jobs": [{"params", "scope", "blobs"}, ...]}` and `<k>` how
//! many it holds, so the worker's reader registers a cancellation token for
//! each job as it reads the request, before the job starts. While a group is
//! in flight the worker sends `<id> progress[+<n>] <json>` frames
//! ([`Progress`] and the index of the job it is about, a chunk blob beside
//! it), each of which renews the deadline, and the parent sends one
//! `<id> cancel <job>` for each job whose caller asks, which fires that job's
//! [`capability::CancelToken`] in the worker and no other. A single job is a
//! group of one.
//!
//! An open group (`Instance::run_open`, continuous batching) is `group[+<n>]
//! <action> <k> <jobs>`, answered `{"open": bool}`: false when the worker's
//! instance runs the action only as closed groups, having done nothing (the
//! parent then sends the group as a `batch`, and does not ask again for that
//! action). While it runs the worker reaches the executor's `JobFeed` through
//! the parent: `<id> take <max>` is answered `<id> joined[+<n>] <k> <jobs>`
//! with the jobs that joined (numbered after the group's own, in order; the
//! reader registers their tokens as it reads them), `<id> done[+<n>] <json>`
//! (`{"job": i}` and its result) finishes one job at once, and `<id> close`
//! is answered `<id> closed true|false`. The feed's screening crosses the
//! same way: `<id> screen` is answered `<id> waiting[+<n>] <k> <jobs>` with
//! the jobs waiting to join that were not screened yet (their tokens are not
//! registered: they are not the group's), the worker's instance asks its own
//! check of each, and `<id> refused <json>` (`{"refused": [{"job", "why"}]}`,
//! `job` its index in that `waiting` frame; sent only when one is refused)
//! has the parent's feed refuse each at once. `<id> metrics <json>`
//! (`{"metrics": [...]}`, sent after a scheduler pass) is the instance's
//! metrics as of now, which the parent's feed publishes for the dispatcher
//! (`JobFeed::publish`). A worker that dies mid-group
//! fails every job the group had not finished, the joined ones included,
//! once.
//!
//! A frame's blobs travel in one sealed memfd beside it, never
//! on the line; an envelope or answer too long for a line travels the same
//! way, as the first descriptor, with `-` in its place. `<device>` is `gpu<i>`,
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

use crate::multi::MultiDeviceResidentModel;
use crate::{AdapterIdentity, Device, Instance, InstanceKey, JobFeed, MemCost, ResidentModel, Tier};

/// The protocol version a worker and its parent must agree on (`hello`).
pub const PROTOCOL: u32 = 6;

mod adapters;
#[cfg(unix)]
mod codec;
#[cfg(unix)]
mod host;

pub use adapters::{AdapterSink, AdapterSource, HostedAdapter};
use adapters::AdapterFeed;

mod multi;

#[cfg(target_os = "linux")]
pub use host::host;
#[cfg(unix)]
pub use host::{host_on, report_device_fault};
pub use multi::SupervisedMultiModel;

/// Device memory a worker's own device context holds on its card beside the instance it hosts: what every in-process
/// instance of a card shares through the serving process's one context and each worker pays again, so every hosted
/// instance is charged it on top of its estimate ([`MemCost::with_device_context`]). It is the card's used memory
/// with a worker built on it less what the same instance adds beside an open context, as the driver reports both, and
/// rounded up to a whole 64 MiB; `a_worker_hosted_encoder_costs_a_round_trip_per_group_and_a_context_per_card` (the
/// brain CLI's `worker_hosted_lfm` tests) measures it on a real card and fails when a worker holds more.
pub const WORKER_DEVICE_CONTEXT_BYTES: u64 = 192 << 20;

/// Host memory a worker process holds of its own: the `brain` binary's mappings, its runtime and its copy of what the
/// model keeps on the host, none of it shared with the serving process, so every hosted instance is charged it against
/// the host budget beside its card ([`MemCost::with_process_host`]). It is the worker's resident set with its instance
/// built and run, rounded up to a whole 64 MiB; the same `worker_hosted_lfm` test measures it and fails when a worker
/// holds more.
pub const WORKER_HOST_BYTES: u64 = 320 << 20;

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
    pub open: fn(&Value) -> Result<HostedModel, String>,
}

/// What a worker rebuilt from a [`HostSpec`]: the resident it activates, and
/// for a base that serves adapters, the registry its parent's `attach` and
/// `detach` rebuild ([`AdapterSink`]).
pub struct HostedModel {
    pub resident: HostedResident,
    pub adapters: Option<Arc<dyn AdapterSink>>,
}

/// The resident a worker activates: on the one card `open` names, or across
/// every card it names, all owned by the one worker (a device's object graph
/// lives in the process that created it, so the cards of one instance cannot
/// be split between workers).
pub enum HostedResident {
    Single(Arc<dyn ResidentModel>),
    Multi(Arc<dyn MultiDeviceResidentModel>),
}

impl HostedResident {
    /// The model, for what is asked of it without a device.
    pub fn model(&self) -> &dyn ResidentModel {
        match self {
            HostedResident::Single(r) => &**r,
            HostedResident::Multi(m) => &**m,
        }
    }

    /// Build the instance on `devices`: exactly one for a single-device resident.
    fn activate(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        match (self, devices) {
            (HostedResident::Single(r), [device]) => r.activate(key, *device),
            (HostedResident::Single(_), _) => Err(format!("{key} runs on one device, and the worker was asked to open {}", devices.len())),
            (HostedResident::Multi(m), _) => m.activate_multi(key, devices),
        }
    }
}

impl HostedModel {
    /// A model on one card that serves no adapters.
    pub fn new(resident: Arc<dyn ResidentModel>) -> HostedModel {
        HostedModel { resident: HostedResident::Single(resident), adapters: None }
    }

    /// A model across several cards, every one owned by the worker.
    pub fn multi(resident: Arc<dyn MultiDeviceResidentModel>) -> HostedModel {
        HostedModel { resident: HostedResident::Multi(resident), adapters: None }
    }

    /// A base whose adapters the parent offers reach `adapters`.
    pub fn with_adapters(mut self, adapters: Arc<dyn AdapterSink>) -> HostedModel {
        self.adapters = Some(adapters);
        self
    }
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

impl std::fmt::Debug for WorkerHosting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerHosting").field("command", &self.command).field("deadlines", &self.deadlines).field("families", &self.families).finish_non_exhaustive()
    }
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
            Some(h) if h.hosts(&spec.family) => Arc::new(h.supervised(resident, spec)),
            _ => resident,
        }
    }

    /// `resident` served from this hosting's workers, which rebuild it from `spec`, whether or not its family is
    /// one an operator named: for a model that holds its hosted form itself (a base adapters are served over, whose
    /// views must build through the worker) and checks [`Self::hosts`] first.
    pub fn supervised(&self, resident: Arc<dyn ResidentModel>, spec: HostSpec) -> SupervisedModel {
        SupervisedModel::new(resident, spec, self.command.clone(), self.health.clone(), self.deadlines)
    }

    /// [`Self::serve`] for a model across several cards, which a worker hosts whole; register the result with
    /// `Executor::register_multi`.
    pub fn serve_multi(hosting: Option<&WorkerHosting>, resident: Arc<dyn MultiDeviceResidentModel>, spec: HostSpec) -> Arc<dyn MultiDeviceResidentModel> {
        match hosting {
            Some(h) if h.hosts(&spec.family) => Arc::new(SupervisedMultiModel::new(resident, spec, h.command.clone(), h.health.clone(), h.deadlines)),
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
    adapters: Option<Arc<dyn AdapterSource>>,
}

impl SupervisedModel {
    /// `inner` answers everything the scheduler asks without a device; a
    /// worker started from `command` rebuilds it from `spec` and runs it.
    pub fn new(inner: Arc<dyn ResidentModel>, spec: HostSpec, command: WorkerCommand, health: Arc<Health<Device>>, deadlines: Deadlines) -> SupervisedModel {
        SupervisedModel { inner, spec, command, health, deadlines, adapters: None }
    }

    /// A base whose adapters `adapters` offers: every worker of it is told each change (`attach`, `detach`), and
    /// rebuilds its own registry from what it is told ([`HostedModel::with_adapters`]).
    pub fn with_adapters(mut self, adapters: Arc<dyn AdapterSource>) -> SupervisedModel {
        self.adapters = Some(adapters);
        self
    }
}

impl ResidentModel for SupervisedModel {
    fn manifest(&self) -> Manifest {
        self.inner.manifest()
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }

    /// The model's own estimate plus its worker's device context and host memory: each instance is one worker on one
    /// card. For a model that demotes, the context is also what its demotion keeps on the card ([`MemCost::context`]).
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let hot = self.inner.estimate(key);
        let placed = hot.with_device_context(WORKER_DEVICE_CONTEXT_BYTES).with_process_host(WORKER_HOST_BYTES);
        if self.demotes(key) {
            placed.with_kept_context(placed.vram - hot.vram)
        } else {
            placed
        }
    }

    /// Hot is [`Self::estimate`]. Warm and Cold keep the worker, whose device context stays on the card and whose
    /// process stays on the host: they are charged that context there and the worker's host memory, which covers the
    /// host copy it keeps. A model that never demotes is never asked about them.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        let hot = self.estimate(key);
        if tier == Tier::Hot || !self.demotes(key) {
            return hot;
        }
        let warm = self.inner.estimate_at(key, tier);
        if hot.context == 0 {
            return warm.with_process_host(hot.process_host);
        }
        MemCost::new(hot.context, 0).with_mapped(warm.mapped).with_kept_context(hot.context).with_process_host(hot.process_host)
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
        let start = WorkerStart::new(&self.spec, key, &self.command, &self.health, self.deadlines);
        SupervisedInstance::open(start, &[device], self.adapters.clone())
    }
}

impl SupervisedModel {
    /// Whether the model opts into demotion: a model that never demotes keeps the default
    /// [`ResidentModel::estimate_at`], its Hot estimate at every tier.
    fn demotes(&self, key: &InstanceKey) -> bool {
        self.inner.estimate_at(key, Tier::Warm) != self.inner.estimate(key)
    }
}

/// How an instance's worker is started and its instance built: what activation does, and what a promotion onto
/// another card does again (a worker owns the cards it opened).
struct WorkerStart {
    command: WorkerCommand,
    /// The `open` request's spec.
    open: Value,
    /// The instance key, for errors.
    key: String,
    health: Arc<Health<Device>>,
    deadlines: Deadlines,
}

impl WorkerStart {
    fn new(spec: &HostSpec, key: &InstanceKey, command: &WorkerCommand, health: &Arc<Health<Device>>, deadlines: Deadlines) -> WorkerStart {
        let open = json!({ "family": spec.family, "args": spec.args, "key": { "model": key.model, "config": key.config } });
        WorkerStart { command: command.clone(), open, key: key.to_string(), health: health.clone(), deadlines }
    }

    /// A new worker with the instance built on `devices` (one, or every card of a multi-device instance: one worker
    /// owns them all), and the instance's metrics.
    fn open_on(&self, devices: &[Device]) -> Result<(Worker, Vec<(String, Value)>), String> {
        let key = &self.key;
        let mut worker = Worker::spawn_with(&self.command, self.deadlines.reap).map_err(|e| format!("{key}: could not start worker {} ({e})", self.command.program().display()))?;
        control(&self.health, devices, &mut worker, &format!("hello {PROTOCOL}"), self.deadlines.open).map_err(|e| format!("{key}: {}", e.why))?;
        let labels: Vec<String> = devices.iter().map(|&d| device_label(d)).collect();
        let metrics = control(&self.health, devices, &mut worker, &format!("open {} {}", labels.join(","), self.open), self.deadlines.open).map_err(|e| format!("{key}: {}", e.why))?;
        Ok((worker, metrics))
    }
}

/// The devices a worker said faulted while one request was in flight, from its `fault <device> <why>` frames: a
/// driver call its model saw fail on one card of several. They alone are quarantined if the worker then dies, and
/// they are quarantined even if it answers, which loses the instance (its card is suspect).
#[derive(Default)]
struct Blamed(std::cell::RefCell<Vec<(Device, String)>>);

impl Blamed {
    /// One `fault` frame's payload, which must name a device of `devices`.
    fn note(&self, devices: &[Device], payload: &str) -> Result<(), String> {
        let (label, why) = payload.split_once(' ').unwrap_or((payload, "a device fault"));
        let device = parse_device_label(label)?;
        if !devices.contains(&device) {
            return Err(format!("the worker reported a fault on {label}, which it does not hold"));
        }
        self.0.borrow_mut().push((device, why.to_string()));
        Ok(())
    }

    fn devices(&self) -> Vec<Device> {
        self.0.borrow().iter().map(|&(d, _)| d).collect()
    }

    /// After an answer: quarantine each device the worker blamed, and say why; `None` when it blamed none.
    fn record(&self, health: &Health<Device>) -> Option<String> {
        let blamed = self.0.borrow();
        let causes: Vec<String> = blamed
            .iter()
            .map(|(device, why)| {
                let cause = format!("{device:?}: the worker reported a device fault ({why}); the device is quarantined until a health probe passes");
                health.record_fault(device, std::time::Instant::now(), &cause);
                cause
            })
            .collect();
        (!causes.is_empty()).then(|| causes.join("; "))
    }
}

/// The parent's side of a request that runs no job: it takes only `fault` frames.
#[cfg(unix)]
struct FaultTalk<'a> {
    devices: &'a [Device],
    blamed: &'a Blamed,
}

#[cfg(unix)]
impl gpu_supervisor::Conversation for FaultTalk<'_> {
    fn frame(&mut self, kind: &str, payload: &str, _fds: Vec<std::os::fd::OwnedFd>) -> Result<Vec<gpu_supervisor::Outgoing>, String> {
        match kind {
            "fault" => self.blamed.note(self.devices, payload).map(|()| Vec::new()),
            other => Err(format!("a request that runs no job takes fault frames, not {other:?}")),
        }
    }

    fn poll(&mut self) -> Vec<gpu_supervisor::Outgoing> {
        Vec::new()
    }

    fn polls(&self) -> bool {
        false
    }
}

/// Why a request that runs no job failed, and whether the worker is gone with it.
struct ControlError {
    why: String,
    lost: bool,
}

/// A request that runs no job (`hello`, `open`, `demote`, `promote`, `attach`, `detach`) on the worker of
/// `devices`: the instance's metrics after it. A model panic is answered and ends the worker, and a device the
/// worker blamed is quarantined, so either loses the worker as a fault does.
#[cfg(unix)]
fn control(health: &Health<Device>, devices: &[Device], worker: &mut Worker, request: &str, deadline: Duration) -> Result<Vec<(String, Value)>, ControlError> {
    use gpu_supervisor::{CallError, Exchange};
    let blamed = Blamed::default();
    let mut talk = FaultTalk { devices, blamed: &blamed };
    let x = Exchange { fds: &[], talk: Some(&mut talk), renew: false };
    let answer = health.exchange_on(devices, worker, request, deadline, x, &|| blamed.devices());
    let faulted = || answer.as_ref().is_err_and(|e| matches!(e, CallError::Fault(_)));
    if let Some(cause) = (!faulted()).then(|| blamed.record(health)).flatten() {
        return Err(ControlError { why: cause, lost: true });
    }
    match answer {
        // The greeting's answer names the worker, not an instance.
        Ok(_) if request.starts_with("hello ") => Ok(Vec::new()),
        Ok((payload, fds)) => codec::decode_control(&payload, fds).map_err(|e| ControlError { why: format!("the worker's answer breaks the protocol ({e})"), lost: true }),
        Err(CallError::Failed(message)) => {
            let lost = message.starts_with(PANICKED);
            Err(ControlError { why: message, lost })
        }
        Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => Err(ControlError { why: e.to_string(), lost: true }),
        Err(e @ CallError::Invalid(_)) => Err(ControlError { why: e.to_string(), lost: false }),
    }
}

#[cfg(not(unix))]
fn control(_health: &Health<Device>, _devices: &[Device], _worker: &mut Worker, _request: &str, _deadline: Duration) -> Result<Vec<(String, Value)>, ControlError> {
    Err(ControlError { why: "supervised device workers need a Unix host".into(), lost: true })
}

/// The word a `demote` request names `tier` by.
fn tier_label(tier: Tier) -> &'static str {
    match tier {
        Tier::Hot => "hot",
        Tier::Warm => "warm",
        Tier::Cold => "cold",
    }
}

/// An instance whose state lives in its worker process.
struct SupervisedInstance {
    /// `None` only while [`Self::lost`] says why: between the old worker of a promotion onto another card being
    /// killed and its new one opening, or when that opening failed.
    worker: Option<Worker>,
    /// The cards the worker owns: one, or every card of a multi-device instance.
    devices: Vec<Device>,
    start: WorkerStart,
    /// Why the worker is gone. Every later run fails with it at once: a run
    /// is not idempotent, so nothing is retried on a replacement worker; the
    /// executor drops this instance and a later job builds a fresh one.
    lost: Option<String>,
    /// Actions the worker's instance runs only as closed groups: its
    /// `run_open` said so, having done nothing, so it is not asked again and
    /// such a group costs one request, not two. Whether a model runs a group
    /// open depends on the action alone in every model that does; were it to
    /// depend on more, a remembered answer would only make it batch a closed
    /// group, which every model answers correctly.
    closed_only: std::collections::BTreeSet<String>,
    /// The worker's instance's metrics as its last answer carried them: the dispatcher reads them between runs
    /// ([`Instance::metrics`]) and never waits on the worker for them.
    metrics: Vec<(String, Value)>,
    /// What the worker was told of its base's adapters, for a base that serves them.
    adapters: Option<AdapterFeed>,
}

impl Instance for SupervisedInstance {
    fn metrics(&self) -> Vec<(String, Value)> {
        self.metrics.clone()
    }

    /// `demote <tier>` in the worker, bounded like a job by the run deadline: the dispatcher calls it, so a worker
    /// wedged in it holds the dispatcher at most that long, and is then killed and its card quarantined.
    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        let deadline = self.start.deadlines.run;
        self.control(&format!("demote {}", tier_label(tier)), deadline)
    }

    /// `promote <device>` in the worker when `device` is its card. Onto another card, or after its worker was lost,
    /// the instance is rebuilt in a new worker there (the old one is killed): a worker owns one card, and the
    /// rebuild is the activation the model would get anyway.
    fn promote(&mut self, device: Device) -> Result<(), String> {
        if self.lost.is_some() || self.devices != [device] {
            // The old worker goes first: its device context is no longer charged to its card.
            self.worker = None;
            let (worker, metrics) = self.start.open_on(&[device]).map_err(|e| self.lose(e))?;
            self.worker = Some(worker);
            self.devices = vec![device];
            self.lost = None;
            self.closed_only.clear();
            self.metrics = metrics;
            if let Some(feed) = self.adapters.as_mut() {
                feed.reset();
            }
            return Ok(());
        }
        let deadline = self.start.deadlines.open;
        self.control(&format!("promote {}", device_label(device)), deadline)
    }

    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut results = self.run_batch(action, std::slice::from_ref(inv), &mut |_, p| progress(p));
        match results.pop() {
            Some(result) if results.is_empty() => result,
            _ => Err(format!("{action}: the worker answered {} results for one job", results.len() + 1)),
        }
    }

    /// The whole group as one `batch` request: the worker's instance runs
    /// its own `run_batch` over it, so a model that batches does so in the
    /// worker as it does in process.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let every = |why: String| invs.iter().map(|_| Err(why.clone())).collect();
        if let Some(lost) = &self.lost {
            return every(lost.clone());
        }
        if action.is_empty() || action.contains(char::is_whitespace) {
            return every(format!("{action:?} is not an action name"));
        }
        self.batch_in_worker(action, invs, progress)
    }

    /// The group as one `group` request the worker's instance runs with its
    /// own `run_open`: the jobs that join while it runs cross as the worker
    /// takes them, and each is finished through `feed` the moment the worker
    /// finishes it. False, having done nothing, when the worker's instance
    /// runs this action only as closed groups (or the worker is lost; the
    /// closed batch then answers every job with why).
    fn run_open(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        if self.lost.is_some() || self.closed_only.contains(action) || action.is_empty() || action.contains(char::is_whitespace) {
            return false;
        }
        self.group_in_worker(action, invs, progress, feed)
    }
}

impl SupervisedInstance {
    /// A worker started from `start` with its instance built on `devices`, told the adapters `adapters` offers.
    fn open(start: WorkerStart, devices: &[Device], adapters: Option<Arc<dyn AdapterSource>>) -> Result<Box<dyn Instance>, String> {
        let (worker, metrics) = start.open_on(devices)?;
        let adapters = adapters.map(AdapterFeed::new);
        Ok(Box::new(SupervisedInstance { worker: Some(worker), devices: devices.to_vec(), start, lost: None, closed_only: std::collections::BTreeSet::new(), metrics, adapters }))
    }

    /// A request that runs no job, its answer's metrics kept.
    fn control(&mut self, request: &str, deadline: Duration) -> Result<(), String> {
        let (Some(worker), None) = (self.worker.as_mut(), &self.lost) else {
            return Err(self.lost.clone().unwrap_or_else(|| "the worker is gone".into()));
        };
        match control(&self.start.health, &self.devices, worker, request, deadline) {
            Ok(metrics) => {
                self.metrics = metrics;
                Ok(())
            }
            Err(ControlError { why, lost: true }) => Err(self.lose(why)),
            Err(ControlError { why, lost: false }) => Err(why),
        }
    }

    /// Tell the worker what its base's adapters changed since it was last told, as requests, before a group
    /// starts. A change the worker refuses (a file whose digest is not the release's) is logged: requests naming
    /// that adapter are answered as for no model.
    fn sync_adapters(&mut self) -> Result<(), String> {
        let Some(changes) = self.adapters.as_mut().map(AdapterFeed::changes) else { return Ok(()) };
        for change in changes {
            if let Err(why) = self.control(&change.line(), self.start.deadlines.open) {
                if self.lost.is_some() {
                    return Err(why);
                }
                crate::log::warn(&format!("{}: {why}", self.start.key));
            }
        }
        Ok(())
    }

    /// One group as one exchange with the worker: the jobs (and their blobs
    /// in a memfd) out, each job's progress and cancellation while it runs,
    /// every job's result back.
    #[cfg(unix)]
    fn batch_in_worker(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        use gpu_supervisor::{CallError, Exchange};
        let every = |why: String| invs.iter().map(|_| Err(why.clone())).collect::<Vec<_>>();
        if let Err(why) = self.sync_adapters() {
            return every(why);
        }
        let mut fds = Vec::new();
        let envelope = match codec::encode_jobs(invs, &mut fds) {
            Ok(e) => e,
            Err(e) => return every(format!("{action}: {e}")),
        };
        let verb = if fds.is_empty() { "batch".to_string() } else { format!("batch+{}", fds.len()) };
        let request = format!("{verb} {action} {} {envelope}", invs.len());

        let Some(worker) = self.worker.as_mut() else { return every("the worker is gone".into()) };
        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        let blamed = Blamed::default();
        let mut talk = JobsTalk::new(invs, progress);
        talk.adapters = self.adapters.as_mut();
        talk.blame = Some((&self.devices, &blamed));
        let x = Exchange { fds: &borrowed, talk: Some(&mut talk), renew: true };
        let answer = self.start.health.exchange_on(&self.devices, worker, &request, self.start.deadlines.run, x, &|| blamed.devices());
        if !matches!(answer, Err(CallError::Fault(_))) {
            if let Some(cause) = blamed.record(&self.start.health) {
                // The jobs keep the answers the worker gave; the instance is lost with its suspect card.
                self.lost = Some(cause);
            }
        }
        match answer {
            Ok((payload, fds)) => match codec::decode_results(&payload, fds) {
                Ok((results, metrics)) => {
                    self.metrics = metrics;
                    results
                }
                Err(e) => every(self.lose(format!("{action}: the worker's answer breaks the protocol ({e})"))),
            },
            Err(CallError::Failed(message)) => match message.strip_prefix(PANICKED) {
                // The same unwind an in-process panic takes: the lane fails
                // the group and the executor rebuilds the instance.
                Some(what) => {
                    let what = format!("{action}: the worker panicked:{what}");
                    self.lost = Some(what.clone());
                    panic!("{what}");
                }
                None => every(message),
            },
            Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => every(self.lose(e.to_string())),
            Err(e @ CallError::Invalid(_)) => every(e.to_string()),
        }
    }

    /// One open group as one exchange with the worker. Whatever ends the
    /// exchange, every job of the group that was not finished yet is
    /// finished with why, once.
    #[cfg(unix)]
    fn group_in_worker(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        use gpu_supervisor::{CallError, Exchange};
        let synced = self.sync_adapters();
        let mut fds = Vec::new();
        let mut talk = JobsTalk::new(invs, progress);
        talk.feed = Some(feed);
        if let Err(why) = synced {
            talk.fail_unfinished(&why);
            return true;
        }
        let envelope = match codec::encode_jobs(invs, &mut fds) {
            Ok(e) => e,
            Err(e) => {
                talk.fail_unfinished(&format!("{action}: {e}"));
                return true;
            }
        };
        let verb = if fds.is_empty() { "group".to_string() } else { format!("group+{}", fds.len()) };
        let request = format!("{verb} {action} {} {envelope}", invs.len());
        let Some(worker) = self.worker.as_mut() else {
            talk.fail_unfinished("the worker is gone");
            return true;
        };
        talk.adapters = self.adapters.as_mut();
        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        let blamed = Blamed::default();
        talk.blame = Some((&self.devices, &blamed));
        let x = Exchange { fds: &borrowed, talk: Some(&mut talk), renew: true };
        let answer = self.start.health.exchange_on(&self.devices, worker, &request, self.start.deadlines.run, x, &|| blamed.devices());
        if !matches!(answer, Err(CallError::Fault(_))) {
            if let Some(cause) = blamed.record(&self.start.health) {
                self.lost = Some(cause);
            }
        }
        // `talk` borrows the adapter feed until the group's jobs are finished, so the worker is marked lost by its field.
        let lose = |lost: &mut Option<String>, why: String| {
            *lost = Some(why.clone());
            why
        };
        let why = match answer {
            Ok((payload, fds)) => match codec::decode_opened(&payload, fds) {
                Ok((true, metrics)) => {
                    self.metrics = metrics;
                    return true;
                }
                Ok((false, _)) if !talk.touched() => {
                    self.closed_only.insert(action.to_string());
                    return false;
                }
                Ok((false, _)) => lose(&mut self.lost, format!("{action}: the worker said it ran no open group after it had run one")),
                Err(e) => lose(&mut self.lost, format!("{action}: the worker's answer breaks the protocol ({e})")),
            },
            Err(CallError::Failed(message)) => match message.strip_prefix(PANICKED) {
                // The same unwind an in-process panic takes; the lane answers
                // every job the group had not finished.
                Some(what) => {
                    let what = format!("{action}: the worker panicked:{what}");
                    self.lost = Some(what.clone());
                    panic!("{what}");
                }
                None => message,
            },
            Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => lose(&mut self.lost, e.to_string()),
            Err(e @ CallError::Invalid(_)) => e.to_string(),
        };
        talk.fail_unfinished(&why);
        true
    }

    #[cfg(not(unix))]
    fn group_in_worker(&mut self, _action: &str, _invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress), _feed: &mut dyn JobFeed) -> bool {
        false
    }

    #[cfg(not(unix))]
    fn batch_in_worker(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().map(|_| Err("supervised device workers need a Unix host".into())).collect()
    }

    /// Mark the worker unusable for every later run, and say why.
    fn lose(&mut self, why: String) -> String {
        self.lost = Some(why.clone());
        why
    }
}

/// The lane's side of a group while it is in flight: each job's progress to
/// its own caller, and one `cancel <job>` for each job whose caller asks. For
/// an open group also the executor's [`JobFeed`]: the worker's `take` is
/// answered `joined` with the jobs the feed gave, its `done` finishes a job
/// through the feed, and its `close` is answered with the feed's verdict.
/// Jobs are numbered as the feed numbers them: the group's own first, then
/// each that joined, in order.
#[cfg(unix)]
struct JobsTalk<'a> {
    /// The cancellation token of each job the worker knows, by index.
    tokens: Vec<capability::CancelToken>,
    cancel_sent: Vec<bool>,
    /// Finished through the feed (an open group's jobs only).
    finished: Vec<bool>,
    own: usize,
    progress: &'a mut dyn FnMut(usize, Progress),
    feed: Option<&'a mut dyn JobFeed>,
    /// The feed's tickets for the waiting jobs the last `waiting` frame
    /// carried, by their index in it, until the worker's verdicts on them.
    screening: Vec<u64>,
    /// The base's adapters, for a base that serves them: each change made while the group runs is sent as a frame,
    /// before the reply to the worker's next `take`, so no job that joins after it reaches the worker first.
    adapters: Option<&'a mut AdapterFeed>,
    /// The worker's cards, and the ones it says faulted while the group runs.
    blame: Option<(&'a [Device], &'a Blamed)>,
}

#[cfg(unix)]
impl<'a> JobsTalk<'a> {
    fn new(invs: &[Invocation], progress: &'a mut dyn FnMut(usize, Progress)) -> JobsTalk<'a> {
        let mut talk = JobsTalk { tokens: Vec::new(), cancel_sent: Vec::new(), finished: Vec::new(), own: invs.len(), progress, feed: None, screening: Vec::new(), adapters: None, blame: None };
        talk.know(invs);
        talk
    }

    fn know(&mut self, invs: &[Invocation]) {
        self.tokens.extend(invs.iter().map(|inv| inv.cancel.clone()));
        self.cancel_sent.resize(self.tokens.len(), false);
        self.finished.resize(self.tokens.len(), false);
    }

    /// Whether the worker took a joiner or finished a job.
    fn touched(&self) -> bool {
        self.tokens.len() > self.own || self.finished.iter().any(|&f| f)
    }

    /// Finish every job of an open group not finished yet with `why`.
    fn fail_unfinished(&mut self, why: &str) {
        let Some(feed) = self.feed.as_mut() else { return };
        for (job, finished) in self.finished.iter_mut().enumerate() {
            if !*finished {
                *finished = true;
                feed.finish(job, Err(why.to_string()));
            }
        }
    }

    /// The adapter changes made since the worker was last told, as frames.
    fn adapter_changes(&mut self) -> Vec<gpu_supervisor::Outgoing> {
        self.adapters.as_mut().map(|feed| feed.changes()).unwrap_or_default().iter().map(|c| gpu_supervisor::Outgoing::line(c.line())).collect()
    }

    fn feed(&mut self, kind: &str) -> Result<&mut dyn JobFeed, String> {
        match self.feed.as_deref_mut() {
            Some(feed) => Ok(feed),
            None => Err(format!("a closed group takes progress frames, not {kind:?}")),
        }
    }
}

#[cfg(unix)]
impl gpu_supervisor::Conversation for JobsTalk<'_> {
    fn frame(&mut self, kind: &str, payload: &str, fds: Vec<std::os::fd::OwnedFd>) -> Result<Vec<gpu_supervisor::Outgoing>, String> {
        use gpu_supervisor::Outgoing;
        match kind {
            "progress" => {
                let (job, p) = codec::decode_progress(payload, fds).map_err(|e| format!("the worker's progress breaks the protocol ({e})"))?;
                if job >= self.tokens.len() {
                    return Err(format!("progress for job {job} of a group of {}", self.tokens.len()));
                }
                (self.progress)(job, p);
                Ok(Vec::new())
            }
            "take" => {
                let max: usize = payload.parse().map_err(|_| format!("take {payload:?} is not a count"))?;
                let joined = self.feed(kind)?.take(max);
                if joined.len() > max {
                    return Err(format!("the feed gave {} jobs for a take of {max}", joined.len()));
                }
                let mut fds = Vec::new();
                let envelope = codec::encode_jobs(&joined, &mut fds)?;
                let verb = if fds.is_empty() { "joined".to_string() } else { format!("joined+{}", fds.len()) };
                self.know(&joined);
                // The adapters first: a job that joined after a change was made must find it in the worker.
                let mut out = self.adapter_changes();
                out.push(Outgoing { line: format!("{verb} {} {envelope}", joined.len()), fds });
                Ok(out)
            }
            "done" => {
                let (job, result) = codec::decode_done(payload, fds)?;
                match self.finished.get(job) {
                    Some(false) => {}
                    Some(true) => return Err(format!("job {job} finished twice")),
                    None => return Err(format!("job {job} finished in a group of {}", self.tokens.len())),
                }
                self.finished[job] = true;
                self.feed(kind)?.finish(job, result);
                Ok(Vec::new())
            }
            "screen" => {
                let (tickets, waiting): (Vec<u64>, Vec<Invocation>) = self.feed(kind)?.unscreened().into_iter().unzip();
                let mut fds = Vec::new();
                let envelope = codec::encode_jobs(&waiting, &mut fds)?;
                let verb = if fds.is_empty() { "waiting".to_string() } else { format!("waiting+{}", fds.len()) };
                self.screening = tickets;
                Ok(vec![Outgoing { line: format!("{verb} {} {envelope}", waiting.len()), fds }])
            }
            "refused" => {
                let verdicts = codec::decode_refused(payload, fds).map_err(|e| format!("the worker's verdicts break the protocol ({e})"))?;
                let tickets = std::mem::take(&mut self.screening);
                let feed = self.feed(kind)?;
                for (job, why) in verdicts {
                    let ticket = *tickets.get(job).ok_or_else(|| format!("a verdict on waiting job {job} of {}", tickets.len()))?;
                    feed.refuse(ticket, why);
                }
                Ok(Vec::new())
            }
            "metrics" => {
                let metrics = codec::decode_control(payload, fds).map_err(|e| format!("the worker's metrics break the protocol ({e})"))?;
                self.feed(kind)?.publish(metrics);
                Ok(Vec::new())
            }
            "close" => {
                let closed = self.feed(kind)?.close();
                Ok(vec![Outgoing::line(format!("closed {closed}"))])
            }
            "fault" => {
                let (devices, blamed) = self.blame.ok_or("a fault frame for a group whose devices are not known")?;
                blamed.note(devices, payload).map(|()| Vec::new())
            }
            other => Err(format!("a group takes progress, take, done, screen, refused, metrics, close and fault frames, not {other:?}")),
        }
    }

    fn poll(&mut self) -> Vec<gpu_supervisor::Outgoing> {
        let mut out = self.adapter_changes();
        for (job, token) in self.tokens.iter().enumerate() {
            if !self.cancel_sent[job] && !self.finished[job] && token.is_cancelled() {
                self.cancel_sent[job] = true;
                out.push(gpu_supervisor::Outgoing::line(format!("cancel {job}")));
            }
        }
        out
    }

    fn polls(&self) -> bool {
        self.adapters.is_some() || self.cancel_sent.iter().zip(&self.finished).any(|(&sent, &finished)| !sent && !finished)
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
