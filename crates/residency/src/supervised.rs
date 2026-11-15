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
//! | `open <device> <spec>` | `{}`; the instance is built on `<device>` |
//! | `batch[+<n>] <action> <k> <jobs>` | `{"results": [...]}`, one per job: `{"outputs", "blobs"}` or `{"error"}` |
//! | `group[+<n>] <action> <k> <jobs>` | `{"open": bool}`; the jobs were finished through `done` frames |
//! | `probe <device>` | anything; the device works (sent to a fresh probe worker) |
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
//! has the parent's feed refuse each at once. A worker that dies mid-group
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

use crate::{AdapterIdentity, Device, Instance, InstanceKey, JobFeed, MemCost, ResidentModel, Tier};

/// The protocol version a worker and its parent must agree on (`hello`).
pub const PROTOCOL: u32 = 3;

#[cfg(unix)]
mod codec;
#[cfg(unix)]
mod host;

#[cfg(target_os = "linux")]
pub use host::host;
#[cfg(unix)]
pub use host::host_on;

/// Device memory a worker's own device context holds on its card beside the instance it hosts: what every in-process
/// instance of a card shares through the serving process's one context and each worker pays again, so every hosted
/// instance is charged it on top of its estimate ([`MemCost::with_device_context`]). It is the card's used memory
/// with a worker built on it less what the same instance adds beside an open context, as the driver reports both, and
/// rounded up to a whole 64 MiB; `a_worker_hosted_encoder_costs_a_round_trip_per_group_and_a_context_per_card` (the
/// brain CLI's `worker_hosted_lfm` tests) measures it on a real card and fails when a worker holds more.
pub const WORKER_DEVICE_CONTEXT_BYTES: u64 = 192 << 20;

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

    /// The model's own estimate plus its worker's device context: each instance is one worker on one card.
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key).with_device_context(WORKER_DEVICE_CONTEXT_BYTES)
    }

    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        self.inner.estimate_at(key, tier).with_device_context(WORKER_DEVICE_CONTEXT_BYTES)
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
        Ok(Box::new(SupervisedInstance { worker, device, health: self.health.clone(), run: self.deadlines.run, lost: None, closed_only: std::collections::BTreeSet::new() }))
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
    /// Actions the worker's instance runs only as closed groups: its
    /// `run_open` said so, having done nothing, so it is not asked again and
    /// such a group costs one request, not two. Whether a model runs a group
    /// open depends on the action alone in every model that does; were it to
    /// depend on more, a remembered answer would only make it batch a closed
    /// group, which every model answers correctly.
    closed_only: std::collections::BTreeSet<String>,
}

impl Instance for SupervisedInstance {
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
    /// One group as one exchange with the worker: the jobs (and their blobs
    /// in a memfd) out, each job's progress and cancellation while it runs,
    /// every job's result back.
    #[cfg(unix)]
    fn batch_in_worker(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        use gpu_supervisor::{CallError, Exchange};
        let every = |why: String| invs.iter().map(|_| Err(why.clone())).collect::<Vec<_>>();
        let mut fds = Vec::new();
        let envelope = match codec::encode_jobs(invs, &mut fds) {
            Ok(e) => e,
            Err(e) => return every(format!("{action}: {e}")),
        };
        let verb = if fds.is_empty() { "batch".to_string() } else { format!("batch+{}", fds.len()) };
        let request = format!("{verb} {action} {} {envelope}", invs.len());

        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        let mut talk = JobsTalk::new(invs, progress);
        let x = Exchange { fds: &borrowed, talk: Some(&mut talk), renew: true };
        let answer = self.health.exchange(&self.device, &mut self.worker, &request, self.run, x);
        match answer {
            Ok((payload, fds)) => match codec::decode_results(&payload, fds) {
                Ok(results) => results,
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
        let mut fds = Vec::new();
        let mut talk = JobsTalk::new(invs, progress);
        talk.feed = Some(feed);
        let envelope = match codec::encode_jobs(invs, &mut fds) {
            Ok(e) => e,
            Err(e) => {
                talk.fail_unfinished(&format!("{action}: {e}"));
                return true;
            }
        };
        let verb = if fds.is_empty() { "group".to_string() } else { format!("group+{}", fds.len()) };
        let request = format!("{verb} {action} {} {envelope}", invs.len());
        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        let x = Exchange { fds: &borrowed, talk: Some(&mut talk), renew: true };
        let answer = self.health.exchange(&self.device, &mut self.worker, &request, self.run, x);
        let why = match answer {
            Ok((payload, fds)) => match codec::decode_opened(&payload, fds) {
                Ok(true) => return true,
                Ok(false) if !talk.touched() => {
                    self.closed_only.insert(action.to_string());
                    return false;
                }
                Ok(false) => self.lose(format!("{action}: the worker said it ran no open group after it had run one")),
                Err(e) => self.lose(format!("{action}: the worker's answer breaks the protocol ({e})")),
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
            Err(e @ (CallError::Fault(_) | CallError::Refused { .. })) => self.lose(e.to_string()),
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
}

#[cfg(unix)]
impl<'a> JobsTalk<'a> {
    fn new(invs: &[Invocation], progress: &'a mut dyn FnMut(usize, Progress)) -> JobsTalk<'a> {
        let mut talk = JobsTalk { tokens: Vec::new(), cancel_sent: Vec::new(), finished: Vec::new(), own: invs.len(), progress, feed: None, screening: Vec::new() };
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
                Ok(vec![Outgoing { line: format!("{verb} {} {envelope}", joined.len()), fds }])
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
            "close" => {
                let closed = self.feed(kind)?.close();
                Ok(vec![Outgoing::line(format!("closed {closed}"))])
            }
            other => Err(format!("a group takes progress, take, done, screen, refused and close frames, not {other:?}")),
        }
    }

    fn poll(&mut self) -> Vec<gpu_supervisor::Outgoing> {
        let mut out = Vec::new();
        for (job, token) in self.tokens.iter().enumerate() {
            if !self.cancel_sent[job] && !self.finished[job] && token.is_cancelled() {
                self.cancel_sent[job] = true;
                out.push(gpu_supervisor::Outgoing::line(format!("cancel {job}")));
            }
        }
        out
    }

    fn polls(&self) -> bool {
        self.cancel_sent.iter().zip(&self.finished).any(|(&sent, &finished)| !sent && !finished)
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
