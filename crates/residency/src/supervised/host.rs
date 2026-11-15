// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running model code in
// killable worker processes for its clients. If your team needs expertise in
// process isolation for accelerator workloads then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The worker side of a hosted model: the request loop a supervised worker
//! runs ([`host`]), over any [`ResidentModel`] a [`HostedFamily`] rebuilds.

use std::cell::Cell;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use capability::{CancelToken, Progress};
use serde_json::{json, Value};

use super::codec::{decode_jobs, encode_control, encode_done, encode_opened, encode_progress, encode_refused, encode_results, object};
use super::adapters::Change;
use super::{parse_device_label, AdapterSink, HostedFamily, PANICKED, PROTOCOL};
use crate::{Device, Instance, InstanceKey, Tier};

/// The worker side: host the model a supervisor asks for on the socket this
/// process was started with (its stdin), until the supervisor closes it.
/// Returns the process exit code.
///
/// Refused (exit code 2, nothing served) unless stdin is a socket and the
/// first request is `hello` with this [`PROTOCOL`]: this is an internal entry
/// point a supervisor reaches, not a command. Once accepted, the protocol
/// moves to a private descriptor, stdin reads nothing and stdout goes to
/// stderr, so model code that prints cannot corrupt the protocol.
#[cfg(target_os = "linux")]
pub fn host(families: &[HostedFamily]) -> i32 {
    match gpu_supervisor::take_worker_socket() {
        Ok(sock) => host_on(sock, families),
        Err(why) => {
            eprintln!("brain: this is an internal entry point: `brain serve` starts it to host a model in a supervised worker, and it is not run by hand ({why})");
            2
        }
    }
}

/// One frame a worker read, with the descriptors it declared and, for a
/// group, the token of each job it carries.
struct Incoming {
    id: u64,
    verb: String,
    rest: String,
    fds: Vec<OwnedFd>,
    tokens: Vec<CancelToken>,
}

/// The group read last and not finished yet, for a cancel frame to reach:
/// its request id and one token per job, by index. One request is in flight
/// at a time, so there is at most one.
type Current = Arc<Mutex<Option<(u64, Vec<CancelToken>)>>>;

/// The hosted model's adapter registry, once `open` built a base that serves adapters: the request loop applies
/// `attach` and `detach` requests to it, the reader thread the frames that arrive while a group runs.
type Adapters = Arc<Mutex<Option<Arc<dyn AdapterSink>>>>;

/// [`host`] on a socket the caller already holds: the worker's request loop.
/// The calling thread runs every request and writes every frame (one writer);
/// a reader thread reads frames and delivers a `cancel <job>` to the job it
/// names, and only to that one. A group's tokens are registered as the reader
/// reads the group, so a cancellation that arrives before a job starts still
/// reaches it, and the host clears only its own entry when the group ends,
/// so one that arrives after the group ended reaches nothing.
pub fn host_on(sock: UnixStream, families: &[HostedFamily]) -> i32 {
    let current: Current = Arc::new(Mutex::new(None));
    let adapters: Adapters = Arc::new(Mutex::new(None));
    let (tx, rx) = std::sync::mpsc::channel::<Result<Incoming, String>>();
    let reading = match sock.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain worker: cannot read the supervisor's socket: {e}");
            return 2;
        }
    };
    let cancels = current.clone();
    let sink = adapters.clone();
    let spawned = std::thread::Builder::new().name("brain-worker-reader".into()).spawn(move || read_requests(reading, &cancels, &sink, &tx));
    if let Err(e) = spawned {
        eprintln!("brain worker: cannot start its reader: {e}");
        return 2;
    }
    let mut host = Host { sock, current, adapters, rx, families, greeted: false, instance: None, device: None };
    loop {
        let incoming = match host.rx.recv() {
            Ok(Ok(incoming)) => incoming,
            Ok(Err(why)) => {
                eprintln!("brain worker: {why}");
                return 2;
            }
            Err(_) => return 0,
        };
        if let Step::Exit(code) = host.handle(incoming) {
            return code;
        }
    }
}

/// How many jobs a frame announces before its JSON: `<k>` in
/// `batch <action> <k> <jobs>`, `group <action> <k> <jobs>`,
/// `joined <k> <jobs>` and `waiting <k> <jobs>`.
fn announced_jobs(verb: &str, rest: &str) -> Option<usize> {
    match verb {
        "batch" | "group" => rest.split(' ').nth(1)?.parse().ok(),
        "joined" | "waiting" => rest.split(' ').next()?.parse().ok(),
        _ => Some(0),
    }
}

/// The reader thread: frames in, requests to the host loop, cancellations to
/// the job they name, and an adapter change sent as a frame about a request
/// already read (a group that runs, or one that just ended) to the model's
/// registry at once, so changes apply in the order they were sent and a
/// running group refreshes from them as it would in process. Ends at end of
/// stream or a broken frame.
fn read_requests(sock: UnixStream, current: &Current, adapters: &Adapters, tx: &std::sync::mpsc::Sender<Result<Incoming, String>>) {
    let mut reader = gpu_supervisor::wire::Reader::new(gpu_supervisor::MAX_REQUEST_BYTES);
    // The newest request id read: a supervisor numbers its requests in order, so an id up to it names a request
    // already read, and the frame is about that request.
    let mut newest = 0u64;
    loop {
        let line = match reader.line() {
            Ok(Some(line)) => line,
            Ok(None) => match reader.fill(&sock) {
                Ok(0) => return,
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    let _ = tx.send(Err(format!("the supervisor's socket failed: {e}")));
                    return;
                }
            },
            Err(why) => {
                let _ = tx.send(Err(why));
                return;
            }
        };
        let mut parts = line.splitn(3, ' ');
        let (Some(id), Some(word)) = (parts.next().and_then(|i| i.parse::<u64>().ok()), parts.next()) else {
            let _ = tx.send(Err(format!("a frame without an id and a request: {:?}", line.chars().take(80).collect::<String>())));
            return;
        };
        let rest = parts.next().unwrap_or("").to_string();
        let (verb, n) = match word.split_once('+') {
            Some((verb, n)) => match n.parse::<usize>() {
                Ok(n) if (1..=gpu_supervisor::wire::MAX_FDS).contains(&n) => (verb, n),
                _ => {
                    let _ = tx.send(Err(format!("request {id} declares descriptors it may not carry ({word:?})")));
                    return;
                }
            },
            None => (word, 0),
        };
        let fds = match reader.take_fds(n) {
            Ok(fds) => fds,
            Err(why) => {
                let _ = tx.send(Err(why));
                return;
            }
        };
        let Some(jobs) = announced_jobs(verb, &rest) else {
            let _ = tx.send(Err(format!("request {id} does not say how many jobs it carries")));
            return;
        };
        let tokens: Vec<CancelToken> = (0..jobs).map(|_| CancelToken::armed()).collect();
        let fresh = id > newest;
        newest = newest.max(id);
        {
            let mut current = current.lock().unwrap_or_else(|e| e.into_inner());
            match verb {
                "attach" | "detach" if !fresh => {
                    let sink = adapters.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    let applied = Change::parse(verb, &rest).and_then(|change| match sink {
                        Some(sink) => change.apply(&*sink),
                        None => Err("this worker's model serves no adapters".into()),
                    });
                    if let Err(why) = applied {
                        eprintln!("brain worker: {why}");
                    }
                    continue;
                }
                "cancel" => {
                    if let (Some((running, held)), Ok(job)) = (current.as_ref(), rest.parse::<usize>()) {
                        if *running == id {
                            if let Some(token) = held.get(job) {
                                token.cancel();
                            }
                        }
                    }
                    continue;
                }
                "batch" | "group" => *current = Some((id, tokens.clone())),
                // The jobs that joined a running group follow its own, in
                // the order they joined, as the parent numbers them. Jobs
                // only `waiting` to join are not the group's: a cancel cannot
                // name them, so their tokens are registered nowhere.
                "joined" => {
                    if let Some((running, held)) = current.as_mut() {
                        if *running == id {
                            held.extend(tokens.iter().cloned());
                        }
                    }
                }
                _ => {}
            }
        }
        if tx.send(Ok(Incoming { id, verb: verb.to_string(), rest, fds, tokens })).is_err() {
            return;
        }
    }
}

enum Step {
    Continue,
    Exit(i32),
}

struct Host<'a> {
    sock: UnixStream,
    current: Current,
    adapters: Adapters,
    rx: Receiver<Result<Incoming, String>>,
    families: &'a [HostedFamily],
    greeted: bool,
    instance: Option<Box<dyn Instance>>,
    /// The device `open` built the instance on: the only one this worker ever opens.
    device: Option<Device>,
}

impl Host<'_> {
    /// Write one frame; a supervisor that is gone ends the worker.
    fn send(&self, line: &str, fds: &[OwnedFd]) -> Result<(), Step> {
        let mut frame = line.replace(['\n', '\r'], " ");
        frame.push('\n');
        let borrowed: Vec<_> = fds.iter().map(std::os::fd::AsFd::as_fd).collect();
        gpu_supervisor::wire::send(&self.sock, frame.as_bytes(), &borrowed).map_err(|_| Step::Exit(0))
    }

    /// `<id> <kind>[+<n>] <payload>` with `fds` beside it.
    fn send_frame(&self, id: u64, kind: &str, payload: &str, fds: &[OwnedFd]) -> Result<(), Step> {
        if fds.is_empty() {
            self.send(&format!("{id} {kind} {payload}"), &[])
        } else {
            self.send(&format!("{id} {kind}+{} {payload}", fds.len()), fds)
        }
    }

    fn answer(&self, id: u64, result: Result<(String, Vec<OwnedFd>), String>) -> Step {
        let sent = match result {
            Ok((payload, fds)) => self.send_frame(id, "ok", &payload, &fds),
            Err(message) => self.send(&format!("{id} err {message}"), &[]),
        };
        match sent {
            Ok(()) => Step::Continue,
            Err(step) => step,
        }
    }

    fn handle(&mut self, incoming: Incoming) -> Step {
        let id = incoming.id;
        let step = self.dispatch(incoming);
        // The request is answered: a cancel naming it from now on reaches nothing.
        let mut current = self.current.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|(running, _)| *running == id) {
            *current = None;
        }
        step
    }

    fn dispatch(&mut self, incoming: Incoming) -> Step {
        let Incoming { id, verb, rest, fds, tokens } = incoming;
        if !self.greeted {
            let want = PROTOCOL.to_string();
            return if verb == "hello" && rest == want {
                self.greeted = true;
                self.answer(id, Ok((json!({ "protocol": PROTOCOL, "pid": std::process::id() }).to_string(), Vec::new())))
            } else {
                // Not a supervisor that speaks this protocol: serve nothing.
                let _ = self.answer(id, Err(format!("the first request must be `hello {PROTOCOL}`, not {verb:?}")));
                Step::Exit(2)
            };
        }
        match verb.as_str() {
            "open" => {
                let opened = self.open(&rest);
                self.control_answer(id, opened)
            }
            "demote" => {
                let demoted = self.demote(&rest);
                self.control_answer(id, demoted)
            }
            "promote" => {
                let promoted = self.promote(&rest);
                self.control_answer(id, promoted)
            }
            "attach" | "detach" => {
                let sink = self.adapters.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let applied = Change::parse(&verb, &rest).and_then(|change| match sink {
                    Some(sink) => change.apply(&*sink),
                    None => Err("this worker's model serves no adapters".into()),
                });
                self.control_answer(id, applied)
            }
            "batch" => self.batch(id, &rest, fds, tokens),
            "group" => self.group(id, &rest, fds, tokens),
            other => self.answer(id, Err(format!("a hosted model takes hello, open, demote, promote, attach, detach, batch and group, not {other:?}"))),
        }
    }

    fn open(&mut self, rest: &str) -> Result<(), String> {
        if self.instance.is_some() {
            return Err("this worker already hosts an instance".into());
        }
        let (label, spec) = rest.split_once(' ').ok_or("open needs a device and a spec")?;
        let device = parse_device_label(label)?;
        let spec: Value = serde_json::from_str(spec).map_err(|e| format!("open: the spec is not JSON ({e})"))?;
        let spec = object(&spec, "the open spec", &["family", "args", "key"])?;
        let family = spec.get("family").and_then(Value::as_str).ok_or("the open spec names no family")?;
        let key = object(spec.get("key").unwrap_or(&Value::Null), "the instance key", &["model", "config"])?;
        let field = |k: &str| key.get(k).and_then(Value::as_str).ok_or_else(|| format!("the instance key has no {k}"));
        let key = InstanceKey::new(field("model")?, field("config")?);
        let hosted = self.families.iter().find(|f| f.name == family).ok_or_else(|| {
            let known: Vec<&str> = self.families.iter().map(|f| f.name).collect();
            format!("this worker hosts no family {family:?} (it hosts {known:?})")
        })?;
        let args = spec.get("args").cloned().unwrap_or(Value::Null);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (hosted.open)(&args).and_then(|model| model.resident.activate(&key, device).map(|instance| (instance, model.adapters)))));
        match built {
            Ok(built) => {
                let (instance, adapters) = built?;
                self.instance = Some(instance);
                self.device = Some(device);
                *self.adapters.lock().unwrap_or_else(|e| e.into_inner()) = adapters;
                Ok(())
            }
            Err(p) => Err(format!("{PANICKED} {}", panic_text(p.as_ref()))),
        }
    }

    /// The answer to a request that runs no job: the instance's metrics after it, or why it failed. A panic in
    /// model code ends the worker once it is answered, as in a job.
    fn control_answer(&mut self, id: u64, done: Result<(), String>) -> Step {
        let panicked = done.as_ref().err().is_some_and(|e| e.starts_with(PANICKED));
        let reply = done.and_then(|()| {
            let mut fds = Vec::new();
            encode_control(&self.metrics(), &mut fds).map(|payload| (payload, fds))
        });
        let step = self.answer(id, reply);
        if panicked {
            Step::Exit(101)
        } else {
            step
        }
    }

    fn metrics(&self) -> super::codec::Metrics {
        self.instance.as_ref().map(|i| i.metrics()).unwrap_or_default()
    }

    /// `demote warm|cold`: the instance releases its device buffers and keeps what it holds on the host. The
    /// worker keeps its device context, which is why a demoted hosted instance stays charged it on its card.
    fn demote(&mut self, rest: &str) -> Result<(), String> {
        let tier = match rest {
            "warm" => Tier::Warm,
            "cold" => Tier::Cold,
            "hot" => Tier::Hot,
            other => return Err(format!("demote {other:?}: the tier is warm or cold")),
        };
        let instance = self.instance.as_mut().ok_or("no instance is open; send `open` first")?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.demote(tier))).unwrap_or_else(|p| Err(format!("{PANICKED} {}", panic_text(p.as_ref()))))
    }

    /// `promote <device>`: the instance rebuilds what its demotion released, on the card this worker owns. Another
    /// card is refused: a worker opens one, and the parent moves an instance by starting a new worker.
    fn promote(&mut self, rest: &str) -> Result<(), String> {
        let device = parse_device_label(rest)?;
        if self.device != Some(device) {
            return Err(format!("this worker owns {:?}; promoting onto {rest} needs a worker of its own", self.device));
        }
        let instance = self.instance.as_mut().ok_or("no instance is open; send `open` first")?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.promote(device))).unwrap_or_else(|p| Err(format!("{PANICKED} {}", panic_text(p.as_ref()))))
    }

    /// A job's progress as a frame; once the supervisor is gone nothing more is sent.
    fn progress(&self, id: u64, job: usize, p: &Progress, gone: &Cell<bool>) {
        if gone.get() {
            return;
        }
        let mut fds = Vec::new();
        match encode_progress(job, p, &mut fds) {
            Ok(payload) => gone.set(self.send_frame(id, "progress", &payload, &fds).is_err()),
            Err(e) => eprintln!("brain worker: a progress update could not be sent: {e}"),
        }
    }

    /// `batch <action> <k> <jobs>`: the instance's own `run_batch` over the group.
    fn batch(&mut self, id: u64, rest: &str, fds: Vec<OwnedFd>, tokens: Vec<CancelToken>) -> Step {
        let prepared = (|| {
            let mut words = rest.splitn(3, ' ');
            let (Some(action), Some(_), Some(jobs)) = (words.next(), words.next(), words.next()) else {
                return Err("batch needs an action, a job count and the jobs".to_string());
            };
            Ok((action.to_string(), decode_jobs(jobs, fds, tokens)?))
        })();
        let (action, invs) = match prepared {
            Ok(p) => p,
            Err(why) => return self.answer(id, Err(why)),
        };
        let Some(mut instance) = self.instance.take() else {
            return self.answer(id, Err("no instance is open; send `open` first".into()));
        };
        let gone = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.run_batch(&action, &invs, &mut |job, p| self.progress(id, job, &p, &gone))));
        if gone.get() {
            return Step::Exit(0);
        }
        match result {
            Ok(results) => {
                self.instance = Some(instance);
                let mut fds = Vec::new();
                let reply = encode_results(&results, &self.metrics(), &mut fds).map(|payload| (payload, fds));
                self.answer(id, reply)
            }
            Err(p) => {
                // The instance may be broken; its parent rebuilds it in a new worker.
                let _ = self.answer(id, Err(format!("{PANICKED} {}", panic_text(p.as_ref()))));
                Step::Exit(101)
            }
        }
    }
}

impl Host<'_> {
    /// `group <action> <k> <jobs>`: the instance's own `run_open` over the
    /// group, with a [`WorkerFeed`] that reaches the parent's. The answer says
    /// whether it ran the group open; when not, it did nothing and the parent
    /// sends the group again as a closed batch.
    fn group(&mut self, id: u64, rest: &str, fds: Vec<OwnedFd>, tokens: Vec<CancelToken>) -> Step {
        let prepared = (|| {
            let mut words = rest.splitn(3, ' ');
            let (Some(action), Some(_), Some(jobs)) = (words.next(), words.next(), words.next()) else {
                return Err("group needs an action, a job count and the jobs".to_string());
            };
            Ok((action.to_string(), decode_jobs(jobs, fds, tokens)?))
        })();
        let (action, invs) = match prepared {
            Ok(p) => p,
            Err(why) => return self.answer(id, Err(why)),
        };
        let Some(mut instance) = self.instance.take() else {
            return self.answer(id, Err("no instance is open; send `open` first".into()));
        };
        let gone = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut feed = WorkerFeed { host: self, id, gone: &gone };
            let mut progress = |job: usize, p: Progress| self.progress(id, job, &p, &gone);
            instance.run_open(&action, &invs, &mut progress, &mut feed)
        }));
        if gone.get() {
            return Step::Exit(0);
        }
        match result {
            Ok(open) => {
                self.instance = Some(instance);
                let mut fds = Vec::new();
                let reply = encode_opened(open, &self.metrics(), &mut fds).map(|payload| (payload, fds));
                self.answer(id, reply)
            }
            Err(p) => {
                let _ = self.answer(id, Err(format!("{PANICKED} {}", panic_text(p.as_ref()))));
                Step::Exit(101)
            }
        }
    }

    /// The parent's reply to a frame this worker sent about request `id`:
    /// the next frame the reader hands over, which must be `verb` for `id`.
    /// `None` once the supervisor is gone or broke the protocol; the worker
    /// then sends nothing more and exits after the group.
    fn reply(&self, id: u64, verb: &str, gone: &Cell<bool>) -> Option<Incoming> {
        if gone.get() {
            return None;
        }
        match self.rx.recv() {
            Ok(Ok(incoming)) if incoming.id == id && incoming.verb == verb => Some(incoming),
            Ok(Ok(other)) => {
                eprintln!("brain worker: request {id} waited for {verb:?} and got {:?} for request {}", other.verb, other.id);
                gone.set(true);
                None
            }
            Ok(Err(why)) => {
                eprintln!("brain worker: {why}");
                gone.set(true);
                None
            }
            Err(_) => {
                gone.set(true);
                None
            }
        }
    }
}

/// The worker's side of the parent's [`crate::JobFeed`] for one open group:
/// `take`, `screen` and `close` are round trips to the parent's feed
/// (`take <max>` answered `joined`, `screen` answered `waiting` and followed
/// by `refused` when the check turns a job away, `close` answered `closed`),
/// `finish` is one `done` frame. Every frame is about the group's request and
/// written by the request loop, the socket's one writer.
struct WorkerFeed<'h, 'a> {
    host: &'h Host<'a>,
    id: u64,
    gone: &'h Cell<bool>,
}

impl crate::JobFeed for WorkerFeed<'_, '_> {
    fn take(&mut self, max: usize) -> Vec<capability::Invocation> {
        if max == 0 || self.gone.get() {
            return Vec::new();
        }
        if self.host.send(&format!("{} take {max}", self.id), &[]).is_err() {
            self.gone.set(true);
            return Vec::new();
        }
        let Some(Incoming { rest, fds, tokens, .. }) = self.host.reply(self.id, "joined", self.gone) else { return Vec::new() };
        let Some((_, jobs)) = rest.split_once(' ') else {
            eprintln!("brain worker: a joined frame without its jobs");
            self.gone.set(true);
            return Vec::new();
        };
        match decode_jobs(jobs, fds, tokens) {
            Ok(invs) => invs,
            Err(why) => {
                eprintln!("brain worker: the jobs that joined break the protocol ({why})");
                self.gone.set(true);
                Vec::new()
            }
        }
    }

    fn finish(&mut self, index: usize, result: capability::ActionResult) {
        if self.gone.get() {
            return;
        }
        let mut fds = Vec::new();
        let sent = match encode_done(index, &result, &mut fds) {
            Ok(payload) => self.host.send_frame(self.id, "done", &payload, &fds),
            Err(e) => {
                let mut fds = Vec::new();
                match encode_done(index, &Err(format!("the result could not cross to the server: {e}")), &mut fds) {
                    Ok(payload) => self.host.send_frame(self.id, "done", &payload, &fds),
                    Err(_) => Err(Step::Exit(0)),
                }
            }
        };
        self.gone.set(sent.is_err());
    }

    /// The check runs here, in the worker, on copies of the parent's waiting
    /// jobs; the parent's feed refuses the ones it turns away.
    fn screen(&mut self, check: &mut dyn FnMut(&capability::Invocation) -> Option<String>) {
        if self.gone.get() {
            return;
        }
        if self.host.send(&format!("{} screen", self.id), &[]).is_err() {
            self.gone.set(true);
            return;
        }
        let Some(Incoming { rest, fds, tokens, .. }) = self.host.reply(self.id, "waiting", self.gone) else { return };
        let Some((_, jobs)) = rest.split_once(' ') else {
            eprintln!("brain worker: a waiting frame without its jobs");
            self.gone.set(true);
            return;
        };
        let waiting = match decode_jobs(jobs, fds, tokens) {
            Ok(invs) => invs,
            Err(why) => {
                eprintln!("brain worker: the waiting jobs break the protocol ({why})");
                self.gone.set(true);
                return;
            }
        };
        let refused: Vec<(usize, String)> = waiting.iter().enumerate().filter_map(|(job, inv)| check(inv).map(|why| (job, why))).collect();
        if refused.is_empty() {
            return;
        }
        let mut fds = Vec::new();
        let sent = match encode_refused(&refused, &mut fds) {
            Ok(payload) => self.host.send_frame(self.id, "refused", &payload, &fds),
            Err(e) => {
                eprintln!("brain worker: the verdicts could not be sent: {e}");
                Err(Step::Exit(0))
            }
        };
        self.gone.set(sent.is_err());
    }

    fn close(&mut self) -> bool {
        if self.gone.get() || self.host.send(&format!("{} close", self.id), &[]).is_err() {
            self.gone.set(true);
            return true;
        }
        match self.host.reply(self.id, "closed", self.gone) {
            Some(incoming) => match incoming.rest.as_str() {
                "true" => true,
                "false" => false,
                other => {
                    eprintln!("brain worker: closed {other:?} is neither true nor false");
                    self.gone.set(true);
                    true
                }
            },
            // Nobody to ask: end the group.
            None => true,
        }
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "a panic without a message".into())
}
