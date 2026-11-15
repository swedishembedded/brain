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

use super::codec::{decode_jobs, encode_progress, encode_results, object};
use super::{parse_device_label, HostedFamily, PANICKED, PROTOCOL};
use crate::{Instance, InstanceKey};

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

/// [`host`] on a socket the caller already holds: the worker's request loop.
/// The calling thread runs every request and writes every frame (one writer);
/// a reader thread reads frames and delivers a `cancel <job>` to the job it
/// names, and only to that one. A group's tokens are registered as the reader
/// reads the group, so a cancellation that arrives before a job starts still
/// reaches it, and the host clears only its own entry when the group ends,
/// so one that arrives after the group ended reaches nothing.
pub fn host_on(sock: UnixStream, families: &[HostedFamily]) -> i32 {
    let current: Current = Arc::new(Mutex::new(None));
    let (tx, rx) = std::sync::mpsc::channel::<Result<Incoming, String>>();
    let reading = match sock.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain worker: cannot read the supervisor's socket: {e}");
            return 2;
        }
    };
    let cancels = current.clone();
    let spawned = std::thread::Builder::new().name("brain-worker-reader".into()).spawn(move || read_requests(reading, &cancels, &tx));
    if let Err(e) = spawned {
        eprintln!("brain worker: cannot start its reader: {e}");
        return 2;
    }
    let mut host = Host { sock, current, rx, families, greeted: false, instance: None };
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
/// `batch <action> <k> <jobs>`.
fn announced_jobs(verb: &str, rest: &str) -> Option<usize> {
    match verb {
        "batch" => rest.split(' ').nth(1)?.parse().ok(),
        _ => Some(0),
    }
}

/// The reader thread: frames in, requests to the host loop, cancellations to
/// the job they name. Ends at end of stream or a broken frame.
fn read_requests(sock: UnixStream, current: &Current, tx: &std::sync::mpsc::Sender<Result<Incoming, String>>) {
    let mut reader = gpu_supervisor::wire::Reader::new(gpu_supervisor::MAX_REQUEST_BYTES);
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
        {
            let mut current = current.lock().unwrap_or_else(|e| e.into_inner());
            match verb {
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
                "batch" => *current = Some((id, tokens.clone())),
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
    rx: Receiver<Result<Incoming, String>>,
    families: &'a [HostedFamily],
    greeted: bool,
    instance: Option<Box<dyn Instance>>,
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
                let failed = opened.as_ref().err().is_some_and(|e| e.starts_with(PANICKED));
                let step = self.answer(id, opened.map(|()| ("{}".to_string(), Vec::new())));
                if failed {
                    Step::Exit(101)
                } else {
                    step
                }
            }
            "batch" => self.batch(id, &rest, fds, tokens),
            other => self.answer(id, Err(format!("a hosted model takes hello, open and batch, not {other:?}"))),
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
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (hosted.open)(&args).and_then(|resident| resident.activate(&key, device))));
        match built {
            Ok(instance) => {
                self.instance = Some(instance?);
                Ok(())
            }
            Err(p) => Err(format!("{PANICKED} {}", panic_text(p.as_ref()))),
        }
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
                let reply = encode_results(&results, &mut fds).map(|payload| (payload, fds));
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

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "a panic without a message".into())
}
