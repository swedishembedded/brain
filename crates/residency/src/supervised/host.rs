// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running model code in
// killable worker processes for its clients. If your team needs expertise in
// process isolation for accelerator workloads then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The worker side of a hosted model: the request loop a supervised worker
//! runs ([`host`]), over any [`ResidentModel`] a [`HostedFamily`] rebuilds.

use std::collections::VecDeque;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};

use capability::{CancelToken, Invocation, Progress};
use serde_json::{json, Value};

use super::codec::{encode_progress, no_leftovers, object, pack, read_blobs, unpack, write_blobs};
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

/// One request a worker read, with the descriptors it declared and, for a
/// run, the token its cancellation fires.
struct Incoming {
    id: u64,
    verb: String,
    rest: String,
    fds: Vec<OwnedFd>,
    cancel: CancelToken,
}

/// The run read last and not finished yet, for a cancel frame to reach. One
/// request is in flight at a time, so there is at most one.
type Current = Arc<Mutex<Option<(u64, CancelToken)>>>;

/// [`host`] on a socket the caller already holds: the worker's request loop.
/// The calling thread runs every request and writes every frame (one writer);
/// a reader thread reads frames and delivers a `cancel` to the run it names,
/// and only to that one. A run's token is registered as the reader reads the
/// run, so a cancellation that arrives before the run starts still reaches
/// it, and the host clears only its own entry when the run ends, so one that
/// arrives after its run ended reaches nothing.
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
    let mut host = Host { sock, current, families, greeted: false, instance: None };
    while let Ok(next) = rx.recv() {
        let Incoming { id, verb, rest, fds, cancel } = match next {
            Ok(incoming) => incoming,
            Err(why) => {
                eprintln!("brain worker: {why}");
                return 2;
            }
        };
        match host.handle(id, &verb, &rest, fds, cancel) {
            Step::Continue => {}
            Step::Exit(code) => return code,
        }
    }
    0
}

/// The reader thread: frames in, requests to the host loop, cancellations to
/// the run they name. Ends at end of stream or a broken frame.
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
        let cancel = CancelToken::armed();
        {
            let mut current = current.lock().unwrap_or_else(|e| e.into_inner());
            match verb {
                "cancel" => {
                    if let Some((running, token)) = current.as_ref() {
                        if *running == id {
                            token.cancel();
                        }
                    }
                    continue;
                }
                "run" => *current = Some((id, cancel.clone())),
                _ => {}
            }
        }
        if tx.send(Ok(Incoming { id, verb: verb.to_string(), rest, fds, cancel })).is_err() {
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

    fn answer(&self, id: u64, result: Result<(String, Vec<OwnedFd>), String>) -> Step {
        let sent = match result {
            Ok((payload, fds)) if fds.is_empty() => self.send(&format!("{id} ok {payload}"), &[]),
            Ok((payload, fds)) => self.send(&format!("{id} ok+{} {payload}", fds.len()), &fds),
            Err(message) => self.send(&format!("{id} err {message}"), &[]),
        };
        match sent {
            Ok(()) => Step::Continue,
            Err(step) => step,
        }
    }

    fn handle(&mut self, id: u64, verb: &str, rest: &str, fds: Vec<OwnedFd>, cancel: CancelToken) -> Step {
        let step = self.dispatch(id, verb, rest, fds, cancel);
        // The request is answered: a cancel naming it from now on reaches nothing.
        let mut current = self.current.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|(running, _)| *running == id) {
            *current = None;
        }
        step
    }

    fn dispatch(&mut self, id: u64, verb: &str, rest: &str, fds: Vec<OwnedFd>, cancel: CancelToken) -> Step {
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
        match verb {
            "open" => {
                let opened = self.open(rest);
                let failed = opened.as_ref().err().is_some_and(|e| e.starts_with(PANICKED));
                let step = self.answer(id, opened.map(|()| ("{}".to_string(), Vec::new())));
                if failed {
                    Step::Exit(101)
                } else {
                    step
                }
            }
            "run" => self.run(id, rest, fds, cancel),
            other => self.answer(id, Err(format!("a hosted model takes hello, open and run, not {other:?}"))),
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

    fn run(&mut self, id: u64, rest: &str, fds: Vec<OwnedFd>, cancel: CancelToken) -> Step {
        let prepared = (|| {
            let (action, envelope) = rest.split_once(' ').ok_or("run needs an action and an envelope")?;
            let mut fds = VecDeque::from(fds);
            let v = unpack(envelope, &mut fds)?;
            let map = object(&v, "the run envelope", &["params", "scope", "blobs"])?;
            let blobs = read_blobs(map.get("blobs"), &mut fds)?;
            no_leftovers(&fds)?;
            let params = map.get("params").cloned().unwrap_or_else(|| json!({}));
            let scope = map.get("scope").and_then(Value::as_str).unwrap_or("").to_string();
            Ok::<_, String>((action.to_string(), Invocation { params, blobs, cancel, scope }))
        })();
        let (action, inv) = match prepared {
            Ok(p) => p,
            Err(why) => return self.answer(id, Err(why)),
        };
        let Some(mut instance) = self.instance.take() else {
            return self.answer(id, Err("no instance is open; send `open` first".into()));
        };
        let mut gone = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut progress = |p: Progress| {
                if gone {
                    return;
                }
                let mut fds = Vec::new();
                let sent = match encode_progress(&p, &mut fds) {
                    Ok(payload) if fds.is_empty() => self.send(&format!("{id} progress {payload}"), &[]),
                    Ok(payload) => self.send(&format!("{id} progress+{} {payload}", fds.len()), &fds),
                    Err(e) => {
                        eprintln!("brain worker: a progress update could not be sent: {e}");
                        Ok(())
                    }
                };
                gone = sent.is_err();
            };
            instance.run(&action, &inv, &mut progress)
        }));
        if gone {
            return Step::Exit(0);
        }
        match result {
            Ok(outcome) => {
                self.instance = Some(instance);
                let reply = outcome.and_then(|o| {
                    let mut fds = Vec::new();
                    let blobs = write_blobs(&o.blobs, &mut fds)?;
                    let payload = pack(json!({ "outputs": o.outputs, "blobs": blobs }).to_string(), &mut fds)?;
                    Ok((payload, fds))
                });
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

