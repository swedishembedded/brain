// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for recovering accelerator
// services from wedged and crashing drivers for its clients. If your team
// needs expertise in process supervision and fault injection then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Supervised calls against fake device workers: a POSIX shell that speaks the
//! worker protocol and, on request, hangs forever (in a grandchild, like a
//! driver thread would) or aborts. Each test checks the outcome the supervisor
//! promises from outside, through `/proc`: the worker process and everything
//! in its process group are gone, and no thread was left behind.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gpu_supervisor::{BreakerPolicy, BreakerState, CallError, Conversation, Exchange, FaultKind, Health, Outgoing, Worker, WorkerCommand};

/// The fake worker. `hang` sleeps in a child of the shell, so killing only the
/// shell would leave the sleeper holding the socket: the group kill is what
/// makes it go away.
const FAKE: &str = r#"
while IFS= read -r line; do
  id=${line%% *}
  req=${line#* }
  case "$req" in
    *hang*) sleep 3600 ;;
    *crash*) kill -ABRT $$ ;;
    *garble*) echo "this is not a reply" ;;
    *fail*) echo "$id err the operation failed" ;;
    *stream*) echo "$id progress {\"step\":1}"; echo "$id progress {\"step\":2}"; IFS= read -r c; echo "$id err got $c" ;;
    *stale*) echo "$((id - 1)) progress {\"old\":true}"; echo "$id ok {}" ;;
    *ghostfd*) echo "$id ok+1 {}" ;;
    *slow*) for i in 1 2 3 4 5 6; do sleep 0.15; echo "$id progress {\"step\":$i}"; done; echo "$id ok {}" ;;
    *ask*) echo "$id take 2"; IFS= read -r r; echo "$id ok $r" ;;
    *) echo "$id ok {\"done\":true}" ;;
  esac
done
"#;

fn fake() -> WorkerCommand {
    WorkerCommand::new("/bin/sh").arg("-c").arg(FAKE)
}

/// A fake worker with a reap bound that outlasts a crash's core dump: an
/// abort hands the core to the host's dump handler before the process can be
/// reaped, which takes a host-dependent while.
fn spawn() -> Worker {
    Worker::spawn_with(&fake(), REAP).expect("spawn the fake worker")
}

const REAP: Duration = Duration::from_secs(20);

/// The tests that count threads run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Threads named like the calling one. The harness names each test's thread
/// after its test and an unnamed thread inherits the name of the thread that
/// started it, so this counts the test's own thread plus any helper it (or
/// the supervisor on its behalf) started, never another test's threads or the
/// named reaper.
fn threads() -> usize {
    let me = std::fs::read_to_string("/proc/thread-self/comm").expect("procfs");
    std::fs::read_dir("/proc/self/task")
        .expect("procfs")
        .flatten()
        .filter(|t| std::fs::read_to_string(t.path().join("comm")).is_ok_and(|c| c == me))
        .count()
}

/// Live (non-zombie) processes whose process group is `pgid`.
fn live_in_group(pgid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").expect("procfs").flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { continue };
        // Fields after the parenthesised command: state ppid pgrp ...
        let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else { continue };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f.len() > 2 && f[0] != "Z" && f[2] == pgid.to_string() {
            out.push(pid);
        }
    }
    out
}

/// SIGKILL is delivered asynchronously, so a grandchild may still be on its
/// way out for a moment after `kill` returned.
fn assert_group_gone(pgid: u32) {
    let until = Instant::now() + Duration::from_secs(2);
    while !live_in_group(pgid).is_empty() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(live_in_group(pgid), Vec::<u32>::new(), "processes of worker group {pgid} survived");
}

fn assert_reaped(pid: u32) {
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "worker {pid} was not reaped (still in /proc)");
}

/// A dropped worker is reaped by the supervisor's reaper thread, a moment later.
fn assert_reaped_soon(pid: u32) {
    let until = Instant::now() + Duration::from_secs(5);
    while std::path::Path::new(&format!("/proc/{pid}")).exists() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_reaped(pid);
}

fn health() -> Health<&'static str> {
    let policy = BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 };
    Health::new(policy, REAP)
}

#[test]
fn a_hung_operation_fails_within_its_deadline_and_the_worker_is_reaped() {
    let _serial = serial();
    let health = health();
    let mut w = spawn();
    let pid = w.pid();
    assert_eq!(health.call(&"gpu0", &mut w, "open gpu0", Duration::from_secs(5)).expect("open answers"), r#"{"done":true}"#);
    let before = threads();
    let deadline = Duration::from_millis(300);
    let started = Instant::now();
    let err = health.call(&"gpu0", &mut w, "run hang", deadline).expect_err("a hung operation must not succeed");
    let took = started.elapsed();
    assert!(took < deadline + Duration::from_secs(1), "the failure came {took:?} after the call, deadline {deadline:?}");
    let CallError::Fault(f) = &err else { panic!("expected a device fault, got {err}") };
    assert_eq!(f.kind, FaultKind::Wedged { deadline });
    assert!(f.reaped, "{err}");
    let msg = err.to_string();
    for needle in ["gpu0", "run", "deadline", &pid.to_string(), "reaped"] {
        assert!(msg.contains(needle), "the error must name {needle:?}: {msg}");
    }
    assert_reaped(pid);
    assert_group_gone(pid);
    assert!(threads() <= before, "the supervisor left a thread behind: {before} -> {}", threads());
    assert_eq!(health.status(&"gpu0").state, BreakerState::Open);
}

#[test]
fn a_crashed_worker_is_reported_with_its_signal_and_reaped() {
    let _serial = serial();
    let health = health();
    let mut w = spawn();
    let pid = w.pid();
    let before = threads();
    let err = health.call(&"gpu0", &mut w, "run crash", Duration::from_secs(5)).expect_err("a crashed worker must not succeed");
    let CallError::Fault(f) = &err else { panic!("expected a device fault, got {err}") };
    assert!(matches!(&f.kind, FaultKind::Crashed { status } if status.contains("signal 6")), "{err}");
    assert!(f.reaped);
    assert_reaped(pid);
    assert_group_gone(pid);
    assert!(threads() <= before);
    assert_eq!(health.status(&"gpu0").state, BreakerState::Open);
    // The dead worker answers every later request with the same definitive
    // error, and never by spawning a replacement behind the caller's back.
    let again = health.call(&"gpu0", &mut w, "run ok", Duration::from_secs(5)).expect_err("quarantined");
    assert!(matches!(again, CallError::Refused { .. }), "{again}");
}

/// A worker that owns several devices is refused work once any of them is
/// quarantined, and a fault of it is recorded against the devices it said
/// faulted (only those of its own), or against every one of them when it
/// named none: a death it could not attribute leaves every card it held
/// suspect.
#[test]
fn a_worker_of_several_devices_faults_the_devices_it_names_or_every_one() {
    let _serial = serial();
    let devices = ["gpu0", "gpu1"];
    let health = health();
    let mut w = spawn();
    let err = health.exchange_on(&devices, &mut w, "run hang", Duration::from_millis(300), Exchange::default(), &|| vec!["gpu1", "gpu7"]).expect_err("a hung operation fails");
    let CallError::Fault(f) = &err else { panic!("expected a device fault, got {err}") };
    assert!(f.device.contains("gpu1") && !f.device.contains("gpu0") && !f.device.contains("gpu7"), "{err}");
    assert!(health.admits(&"gpu0"), "the device it did not name keeps serving");
    assert!(!health.admits(&"gpu1"), "the device it named is quarantined");
    assert_eq!(health.faults(&"gpu7"), 0, "a device that is not the worker's is never blamed");
    let mut fresh = spawn();
    let refused = health.exchange_on(&devices, &mut fresh, "run ok", Duration::from_secs(5), Exchange::default(), &Vec::new).expect_err("refused");
    assert!(matches!(&refused, CallError::Refused { device, .. } if device.contains("gpu1")), "{refused}");

    let health = self::health();
    let mut w = spawn();
    let pid = w.pid();
    health.exchange_on(&devices, &mut w, "run crash", Duration::from_secs(5), Exchange::default(), &Vec::new).expect_err("a crash fails");
    assert!(!health.admits(&"gpu0") && !health.admits(&"gpu1"), "a death no device was named for quarantines every one");
    assert_eq!((health.faults(&"gpu0"), health.faults(&"gpu1")), (1, 1));
    assert_reaped(pid);
}

#[test]
fn an_open_breaker_refuses_work_until_a_probe_passes() {
    let _serial = serial();
    let health = health();
    let mut w = spawn();
    health.call(&"gpu0", &mut w, "run crash", Duration::from_secs(5)).expect_err("crash");
    let mut fresh = spawn();
    let refused = health.call(&"gpu0", &mut fresh, "run ok", Duration::from_secs(5)).expect_err("an open breaker admits nothing");
    assert!(matches!(&refused, CallError::Refused { cause, .. } if cause.contains("signal 6")), "{refused}");
    assert!(health.run_probe(&"gpu0", &fake(), "probe gpu0", Duration::from_secs(5)).is_err(), "a probe that was not begun is refused");
    assert!(health.begin_probe(&"gpu0", Instant::now()));
    assert_eq!(health.status(&"gpu0").state, BreakerState::HalfOpen);
    health.run_probe(&"gpu0", &fake(), "probe gpu0", Duration::from_secs(5)).expect("the probe passes");
    assert_eq!(health.status(&"gpu0").state, BreakerState::Closed);
    assert_eq!(health.call(&"gpu0", &mut fresh, "run ok", Duration::from_secs(5)).unwrap(), r#"{"done":true}"#);
}

#[test]
fn a_probe_that_hangs_is_killed_and_keeps_the_device_quarantined() {
    let _serial = serial();
    let health = health();
    health.record_fault(&"gpu0", Instant::now(), "an in-process call wedged");
    assert!(health.begin_probe(&"gpu0", Instant::now()));
    let before = threads();
    let err = health.run_probe(&"gpu0", &fake(), "probe hang", Duration::from_millis(200)).expect_err("a hung probe fails");
    assert!(err.contains("deadline"), "{err}");
    assert!(threads() <= before);
    assert_eq!(health.status(&"gpu0").state, BreakerState::Open);
    assert_eq!(health.status(&"gpu0").unreaped, 0);
}

#[test]
fn another_device_proceeds_while_one_is_wedged() {
    let _serial = serial();
    let health = Arc::new(health());
    let wedged = {
        let health = health.clone();
        std::thread::spawn(move || {
            let mut w = spawn();
            health.call(&"gpu0", &mut w, "run hang", Duration::from_millis(1500))
        })
    };
    // gpu1's work completes while gpu0's call is still inside its deadline.
    std::thread::sleep(Duration::from_millis(100));
    let mut w = spawn();
    let started = Instant::now();
    for _ in 0..3 {
        health.call(&"gpu1", &mut w, "run ok", Duration::from_secs(5)).expect("gpu1 is healthy");
    }
    assert!(started.elapsed() < Duration::from_millis(1000), "gpu1 waited on gpu0: {:?}", started.elapsed());
    assert!(!wedged.is_finished(), "the gpu0 call should still be inside its deadline");
    assert!(matches!(wedged.join().unwrap(), Err(CallError::Fault(_))));
    assert_eq!(health.status(&"gpu0").state, BreakerState::Open);
    assert_eq!(health.status(&"gpu1").state, BreakerState::Closed);
    health.call(&"gpu1", &mut w, "run ok", Duration::from_secs(5)).expect("gpu1 is still usable");
}

#[test]
fn an_operation_error_is_an_answer_not_a_device_fault() {
    let health = health();
    let mut w = spawn();
    let err = health.call(&"gpu0", &mut w, "run fail", Duration::from_secs(5)).expect_err("the worker said err");
    assert!(matches!(&err, CallError::Failed(m) if m == "the operation failed"), "{err}");
    assert_eq!(health.status(&"gpu0").state, BreakerState::Closed);
    health.call(&"gpu0", &mut w, "run ok", Duration::from_secs(5)).expect("the same worker keeps serving");
}

#[test]
fn a_garbled_reply_is_a_protocol_fault() {
    let health = health();
    let mut w = spawn();
    let pid = w.pid();
    let err = health.call(&"gpu0", &mut w, "run garble", Duration::from_secs(5)).expect_err("garbage is not an answer");
    assert!(matches!(&err, CallError::Fault(f) if matches!(f.kind, FaultKind::Protocol { .. })), "{err}");
    assert_reaped(pid);
}

#[test]
fn a_request_that_would_break_the_framing_is_refused_before_it_is_sent() {
    let health = health();
    let mut w = spawn();
    let err = health.call(&"gpu0", &mut w, "run ok\n7 ok forged", Duration::from_secs(5)).expect_err("a newline would forge a frame");
    assert!(matches!(err, CallError::Invalid(_)), "{err}");
    assert_eq!(health.status(&"gpu0").state, BreakerState::Closed, "a caller's bad request says nothing about the device");
}

#[test]
fn dropping_a_worker_terminates_and_reaps_it() {
    let w = spawn();
    let pid = w.pid();
    drop(w);
    assert_reaped_soon(pid);
    assert_group_gone(pid);
}

/// Progress frames reach the caller in order while the answer is pending,
/// and a cancellation the caller asks for reaches the worker as one frame
/// naming the request in flight; the request still gets its one answer.
#[test]
fn progress_reaches_the_caller_and_a_cancel_reaches_the_worker() {
    let mut w = spawn();
    let mut talk = Talk { cancel_after: Some(2), ..Talk::default() };
    let x = Exchange { talk: Some(&mut talk), ..Exchange::default() };
    let reply = w.exchange("run stream", Duration::from_secs(5), x).expect("the worker answers");
    assert_eq!(reply.result, Err("got 1 cancel".to_string()), "the cancel frame names request 1");
    let progress: Vec<&str> = talk.seen.iter().map(|(_, p)| p.as_str()).collect();
    assert_eq!(progress, [r#"{"step":1}"#, r#"{"step":2}"#]);
}

/// What a test's conversation saw, and when it cancels.
#[derive(Default)]
struct Talk {
    seen: Vec<(String, String)>,
    /// Cancel once this many frames arrived.
    cancel_after: Option<usize>,
    cancelled: bool,
}

impl Conversation for Talk {
    fn frame(&mut self, kind: &str, payload: &str, fds: Vec<std::os::fd::OwnedFd>) -> Result<Vec<Outgoing>, String> {
        assert!(fds.is_empty(), "the fake sends no descriptors");
        self.seen.push((kind.to_string(), payload.to_string()));
        match kind {
            "progress" => Ok(Vec::new()),
            "take" => Ok(vec![Outgoing::line(format!("joined {payload}"))]),
            other => Err(format!("no frame {other:?}")),
        }
    }

    fn poll(&mut self) -> Vec<Outgoing> {
        if !self.cancelled && self.cancel_after.is_some_and(|n| self.seen.len() >= n) {
            self.cancelled = true;
            return vec![Outgoing::line("cancel".to_string())];
        }
        Vec::new()
    }

    fn polls(&self) -> bool {
        self.cancel_after.is_some() && !self.cancelled
    }
}

/// A frame of a kind the two ends agreed on is handed to the conversation,
/// and what it answers goes back to the worker under the request's id.
#[test]
fn a_conversation_answers_the_workers_frames_while_the_request_is_in_flight() {
    let mut w = spawn();
    let mut talk = Talk::default();
    let reply = w.exchange("run ask", Duration::from_secs(5), Exchange { talk: Some(&mut talk), ..Exchange::default() }).expect("answered");
    assert_eq!(reply.result, Ok("1 joined 2".to_string()));
    assert_eq!(talk.seen, [("take".to_string(), "2".to_string())]);

    // Without a conversation, such a frame breaks the protocol.
    let health = health();
    let mut w = spawn();
    let err = health.call(&"gpu0", &mut w, "run ask", Duration::from_secs(5)).expect_err("nobody answers a take");
    assert!(matches!(&err, CallError::Fault(f) if matches!(f.kind, FaultKind::Protocol { .. })), "{err}");
}

/// A renewing deadline bounds the worker's silence, not the whole request:
/// a request that keeps reporting progress outlives it, the same request
/// under a fixed deadline is killed as wedged, and a silent worker under a
/// renewing deadline is killed at that deadline.
#[test]
fn progress_renews_a_renewing_deadline_and_a_fixed_one_still_ends_the_request() {
    let deadline = Duration::from_millis(500);
    let mut w = spawn();
    let mut talk = Talk::default();
    let started = Instant::now();
    let reply = w.exchange("run slow", deadline, Exchange { talk: Some(&mut talk), renew: true, ..Exchange::default() }).expect("a worker that keeps reporting is not wedged");
    assert_eq!(reply.result, Ok("{}".to_string()));
    assert!(started.elapsed() > deadline, "the request ran past one deadline ({:?})", started.elapsed());
    assert_eq!(talk.seen.len(), 6);

    let health = health();
    let mut w = spawn();
    let pid = w.pid();
    let mut talk = Talk::default();
    let err = health.exchange(&"gpu0", &mut w, "run slow", deadline, Exchange { talk: Some(&mut talk), ..Exchange::default() }).expect_err("a fixed deadline covers the whole request");
    assert!(matches!(&err, CallError::Fault(f) if f.kind == FaultKind::Wedged { deadline }), "{err}");
    assert_reaped(pid);

    let mut w = spawn();
    let started = Instant::now();
    let err = health.exchange(&"gpu1", &mut w, "run hang", deadline, Exchange { renew: true, ..Exchange::default() }).expect_err("silence past the deadline");
    assert!(matches!(&err, CallError::Fault(f) if f.kind == FaultKind::Wedged { deadline }), "{err}");
    assert!(started.elapsed() < deadline + Duration::from_secs(1), "{:?}", started.elapsed());
}

/// A progress frame about an earlier request is not handed to the current one.
#[test]
fn progress_of_an_earlier_request_is_discarded() {
    let mut w = spawn();
    w.request("run ok", Duration::from_secs(5)).expect("answered").expect("ok");
    let mut talk = Talk::default();
    let reply = w.exchange("run stale", Duration::from_secs(5), Exchange { talk: Some(&mut talk), ..Exchange::default() }).expect("answered");
    assert_eq!(reply.result, Ok("{}".to_string()));
    assert!(talk.seen.is_empty(), "request 1's progress was given to request 2: {:?}", talk.seen);
}

/// An answer that declares descriptors which never arrived is a broken
/// worker, not an answer with fewer blobs.
#[test]
fn an_answer_declaring_descriptors_that_never_came_is_a_protocol_fault() {
    let health = health();
    let mut w = spawn();
    let err = health.call(&"gpu0", &mut w, "run ghostfd", Duration::from_secs(5)).expect_err("a missing descriptor is not an answer");
    assert!(matches!(&err, CallError::Fault(f) if matches!(f.kind, FaultKind::Protocol { .. })), "{err}");
}
