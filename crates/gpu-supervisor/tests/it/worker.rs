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

use gpu_supervisor::{BreakerPolicy, BreakerState, CallError, FaultKind, Health, Worker, WorkerCommand};

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

/// Thread counts are per process, so the tests that compare them run one at a
/// time; the others still run in parallel threads of the harness, which only
/// ever lowers the count between two reads taken under this lock.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").expect("procfs").count()
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
    assert_reaped(pid);
    assert_group_gone(pid);
}
