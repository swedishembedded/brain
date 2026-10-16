// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for servers that stop cleanly on
// SIGTERM whatever state their log pipe is in for its clients. If your team
// needs expertise in service lifecycle and graceful shutdown then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A server whose stdout/stderr reader has gone away must still stop on
//! SIGINT/SIGTERM. The signal handler runs in a real child process (this test
//! binary re-run on [`CHILD`]) because the property is about process-wide
//! signal disposition and a broken pipe, neither of which an in-process test
//! can arrange.

#![cfg(unix)]

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CHILD: &str = "child_waits_for_shutdown";
const READY: &str = "ready";
const FIRED_EXIT: i32 = 0;
const NOT_FIRED_EXIT: i32 = 3;

/// The child half: installs the signal handling exactly as `brain serve` does
/// and exits 0 once the shutdown fires. Ignored so only the parent runs it.
#[test]
#[ignore = "runs only as the child of the signal tests"]
fn child_waits_for_shutdown() {
    let (trigger, shutdown) = brain_shutdown::channel();
    brain_shutdown::install_signals(trigger);
    println!("{READY}");
    let until = Instant::now() + Duration::from_secs(10);
    while Instant::now() < until {
        if shutdown.is_shutdown() {
            std::process::exit(FIRED_EXIT);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    std::process::exit(NOT_FIRED_EXIT);
}

/// Start the child with its stderr reader already closed, wait until it has
/// installed its handlers, send `signal`, and return how it ended.
fn signal_child_with_closed_stderr(signal: i32) -> Option<i32> {
    let mut child = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--ignored", "--exact", CHILD, "--nocapture", "--test-threads=1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child");
    drop(child.stderr.take()); // the reader of stderr is gone: every write fails with EPIPE
    let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
    let mut line = String::new();
    // The libtest progress line precedes the marker on the same line.
    while !line.trim_end().ends_with(READY) {
        line.clear();
        assert!(stdout.read_line(&mut line).expect("read the child's stdout") > 0, "the child exited before it was ready");
    }
    // SAFETY: `kill` on a child this test spawned.
    assert_eq!(unsafe { libc::kill(child.id() as i32, signal) }, 0);

    let until = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("poll the child") {
            return status.code();
        }
        if Instant::now() >= until {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn sigterm_stops_a_process_whose_stderr_reader_is_gone() {
    assert_eq!(signal_child_with_closed_stderr(libc::SIGTERM), Some(FIRED_EXIT), "SIGTERM must fire the shutdown even when logging it fails");
}

#[test]
fn sigint_stops_a_process_whose_stderr_reader_is_gone() {
    assert_eq!(signal_child_with_closed_stderr(libc::SIGINT), Some(FIRED_EXIT), "SIGINT must fire the shutdown even when logging it fails");
}
