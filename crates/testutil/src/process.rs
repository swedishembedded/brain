// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running a child process under a deadline, so a test whose program never exits fails instead of hanging the suite.
//!
//! Swedish Embedded AB implements test suites that fail fast and say why instead of stalling a CI run for its
//! clients. If your team needs expertise in testing command-line programs and servers, you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::io::Read;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// [`Command::output`] with a deadline: stdin is closed, stdout and stderr are captured, and a child still running
/// after `limit` is killed and the test panics with what it printed so far. A program that should refuse or answer
/// and exit can then never hang the suite, whatever it finds on the host.
pub fn output_within(cmd: &mut Command, limit: Duration) -> Output {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {cmd:?}: {e}"));
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            pipe.read_to_end(&mut bytes).ok();
            bytes
        })
    };
    let stdout = drain(Box::new(child.stdout.take().expect("piped stdout")));
    let stderr = drain(Box::new(child.stderr.take().expect("piped stderr")));
    let deadline = Instant::now() + limit;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap_or_else(|e| panic!("wait for {cmd:?}: {e}")) {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().ok();
            child.wait().ok();
            let (out, err) = (stdout.join().unwrap_or_default(), stderr.join().unwrap_or_default());
            panic!(
                "{cmd:?} was still running after {limit:?} and was killed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out),
                String::from_utf8_lossy(&err)
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Output { status, stdout: stdout.join().unwrap_or_default(), stderr: stderr.join().unwrap_or_default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A child that exits is answered with its status and output; one that does not is killed at the deadline and
    /// the call panics instead of waiting for it.
    #[test]
    fn a_child_past_its_deadline_is_killed_not_waited_for() {
        let out = output_within(Command::new("sh").args(["-c", "echo hi; echo err >&2; exit 3"]), Duration::from_secs(30));
        assert_eq!((out.status.code(), out.stdout.as_slice(), out.stderr.as_slice()), (Some(3), &b"hi\n"[..], &b"err\n"[..]));

        let start = Instant::now();
        let hung = std::panic::catch_unwind(|| output_within(Command::new("sleep").arg("30"), Duration::from_millis(200)));
        assert!(hung.is_err(), "a hung child must fail the call");
        assert!(start.elapsed() < Duration::from_secs(10), "the call returned at the deadline, not when the child exited");
    }
}
