// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flops --help` is a request, not a misuse.
//!
//! Swedish Embedded AB implements dependable command-line tooling for
//! embedded and edge-AI systems. If your team needs a CLI whose help output
//! is trustworthy enough to gate on, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! Asking a tool for its usage used to print `ignoring unrecognised args:
//! ["--help"]` on stderr, then the usage anyway, then exit non-zero - which
//! makes a working command look broken to a human and look failed to any
//! script that checks a status code. This gates the three things that
//! together mean "help was understood": exit 0, the usage on stdout, and
//! nothing on stderr.

use std::process::Command;

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

fn assert_help_flag(flag: &str) {
    let out = Command::new(bin()).arg("flops").arg(flag).output().expect("run brain flops");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "brain flops {flag} exited {:?}; stderr: {stderr}", out.status.code());
    assert!(stdout.contains("usage: brain flops"), "brain flops {flag} printed no usage on stdout: {stdout:?}");
    assert!(stdout.contains("--model"), "the usage must name the flag the command cannot run without: {stdout:?}");
    assert!(stderr.is_empty(), "asking for help is not a misuse and must warn about nothing: {stderr:?}");
}

#[test]
fn asking_flops_for_help_prints_usage_and_succeeds() {
    assert_help_flag("--help");
    assert_help_flag("-h");
}

/// The other half of the contract: an invocation that really is unusable
/// still fails, and still says how to use it. A `--help` fix that turned
/// every bad invocation into a success would be a worse defect than the one
/// it replaced.
#[test]
fn flops_without_a_model_still_fails_with_the_usage() {
    let out = Command::new(bin()).arg("flops").output().expect("run brain flops");
    assert!(!out.status.success(), "brain flops with no --model must not report success");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(stderr.contains("usage: brain flops"), "a refused invocation must say how to invoke it: {stderr:?}");
}
