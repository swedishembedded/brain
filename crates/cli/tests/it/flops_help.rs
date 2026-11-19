// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain perf flops --help` is a request, not a misuse.
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
    // Cargo names the binary it built for this test; a path derived from the test
    // executable's own location breaks whenever cargo moves its build directory.
    brain_testutil::package_bin("brain").to_string_lossy().into_owned()
}

fn assert_help_flag(flag: &str) {
    // `--device cpu` keeps the machine out of stderr: with no flag the CLI resolves every device it has, and says
    // so when `BRAIN_DEVICE`/`BRAIN_BACKEND` are set or an NVIDIA card is driven without CUDA. Those notices are
    // the start-up's, not the help's.
    let out = Command::new(bin()).args(["--device", "cpu", "perf", "flops"]).arg(flag).output().expect("run brain perf flops");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "brain perf flops {flag} exited {:?}; stderr: {stderr}", out.status.code());
    assert!(stdout.contains("usage: brain perf flops"), "brain perf flops {flag} printed no usage on stdout: {stdout:?}");
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
    let out = Command::new(bin()).args(["perf", "flops"]).output().expect("run brain perf flops");
    assert!(!out.status.success(), "brain perf flops with no --model must not report success");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(stderr.contains("usage: brain perf flops"), "a refused invocation must say how to invoke it: {stderr:?}");
}
