// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Process-level tests for `brain serve` / `brain run`'s flag parsing: unknown
//! flags must be a hard error (not the old warn-and-continue), and `--help`
//! must actually print usage instead of falling into the blocking stdio loop.
//!
//! Regression coverage for an incident where `brain serve --listen HOST:PORT`
//! (a flag that never existed) used to be silently ignored, exit 0, and
//! never open a listener.
//!
//! IMPORTANT: every test here passes `.stdin(Stdio::null())`. Without it, a
//! regression back to warn-and-continue falls into the blocking stdio JSONL
//! loop, which reads from whatever stdin `cargo test` gave the child — this is
//! what turns a silent regression into a LOUD, fast test failure (wrong exit
//! code) instead of either a hang or a flaky pass, since `Stdio::null()`
//! guarantees stdin reads EOF immediately either way.

use std::process::{Command, Stdio};

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(bin())
        .args(args)
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .unwrap_or_else(|e| panic!("run brain {args:?}: {e}"))
}

#[test]
fn unknown_flag_is_a_hard_error_with_usage() {
    // The exact flag from the bench-integration-friction incident.
    let out = run(&["serve", "--listen", "0.0.0.0:8788"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown flag"), "stderr: {stderr}");
    assert!(stderr.contains("--listen"), "stderr: {stderr}");
    // The usage text (printed on the error path) must actually help the reader
    // find the real flag.
    assert!(stderr.contains("--openai"), "stderr: {stderr}");
}

#[test]
fn unknown_flag_is_a_hard_error_for_the_run_alias() {
    let out = run(&["run", "--nope"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--nope"));
}

#[test]
fn non_numeric_port_errors_instead_of_silently_defaulting() {
    // Before this change, a non-numeric token after --openai was left
    // unconsumed by take_port, silently fell through with the default port
    // 8788 bound, and "foo" was then dropped on the floor by the old
    // warn-and-continue arm.
    let out = run(&["serve", "--openai", "foo"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("foo"));
}

#[test]
fn out_of_range_port_errors() {
    let out = run(&["serve", "--openai", "99999"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn bare_positional_is_an_error() {
    let out = run(&["serve", "models.safetensors"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn value_taking_flag_with_no_value_errors() {
    let out = run(&["serve", "--models-dir"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--models-dir needs a value"));
}

#[test]
fn serve_help_exits_zero_with_usage_on_stdout() {
    for flag in ["--help", "-h"] {
        let out = run(&["serve", flag]);
        assert!(out.status.success(), "brain serve {flag}: {}", String::from_utf8_lossy(&out.stderr));
        assert!(out.stderr.is_empty(), "stderr should be empty for {flag}: {}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        for f in ["--openai", "--anthropic", "--openrouter", "--dbus", "--models-dir", "--api-keys-out", "--reserve-gb", "--ready-file"] {
            assert!(stdout.contains(f), "brain serve {flag} stdout missing {f}:\n{stdout}");
        }
    }
}

#[test]
fn run_help_alias_also_exits_zero_with_usage() {
    let out = run(&["run", "--help"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("--openai"));
}
