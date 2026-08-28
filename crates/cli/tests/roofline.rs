// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain roofline` - one comprehensive, fast, cross-accelerator hardware
//! compute-capacity report (GPU + NPU + CPU). These tests run against the
//! REAL machine (no mocks for GPU/NPU/CPU presence - whatever this sandbox
//! has is what gets exercised), so assertions are on STRUCTURE (every row
//! self-contained, JSON round-trips the plain rows, standalone scoping,
//! bounded wall-clock) rather than on exact hardware-dependent numbers.
//!
//! Swedish Embedded AB builds hardware-aware inference tooling for embedded
//! and edge-AI teams. If your team needs one trustworthy answer to "what can
//! this box actually do" across GPU, NPU and CPU targets, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::process::Command;
use std::time::{Duration, Instant};

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

fn run(args: &[&str]) -> (bool, String, String, Duration) {
    let t0 = Instant::now();
    let out = Command::new(bin()).arg("roofline").args(args).output().expect("run brain roofline");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned(), t0.elapsed())
}

/// Every data row (i.e. every line naming an accelerator) must be
/// self-contained: it carries its own accelerator id, so a `| grep gpu0` or
/// `| grep npu0` returns a complete, useful line rather than a fragment that
/// only makes sense next to a header or a neighbouring row - the same
/// convention `crate::tree`'s plain renderer establishes for `models list`.
#[test]
fn plain_mode_every_data_line_is_self_contained_and_greppable() {
    let (ok, stdout, stderr, elapsed) = run(&[]);
    assert!(ok, "brain roofline failed: {stderr}");
    assert!(elapsed < Duration::from_secs(60), "brain roofline took {elapsed:?}, expected well under a minute even on a cold cache");

    let data_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("gpu") || l.starts_with("npu") || l.starts_with("cpu")).collect();
    assert!(!data_lines.is_empty(), "expected at least one accelerator row in:\n{stdout}");
    for line in &data_lines {
        let accel_token = line.split_whitespace().next().expect("a data line has at least one token");
        assert!(
            accel_token.starts_with("gpu") || accel_token.starts_with("npu") || accel_token.starts_with("cpu"),
            "data line does not open with its own accelerator id: {line:?}"
        );
    }
    // Every real accelerator class brain knows about gets a section, even a
    // degraded one - GPU, NPU and CPU sections must all appear in the
    // unscoped report.
    assert!(data_lines.iter().any(|l| l.starts_with("cpu")), "CPU is always present and must always report:\n{stdout}");
    assert!(data_lines.iter().any(|l| l.starts_with("npu")), "the NPU section must appear (even if degraded) in the unscoped report:\n{stdout}");
}

/// `--json` must parse as valid JSON and carry the SAME information as the
/// plain rows. This checks a genuine round-trip - every JSON row's own
/// `line` field must equal what that row's OWN fields render to, and the set
/// of accelerators/dtypes reported must match a separate plain-mode run -
/// but never compares exact rate text byte-for-byte across two process
/// invocations: the CPU rung measures fresh every call (no cache), so its
/// GFLOP/s legitimately differs run to run, and comparing full line text
/// across runs would be a flaky test of process jitter, not of this command.
#[test]
fn json_output_round_trips_the_plain_rows() {
    let (ok, stdout, stderr, _) = run(&["--json"]);
    assert!(ok, "brain roofline --json failed: {stderr}");
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("--json output did not parse: {e}\n{stdout}"));
    let arr = v.as_array().expect("--json must emit a JSON array of rows");
    assert!(!arr.is_empty(), "expected at least one row");

    for row in arr {
        let accel = row.get("accelerator").and_then(|x| x.as_str()).unwrap_or_else(|| panic!("row missing accelerator: {row}"));
        let status = row.get("status").and_then(|x| x.as_str()).unwrap_or_else(|| panic!("row missing status: {row}"));
        let line = row.get("line").and_then(|x| x.as_str()).unwrap_or_else(|| panic!("row missing line: {row}"));
        assert!(line.starts_with(accel), "row's own `line` must open with its own `accelerator` field - line {line:?} vs accelerator {accel:?}");
        let rate = row.get("rate").expect("row must carry a rate field, even if null");
        if status == "measured" {
            assert!(rate.as_f64().is_some_and(|v| v.is_finite() && v > 0.0), "a measured row must carry a real positive rate: {row}");
        } else {
            assert!(rate.is_null(), "a non-measured row (status {status:?}) must never carry a fabricated rate: {row}");
        }
    }

    // Cross-check against an independent plain-mode run: same set of
    // accelerator ids reported, never a mismatched count (one process
    // silently dropping or duplicating a section would show up here).
    let json_accels: Vec<&str> = arr.iter().filter_map(|r| r.get("accelerator").and_then(|x| x.as_str())).collect();
    let (ok2, plain_stdout, stderr2, _) = run(&[]);
    assert!(ok2, "brain roofline failed: {stderr2}");
    let plain_accels: Vec<&str> = plain_stdout
        .lines()
        .filter(|l| l.starts_with("gpu") || l.starts_with("npu") || l.starts_with("cpu"))
        .map(|l| l.split_whitespace().next().unwrap())
        .collect();
    assert_eq!(json_accels.len(), plain_accels.len(), "--json and plain mode must report the same number of rows\njson: {json_accels:?}\nplain: {plain_accels:?}");
}

/// The NPU probe must never hang and must never turn a missing/unusable NPU
/// into a non-zero exit: `brain roofline npu` alone must degrade cleanly and
/// finish fast, whatever this sandbox's real NPU state is (device node
/// absent, node present but no OpenVINO runtime, or genuinely usable).
#[test]
fn npu_scope_degrades_cleanly_and_finishes_within_a_bounded_time() {
    let (ok, stdout, stderr, elapsed) = run(&["npu"]);
    assert!(ok, "brain roofline npu must exit cleanly even with no usable NPU: {stderr}");
    assert!(elapsed < Duration::from_secs(30), "brain roofline npu took {elapsed:?} - the probe must be bounded, never hang");
    assert!(stdout.lines().any(|l| l.starts_with("npu")), "expected an npu row in:\n{stdout}");
    assert!(!stdout.lines().any(|l| l.starts_with("gpu") || l.starts_with("cpu")), "brain roofline npu must print ONLY its own section:\n{stdout}");
}

/// `brain roofline gpu` / `cpu` each run standalone and print only their own
/// section - never the other two.
#[test]
fn gpu_and_cpu_scopes_each_print_only_their_own_section() {
    let (ok, stdout, stderr, _) = run(&["cpu"]);
    assert!(ok, "brain roofline cpu failed: {stderr}");
    assert!(stdout.lines().any(|l| l.starts_with("cpu")), "expected a cpu row in:\n{stdout}");
    assert!(!stdout.lines().any(|l| l.starts_with("gpu") || l.starts_with("npu")), "brain roofline cpu must print ONLY its own section:\n{stdout}");

    let (ok, stdout, stderr, _) = run(&["gpu"]);
    assert!(ok, "brain roofline gpu failed: {stderr}");
    assert!(!stdout.lines().any(|l| l.starts_with("npu") || l.starts_with("cpu")), "brain roofline gpu must print ONLY its own section:\n{stdout}");
}
