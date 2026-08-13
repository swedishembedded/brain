// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every NPU-capable subcommand must read the SAME resolved
//! `--device`/`BRAIN_DEVICE` `ComputeSet` gpu_core publishes - there is no
//! process-global sidecar duplicating that resolution, and `brain npu ...`
//! subcommands are not exempt from it.
//!
//! This sandbox has exactly one `/dev/accel/accel*` node (`accel0`), so
//! `Inventory::probe().npus == 1` here - a bare `--device npu`/`npu0`
//! resolves successfully on THIS box: `Inventory::probe` only counts device
//! nodes, it does not check firmware liveness, and (per prior investigation
//! of this specific sandbox) OpenVINO silently retargets to its GPU plugin
//! at compile time rather than erroring at resolve time when the firmware
//! is not actually functional. So "no NPU present" cannot be exercised via
//! a clean absence here; `npu5` (out of range regardless of how many accel
//! nodes exist) is used instead to reliably hit `DeviceSpec::resolve`'s NPU
//! error path on any machine.

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

/// `--device npu<N>` for an out-of-range NPU index must produce a clear,
/// specific diagnostic and a clean non-zero exit - never a panic (which would
/// surface as a signal termination, `status.code() == None` on Unix) and
/// never a silent fallback.
#[test]
fn device_npu_out_of_range_index_reports_clear_diagnostic_and_exits_nonzero() {
    let output = Command::new(bin())
        .args(["devices", "--device", "npu5"])
        .output()
        .expect("run brain devices --device npu5");

    assert!(
        !output.status.success(),
        "expected a non-zero exit for an out-of-range NPU index; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.code().is_some(),
        "must exit cleanly (std::process::exit), not be killed by a signal (a panic/abort): {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("npu5 requested but this machine has"),
        "stderr should name the specific out-of-range NPU request, got: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "must not panic on a bad --device value: {stderr}"
    );
}

/// `brain npu ...` used to bypass `select_backend` entirely (the
/// `argv.get(1) == Some("npu")` special case A4 deleted). Confirm it still
/// flows through the exact same `--device` resolution as any other
/// subcommand by checking the identical out-of-range diagnostic fires there
/// too - not a different error shape, not a silent pass-through.
///
/// `brain npu ...`'s OWN `--device` flag is a different, deprecated-alias
/// grammar (the OpenVINO target device, translated to `--ov-device`), so this
/// uses `BRAIN_DEVICE` - untouched by that translation - to reach brain's own
/// compute-device grammar for this subcommand specifically.
#[test]
fn brain_npu_subcommand_flows_through_the_same_device_resolution() {
    let output = Command::new(bin())
        .args(["npu", "check"])
        .env("BRAIN_DEVICE", "npu5")
        .output()
        .expect("run BRAIN_DEVICE=npu5 brain npu check");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("npu5 requested but this machine has"),
        "brain npu ... should hit the SAME DeviceSpec::resolve diagnostic as \
         every other subcommand, got stderr: {stderr}"
    );
    assert!(
        output.status.code().is_some(),
        "must not be killed by a signal (a panic/abort): {:?}",
        output.status
    );
}

/// The `NPU_REQUESTED` global/reader this phase deleted must never reappear
/// anywhere under `crates/cli/src/` (or, as a broader belt-and-suspenders
/// check, the whole workspace) - that is the difference between "migrated
/// every call site" and "migrated every call site we happened to remember,
/// left a dead sidecar nobody reads". A textual grep, not a behavioural test,
/// is deliberately used here: the property under test IS the absence of the
/// symbol, not any runtime behaviour.
#[test]
fn npu_requested_sidecar_has_zero_occurrences() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let mut offenders = Vec::new();
    for entry in walk(std::path::Path::new(root)) {
        // Skip build output / vendored dirs - only source is in scope.
        let s = entry.to_string_lossy();
        // `/target/`, `/.git/` - build output / VCS internals, never source.
        // `/.claude/worktrees/` - sibling agents' independent git worktrees
        // nested under this repo's `.claude/` dir; each is its own checkout
        // with its own copy of these files, out of scope for THIS phase's
        // migration (and actively being edited by other agents concurrently).
        if s.contains("/target/") || s.contains("/.git/") || s.contains("/.claude/worktrees/") {
            continue;
        }
        // This file itself necessarily names the banned symbols (in this
        // very doc comment and in the literal strings the scan below greps
        // for) - it is the gate, not a call site.
        if entry.ends_with("tests/device_routing.rs") {
            continue;
        }
        if entry.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&entry) {
            if text.contains("NPU_REQUESTED") || text.contains("npu_requested") {
                offenders.push(entry);
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "NPU_REQUESTED/npu_requested must have zero occurrences after phase C3, found in: {offenders:?}"
    );
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}
