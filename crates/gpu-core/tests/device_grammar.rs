// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate: the strong `--device` grammar (`DeviceSpec::parse` +
//! `resolve`, what `crates/cli`'s `select_backend` uses for `--device` /
//! `BRAIN_DEVICE`) and `gpu_core::ambient_compute_set()` (what every
//! non-CLI caller - every test binary, every library caller - now goes
//! through for a bare `BRAIN_DEVICE`) must resolve the SAME token to
//! IDENTICAL `ComputeSet`s. Before this gate, `BRAIN_DEVICE` fed to a
//! non-CLI caller went through `resolve_backend_name`'s weak ladder (only
//! bare `"cpu"`/`"vulkan"` understood, everything else - `gpu0`, `npu`,
//! `cpu0-7`, `wgpu` - silently mangled to "just use wgpu, ambient card").
//!
//! `ambient_compute_set()` memoizes for the process lifetime (deliberate -
//! same precedent as `BRAIN_GPU_INDEX`, see `gpu_core::devices`'s module
//! doc), so each distinct `BRAIN_DEVICE` value under test needs its own
//! fresh process: this file re-execs itself as a subprocess per case rather
//! than asserting in-process, where the first resolution would stick for
//! every later case.

use std::process::Command;

/// Re-run this test binary in a child process with `BRAIN_DEVICE` set (or
/// unset, for `None`), executing only the `print_ambient` helper below, and
/// return its `(compute_set debug, ambient_gpu pin debug, stderr)`.
fn run_ambient(device: Option<&str>) -> (String, String, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", "print_ambient", "--ignored", "--nocapture", "--test-threads=1"]);
    match device {
        Some(v) => {
            cmd.env("BRAIN_DEVICE", v);
        }
        None => {
            cmd.env_remove("BRAIN_DEVICE");
        }
    }
    let out = cmd.output().expect("spawn subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success(),
        "subprocess (BRAIN_DEVICE={device:?}) exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
    // `--nocapture` prints libtest's own `test <name> ... ` progress prefix on
    // the SAME line as whatever the test writes first, so the marker may not
    // be at line start - search for the substring, not a line prefix.
    let extract = |marker: &str| -> String {
        stdout
            .lines()
            .find_map(|l| l.split_once(marker).map(|(_, rest)| rest.to_string()))
            .unwrap_or_else(|| panic!("no {marker} line in stdout:\n{stdout}"))
    };
    let set_dbg = extract("AMBIENT_SET_DEBUG:");
    let pin_dbg = extract("AMBIENT_GPU_PIN:");
    (set_dbg, pin_dbg, stderr)
}

/// The subprocess entry point `run_ambient` invokes. Never runs as part of
/// a normal `cargo test` pass (kept `#[ignore]`, selected explicitly via
/// `--exact print_ambient --ignored`).
#[test]
#[ignore = "invoked as a subprocess by the other tests in this file"]
fn print_ambient() {
    let set = gpu_core::ambient_compute_set();
    println!("AMBIENT_SET_DEBUG:{set:?}");
    println!("AMBIENT_GPU_PIN:{:?}", gpu_core::devices::ambient_gpu());
}

/// Table test: every representative `--device` token the strong grammar
/// accepts on THIS machine (adapted to its real GPU/CPU/NPU counts - a
/// synthetic `Inventory` would not exercise `ambient_compute_set()`'s own
/// real `Inventory::probe()` call) resolves identically whether reached via
/// `DeviceSpec::parse(...).resolve(...)` directly (the `--device` path) or
/// via `BRAIN_DEVICE=<token>` with nothing published (the `ambient_compute_set`
/// path).
#[test]
fn device_and_brain_device_agree_on_every_token() {
    let inv = gpu_core::Inventory::probe();
    let mut tokens: Vec<String> = vec!["".into(), "cpu".into(), "vulkan".into(), "wgpu".into()];
    if inv.cpu_cores > 7 {
        tokens.push("cpu0-7".into());
    }
    if inv.gpus > 0 {
        tokens.push("gpu".into());
        tokens.push("gpu0".into());
        if inv.cpu_cores > 3 {
            tokens.push("gpu0,cpu0-3".into());
        }
    }
    if inv.gpus > 1 {
        tokens.push("gpu1".into());
        tokens.push("gpu1,cpu0-3".into());
    }
    if inv.npus > 0 {
        tokens.push("npu".into());
        tokens.push("npu0".into());
    }
    assert!(tokens.len() > 4, "expected more than the always-present tokens on {inv:?}");

    for tok in &tokens {
        let want = gpu_core::DeviceSpec::parse(tok)
            .unwrap_or_else(|e| panic!("token {tok:?} must parse: {e}"))
            .resolve(&inv)
            .unwrap_or_else(|e| panic!("token {tok:?} must resolve on {inv:?}: {e}"));
        let (got_dbg, _pin, stderr) = run_ambient(Some(tok));
        assert_eq!(
            got_dbg,
            format!("{want:?}"),
            "token {tok:?}: --device and BRAIN_DEVICE disagree (stderr: {stderr})"
        );
    }
}

/// `BRAIN_DEVICE=gpu<last>` (nothing published) must pin THAT specific
/// card - not merely select the wgpu backend and leave the ambient GPU
/// selection untouched (which, before this gate, meant "whatever card
/// `selected_device()` defaults to", never validated against the index the
/// user actually asked for).
#[test]
fn brain_device_gpu_index_pins_the_specific_card_not_just_the_backend() {
    let inv = gpu_core::Inventory::probe();
    if inv.gpus == 0 {
        eprintln!("skipping: no GPU present on this machine");
        return;
    }
    let idx = inv.gpus - 1;
    let tok = format!("gpu{idx}");
    let (set_dbg, pin_dbg, _stderr) = run_ambient(Some(&tok));
    let want = gpu_core::DeviceSpec::parse(&tok).unwrap().resolve(&inv).unwrap();
    assert_eq!(set_dbg, format!("{want:?}"));
    assert_eq!(
        pin_dbg,
        format!("Some({idx})"),
        "BRAIN_DEVICE={tok:?} must explicitly pin the ambient GPU to {idx}, got {pin_dbg}"
    );
}

/// An out-of-range `BRAIN_DEVICE=gpu<N>` must fall back to the default
/// "all devices" set - never panic, never silently reinterpret the
/// unrecognised index as "just use wgpu, ambient card".
#[test]
fn brain_device_invalid_gpu_index_falls_back_to_all_devices_without_panicking() {
    let inv = gpu_core::Inventory::probe();
    let bad = inv.gpus + 99;
    let tok = format!("gpu{bad}");
    let (set_dbg, _pin, stderr) = run_ambient(Some(&tok));
    let want_fallback = gpu_core::DeviceSpec::default().resolve(&inv).unwrap();
    assert_eq!(
        set_dbg,
        format!("{want_fallback:?}"),
        "an out-of-range index must fall back to the all-devices default set"
    );
    assert!(
        !stderr.is_empty(),
        "a malformed/unresolvable BRAIN_DEVICE must print a warning to stderr"
    );
    assert_eq!(
        stderr.matches("BRAIN_DEVICE").count(),
        1,
        "expected exactly one warning line mentioning BRAIN_DEVICE, got:\n{stderr}"
    );
}

/// Absent `BRAIN_DEVICE` must resolve identically to an absent `--device`:
/// every device on the machine.
#[test]
fn absent_brain_device_matches_absent_device_flag() {
    let inv = gpu_core::Inventory::probe();
    let want = gpu_core::DeviceSpec::parse("").unwrap().resolve(&inv).unwrap();
    let (set_dbg, _pin, _stderr) = run_ambient(None);
    assert_eq!(set_dbg, format!("{want:?}"));
}
