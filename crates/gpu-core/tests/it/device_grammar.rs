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
//! one resolution per process), so each distinct `BRAIN_DEVICE` value under test needs its own
//! fresh process: this file re-execs itself as a subprocess per case rather
//! than asserting in-process, where the first resolution would stick for
//! every later case.

use std::process::Command;

/// Re-run this test binary in a child process with `BRAIN_DEVICE` set (or
/// unset, for `None`), executing only the `print_ambient` helper below, and
/// return its `(compute_set debug, ambient_gpu pin debug, stderr)`.
fn run_ambient(device: Option<&str>) -> (String, String, String, String) {
    let (out, stdout, stderr) = spawn_ambient(device);
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
    let selected = extract("AMBIENT_SELECTED:");
    let set_dbg = extract("AMBIENT_SET_DEBUG:");
    let pin_dbg = extract("AMBIENT_GPU_PIN:");
    (selected, set_dbg, pin_dbg, stderr)
}

/// The `print_ambient` child under `BRAIN_DEVICE=device` (unset for `None`),
/// whatever its exit: `(output, stdout, stderr)`.
fn spawn_ambient(device: Option<&str>) -> (std::process::Output, String, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", "device_grammar::print_ambient", "--ignored", "--nocapture", "--test-threads=1"]);
    // `BRAIN_BACKEND` overrides the backend a device token implies, in the
    // child as in this process. The comparison is of the token alone, so a
    // lane that exports it (the CUDA validation run) must not reach the child.
    cmd.env_remove("BRAIN_BACKEND");
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
    (out, stdout, stderr)
}

/// The subprocess entry point `run_ambient` invokes. Never runs as part of
/// a normal `cargo test` pass (kept `#[ignore]`, selected explicitly via
/// `--exact device_grammar::print_ambient --ignored`).
#[test]
#[ignore = "invoked as a subprocess by the other tests in this file"]
fn print_ambient() {
    // First, before anything resolves the compute set: the card a constructor
    // that never asks for the set (`Gpu::try_new_cuda`) would open.
    println!("AMBIENT_SELECTED:{:?}", gpu_core::devices::selected_device().map(|d| d.index));
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
    let mut tokens: Vec<String> = vec!["".into(), "cpu".into()];
    if inv.cpu_cores > 7 {
        tokens.push("cpu0-7".into());
    }
    // A backend token asks for GPU compute, so it resolves only where a card exists.
    if inv.gpus > 0 {
        tokens.push("vulkan".into());
        tokens.push("wgpu".into());
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
    assert!(tokens.len() > 2, "expected more than the always-present tokens on {inv:?}");

    for tok in &tokens {
        let want = gpu_core::DeviceSpec::parse(tok)
            .unwrap_or_else(|e| panic!("token {tok:?} must parse: {e}"))
            .resolve(&inv)
            .unwrap_or_else(|e| panic!("token {tok:?} must resolve on {inv:?}: {e}"));
        let (_selected, got_dbg, _pin, stderr) = run_ambient(Some(tok));
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
        brain_testutil::skip_unavailable("no GPU present on this machine");
        return;
    }
    let idx = inv.gpus - 1;
    let tok = format!("gpu{idx}");
    let (selected, set_dbg, pin_dbg, _stderr) = run_ambient(Some(&tok));
    let want = gpu_core::DeviceSpec::parse(&tok).unwrap().resolve(&inv).unwrap();
    assert_eq!(set_dbg, format!("{want:?}"));
    assert_eq!(
        pin_dbg,
        format!("Some({idx})"),
        "BRAIN_DEVICE={tok:?} must explicitly pin the ambient GPU to {idx}, got {pin_dbg}"
    );
    // The first selection a process makes, with nothing resolved yet, is that
    // card too: a test binary building its handle with `Gpu::try_new_cuda`
    // under `BRAIN_DEVICE=gpu1` once opened gpu0.
    assert_eq!(selected, format!("Some({idx})"), "BRAIN_DEVICE={tok:?} must select gpu{idx} before anything resolved the set");
}

/// An out-of-range `BRAIN_DEVICE=gpu<N>` stops the program at its first
/// device use, naming the variable and the token - never a run on every
/// device instead, never the unrecognised index read as "the ambient card".
#[test]
fn brain_device_invalid_gpu_index_is_refused_naming_it() {
    let inv = gpu_core::Inventory::probe();
    let tok = format!("gpu{}", inv.gpus + 99);
    let (out, stdout, stderr) = spawn_ambient(Some(&tok));
    assert!(!out.status.success(), "BRAIN_DEVICE={tok} must not resolve:\nstdout:\n{stdout}");
    assert!(!stdout.contains("AMBIENT_SET_DEBUG:"), "no compute set may be resolved for {tok}:\n{stdout}");
    let said = format!("{stdout}{stderr}");
    assert!(said.contains("BRAIN_DEVICE") && said.contains(&tok), "the refusal names the variable and its value:\n{said}");
}

/// Absent `BRAIN_DEVICE` must resolve identically to an absent `--device`:
/// every device on the machine.
#[test]
fn absent_brain_device_matches_absent_device_flag() {
    let inv = gpu_core::Inventory::probe();
    let want = gpu_core::DeviceSpec::parse("").unwrap().resolve(&inv).unwrap();
    let (_selected, set_dbg, _pin, _stderr) = run_ambient(None);
    assert_eq!(set_dbg, format!("{want:?}"));
}
