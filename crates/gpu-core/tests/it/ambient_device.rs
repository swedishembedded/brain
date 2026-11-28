// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a process nothing configured runs on the CPU, whatever its environment says. `BRAIN_DEVICE` and
//! `BRAIN_BACKEND` were once read by every test binary and library caller to pick its hardware, so a shell that
//! exported them (or a node whose cards another job held) sent work to a GPU nobody asked for. A device is now asked
//! for explicitly: by the CLI's `--device`/`--backend`, a tool's own flags (`select_compute`), or a caller's own
//! placement.
//!
//! `ambient_compute_set()` is resolved once per process, so each case runs in a fresh child of this test binary. The
//! harness names a device in `BRAIN_TEST_DEVICE` (`brain_testutil::device`), which this binary takes when it loads.

use std::process::Command;

/// This test binary re-run as a child that prints its ambient compute set and the backend `Gpu::new` would build
/// on, with `env` set (and every GPU hidden from CUDA, so a regression cannot reach a card): `(set, backend)`.
fn ambient_in_child(env: &[(&str, &str)]) -> (String, String) {
    child(env, None)
}

/// [`ambient_in_child`], expecting the child to refuse to start, naming `refused`.
fn refused_in_child(env: &[(&str, &str)], refused: &str) {
    child(env, Some(refused));
}

fn child(env: &[(&str, &str)], refusal: Option<&str>) -> (String, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", "ambient_device::print_ambient", "--ignored", "--nocapture", "--test-threads=1"]);
    cmd.env_remove("BRAIN_DEVICE").env_remove("BRAIN_BACKEND").env_remove("BRAIN_TEST_DEVICE").env("CUDA_VISIBLE_DEVICES", "").env("VK_ICD_FILENAMES", "/dev/null");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if let Some(refused) = refusal {
        assert!(!out.status.success(), "the child ran:\n{stdout}");
        assert!(stderr.contains("BRAIN_TEST_DEVICE") && stderr.contains(refused), "the refusal names the variable and the device:\n{stderr}");
        return (String::new(), String::new());
    }
    assert!(out.status.success(), "the child failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let field = |marker: &str| {
        stdout
            .lines()
            .find_map(|l| l.split_once(marker).map(|(_, rest)| rest.to_string()))
            .unwrap_or_else(|| panic!("no {marker} in:\n{stdout}"))
    };
    (field("AMBIENT_SET:"), field("AMBIENT_BACKEND:"))
}

/// The child's entry point; never part of a normal run.
#[test]
#[ignore = "invoked as a child by the other tests in this file"]
fn print_ambient() {
    println!("AMBIENT_SET:{:?}", gpu_core::ambient_compute_set());
    println!("AMBIENT_BACKEND:{}", gpu_core::backend_name());
}

/// The CPU set as `--device cpu` resolves it, without probing a card.
fn cpu_set() -> String {
    let set = gpu_core::DeviceSpec::parse("cpu").unwrap().resolve(&gpu_core::Inventory::probe_host()).unwrap();
    format!("{set:?}")
}

#[test]
fn an_unconfigured_process_runs_on_the_cpu() {
    assert_eq!(ambient_in_child(&[]), (cpu_set(), "cpu".to_string()));
}

#[test]
fn the_environment_chooses_no_device() {
    for env in [
        &[("BRAIN_DEVICE", "gpu0")][..],
        &[("BRAIN_DEVICE", "gpu"), ("BRAIN_BACKEND", "cuda")],
        &[("BRAIN_DEVICE", "vulkan")],
        &[("BRAIN_BACKEND", "wgpu")],
    ] {
        assert_eq!(ambient_in_child(env), (cpu_set(), "cpu".to_string()), "{env:?} is ignored");
    }
}

/// The harness names the device in `BRAIN_TEST_DEVICE`; the CPU is the CPU, a GPU this machine cannot give (here every
/// card is hidden) stops the binary naming the variable, never runs on the CPU in its place.
#[test]
fn the_harness_names_the_device_and_a_missing_gpu_is_refused() {
    assert_eq!(ambient_in_child(&[("BRAIN_TEST_DEVICE", "cpu")]), (cpu_set(), "cpu".to_string()));
    refused_in_child(&[("BRAIN_TEST_DEVICE", "gpu0:cuda")], "gpu0:cuda");
    refused_in_child(&[("BRAIN_TEST_DEVICE", "vulkan")], "vulkan");
    refused_in_child(&[("BRAIN_TEST_DEVICE", "gpu0:cuda:x")], "gpu0:cuda:x");
}
