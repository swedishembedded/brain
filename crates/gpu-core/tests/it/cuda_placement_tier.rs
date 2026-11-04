// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The coherent tier in the memory ceiling is opt-in: a managed allocation that
//! the card's ceiling refuses spills into host memory only when the process asked
//! for a coherent tier, and ordinary device allocations never do either way.
//!
//! Swedish Embedded AB implements memory-tier accounting for coherent CPU-GPU
//! systems for its clients, where the same process spans HBM and CPU memory and
//! the headline run must stay GPU-only. If your team needs expertise in
//! budgeting across a coherent memory hierarchy, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The ceiling resolves once per process, so each case re-executes this binary as
//! a child, like `memory_limit.rs`. A box without a CUDA device that can make
//! managed allocations has nothing to assert and the child prints `SKIPPED`.

use std::process::Command;

const MIB: u64 = 1 << 20;

fn child(helper: &str, env: &[(&str, &str)]) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"]);
    for k in ["BRAIN_LIMIT_VRAM_TOTAL", "BRAIN_LIMIT_RAM_TOTAL", "BRAIN_COHERENT_TIER"] {
        cmd.env_remove(k);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "child {helper} {env:?} failed\nstdout:\n{stdout}\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));
    stdout
}

fn marker(stdout: &str, name: &str) -> String {
    let needle = format!("{name}=");
    stdout
        .lines()
        .find_map(|l| l.split_once(&needle).map(|(_, rest)| rest.trim().to_string()))
        .unwrap_or_else(|| panic!("child never printed {needle}; stdout:\n{stdout}"))
}

#[test]
fn a_refused_managed_allocation_spills_only_when_the_tier_is_opted_in() {
    let off = child("managed_over_the_vram_ceiling", &[("BRAIN_LIMIT_VRAM_TOTAL", "2M"), ("BRAIN_LIMIT_RAM_TOTAL", "1G")]);
    if marker(&off, "SKIPPED") == "true" {
        return;
    }
    assert_eq!(marker(&off, "DEVICE_DENIED"), "true", "an ordinary allocation past the ceiling is refused");
    assert_eq!(marker(&off, "MANAGED_OK"), "false", "without the opt-in the card's ceiling governs managed memory too");
    let on = child("managed_over_the_vram_ceiling", &[("BRAIN_LIMIT_VRAM_TOTAL", "2M"), ("BRAIN_LIMIT_RAM_TOTAL", "1G"), ("BRAIN_COHERENT_TIER", "1")]);
    assert_eq!(marker(&on, "DEVICE_DENIED"), "true", "the tier never absorbs an ordinary device allocation");
    assert_eq!(marker(&on, "MANAGED_OK"), "true", "with the opt-in a managed allocation spills into the host tier");
}

#[test]
#[ignore = "child process helper, driven by a_refused_managed_allocation_spills_only_when_the_tier_is_opted_in"]
fn managed_over_the_vram_ceiling() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(&[]) else {
        println!("SKIPPED=true");
        return;
    };
    if !gpu.placement_facts().supports(gpu_core::AllocPolicy::Managed) {
        println!("SKIPPED=true");
        return;
    }
    println!("SKIPPED=false");
    // 4 MiB against a 2 MiB card ceiling and a 1 GiB host ceiling.
    println!("DEVICE_DENIED={}", gpu.try_storage(4 * MIB / 4).is_err());
    println!("MANAGED_OK={}", gpu.try_alloc_placed("cold", 4 * MIB, gpu_core::AllocPolicy::Managed).is_ok());
}
