// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--limit-vram-total` / `--limit-ram-total` at the one allocation
//! chokepoint: `Gpu::{storage,storage_init,buffer,uniform_dynamic}`.
//!
//! Two properties, and the second is the load-bearing one:
//!
//! 1. An allocation that would cross the ceiling is refused CLEANLY - a named
//!    error naming the flag - instead of OOMing the driver or the box.
//! 2. A run under a GENEROUS (non-binding) ceiling is **bit-identical** to a
//!    run with no ceiling at all. A limit that silently changed a computed
//!    number - by throttling, by rounding a size, by taking a different code
//!    path - would be a far worse bug than the OOM it prevents.
//!
//! `memauth::limits()` resolves ONCE per process (a `OnceLock`, same shape and
//! same rationale as `BRAIN_DEVICE`/`ambient_compute_set`), so each ceiling
//! under test needs its own fresh process: this file re-execs itself as a
//! subprocess per case, exactly like `device_grammar.rs`.
//!
//! Everything runs on the CPU backend on purpose. The ceiling lives in the
//! facade, above the backend dispatch, so it behaves identically on every
//! backend - and a test that needs no driver, no card and no
//! `MOE_SKIP_GPU_TESTS` escape hatch is one that actually runs in the fast
//! lane, where a regression gets caught.

use std::process::Command;

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

/// Re-run this test binary in a child process with the given environment,
/// executing only the named `#[ignore]`d helper below. Returns its stdout;
/// panics with the child's full output if it failed.
fn child(helper: &str, env: &[(&str, &str)]) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"]);
    // The parent's own environment must never leak a ceiling into a child that
    // is supposed to have none.
    cmd.env_remove("BRAIN_LIMIT_VRAM_TOTAL");
    cmd.env_remove("BRAIN_LIMIT_RAM_TOTAL");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "child {helper} {env:?} exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", out.status.code());
    stdout
}

/// Pull `MARKER=...` out of a child's stdout. `--nocapture` puts libtest's own
/// progress prefix on the same line as the first thing the test writes, so
/// search for the substring rather than a line prefix.
fn marker(stdout: &str, name: &str) -> String {
    let needle = format!("{name}=");
    stdout
        .lines()
        .find_map(|l| l.split_once(&needle).map(|(_, rest)| rest.trim().to_string()))
        .unwrap_or_else(|| panic!("child never printed {needle}; stdout:\n{stdout}"))
}

/// An allocation past the ceiling is refused, and the refusal names the flag.
#[test]
fn an_allocation_over_the_ceiling_is_refused_cleanly() {
    let out = child("ceiling_denies_an_oversized_allocation", &[("BRAIN_LIMIT_RAM_TOTAL", "64M")]);
    assert_eq!(marker(&out, "SMALL_OK"), "true");
    assert_eq!(marker(&out, "BIG_DENIED"), "true");
    assert!(marker(&out, "MESSAGE").contains("--limit-ram-total"), "the panic must name the flag to raise");
}

/// The regression gate: a generous ceiling must not perturb a single bit of a
/// real computation, nor the number of allocations it takes to get there.
#[test]
fn a_generous_ceiling_is_bit_identical_to_no_ceiling() {
    let free = child("compute_and_print_a_digest", &[]);
    let limited = child("compute_and_print_a_digest", &[("BRAIN_LIMIT_RAM_TOTAL", "64G"), ("BRAIN_LIMIT_VRAM_TOTAL", "64G")]);
    assert_eq!(marker(&free, "DIGEST"), marker(&limited, "DIGEST"), "a non-binding ceiling changed the computed result");
    assert_eq!(marker(&free, "RESULT"), marker(&limited, "RESULT"), "a non-binding ceiling changed the computed result");
    assert_eq!(marker(&free, "CHARGED"), "0", "with no ceiling nothing may be charged at all");
    assert_ne!(marker(&limited, "CHARGED"), "0", "under a ceiling the same allocations must actually be accounted");
}

/// With no ceiling set, no authority is built, nothing is probed and nothing
/// is charged - the path is exactly what it was before this feature existed.
#[test]
fn no_ceiling_set_is_a_true_no_op() {
    let out = child("report_the_unset_state", &[]);
    assert_eq!(marker(&out, "ENFORCING"), "false");
    assert_eq!(marker(&out, "AUTHORITY"), "none");
    assert_eq!(marker(&out, "CHARGED"), "0");
}

// ---- child helpers (run only as subprocesses of the tests above) -----------

#[test]
#[ignore = "child process helper, driven by an_allocation_over_the_ceiling_is_refused_cleanly"]
fn ceiling_denies_an_oversized_allocation() {
    let gpu = gpu_core::Gpu::new_cpu(KERNELS);
    // 1 MiB (262144 f32 words) against a 64 MiB ceiling: fits.
    println!("SMALL_OK={}", gpu.try_storage(256 * 1024).is_ok());
    // 512 MiB: cannot fit the whole ceiling, ever.
    let denied = gpu.try_storage(128 * 1024 * 1024);
    println!("BIG_DENIED={}", denied.is_err());
    // The infallible facade method keeps its existing shape - it panics, the
    // way the backends themselves do on a failed allocation - but with a
    // message an operator can act on instead of a driver abort.
    let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.buffer("big", 512 << 20, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)))
        .err()
        .map(|e| e.downcast_ref::<String>().cloned().unwrap_or_else(|| "<non-string panic>".to_string()))
        .unwrap_or_else(|| "<did not panic>".to_string());
    println!("MESSAGE={}", msg.replace('\n', " "));
}

/// A real multi-kernel computation over many allocations, reduced to one
/// digest so the parent can compare two processes bit-for-bit.
#[test]
#[ignore = "child process helper, driven by a_generous_ceiling_is_bit_identical_to_no_ceiling"]
fn compute_and_print_a_digest() {
    const N: usize = 4096;
    let gpu = gpu_core::Gpu::new_cpu(KERNELS);
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let mut acc = gpu.storage_init("acc", &a);
    for round in 0..16 {
        let b: Vec<f32> = (0..N).map(|i| ((i + round) as f32) * 0.125).collect();
        let rhs = gpu.storage_init("rhs", &b);
        let out = gpu.storage(N as u64);
        let step = gpu.step(0, &[&acc, &rhs, &out], &[N as u32], N as u32);
        gpu.submit(&[], &[step]);
        acc = out;
    }
    let result = gpu.read(&acc, N);
    // Digest the RAW BITS, not a rounded float: a bit-identity claim must be
    // checked on bits.
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for v in &result {
        digest ^= v.to_bits() as u64;
        digest = digest.wrapping_mul(0x100_0000_01b3);
    }
    println!("DIGEST={digest:016x}");
    println!("RESULT={:?}", &result[..4]);
    println!("CHARGED={}", gpu.charged_bytes());
}

#[test]
#[ignore = "child process helper, driven by no_ceiling_set_is_a_true_no_op"]
fn report_the_unset_state() {
    let gpu = gpu_core::Gpu::new_cpu(KERNELS);
    let _b = gpu.storage(1024);
    println!("ENFORCING={}", memauth::enforcing());
    println!("AUTHORITY={}", if memauth::authority().is_some() { "some" } else { "none" });
    println!("CHARGED={}", gpu.charged_bytes());
}
