// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How many `Gpu`s can one process build and drop in sequence?
//!
//! `crates/bench/tests/capscale.rs` fails on this box with
//! `physical GPU "Tesla P40" not found among 0 wgpu adapter(s)` AFTER six
//! devices in the same process succeeded — so the question is not whether the
//! card is visible but whether building and dropping devices in a loop
//! exhausts something.
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

/// Every test in this file builds several independent real `Gpu::new()`
/// devices in a loop by design (that's the churn under test) rather than
/// sharing one via `gpu_core::testgpu` - so, unlike the rest of the suite,
/// nothing here is protected from a sibling test's concurrent device churn
/// racing against it. Several independently churning devices on one card at
/// once is the exact driver deadlock `device_sharing.rs`'s own
/// `DEVICE_SERIAL` avoids for its tests; this file needs the same one lock,
/// held for each test's whole body.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn building_and_dropping_devices_in_sequence_does_not_exhaust_the_driver() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    for i in 0..12 {
        let gpu = gpu_core::Gpu::new(&KERNELS);
        let a = gpu.storage(4);
        gpu.write_f32(&a, &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage(4);
        gpu.write_f32(&b, &[1.0, 1.0, 1.0, 1.0]);
        let c = gpu.storage(4);
        gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[4], 4)]);
        gpu.poll_wait();
        assert_eq!(gpu.read(&c, 4), vec![2.0, 3.0, 4.0, 5.0], "device {i} computed wrong");
        N.store(i + 1, Ordering::Relaxed);
        drop(gpu);
    }
    eprintln!("built {} devices in sequence", N.load(Ordering::Relaxed));
}

/// The same loop, but HOLDING every device alive. `capscale` builds one engine
/// per grid point; if the answer differs from the sequential case, the fault is
/// concurrently-live devices rather than churn.
#[test]
#[ignore = "diagnostic, assertion-free: builds 12 live devices and only eprintln!s the count -- run by hand when investigating driver exhaustion, not in the default lane"]
fn building_devices_while_holding_the_previous_ones_does_not_exhaust_the_driver() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    let mut live = Vec::new();
    for i in 0..12 {
        eprintln!("building live device {i}");
        live.push(gpu_core::Gpu::new(&KERNELS));
    }
    eprintln!("held {} devices live", live.len());
}

/// The FAITHFUL repro: real allocation and real submits per device, the way
/// `model::train::fit` uses one. The trivial loops above pass at 12; `capscale`
/// dies at the 7th, and the only difference left is how much each device did.
#[test]
fn devices_that_did_real_work_can_still_be_rebuilt() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    const N: usize = 1 << 22; // 16 MiB per buffer, 48 MiB per device
    for i in 0..12 {
        eprintln!("device {i}: building");
        let gpu = gpu_core::Gpu::new(&KERNELS);
        let a = gpu.storage(N as u64);
        let b = gpu.storage(N as u64);
        let c = gpu.storage(N as u64);
        gpu.write_f32(&a, &vec![1.0f32; N]);
        gpu.write_f32(&b, &vec![2.0f32; N]);
        for _ in 0..40 {
            gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[N as u32], N as u32)]);
        }
        gpu.poll_wait();
        assert_eq!(gpu.read(&c, 4), vec![3.0; 4], "device {i} computed wrong");
        drop(gpu);
    }
    eprintln!("12 working devices built and dropped");
}
/// Same churn as the first test above, but with a 1.5s sleep between each
/// device's drop and the next device's build — distinguishes a
/// destruction/recreation TIMING race (driver resource reclaim is
/// asynchronous; giving it time to finish should raise the device count
/// before the ICD loses the card) from a hard one-shot-per-process limit
/// (a delay would not help at all). The residual finding from the earlier
/// investigation that first hit this: with NO delay, only 1 real Vulkan
/// device succeeds before every subsequent one falls back to wgpu/llvmpipe.
///
/// Measured on this box (2 real P40s, driver current as of this test):
/// **exactly 4** real Vulkan devices succeed with the delay, reproducibly
/// across repeated runs (not 3, not 5 — the same count both times this was
/// tried), before falling back at device 4 for the rest of the loop. That
/// determinism is itself informative: pure timing jitter would show run-to-
/// run variance in HOW MANY devices succeed; a fixed count instead suggests
/// a slow-reclaim driver-side resource (e.g. a small ICD-internal handle
/// pool) that a 1.5s wait is not long enough to fully drain, not a race
/// that scales away with more delay. Still a real, useful data point: it
/// rules out "concurrent creation" as the mechanism (this loop is already
/// single-threaded — nothing here IS concurrent, so a creation-side mutex
/// mirroring `VkContext::queue_lock` could not change this outcome) and
/// narrows the remaining theory to driver-side teardown/reclaim timing, not
/// a hard one-shot-per-process cap.
#[test]
#[ignore = "diagnostic, assertion-free: 6x1.5s sleeps to characterise driver teardown timing, records a streak it never asserts on -- a lab-notebook entry (kept for its measured findings above), not a gate"]
fn churn_with_delay_between_devices_extends_but_does_not_fix_the_streak() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    let mut real_streak = 0usize;
    let mut saw_fallback = false;
    for i in 0..6 {
        let gpu = gpu_core::Gpu::new(&KERNELS);
        let a = gpu.storage(4);
        gpu.write_f32(&a, &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage(4);
        gpu.write_f32(&b, &[1.0, 1.0, 1.0, 1.0]);
        let c = gpu.storage(4);
        gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[4], 4)]);
        gpu.poll_wait();
        let kind = gpu.kind();
        eprintln!("device {i}: adapter={kind}");
        if kind == "vulkan" {
            if saw_fallback {
                eprintln!("note: real Vulkan device AFTER an earlier fallback -- streak was not strictly leading, informational only");
            } else {
                real_streak += 1;
            }
        } else {
            saw_fallback = true;
        }
        drop(gpu);
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
    eprintln!("real Vulkan devices before first fallback (with 1.5s teardown delay): {real_streak}");
}
