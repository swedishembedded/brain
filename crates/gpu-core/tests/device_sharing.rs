// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One wgpu device per process, not one per model object.
//!
//! Building a `Gpu` used to build a whole Vulkan device — instance, adapter,
//! queue and one shader compile per kernel. Since a `Gpu` is constructed per
//! *model object*, a test binary or a serving process ended up with many
//! concurrent devices on one physical card, which:
//!   * deadlocked ~50% of runs when several test threads did it at once
//!     (every thread parked in futex wait), and
//!   * made a second engine cost as much as the first (`brain perf startup`
//!     measured no warm path at all).
//!
//! These tests pin the fix: the device is shared, and concurrent construction
//! from many threads completes.

/// `devices_built()` is a process-global counter, so two tests observing a
/// *delta* must not overlap. Serialise them rather than weakening the assertion.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok() || gpu_core::backend_name() != "wgpu"
}

const K: &[(&str, &str)] = &[("add2", kernels::ADD2)];

/// Many `Gpu`s over the same kernel set must build **one** device between them.
#[test]
fn same_kernel_set_shares_one_device() {
    if skip() {
        return;
    }
    let _x = exclusive();
    let before = gpu_core::devices_built();
    let gpus: Vec<gpu_core::Gpu> = (0..8).map(|_| gpu_core::Gpu::new(K)).collect();
    let built = gpu_core::devices_built() - before;
    assert!(built <= 1, "8 Gpus over one kernel set built {built} devices; expected at most 1");
    drop(gpus);
}

/// Concurrent construction from many threads must complete — this is the case
/// that used to deadlock. A hang fails the test by the harness timeout; the
/// assertion catches the sharing regression that would bring the hang back.
#[test]
fn concurrent_construction_does_not_deadlock() {
    if skip() {
        return;
    }
    let _x = exclusive();
    let before = gpu_core::devices_built();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                let g = gpu_core::Gpu::new(K);
                // Touch the device so this is a real use, not just construction.
                let b = g.storage_init("x", &[1.0f32, 2.0, 3.0, 4.0]);
                g.read(&b, 4)
            })
        })
        .collect();
    for h in handles {
        let v = h.join().expect("a construction thread panicked");
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }
    let built = gpu_core::devices_built() - before;
    assert!(built <= 1, "8 threads built {built} devices; expected at most 1");
}
