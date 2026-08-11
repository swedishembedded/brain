// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Explicit device sharing: one device, many handles.
//!
//! Building a `Gpu` builds a whole device — instance, adapter, queue and one
//! shader compile per kernel. That costs seconds, and running *many* concurrent
//! devices on one physical card is hostile to the driver: it deadlocked the test
//! suite roughly half the time (every thread parked in futex wait) and gave
//! `brain perf startup` no warm path at all.
//!
//! The fix is explicit, not a hidden cache: [`gpu_core::Gpu::share`] hands out a
//! second handle onto the same device (same queue and compiled pipelines, its
//! own command stream). These tests pin that a shared handle really is the same
//! device doing correct work, including from many threads at once.
//!
//! Runs on both `wgpu` and `vulkan`: `VulkanBackend::share`/`new_like` used to
//! silently fall through to the `Backend` trait's `None` default (no Vulkan
//! implementation existed), so `Gpu::share` on this backend built a WHOLE NEW
//! device instead of truly sharing one — the exact "many concurrent devices on
//! one card" shape this file exists to rule out, just never exercised here.
//! See `.agents/rules/lessons.md`'s Vulkan-device-sharing entry.

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok() || !matches!(gpu_core::backend_name(), "wgpu" | "vulkan")
}

const K: &[(&str, &str)] = &[("add2", kernels::ADD2)];

/// A shared handle computes correctly and independently of its parent: each
/// handle has its own command stream, so interleaved use must not corrupt
/// either's batches.
/// These tests build REAL devices on purpose — device lifecycle/sharding is
/// the thing under test, so the pooled test device would defeat them. They
/// must therefore not run concurrently with EACH OTHER: several fresh devices
/// on one card is the exact driver deadlock the rest of the suite avoids via
/// gpu_core::testgpu. One lock, held for each test's whole body.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn shared_handle_computes_independently() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let parent = gpu_core::Gpu::new(K);
    let child = parent.share();

    let pa = parent.storage_init("a", &[1.0f32, 2.0, 3.0, 4.0]);
    let pb = parent.storage_init("b", &[10.0f32, 20.0, 30.0, 40.0]);
    let po = parent.storage(4);
    let ca = child.storage_init("a", &[5.0f32, 6.0, 7.0, 8.0]);
    let cb = child.storage_init("b", &[1.0f32, 1.0, 1.0, 1.0]);
    let co = child.storage(4);

    // Interleave: record on both, then read both.
    let ps = parent.step(0, &[&pa, &pb, &po], &[4], 4);
    let cs = child.step(0, &[&ca, &cb, &co], &[4], 4);
    parent.submit(&[], &[ps]);
    child.submit(&[], &[cs]);

    assert_eq!(parent.read(&po, 4), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(child.read(&co, 4), vec![6.0, 7.0, 8.0, 9.0]);
}

/// Buffers created on one handle are usable from another — they live on the
/// same device. This is what lets a serving process share weights between
/// handles instead of uploading them once per model object.
#[test]
fn buffers_are_usable_across_handles() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let parent = gpu_core::Gpu::new(K);
    let child = parent.share();
    let a = parent.storage_init("a", &[1.0f32, 1.0]);
    let b = parent.storage_init("b", &[2.0f32, 3.0]);
    let out = child.storage(2);
    let s = child.step(0, &[&a, &b, &out], &[2], 2);
    child.submit(&[], &[s]);
    assert_eq!(child.read(&out, 2), vec![3.0, 4.0]);
}

/// Many threads hammering shared handles must complete — the shape that used to
/// deadlock when each thread owned its own device. A hang here fails via the
/// suite timeout; the assertions catch wrong results.
#[test]
fn concurrent_shared_handles_do_not_deadlock() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let parent = std::sync::Arc::new(gpu_core::Gpu::new(K));
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let p = parent.clone();
            std::thread::spawn(move || {
                let g = p.share();
                let x = i as f32;
                let a = g.storage_init("a", &[x, x + 1.0]);
                let b = g.storage_init("b", &[1.0f32, 1.0]);
                let o = g.storage(2);
                let s = g.step(0, &[&a, &b, &o], &[2], 2);
                g.submit(&[], &[s]);
                g.read(&o, 2)
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let v = h.join().expect("worker panicked");
        assert_eq!(v, vec![i as f32 + 1.0, i as f32 + 2.0]);
    }
}
