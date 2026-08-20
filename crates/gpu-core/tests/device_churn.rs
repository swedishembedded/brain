// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How many `Gpu`s can one process build and drop in sequence?
//!
//! The answer used to be "not many, and it fails strangely": a process would
//! build a handful of devices on a card and then report
//! `physical GPU ... not found among 0 wgpu adapter(s)`, while every other
//! process on the machine still saw the card perfectly. The cause was
//! repeated creation and destruction of Vulkan instances - see
//! `backend_wgpu`'s `instance` and `brain_vulkan::context`'s
//! `shared_instance` - and the answer is now "unbounded".
//!
//! These tests keep it that way. The two that matter gate PLACEMENT, not
//! just arithmetic: a device that quietly lands on a software rasteriser
//! still computes the right answer, which is exactly why the older loops
//! here passed while the bug was live.
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

/// The same loop with real allocation and real submits per device, the way
/// `model::train::fit` uses one - a device that did actual work has more
/// driver state to unwind on the way out than one that only compiled a
/// pipeline.
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

/// The invariant the loops above never actually checked: that each rebuilt
/// device is still **the physical card that was asked for**.
///
/// They all call `Gpu::new`/`Gpu::new_on`, which fall back to wgpu's own
/// adapter request when a card cannot be found - and on a machine with a
/// software rasteriser installed (most Mesa systems have lavapipe) that
/// fallback succeeds, so a loop that only asserts "it computed the right
/// answer" passes while silently running on the CPU. What it costs is not
/// only speed: the software adapter reports a fraction of the real card's
/// buffer limits, so the first genuinely large allocation after the switch
/// dies in `create_bind_group` validation instead.
///
/// This is the shape every "fresh device per forward call" model produces
/// (ltxv's DiT opens one per denoise step, then one more for VAE decode), so
/// it gates on placement, not just on arithmetic.
#[test]
fn repeated_device_opens_keep_landing_on_the_physical_card() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let Some(dev) = gpu_core::devices::gpus().first() else {
        eprintln!("no physical GPU present; placement is moot");
        return;
    };
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    for i in 0..24 {
        let gpu = gpu_core::Gpu::new_on(dev, &KERNELS);
        let class = gpu.caps().class;
        let a = gpu.storage(4);
        gpu.write_f32(&a, &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage(4);
        gpu.write_f32(&b, &[1.0, 1.0, 1.0, 1.0]);
        let c = gpu.storage(4);
        gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[4], 4)]);
        gpu.poll_wait();
        assert_eq!(gpu.read(&c, 4), vec![2.0, 3.0, 4.0, 5.0], "device {i} computed wrong");
        assert_ne!(
            class,
            backend_api::DeviceClass::Cpu,
            "device {i} landed on a software adapter instead of {:?} (pci {:?})",
            dev.identity.name,
            dev.identity.pci_bus
        );
        drop(gpu);
    }
}

/// The same invariant on the NATIVE Vulkan backend, which reaches the loader
/// through `ash` rather than through wgpu and so has its own instance
/// lifecycle to get right.
///
/// Worth gating separately: this path has no software rasteriser to fall back
/// to, so the identical root cause surfaces as a hard
/// `not found by the Vulkan ICD` error instead of a silent demotion - a
/// different symptom from the wgpu path's, from one shared mistake.
#[test]
fn repeated_native_vulkan_contexts_keep_finding_the_physical_card() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let Some(dev) = gpu_core::devices::gpus().first() else {
        eprintln!("no physical GPU present; placement is moot");
        return;
    };
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    // The first build is what says whether a usable native Vulkan stack
    // exists at all; only a LATER one failing is the regression.
    let Ok(first) = backend_vulkan::VulkanBackend::try_new_on(&KERNELS, &dev.identity) else {
        eprintln!("no native Vulkan backend available; nothing to churn");
        return;
    };
    drop(first);
    for i in 1..24 {
        match backend_vulkan::VulkanBackend::try_new_on(&KERNELS, &dev.identity) {
            Ok(b) => drop(b),
            Err(e) => panic!(
                "native Vulkan context {i} could no longer reach {:?} (pci {:?}) \
                 though the first one did: {e}",
                dev.identity.name, dev.identity.pci_bus
            ),
        }
    }
}
