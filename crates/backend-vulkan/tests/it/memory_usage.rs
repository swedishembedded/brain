// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Vulkan backend accounts for its buffers like the CPU, wgpu and CUDA backends do: `memory_usage` reports the
//! bytes a handle family's buffers hold, the peak since the last reset, and what a dropped buffer still holds until
//! the device is done with it.
//!
//! Swedish Embedded AB implements memory-aware GPU runtimes for its clients, where a backend that cannot say what it
//! holds leaves a planner sizing a model's residency blind. If your team needs expertise in device memory accounting
//! across GPU backends, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The counting logic itself is gated without a device by `backend_api::MemoryTally`'s own tests; these gates check
//! that the Vulkan buffers feed it at the right moments. They skip (pass trivially) when no Vulkan device is present,
//! and run on a software adapter such as lavapipe where one is installed.

use backend_api::Backend;
use backend_vulkan::VulkanBackend;

/// Each test builds its own device below `gpu_core::Gpu`; see `deferred_reclaim`'s identical serialisation.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("axpy", kernels::AXPY)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

fn live(be: &dyn Backend) -> u64 {
    be.memory_usage().expect("the Vulkan backend accounts for its allocations").live
}

#[test]
fn live_bytes_follow_buffer_creation_and_drop() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let base = live(&be);
    let a = Backend::storage(&be, 1024);
    assert_eq!(live(&be), base + 4096);
    let b = Backend::storage_init(&be, "b", &[1.0; 256]);
    assert_eq!(live(&be), base + 4096 + 1024);
    drop(a);
    assert_eq!(live(&be), base + 1024);
    let usage = be.memory_usage().unwrap();
    assert!(usage.reserved >= usage.live, "reserved is live plus what dropped buffers still hold");
    drop(b);
    assert_eq!(live(&be), base);
}

#[test]
fn the_peak_keeps_its_high_water_mark_until_reset() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    be.reset_memory_peak();
    let base = live(&be);
    drop(Backend::storage(&be, 2048));
    assert_eq!(live(&be), base);
    assert_eq!(be.memory_usage().unwrap().peak_live, base + 8192);
    be.reset_memory_peak();
    assert_eq!(be.memory_usage().unwrap().peak_live, base);
}

#[test]
fn a_shared_handle_counts_into_its_parents_tally() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let sibling = be.share().expect("the Vulkan backend shares its device");
    let base = live(&be);
    let buf = sibling.storage(512);
    assert_eq!(live(&be), base + 2048, "the parent sees what its sibling allocated");
    assert_eq!(live(sibling.as_ref()), base + 2048);
    drop(buf);
    assert_eq!(live(&be), base);
}
