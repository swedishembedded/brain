// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-frame performance contract of the Vulkan backend.
//!
//! Inference issues hundreds of dispatches per frame. If building a dispatch
//! touches the GPU queue (a submit + fence wait), the frame serializes into
//! hundreds of host<->GPU round trips and an integrated GPU runs orders of
//! magnitude slower than the same kernels batched - measured for ZipDepth on
//! Intel Arc (MTL), where nearly all of a frame was round trips rather than
//! GPU work. These tests pin the contract
//! that makes batching real:
//!
//!   1. `step()` / `step_sliced()` are HOST-side work only — no queue submits.
//!   2. A steady-state frame loop (build steps -> submit -> read) performs a
//!      BOUNDED number of submits per frame (the flush + the readback), not
//!      O(dispatches).
//!   3. Transient uniform buffers are recycled across flushes — a camera loop
//!      must not grow GPU allocations per frame.
//!
//! All tests skip (pass trivially) when no Vulkan device is present, like
//! `gpu-core`'s `vulkan_dispatch_storage_and_readback`.

use backend_api::Backend;
use backend_vulkan::VulkanBackend;

/// Each of this file's 4 tests calls `backend()` to build its own real
/// Vulkan device directly (below `gpu_core::Gpu`, so `gpu_core::testgpu::dev`
/// does not apply) - under `cargo test`'s default multi-threaded run they
/// can race their own independent device builds against each other. Same
/// hazard, same fix as this crate's `kernel_timing.rs`.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("add2", kernels::ADD2)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

#[test]
fn step_creation_performs_no_queue_submits() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    let base = be.queue_submits();
    let steps: Vec<_> = (0..16).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    assert_eq!(
        be.queue_submits() - base,
        0,
        "building a dispatch must not submit to the GPU queue (uniform writes \
         must go through mapped host-visible memory, not zero/upload commands)"
    );

    // The batch still computes the right thing.
    be.submit(&[], &steps);
    assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn frame_loop_submits_are_bounded_per_frame() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    // Warm frame: first-touch allocations (uniform pool, descriptor pool).
    let steps: Vec<_> = (0..32).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    be.submit(&[], &steps);
    let _ = be.read(&out, 4);

    // Steady-state frame: 32 dispatches must cost O(1) submits (the batched
    // flush + the readback), NOT O(dispatches).
    let base = be.queue_submits();
    let steps: Vec<_> = (0..32).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    be.submit(&[], &steps);
    let r = be.read(&out, 4);
    assert_eq!(r, vec![11.0, 22.0, 33.0, 44.0]);
    let submits = be.queue_submits() - base;
    assert!(
        submits <= 4,
        "a 32-dispatch frame performed {submits} queue submits — the batch is \
         not actually batching (expected <= 4: flush + readback copy)"
    );
}

#[test]
fn transient_uniforms_are_recycled_across_flushes() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    // Three frames of 16 transient-uniform dispatches each. Without recycling
    // the live transient-uniform count grows by 16 per frame (at camera frame
    // rates that is thousands of leaked buffers + descriptor sets per second);
    // with it, the pool peaks
    // at one frame's worth.
    for _ in 0..3 {
        let steps: Vec<_> = (0..16).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
        be.submit(&[], &steps);
        assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    }
    let live = be.transient_uniform_count();
    assert!(
        live <= 16,
        "{live} transient uniforms live after 3 flushed frames of 16 dispatches \
         — transients are not being recycled across flushes"
    );
}

/// `storage()`/`storage_init()` used to hardcode `host_visible: false` on
/// every buffer, regardless of the memory type actually bound — so on a
/// unified-memory device (an integrated GPU with no separate VRAM, where the
/// `DEVICE_LOCAL` heap is *also* `HOST_VISIBLE | HOST_COHERENT`) every
/// `storage_init` paid a staging-buffer + `run_cmd` (a full submit+fence) for
/// memory a direct `memcpy` could have reached. This does not assert the box
/// is unified memory (a discrete-GPU CI runner is a legitimate
/// `host_visible: false` outcome) — it asserts the two are consistent: IF
/// this device reports `unified_memory` (via `DeviceCaps`, queried the same
/// way the rest of the engine does), THEN `storage_init` (a host write) must
/// cost zero queue submits, matching `uniform_dynamic`'s existing zero-submit
/// contract.
///
/// `read` (a host read of GPU-written data) deliberately does NOT get the
/// same claim — `VkContext::download` always stages, even on a host-visible
/// buffer, because a direct-map readback was measured live on this box
/// (Intel Arc MTL / Mesa ANV 25.0.7) to race with the driver's cache
/// write-back: a dispatch's writes were sometimes not yet visible to a host
/// read performed immediately after `vkWaitForFences` returned. See
/// `VkContext::download`'s doc for the full investigation. So `read` costs
/// exactly one submit (the staging copy) here, same as everywhere else.
#[test]
fn storage_buffers_skip_staging_on_a_unified_memory_device() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.caps().unified_memory {
        brain_testutil::skip_unavailable("this device is not unified memory (a real staging path is correct here)");
        return;
    }
    let base = be.queue_submits();
    let buf = be.storage_init("x", &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        be.queue_submits() - base,
        0,
        "storage_init on a unified-memory device cost a queue submit — \
         alloc_raw is not deriving host_visible from the memory type actually bound"
    );
    let got = be.read(&buf, 4);
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        be.queue_submits() - base,
        1,
        "read cost a different number of queue submits than the staging path — \
         download must always stage (see its doc for why a direct-map readback \
         is unsafe on this driver), not vary submits by host_visible"
    );
}
