// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deferred buffer reclaim must not free memory a live descriptor set still
//! names - the bug behind every "GPU device lost while waiting for a submit to
//! complete" panic this backend produced on real models.
//!
//! `VkOwnedBuffer::drop` does not free: it *buries* the handle, and
//! `VkContext::reclaim_dead` destroys it at a point the device is provably
//! done. The original safety condition was a single counter of dispatches that
//! had reached `submit`. That counter is incremented too late: a descriptor set
//! starts naming a raw `vk::Buffer` when it is **written**, in `record()`,
//! which happens while the step is being *built*. So a caller that built a
//! batch of steps, dropped a scratch buffer, and only then submitted left the
//! counter reading zero with live sets still pointing at the scratch - and the
//! next flush of an empty pending list destroyed it underneath them. The queued
//! dispatches then read freed device memory, which a Tesla P40 reports as
//! `VK_ERROR_DEVICE_LOST`. Vulkan validation named it exactly:
//! `VUID-vkCmdDispatch-None-08114`, "the descriptor ... is using buffer ...
//! that is invalid or has been destroyed".
//!
//! Both tests gate the invariant directly - a buried-but-still-named buffer is
//! observably NOT destroyed (`buried_bytes`), and the dispatch that names it
//! computes the right answer - rather than gating on "the process did not
//! crash", which a use-after-free is only sometimes impolite enough to do.
//! Each also asserts the buffer IS reclaimed once nothing names it, so a
//! future regression cannot be "fixed" by disabling reclaim.
//!
//! Skips (passes trivially) when no Vulkan device is present.

use backend_vulkan::VulkanBackend;

/// Both tests build their own real Vulkan device directly (below
/// `gpu_core::Gpu`, so `gpu_core::testgpu::dev` does not apply) - the
/// concurrent-device-construction hazard `crates/gpu-core/tests/
/// device_sharing.rs` documents. Same `DEVICE_SERIAL` fix as this crate's
/// sibling test files.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

const N: usize = 1024;

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("axpy", kernels::AXPY)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

/// `axpy` is `out[i] += s * inp[i]`, so an `out` cleared to zero and an `inp`
/// of ones must come back as exactly `s` everywhere. Reading freed memory
/// would show up here as garbage even on a run the driver tolerated.
fn assert_axpy_result(got: &[f32], s: f32) {
    assert_eq!(got.len(), N);
    for (i, v) in got.iter().enumerate() {
        assert!((v - s).abs() < 1e-6, "element {i}: expected {s}, got {v} - a dispatch read the wrong memory");
    }
}

#[test]
fn a_buffer_dropped_between_building_a_step_and_submitting_it_is_not_freed() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };

    let out = be.storage(N as u64);
    // The scratch input. Built, named by a step, then dropped BEFORE the step
    // it belongs to is ever submitted - the exact ordering a graph-style
    // caller produces when a helper returns its temporaries.
    let scratch = be.storage_init("scratch", &vec![1.0f32; N]);
    let step = be.step(0, &[&out, &scratch], &[N as u32, backend_api::f(2.5)], N as u32);
    drop(scratch);
    // Burial is what `drop` does here, so this reads the scratch buffer's own
    // size without needing the backend's private `bytes()`.
    let bytes = be.buried_bytes();
    assert!(bytes > 0, "dropping a device buffer must bury it, not free it on the spot");

    // Flush an EMPTY pending list: `step` exists but has not been submitted,
    // so the old dispatch counter read zero here and this call destroyed
    // `scratch`. Any `read`/`write`/`poll_wait` reaches the same code path.
    be.poll_wait();
    assert!(
        be.buried_bytes() >= bytes,
        "the dropped buffer was destroyed while a recorded, unsubmitted dispatch still named it \
         (buried {} B, expected at least {bytes} B) - this is the use-after-free that surfaces as \
         'GPU device lost'",
        be.buried_bytes()
    );

    be.submit(&[], &[step]);
    be.poll_wait();
    assert_axpy_result(&be.read(&out, N), 2.5);

    // ...and reclaim must still actually happen: nothing names the buffer now
    // that the batch has been flushed and its transient set retired.
    assert_eq!(be.buried_bytes(), 0, "nothing names the scratch buffer any more, so it must have been reclaimed");
}

#[test]
fn a_caller_held_step_pins_only_the_buffers_it_names() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };

    // `step_buf` steps are caller-owned and re-submitted across flushes (the
    // `uniform_dynamic` training-loop reuse pattern), so their descriptor set
    // never retires - it must keep pinning its buffers for as long as it lives.
    let ubuf = be.uniform_dynamic(2);
    be.write(&ubuf, &[N as u32, backend_api::f(1.0)]);
    let out = be.storage(N as u64);
    let scratch = be.storage_init("scratch", &vec![1.0f32; N]);
    let step = be.step_buf(0, &ubuf, &[&out, &scratch], N as u32);

    be.submit(&[], &[step.clone()]);
    be.poll_wait();
    assert_axpy_result(&be.read(&out, N), 1.0);

    // An unrelated buffer, dropped at the same moment, is named by nothing -
    // it must still be reclaimed. A held step is not allowed to freeze reclaim
    // device-wide, only to protect its own operands.
    let unrelated = be.storage_init("unrelated", &vec![7.0f32; N]);
    drop(unrelated);
    let unrelated_bytes = be.buried_bytes();
    drop(scratch);
    let scratch_bytes = be.buried_bytes() - unrelated_bytes;
    assert!(scratch_bytes > 0 && unrelated_bytes > 0, "both drops must bury something");

    be.poll_wait();
    let buried = be.buried_bytes();
    assert!(
        buried >= scratch_bytes,
        "the held step's own input was destroyed under it (buried {buried} B, expected at least {scratch_bytes} B)"
    );
    assert_eq!(
        buried, scratch_bytes,
        "an unreferenced buffer stayed buried alongside the pinned one - a held step must pin only \
         what it names, not disable reclaim device-wide"
    );

    // Re-submitting the held step reads `scratch` again: correct only because
    // it was never destroyed. `out` already holds 1.0, so this lands on 2.0.
    be.submit(&[], &[step]);
    be.poll_wait();
    assert_axpy_result(&be.read(&out, N), 2.0);
}
