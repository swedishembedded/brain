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

/// `flush_chunk` used to insert a blanket `VkMemoryBarrier` before every
/// dispatch but the first in a batch, regardless of whether it touched any
/// buffer an earlier dispatch in the batch had written - an `n`-dispatch
/// batch always paid `n-1` barriers. It now runs a per-buffer read/write-set
/// hazard analysis and emits a `VkBufferMemoryBarrier` only for a buffer a
/// later dispatch actually depends on.
///
/// This batch is 3 dispatches: `out1 = a + b`, then `out2 = out1 + c` (a real
/// dependency - reads `out1`), then `out3 = d + e` (fully independent - shares
/// no buffer with either of the first two). The blanket barrier would cost 2;
/// the hazard analysis must cost exactly 1 (before the `out1` read), with
/// zero inserted before the independent third dispatch.
#[test]
fn independent_dispatches_in_a_batch_cost_no_barrier() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let c = be.storage_init("c", &[100.0, 200.0, 300.0, 400.0]);
    let d = be.storage_init("d", &[1000.0, 2000.0, 3000.0, 4000.0]);
    let e = be.storage_init("e", &[1.0, 1.0, 1.0, 1.0]);
    let out1 = be.storage(4);
    let out2 = be.storage(4);
    let out3 = be.storage(4);

    let base = be.barrier_count();
    let steps = vec![
        be.step(0, &[&a, &b, &out1], &[4], 4),   // out1 = a + b
        be.step(0, &[&out1, &c, &out2], &[4], 4), // out2 = out1 + c  (depends on out1)
        be.step(0, &[&d, &e, &out3], &[4], 4),   // out3 = d + e     (independent)
    ];
    be.submit(&[], &steps);
    assert_eq!(be.read(&out2, 4), vec![111.0, 222.0, 333.0, 444.0]);
    assert_eq!(be.read(&out3, 4), vec![1001.0, 2001.0, 3001.0, 4001.0]);

    let barriers = be.barrier_count() - base;
    assert_eq!(
        barriers, 1,
        "expected exactly 1 buffer barrier (the real out1 dependency) out of \
         3 dispatches - got {barriers}; either a real dependency was missed \
         (correctness) or an independent dispatch was barriered anyway \
         (not minimal)"
    );
}

/// `record_dispatches`'s hazard analysis used to mark a buffer `dirty` only
/// when a dispatch WROTE it, and insert a barrier only when a LATER dispatch
/// touched a `dirty` buffer - catching read-after-write and write-after-
/// write, but NOT write-AFTER-read: a dispatch that only READS buffer `x`
/// never marked it dirty, so a LATER dispatch overwriting `x` got no barrier
/// at all - the write could start before the earlier read had actually
/// consumed the old value (M6.7).
///
/// This batch is 2 dispatches: `out1 = x + b` (reads `x`, does not write it),
/// then `x = c + d` (overwrites `x` with unrelated values - a real
/// dependency: dispatch 2 must not start until dispatch 1 has read `x`'s OLD
/// value). Root-caused via a real Qwen3.5 GQA-layer forward on this
/// workspace's Intel Arc (Meteor Lake) iGPU, where exactly this class of gap
/// crashed the device outright (`ERROR_DEVICE_LOST`) rather than merely
/// computing a wrong answer - `BRAIN_VK_SERIAL=1` (forcing a barrier before
/// every dispatch, unconditionally) made the crash go away, which is what
/// pointed at the hazard analysis rather than a driver bug unrelated to it.
/// Before the fix this test's own `barrier_count()` assertion caught the gap
/// directly (0 barriers for a real dependency); the tiny 4-element scale
/// here did not reliably reproduce a wrong VALUE the way the real GQA
/// traffic reproduced a crash, which is exactly why the barrier-count
/// assertion is the load-bearing one, not the value check.
#[test]
fn a_write_after_read_hazard_gets_a_barrier() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let x = be.storage_init("x", &[1.0, 1.0, 1.0, 1.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let c = be.storage_init("c", &[100.0, 200.0, 300.0, 400.0]);
    let d = be.storage_init("d", &[1.0, 1.0, 1.0, 1.0]);
    let out1 = be.storage(4);

    let base = be.barrier_count();
    let steps = vec![
        be.step(0, &[&x, &b, &out1], &[4], 4), // out1 = x + b   (reads x)
        be.step(0, &[&c, &d, &x], &[4], 4),    // x = c + d      (overwrites x - WAR on dispatch 1's read)
    ];
    be.submit(&[], &steps);
    let got = be.read(&out1, 4);

    let barriers = be.barrier_count() - base;
    assert_eq!(
        barriers, 1,
        "expected exactly 1 buffer barrier (the write-after-read dependency on x) - got \
         {barriers}; a regression here means dispatch 2's write can race dispatch 1's read again, \
         the same class of gap that crashed the native Vulkan backend outright on real GQA-layer \
         traffic (ERROR_DEVICE_LOST)"
    );
    assert_eq!(
        got,
        vec![11.0, 21.0, 31.0, 41.0],
        "out1 = x + b computed the wrong answer ({got:?}, expected [11, 21, 31, 41] from x's \
         value BEFORE dispatch 2 overwrote it)"
    );
}

/// `submit` used to accumulate an UNBOUNDED batch: `step`/`submit` never
/// touch the queue by design (M6.2 - see this file's own header), so a
/// caller that never calls `read`/`poll_wait`/`Backend::flush` between
/// `submit`s (a legitimate pattern this file's own `frame_loop_submits_are_
/// bounded_per_frame` explicitly wants for small kernels) could grow ONE
/// un-synchronised command buffer without limit. Harmless for the catalogue's
/// normal (fast, `@opt` 3-5) kernels even at hundreds of them - but a real
/// Qwen3.5 GQA-layer forward (`qwen35_bench gqa 128 3`, `BRAIN_DEVICE=
/// vulkan`) crashed the device outright (`ERROR_DEVICE_LOST`) this way: its
/// own benchmark harness accumulates several real-shape `matmul.wgsl`
/// dispatches (`@opt 2`, unblocked, ~3.3s EACH at this model's real
/// `d_model=5120` q_proj shape on this workspace's Intel Arc iGPU) with zero
/// host sync between them, and the resulting single, continuous, multi-
/// dispatch GPU submission ran long enough to trip this driver's own hang-
/// detection reset - independent of `record_dispatches`'s buffer-hazard
/// correctness (confirmed unrelated: the identical crash reproduces from a
/// sequence of `matmul` dispatches alone, no GQA attention pattern involved,
/// and `barrier_count()` for the real GQA sequence already matches a full
/// hand trace with zero gaps - see M6.9's ledger entry for the full
/// isolation). Root cause is un-bounded ACCUMULATED dispatch size, not a
/// missing barrier, so the fix is a size-based (not count-based - see
/// `MAX_UNSYNCED_WORKGROUPS`'s own doc for why COUNT does not work here
/// without breaking the bounded-submits contract this same file already
/// tests) ceiling on how much can accumulate in `pending` before `submit`
/// forces a real flush + host wait on its own.
///
/// This test cannot use an actually-slow kernel (multi-second, real-driver
/// dependent, thermal-state dependent - exactly the fragility this repo's
/// own measurement discipline warns against in a routine test) - it proves
/// the MECHANISM instead: one `add2` dispatch (bounds-checked against
/// `Params.total`, so a huge thread count over a 4-element buffer is safe,
/// not a buffer overrun) sized far larger than the ceiling could plausibly
/// be tuned to, but computationally trivial (O(1)/thread) so it completes
/// fast regardless. If `submit` no longer bounds accumulation, this single
/// call would cost zero queue submits until the next explicit sync (matching
/// `step_creation_performs_no_queue_submits`'s own contract for a SMALL
/// batch) - the regression signal is a submit happening with NO `read`/
/// `poll_wait` call anywhere in this test.
#[test]
fn an_oversized_pending_batch_is_auto_flushed_without_an_explicit_sync() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    // Warm-up: first-touch pipeline/descriptor-pool cost, not part of what
    // this test measures.
    be.submit(&[], &[be.step(0, &[&a, &b, &out], &[4], 4)]);
    let _ = be.read(&out, 4);

    // 2,097,152 threads = 32768 workgroups at this kernel's `@workgroup_size
    // (64)` - 8x this crate's own `MAX_UNSYNCED_WORKGROUPS` ceiling (4096),
    // so this stays a clear regression signal even if that ceiling is later
    // retuned. `Params.total = 4` means all but the first 4 threads return
    // immediately (`add2.wgsl`'s own bounds check) - fast on any device.
    let huge = 4_096u32 * 64 * 8;
    let step = be.step(0, &[&a, &b, &out], &[4], huge);

    let base = be.queue_submits();
    be.submit(&[], &[step]);
    // Deliberately NO `read`/`poll_wait`/explicit `flush` call here.
    let submits = be.queue_submits() - base;
    assert_eq!(
        submits, 1,
        "a single dispatch far past the un-synchronised-accumulation ceiling did not trigger \
         exactly one automatic flush with no explicit sync call ({submits} submits) - an \
         unbounded batch of expensive dispatches can once again run long enough, uninterrupted, \
         to trip this driver's own hang-detection reset (M6.9)"
    );
    assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0], "the auto-flushed dispatch must still compute the right answer");
}
