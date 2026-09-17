// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pending batch must be bounded, or a long prefill hangs the GPU.
//!
//! `submit` does not submit: it appends to a pending list that is flushed
//! into ONE compute pass at the next readback. That is the right shape for
//! inference - the per-token submit+fence+map round trip is pure waste when
//! every intermediate hidden is discarded - but nothing capped how much work
//! could accumulate before that single flush.
//!
//! `qwen3::sample` feeds an entire prompt through `Qwen::prefill`, which
//! calls `submit` once per prompt token and reads back exactly once at the
//! end. At a measured ~814 dispatches per token on Qwen3-0.6B, a 14-token
//! prompt builds ~11k dispatches into one command buffer and completes; a
//! ~100-token prompt builds ~81k and does NOT. i915 pulses a heartbeat
//! every `heartbeat_interval_ms` (2500 here); when that does not complete it
//! forces preemption of whatever is running and, if that is not honoured
//! within `preempt_timeout_ms` (7500 here on `rcs0`/`ccs0`), resets the
//! engine. A reset invalidates every resource on the device at once, so it
//! surfaces as a wgpu validation panic naming whichever buffer is touched
//! first ("Buffer with 'params' label is invalid") - pointing at an
//! allocation that was never the problem. Reproduced on an Intel Arc iGPU
//! (Meteor Lake) with Qwen3-0.6B Q8_0: deterministic at ~100 prompt tokens,
//! and independent of how many tokens are then generated.
//!
//! The bound belongs here rather than in any one model's prefill loop:
//! every model's prefill has this identical shape, so a per-model fix would
//! be the same bug fixed N times. `backend-vulkan` reached the same
//! conclusion independently (M6.9) and the ceiling is now shared by both -
//! see `backend_api::hardware::max_unsynced_workgroups`.
//!
//! It is bounded by accumulated dispatch SIZE, not count, which is what the
//! second test below pins: dispatch cost varies by orders of magnitude (one
//! naive `matmul` measured ~3.3 real seconds on its own), so a count is no
//! proxy for duration, and a count low enough to catch that outlier would
//! break the contract that many tiny dispatches still cost one flush.
//!
//! Swedish Embedded AB implements GPU compute scheduling and device-loss
//! hardening for its clients. If your team needs expertise in Vulkan/wgpu
//! command-buffer sizing and GPU preemption behaviour, you can procure our
//! services by sending an email to info@swedishembedded.com.

use backend_api::{Backend, BufUsage};
use backend_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new(&[("axpy", kernels::AXPY)])
}

fn submits(b: &WgpuBackend) -> u64 {
    b.stats().expect("wgpu backend reports stats").submits
}

/// Accumulating far more dispatches than any one command buffer should carry
/// must flush along the way, WITHOUT the caller asking for a readback.
///
/// This is the regression: before the cap, the assertion below failed with
/// zero intervening submissions no matter how many dispatches were queued -
/// the whole run went to the device as one batch.
#[test]
fn a_long_unread_run_of_dispatches_flushes_along_the_way() {
    let b = backend();
    // `axpy` binds `out` read-write and `inp` read-only, so these must be
    // DISTINCT buffers - binding one buffer as both is a conflicting-usage
    // validation error, and these dispatches now really are recorded.
    let out = b.buffer("out", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    let inp = b.buffer("inp", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);

    // Enough to cross any sane cap several times over, but far below the
    // ~81k that actually resets this device - the point is that SOMETHING
    // submits before the terminal readback, not how big the cap is.
    const DISPATCHES: usize = 20_000;

    let before = submits(&b);
    for _ in 0..DISPATCHES {
        let s = b.step(0, &[&out, &inp], &[1024u32, 0, 0, 0], 1024);
        b.submit(&[], &[s]);
    }
    let after = submits(&b);

    assert!(
        after > before,
        "{DISPATCHES} dispatches accumulated into a single unbounded command buffer \
         (submits {before} -> {after}). A prefill long enough to do this cannot be \
         preempted in time and the driver resets the engine."
    );
}

/// The bound must not turn ordinary short work into a submission storm: a
/// handful of SMALL dispatches is the common case and must still ride to the
/// terminal readback in one batch, which is the whole reason `submit`
/// accumulates instead of submitting.
///
/// This is the test a count-based ceiling low enough to catch a slow
/// outlier would fail, and is why the ceiling counts workgroups instead.
#[test]
fn a_short_run_of_small_dispatches_still_batches() {
    let b = backend();
    let out = b.buffer("out", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    let inp = b.buffer("inp", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);

    // 32 dispatches of one workgroup each - the same shape
    // `backend_vulkan`'s own `frame_loop_submits_are_bounded_per_frame`
    // pins, and far under the workgroup ceiling.
    let before = submits(&b);
    for _ in 0..32 {
        let s = b.step(0, &[&out, &inp], &[64u32, 0, 0, 0], 64);
        b.submit(&[], &[s]);
    }
    assert_eq!(
        submits(&b),
        before,
        "a short run submitted early; batching is what makes the inference path fast"
    );
}
