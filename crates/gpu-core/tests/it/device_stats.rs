// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every backend counts device ops. This is a *contract* test, not a perf one.
//!
//! `Backend::stats()` defaults to `None` — "this backend does not count" — and
//! consumers are required to report null for that, never zero. Two of the three
//! backends took the default (`backend-cpu`, `backend-vulkan`), so on those the
//! answer to "how many submits did this cost" was unavailable, and the one
//! in-tree consumer papered over it with `.unwrap_or(0)`. That turned "not
//! counted" into "zero" and made an engine test
//! (`qwen3::serve::tests::prefill_submits_scale_with_chunks_not_with_token_count`)
//! pass **vacuously** on the CPU backend: its first two assertions compared
//! 0 == 0 and only the third noticed.
//!
//! The fix was to make the counters real everywhere rather than to teach each
//! caller to cope. This test is what stops the default creeping back in.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| v != "0").unwrap_or(false)
}

#[test]
fn the_backend_counts_submits_dispatches_and_readbacks() {
    if skip_gpu() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);

    let before = gpu.stats().unwrap_or_else(|| {
        panic!(
            "backend `{}` reports no DeviceStats — a consumer then has to \
             distinguish 'not counted' from 'zero', which is exactly the \
             ambiguity that made an engine test pass vacuously",
            gpu.kind()
        )
    });

    let n = 1024u64;
    let out = gpu.storage(n);
    let inp = gpu.storage(n);
    gpu.write_f32(&inp, &vec![1.0f32; n as usize]);

    let steps: Vec<_> = (0..3)
        .map(|_| gpu.step(0, &[&out, &inp], &[n as u32, backend_api::f(1.0)], n as u32))
        .collect();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let got = gpu.read(&out, n as usize);

    let after = gpu.stats().expect("stats were available a moment ago");

    // The arithmetic is incidental — but if it is wrong the counters are
    // counting the wrong thing, so assert it: three axpy passes of 1.0.
    assert_eq!(got[0], 3.0, "three axpy passes should sum to 3.0");

    assert!(
        after.submits > before.submits,
        "{}: submits did not move ({} -> {})",
        gpu.kind(),
        before.submits,
        after.submits
    );
    assert!(
        after.dispatches >= before.dispatches + 3,
        "{}: 3 dispatches were submitted, counter moved {} -> {}",
        gpu.kind(),
        before.dispatches,
        after.dispatches
    );
    assert!(
        after.readbacks > before.readbacks,
        "{}: one read() happened, counter moved {} -> {}",
        gpu.kind(),
        before.readbacks,
        after.readbacks
    );
}

/// A `Gpu` that has done nothing still answers — `Some(0)` is a measurement,
/// `None` is the absence of one, and the two must not be confused.
#[test]
fn a_fresh_handle_reports_zero_rather_than_nothing() {
    if skip_gpu() {
        return;
    }
    let gpu = Gpu::new(KERNELS);
    let s = gpu.stats().expect("every backend must report DeviceStats");
    // Constructing a device may itself submit (pipeline warm-up differs per
    // backend), so the assertion is that the value EXISTS and is sane, not that
    // it is zero.
    assert!(s.dispatches < u64::MAX, "{:?}", s);
}
