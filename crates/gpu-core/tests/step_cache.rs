// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **`Gpu::enable_step_cache`: replaying a recorded dispatch computes what
//! rebuilding it computes, and only where the call really is the same one.**
//!
//! The cache exists because rebuilding a dispatch is expensive host-side work
//! (a fresh uniform buffer plus a fresh bind group per dispatch on wgpu) that
//! a loop re-recording a wide tape every iteration pays over and over. Its
//! entire correctness claim is that `(kernel, buffers, params, threads)` is
//! everything a backend reads to build one, so:
//!
//! 1. **A replayed tape's results are BIT-IDENTICAL to an unreplayed one's.**
//!    Not close - the same dispatch object, submitted again. This is the
//!    assertion a model adopting it stands on; anything weaker and a decode
//!    step could quietly change because a cache was armed.
//! 2. **Buffer CONTENTS are outside the key.** A bind group names buffers,
//!    never their values, so re-submitting a hit after rewriting its inputs
//!    must compute the NEW values - the property the whole thing is for (a
//!    decode step's tape differs between tokens only in what it reads).
//! 3. **A different call MISSES.** Different params, different buffers, or a
//!    different thread count must each build their own dispatch rather than
//!    silently reusing a neighbour's - the failure mode that would be
//!    invisible in output for exactly as long as the two happened to agree.
//! 4. **The cap is a cap**, and an entry never outlives the buffers its key
//!    names (which is what makes an `alloc_id` key sound at all).
//!
//! Both backends: the default device and the CPU Cranelift JIT.
//!
//! Each case owns its own pipeline-list constant. `gpu_core::testgpu::dev`
//! pools by the slice's ADDRESS, so two cases naming one constant would share
//! a handle - and therefore one cache - which under `--test-threads` would let
//! them count each other's hits. Distinct constants keep them on distinct
//! handles of the one shared physical device.

use gpu_core::Gpu;

/// The fixture kernel: `add2` (`out = a + b`, `params = [total]`). One
/// elementwise dispatch with a plain `(a, b, out)` signature is all identity
/// needs. `axpy` is registered only so the list is not a single entry that
/// could accidentally be the same slice as another crate's.
/// The two lists differ in length on purpose: two `const` slices with
/// identical contents may be merged into one static, which would hand both
/// cases the same pooled handle and the same cache - exactly what naming them
/// separately is meant to avoid.
const REPLAY_PIPES: &[(&str, &str)] = &[("add2", kernels::ADD2), ("axpy", kernels::AXPY)];
const MISS_PIPES: &[(&str, &str)] = &[("add2", kernels::ADD2), ("axpy", kernels::AXPY), ("scale_add", kernels::SCALE_ADD)];

const ADD2: usize = 0;
const N: usize = 64;

/// Both backends, as handle factories over `pipes`: `Gpu` is not `Clone`, and
/// the default device's handle comes from the pool rather than a fresh
/// `Gpu::new` (which deadlocks the driver under `--test-threads`).
fn devices(pipes: &'static [(&'static str, &'static str)]) -> Vec<(&'static str, Box<dyn Fn() -> Gpu>)> {
    let mut v: Vec<(&'static str, Box<dyn Fn() -> Gpu>)> = Vec::new();
    if std::env::var("MOE_SKIP_GPU_TESTS").is_err() {
        v.push(("default", Box::new(move || gpu_core::testgpu::dev(pipes))));
    } else {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set: the default (GPU) device is not exercised");
    }
    v.push(("cpu-jit", Box::new(move || Gpu::new_cpu(pipes))));
    v
}

fn ramp(base: f32) -> Vec<f32> {
    (0..N).map(|i| base + i as f32 * 0.25).collect()
}

/// (1) and (2): the same two-round tape, run once with the cache off and once
/// with it on - identical results both times, and the cached round really did
/// recompute from the rewritten inputs rather than hand back the first round's
/// numbers.
///
/// Both halves run on ONE handle (the pooled device hands out one per pipeline
/// slice, so "a second handle" is not available to ask for), with the cache
/// armed between them and cleared after - which also leaves the pooled handle
/// as this case found it.
#[test]
fn a_replayed_dispatch_computes_what_a_rebuilt_one_computes() {
    for (label, dev) in devices(REPLAY_PIPES) {
        let g = dev();
        g.clear_step_cache();
        let (a0, b0) = (ramp(1.0), ramp(-3.5));
        let (a1, b1) = (ramp(11.0), ramp(0.75));

        // Reference: cache off, one dispatch rebuilt per round.
        let (pa, pb, pout) = (g.storage_init("a", &a0), g.storage_init("b", &b0), g.storage(N as u64));
        let s = g.step(ADD2, &[&pa, &pb, &pout], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        let want0 = g.read(&pout, N);
        g.write_f32(&pa, &a1);
        g.write_f32(&pb, &b1);
        let s = g.step(ADD2, &[&pa, &pb, &pout], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        let want1 = g.read(&pout, N);
        assert!(want0.iter().any(|v| v.abs() > 1e-6), "[{label}] reference round 0 is ~zero");
        assert_ne!(want0, want1, "[{label}] the two rounds must differ, or the case proves nothing");

        // Same tape on distinct buffers, cache armed: round 1 is a hit.
        g.enable_step_cache(64);
        let (ma, mb, mout) = (g.storage_init("a", &a0), g.storage_init("b", &b0), g.storage(N as u64));
        let s = g.step(ADD2, &[&ma, &mb, &mout], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        let got0 = g.read(&mout, N);
        g.write_f32(&ma, &a1);
        g.write_f32(&mb, &b1);
        let s = g.step(ADD2, &[&ma, &mb, &mout], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        let got1 = g.read(&mout, N);

        let (hits, misses, live) = g.step_cache_stats().expect("armed");
        assert_eq!((hits, misses, live), (1, 1, 1), "[{label}] expected one miss then one hit over one entry");
        g.clear_step_cache();

        for (i, (x, w)) in got0.iter().zip(&want0).enumerate() {
            assert_eq!(x.to_bits(), w.to_bits(), "[{label}] round 0 out[{i}]: replayed {x} vs rebuilt {w}");
        }
        for (i, (x, w)) in got1.iter().zip(&want1).enumerate() {
            assert_eq!(x.to_bits(), w.to_bits(), "[{label}] round 1 out[{i}]: replayed {x} vs rebuilt {w}");
        }
    }
}

/// (3): each of the three things that are not the buffers must still separate
/// two calls. A cache that ignored any of them would return a dispatch of the
/// wrong width, over the wrong memory, or with the wrong grid - and would keep
/// looking right for exactly as long as the two calls happened to agree.
#[test]
fn a_different_call_misses() {
    for (label, dev) in devices(MISS_PIPES) {
        let g = dev();
        g.clear_step_cache();
        g.enable_step_cache(64);
        let a = g.storage_init("a", &ramp(1.0));
        let b = g.storage_init("b", &ramp(2.0));
        let out = g.storage(N as u64);
        let other = g.storage(N as u64);

        let s = g.step(ADD2, &[&a, &b, &out], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        assert_eq!(g.step_cache_stats().unwrap().0, 0, "[{label}] the first call cannot be a hit");

        let s = g.step(ADD2, &[&a, &b, &out], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        assert_eq!(g.step_cache_stats().unwrap().0, 1, "[{label}] the identical call must hit");

        // Different params.
        let s = g.step(ADD2, &[&a, &b, &out], &[(N / 2) as u32], (N / 2) as u32);
        g.submit(&[], &[s]);
        // Different output buffer, same params.
        let s = g.step(ADD2, &[&a, &b, &other], &[N as u32], N as u32);
        g.submit(&[], &[s]);
        // Same buffers and params, different thread count.
        let s = g.step(ADD2, &[&a, &b, &out], &[N as u32], (N / 2) as u32);
        g.submit(&[], &[s]);

        let (hits, misses, live) = g.step_cache_stats().unwrap();
        g.clear_step_cache();
        assert_eq!(hits, 1, "[{label}] only the repeated call may hit ({hits} hits)");
        assert_eq!(misses, 4, "[{label}] four distinct calls must each miss ({misses} misses)");
        assert_eq!(live, 4, "[{label}] four distinct entries");
    }
}

/// (4): the cap holds, and clearing releases. The buffer handles an entry pins
/// are what make an `alloc_id` key sound at all, so "cleared" has to mean the
/// entries AND their handles are gone - `DeviceBuffer::is_unique` on a buffer
/// the cache had pinned is the only way to observe that from outside.
///
/// CPU JIT only, deliberately: this asserts on handle counts, not on device
/// arithmetic, and a fresh `Gpu::new_cpu` is a handle no other case shares.
#[test]
fn the_cap_holds_and_clearing_releases_the_pinned_buffers() {
    let g = Gpu::new_cpu(REPLAY_PIPES);
    g.enable_step_cache(2);
    let a = g.storage_init("a", &ramp(1.0));
    let b = g.storage_init("b", &ramp(2.0));
    let out = g.storage(N as u64);
    assert!(out.is_unique(), "the fixture starts with one handle to `out`");

    // Eight calls that never repeat: nothing is ever reused, so eviction has
    // only unreused entries to drop and the cap is what bounds the map.
    for i in 0..8u32 {
        let s = g.step(ADD2, &[&a, &b, &out], &[N as u32 - i], N as u32);
        g.submit(&[], &[s]);
    }
    let (_, _, live) = g.step_cache_stats().unwrap();
    assert!(live <= 2, "cap 2 exceeded: {live} live entries");
    assert!(!out.is_unique(), "a live entry must pin the buffers its key names");

    g.clear_step_cache();
    assert!(g.step_cache_stats().is_none(), "a cleared cache reports nothing");
    assert!(out.is_unique(), "clearing must release the pinned handles");

    // Disarmed: dispatching still works, and records nothing.
    let s = g.step(ADD2, &[&a, &b, &out], &[N as u32], N as u32);
    g.submit(&[], &[s]);
    assert_eq!(g.read(&out, N)[0], 3.0, "a disarmed handle still dispatches");
    assert!(g.step_cache_stats().is_none(), "a disarmed handle records nothing");
}
