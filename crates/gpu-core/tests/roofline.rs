// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The measured roofline: does the probe produce a number that can be believed?
//!
//! These assertions are deliberately device-INDEPENDENT. A test that pins
//! "11.76 TFLOP/s" would pass on exactly one card and would be the very defect
//! this module exists to remove. What is checked instead is that the
//! measurement cannot be one of the two ways this probe can lie:
//!
//! 1. **A folded loop.** If a compiler proves the FMA chain away, the reported
//!    rate goes to absurdity. `roof_fma` sources its multiplier and addend from
//!    the uniform to prevent that; the sanity ceiling here is what would catch
//!    a regression in that guarantee.
//! 2. **A host-timed region.** `submit` only appends to the pending list on
//!    the wgpu backend, so an unbracketed loop
//!    times host recording and reports a rate above the hardware roof. A
//!    bandwidth number at or above the arithmetic rate is that failure.

use gpu_core::roof::{self, Bound, Roofs};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| v != "0").unwrap_or(false)
}

/// The probes must not run concurrently **with each other**.
///
/// A roofline probe measures the throughput available to it, so two of them
/// sharing a device measure the contended device and disagree — which is not a
/// bug in the probe, it is what the probe is for. The suite runs at
/// `--test-threads=8`, so without this the reproducibility check fails purely
/// because its neighbour is also saturating the card.
///
/// Production is unaffected: `roof::ensure` measures once and caches. This is
/// also why `ensure` is the API benches call, and why any *use* of the roofs
/// under load is reading a cached idle number rather than re-measuring.
static PROBE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn probe_lock() -> std::sync::MutexGuard<'static, ()> {
    PROBE.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn the_measured_roofs_are_physically_plausible() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    let Some(r) = roof::measure(&gpu) else {
        return; // unprobeable device — callers print `-`, never a guess
    };

    println!(
        "measured roofline on {}: {:.0} GFLOP/s fp32, {} int8, {:.1} GB/s DRAM, \
         {:.1} GB/s cache, ridge {:.1} FLOP/byte",
        gpu.kind(),
        r.gflops,
        r.int8_gops.map(|g| format!("{g:.0} GOP/s")).unwrap_or_else(|| "-".into()),
        r.gbs,
        r.cache_gbs,
        r.ridge()
    );

    // If the device claims an int8 dot path, the measured rate must exceed the
    // fp32 one — otherwise grading int8 kernels against fp32 was never the
    // distortion it was assumed to be, and the extra probe is not earning its
    // keep. Recorded either way rather than assumed.
    if let Some(g) = r.int8_gops {
        println!("  int8/fp32 rate ratio: {:.2}x", g / r.gflops);
        assert!(g.is_finite() && g > 0.0, "int8 roof {g}");
    }

    assert!(r.gflops.is_finite() && r.gflops > 0.0, "compute roof {}", r.gflops);
    assert!(r.gbs.is_finite() && r.gbs > 0.0, "bandwidth roof {}", r.gbs);

    // (1) A folded FMA chain reports absurdity. No device brain targets is
    // within two orders of magnitude of a petaflop of fp32.
    assert!(r.gflops < 1.0e6, "compute roof {} GFLOP/s implies the FMA loop folded", r.gflops);

    // (2) Bandwidth at or above the arithmetic rate means the timed region was
    // the host, not the device (§E.0). Every real device has a ridge > 1.
    assert!(
        r.gbs < r.gflops,
        "bandwidth {} GB/s >= compute {} GFLOP/s — the timing is measuring the host",
        r.gbs,
        r.gflops
    );
    assert!(r.ridge() > 1.0, "ridge {} FLOP/byte", r.ridge());
}

#[test]
fn measuring_twice_agrees() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    let (Some(a), Some(b)) = (roof::measure(&gpu), roof::measure(&gpu)) else {
        return;
    };
    // A probe whose answer moves run to run cannot rank anything. The band is
    // wide because these cards throttle under sustained load — the point is to
    // catch a probe that is timing noise, not to pin a clock.
    let spread = |x: f32, y: f32| (x - y).abs() / x.max(y);
    assert!(spread(a.gflops, b.gflops) < 0.25, "compute {a:?} vs {b:?}");
    assert!(spread(a.gbs, b.gbs) < 0.25, "bandwidth {a:?} vs {b:?}");
}

#[test]
fn caps_expose_the_roofs_only_after_something_measured_them() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    // `ensure` persists what it measures — point that at a scratch dir so a
    // TEST never writes the developer's real ~/.cache/brain (state outside
    // the repo). This is the only test in the file that calls the persisting
    // path on a probeable device.
    let scratch = std::env::temp_dir().join(format!("brain-roofline-test-{}", std::process::id()));
    roof::set_cache_dir(Some(scratch.clone()));
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    // `ensure` is the only thing that may run probe kernels; reading caps must
    // never have that side effect.
    let r = roof::ensure(&gpu);
    roof::set_cache_dir(None);
    std::fs::remove_dir_all(&scratch).ok();
    let Some(r) = r else {
        return;
    };
    let caps = gpu.caps();
    assert_eq!(caps.peak_gflops, Some(r.gflops));
    assert_eq!(caps.peak_bandwidth_gbs, Some(r.gbs));
    assert_eq!(caps.ridge_flops_per_byte(), Some(r.ridge()));
}

/// `ensure` must not run any probe kernel at all on the CPU backend unless the
/// caller explicitly opts in (`BRAIN_NO_ROOF=0`) — see the doc on `roof::ensure`.
/// A probe that actually launched would take at minimum `MIN_PROBE_SECONDS`
/// per rung; this test's bound (well under that) is only clearable by the
/// default-off skip returning immediately.
#[test]
fn ensure_defaults_off_on_the_cpu_backend() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    if gpu.caps().class != backend_api::DeviceClass::Cpu {
        return; // this behaviour is CPU-specific; nothing to check elsewhere
    }
    let t0 = std::time::Instant::now();
    let r = roof::ensure(&gpu);
    assert!(r.is_none(), "CPU backend must default to unprobed, got {r:?}");
    assert!(
        t0.elapsed() < std::time::Duration::from_millis(50),
        "ensure() took {:?} on the CPU backend with no opt-in -- it ran a probe \
         instead of skipping",
        t0.elapsed()
    );
}

/// The calibration loops (`measure`) must never block past their wall-clock
/// budget, regardless of backend. This is the actual regression test for the
/// unbounded-hang defect: before the fix, a CPU-backend probe that stalled
/// inside `backend-cpu`'s dispatch never returned at all. Now the shared
/// `roof_budget()` ceiling bounds every one of `measure`'s (up to four)
/// internal calibration loops, so total wall time is provably finite even on
/// a device where the underlying stall reproduces -- "a slow measurement, not
/// a hang" (see `roof.rs`'s `1a` fix note).
#[test]
fn measure_is_bounded_by_the_roof_budget_even_if_a_rung_stalls() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    let t0 = std::time::Instant::now();
    let _ = roof::measure(&gpu); // Some() or None both fine -- only the bound matters
    // Default budget is 10s per loop, up to 4 loops (compute/bandwidth/cache/int8),
    // and `best_of` itself now stops repping once its loop's deadline passes (it
    // used to be checked only BETWEEN calls, so one call started just under the
    // wire could legally run 4 more 15s-bounded dispatches past it). A generous
    // multiple still leaves slack for scheduling jitter without re-admitting an
    // effectively-unbounded wait.
    let elapsed = t0.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "measure() took {elapsed:?} -- the per-loop deadline did not bound it"
    );
}

#[test]
fn a_streaming_kernel_is_graded_against_bandwidth_not_flops() {
    // Pure arithmetic on the type — no device needed, so this runs everywhere
    // and pins the classification rule itself.
    let r = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0) };
    // `axpy`: 2 FLOP per 12 bytes moved.
    assert_eq!(r.classify(2, 12), Bound::Memory);
    // `col2im` measured 23.2 GB/s on a P40 — 6.7% of that roof, far under the
    // memory-bound defect line, which is what makes it a defect and not merely
    // an untuned kernel.
    let u = r.utilisation(0, 23_200_000_000, 1.0).unwrap();
    assert!(u < Bound::Memory.defect_pct(), "col2im-shaped utilisation {u}%");
}
