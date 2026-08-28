// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The measured roofline: does the probe produce a number that can be believed?
//!
//! These assertions are deliberately device-INDEPENDENT. A test that pinned
//! one card's datasheet peak would pass on exactly that card and would be the
//! very defect this module exists to remove. What is checked instead is that the
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
//! 3. **A roof measured on a parked device.** A GPU that has been idle runs at
//!    its frequency floor and takes seconds of continuous work to reach the
//!    clock it will run a job at. A probe that does not ramp it first reports
//!    the idle clock as the silicon's ceiling - and this module CACHES and
//!    PERSISTS what it measures, so that number then divides every later
//!    "% of roof" on the machine. The signature is a roof BELOW what a real
//!    memory-fed kernel achieves, which is physically impossible, and it is
//!    what `real_work_can_never_beat_the_roof` below checks.

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

/// **A real kernel cannot beat the silicon's ceiling.** `roof_fma` is a
/// register-resident, dependency-free FMA chain with no memory traffic in the
/// loop; `matmul_reg2` is a memory-fed GEMM. Whatever the tiling, the GEMM
/// cannot do more arithmetic per second than the chain. If it does, the roof
/// was measured in a different (slower) regime than the kernel it is used to
/// grade - which is exactly what a probe run on an idle, frequency-parked GPU
/// produces, and what silently invalidated every "% of peak" figure on such a
/// box until the probe learned to ramp the device first.
///
/// The GEMM runs FIRST and is itself ramped, so both sides are measured on an
/// awake device and the comparison is like-for-like. The margin below is
/// deliberately generous, because these parts throttle under sustained load
/// and the two measurements cannot occupy the same instant; the defect it
/// exists to catch overshot it comfortably.
#[test]
fn real_work_can_never_beat_the_roof() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);
    let g = gpu.new_like(&[("matmul_reg2", kernels::MATMUL_REG2)]);

    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let fill = |len: usize, seed: usize| -> Vec<f32> {
        (0..len).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
    };
    let xb = g.storage_init("x", &fill(m * k, 1));
    let wb = g.storage_init("w", &fill(n * k, 2));
    let ob = g.storage((m * n) as u64);
    let params = [m as u32, k as u32, n as u32];
    let threads = (m.div_ceil(128) * n.div_ceil(128) * 256) as u32;
    let step = g.step(0, &[&xb, &wb, &ob], &params, threads);
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;

    // Ramp and measure in one loop: the best single dispatch over a window of
    // continuous work is the fastest this kernel goes on this device.
    let mut best = f64::INFINITY;
    let t0 = std::time::Instant::now();
    while t0.elapsed() < std::time::Duration::from_secs(3) {
        let t = std::time::Instant::now();
        g.submit(&[], std::slice::from_ref(&step));
        g.poll_wait();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gemm = gflop / best;

    let Some(r) = roof::measure(&gpu) else {
        return; // unprobeable device: callers print `-`, never a guess
    };
    println!(
        "matmul_reg2 {m}x{k}x{n}: {gemm:.0} GFLOP/s against a measured roof of {:.0} GFLOP/s \
         ({:.0}% of it)",
        r.gflops,
        100.0 * gemm / f64::from(r.gflops)
    );
    assert!(
        f64::from(r.gflops) * 1.5 >= gemm,
        "the measured compute roof is {:.0} GFLOP/s but a real GEMM does {gemm:.0} GFLOP/s on \
         the same device -- the roof was measured in a slower regime than the kernels it grades",
        r.gflops
    );
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
    // wide because these parts throttle under sustained load, and the second
    // call is by construction made on a device the first one just heated: on an
    // integrated Arc under a binding package power limit
    // (`throttle_reason_pl1` set) a back-to-back pair straddled the old 0.25
    // band while each call was reproducible on its own. That is the chassis,
    // not the probe. The bar is set where it still catches what it is for - a
    // probe timing noise rather than the device, which misses by orders of
    // magnitude, not by tens of percent.
    let spread = |x: f32, y: f32| (x - y).abs() / x.max(y);
    assert!(spread(a.gflops, b.gflops) < 0.5, "compute {a:?} vs {b:?}");
    assert!(spread(a.gbs, b.gbs) < 0.5, "bandwidth {a:?} vs {b:?}");
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

/// `reprofile` must overwrite BOTH the in-memory and on-disk cache `ensure`
/// would otherwise serve - the `brain models list --reprofile` entry point,
/// which needs a fresh number even when a (possibly stale) one is already
/// cached, unlike every other caller.
#[test]
fn reprofile_overwrites_a_stale_cached_value() {
    if skip_gpu() {
        return;
    }
    let _probe = probe_lock();
    let scratch = std::env::temp_dir().join(format!("brain-roofline-reprofile-test-{}", std::process::id()));
    roof::set_cache_dir(Some(scratch.clone()));
    let gpu = gpu_core::testgpu::dev(&[("axpy", kernels::AXPY)]);

    // `reprofile` (unlike `ensure`) never consults the in-memory CACHE on
    // entry, so calling it first (rather than `ensure`) is what makes this
    // test self-contained regardless of what an earlier test in this same
    // process already measured and cached for this backend name.
    let Some(_) = roof::reprofile(&gpu) else {
        roof::set_cache_dir(None);
        std::fs::remove_dir_all(&scratch).ok();
        return; // unprobeable device - nothing to reprofile
    };
    let stale_path = std::fs::read_dir(&scratch).unwrap().next().unwrap().unwrap().path();
    // A hand-planted stale record: the SECOND `reprofile` below must not
    // trust it, and must replace it on disk, not merely shadow it in memory.
    std::fs::write(&stale_path, "gflops=1\ngbs=1\ncache_gbs=1\n").unwrap();

    let fresh = roof::reprofile(&gpu).expect("reprofile must measure a probeable device");
    assert_ne!((fresh.gflops, fresh.gbs), (1.0, 1.0), "reprofile served the hand-planted stale record instead of measuring");
    assert_eq!(roof::known(gpu.kind()), Some(fresh), "reprofile must update the in-memory cache too, not just the disk file");

    let reloaded_text = std::fs::read_to_string(&stale_path).unwrap();
    assert!(reloaded_text.contains(&format!("gflops={}", fresh.gflops)), "reprofile must overwrite the on-disk record: got {reloaded_text:?}");

    roof::set_cache_dir(None);
    std::fs::remove_dir_all(&scratch).ok();
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
    // Default budget is `roof_budget()` per loop, up to 4 loops
    // (compute/bandwidth/cache/int8),
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
    // `col2im` measured a small fraction of that roof, far under the
    // memory-bound defect line, which is what makes it a defect and not merely
    // an untuned kernel.
    let u = r.utilisation(0, 23_200_000_000, 1.0).unwrap();
    assert!(u < Bound::Memory.defect_pct(), "col2im-shaped utilisation {u}%");
}
