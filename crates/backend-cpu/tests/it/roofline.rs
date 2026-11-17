// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `backend_cpu::roofline::measure` must produce real, fast, sane fp32 and
//! memory numbers for `brain perf roofline`'s CPU rung: the silicon's SIMD FMA
//! rate and the memory bandwidth over every core the process may use, brain's
//! own conv kernel mix beside them, and the published peak they are compared
//! with where the CPU is known.
//!
//! `gpu_core::roof::ensure` skips the CPU device class by default (a
//! known-bad interaction between its calibration loop and this crate's own
//! rayon dispatch, see the comment on `roof::ensure`), so the CPU otherwise
//! reports no roofline at all. This test suite gates the replacement: the
//! lifted methodology from `fast_conv::tests::bench_conv_gflops` (min-of-N
//! wall time over a representative NCHW conv2d mix) must still land in the
//! same physical ballpark once it is a real, fast, non-`#[ignore]`d API.

use backend_cpu::par::allowed_cores;
use backend_cpu::roofline::{last_level_cache_bytes, measure, CpuRoofline};
use std::sync::{Mutex, OnceLock};

/// The probes own every core while they run: two at once measure each other. Every measurement here takes this.
static CORES: Mutex<()> = Mutex::new(());

/// One measurement shared by the checks that only read it.
fn measured() -> &'static CpuRoofline {
    static R: OnceLock<CpuRoofline> = OnceLock::new();
    R.get_or_init(|| {
        let _cores = CORES.lock().unwrap_or_else(|e| e.into_inner());
        measure()
    })
}

/// A real, measured baseline this suite is graded against: the aggregate line
/// of the output of `cargo test -p brain-backend-cpu --release
/// bench_conv_gflops -- --ignored --nocapture` run on this development host
/// (22 threads; it prints one row per conv shape plus the aggregate).
///
/// ```text
/// TOTAL     (aggregate)  70.7 GFLOP/s   perf-number: the fixture the constant encodes
/// ```
///
/// This is not a portable constant (a CI runner's CPU differs from this dev
/// host), which is why the check below is an order-of-magnitude bound, not
/// an equality - the point is to catch the lift silently changing the
/// methodology (e.g. a units bug, or measuring something else entirely), not
/// to pin an exact clock speed.
const IGNORED_BENCH_AGGREGATE_GFLOPS: f32 = 70.7;

/// Requirement 1: the number is positive, and sane for a CPU - not zero
/// (a stuck timer or an eliminated call), not absurdly huge (a units bug, or
/// the optimizer proving the repeated call loop-invariant and hoisting the
/// real work out from under the timer, as `fast_ops.rs`'s `moe_linear_gated_
/// bench` module comment describes hitting for a different shape). The
/// ceiling is a generous multiple of any real single-node CPU's fp32 peak,
/// not a tight fit to this dev host.
#[test]
fn measure_returns_sane_gflops() {
    let r = measured();
    assert!(r.conv_gflops > 1.0, "suspiciously low CPU GFLOP/s: {}", r.conv_gflops);
    assert!(r.conv_gflops < 100_000.0, "suspiciously high CPU GFLOP/s (units bug or elided work?): {}", r.conv_gflops);
}

/// Requirement 2: the lift must not silently change the methodology and
/// produce a different order of magnitude than the proven ignored bench's
/// real output (see `IGNORED_BENCH_AGGREGATE_GFLOPS` above). A factor of 10
/// either way is generous cross-machine headroom while still catching a
/// methodology break (e.g. FLOP-counting only half the multiply-adds, or
/// timing something that isn't the compute at all).
#[test]
fn measure_matches_ignored_bench_order_of_magnitude() {
    let r = measured();
    let ratio = r.conv_gflops / IGNORED_BENCH_AGGREGATE_GFLOPS;
    assert!(
        (0.1..10.0).contains(&ratio),
        "measure() = {} GFLOP/s is not within an order of magnitude of the ignored bench's {} GFLOP/s (ratio {ratio})",
        r.conv_gflops,
        IGNORED_BENCH_AGGREGATE_GFLOPS
    );
}

/// Requirement 3: a real MEMORY roof, not `None`.
///
/// Every hot kernel of a resident decoder's steady state is a GEMV - one new
/// token against a weight read once and never revisited - at 0.5 FLOP/byte.
/// Grading those against a compute roof names the wrong bound, so the CPU rung
/// has to report a bandwidth the same way `gpu_core::roof` does for a card.
/// Sanity, not equality: below any real DDR3 single channel would mean the
/// probe is measuring something other than DRAM, and above a terabyte a second
/// would mean it never left cache.
#[test]
fn measure_returns_a_real_dram_bandwidth() {
    let r = measured();
    for (what, gbs) in [("copy", r.copy_gbs), ("triad", r.triad_gbs)] {
        assert!(gbs > 1.0, "suspiciously low {what} GB/s: {gbs}");
    }
    // The working set must be DRAM's, not the caches': several times every last-level cache the cores share.
    if let Some(llc) = last_level_cache_bytes() {
        assert!(r.working_set_bytes >= 4 * llc, "the bandwidth working set {} B is not past the {llc} B of last-level cache", r.working_set_bytes);
    }
    if let Some(peak) = r.peak.as_ref().and_then(|p| p.memory_gbs) {
        assert!(f64::from(r.triad_gbs) <= 1.1 * peak, "triad {} GB/s above the published {peak} GB/s: the probe read cache", r.triad_gbs);
    }
}

/// Requirement 5: the rung uses every core the process may run on, one thread
/// a core, and names how many and which SIMD path it ran.
#[test]
fn measure_runs_one_simd_thread_on_every_allowed_core() {
    let r = measured();
    assert_eq!(r.threads, allowed_cores().len());
    assert!(!r.simd.is_empty());
}

/// Requirement 6: the FMA probe is the silicon's ceiling. No real kernel beats
/// it (brain's conv mix is held under it), and it never beats the published
/// peak derived from the cores' issue width and clock.
#[test]
fn the_fma_probe_is_a_ceiling_no_kernel_exceeds_and_it_stays_under_the_published_peak() {
    let r = measured();
    assert!(r.fma_gflops * 1.1 >= r.conv_gflops, "the conv mix ({} GFLOP/s) beat the FMA probe ({} GFLOP/s)", r.conv_gflops, r.fma_gflops);
    if let Some(p) = &r.peak {
        assert!(f64::from(r.fma_gflops) <= 1.05 * p.fp32_gflops, "FMA probe {} GFLOP/s above the published {} GFLOP/s ({})", r.fma_gflops, p.fp32_gflops, p.source);
    }
}

/// Requirement 4: fast enough to be one rung of `brain perf roofline`'s "first
/// result within 10 seconds" budget across every accelerator class - well
/// a few seconds on its own, generously bounded (see the assertion) so a loaded CI runner
/// doesn't flake.
#[test]
fn measure_is_fast() {
    let _cores = CORES.lock().unwrap_or_else(|e| e.into_inner());
    let t = std::time::Instant::now();
    let r = measure();
    let elapsed = t.elapsed();
    eprintln!("measure() = {r:?} in {elapsed:?}");
    // perf-number: the 10s budget is a threshold the assertion enforces
    assert!(elapsed.as_secs_f64() < 4.0, "measure() took {:?}, too slow for a <10s multi-accelerator report", elapsed);
    assert!(r.fma_gflops > 0.0);
}
