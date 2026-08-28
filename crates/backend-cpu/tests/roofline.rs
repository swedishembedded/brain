// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `backend_cpu::roofline::measure` must produce a real, fast, sane fp32
//! GFLOP/s number for `brain roofline`'s CPU rung.
//!
//! `gpu_core::roof::ensure` skips the CPU device class by default (a
//! known-bad interaction between its calibration loop and this crate's own
//! rayon dispatch, see the comment on `roof::ensure`), so the CPU otherwise
//! reports no roofline at all. This test suite gates the replacement: the
//! lifted methodology from `fast_conv::tests::bench_conv_gflops` (min-of-N
//! wall time over a representative NCHW conv2d mix) must still land in the
//! same physical ballpark once it is a real, fast, non-`#[ignore]`d API.

use backend_cpu::roofline::measure;

/// A real, measured baseline this suite is graded against: the output of
/// `cargo test -p brain-backend-cpu --release bench_conv_gflops -- --ignored
/// --nocapture` run on this development host (22 threads):
///
/// ```text
/// === conv microbench (min of 30, threads=22) ===
///   stem 3->16 3x3 s2 @640        2.02 ms     43.8 GFLOP/s
///   16->32 3x3 s2 @320            3.11 ms     75.9 GFLOP/s
///   32->32 3x3 s1 @160            7.71 ms     61.2 GFLOP/s
///   64->64 3x3 s1 @80             6.74 ms     70.0 GFLOP/s
///   128->128 3x3 s1 @40           5.73 ms     82.3 GFLOP/s
///   1x1 128->128 @80              2.30 ms     91.1 GFLOP/s
///   1x1 256->256 @20              0.69 ms     75.8 GFLOP/s
///   TOTAL                        28.30 ms     70.7 GFLOP/s (aggregate)
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
    let r = measure();
    assert!(r.gflops > 1.0, "suspiciously low CPU GFLOP/s: {}", r.gflops);
    assert!(r.gflops < 10_000.0, "suspiciously high CPU GFLOP/s (units bug or elided work?): {}", r.gflops);
}

/// Requirement 2: the lift must not silently change the methodology and
/// produce a different order of magnitude than the proven ignored bench's
/// real output (see `IGNORED_BENCH_AGGREGATE_GFLOPS` above). A factor of 10
/// either way is generous cross-machine headroom while still catching a
/// methodology break (e.g. FLOP-counting only half the multiply-adds, or
/// timing something that isn't the compute at all).
#[test]
fn measure_matches_ignored_bench_order_of_magnitude() {
    let r = measure();
    let ratio = r.gflops / IGNORED_BENCH_AGGREGATE_GFLOPS;
    assert!(
        (0.1..10.0).contains(&ratio),
        "measure() = {} GFLOP/s is not within an order of magnitude of the ignored bench's {} GFLOP/s (ratio {ratio})",
        r.gflops,
        IGNORED_BENCH_AGGREGATE_GFLOPS
    );
}

/// Requirement 3: fast enough to be one rung of `brain roofline`'s "first
/// result within 10 seconds" budget across every accelerator class - well
/// under a second on its own, generously bounded at 2s so a loaded CI runner
/// doesn't flake.
#[test]
fn measure_is_fast() {
    let t = std::time::Instant::now();
    let r = measure();
    let elapsed = t.elapsed();
    eprintln!("measure() = {:.1} GFLOP/s (bandwidth_gbs={:?}) in {:?}", r.gflops, r.bandwidth_gbs, elapsed);
    assert!(elapsed.as_secs_f64() < 2.0, "measure() took {:?}, too slow for a <10s multi-accelerator report", elapsed);
    assert!(r.gflops > 0.0);
}
