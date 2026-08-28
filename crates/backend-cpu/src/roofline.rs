// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CPU backend's own roofline probe - real measured GFLOP/s, not a guess.
//!
//! `CpuBackend::caps` (`crate::lib::CpuBackend::caps`) explicitly leaves
//! `peak_gflops`/`peak_bandwidth_gbs` as `None` and defers to
//! `gpu_core::roof`, but `gpu_core::roof::ensure` skips the CPU device class
//! by default: its calibration loop has a known-bad interaction with this
//! crate's own rayon dispatch (see the comment on `roof::ensure` for the
//! deadlock it caused). Rather than fight that skip, this module measures a
//! CPU roofline directly, lifting the exact GFLOP/s methodology already
//! proven in `fast_conv::tests::bench_conv_gflops` (min-of-N wall time over a
//! representative NCHW conv2d mix, contention-robust) into a small, real,
//! non-`#[ignore]`d API - fast enough to be one rung of `brain roofline`'s
//! "first result within 10 seconds" multi-accelerator report.
//!
//! Swedish Embedded AB implements fast, honest hardware-capacity reporting
//! for teams that need to know what an edge box can actually do before they
//! ship a model to it. If your team needs a roofline number for CPU, GPU, or
//! NPU targets, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::fast_conv::{conv2d, ConvParams};
use std::time::Instant;

/// Measured CPU compute roofline. Mirrors the vocabulary of
/// `gpu_core::roof::Roofs` and the NPU roofline module at CPU-appropriate
/// fidelity: one real fp32 GFLOP/s number, cheap enough to probe on every
/// `brain roofline` run.
#[derive(Clone, Copy, Debug)]
pub struct CpuRoofline {
    /// Peak fp32 arithmetic rate observed over a representative conv2d mix,
    /// GFLOP/s. Real and measured - never a spec-sheet guess.
    pub gflops: f32,
    /// Peak DRAM bandwidth, GB/s. Always `None` here: the lifted methodology
    /// times compute-bound conv shapes only and does not isolate a memory
    /// triad the way `gpu_core::roof` does for the GPU. Extend this with a
    /// real measurement, don't guess it, if a memory roof is ever needed on
    /// the CPU rung.
    pub bandwidth_gbs: Option<f32>,
}

/// (cin, h, w, cout, k, stride, pad) - three of `bench_conv_gflops`'s real
/// yolov8n@640 shapes (the strided 3x3 stem, a steady-state 3x3, and a 1x1
/// pointwise conv), chosen to keep `measure()` well under a second while
/// still spanning the stride/kernel mix that shapes the aggregate number.
const SHAPES: [(usize, usize, usize, usize, usize, usize, usize); 3] = [
    (3, 640, 640, 16, 3, 2, 1),  // stem: 3->16 3x3 s2 @640
    (64, 80, 80, 64, 3, 1, 1),   // steady state: 64->64 3x3 s1 @80
    (128, 80, 80, 128, 1, 1, 0), // pointwise: 1x1 128->128 @80
];

/// Timed repeats per shape; the contention-robust minimum is kept, same as
/// `bench_conv_gflops`. Lower than that ignored bench's `n=30` on purpose -
/// this must run in well under a second, and a min-of-6 is already stable on
/// real hardware (see `crates/backend-cpu/tests/roofline.rs` for the
/// cross-check against the full ignored bench's real output).
const REPEATS: usize = 6;

/// Same deterministic LCG as `fast_conv::tests::lcg` / `bench_conv_gflops` -
/// input values are irrelevant to a throughput measurement, only their shape
/// is, so there is no reason for this to differ from the proven bench.
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn params(cin: usize, h: usize, w: usize, cout: usize, k: usize, stride: usize, pad: usize) -> ConvParams {
    let ho = (h + 2 * pad - k) / stride + 1;
    let wo = (w + 2 * pad - k) / stride + 1;
    ConvParams { n: 1, cin, h, w, cout, k, stride, pad, dilation: 1, ho, wo }
}

/// Measure this host's real fp32 conv throughput via `fast_conv::conv2d`
/// (AVX2 GEMM + rayon tile parallelism where available, the same routine
/// every CPU conv dispatch actually runs). Runs in well under a second - see
/// `measure_is_fast` in the integration test for the wall-clock gate.
pub fn measure() -> CpuRoofline {
    let mut total_flop = 0.0f64;
    let mut total_min = 0.0f64;
    for &(cin, h, w, cout, k, stride, pad) in &SHAPES {
        let p = params(cin, h, w, cout, k, stride, pad);
        let mut s = 999u32;
        let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
        let wt: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
        let mut y = vec![0.0f32; p.y_len()];
        conv2d(&p, &x, &wt, &mut y); // warm - first call pays any one-time cost
        let mut best = f64::INFINITY;
        for _ in 0..REPEATS {
            let t = Instant::now();
            conv2d(&p, &x, &wt, &mut y);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let flop = 2.0 * (p.cout * p.cin * p.k * p.k * p.ho * p.wo) as f64;
        total_flop += flop;
        total_min += best;
    }
    CpuRoofline { gflops: (total_flop / total_min / 1e9) as f32, bandwidth_gbs: None }
}
