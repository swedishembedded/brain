// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 activation quant's per-row max: one thread per row
//! (`max_abs_row`) vs one workgroup per row (`max_abs_rows`) — parity +
//! achieved bandwidth.
//!
//! `max_abs_row` gives thread `t` row `t` and walks the whole row from that one
//! invocation, so a warp's 32 loads are `k` floats apart (each 32-byte sector
//! fetched serves one useful float) *and* the row is a serial chain of `k`
//! dependent loads. That is the same trap that produced
//! `gn_stats` (159x), `rmsnorm` (19.4x) and `layernorm` (2.8-10x) —
//! sitting on the int8 path every `qwen3::q8` / `zimage` / FLUX.2-int8 linear
//! quantizes its activations through.
//!
//! ```text
//! DISPLAY= BRAIN_DEVICE=gpu1 cargo test --release -p brain-gpu-core \
//!     --test bench_max_abs_row -- --ignored --nocapture
//! ```
//!
//! `PEAK_GBPS` is the Tesla P40's 346 GB/s; on another card read the achieved
//! column, not the percentage.
//!
//! Both kernels are registered and dispatched **by index** here, with
//! `BRAIN_NO_KERNEL_UPGRADE=1` pinning `Gpu::step` to the index it was given.
//! Without that, `gpu_core::upgrade` would redirect slot 0 to slot 1 and the
//! benchmark would measure the same kernel twice and report a confident 1.00x.
//! The end-to-end check that the redirect *does* fire lives in
//! `tests/kernel_upgrade.rs` (a separate binary, because the switch is
//! process-wide).

use gpu_core::Gpu;

const PEAK_GBPS: f64 = 346.0;

/// (rows, k) — shapes the int8 paths actually quantize at: FLUX.2's text
/// encoder / DiT (d 3072-3584, mlp 9216-12288), Qwen 0.6B (1024 / 3072) and
/// zimage's DiT, plus the decode-regime tail.
const SHAPES: &[(u32, u32)] = &[
    (512, 1024),
    (512, 3072),
    (512, 9216),
    (1024, 1024),
    (1024, 3072),
    (2048, 1024),
    (2048, 3072),
    (4096, 1024),
    (77, 3584),
    (77, 12288),
    (8, 3072),
    (1, 3072),
];

const KS: &[(&str, &str)] =
    &[("max_abs_row", kernels::MAX_ABS_ROW), ("max_abs_rows", kernels::MAX_ABS_ROWS)];

/// A handle with the transparent upgrade DISABLED, so slot 0 really is the
/// per-thread kernel. Set before any device is built (the switch is read once).
fn pinned_gpu() -> Gpu {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::env::set_var("BRAIN_NO_KERNEL_UPGRADE", "1"));
    gpu_core::testgpu::dev(KS)
}

fn fill(n: usize) -> Vec<f32> {
    // Deliberately includes per-row outliers: the point of a per-token scale is
    // that one large activation must not move any other row's scale.
    (0..n)
        .map(|i| {
            let v = (((i * 37) % 197) as f32 / 197.0) - 0.5;
            if i % 1021 == 0 {
                v * 17.0
            } else {
                v
            }
        })
        .collect()
}

/// Min-of-`reps` wall clock for one dispatch (warm-up submitted first).
fn time(
    gpu: &Gpu,
    kind: usize,
    bufs: &[&gpu_core::DeviceBuffer],
    p: &[u32],
    threads: u32,
    reps: usize,
) -> f64 {
    let s = gpu.step(kind, bufs, p, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        // 4 back-to-back dispatches so launch overhead is amortised; the
        // reported time is per dispatch.
        let steps: Vec<_> = (0..4).map(|_| gpu.step(kind, bufs, p, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / 4.0);
    }
    best
}

#[test]
#[ignore]
fn bench_max_abs_row() {
    let gpu = pinned_gpu();
    println!("backend: {}  (peak {PEAK_GBPS} GB/s)", gpu.kind());
    println!(
        "{:>6} {:>6} | {:>10} {:>7} | {:>9} {:>7} | {:>8}",
        "rows", "k", "per-thread", "GB/s", "coop", "GB/s", "speedup"
    );

    let mut worst = f64::INFINITY;
    for &(m, k) in SHAPES {
        let x = gpu.storage_init("x", &fill((m * k) as usize));
        let a = gpu.storage((m * 4) as u64);
        let b = gpu.storage((m * 4) as u64);

        let t_ref = time(&gpu, 0, &[&x, &a], &[m, k], m, 8);
        let t_coop = time(&gpu, 1, &[&x, &b], &[m, k], m * 64, 8);

        // `max` is exact and associative, so splitting a row across 64 lanes
        // cannot change the answer. BIT-identical is the contract, not "close":
        // a drifted scale would silently change every int8 activation.
        assert_eq!(
            gpu.read(&a, m as usize),
            gpu.read(&b, m as usize),
            "max_abs_rows must be bit-identical at {m}x{k}"
        );

        // Bytes: read the whole tensor, write one scale per row.
        let bytes = 4.0 * ((m as f64) * (k as f64) + m as f64);
        let gbps = |t: f64| bytes / t / 1e9;
        let sp = t_ref / t_coop;
        worst = worst.min(sp);
        println!(
            "{m:>6} {k:>6} | {:>10.3} {:>7.1} | {:>9.3} {:>7.1} | {:>7.2}x",
            t_ref * 1e3,
            gbps(t_ref),
            t_coop * 1e3,
            gbps(t_coop),
            sp
        );
    }
    println!("worst speedup across all shapes: {worst:.2}x");
}

/// The correctness half, runnable on any backend without `--ignored`: the
/// cooperative kernel agrees with the reference **bit for bit**, including the
/// all-zero row (whose `1e-8` guard keeps the scale finite) and row widths
/// shorter than, equal to, and not a multiple of the 64-lane stride.
#[test]
fn max_abs_rows_matches_reference_exactly() {
    let gpu = pinned_gpu();
    for &(m, k) in &[(1u32, 1u32), (3, 17), (5, 64), (7, 65), (33, 128), (64, 300), (129, 1024)] {
        let mut x = fill((m * k) as usize);
        // Row 0 all zeros -> exercises the 1e-8 floor on both kernels.
        for v in x.iter_mut().take(k as usize) {
            *v = 0.0;
        }
        let xb = gpu.storage_init("x", &x);
        let a = gpu.storage((m * 4) as u64);
        let b = gpu.storage((m * 4) as u64);
        let s0 = gpu.step(0, &[&xb, &a], &[m, k], m);
        let s1 = gpu.step(1, &[&xb, &b], &[m, k], m * 64);
        gpu.submit(&[], &[s0, s1]);
        assert_eq!(gpu.read(&a, m as usize), gpu.read(&b, m as usize), "{m}x{k}");
    }
}
