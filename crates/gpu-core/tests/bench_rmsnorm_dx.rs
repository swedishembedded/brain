// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RMSNorm backward w.r.t. x: one thread per row (`rmsnorm_dx`) vs one
//! workgroup per row (`rmsnorm_dx_rows`) - parity + achieved bandwidth.
//!
//! `rmsnorm_dx.wgsl` walks its row TWICE from a single thread (once for
//! `sum(x^2)`, once for `sum(dy*w*x)`); a warp's 32 loads are `d` floats
//! apart, so each 32-byte sector fetched serves one useful float - the same
//! `Op::RmsNorm`/`Op::LayerNorm` finding `bench_layernorm.rs` already
//! measured for the forward half and for `layernorm_dx`. This is the
//! RMSNorm backward half, which had no cooperative sibling at all before
//! this milestone.
//!
//! ```text
//! DISPLAY= BRAIN_DEVICE=gpu1 cargo test --release -p brain-gpu-core \
//!     --test bench_rmsnorm_dx -- --ignored --nocapture
//! ```
//!
//! `PEAK_GBPS` is the Tesla P40's datasheet bandwidth; on another card set it
//! to that card's own figure and read the achieved column, not the
//! percentage.
//!
//! ## The cooperative kernel does NOT win at every width - read the sweep
//!
//! The first sweep of this file covered `d` 896-5120 only, and the cooperative
//! kernel won at all of it. Extending it DOWN to the per-head QK-norm widths
//! the Qwen3 family really dispatches (`head_dim` 64/128) REVERSES the result:
//! at `d = 64` the `_rows` kernel is consistently slower than the reference at
//! every row count swept, and at `d = 128` it is a loss at some row counts and
//! a win at others. Read the `speedup` column of a real run - do not assume
//! the sign from the wide shapes.
//!
//! The cause is arithmetic, not memory. `rmsnorm_dx_rows` avoids a second
//! barrier (the CPU JIT permits exactly one) by having ALL 64 threads
//! redundantly fold the 64 partials of BOTH reductions - a fixed cost per row,
//! independent of `d`. At `d = 5120` each thread already did 80 elements of
//! real work and that fold disappears into them; at `d = 64` each thread did
//! ONE element, so the fold does orders of magnitude more arithmetic than the
//! reduction it is folding. Widen the row and the fix pays for itself; narrow
//! it below the workgroup and it cannot.
//!
//! So the crossover is a property of `d` (row WIDTH), not of `rows` - which is
//! a different axis from the `m <= 32` row-COUNT gate that was once a real bug
//! on this op, and must not be confused with it. Re-run this sweep on the card
//! in question rather than trusting either figure.

use gpu_core::Gpu;

const PEAK_GBPS: f64 = 346.0;
/// Relative agreement gate. The cooperative kernel folds its two partial
/// reductions in a different order (`bench_layernorm.rs`'s own tolerance for
/// `layernorm_dx`/`layernorm_dx_rows`) - the same answer to fp32 round-off is
/// the contract, not bit-identity.
const TOL: f32 = 2e-5;

/// (rows, d_model) - the same shape family `bench_layernorm.rs` sweeps: a
/// decode-shaped single row, narrow per-head rows, and prefill/training-width
/// row blocks, at the d_model values the RMSNorm-family models in this repo
/// (Qwen3/Qwen3.5/DeepSeek2/GLM-DSA) actually carry.
const SHAPES: &[(u32, u32)] = &[
    (512, 896),
    (1024, 896),
    (2048, 896),
    (512, 1024),
    (1024, 1024),
    (2048, 1024),
    (512, 2048),
    (1024, 2048),
    (2048, 2048),
    (512, 5120),
    (1024, 5120),
    (2048, 5120),
    // Decode / tiny-row regime: does the cooperative kernel still win when
    // there is only a handful of rows to spread over the card?
    (1, 1024),
    (1, 5120),
    (8, 2048),
    // Per-head QK-norm rows: `head_dim` wide, `b*t*n_heads` of them. This is
    // the NARROWEST row any RMSNorm backward in this repo dispatches (the
    // Qwen3 family's `attn.q_norm`/`attn.k_norm`, which `qwen3tts`'s Talker
    // inherits verbatim), and the per-element kernel's one-thread-per-row
    // layout scatters its reads hardest exactly here.
    // NARROW-ROW regime: `d` at or below the 64-thread workgroup width, which
    // the original sweep (`d` 896-5120) never reached. This is where the
    // Qwen3 family's per-head QK-norms live (`head_dim` 128, `b*t*n_heads`
    // rows), and it is where the cooperative kernel STOPS winning - see this
    // file's own header. Kept in the sweep precisely because it is the
    // counter-example; a sweep that only covers the shapes a kernel wins at
    // is not a measurement.
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (16384, 128),
    (2048, 64),
    (8192, 64),
    (2048, 256),
    (2048, 512),
    (8192, 512),
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5).collect()
}

/// Max absolute difference, normalised by the larger of the reference's own
/// magnitude and the data scale (`fill` spans +-0.5).
const DATA_SCALE: f32 = 0.5;
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let md = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    md / a.iter().fold(DATA_SCALE, |m, &v| m.max(v.abs()))
}

/// Min-of-`reps` wall clock for one dispatch (warm-up submitted first).
fn time(gpu: &Gpu, kind: usize, bufs: &[&gpu_core::DeviceBuffer], p: &[u32], threads: u32, reps: usize) -> f64 {
    let s = gpu.step(kind, bufs, p, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        // 4 back-to-back dispatches so launch overhead is amortised, as in
        // `bench_layernorm`/`bench_backward`; the reported time is per
        // dispatch.
        let steps: Vec<_> = (0..4).map(|_| gpu.step(kind, bufs, p, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / 4.0);
    }
    best
}

#[test]
#[ignore]
fn bench_rmsnorm_dx() {
    let ks = &[("rmsnorm_dx", kernels::RMSNORM_DX), ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS)];
    let (dx, dxr) = (0usize, 1);
    let g = Gpu::new_wgpu(ks);
    let reps = 8;

    println!(
        "\nrmsnorm_dx  (one thread per row -> one workgroup per row)\n{:<14} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10}",
        "rows x d", "ref ms", "ref GB/s", "rows ms", "rows GB/s", "speedup", "rel diff"
    );
    println!("{}", "-".repeat(80));
    for &(rows, d) in SHAPES {
        let n = (rows * d) as usize;
        let xb = g.storage_init("x", &fill(n, 1));
        let wb = g.storage_init("w", &fill(d as usize, 2));
        let dyb = g.storage_init("dy", &fill(n, 3));
        let dxa = g.storage(n as u64);
        let dxb_out = g.storage(n as u64);
        let p = [d, rows];

        // Bytes moved by the minimal implementation: x, w, dy read once, dx
        // written once (the cooperative kernel does not re-read anything the
        // reference did not already read - same traffic, see `cost.rs`).
        let bytes = 3.0 * n as f64 * 4.0 + n as f64 * 4.0;
        let bufs_a: Vec<&gpu_core::DeviceBuffer> = vec![&xb, &wb, &dyb, &dxa];
        let bufs_b: Vec<&gpu_core::DeviceBuffer> = vec![&xb, &wb, &dyb, &dxb_out];

        let ta = time(&g, dx, &bufs_a, &p, rows, reps);
        let tb = time(&g, dxr, &bufs_b, &p, rows * 64, reps);
        let (ra, rb) = (g.read(&dxa, n), g.read(&dxb_out, n));
        let diff = rel(&ra, &rb);
        println!(
            "{:>6} x {:<5} {:>10.3} {:>10.0} {:>10.3} {:>10.0} {:>8.1}x {:>10.1e}",
            rows,
            d,
            ta * 1e3,
            bytes / ta / 1e9,
            tb * 1e3,
            bytes / tb / 1e9,
            ta / tb,
            diff
        );
        assert!(diff < TOL, "rmsnorm_dx {rows}x{d}: cooperative variant diverges (rel {diff:.2e})");
    }
    println!("\n(GB/s vs {PEAK_GBPS} GB/s peak on a Tesla P40)");
}
