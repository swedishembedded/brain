// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LayerNorm: one thread per row (`layernorm`, `ln_stats`, `layernorm_dx`) vs
//! one workgroup per row (`*_rows`) — parity + achieved bandwidth.
//!
//! The per-element kernels give thread `t` row `t`, so a warp's 32 loads are
//! `d` floats apart and each 32-byte sector fetched serves one useful float.
//! This is the LayerNorm half of the finding in `docs/performance/overview.md`
//! (the RMSNorm half measured 19.4x). Shapes are the ones `gpt`, `pid`,
//! `seq2seq` and the ViT trunk actually dispatch.
//!
//! ```text
//! DISPLAY= BRAIN_DEVICE=gpu1 cargo test --release -p brain-gpu-core \
//!     --test bench_layernorm -- --ignored --nocapture
//! ```
//!
//! `PEAK_GBPS` is the Tesla P40's 346 GB/s; on another card read the achieved
//! column, not the percentage.

use gpu_core::Gpu;

const PEAK_GBPS: f64 = 346.0;
/// Relative agreement gate. The cooperative kernels reduce in a different order
/// (and from a shifted origin), so exact equality is not the contract — the
/// same answer to fp32 round-off is.
const TOL: f32 = 2e-5;

/// (rows, d_model) — d_model 768/1024/2048/3072 across the row counts a
/// batch x seq of these models produces.
const SHAPES: &[(u32, u32)] = &[
    (512, 768),
    (1024, 768),
    (2048, 768),
    (512, 1024),
    (1024, 1024),
    (2048, 1024),
    (512, 2048),
    (1024, 2048),
    (2048, 2048),
    (512, 3072),
    (1024, 3072),
    (2048, 3072),
    // Decode / tiny-row regime: does the cooperative kernel still win when
    // there is only a handful of rows to spread over the card?
    (1, 768),
    (8, 2048),
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5).collect()
}

/// Max absolute difference, normalised by the larger of the reference's own
/// magnitude and the **data scale** (`fill` spans ±0.5).
///
/// The floor matters: `ln_stats`' `mean` output is ~1e-3 for zero-mean input,
/// so dividing by its own magnitude would turn a 5e-8 absolute difference into
/// a 5e-5 "relative" one and say nothing about accuracy.
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
        // `bench_backward`; the reported time is per dispatch.
        let steps: Vec<_> = (0..4).map(|_| gpu.step(kind, bufs, p, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / 4.0);
    }
    best
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
#[ignore]
fn bench_layernorm() {
    let ks = &[
        ("layernorm", kernels::LAYERNORM),
        ("layernorm_rows", kernels::LAYERNORM_ROWS),
        ("ln_stats", kernels::LN_STATS),
        ("ln_stats_rows", kernels::LN_STATS_ROWS),
        ("layernorm_dx", kernels::LAYERNORM_DX),
        ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ];
    let (ln, lnr, st, str_, dx, dxr) = (0usize, 1, 2, 3, 4, 5);
    let g = Gpu::new_wgpu(ks);
    let eps = 1e-5f32;
    let reps = 8;

    for (name, a, b) in [
        ("layernorm      ", ln, lnr),
        ("ln_stats       ", st, str_),
        ("layernorm_dx   ", dx, dxr),
    ] {
        println!(
            "\n{name}  (one thread per row -> one workgroup per row)\n{:<14} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10}",
            "rows x d", "ref ms", "ref GB/s", "rows ms", "rows GB/s", "speedup", "rel diff"
        );
        println!("{}", "-".repeat(80));
        for &(rows, d) in SHAPES {
            let n = (rows * d) as usize;
            let xb = g.storage_init("x", &fill(n, 1));
            let gb = g.storage_init("gamma", &fill(d as usize, 2));
            let bb = g.storage_init("beta", &fill(d as usize, 3));
            let dyb = g.storage_init("dy", &fill(n, 4));
            let oa = g.storage(n as u64);
            let ob = g.storage(n as u64);
            let ma = g.storage(rows as u64);
            let ia = g.storage(rows as u64);
            let mb = g.storage(rows as u64);
            let ib = g.storage(rows as u64);
            let p = [d, rows, f(eps)];

            // bytes moved by the minimal implementation of each op
            let (bufs_a, bufs_b, bytes, out_len): (Vec<&gpu_core::DeviceBuffer>, Vec<&gpu_core::DeviceBuffer>, f64, usize) =
                if a == ln {
                    (vec![&xb, &gb, &bb, &oa], vec![&xb, &gb, &bb, &ob], 2.0 * n as f64 * 4.0, n)
                } else if a == st {
                    (vec![&xb, &ma, &ia], vec![&xb, &mb, &ib], n as f64 * 4.0, rows as usize)
                } else {
                    (vec![&xb, &gb, &dyb, &oa], vec![&xb, &gb, &dyb, &ob], 3.0 * n as f64 * 4.0, n)
                };

            let ta = time(&g, a, &bufs_a, &p, rows, reps);
            let tb = time(&g, b, &bufs_b, &p, rows * 64, reps);
            let (ra, rb) = if a == st {
                (g.read(&ma, out_len), g.read(&mb, out_len))
            } else {
                (g.read(&oa, out_len), g.read(&ob, out_len))
            };
            let diff = rel(&ra, &rb);
            // ln_stats also produces `inv`; check it too.
            let diff = if a == st {
                diff.max(rel(&g.read(&ia, out_len), &g.read(&ib, out_len)))
            } else {
                diff
            };
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
            assert!(diff < TOL, "{name} {rows}x{d}: cooperative variant diverges (rel {diff:.2e})");
        }
    }
    println!("\n(GB/s vs {PEAK_GBPS} GB/s peak on a Tesla P40)");
}
