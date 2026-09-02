// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Attention backward dscores: one thread per query row
//! (`attn_bwd_dscores{,_bidir,_cross}`, `gqa_bwd_dscores`) vs one workgroup
//! per query row (their `_rows` twins) - parity + achieved bandwidth (M5.2).
//!
//! The per-element kernels give thread `t` query row `t`; the per-iteration
//! read of `d_out`/`d_ctx` is indexed by that thread-varying row, so a warp's
//! 32 loads land `d_model` floats apart - `Op::MaxAbsRow`'s coalescing bug,
//! not merely a slow reduction (see `backend_api::select::Op::
//! AttnBwdDScores`'s own doc and `attn_bwd_dscores_rows.wgsl`).
//!
//! Swedish Embedded AB implements validated GPU kernel selection for its
//! clients. If your team needs expertise in numerically-gated kernel
//! optimization then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ```text
//! BRAIN_DEVICE=gpu1 cargo test --release -p brain-gpu-core \
//!     --test bench_attn_bwd_dscores -- --ignored --nocapture
//! ```

use gpu_core::Gpu;

/// Relative agreement gate. The cooperative kernels fold the row's `dot`
/// reduction in a different order (64 partials rather than one running sum),
/// so exact equality is not the contract - the same answer to fp32 round-off
/// is (`rmsnorm_rows.wgsl`'s own contract).
const TOL: f32 = 2e-5;

/// (bsz, n_heads, tcols, head_dim) - causal/bidir/GQA shapes; `gpt2`'s own
/// dispatch (`4, 4, T, 16`) plus a wider/narrower spread.
const SHAPES: &[(u32, u32, u32, u32)] = &[(2, 4, 8, 16), (2, 4, 64, 16), (1, 8, 128, 32), (4, 2, 256, 8)];

/// (bsz, n_heads, t_dec, t_enc, head_dim) for the cross family.
const CROSS_SHAPES: &[(u32, u32, u32, u32, u32)] = &[(2, 4, 8, 12, 16), (1, 8, 64, 32, 16), (2, 2, 16, 128, 8)];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let md = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    md / a.iter().fold(0.5f32, |m, &v| m.max(v.abs()))
}

/// Min-of-`reps` wall clock for one dispatch (warm-up submitted first).
fn time(gpu: &Gpu, kind: usize, bufs: &[&gpu_core::DeviceBuffer], p: &[u32], threads: u32, reps: usize) -> f64 {
    let s = gpu.step(kind, bufs, p, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let steps: Vec<_> = (0..4).map(|_| gpu.step(kind, bufs, p, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / 4.0);
    }
    best
}

#[test]
#[ignore]
fn bench_attn_bwd_dscores_causal_and_bidir() {
    let ks = &[
        ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
        ("attn_bwd_dscores_rows", kernels::ATTN_BWD_DSCORES_ROWS),
        ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
        ("attn_bwd_dscores_bidir_rows", kernels::ATTN_BWD_DSCORES_BIDIR_ROWS),
        ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
        ("gqa_bwd_dscores_rows", kernels::GQA_BWD_DSCORES_ROWS),
    ];
    let g = Gpu::new_wgpu(ks);
    let reps = 8;

    for (name, a, b, gqa) in [("attn_bwd_dscores     ", 0usize, 1, false), ("attn_bwd_dscores_bidir", 2, 3, false), ("gqa_bwd_dscores      ", 4, 5, true)] {
        println!(
            "\n{name}  (one thread per row -> one workgroup per row)\n{:<18} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10}",
            "b h T hd", "ref ms", "ref rows", "rows ms", "rows rows", "speedup", "rel diff"
        );
        println!("{}", "-".repeat(90));
        for &(bsz, n_heads, t, hd) in SHAPES {
            let d_model = n_heads * hd;
            let n_kv_heads = if gqa { (n_heads / 2).max(1) } else { n_heads };
            let group = n_heads / n_kv_heads;
            let qkv_stride = 3 * d_model;
            let v_off = 2 * d_model;
            let rows = bsz * n_heads * t;

            let d_out = g.storage_init("d_out", &fill((bsz * t * d_model) as usize, 1));
            let qkv = g.storage_init("qkv", &fill((bsz * t * qkv_stride) as usize, 2));
            let v_kv = g.storage_init("v", &fill((bsz * t * n_kv_heads * hd) as usize, 2));
            let probs = g.storage_init("probs", &fill((bsz * n_heads * t * t) as usize, 3));
            let da = g.storage((bsz * n_heads * t * t) as u64);
            let db = g.storage((bsz * n_heads * t * t) as u64);

            let (bufs_a, bufs_b, p): (Vec<&gpu_core::DeviceBuffer>, Vec<&gpu_core::DeviceBuffer>, Vec<u32>) = if gqa {
                (
                    vec![&d_out, &v_kv, &probs, &da],
                    vec![&d_out, &v_kv, &probs, &db],
                    vec![bsz, n_heads, n_kv_heads, t, hd, group],
                )
            } else {
                (
                    vec![&d_out, &qkv, &probs, &da],
                    vec![&d_out, &qkv, &probs, &db],
                    vec![bsz, n_heads, t, hd, qkv_stride, v_off, d_model],
                )
            };

            let ta = time(&g, a, &bufs_a, &p, rows, reps);
            let tb = time(&g, b, &bufs_b, &p, rows * 64, reps);
            let n = (bsz * n_heads * t * t) as usize;
            let diff = rel(&g.read(&da, n), &g.read(&db, n));
            println!(
                "{:>2} {:>2} {:>4} {:>3} {:>10.3} {:>10} {:>10.3} {:>10} {:>8.1}x {:>10.1e}",
                bsz, n_heads, t, hd, ta * 1e3, rows, tb * 1e3, rows * 64, ta / tb, diff
            );
            assert!(diff < TOL, "{name} b={bsz} h={n_heads} T={t} hd={hd}: cooperative variant diverges (rel {diff:.2e})");
        }
    }
}

#[test]
#[ignore]
fn bench_attn_bwd_dscores_cross() {
    let ks = &[("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS), ("attn_bwd_dscores_cross_rows", kernels::ATTN_BWD_DSCORES_CROSS_ROWS)];
    let g = Gpu::new_wgpu(ks);
    let reps = 8;

    println!(
        "\nattn_bwd_dscores_cross  (one thread per row -> one workgroup per row)\n{:<22} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10}",
        "b h Tdec Tenc hd", "ref ms", "ref rows", "rows ms", "rows rows", "speedup", "rel diff"
    );
    println!("{}", "-".repeat(95));
    for &(bsz, n_heads, t_dec, t_enc, hd) in CROSS_SHAPES {
        let d_model = n_heads * hd;
        let kv_stride = 2 * d_model;
        let v_off = d_model;
        let rows = bsz * n_heads * t_dec;

        let d_out = g.storage_init("d_out", &fill((bsz * t_dec * d_model) as usize, 1));
        let kv = g.storage_init("kv", &fill((bsz * t_enc * kv_stride) as usize, 2));
        let probs = g.storage_init("probs", &fill((bsz * n_heads * t_dec * t_enc) as usize, 3));
        let da = g.storage((bsz * n_heads * t_dec * t_enc) as u64);
        let db = g.storage((bsz * n_heads * t_dec * t_enc) as u64);
        let p = [bsz, n_heads, t_dec, t_enc, hd, kv_stride, v_off, d_model];

        let ta = time(&g, 0, &[&d_out, &kv, &probs, &da], &p, rows, reps);
        let tb = time(&g, 1, &[&d_out, &kv, &probs, &db], &p, rows * 64, reps);
        let n = (bsz * n_heads * t_dec * t_enc) as usize;
        let diff = rel(&g.read(&da, n), &g.read(&db, n));
        println!(
            "{:>2} {:>2} {:>6} {:>6} {:>3} {:>10.3} {:>10} {:>10.3} {:>10} {:>8.1}x {:>10.1e}",
            bsz, n_heads, t_dec, t_enc, hd, ta * 1e3, rows, tb * 1e3, rows * 64, ta / tb, diff
        );
        assert!(diff < TOL, "attn_bwd_dscores_cross b={bsz} h={n_heads} Tdec={t_dec} Tenc={t_enc} hd={hd}: cooperative variant diverges (rel {diff:.2e})");
    }
}
