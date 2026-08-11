// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured DiT batch scaling at REAL klein-4B dims — the profile behind the
//! throughput ladder.
//!
//! Times `Flux2Model::forward_batch` at B = 1, 2, 4, 8 for the 512² shape
//! (512 text + 1024 image = 1536 joint tokens per sample), reporting ms per
//! forward and ms **per image**, which is the number that decides whether
//! batching buys anything. Ignored by default: it loads the 4B checkpoint.
//!
//! ```text
//! BRAIN_DEVICE=gpu0 BRAIN_FLUX2_TRANSFORMER=<transformer/> \
//!   cargo test --release -p brain-flux2 --test batch_time -- --ignored --nocapture
//! ```
//! `BRAIN_FLUX2_BATCH_LADDER=1,2,4` overrides the ladder,
//! `BRAIN_FLUX2_BATCH_PRECISION=fp32` the tier (default int8).

use flux2::{position_ids, Flux2Config, Flux2Model, Precision, Sample};

fn load_weights() -> Option<flux2::Tensors> {
    let Ok(dir) = std::env::var("BRAIN_FLUX2_TRANSFORMER") else {
        eprintln!("SKIP: BRAIN_FLUX2_TRANSFORMER unset");
        return None;
    };
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("transformer dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut tensors = Vec::new();
    for f in files {
        tensors.extend(checkpoint::safetensors::read(f.to_str().unwrap()).unwrap());
    }
    Some(flux2::import_diffusers(tensors, &Flux2Config::klein_4b()).unwrap())
}

/// Weight-free roofline probe for the question the ladder raises: does the
/// GEMM itself lose throughput as the batch raises `M`?
///
/// Runs klein-4B's two hot single-block shapes (`[M,3072]·[3072,9216]ᵀ` and
/// `[M,9216]·[9216,3072]ᵀ`) at M = B·1536 for B = 1..8, in isolation, and
/// reports GFLOP/s. If GFLOP/s is flat, batching is neutral in the GEMM and any
/// ladder regression is elsewhere; if it falls, the GEMM is the ceiling.
#[test]
#[ignore = "measurement, not a gate"]
fn gemm_throughput_vs_batch_rows() {
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        eprintln!("SKIP: needs a GPU backend");
        return;
    }
    // resolve `matmul_reg3` the way the model does (by registered name order)
    let k_mm = flux2::KERNELS.iter().position(|(n, _)| *n == "matmul_reg3").expect("matmul_reg3");
    eprintln!("\n   shape (K x N) |     M |   ms | GFLOP/s");
    eprintln!("-----------------+-------+------+--------");
    for (k, n) in [(3072u32, 9216u32), (9216u32, 3072u32)] {
        let w = gpu.storage(k as u64 * n as u64);
        for b in [1u32, 2, 3, 4, 6, 8] {
            let m = b * 1536;
            let x = gpu.storage(m as u64 * k as u64);
            let o = gpu.storage(m as u64 * n as u64);
            let step = gpu.step(k_mm, &[&x, &w, &o], &[m, k, n], m.div_ceil(128) * n.div_ceil(128) * 256);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                gpu.submit(&[], std::slice::from_ref(&step));
                let _ = gpu.read(&o, 1);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            let gf = 2.0 * m as f64 * k as f64 * n as f64 / best / 1e9;
            eprintln!("  {k:5} x {n:5} | {m:5} | {:4.0} | {gf:7.0}", best * 1000.0);
        }
    }
}

#[test]
#[ignore = "loads the 4B checkpoint; measurement, not a gate"]
fn klein_4b_forward_batch_scaling() {
    let ladder: Vec<u32> = std::env::var("BRAIN_FLUX2_BATCH_LADDER")
        .unwrap_or_else(|_| "1,2,4,8".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let precision = match std::env::var("BRAIN_FLUX2_BATCH_PRECISION").as_deref() {
        Ok("fp32") => Precision::F32,
        _ => Precision::Int8,
    };
    let b_max = *ladder.iter().max().unwrap();
    let Some(ts) = load_weights() else { return };
    let cfg = Flux2Config::klein_4b();
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let (lh, lw) = (32usize, 32usize); // 512×512
    let n_img = lh * lw;
    let ids = position_ids(cfg.txt_len, lh, lw, &[]);
    let n_max = cfg.txt_len as u32 + n_img as u32;

    let t0 = std::time::Instant::now();
    let model = Flux2Model::new_batched(&cfg, &ts, gpu, n_max, b_max, precision);
    eprintln!("built {} DiT (b_max {b_max}) in {:.1} s", precision.name(), t0.elapsed().as_secs_f64());
    drop(ts);

    let imgs: Vec<Vec<f32>> = (0..b_max)
        .map(|b| model::hostmath::randn(n_img * cfg.in_channels, 100 + b as u64))
        .collect();
    let ctxs: Vec<Vec<f32>> = (0..b_max)
        .map(|b| model::hostmath::randn(cfg.txt_len * cfg.context_in_dim, 200 + b as u64))
        .collect();

    // Two ladders. `mixed` gives every sample its own timestep (the
    // continuous-batching case: B distinct host modulation solves); `lockstep`
    // gives them all the same one (requests that started together), which the
    // host-side dedup collapses to a single solve. The gap between the two IS
    // the per-forward host conditioning cost.
    let mut rows: Vec<(u32, f64, f64)> = Vec::new();
    for &b in &ladder {
        let mut per = [0.0f64; 2];
        for (mode, slot) in [(0usize, 0usize), (1, 1)] {
            let samples: Vec<Sample<'_>> = (0..b as usize)
                .map(|i| Sample {
                    img_tokens: &imgs[i],
                    ctx: &ctxs[i],
                    t: if mode == 0 { 0.9 - 0.1 * i as f32 } else { 0.9 },
                })
                .collect();
            let _ = model.forward_batch(&samples, &ids, n_img); // warm
            // MIN of N, not mean: this box shares its GPUs, and an interfering
            // process can only ever make a sample slower. The minimum is the
            // uncontended time; a mean silently reports the neighbour's load.
            let reps = std::env::var("BRAIN_FLUX2_BATCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                let _ = model.forward_batch(&samples, &ids, n_img);
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            per[slot] = best;
        }
        rows.push((b, per[0], per[1]));
    }
    eprintln!("\n  B | mixed-t ms/fwd | ms/image | lockstep ms/fwd | ms/image | speedup vs B=1 (mixed / lockstep)");
    eprintln!("----+----------------+----------+-----------------+----------+---------------------------------");
    let (b0, m0, l0) = rows[0];
    let (bm0, bl0) = (m0 / b0 as f64, l0 / b0 as f64);
    for &(b, m, l) in &rows {
        let (mi, li) = (m / b as f64, l / b as f64);
        eprintln!("{b:3} | {m:14.1} | {mi:8.1} | {l:15.1} | {li:8.1} | {:.3}x / {:.3}x", bm0 / mi, bl0 / li);
    }
}
