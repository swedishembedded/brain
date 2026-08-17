// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free forward smoke test at toy dims: exercises every step kind in the
//! FLUX.1 graph (device modulation GEMVs, per-stream sliced ranges, the
//! LN→film→gate modulation chain, RoPE, joint attention, GELU MLPs, the
//! column-split `linear2`, the final head) on the pooled test device. Catches
//! buffer-sizing/binding bugs without the 12 B checkpoint.
//!
//! `SMOKE_DBL` / `SMOKE_SGL` override the block counts and `SMOKE_STEPS=k`
//! submits only the first `k` steps — the bisection handles the playbook §4
//! prescribes.

use flux1::{position_ids, Flux1Config, Flux1Model, Precision};

/// Toy config whose every sliced width is a multiple of 64 floats (the
/// 256-byte `min_storage_buffer_offset_alignment`).
fn tiny() -> Flux1Config {
    Flux1Config {
        in_channels: 64,
        context_in_dim: 128,
        vec_in_dim: 64,
        hidden: 128,
        n_heads: 2,
        depth_double: std::env::var("SMOKE_DBL").map(|v| v.parse().unwrap()).unwrap_or(2),
        depth_single: std::env::var("SMOKE_SGL").map(|v| v.parse().unwrap()).unwrap_or(2),
        axes_dim: [8, 28, 28],
        ..Flux1Config::dev()
    }
}

fn fake_tensors(cfg: &Flux1Config) -> flux1::Tensors {
    let mut ts = flux1::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        ts.insert(name, (shape, data));
    }
    ts
}

fn run(cfg: &Flux1Config, precision: Precision) -> Vec<f32> {
    let gpu = gpu_core::testgpu::dev(flux1::KERNELS);
    let nt = 64usize;
    let n_img = 4 * 4 + 2 * 2; // generated 4x4 + one 2x2 Kontext reference
    let model =
        Flux1Model::new_with(cfg, &fake_tensors(cfg), gpu, (nt + n_img) as u32, precision);
    let ids = position_ids(nt, 4, 4, &[(2, 2)]);
    let img: Vec<f32> = (0..n_img * cfg.in_channels).map(|i| (i as f32 * 0.7).sin()).collect();
    let ctx: Vec<f32> = (0..nt * cfg.context_in_dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let pooled: Vec<f32> = (0..cfg.vec_in_dim).map(|i| (i as f32 * 0.11).sin()).collect();
    // n_pred = the noise span only: the 4x4 generated tokens, not the reference
    model.forward(&img, &ctx, &pooled, 0.7, 3.5, &ids, 16)
}

#[test]
fn tiny_forward_runs_and_is_finite() {
    let cfg = tiny();
    let out = run(&cfg, Precision::F32);
    assert_eq!(out.len(), 16 * cfg.in_channels);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    let mean_abs: f32 = out.iter().map(|v| v.abs()).sum::<f32>() / out.len() as f32;
    assert!(mean_abs > 0.0, "all-zero output");
}

/// The traced forward must produce the same prediction as the plain one — the
/// trace only inserts submit/readback boundaries, never changes the math.
#[test]
fn traced_forward_matches_and_taps_every_block() {
    let cfg = tiny();
    let gpu = gpu_core::testgpu::dev(flux1::KERNELS);
    let nt = 64usize;
    let n_img = 4 * 4 + 2 * 2;
    let model = Flux1Model::new(&cfg, &fake_tensors(&cfg), gpu, (nt + n_img) as u32);
    let ids = position_ids(nt, 4, 4, &[(2, 2)]);
    let img: Vec<f32> = (0..n_img * cfg.in_channels).map(|i| (i as f32 * 0.7).sin()).collect();
    let ctx: Vec<f32> = (0..nt * cfg.context_in_dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let pooled: Vec<f32> = (0..cfg.vec_in_dim).map(|i| (i as f32 * 0.11).sin()).collect();
    let plain = model.forward(&img, &ctx, &pooled, 0.7, 3.5, &ids, 16);
    let (traced, tr) = model.forward_traced(&img, &ctx, &pooled, 0.7, 3.5, &ids, 16);
    assert_eq!(tr.stages.len(), cfg.depth_double + cfg.depth_single);
    assert_eq!(tr.pre_final.len(), n_img * cfg.hidden);
    assert_eq!(tr.temb.len(), cfg.hidden);
    let max_abs =
        plain.iter().zip(&traced).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert_eq!(max_abs, 0.0, "tracing perturbed the forward");
}

/// Int8 smoke: every quant + sliced-DP4A-GEMM site runs, including the 77
/// `m = 1` modulation GEMVs, and the output loosely tracks fp32. GPU only.
#[test]
fn tiny_forward_int8_runs_and_tracks_fp32() {
    let gpu = gpu_core::testgpu::dev(flux1::KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable(&format!("int8 needs a GPU backend, current is {}", gpu.kind()));
        return;
    }
    drop(gpu);
    let cfg = tiny();
    let a = run(&cfg, Precision::F32);
    let b = run(&cfg, Precision::Int8);
    assert!(b.iter().all(|v| v.is_finite()), "non-finite int8 output");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("int8 vs fp32 toy cosine {cos:.6}");
    assert!(cos > 0.9, "int8 toy forward diverged: cosine {cos:.6}");
}
