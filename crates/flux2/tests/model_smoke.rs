// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free forward smoke test at toy dims: exercises every step kind in
//! the DiT graph (double + single blocks, sliced ranges, rope, attention,
//! final head) on the pooled test device. Catches buffer-sizing/binding bugs
//! without the 4B checkpoint.

use flux2::{position_ids, Flux2Config, Flux2Model, Precision};

#[test]
fn tiny_forward_runs_and_is_finite() {
    let cfg = Flux2Config {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 16,
        n_heads: 2,
        depth_double: std::env::var("SMOKE_DBL").map(|v| v.parse().unwrap()).unwrap_or(2),
        depth_single: std::env::var("SMOKE_SGL").map(|v| v.parse().unwrap()).unwrap_or(2),
        axes_dim: [2, 2, 2, 2],
        txt_len: 8,
        ..Flux2Config::klein_4b()
    };
    let mut ts = flux2::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        // small deterministic non-constant fill
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        ts.insert(name, (shape, data));
    }
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let n_img = 4 * 4 + 2 * 2; // gen 4x4 + one 2x2 ref
    let model = Flux2Model::new(&cfg, &ts, gpu, (cfg.txt_len + n_img) as u32);
    let ids = position_ids(cfg.txt_len, 4, 4, &[(2, 2)]);
    let img: Vec<f32> = (0..n_img * cfg.in_channels).map(|i| (i as f32 * 0.7).sin()).collect();
    let ctx: Vec<f32> = (0..cfg.txt_len * cfg.context_in_dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let out = model.forward(&img, &ctx, 0.7, &ids, 16);
    assert_eq!(out.len(), 16 * cfg.in_channels);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    let mean_abs: f32 = out.iter().map(|v| v.abs()).sum::<f32>() / out.len() as f32;
    assert!(mean_abs > 0.0, "all-zero output");
}

/// Int8 smoke at the smallest dims meeting the DP4A slicing alignment
/// (txt_len/hidden/mlp multiples of 64): every quant + sliced-i8-GEMM site
/// runs, output is finite and loosely tracks the fp32 forward. GPU only.
#[test]
fn tiny_forward_int8_runs_and_tracks_fp32() {
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        eprintln!("SKIP: int8 needs a GPU backend, current is {}", gpu.kind());
        return;
    }
    let cfg = Flux2Config {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 64,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        axes_dim: [8, 8, 8, 8],
        txt_len: 64,
        ..Flux2Config::klein_4b()
    };
    let mut ts = flux2::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        ts.insert(name, (shape, data));
    }
    let n_img = 4 * 4 + 2 * 2;
    let n_max = (cfg.txt_len + n_img) as u32;
    let ids = position_ids(cfg.txt_len, 4, 4, &[(2, 2)]);
    let img: Vec<f32> = (0..n_img * cfg.in_channels).map(|i| (i as f32 * 0.7).sin()).collect();
    let ctx: Vec<f32> = (0..cfg.txt_len * cfg.context_in_dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let f32_out = Flux2Model::new(&cfg, &ts, gpu.share(), n_max).forward(&img, &ctx, 0.7, &ids, 16);
    let i8_out = Flux2Model::new_with(&cfg, &ts, gpu, n_max, Precision::Int8).forward(&img, &ctx, 0.7, &ids, 16);
    assert!(i8_out.iter().all(|v| v.is_finite()), "non-finite int8 output");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in i8_out.iter().zip(&f32_out) {
        dot += a as f64 * b as f64;
        na += a as f64 * a as f64;
        nb += b as f64 * b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(cos > 0.9, "int8 tiny forward diverges from fp32: cosine {cos:.4}");
}
