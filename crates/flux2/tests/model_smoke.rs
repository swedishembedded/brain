// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free forward smoke test at toy dims: exercises every step kind in
//! the DiT graph (double + single blocks, sliced ranges, rope, attention,
//! final head) on the pooled test device. Catches buffer-sizing/binding bugs
//! without the 4B checkpoint.

use flux2::{position_ids, Flux2Config, Flux2Model};

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
