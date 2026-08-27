// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host trainer forward vs the parity-proven device forward, tiny dims: the
//! f32 instantiation of [`flux2::modelgrad::forward`] (the finetune trainer's
//! math) must reproduce [`flux2::Flux2Model::forward`] (cosine 1.0 vs the
//! diffusers reference at 4B) on the same imported tensors. This transitively
//! validates the host path's fused→split slicing, modulation fold, RoPE
//! tables, and op order against the checked-in device graph. Pooled test
//! device (`BRAIN_DEVICE=cpu` works).
//!
//! Dims are NOT free: the device leg binds each block's modulation slice at a
//! `3 * hidden` float offset, and a storage binding must respect the 256-byte
//! `min_storage_buffer_offset_alignment` (= 64 floats). `hidden: 16` put that
//! offset at 48 floats / 192 bytes and the test failed with a wgpu validation
//! error on every GPU — real dims are fine (klein-4B's `3 * 3072` is 36 864 B),
//! so this was a test-config fault, not a model one. Keep `hidden` and
//! `mlp_hidden` multiples of 64 floats, the same rule `tests/batch_parity.rs`
//! documents and `Flux2Model::forward_batch` asserts.

use flux2::modelgrad::{forward, make_flow_batch, Cfg, ModelWeights};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

#[test]
fn host_f32_forward_matches_device_forward() {
    let fc = flux2::Flux2Config {
        in_channels: 4,
        context_in_dim: 6,
        hidden: 64,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        mlp_ratio: 1.0, // mlp_hidden 64
        axes_dim: [8, 8, 8, 8],
        txt_len: 3,
        ..flux2::Flux2Config::klein_4b()
    };
    let cfg = Cfg::from_flux2(&fc, 2, 2);
    let mut r = rng(0xACC0_5EED);
    let mut ts = flux2::Tensors::new();
    for (name, shape) in fc.tensor_manifest() {
        let n: usize = shape.iter().product();
        let (base, scale) = if name.ends_with("norm.scale") { (1.0, 0.1) } else { (0.0, 0.25) };
        let data: Vec<f32> = (0..n).map(|_| (base + r() * scale) as f32).collect();
        ts.insert(name, (shape, data));
    }

    let x0: Vec<f32> = (0..cfg.n_img() * cfg.in_channels).map(|_| r() as f32).collect();
    let ctx: Vec<f32> = (0..cfg.txt_len * cfg.context_in_dim).map(|_| r() as f32).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|_| r() as f32).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.4, &noise);

    // host f32 trainer path
    // `from_tensors` consumes the map; the device build below needs the fused
    // layout, so the host extraction gets a copy (tiny config).
    let w = ModelWeights::from_tensors(&cfg, &mut ts.clone()).unwrap();
    let (host, _) = forward(&cfg, &w, &b.img, &b.ctx, b.t, &b.cos, &b.sin);

    // device path on the same tensors/inputs
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let model = flux2::Flux2Model::new(&fc, &ts, gpu, cfg.n() as u32);
    let ids = flux2::position_ids(cfg.txt_len, cfg.lh, cfg.lw, &[]);
    let dev = model.forward(&b.img, &b.ctx, b.t as f32, &ids, cfg.n_img());

    assert_eq!(host.len(), dev.len());
    let mut max_abs = 0.0f32;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &d) in host.iter().zip(&dev) {
        max_abs = max_abs.max((a - d).abs());
        dot += a as f64 * d as f64;
        na += a as f64 * a as f64;
        nb += d as f64 * d as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("host-f32 vs device forward: cosine={cos:.7} max_abs={max_abs:.2e}");
    assert!(cos > 0.99999, "host trainer forward diverges from the device forward (cos {cos})");
    assert!(max_abs < 1e-3, "host trainer forward diverges (max abs {max_abs})");
}
