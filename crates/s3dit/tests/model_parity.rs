// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full Z-Image S³-DiT forward parity vs diffusers (small config).
//!
//! Golden (`tests/golden/zimage_model.safetensors`, committed, baked by
//! `tools/goldens/zimage_model_dump_reference.py`): a small model (dim 48,
//! 2 layers, 1 refiner, cap_feat_dim 16) with random weights + inputs and its
//! reference output. Validates the whole assembly — timestep/x/cap embedders,
//! patchify, noise/context refiners, [image, caption] unified sequence, main
//! layers, FinalLayer, unpatchify — over the (separately bit-exact) block.
//! Self-contained; runs on the CPU backend.

use std::collections::HashMap;

use s3dit::{ZImageConfig, ZImageModel};

/// Resolve a fixture under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
use brain_testutil::testdata;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn zimage_full_model_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_model.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    let st = checkpoint::safetensors::read(&fixture).expect("read model golden");
    let mut weights: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut input: HashMap<String, Vec<f32>> = HashMap::new();
    for t in st {
        if let Some(name) = t.name.strip_prefix('_') {
            input.insert(name.to_string(), t.data);
        } else {
            weights.insert(t.name, (t.shape, t.data));
        }
    }

    let cfg = ZImageConfig {
        dim: 48,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 2,
        cap_feat_dim: 16,
        in_channels: 16,
        patch_size: 2,
        f_patch_size: 1,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 32, 32],
        rope_theta: 256.0,
        t_scale: 1000.0,
        norm_eps: 1e-5,
    };
    let (f, h, w, cap_len) = (1u32, 16u32, 8u32, 32u32);
    let model = ZImageModel::new(cfg, weights, Some("cpu"));

    let got = model.forward(&input["latent"], f, h, w, &input["cap"], cap_len, input["t"][0]);
    let want = &input["out"];
    assert_eq!(got.len(), want.len(), "output len {} != golden {}", got.len(), want.len());

    let cos = cosine(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let want_max = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    eprintln!("Z-Image full-model parity: cosine={cos:.6}  max_abs={max_abs:.5}  (|want|max={want_max:.3})");
    assert!(cos >= 0.9999, "cosine {cos:.6} < 0.9999");
    assert!(max_abs <= 2e-2 * want_max.max(1.0), "max_abs {max_abs:.5} too large");
}
