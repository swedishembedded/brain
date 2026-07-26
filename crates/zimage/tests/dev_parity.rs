// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-resident ZImageDit parity vs the reference ZImageModel (and thus
//! diffusers) on the small full-model golden. Confirms the resident-graph
//! forward is numerically identical to the per-block reference.

use std::collections::HashMap;

use zimage::{ZImageConfig, ZImageDit};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/zimage_model.safetensors");

fn small_cfg() -> ZImageConfig {
    ZImageConfig {
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
    }
}

#[test]
fn zimage_dev_matches_reference_golden() {
    let st = checkpoint::safetensors::read(GOLDEN).expect("read model golden");
    let mut weights: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut input: HashMap<String, Vec<f32>> = HashMap::new();
    for t in st {
        if let Some(name) = t.name.strip_prefix('_') {
            input.insert(name.to_string(), t.data);
        } else {
            weights.insert(t.name, (t.shape, t.data));
        }
    }

    let dit = ZImageDit::build(small_cfg(), weights, 1, 16, 8, 32, Some("cpu"));
    let got = dit.forward(&input["latent"], &input["cap"], input["t"][0]);
    let want = &input["out"];
    assert_eq!(got.len(), want.len());
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let want_max = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    eprintln!("ZImageDit (device-resident) parity: max_abs={max_abs:.6} (|want|max={want_max:.3})");
    assert!(max_abs <= 1e-3 * want_max.max(1.0), "max_abs {max_abs:.6} too large");
}
