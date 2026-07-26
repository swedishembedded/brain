// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real Z-Image-Turbo (6B) DiT forward parity vs diffusers, on the shipped
//! weights.
//!
//! Golden (`tests/golden/zimage_real.safetensors`, committed — small: inputs +
//! output only): a forward of the turbo-config model loaded from the real Comfy
//! weights, baked by `resources/image-models/_goldens/gen_zimage_real.py`. brain
//! imports the SAME weights (`import_comfy`) and must match. The 12 GB weights
//! are NOT committed — set `BRAIN_ZIMAGE_DIT` (a resources default is tried);
//! skips if absent. Heavy (loads ~24 GB fp32, 30-layer CPU forward).

use std::path::Path;

use zimage::{import::import_comfy, ZImageConfig, ZImageModel};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/zimage_real.safetensors");

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

#[test]
fn zimage_real_dit_matches_diffusers() {
    let dit = match std::env::var("BRAIN_ZIMAGE_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("SKIP: set BRAIN_ZIMAGE_DIT to the z_image_turbo_bf16 safetensors");
            return;
        }
    };
    if !Path::new(&dit).exists() {
        eprintln!("SKIP: BRAIN_ZIMAGE_DIT={dit} not found");
        return;
    }
    let fx = checkpoint::safetensors::read(GOLDEN).expect("read real golden");
    let g = |n: &str| &fx.iter().find(|t| t.name == n).unwrap().data;
    let (latent, cap, tt, want) = (g("_latent"), g("_cap"), g("_t"), g("_out"));

    let cfg = ZImageConfig::turbo();
    let tensors = checkpoint::safetensors::read(&dit).expect("read DiT weights");
    let weights = import_comfy(tensors, &cfg);
    let model = ZImageModel::new(cfg, weights, Some("cpu"));
    let got = model.forward(latent, 1, 16, 16, cap, 32, tt[0]);

    assert_eq!(got.len(), want.len(), "output len {} != golden {}", got.len(), want.len());
    let cos = cosine(&got, want);
    let rl2 = rel_l2(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("Z-Image REAL 6B DiT parity: cosine={cos:.6}  rel_l2={rl2:.5}  max_abs={max_abs:.4}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
    assert!(rl2 <= 0.03, "rel_l2 {rl2:.5} > 0.03");
}
