// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image VAE (`AutoencoderKL`) ENCODE parity vs the diffusers reference.
//!
//! Golden fixture (`testdata/golden/vae/zimage_vae_encode.safetensors`, fetched): a
//! fixed image `[1,3,64,64]` and diffusers' `vae.encode(x).latent_dist`
//! parameters `[1,32,8,8]` (mean‖logvar) + `mean` `[1,16,8,8]`. Unlike
//! `decode_parity.rs`'s golden, this one has **no in-repo generator** —
//! `tools/goldens/vae_dump_reference.py` only bakes the decode direction; a
//! `<something>_encode` counterpart does not exist in this repo. This VAE has no
//! `quant_conv`, so the posterior parameters are exactly the encoder `conv_out`
//! that brain's `VaeEncoder` returns. Weights gated on `BRAIN_S3DIT_VAE` (skips
//! if absent). CPU backend by default; `BRAIN_VAE_DEVICE=gpu` runs it on wgpu.

use std::collections::HashMap;
use std::path::Path;

use vae::{VaeConfig, VaeEncoder};

/// Resolve a fixture under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
use brain_testutil::testdata;

fn load_tensors(path: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    checkpoint::safetensors::read(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect()
}

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
fn zimage_vae_encode_matches_diffusers() {
    let fixture = testdata("golden/vae/zimage_vae_encode.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        brain_testutil::skip(&format!("fixture {fixture} absent - run `make fetch/testdata`"));
        return;
    }
    let vae_path = match std::env::var("BRAIN_S3DIT_VAE") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            brain_testutil::skip("set BRAIN_S3DIT_VAE to the Z-Image vae/ safetensors");
            return;
        }
    };
    if !Path::new(&vae_path).exists() {
        brain_testutil::skip(&format!("BRAIN_S3DIT_VAE={vae_path} not found"));
        return;
    }

    let cfg_path = Path::new(&vae_path).with_file_name("config.json");
    let cfg = VaeConfig::from_json(
        &serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("vae config.json"))
            .expect("parse config.json"),
    );

    let weights = load_tensors(&vae_path);
    let fixture = load_tensors(&fixture);
    let (ishape, image) = &fixture["image"];
    let (mshape, want_moments) = &fixture["moments"];
    assert_eq!(ishape, &vec![1, cfg.in_channels as usize, 64, 64], "image shape");
    let (h, w) = (ishape[2] as u32, ishape[3] as u32);
    let (lh, lw) = (mshape[2] as u32, mshape[3] as u32);

    let device = std::env::var("BRAIN_VAE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let enc = VaeEncoder::from_diffusers(cfg.clone(), &weights, h, w, Some(&device));
    let got = enc.encode(image); // [32, lh, lw] moments

    assert_eq!(got.len(), want_moments.len(), "moments {} != golden {} ({mshape:?})", got.len(), want_moments.len());
    let cos = cosine(&got, want_moments);
    let max_abs = got.iter().zip(want_moments).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let want_max = want_moments.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    eprintln!("VAE encode parity: cosine={cos:.6}  max_abs={max_abs:.5}  (golden max |v|={want_max:.3})");

    // The mean channels (used by image-conditioned generation) must also match.
    let (_, want_mean) = &fixture["mean"];
    let mean = enc.encode_mean(image, lh, lw);
    let cos_mean = cosine(&mean, want_mean);
    eprintln!("VAE encode MEAN parity: cosine={cos_mean:.6}");

    assert!(cos >= 0.9990, "moments cosine {cos:.6} < 0.999");
    assert!(cos_mean >= 0.9990, "mean cosine {cos_mean:.6} < 0.999");
    assert!(max_abs <= 0.05 * want_max, "max_abs {max_abs:.5} > 5% of golden range");
}
