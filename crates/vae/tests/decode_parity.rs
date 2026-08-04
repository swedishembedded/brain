// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image VAE (`AutoencoderKL`) decode parity vs the diffusers reference.
//!
//! Golden fixture (`testdata/golden/vae/zimage_vae_decode.safetensors`, fetched):
//! a fixed latent `[1,16,8,8]` and diffusers' `vae.decode(z).sample`
//! `[1,3,64,64]`, baked by `tools/goldens/vae_dump_reference.py`.
//! The 168 MB reference weights are NOT committed — point `BRAIN_ZIMAGE_VAE` at
//! `.../z-image/weights/vae/diffusion_pytorch_model.safetensors` (a default
//! resources path is tried); the test skips if absent (like brain's other
//! weight-gated tests). Runs on the CPU backend (deterministic, no GPU needed).

use std::collections::HashMap;
use std::path::Path;

use vae::{VaeConfig, VaeDecoder};

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

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    // Reference is ~[-1,1] → peak-to-peak ≈ 2.
    10.0 * (4.0f64 / mse).log10()
}

#[test]
fn zimage_vae_decode_matches_diffusers() {
    let fixture = testdata("golden/vae/zimage_vae_decode.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    let vae_path = match std::env::var("BRAIN_ZIMAGE_VAE") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("SKIP: set BRAIN_ZIMAGE_VAE to the Z-Image vae/ safetensors");
            return;
        }
    };
    if !Path::new(&vae_path).exists() {
        eprintln!("SKIP: BRAIN_ZIMAGE_VAE={vae_path} not found");
        return;
    }

    let cfg_path = Path::new(&vae_path).with_file_name("config.json");
    let cfg = VaeConfig::from_json(
        &serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("vae config.json"))
            .expect("parse config.json"),
    );

    let weights = load_tensors(&vae_path);
    let fixture = load_tensors(&fixture);
    let (lshape, latent) = &fixture["latent"];
    let (ishape, want) = &fixture["image"];
    assert_eq!(lshape, &vec![1, cfg.latent_channels as usize, 8, 8], "latent shape");
    let (h, w) = (lshape[2] as u32, lshape[3] as u32);

    // Backend selectable for cross-backend parity (default CPU: deterministic,
    // needs no GPU). `BRAIN_VAE_DEVICE=gpu` runs it on wgpu/Vulkan (the P40s).
    let device = std::env::var("BRAIN_VAE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let dec = VaeDecoder::from_diffusers(cfg, &weights, h, w, Some(&device));
    let got = dec.decode(latent);

    assert_eq!(got.len(), want.len(), "output {} != golden {} ({ishape:?})", got.len(), want.len());
    let cos = cosine(&got, want);
    let db = psnr(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("VAE decode parity: cosine={cos:.6}  PSNR={db:.2}dB  max_abs={max_abs:.5}");
    assert!(cos >= 0.9990, "cosine {cos:.6} < 0.999");
    assert!(db >= 40.0, "PSNR {db:.2}dB < 40");
}
