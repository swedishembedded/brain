// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein VAE (`AutoencoderKLFlux2`) parity vs the diffusers reference.
//!
//! Golden fixture (`testdata/flux2/klein-4b/vae.safetensors`, fetched): a fixed
//! image `[3,512,512]` in [-1,1], the encoder posterior `moments` `[64,64,64]`
//! (after `quant_conv`), its `latent_mean` `[32,64,64]`, the packed+normalized
//! DiT latent `latent_packed_norm` `[128,32,32]`, the decode `decoded`
//! `[3,512,512]`, and the checkpoint's `bn_running_{mean,var}` + `bn_eps`.
//! Weights gated on `BRAIN_FLUX2_VAE` (the `vae/` dir or the safetensors file;
//! skips if unset/absent). CPU backend by default; `BRAIN_VAE_DEVICE=gpu` runs
//! it on wgpu.

use std::collections::HashMap;
use std::path::Path;

use vae::{latent, VaeConfig, VaeDecoder, VaeEncoder};

/// Resolve a fixture under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

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
fn flux2_vae_parity() {
    let fixture = testdata("flux2/klein-4b/vae.safetensors");
    if !Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    let vae_env = match std::env::var("BRAIN_FLUX2_VAE") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("SKIP: set BRAIN_FLUX2_VAE to the FLUX.2 Klein vae/ dir");
            return;
        }
    };
    // Accept the vae/ directory (diffusers layout) or the safetensors file.
    let p = Path::new(&vae_env);
    let vae_path = if p.is_dir() { p.join("diffusion_pytorch_model.safetensors") } else { p.to_path_buf() };
    if !vae_path.exists() {
        eprintln!("SKIP: BRAIN_FLUX2_VAE={vae_env} has no vae safetensors");
        return;
    }

    let cfg_path = vae_path.with_file_name("config.json");
    let cfg = VaeConfig::from_json(
        &serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("vae config.json"))
            .expect("parse config.json"),
    );
    assert!(cfg.use_quant_conv && cfg.use_post_quant_conv, "not a FLUX.2 VAE config: {cfg:?}");
    assert_eq!(cfg.latent_channels, 32, "FLUX.2 latent channels");
    assert_eq!(cfg.patch_size, [2, 2], "FLUX.2 patch size");

    let weights = load_tensors(vae_path.to_str().unwrap());
    let fx = load_tensors(&fixture);
    let (ishape, image) = &fx["image"];
    let (mshape, want_moments) = &fx["moments"];
    assert_eq!(ishape, &vec![3, 512, 512], "image shape");
    let (h, w) = (ishape[1] as u32, ishape[2] as u32);
    let (lh, lw) = (mshape[1], mshape[2]);
    assert_eq!(mshape[0], 2 * cfg.latent_channels as usize, "moments channels");

    let device = std::env::var("BRAIN_VAE_DEVICE").unwrap_or_else(|_| "cpu".to_string());

    // 1) Encode: image → posterior moments (encoder conv_out + quant_conv).
    {
        let enc = VaeEncoder::from_diffusers(cfg.clone(), &weights, h, w, Some(&device));
        let got = enc.encode(image);
        assert_eq!(got.len(), want_moments.len(), "moments len ({mshape:?})");
        let cos = cosine(&got, want_moments);
        eprintln!("FLUX.2 VAE encode parity: cosine={cos:.6}");
        assert!(cos >= 0.9999, "moments cosine {cos:.6} < 0.9999");
    }

    // 2) Pack: golden latent_mean → packed+normalized DiT latent, exact math.
    let (_, mean) = &fx["latent_mean"];
    let (_, want_packed) = &fx["latent_packed_norm"];
    let (_, bn_mean) = &fx["bn_running_mean"];
    let (_, bn_var) = &fx["bn_running_var"];
    let eps = fx["bn_eps"].1[0];
    assert_eq!(eps, cfg.batch_norm_eps, "bn eps golden vs config");
    // The checkpoint's own bn stats must match the golden's (same provenance).
    let ck_mean = &weights["bn.running_mean"].1;
    let ck_var = &weights["bn.running_var"].1;
    assert_eq!(ck_mean, bn_mean, "checkpoint bn.running_mean vs golden");
    assert_eq!(ck_var, bn_var, "checkpoint bn.running_var vs golden");
    let packed = latent::pack(mean, cfg.latent_channels as usize, lh, lw, bn_mean, bn_var, eps);
    let max_abs =
        packed.iter().zip(want_packed).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("FLUX.2 VAE pack parity: max_abs={max_abs:.2e}");
    assert!(max_abs < 1e-4, "pack max_abs {max_abs:.2e} >= 1e-4");

    // 3) Decode: unpack(golden packed latent) → post_quant_conv + decoder.
    let z = latent::unpack(want_packed, cfg.latent_channels as usize, lh, lw, bn_mean, bn_var, eps);
    let (dshape, want_dec) = &fx["decoded"];
    let dec = VaeDecoder::from_diffusers(cfg, &weights, lh as u32, lw as u32, Some(&device));
    let got = dec.decode(&z);
    assert_eq!(got.len(), want_dec.len(), "decoded len ({dshape:?})");
    let cos = cosine(&got, want_dec);
    eprintln!("FLUX.2 VAE decode parity: cosine={cos:.6}");
    assert!(cos >= 0.9999, "decoded cosine {cos:.6} < 0.9999");
}
