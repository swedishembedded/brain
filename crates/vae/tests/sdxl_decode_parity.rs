// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL VAE (`AutoencoderKL`, 4 latent channels) decode parity vs diffusers.
//!
//! Why this file exists, separately from `decode_parity.rs`: that gate covers
//! Z-Image's VAE, which is a *16*-channel `AutoencoderKL`, and it skips unless
//! `BRAIN_ZIMAGE_VAE` is set — so on this machine, and on CI, **nothing gated a
//! VAE decode at all**. `crates/unet`'s SDXL pipeline shipped a decode that
//! produced structurally-correct but visibly corrupted pictures, and no test
//! could see it: the UNet's own 165-tap parity is green, and a
//! gradient-magnitude sanity check on the output passes on a broken image.
//!
//! Golden fixture: `testdata/golden/vae/sdxl_vae_decode.safetensors`, a fixed
//! `[1,4,16,16]` latent and diffusers' `vae.decode(z).sample` `[1,3,128,128]`,
//! baked by `tools/goldens/sdxl_dump_vae_decode.py` from the released fp16 weights
//! upcast to fp32 (which is what brain loads).
//!
//! Run:
//!   BRAIN_SDXL=/path/to/stable-diffusion-xl-base-1.0 \
//!     cargo test --release -p brain-vae --test sdxl_decode_parity -- --nocapture

use std::collections::HashMap;
use std::path::Path;

use vae::{VaeConfig, VaeDecoder};

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

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 =
        a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    // Reference is ~[-1,1] → peak-to-peak ≈ 2.
    10.0 * (4.0f64 / mse).log10()
}

#[test]
fn sdxl_vae_decode_matches_diffusers() {
    let fixture = testdata("golden/vae/sdxl_vae_decode.safetensors");
    if !Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata` (or tools/goldens/sdxl_dump_vae_decode.py)");
        return;
    }
    let Ok(root) = std::env::var("BRAIN_SDXL") else {
        eprintln!("SKIP: BRAIN_SDXL unset (point it at stable-diffusion-xl-base-1.0)");
        return;
    };
    let vae_dir = Path::new(&root).join("vae");
    if !vae_dir.exists() {
        eprintln!("SKIP: {} absent", vae_dir.display());
        return;
    }

    let cfg = VaeConfig::from_json(
        &serde_json::from_str(
            &std::fs::read_to_string(vae_dir.join("config.json")).expect("vae config.json"),
        )
        .expect("parse config.json"),
    );
    assert_eq!(cfg.latent_channels, 4, "SDXL's AutoencoderKL is 4-channel");

    // Load whichever weight file the release ships (it is `*.fp16.safetensors`
    // for SDXL base); brain upcasts F16 to f32 on read.
    let wf = std::fs::read_dir(&vae_dir)
        .expect("read vae dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .expect("a .safetensors in vae/");
    let weights = load_tensors(wf.to_str().unwrap());

    let fixture = load_tensors(&fixture);
    let (lshape, latent) = &fixture["z"];
    let (ishape, want) = &fixture["decoded"];
    assert_eq!(lshape, &vec![1, 4, 16, 16], "latent shape");
    let (h, w) = (lshape[2] as u32, lshape[3] as u32);

    let device = std::env::var("BRAIN_VAE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let dec = VaeDecoder::from_diffusers(cfg, &weights, h, w, Some(&device));
    let got = dec.decode(latent);

    assert_eq!(got.len(), want.len(), "output {} != golden {} ({ishape:?})", got.len(), want.len());
    let cos = cosine(&got, want);
    let db = psnr(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (gmin, gmax) = got.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let (wmin, wmax) = want.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    eprintln!("SDXL VAE decode: cosine={cos:.6}  PSNR={db:.2}dB  max_abs={max_abs:.5}");
    eprintln!("  range got [{gmin:.4}, {gmax:.4}]  want [{wmin:.4}, {wmax:.4}]");
    assert!(cos >= 0.9990, "cosine {cos:.6} < 0.999");
    assert!(db >= 40.0, "PSNR {db:.2}dB < 40");
}
