// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real Z-Image-Turbo (6B) DiT forward parity vs diffusers, on the shipped
//! weights.
//!
//! Golden (`tests/golden/zimage_real.safetensors`, committed — small: inputs +
//! output only): a forward of the turbo-config model loaded from the real Comfy
//! weights, baked by `tools/goldens/zimage_real_dump_reference.py`. brain
//! imports the SAME weights (`import_comfy`) and must match. The 12 GB weights
//! are NOT committed — set `BRAIN_ZIMAGE_DIT` (a resources default is tried);
//! skips if absent. Heavy (loads ~24 GB fp32, 30-layer CPU forward).

use std::path::Path;

use zimage::{import::import_comfy, ZImageConfig, ZImageDitI8, ZImageDitShard, ZImageModel};

/// Resolve a fixture under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
use brain_testutil::testdata;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    model::hostmath::cosine(a, b) as f64
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
    let fixture = testdata("golden/zimage/zimage_real.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
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
    let fx = checkpoint::safetensors::read(&fixture).expect("read real golden");
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

/// 2-GPU sharded forward on the real 6B weights, matched against the diffusers
/// golden. Validates that splitting the stack across both P40s (with the
/// host-staged residual at the cut) is numerically correct — not just that it
/// runs. Needs BOTH cards + BRAIN_ZIMAGE_DIT + BRAIN_ZIMAGE_SHARD=1; skips
/// otherwise (it allocates ~24 GB per card).
#[test]
fn zimage_shard_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_real.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    if std::env::var("BRAIN_ZIMAGE_SHARD").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_ZIMAGE_SHARD=1 (+ BRAIN_ZIMAGE_DIT, 2 GPUs) to run the 2-GPU shard parity");
        return;
    }
    let dit = match std::env::var("BRAIN_ZIMAGE_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    let fx = checkpoint::safetensors::read(&fixture).expect("read real golden");
    let g = |n: &str| &fx.iter().find(|t| t.name == n).unwrap().data;
    let (latent, cap, tt, want) = (g("_latent"), g("_cap"), g("_t"), g("_out"));

    let cfg = ZImageConfig::turbo();
    let weights = import_comfy(checkpoint::safetensors::read(&dit).expect("read DiT"), &cfg);
    // Golden latent is 16×16 (H=W=16) with 32 caption tokens.
    let shard = ZImageDitShard::build(cfg, weights, 1, 16, 16, 32);
    let got = shard.forward(latent, cap, tt[0]);

    assert_eq!(got.len(), want.len());
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    let rl2 = (num / den).sqrt();
    eprintln!("Z-Image 2-GPU SHARD parity: rel_l2={rl2:.5}  max_abs={max_abs:.4}");
    assert!(rl2 <= 0.03, "shard rel_l2 {rl2:.5} > 0.03");
}

/// int8 (DP4A) single-GPU forward on the real 6B weights vs the diffusers golden.
/// Per-tensor int8 across 34 blocks accumulates error, so the gate is cosine
/// (structure preserved), not bit-exact. BRAIN_ZIMAGE_I8=1 + BRAIN_ZIMAGE_DIT.
#[test]
fn zimage_int8_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_real.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    if std::env::var("BRAIN_ZIMAGE_I8").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_ZIMAGE_I8=1 (+ BRAIN_ZIMAGE_DIT, GPU) for the int8 parity test");
        return;
    }
    let dit = match std::env::var("BRAIN_ZIMAGE_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    let fx = checkpoint::safetensors::read(&fixture).expect("read real golden");
    let g = |n: &str| &fx.iter().find(|t| t.name == n).unwrap().data;
    let (latent, cap, tt, want) = (g("_latent"), g("_cap"), g("_t"), g("_out"));

    let cfg = ZImageConfig::turbo();
    let weights = import_comfy(checkpoint::safetensors::read(&dit).expect("read DiT"), &cfg);
    let i8 = ZImageDitI8::build(cfg, weights, 1, 16, 16, 32);
    let got = i8.forward(latent, cap, tt[0]);

    let cos = cosine(&got, want);
    let rl2 = rel_l2(&got, want);
    eprintln!("Z-Image int8 (1 GPU) parity: cosine={cos:.5}  rel_l2={rl2:.4}");
    assert!(cos >= 0.99, "int8 cosine {cos:.5} < 0.99");
}
