// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real Z-Image-Turbo (6B) DiT forward parity vs diffusers, on the shipped
//! weights.
//!
//! Golden (`tests/golden/zimage_real.safetensors`, committed - small: inputs +
//! output only): a forward of the turbo-config model loaded from the real Comfy
//! weights, baked by `tools/goldens/s3dit_real_dump_reference.py`. brain
//! imports the SAME weights (`import_comfy`) and must match. The 12 GB weights
//! are NOT committed - set `BRAIN_S3DIT_DIT` (a resources default is tried);
//! skips if absent. Heavy (loads ~24 GB fp32, 30-layer CPU forward).

use std::path::Path;

use s3dit::{import::import_comfy, ZImageConfig, ZImageDitI8, ZImageDitShard, ZImageModel};

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
        brain_testutil::skip(&format!("fixture {fixture} absent - run `make fetch/testdata`"));
        return;
    }
    let dit = match std::env::var("BRAIN_S3DIT_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            brain_testutil::skip("set BRAIN_S3DIT_DIT to the z_image_turbo_bf16 safetensors");
            return;
        }
    };
    if !Path::new(&dit).exists() {
        brain_testutil::skip(&format!("BRAIN_S3DIT_DIT={dit} not found"));
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

/// Read DiT weights from `path`: a single (Comfy) safetensors file, or an
/// HF-style directory (already diffusers-named, sharded or not) - the same
/// leniency `s3dit::pipeline::read_component_tensors` gives the real CLI, so
/// this test loads a directly-fetched `Tongyi-MAI/Z-Image-Turbo` tree exactly
/// like `brain s3dit text2image` does, with no manual repacking.
fn read_dit_tensors(path: &str) -> Vec<checkpoint::safetensors::StTensor> {
    let p = Path::new(path);
    if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p).expect("read DiT weights dir")
    } else {
        checkpoint::safetensors::read(path).expect("read DiT weights")
    }
}

/// Real Z-Image-Turbo forward at the CLI's actual 512x512 generation scale:
/// 1024 image tokens + 64 caption tokens = 1088 joint tokens through
/// `layers.*` (vs [`zimage_real_dit_matches_diffusers`]'s 64+32=96-token
/// toy scale). This size is deliberate, not padding for its own sake: a
/// `flip_sin_to_cos` sign/order bug in the timestep-embedding conditioning
/// (fixed alongside this test) corrupted every block's adaLN modulation
/// globally, at every resolution - but its effect on the final cosine was
/// small enough at the 96-token toy scale to pass `cos >= 0.999` there,
/// while at this realistic scale it collapsed `layers.0`'s own output to
/// cosine ~0.80. Catching a regression like that requires running parity at
/// (approximately) the scale real generations actually use, not just a small
/// smoke shape.
///
/// CPU, like [`zimage_real_dit_matches_diffusers`] (not GPU): the bug this
/// guards is host-side math (`crate::model::timestep_embedding`), identical
/// on every device, and `ZImageModel::forward` builds one fresh `Gpu` per
/// block (34 for a full forward) - fine on CPU, but on this box's wgpu/Vulkan
/// backend that much create/destroy churn against the SAME physical adapter
/// intermittently drops to a fallback software adapter with a much smaller
/// binding limit and hard-faults on these tensor sizes, unrelated to
/// anything this test is meant to check. Heavy (30-layer CPU forward at
/// 1088 tokens); skips without the fixture/weights.
#[test]
fn zimage_real_dit_matches_diffusers_at_512() {
    let fixture = testdata("golden/zimage/zimage_real_512.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        brain_testutil::skip(&format!("fixture {fixture} absent - run `tools/goldens/s3dit_real_512_dump_reference.py`"));
        return;
    }
    let dit = match std::env::var("BRAIN_S3DIT_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            brain_testutil::skip("set BRAIN_S3DIT_DIT to the Z-Image-Turbo transformer weights (file or HF dir)");
            return;
        }
    };
    if !Path::new(&dit).exists() {
        brain_testutil::skip(&format!("BRAIN_S3DIT_DIT={dit} not found"));
        return;
    }

    let fx = checkpoint::safetensors::read(&fixture).expect("read real 512 golden");
    let g = |n: &str| &fx.iter().find(|t| t.name == n).unwrap().data;
    let (latent, cap, tt, want) = (g("_latent"), g("_cap"), g("_t"), g("_out"));

    let cfg = ZImageConfig::turbo();
    let weights = import_comfy(read_dit_tensors(&dit), &cfg);
    let model = ZImageModel::new(cfg, weights, Some("cpu"));
    // latent [16,1,64,64] -> 32x32 = 1024 image patches; cap_len=64 -> 1088
    // joint tokens, the real 512x512 CLI shape.
    let got = model.forward(latent, 1, 64, 64, cap, 64, tt[0]);

    assert_eq!(got.len(), want.len(), "output len {} != golden {}", got.len(), want.len());
    let cos = cosine(&got, want);
    let rl2 = rel_l2(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("Z-Image REAL 6B DiT parity @512x512 (1088 tok): cosine={cos:.6}  rel_l2={rl2:.5}  max_abs={max_abs:.4}");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
    assert!(rl2 <= 0.15, "rel_l2 {rl2:.5} > 0.15");
}

/// 2-GPU sharded forward on the real 6B weights, matched against the diffusers
/// golden. Validates that splitting the stack across both P40s (with the
/// host-staged residual at the cut) is numerically correct - not just that it
/// runs. Needs BOTH cards + BRAIN_S3DIT_DIT + BRAIN_S3DIT_SHARD=1; skips
/// otherwise (it allocates ~24 GB per card).
#[test]
fn zimage_shard_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_real.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        brain_testutil::skip(&format!("fixture {fixture} absent - run `make fetch/testdata`"));
        return;
    }
    if std::env::var("BRAIN_S3DIT_SHARD").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_S3DIT_SHARD=1 (+ BRAIN_S3DIT_DIT, 2 GPUs) to run the 2-GPU shard parity");
        return;
    }
    let dit = match std::env::var("BRAIN_S3DIT_DIT") {
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
/// (structure preserved), not bit-exact. BRAIN_S3DIT_I8=1 + BRAIN_S3DIT_DIT.
#[test]
fn zimage_int8_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_real.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        brain_testutil::skip(&format!("fixture {fixture} absent - run `make fetch/testdata`"));
        return;
    }
    if std::env::var("BRAIN_S3DIT_I8").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_S3DIT_I8=1 (+ BRAIN_S3DIT_DIT, GPU) for the int8 parity test");
        return;
    }
    let dit = match std::env::var("BRAIN_S3DIT_DIT") {
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
