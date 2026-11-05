// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An IC-LoRA reference video reaches generation: encoded through the real
//! video VAE at the stage's resolution and appended as clean tokens beside
//! the clip being generated, with the geometry the adapter declares in its
//! own metadata - and refused, by name, wherever it cannot apply.
//!
//! The DiT here is the tiny random-weight one (the reference's tokens change
//! its output exactly as they would the real one's); the VAE is the real
//! checkpoint from the model store, so the test skips without it.

use std::path::{Path, PathBuf};

use ltxv::lora::reference_scale_factors;
use ltxv::pipeline::{generate, GenOpts, Paths, Video, VideoReference};
use model::adapter::AdapterSpec;

fn vae_path() -> Option<String> {
    let p = format!("{}/ltx-2.5-video-vae-conv-bf16.safetensors", brain_testutil::model_dir("Lightricks/LTX-2.5")?);
    Path::new(&p).exists().then_some(p)
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ltxv-refvid-{tag}-{}-{}", std::process::id(), data::rng::random_seed()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A raw safetensors file: `tensors` as f32 plus free-form `metadata`, the
/// way an IC-LoRA records its reference geometry.
fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)], metadata: &[(&str, &str)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let end = offset + data.len() * 4;
        header.insert(name.to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}));
        offset = end;
    }
    let meta: serde_json::Map<String, serde_json::Value> = metadata.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect();
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let h = serde_json::to_vec(&header).unwrap();
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    for (_, _, data) in tensors {
        bytes.extend(data.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(path, bytes).unwrap();
}

/// An adapter over the tiny DiT's first block, declaring `metadata`.
fn adapter(dir: &Path, name: &str, metadata: &[(&str, &str)]) -> AdapterSpec {
    let d = ltxv::LtxDitConfig::tiny().inner_dim as usize;
    let mut rng = data::rng::Lcg::new(31);
    let path = dir.join(name);
    write_safetensors(
        &path,
        &[
            ("diffusion_model.transformer_blocks.0.attn1.to_v.lora_A.weight", vec![4, d], rng.vec_scaled(4 * d, 0.5)),
            ("diffusion_model.transformer_blocks.0.attn1.to_v.lora_B.weight", vec![d, 4], rng.vec_scaled(d * 4, 0.5)),
        ],
        metadata,
    );
    AdapterSpec { path: path.to_string_lossy().into_owned(), scale: 1.0 }
}

fn clip(frames: usize, side: u32, seed: u64) -> Video {
    let mut rng = data::rng::Lcg::new(seed);
    let px = (side * side * 3) as usize;
    let frames = (0..frames).map(|_| rng.vec_scaled(px, 1.0).iter().map(|v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8).collect()).collect();
    Video { width: side, height: side, fps: 8, frames, audio: None }
}

fn opts(adapters: Vec<AdapterSpec>, reference: Option<VideoReference>) -> GenOpts {
    GenOpts { seed: 7, eta: 0.0, adapters, reference, device: Some("cpu".into()), ..GenOpts::default() }
}

fn run(paths: &Paths, o: &GenOpts) -> Result<Vec<Vec<u8>>, String> {
    generate(paths, "a red ball", o, &Default::default(), |_, _, _| {}).map(|(v, _)| v.frames)
}

/// The reference's tokens change what the in-context stage generates, and
/// two different references give two different clips - while the clip's own
/// shape is untouched.
#[test]
fn a_reference_video_reaches_the_in_context_stage() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    let dir = tempdir("reaches");
    let a = adapter(&dir, "ic.safetensors", &[]);
    let without = run(&paths, &opts(vec![a.clone()], None)).unwrap();
    let with_a = run(&paths, &opts(vec![a.clone()], Some(VideoReference::new(clip(9, 64, 1), 1.0)))).unwrap();
    let with_b = run(&paths, &opts(vec![a], Some(VideoReference::new(clip(9, 64, 2), 1.0)))).unwrap();
    assert_eq!(with_a.len(), without.len(), "the reference must not change the clip's own length");
    assert_eq!(with_a[0].len(), without[0].len(), "nor its frame size");
    assert_ne!(with_a, without, "the reference tokens must reach the model");
    assert_ne!(with_a, with_b, "a different reference must give a different clip");
}

/// A constant `side`x`side` mask video of `frames` frames at gray level `v`.
fn mask(frames: usize, side: u32, v: u8) -> Video {
    Video { width: side, height: side, fps: 8, frames: vec![vec![v; (side * side * 3) as usize]; frames], audio: None }
}

/// The attention strength cuts the reference off from the generated tokens:
/// at 0 the clip is the clip generated with no reference at all, and between
/// 0 and 1 it is neither that nor the fully attended one. A white mask is the
/// strength alone, a black one cuts it off like strength 0.
#[test]
fn the_reference_attention_strength_weights_what_the_clip_draws_from_it() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    let dir = tempdir("attention");
    let a = adapter(&dir, "ic.safetensors", &[]);
    let at = |attention_strength: f32, attention_mask: Option<Video>| {
        let reference = VideoReference { attention_strength, attention_mask, ..VideoReference::new(clip(9, 64, 1), 1.0) };
        run(&paths, &opts(vec![a.clone()], Some(reference))).unwrap()
    };
    let max_diff = |x: &[Vec<u8>], y: &[Vec<u8>]| x.iter().flatten().zip(y.iter().flatten()).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0);
    let without = run(&paths, &opts(vec![a.clone()], None)).unwrap();
    let full = at(1.0, None);
    let cut = at(0.0, None);
    let half = at(0.5, None);
    assert_ne!(full, without, "the reference reaches the clip at full strength");
    assert!(max_diff(&cut, &without) <= 1, "at strength 0 the clip is the unreferenced one (max pixel diff {})", max_diff(&cut, &without));
    assert_ne!(half, full, "a half-strength reference is not the full one");
    assert_ne!(half, cut, "nor the cut-off one");
    assert_eq!(at(0.5, Some(mask(9, 64, 255))), half, "a white mask is the strength alone");
    assert!(max_diff(&at(1.0, Some(mask(9, 64, 0))), &cut) <= 1, "a black mask cuts the reference off");
    let e = run(&paths, &opts(vec![a], Some(VideoReference { attention_mask: Some(mask(5, 64, 255)), ..VideoReference::new(clip(9, 64, 1), 1.0) }))).unwrap_err();
    assert!(e.contains("attention mask video has 5 frames"), "{e}");
}

/// A reference is meaningless to the base checkpoint, and one shorter than
/// the clip cannot cover it: both are refused before any denoising.
#[test]
fn a_reference_without_an_adapter_or_too_short_is_refused() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    let e = run(&paths, &opts(Vec::new(), Some(VideoReference::new(clip(9, 64, 3), 1.0)))).unwrap_err();
    assert!(e.contains("--adapter"), "{e}");
    let dir = tempdir("short");
    let e = run(&paths, &opts(vec![adapter(&dir, "ic.safetensors", &[])], Some(VideoReference::new(clip(5, 64, 4), 1.0)))).unwrap_err();
    assert!(e.contains("fewer than"), "{e}");
}

/// The reference geometry comes from the adapter's own metadata, defaults to
/// 1, and two adapters that disagree about it are refused.
#[test]
fn reference_geometry_is_read_from_the_adapters_metadata() {
    let dir = tempdir("factors");
    let plain = adapter(&dir, "plain.safetensors", &[]);
    let half = adapter(&dir, "half.safetensors", &[("reference_downscale_factor", "2"), ("reference_temporal_scale_factor", "2")]);
    let quarter = adapter(&dir, "quarter.safetensors", &[("reference_downscale_factor", "4")]);
    assert_eq!(reference_scale_factors(std::slice::from_ref(&plain)).unwrap(), (1, 1));
    assert_eq!(reference_scale_factors(&[plain, half.clone()]).unwrap(), (2, 2));
    let e = reference_scale_factors(&[half, quarter]).unwrap_err();
    assert!(e.contains("reference_downscale_factor"), "{e}");
}
