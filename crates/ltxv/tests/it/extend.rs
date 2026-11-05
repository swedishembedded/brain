// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Extension continues a clip (`video_extend_lora`'s forward extension): the
//! clip comes back untouched with whole latent frames of new content after
//! it, generated against the clip's held tail. The DiT is the tiny
//! random-weight one, so the new frames are noise - which is what makes them
//! easy to tell from a copy of the clip's last frame; the VAE is the real
//! checkpoint from the model store, so the test skips without it.

use std::path::Path;

use ltxv::pipeline::{extend, ExtendOpts, GenOpts, Paths, Video};

fn vae_path() -> Option<String> {
    let p = format!("{}/ltx-2.5-video-vae-conv-bf16.safetensors", brain_testutil::model_dir("Lightricks/LTX-2.5")?);
    Path::new(&p).exists().then_some(p)
}

/// Mean absolute byte difference between two frames.
fn diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

#[test]
fn an_extension_keeps_the_clip_and_appends_generated_frames() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    let side = 64usize;
    let frame = |k: usize| (0..side * side).flat_map(|i| [((i % side) * 4) as u8, ((i / side) * 4) as u8, (k * 8) as u8]).collect::<Vec<u8>>();
    let clip = Video { width: side as u32, height: side as u32, fps: 8, frames: (0..20).map(frame).collect(), audio: None };
    let o = ExtendOpts { frames: 5, context_frames: 9, base: GenOpts { seed: 7, eta: 0.0, device: Some("cpu".into()), ..Default::default() } };
    let (out, _) = extend(&paths, "a gradient", &clip, &o, &Default::default(), |_, _, _| {}).expect("an extension runs");
    assert_eq!(out.frames.len(), 20 + 8, "five frames round up to one whole latent frame");
    assert!(out.frames[..20] == clip.frames[..], "the clip itself comes back untouched");
    let last = clip.frames.last().unwrap();
    let new: f64 = out.frames[20..].iter().map(|f| diff(f, last)).sum::<f64>() / 8.0;
    eprintln!("mean |new - last source frame|: {new:.2}");
    assert!(new > 10.0, "the added frames must be generated, not a copy of the last frame (diff {new:.2})");
}

/// The same extension through the capability surface: the clip arrives as a
/// video blob and comes back as one, the source frames first.
#[test]
fn the_extend_action_takes_and_returns_a_video_blob() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    let side = 64u32;
    let frame = |k: usize| ((0..side * side).flat_map(|i| [((i % side) * 4) as f32 / 255.0, ((i / side) * 4) as f32 / 255.0, (k * 8) as f32 / 255.0]).collect::<Vec<f32>>(), side, side);
    let source: Vec<(Vec<f32>, u32, u32)> = (0..9).map(frame).collect();
    let inv = capability::Invocation::new()
        .blob("video", capability::blob::video_blob(&source).unwrap())
        .set("prompt", serde_json::json!("a gradient"))
        .set("fps", serde_json::json!(8))
        .set("frames", serde_json::json!(8))
        .set("seed", serde_json::json!(3))
        .set("eta", serde_json::json!(0.0));
    let cpu = |o: &mut GenOpts| o.device = Some("cpu".into());
    let out = ltxv::caps::clip_action_on("extend", &paths, &inv, &cpu, &mut |_| {}).expect("the action runs");
    let got = capability::blob::decode_video(&capability::Invocation::new().blob("video", out.blobs["video"].clone()), "video").expect("a video blob comes back");
    assert_eq!(got.len(), 17);
    for (k, (px, _, _)) in got.iter().take(9).enumerate() {
        let want: Vec<f32> = source[k].0.iter().map(|v| (v * 255.0).round() / 255.0).collect();
        assert!(px.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6), "source frame {k} must come back unchanged");
    }
}
