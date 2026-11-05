// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A retake regenerates a time window of a clip and holds the rest
//! (`ltx_pipelines.retake`'s `TemporalRegionMask`): the held frames come
//! back as the VAE's own reconstruction of the source, the window as
//! whatever the model denoises there. The DiT is the tiny random-weight one,
//! so the window turns to noise - which is exactly what makes the two
//! regions easy to tell apart; the VAE is the real checkpoint from the model
//! store, so the test skips without it.

use std::path::Path;

use ltxv::pipeline::{generate_masked, MaskedOpts, Paths, Video};

fn vae_path() -> Option<String> {
    let p = format!("{}/ltx-2.5-video-vae-conv-bf16.safetensors", brain_testutil::model_dir("Lightricks/LTX-2.5")?);
    Path::new(&p).exists().then_some(p)
}

/// Mean absolute byte difference between two frames.
fn diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

#[test]
fn a_retake_regenerates_its_window_and_holds_the_rest() {
    let Some(vae) = vae_path() else {
        brain_testutil::skip("no ltx-2.5-video-vae-conv-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let paths = Paths::new(vae.clone());
    // A smooth gradient the VAE reconstructs well, 17 frames at 8 fps:
    // latent frames span [0, 1/8), [1/8, 9/8) and [9/8, 17/8) seconds.
    let side = 64usize;
    let frame = |k: usize| (0..side * side).flat_map(|i| [((i % side) * 4) as u8, ((i / side) * 4) as u8, (k * 8) as u8]).collect::<Vec<u8>>();
    let clip = Video { width: side as u32, height: side as u32, fps: 8, frames: (0..17).map(frame).collect(), audio: None };
    let o = MaskedOpts { window: Some((1.2, 2.0)), base: ltxv::pipeline::GenOpts { seed: 7, eta: 0.0, device: Some("cpu".into()), ..Default::default() }, ..Default::default() };
    let (out, _) = generate_masked(&paths, "a gradient", &clip, &o, &Default::default(), |_, _, _| {}).expect("a retake runs");
    assert_eq!(out.frames.len(), clip.frames.len());
    let held: f64 = (0..9).map(|k| diff(&out.frames[k], &clip.frames[k])).sum::<f64>() / 9.0;
    let window: f64 = (9..17).map(|k| diff(&out.frames[k], &clip.frames[k])).sum::<f64>() / 8.0;
    eprintln!("mean |out - source|: held {held:.2}, window {window:.2}");
    assert!(window > 3.0 * held, "the window must be regenerated (diff {window:.2}) and the rest held (diff {held:.2})");
}
