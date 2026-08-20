// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vocoder parity vs the `diffusers` reference, at both `::tiny()` (random
//! weights, recorded in the golden's own state-dict fixture) and real dims
//! (real weights, resolved via `BRAIN_MINIMAXMUSIC3_VOCODER`).
//!
//! Regenerate goldens with `tools/goldens/minimaxmusic3_dump_reference.py`.
//! Skips cleanly when the golden or the checkpoint is absent.

use std::path::Path;

use brain_testutil::{golden::Source, parity::compare, testdata_path};
use gpu_core::Gpu;
use minimaxmusic3::config::VocoderConfig;
use minimaxmusic3::vocoder::{forward, from_tensors, PIPELINES};

const DUMPER: &str = "tools/goldens/minimaxmusic3_dump_reference.py";
const COS_FLOOR: f64 = 0.999;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn check(tag: &str, cfg: &VocoderConfig, ident: &[(&str, i64)], weights_dir: &Path) {
    let dir = testdata_path("golden/minimaxmusic3");
    let meta = dir.join(format!("vocoder_{tag}_meta.json"));
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(ident) {
        return;
    }

    let tensors = match checkpoint::safetensors::read_model_dir(weights_dir) {
        Ok(t) => t,
        Err(_) if weights_dir.is_file() => checkpoint::safetensors::read(weights_dir.to_str().unwrap()).unwrap(),
        Err(e) => {
            brain_testutil::skip(&format!("vocoder[{tag}]: cannot read {}: {e}", weights_dir.display()));
            return;
        }
    };
    let w = from_tensors(tensors, cfg).expect("import");

    let latents = read_f32(&dir.join(format!("vocoder_{tag}_in.f32")));
    let want = read_f32(&dir.join(format!("vocoder_{tag}_out.f32")));
    let length = latents.len() / cfg.latent_channels as usize;

    let gpu = Gpu::new_cpu(PIPELINES);
    let got = forward(&gpu, cfg, &w, &latents, 1, length);
    assert_eq!(got.len(), want.len(), "vocoder[{tag}]: output length mismatch");

    let (cos, max_abs) = compare(&got, &want);
    println!("vocoder[{tag}]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= COS_FLOOR, "vocoder[{tag}]: cosine {cos} below floor {COS_FLOOR}");
}

fn ident(cfg: &VocoderConfig) -> Vec<(&'static str, i64)> {
    vec![
        ("latent_channels", cfg.latent_channels as i64),
        ("decoder_input_dim", cfg.decoder_input_dim as i64),
        ("decoder_hidden_dim", cfg.decoder_hidden_dim as i64),
        ("num_upsample_stages", cfg.upsampling_ratios.len() as i64),
        ("sampling_rate", cfg.sampling_rate as i64),
    ]
}

#[test]
fn tiny_matches_diffusers_reference() {
    let cfg = VocoderConfig::tiny();
    let weights = testdata_path("golden/minimaxmusic3/vocoder_tiny_state_dict.safetensors");
    check("tiny", &cfg, &ident(&cfg), &weights);
}

#[test]
fn real_matches_diffusers_reference() {
    let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_VOCODER") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_VOCODER unset");
        return;
    };
    let cfg = VocoderConfig::real();
    check("real", &cfg, &ident(&cfg), Path::new(&dir));
}
