// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Condition encoder parity vs the `diffusers` reference, at both `::tiny()`
//! (random weights, recorded in the golden's own state-dict fixture) and
//! real dims (real weights, resolved via `BRAIN_MINIMAXMUSIC3_CONDITION` -
//! the same env var the arch registry's `weights_env` names for serving).
//!
//! Regenerate goldens with `tools/goldens/minimaxmusic3_dump_reference.py`.
//! Skips cleanly when the golden or the checkpoint is absent (a fixture
//! problem, not a parity failure) - hard failure under
//! `BRAIN_REQUIRE_FIXTURES=1`.

use std::path::Path;

use brain_testutil::{golden::Source, parity::compare, testdata_path};
use minimaxmusic3::condition_encoder::{forward, from_tensors};
use minimaxmusic3::config::ConditionEncoderConfig;

const DUMPER: &str = "tools/goldens/minimaxmusic3_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn check(tag: &str, cfg: &ConditionEncoderConfig, ident: &[(&str, i64)], weights_dir: &Path) {
    let dir = testdata_path("golden/minimaxmusic3");
    let meta = dir.join(format!("condition_encoder_{tag}_meta.json"));
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(ident) {
        return;
    }

    let tensors = match checkpoint::safetensors::read_model_dir(weights_dir) {
        Ok(t) => t,
        Err(_) if weights_dir.is_file() => checkpoint::safetensors::read(weights_dir.to_str().unwrap()).unwrap(),
        Err(e) => {
            brain_testutil::skip(&format!("condition_encoder[{tag}]: cannot read {}: {e}", weights_dir.display()));
            return;
        }
    };
    let w = from_tensors(tensors, &weights_dir.display().to_string()).expect("import");

    let hidden = read_f32(&dir.join(format!("condition_encoder_{tag}_in.f32")));
    let want = read_f32(&dir.join(format!("condition_encoder_{tag}_out.f32")));

    let frames = hidden.len() / (cfg.num_condition_layers as usize * cfg.condition_hidden_dim as usize);
    let (got, lo) = forward(cfg, &w, &hidden, 1, frames);
    assert_eq!(got.len(), want.len(), "condition_encoder[{tag}]: output length mismatch (latent_length={lo})");

    let (cos, max_abs) = compare(&got, &want);
    println!("condition_encoder[{tag}]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= COS_FLOOR, "condition_encoder[{tag}]: cosine {cos} below floor {COS_FLOOR}");
}

#[test]
fn tiny_matches_diffusers_reference() {
    let cfg = ConditionEncoderConfig::tiny();
    let ident: &[(&str, i64)] = &[
        ("condition_hidden_dim", cfg.condition_hidden_dim as i64),
        ("num_condition_layers", cfg.num_condition_layers as i64),
        ("out_dim", cfg.out_dim as i64),
    ];
    let weights = testdata_path("golden/minimaxmusic3/condition_encoder_tiny_state_dict.safetensors");
    check("tiny", &cfg, ident, &weights);
}

#[test]
fn real_matches_diffusers_reference() {
    let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_CONDITION") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_CONDITION unset");
        return;
    };
    let cfg = ConditionEncoderConfig::real();
    let ident: &[(&str, i64)] = &[
        ("condition_hidden_dim", cfg.condition_hidden_dim as i64),
        ("num_condition_layers", cfg.num_condition_layers as i64),
        ("out_dim", cfg.out_dim as i64),
    ];
    check("real", &cfg, ident, Path::new(&dir));
}
