// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DiT parity vs the `diffusers` reference, at both `::tiny()` (random
//! weights, recorded in the golden's own state-dict fixture) and real dims
//! (real weights, resolved via `BRAIN_MINIMAXMUSIC3_DIT`).
//!
//! Regenerate goldens with `tools/goldens/minimaxmusic3_dump_reference.py`.
//! Skips cleanly when the golden or the checkpoint is absent.

use std::path::Path;

use brain_testutil::{golden::Source, parity::compare, testdata_path};
use minimaxmusic3::config::DitConfig;
use minimaxmusic3::dit::{forward, from_tensors, PIPELINES};

const DUMPER: &str = "tools/goldens/minimaxmusic3_dump_reference.py";
const COS_FLOOR: f64 = 0.999;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn ident(cfg: &DitConfig) -> Vec<(&'static str, i64)> {
    vec![
        ("in_channels", cfg.in_channels as i64),
        ("condition_dim", cfg.condition_dim as i64),
        ("num_layers", cfg.num_layers as i64),
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("attention_head_dim", cfg.attention_head_dim as i64),
        ("ff_inner_dim", cfg.ff_inner_dim as i64),
        ("rotary_dim", cfg.rotary_dim as i64),
        ("fourier_embedding_dim", cfg.fourier_embedding_dim as i64),
    ]
}

fn check(tag: &str, cfg: &DitConfig, weights_dir: &Path) {
    let dir = testdata_path("golden/minimaxmusic3");
    let meta = dir.join(format!("dit_{tag}_meta.json"));
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(&ident(cfg)) {
        return;
    }

    let tensors = match checkpoint::safetensors::read_model_dir(weights_dir) {
        Ok(t) => t,
        Err(_) if weights_dir.is_file() => checkpoint::safetensors::read(weights_dir.to_str().unwrap()).unwrap(),
        Err(e) => {
            brain_testutil::skip(&format!("dit[{tag}]: cannot read {}: {e}", weights_dir.display()));
            return;
        }
    };
    let w = from_tensors(tensors, cfg, tag).expect("import");

    let hidden_states = read_f32(&dir.join(format!("dit_{tag}_hidden_states.f32")));
    let timestep = read_f32(&dir.join(format!("dit_{tag}_timestep.f32")))[0];
    let encoder_hidden_states = read_f32(&dir.join(format!("dit_{tag}_encoder_hidden_states.f32")));
    let want = read_f32(&dir.join(format!("dit_{tag}_out.f32")));
    let length = hidden_states.len() / cfg.in_channels as usize;

    // The pooled test device, NOT `Gpu::new_cpu`: this parity gate has to
    // be runnable on BOTH backends (`make parity` is the cross-backend
    // gate, and this repo has already paid for a defect that appeared on
    // one backend only), and hardcoding the CPU JIT made that impossible.
    // `testgpu::dev` honours the ambient `--device`/`BRAIN_DEVICE`
    // selection and shares one device per test binary, per the
    // one-device-per-process rule.
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let got = forward(&gpu, cfg, &w, &hidden_states, &encoder_hidden_states, timestep, length);
    assert_eq!(got.len(), want.len(), "dit[{tag}]: output length mismatch");

    let (cos, max_abs) = compare(&got, &want);
    println!("dit[{tag}]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= COS_FLOOR, "dit[{tag}]: cosine {cos} below floor {COS_FLOOR}");
}

#[test]
fn tiny_matches_diffusers_reference() {
    let cfg = DitConfig::tiny();
    let weights = testdata_path("golden/minimaxmusic3/dit_tiny_state_dict.safetensors");
    check("tiny", &cfg, &weights);
}

#[test]
fn real_matches_diffusers_reference() {
    let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_DIT") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_DIT unset");
        return;
    };
    let cfg = DitConfig::real();
    check("real", &cfg, Path::new(&dir));
}
