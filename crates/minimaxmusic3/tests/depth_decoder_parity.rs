// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RVQ depth decoder parity vs the `diffusers` reference, at both
//! `::tiny()` (random weights, recorded in the golden's own state-dict
//! fixture) and real dims (real weights, resolved via
//! `BRAIN_MINIMAXMUSIC3_DEPTH`).
//!
//! Four independently-checked pieces, matching the golden dumper's own
//! four taps: `forward` (the transformer stack), `projection`,
//! `audio_embeddings` (gather), and `audio_heads[i]` (one per residual
//! codebook).
//!
//! Regenerate goldens with `tools/goldens/minimaxmusic3_dump_reference.py`.
//! Skips cleanly when the golden or the checkpoint is absent.

use std::path::Path;

use brain_testutil::{golden::Source, parity::compare, testdata_path};
use minimaxmusic3::config::DepthDecoderConfig;
use minimaxmusic3::depth_decoder::{audio_embedding_row, audio_head, forward, from_tensors, projection};

const DUMPER: &str = "tools/goldens/minimaxmusic3_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u32(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn ident(cfg: &DepthDecoderConfig) -> Vec<(&'static str, i64)> {
    vec![
        ("hidden_size", cfg.hidden_size as i64),
        ("num_layers", cfg.num_layers as i64),
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("intermediate_size", cfg.intermediate_size as i64),
        ("audio_vocab_size", cfg.audio_vocab_size as i64),
        ("num_codebooks", cfg.num_codebooks as i64),
        ("max_position_embeddings", cfg.max_position_embeddings as i64),
    ]
}

fn check(tag: &str, cfg: &DepthDecoderConfig, weights_dir: &Path) {
    let dir = testdata_path("golden/minimaxmusic3");
    let meta = dir.join(format!("depth_decoder_{tag}_meta.json"));
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(&ident(cfg)) {
        return;
    }

    let tensors = match checkpoint::safetensors::read_model_dir(weights_dir) {
        Ok(t) => t,
        Err(_) if weights_dir.is_file() => checkpoint::safetensors::read(weights_dir.to_str().unwrap()).unwrap(),
        Err(e) => {
            brain_testutil::skip(&format!("depth_decoder[{tag}]: cannot read {}: {e}", weights_dir.display()));
            return;
        }
    };
    let w = from_tensors(tensors, cfg, &weights_dir.display().to_string()).expect("import");

    // Tap 1: forward (the transformer stack).
    let inputs_embeds = read_f32(&dir.join(format!("depth_decoder_{tag}_inputs_embeds.f32")));
    let want_hidden = read_f32(&dir.join(format!("depth_decoder_{tag}_hidden_out.f32")));
    let steps = cfg.num_codebooks as usize;
    let (got_hidden, _) = forward(&w, cfg, &inputs_embeds, steps);
    let (cos, max_abs) = compare(&got_hidden, &want_hidden);
    println!("depth_decoder[{tag}] forward: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= COS_FLOOR, "depth_decoder[{tag}] forward: cosine {cos} below floor {COS_FLOOR}");

    // Tap 2: projection.
    let proj_in = read_f32(&dir.join(format!("depth_decoder_{tag}_proj_in.f32")));
    let want_proj = read_f32(&dir.join(format!("depth_decoder_{tag}_proj_out.f32")));
    let got_proj = projection(&w, cfg, &proj_in);
    let (cos, _) = compare(&got_proj, &want_proj);
    println!("depth_decoder[{tag}] projection: cosine={cos:.9}");
    assert!(cos >= COS_FLOOR, "depth_decoder[{tag}] projection: cosine {cos} below floor {COS_FLOOR}");

    // Tap 3: audio_embeddings gather - the golden dumper's own tap is the
    // RAW `audio_embeddings(codes + offsets)` (one row per residual
    // codebook, shape `[num_codebooks-1, hidden]`), NOT the pipeline's
    // `.sum(dim=1)` reduction (that summing happens one layer up, in
    // `_embed_audio_frame`, not inside this module).
    let codes = read_u32(&dir.join(format!("depth_decoder_{tag}_codes.u32")));
    let want_embed = read_f32(&dir.join(format!("depth_decoder_{tag}_embed_out.f32")));
    let n_res = cfg.num_codebooks as usize - 1;
    let d = cfg.hidden_size as usize;
    let mut worst_cos = f64::MAX;
    for (i, &code) in codes.iter().enumerate() {
        let row = audio_embedding_row(&w, cfg, code as usize + i * cfg.audio_vocab_size as usize);
        let want = &want_embed[i * d..(i + 1) * d];
        let (cos, _) = compare(&row, want);
        worst_cos = worst_cos.min(cos);
    }
    println!("depth_decoder[{tag}] audio_embeddings: worst cosine={worst_cos:.9}");
    assert!(worst_cos >= COS_FLOOR, "depth_decoder[{tag}] audio_embeddings: worst cosine {worst_cos} below floor {COS_FLOOR}");

    // Tap 4: audio_heads[i], one per residual codebook.
    let head_ins = read_f32(&dir.join(format!("depth_decoder_{tag}_head_ins.f32")));
    let want_heads = read_f32(&dir.join(format!("depth_decoder_{tag}_head_outs.f32")));
    let d = cfg.hidden_size as usize;
    let v = cfg.audio_vocab_size as usize;
    let mut worst_cos = f64::MAX;
    for i in 0..n_res {
        let x = &head_ins[i * d..(i + 1) * d];
        let got = audio_head(&w, cfg, i, x);
        let want = &want_heads[i * v..(i + 1) * v];
        let (cos, _) = compare(&got, want);
        worst_cos = worst_cos.min(cos);
    }
    println!("depth_decoder[{tag}] audio_heads: worst cosine={worst_cos:.9}");
    assert!(worst_cos >= COS_FLOOR, "depth_decoder[{tag}] audio_heads: worst cosine {worst_cos} below floor {COS_FLOOR}");
}

#[test]
fn tiny_matches_diffusers_reference() {
    let cfg = DepthDecoderConfig::tiny();
    let weights = testdata_path("golden/minimaxmusic3/depth_decoder_tiny_state_dict.safetensors");
    check("tiny", &cfg, &weights);
}

#[test]
fn real_matches_diffusers_reference() {
    let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_DEPTH") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_DEPTH unset");
        return;
    };
    let cfg = DepthDecoderConfig::real();
    check("real", &cfg, Path::new(&dir));
}
