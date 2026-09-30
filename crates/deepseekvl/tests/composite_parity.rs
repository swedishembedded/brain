// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL-7B-chat end to end on the real checkpoint, against the pinned
//! reference's fp32 run (`tools/goldens/deepseekvl_dump_reference.py`) over a
//! fixed synthetic image. Each stage is fed the reference's own input, so a
//! failure names the stage that broke:
//!
//! 1. preprocessing: the processor's `pixel_values` and the low branch's
//!    resize;
//! 2. the prompt: rendered text and token ids, image placeholders expanded;
//! 3. the tower: both branches' features and the aligner's rows;
//! 4. the decoder at the checkpoint's own fp16 weights: the prompt's
//!    next-token logits, and 16 greedy tokens identical to the reference's.
//!
//! The towers' blocks have their own gates (`sam1`'s and `clip`'s real-weight
//! parity tests). Skips when the checkpoint or the golden is absent.

use std::path::Path;

use brain_testutil::parity::{load, rel_l2, Report};
use deepseekvl::prompt::{Role, Turn};
use imaging::pixels::Rgb8;
use qwen3::model::PrefillInput;

const REPO: &str = "deepseek-ai/deepseek-vl-7b-chat";
const DUMPER: &str = "tools/goldens/deepseekvl_dump_reference.py";
const FLOOR: f64 = 0.9999;

#[test]
fn deepseek_vl_matches_the_reference_end_to_end() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let Some(dir) = brain_testutil::model_dir(REPO).filter(|d| Path::new(d).join("model.safetensors.index.json").exists()) else {
        return brain_testutil::skip(&format!("{REPO} not in the model store"));
    };
    let golden_dir = brain_testutil::testdata_path("deepseek-vl/composite");
    let Some(src) = brain_testutil::golden::Source::open(&golden_dir, DUMPER) else { return };
    if !src.require(&[("n_layers", 30), ("d_model", 4096), ("vocab", 102400), ("image_tokens", 576)]) {
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden_dir.join("manifest.json")).unwrap()).unwrap();
    let ids_of = |k: &str| -> Vec<u32> { manifest[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let g = load(&golden_dir.join("golden.safetensors"));

    let m = deepseekvl::load(Path::new(&dir), qwen3::Dtype::F16, 1024).expect("load DeepSeek-VL");
    let mut r = Report::new(FLOOR);

    // ---- 1. preprocessing ----
    let shape = &g["image"].shape;
    let img = Rgb8 { w: shape[1] as u32, h: shape[0] as u32, px: g["image"].data.iter().map(|&v| v as u8).collect() };
    let px = m.processor.pixel_values(&img).unwrap();
    let max_abs = px.iter().zip(&g["pixel_values"].data).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(max_abs <= 1e-6, "pixel_values differ by {max_abs:e}: the resize, the padding or the rescale");
    let low = deepseekvl::preprocess::low_resolution(&g["pixel_values"].data, 1024, 384);
    r.check("low_images", &low, &g["low_images"].data);

    // ---- 2. the prompt ----
    let turns = [Turn { role: Role::User, content: "<image_placeholder>Describe this image.".into() }];
    assert_eq!(deepseekvl::prompt::render(&m.style, deepseekvl::prompt::SYSTEM_PROMPT, &turns, &m.eos).unwrap(), manifest["prompt"].as_str().unwrap());
    let ids = m.prompt_ids(&turns).unwrap();
    assert_eq!(manifest["image_token_id"].as_u64(), Some(m.splice.image_id as u64));
    assert_eq!(m.splice.expand_ids(&ids), ids_of("input_ids"), "prompt token ids");

    // ---- 3. the tower ----
    let f = m.tower.encode(&g["pixel_values"].data);
    r.check("high_features", &f.streams[0], &g["high_features"].data);
    r.check("low_features", &f.streams[1], &g["low_features"].data);
    r.check("aligner_out", &f.embeds, &g["aligner_out"].data);
    let rel = rel_l2(&f.embeds, &g["aligner_out"].data);
    assert!(rel < 1e-2, "aligner_out rel_l2 {rel:.3e}");

    // ---- 4. the decoder ----
    let inputs = m.inputs(&ids, &f.embeds).unwrap();
    let spliced: Vec<f32> = inputs
        .iter()
        .flat_map(|i| match i {
            PrefillInput::Token(t) => m.decoder.embed_row(*t),
            PrefillInput::Embed(row) => row.to_vec(),
        })
        .collect();
    r.check("inputs_embeds", &spliced, &g["inputs_embeds"].data);
    m.decoder.reset_cache();
    m.decoder.prefill(&inputs);
    r.check("logits_last", &m.decoder.decode_logits(), &g["logits_last"].data);
    r.finish("DeepSeek-VL composite");

    let want = ids_of("greedy_ids");
    let got = m.generate_greedy(&ids, &f.embeds, want.len(), &mut |_| true).unwrap();
    assert_eq!(got, want, "greedy continuation: {:?}", data::tokenizer::Tokenizer::decode(&m.tokenizer, &got));
}
