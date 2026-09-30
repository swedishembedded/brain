// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro-7B on the real checkpoint against the pinned reference's fp32
//! run (`tools/goldens/janus_dump_reference.py`). One decoder load serves
//! both halves, at the checkpoint's own bf16 weights.
//!
//! * Understanding, stage by stage from the reference's own inputs:
//!   `pixel_values` of a fixed synthetic image, the prompt with its image
//!   wrapped in begin/end tags, SigLIP features, the aligner, the next-token
//!   logits, and 16 greedy tokens identical to the reference's.
//! * Generation, teacher-forced: the reference sampled 8 image tokens under
//!   classifier-free guidance. Fed the same tokens, the conditional and the
//!   unconditional sequences (the prompt with its interior padded out) are
//!   independent, so each runs alone on the decoder. At every step the
//!   generation head's logits for both, their guided blend, and the fed-back
//!   token rows must match.
//!
//! The VQ-16 decoder that turns tokens into pixels has its own gate
//! (`brain-vqgan`). Skips when the checkpoint or a golden is absent.

use std::path::Path;

use brain_testutil::parity::{load, Report};
use deepseekvl::prompt::{Role, Turn};
use imaging::pixels::Rgb8;
use qwen3::model::PrefillInput;

const REPO: &str = "deepseek-ai/Janus-Pro-7B";
const DUMPER: &str = "tools/goldens/janus_dump_reference.py";
const FLOOR: f64 = 0.9999;

fn manifest(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn ids(m: &serde_json::Value, key: &str) -> Vec<u32> {
    m[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

#[test]
fn janus_pro_matches_the_reference() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let Some(dir) = brain_testutil::model_dir(REPO).filter(|d| Path::new(d).join("pytorch_model.bin.index.json").exists()) else {
        return brain_testutil::skip(&format!("{REPO} not in the model store"));
    };
    let (u_dir, g_dir) = (brain_testutil::testdata_path("janus/understanding"), brain_testutil::testdata_path("janus/generation"));
    let identity = [("n_layers", 30), ("d_model", 4096), ("vocab", 102400), ("image_vocab", 16384)];
    for d in [&u_dir, &g_dir] {
        let Some(src) = brain_testutil::golden::Source::open(d, DUMPER) else { return };
        if !src.require(&identity) {
            return;
        }
    }
    let m = januspro::model::load_understanding(Path::new(&dir), qwen3::Dtype::BF16, 1024).expect("load Janus-Pro");
    let mut r = Report::new(FLOOR);

    // ---- understanding ----
    let (um, g) = (manifest(&u_dir), load(&u_dir.join("golden.safetensors")));
    let shape = &g["image"].shape;
    let img = Rgb8 { w: shape[1] as u32, h: shape[0] as u32, px: g["image"].data.iter().map(|&v| v as u8).collect() };
    let px = m.processor.pixel_values(&img).unwrap();
    let max_abs = px.iter().zip(&g["pixel_values"].data).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(max_abs <= 1e-6, "pixel_values differ by {max_abs:e}");

    let turns = [Turn { role: Role::User, content: "<image_placeholder>\nDescribe this image.".into() }];
    assert_eq!(deepseekvl::prompt::render(&m.style, deepseekvl::prompt::SYSTEM_PROMPT, &turns, &m.eos).unwrap(), um["prompt"].as_str().unwrap());
    let prompt = m.prompt_ids(&turns).unwrap();
    let tag = |k: &str| um[k].as_u64().unwrap() as u32;
    assert_eq!(m.splice.wrap, Some((tag("image_start_id"), tag("image_end_id"))), "the image tags");
    assert_eq!(m.splice.expand_ids(&prompt), ids(&um, "input_ids"), "prompt token ids");

    let f = m.tower.encode(&g["pixel_values"].data);
    r.check("features", &f.streams[0], &g["features"].data);
    r.check("aligner_out", &f.embeds, &g["aligner_out"].data);
    let inputs = m.inputs(&prompt, &f.embeds).unwrap();
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
    let want = ids(&um, "greedy_ids");
    let got = m.generate_greedy(&prompt, &f.embeds, want.len(), &mut |_| {}).unwrap();
    assert_eq!(got, want, "greedy continuation: {:?}", data::tokenizer::Tokenizer::decode(&m.tokenizer, &got));

    // ---- generation, teacher-forced ----
    let (gm, g) = (manifest(&g_dir), load(&g_dir.join("golden.safetensors")));
    let (cfg, rd) = januspro::model::open(Path::new(&dir)).unwrap();
    let heads = januspro::gen::GenHeads::load(&rd, &cfg, 1).expect("load the generation heads");
    let sampled = ids(&gm, "sampled");
    let weight = gm["cfg_weight"].as_f64().unwrap() as f32;
    let (v, d) = (heads.vocab(), m.decoder_cfg.d_model as usize);
    let step_rows = |key: &str, t: usize, width: usize| g[key].data[t * width..(t + 1) * width].to_vec();
    let mut branch_logits = Vec::new();
    for key in ["cond_ids", "uncond_ids"] {
        let prompt: Vec<PrefillInput> = ids(&gm, key).into_iter().map(PrefillInput::Token).collect();
        m.decoder.reset_cache();
        let mut hidden = m.decoder.prefill(&prompt);
        let mut logits = Vec::new();
        for (t, &token) in sampled.iter().enumerate() {
            logits.push(heads.logits(&hidden));
            let fed = heads.token_embeds(&[token]).unwrap();
            if key == "cond_ids" {
                r.check(&format!("token_embeds[{t}]"), &fed, &step_rows("token_embeds", t, d));
            }
            hidden = m.decoder.step_embed(&fed);
        }
        branch_logits.push(logits);
    }
    for t in 0..sampled.len() {
        let (cond, uncond) = (&branch_logits[0][t], &branch_logits[1][t]);
        r.check(&format!("logits_cond[{t}]"), cond, &step_rows("logits_cond", t, v));
        r.check(&format!("logits_uncond[{t}]"), uncond, &step_rows("logits_uncond", t, v));
        r.check(&format!("logits_cfg[{t}]"), &model::hostmath::cfg_blend(cond, uncond, weight), &step_rows("logits_cfg", t, v));
    }
    r.finish("Janus-Pro");
}
