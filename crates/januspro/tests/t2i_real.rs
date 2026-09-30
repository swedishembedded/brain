// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro-7B text-to-image on the serving engine, on the real checkpoint.
//!
//! The engine holds the 7B decoder at the checkpoint's own bf16. Two checks:
//!
//! * teacher-forced against the pinned reference's fp32 run
//!   (`tools/goldens/janus_dump_reference.py`): fed the reference's sampled
//!   tokens, the guided logits at each of its 8 steps match - the batched
//!   engine computes what the single-sequence decoder does in
//!   `reference_parity.rs`;
//! * a whole 576-token image from a fixed seed decodes to a real picture:
//!   finite, with contrast, and red dominant for a prompt asking for a red
//!   apple.
//!
//! Skips when the checkpoint or the golden is absent.

use std::path::Path;

use brain_testutil::parity::{load, Report};
use januspro::t2i::{Request, TextToImage};

const REPO: &str = "deepseek-ai/Janus-Pro-7B";
const DUMPER: &str = "tools/goldens/janus_dump_reference.py";

#[test]
fn janus_pro_generates_images_on_the_engine() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let Some(dir) = brain_testutil::model_dir(REPO).filter(|d| Path::new(d).join("pytorch_model.bin.index.json").exists()) else {
        return brain_testutil::skip(&format!("{REPO} not in the model store"));
    };
    let g_dir = brain_testutil::testdata_path("janus/generation");
    let Some(src) = brain_testutil::golden::Source::open(&g_dir, DUMPER) else { return };
    if !src.require(&[("n_layers", 30), ("d_model", 4096), ("vocab", 102400), ("image_vocab", 16384)]) {
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(g_dir.join("manifest.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { manifest[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let g = load(&g_dir.join("golden.safetensors"));

    let mut t2i = TextToImage::load(Path::new(&dir), 1, qwen3::Dtype::BF16).expect("load Janus-Pro generation");
    let prompt = "A red apple on a wooden table.";
    let (cond, uncond) = t2i.prompt_ids(prompt).unwrap();
    assert_eq!((cond.clone(), uncond.clone()), (ids("cond_ids"), ids("uncond_ids")), "the guided pair of prompts");

    // ---- teacher-forced ----
    let sampled = ids("sampled");
    let weight = manifest["cfg_weight"].as_f64().unwrap() as f32;
    let want = &g["logits_cfg"].data;
    let v = want.len() / sampled.len();
    let mut r = Report::new(0.9999);
    t2i.run_tokens(&cond, &uncond, weight, sampled.len(), &mut |step, blended| {
        r.check(&format!("logits_cfg[{step}]"), blended, &want[step * v..(step + 1) * v]);
        Ok(vec![sampled[step]])
    })
    .unwrap();
    r.finish("Janus-Pro guided logits (bf16 engine)");

    // ---- a whole image ----
    let req = Request { prompt, cfg_weight: weight, temperature: 1.0, seed: 7 };
    let mut last = 0;
    let images = t2i.generate(&req, &|| false, &mut |step, _| last = step).unwrap();
    assert_eq!((images.len(), last), (1, t2i.image_tokens()));
    let img = &images[0];
    let n = (img.w * img.h) as f64;
    let mean = |c: usize| img.px.iter().skip(c).step_by(3).map(|&v| v as f64).sum::<f64>() / n;
    let (r_mean, g_mean, b_mean) = (mean(0), mean(1), mean(2));
    let luma: Vec<f64> = img.px.chunks(3).map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64).collect();
    let l_mean = luma.iter().sum::<f64>() / n;
    let l_std = (luma.iter().map(|l| (l - l_mean).powi(2)).sum::<f64>() / n).sqrt();
    eprintln!("generated {}x{}: mean rgb ({r_mean:.1}, {g_mean:.1}, {b_mean:.1}), luma std {l_std:.1}", img.w, img.h);
    let out = std::env::temp_dir().join(format!("janus-t2i-{}.png", std::process::id()));
    imaging::codec::save_png(&out, img).expect("write the generated image");
    eprintln!("wrote {} for inspection", out.display());
    assert!(l_std > 20.0, "the image has no contrast (luma std {l_std:.1})");
    assert!(r_mean > g_mean && r_mean > b_mean, "a red apple should make red the dominant channel: ({r_mean:.1}, {g_mean:.1}, {b_mean:.1})");
}
