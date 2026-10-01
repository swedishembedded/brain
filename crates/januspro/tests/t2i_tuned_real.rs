// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A generation fine-tune served by the drawing engine reproduces what it was
//! trained on. Needs the real checkpoint, a fine-tune directory
//! (`JANUS_TUNED_DIR`, from `brain januspro finetune --mode generation`) and
//! the image and prompt it was trained on (`JANUS_TUNED_IMAGE`,
//! `JANUS_TUNED_PROMPT`); run with `--release -- --ignored`. Drawn without
//! guidance (weight 1) and near-greedily, the tuned model's picture is much
//! closer to the training image than the base model's.

use std::path::Path;

use januspro::t2i::{Request, TextToImage};

fn mse(a: &imaging::pixels::Rgb8, target: &imaging::pixels::Rgb8) -> f64 {
    assert_eq!((a.w, a.h), (target.w, target.h));
    a.px.iter().zip(&target.px).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.px.len() as f64
}

#[test]
#[ignore]
fn a_generation_fine_tune_draws_its_training_image() {
    let (Ok(tuned), Ok(image), Ok(prompt)) = (std::env::var("JANUS_TUNED_DIR"), std::env::var("JANUS_TUNED_IMAGE"), std::env::var("JANUS_TUNED_PROMPT")) else {
        return brain_testutil::skip("JANUS_TUNED_DIR, JANUS_TUNED_IMAGE and JANUS_TUNED_PROMPT are not set");
    };
    let Some(dir) = brain_testutil::model_dir("deepseek-ai/Janus-Pro-7B") else { return brain_testutil::skip("Janus-Pro-7B not in the model store") };
    let target = imaging::codec::decode(&std::fs::read(&image).unwrap()).unwrap();
    let req = Request { prompt: &prompt, cfg_weight: 1.0, temperature: 0.05, seed: 3 };
    let mut draw = |tuned: Option<&Path>| {
        let mut t2i = TextToImage::load_tuned(Path::new(&dir), 1, qwen3::Dtype::BF16, 1024, tuned).unwrap();
        t2i.generate(&req, &|| false, &mut |_, _| {}).unwrap().remove(0)
    };
    let base = mse(&draw(None), &target);
    let tuned_img = draw(Some(Path::new(&tuned)));
    let tuned_mse = mse(&tuned_img, &target);
    imaging::codec::save_png(&std::env::temp_dir().join("janus-tuned.png"), &tuned_img).unwrap();
    eprintln!("mse to the training image: base {base:.0}, tuned {tuned_mse:.0}");
    assert!(tuned_mse < base * 0.25, "tuned {tuned_mse:.0} vs base {base:.0}");
}
