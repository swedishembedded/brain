// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A saved tune serves what the trainer trained. Every trained part - decoder
//! LoRA, the connector, tower LoRA - is moved off its initial value, the
//! trainer's loss of one next token after an image prompt is taken, the tune
//! is saved and loaded the way serving loads it (`MiniCpmV::load_tuned`),
//! and the served model's `-log p` of that token must be the same number.

mod common;

use minicpmv::dataset::Sample;
use minicpmv::preprocess::{process_image, ImageOptions};
use minicpmv::prompt::{view_layout, Formatter, Message, Part, IMAGE_PAD};
use minicpmv::train::{PartMode, TrainSpec, Trainer};
use minicpmv::vision_train::train_pipelines;
use minicpmv::{GenOptions, MiniCpmV};
use model::serve::PagedDecoder;

const QUESTION: &str = "What colour dominates this picture? Answer in one word.";

/// Move every trained tensor of `t` (LoRA factors and the connector; not the
/// optimiser moments) by small seeded noise, through its own state file.
fn perturb(t: &mut Trainer, path: &std::path::Path) {
    t.save_state(path, 0).unwrap();
    let p = path.to_str().unwrap();
    let st = checkpoint::st::load_safetensors(p).unwrap();
    let config = st.config();
    // seed: a fixed perturbation keeps the compared token's loss reproducible
    let mut rng = data::rng::Rng::new(77);
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = st
        .tensors
        .into_iter()
        .map(|(n, mut v)| {
            if !n.starts_with("adam_") {
                for x in v.iter_mut() {
                    *x += (rng.next_f32() * 2.0 - 1.0) * 0.01;
                }
            }
            (n, vec![v.len() as u64], v)
        })
        .collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    checkpoint::st::save_safetensors(p, &tensors, &config, None).unwrap();
    t.restore(path).unwrap();
}

#[test]
fn a_saved_tune_serves_the_trained_numbers() {
    let Some(fx) = common::fixture() else { return };
    let fmt = Formatter::from_dir(&fx.dir).unwrap();
    let opts = ImageOptions::from_dir(&fx.dir, fx.cfg.vision.patch, fx.cfg.downsample).unwrap();
    let rgb = imaging::codec::load(fx.golden.join("inputs/room.png")).unwrap();
    let image = process_image(&rgb, &opts);
    let rendered = fmt.render(&[Message::user(vec![Part::Image, Part::Text(QUESTION.into())])], true, false).unwrap();
    let prompt = fmt.expand(&rendered, &[view_layout(&image, opts.downsample, IMAGE_PAD)], &[], opts.use_image_id).unwrap();
    let token = data::tokenizer::Tokenizer::encode(&*fmt.tokenizer, "Blue")[0];
    let mut ids = prompt.ids.clone();
    ids.push(token);
    let targets = (0..ids.len()).map(|i| if i + 2 == ids.len() { token } else { model::IGNORE }).collect();
    let sample = Sample { ids: ids.clone(), targets, image_runs: prompt.runs.clone(), video_runs: vec![], images: vec![image], videos: vec![] };

    let work = std::env::temp_dir().join(format!("brain-minicpmv-reload-{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let gpu = gpu_core::testgpu::dev(train_pipelines());
    let trained = gpu_core::reclaiming(&gpu, || {
        let spec = TrainSpec {
            decoder: PartMode::Lora { rank: 8, alpha: 16.0 },
            connector: true,
            vision: PartMode::Lora { rank: 4, alpha: 8.0 },
            block: ids.len() as u32,
            downsample: opts.downsample,
            lr: minicpmv::train::Rates::uniform(0.0),
            weight_decay: 0.0,
            grad_clip: None,
            seed: data::rng::random_seed(),
        };
        let vision = minicpmv::import::vision_weights(&fx.dir, &fx.cfg).unwrap();
        let decoder = qwen35::base_source::BaseCheckpoint { cfg: fx.cfg.text.clone(), src: Box::new(minicpmv::import::open_decoder(&fx.dir, &fx.cfg).unwrap()) };
        let mut t = Trainer::new(&gpu, fx.cfg.clone(), &vision, decoder, spec).unwrap();
        drop(vision);
        let base = t.loss(&sample).unwrap();
        perturb(&mut t, &work.join("state.safetensors"));
        let trained = t.loss(&sample).unwrap();
        assert!((trained - base).abs() > 1e-2, "the perturbation moved the loss only {base} -> {trained}");
        t.save_tune(&work.join("tune"), common::REFERENCE, "test:reload:v1").unwrap();
        trained
    });

    let mut m = MiniCpmV::load_tuned(&fx.dir, Some(&work.join("tune")), &gpu, 2048, 1, model::kv_tier::KvTier::F32).unwrap();
    let prepared = m.prepare(&[Message::user(vec![Part::Image, Part::Text(QUESTION.into())])], &[rgb], &GenOptions::default()).unwrap();
    assert_eq!(prepared.prompt.ids, prompt.ids, "serving lays the prompt out as training did");
    let mut tbl = model::paged::BlockTable::new();
    let hidden = m.engine.prefill_embeds(&mut tbl, &prepared.embeds);
    let logits = m.engine.logits(&hidden);
    m.engine.release_table(&mut tbl);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
    let served = lse - logits[token as usize];
    eprintln!("trained loss {trained}, served -log p {served}");
    assert!((served - trained).abs() <= 2e-3 * trained.abs().max(1.0), "the served tune gives {served}, the trainer {trained}");
    std::fs::remove_dir_all(&work).ok();
}
