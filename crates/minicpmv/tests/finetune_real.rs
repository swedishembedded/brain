// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fine-tuning the real checkpoint end to end: a run over a small synthetic
//! task (the colour of a disc on a noisy background names a made-up
//! codeword) with every part training - decoder LoRA, the connector and
//! tower LoRA - must cut the held-out loss on images it never saw, and the
//! tune it writes, loaded the way serving loads it, must answer those
//! held-out images with their codewords. At the default per-part rates
//! (the connector at a tenth of the LoRA rate) the held-out loss falls from
//! 11.6 to under 0.01 in 32 steps; the run takes about 16 minutes on a P40.

mod common;

use std::path::Path;

use data::rng::Rng;
use minicpmv::finetune::{finetune, Event, Job, Options};
use minicpmv::prompt::{Message, Part};
use minicpmv::train::PartMode;
use minicpmv::{GenOptions, MiniCpmV};

const QUESTION: &str = "Which codeword does this picture show?";
const CODES: [(&str, [u8; 3]); 3] = [("falcon", [220, 30, 30]), ("maple", [30, 200, 40]), ("quartz", [40, 60, 230])];

/// A 224x224 image: grey noise with a disc of `color` somewhere on it.
fn picture(color: [u8; 3], rng: &mut Rng) -> imaging::Rgb8 {
    let (w, h) = (224u32, 224u32);
    let (cx, cy, r) = (rng.uniform(60.0, 164.0), rng.uniform(60.0, 164.0), rng.uniform(30.0, 55.0));
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let inside = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) < r * r;
            let g = rng.gen_range_inclusive(90, 160) as u8;
            px.extend(if inside { color } else { [g, g, g] });
        }
    }
    imaging::Rgb8 { w, h, px }
}

/// `per_color` pictures of each colour under `dir`, and their JSONL.
fn dataset(dir: &Path, name: &str, per_color: usize, rng: &mut Rng) -> std::path::PathBuf {
    let mut lines = String::new();
    for (code, color) in CODES {
        for i in 0..per_color {
            let file = format!("{name}-{code}-{i}.png");
            imaging::codec::save_png(dir.join(&file), &picture(color, rng)).unwrap();
            let record = serde_json::json!({"messages": [
                {"role": "user", "train": false, "content": [{"type": "image", "path": file}, {"type": "text", "text": QUESTION}]},
                {"role": "assistant", "train": true, "content": code}]});
            lines.push_str(&format!("{record}\n"));
        }
    }
    let path = dir.join(format!("{name}.jsonl"));
    std::fs::write(&path, lines).unwrap();
    path
}

#[test]
fn every_part_trains_and_the_saved_tune_answers_unseen_images() {
    let Some(dir) = common::checkpoint() else { return };
    let work = std::env::temp_dir().join(format!("brain-minicpmv-finetune-{}", std::process::id()));
    std::fs::remove_dir_all(&work).ok();
    std::fs::create_dir_all(&work).unwrap();
    // seed: a fixed dataset keeps the loss floor comparable between runs
    let mut rng = Rng::new(41);
    let train = dataset(&work, "train", 4, &mut rng);
    let held_out = dataset(&work, "held-out", 2, &mut rng);
    let out = work.join("tune");

    let opts = Options {
        decoder: PartMode::Lora { rank: 16, alpha: 32.0 },
        connector: true,
        vision: PartMode::Lora { rank: 4, alpha: 8.0 },
        epochs: 8,
        batch: 3,
        lr: 5e-4,
        ..Options::new(7)
    };
    let gpu = gpu_core::testgpu::dev(qwen35::model::pipelines());
    let job = Job { dir: &dir, base_id: common::REFERENCE, train: &train, held_out: Some(&held_out), out: &out, label: "test:codewords:v1" };
    let mut evals = Vec::new();
    let outcome = finetune(&gpu, job, &opts, &mut |e| {
        eprintln!("{e:?}");
        if let Event::Eval { loss, .. } = e {
            evals.push(*loss);
        }
    })
    .expect("fine-tune");
    let (before, after) = (outcome.initial_eval.unwrap(), outcome.final_eval.unwrap());
    assert_eq!(evals.len(), 1 + opts.epochs as usize, "held-out scored before training and after every epoch");
    assert!(after < 0.01 * before, "held-out loss {before} -> {after}: {evals:?}");

    let mut m = MiniCpmV::load_tuned(&dir, Some(&out), &gpu, 1024, 1, model::kv_tier::KvTier::F32).expect("load the tune");
    // seed: the exact-code assertions on unseen pictures were validated on this picture stream
    let mut rng = Rng::new(97);
    for (code, color) in CODES {
        let got = m.generate(&[Message::user(vec![Part::Image, Part::Text(QUESTION.into())])], &[picture(color, &mut rng)], &GenOptions { max_new: 8, ..GenOptions::default() }, &mut |_| true).unwrap();
        assert_eq!(m.decode(&got).trim(), code, "an unseen {code} picture");
    }
    std::fs::remove_dir_all(&work).ok();
}
