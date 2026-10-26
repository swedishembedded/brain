// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The trainer's forward is the reference's: on the real checkpoint, the
//! label loss `Trainer::loss` computes for the golden photo, prompt and
//! answer (the reference's own greedy reply, prompt rows unsupervised) is the
//! loss `transformers` computes for the same labels. Inference parity alone
//! does not cover this - the training path encodes and splices the image
//! through its own graph.

mod common;

use minicpmv::dataset::Sample;
use minicpmv::preprocess::{process_image, ImageOptions};
use minicpmv::prompt::{view_layout, Formatter, Message, Part, IMAGE_PAD};
use minicpmv::train::{PartMode, TrainSpec, Trainer};
use minicpmv::vision_train::train_pipelines;

#[test]
fn the_training_loss_is_the_references_label_loss() {
    let Some(fx) = common::fixture() else { return };
    let entry = &fx.manifest()["model"]["room_16x"];
    let golden = fx.tensors("model/room_16x.safetensors");
    let fmt = Formatter::from_dir(&fx.dir).unwrap();
    let opts = ImageOptions::from_dir(&fx.dir, fx.cfg.vision.patch, fx.cfg.downsample).unwrap();
    let image = process_image(&imaging::codec::load(fx.golden.join("inputs/room.png")).unwrap(), &opts);
    let rendered = fmt.render(&[Message::user(vec![Part::Image, Part::Text(entry["question"].as_str().unwrap().into())])], true, false).unwrap();
    let prompt = fmt.expand(&rendered, &[view_layout(&image, opts.downsample, IMAGE_PAD)], &[], opts.use_image_id).unwrap();
    assert_eq!(prompt.ids, common::ids(&golden["input_ids"].1), "prompt ids");

    let ids = common::ids(&golden["loss_ids"].1);
    let first = prompt.ids.len();
    let targets = (0..ids.len()).map(|i| if i + 1 >= first && i + 1 < ids.len() { ids[i + 1] } else { model::IGNORE }).collect();
    let sample = Sample { ids: ids.clone(), targets, image_runs: prompt.runs.clone(), video_runs: vec![], images: vec![image], videos: vec![] };

    let spec = TrainSpec {
        decoder: PartMode::Frozen,
        connector: true,
        vision: PartMode::Frozen,
        block: ids.len() as u32,
        downsample: opts.downsample,
        lr: minicpmv::train::Rates::uniform(0.0),
        weight_decay: 0.0,
        grad_clip: None,
        seed: data::rng::random_seed(),
    };
    let vision = minicpmv::import::vision_weights(&fx.dir, &fx.cfg).unwrap();
    let decoder = qwen35::base_source::BaseCheckpoint { cfg: fx.cfg.text.clone(), src: Box::new(minicpmv::import::open_decoder(&fx.dir, &fx.cfg).unwrap()) };
    let mut t = Trainer::new(&gpu_core::testgpu::dev(train_pipelines()), fx.cfg.clone(), &vision, decoder, spec).unwrap();
    drop(vision);
    let got = t.loss(&sample).unwrap();
    let want = golden["loss"].1[0];
    eprintln!("trainer loss {got}, reference {want}");
    assert!((got - want).abs() <= 2e-3 * want.abs().max(1.0), "trainer loss {got} vs reference {want}");
}
