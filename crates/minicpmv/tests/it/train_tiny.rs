// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The trainer at the tiny configuration, no checkpoint needed: frozen parts
//! stay bit-identical while trained parts move, the loss falls on an overfit
//! run, and a run saved and restored halfway ends exactly where an
//! uninterrupted run does.

use data::rng::Rng;
use minicpmv::dataset::Sample;
use minicpmv::preprocess::ProcessedImage;
use minicpmv::train::{PartMode, TrainSpec, Trainer};
use minicpmv::vision::View;
use minicpmv::vision_train::train_pipelines;
use minicpmv::{Downsample, MiniCpmVConfig};

fn trainer(spec: TrainSpec) -> Trainer {
    let cfg = MiniCpmVConfig::tiny();
    // seed: fixed weights make the two halves of the resume test comparable
    let mut rng = Rng::new(17);
    let vision: std::collections::HashMap<String, Vec<f32>> = minicpmv::vision::tensor_manifest(&cfg)
        .into_iter()
        .map(|(n, k)| {
            let gain = n.contains("norm") && n.ends_with(".weight");
            (n, (0..k).map(|_| if gain { 1.0 } else { 0.4 * (rng.next_f32() - 0.5) }).collect())
        })
        .collect();
    let dec = qwen35::init::init_weights(&cfg.text, 5);
    let base = qwen35::base_source::BaseCheckpoint { cfg: cfg.text.clone(), src: Box::new(dec) };
    Trainer::new(&gpu_core::testgpu::dev(train_pipelines()), cfg, &vision, base, spec).expect("trainer")
}

fn spec(decoder: PartMode, connector: bool, vision: PartMode) -> TrainSpec {
    TrainSpec { decoder, connector, vision, block: 12, downsample: Downsample::X16, lr: minicpmv::train::Rates::uniform(2e-2), weight_decay: 0.0, grad_clip: Some(1.0), seed: 3 } // seed: the 0.3x loss-fall margin of the overfit test was tuned on this adapter init
}

fn sample() -> Sample {
    let cfg = MiniCpmVConfig::tiny();
    // seed: the 0.3x loss-fall margin of the overfit test was tuned on this patch draw
    let mut rng = Rng::new(23);
    let views = vec![View { h: 4, w: 4 }, View { h: 4, w: 8 }];
    let pp = 3 * cfg.vision.patch * cfg.vision.patch;
    let patches = (0..48 * pp).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let ids: Vec<u32> = vec![3, 7, 0, 9, 4, 0, 0, 11, 5, 8, 2, 6];
    let targets = (0..ids.len()).map(|i| if (8..ids.len()).contains(&(i + 1)) { ids[i + 1] } else { model::IGNORE }).collect();
    Sample { ids, targets, image_runs: vec![(2, 1), (5, 2)], video_runs: vec![], images: vec![ProcessedImage { views, patches, grid: None, resized: Vec::new() }], videos: vec![] }
}

fn train(t: &mut Trainer, s: &Sample, steps: usize) -> Vec<f32> {
    (0..steps)
        .map(|_| {
            let l = t.accumulate(s).unwrap();
            t.step(1.0, 1);
            l
        })
        .collect()
}

#[test]
fn frozen_parts_stay_put_and_trained_parts_move() {
    let s = sample();
    // LoRA everywhere plus the connector: bases frozen, adapters and merger move.
    let mut t = trainer(spec(PartMode::Lora { rank: 2, alpha: 4.0 }, true, PartMode::Lora { rank: 2, alpha: 4.0 }));
    let frozen_dec = "blocks.0.mlp.gate.weight";
    let (dec0, norm0, pos0, merger0) = (t.decoder.read_weight(frozen_dec), t.tower.read_weight("blocks.0.norm1.weight"), t.tower.read_weight("pos.weight"), t.tower.read_weight("merger.fc1.weight"));
    let before = t.trained_tensors();
    train(&mut t, &s, 3);
    assert_eq!(t.decoder.read_weight(frozen_dec), dec0, "the decoder base moved under LoRA");
    assert_eq!(t.tower.read_weight("blocks.0.norm1.weight"), norm0, "an untargeted tower tensor moved under LoRA");
    assert_eq!(t.tower.read_weight("pos.weight"), pos0, "the position table moved under LoRA");
    assert_ne!(t.tower.read_weight("merger.fc1.weight"), merger0, "the connector did not train");
    let after = t.trained_tensors();
    let moved = before.iter().zip(&after).filter(|((_, a), (_, b))| a != b).count();
    assert!(moved * 2 > before.len(), "only {moved} of {} trained tensors moved", before.len());

    // Full vision, frozen decoder: every decoder tensor stays, the tower moves.
    let mut t = trainer(spec(PartMode::Frozen, false, PartMode::Full));
    let dec_names: Vec<String> = ["blocks.3.self_attn.q_proj.weight", "tok.weight"].iter().map(|s| s.to_string()).collect();
    let dec0: Vec<Vec<f32>> = dec_names.iter().map(|n| t.decoder.read_weight(n)).collect();
    let (tower0, merger0) = (t.tower.read_weight("blocks.1.attn.qkv.weight"), t.tower.read_weight("merger.fc2.weight"));
    train(&mut t, &s, 2);
    for (n, w) in dec_names.iter().zip(&dec0) {
        assert_eq!(&t.decoder.read_weight(n), w, "{n} moved with the decoder frozen");
    }
    assert_ne!(t.tower.read_weight("blocks.1.attn.qkv.weight"), tower0, "the tower did not train");
    assert_eq!(t.tower.read_weight("merger.fc2.weight"), merger0, "the frozen connector moved");
}

/// The whole pipeline - full decoder, connector and tower LoRA - fits one
/// sample. (A LoRA-only decoder learns slowly on the random tiny model, as
/// `qwen35`'s own convergence test records; the real-checkpoint test is its
/// overfit gate.)
#[test]
fn an_overfit_run_drives_the_loss_down() {
    let s = sample();
    let mut t = trainer(spec(PartMode::Full, true, PartMode::Lora { rank: 2, alpha: 4.0 }));
    let losses = train(&mut t, &s, 40);
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(last < 0.3 * first, "loss {first} -> {last} over 40 steps: {losses:?}");
}

#[test]
fn a_restored_run_ends_where_an_uninterrupted_one_does() {
    let s = sample();
    let mk = || trainer(spec(PartMode::Lora { rank: 2, alpha: 4.0 }, true, PartMode::Lora { rank: 2, alpha: 4.0 }));
    let mut whole = mk();
    train(&mut whole, &s, 4);
    let mut first = mk();
    train(&mut first, &s, 2);
    let state = std::env::temp_dir().join(format!("brain-minicpmv-resume-{}.safetensors", std::process::id()));
    first.save_state(&state, 7).unwrap();
    drop(first);
    let mut second = mk();
    assert_eq!(second.restore(&state).unwrap(), 7, "the data position round-trips");
    assert_eq!(second.steps, 2);
    train(&mut second, &s, 2);
    for ((n, a), (_, b)) in whole.trained_tensors().iter().zip(second.trained_tensors().iter()) {
        assert_eq!(a, b, "{n} differs between the resumed and the uninterrupted run");
    }
    std::fs::remove_file(&state).ok();
}
