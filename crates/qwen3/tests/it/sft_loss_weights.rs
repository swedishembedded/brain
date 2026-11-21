// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines that weigh parts of a
// conversation exactly as asked, and train exactly as before when not asked,
// for its clients. If your team needs expertise in supervised fine-tuning of
// language models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: per-token loss weights on a prepared chat dataset
//! (`<split>.weight.bin`, written from messages' `loss_weight`) change the
//! shared causal-LM objective's step exactly as a weighted loss over the same
//! denominator: `sum(w_t * CE_t)` over the supervised positions divided by
//! their count. Weights of one train exactly as no weights, and a weight `w`
//! on a record's positions scales that record's share of the step's gradient,
//! and of the held-out loss early stopping reads, by `w`.
//!
//! The records supervise 1 and 5 positions, so a denominator that summed the
//! weights instead of counting the positions would bend the gradient away from
//! the line through the unweighted and the record-dropped runs.

use std::path::Path;

use model::Objective;
use qwen3::{Qwen, QwenConfig};

const BLOCK: u32 = 8;
const VOCAB: u32 = 23;

/// The two records: ids and the supervision mask, one supervised position and five.
fn records() -> [(Vec<u32>, Vec<bool>); 2] {
    [(vec![3, 4, 5], vec![false, false, true]), (vec![6, 7, 8, 9, 10, 11, 12], vec![false, false, true, true, true, true, true])]
}

/// A prepared dataset whose second record's supervised tokens weigh `weight` (`None`: no weights at all), both
/// splits holding both records.
fn dataset(tag: &str, weight: Option<f32>) -> brain_testutil::Scratch {
    let dir = brain_testutil::Scratch::dir(&format!("qwen3-sft-loss-weights-{tag}"));
    let mut split = data::chat::EncodedSplit::default();
    for (at, (ids, mask)) in records().into_iter().enumerate() {
        let loss_weights = weight.map(|w| mask.iter().map(|&m| if m && at == 1 { w } else { 1.0 }).collect());
        let behavior = vec![f32::NAN; ids.len()];
        split.push_rollout(&data::chat::EncodedSample { ids, mask, behavior, loss_weights, ..Default::default() }, &[], None);
    }
    for name in ["train", "val"] {
        data::chat::write_split(dir.path(), name, &split).unwrap();
    }
    std::fs::write(dir.path().join("meta.json"), data::binio::Meta::vocab_only(VOCAB as usize)).unwrap();
    dir
}

fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: BLOCK, max_position_embeddings: BLOCK, ..QwenConfig::tiny() }
}

/// One optimiser step's gradient over both records (two micro-steps of one record, accumulated, times the
/// `1 / grad_accum` the loop hands AdamW) and the held-out loss of every validation record, from a fresh model.
fn step(dir: &Path) -> (Vec<(String, Vec<f32>)>, f32) {
    let cfg = cfg();
    let mut model = Qwen::new(cfg.clone(), 1, BLOCK, &qwen3::init_weights(&cfg, 7));
    let opts = model::FitOpts { batch_size: 1, block_size: BLOCK, grad_accum: 2, ..Default::default() };
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, &opts).unwrap();
    let mut objective = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    objective.prepare(&mut model).unwrap();
    // seed: any; an epoch of two draws holds both records in either order
    objective.begin(5, 0);
    model.zero_grads();
    objective.micro_steps(&model, &mut data::rng::Rng::new(5), 2);
    model.poll_wait();
    let grads = cfg.param_list().into_iter().map(|(name, _)| (name.clone(), model.read_grad(&name).iter().map(|g| g * 0.5).collect())).collect();
    let eval = objective.eval(&model, &mut data::rng::Rng::new(5), 0).expect("a held-out loss");
    (grads, eval)
}

/// The largest difference between two gradients, relative to the largest entry of `want`, over every parameter.
fn worst(got: &[(String, Vec<f32>)], want: &[(String, Vec<f32>)]) -> f32 {
    got.iter()
        .zip(want)
        .map(|((name, g), (other, w))| {
            assert_eq!(name, other);
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-8);
            g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
        })
        .fold(0f32, f32::max)
}

#[test]
fn weights_of_one_train_exactly_as_no_weights() {
    let plain = dataset("plain", None);
    let ones = dataset("ones", Some(1.0));
    assert!(!data::loader::split_weights_path(plain.path(), "train").exists() && !data::loader::split_weights_path(ones.path(), "train").exists(), "unit weights write no weight file");
    // The weighted path itself, at weights of one: the file written by hand.
    let tokens = data::binio::read_tokens_u32(&ones.path().join("train")).unwrap().len();
    for name in ["train", "val"] {
        data::loader::write_split_weights(ones.path(), name, Some(&vec![1.0; tokens])).unwrap();
    }
    let (want, want_eval) = step(plain.path());
    let (got, got_eval) = step(ones.path());
    assert!(worst(&got, &want) < 1e-6, "all-ones weighted gradient is {:e} (relative) off the plain one", worst(&got, &want));
    assert!((got_eval - want_eval).abs() <= 1e-6 * want_eval.abs(), "held-out loss {got_eval} vs {want_eval}");
}

#[test]
fn a_weight_scales_its_positions_share_of_the_gradient_and_the_held_out_loss() {
    let (one, dropped, weighted) = (dataset("one", None), dataset("dropped", Some(0.0)), dataset("weighted", Some(0.3)));
    let (g1, e1) = step(one.path());
    let (g0, e0) = step(dropped.path());
    let (gw, ew) = step(weighted.path());
    assert!(worst(&g0, &g1) > 1e-2, "dropping the long record changes the step");
    let line: Vec<(String, Vec<f32>)> = g0.iter().zip(&g1).map(|((name, a), (_, b))| (name.clone(), a.iter().zip(b).map(|(a, b)| a + 0.3 * (b - a)).collect())).collect();
    assert!(worst(&gw, &line) < 1e-4, "a weight of 0.3 is {:e} (relative) off 0.3 of the record's share", worst(&gw, &line));
    let expected = e0 + 0.3 * (e1 - e0);
    assert!((ew - expected).abs() < 1e-5 * expected.abs().max(1.0), "held-out loss {ew} vs {expected}");
}
