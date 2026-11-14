// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines whose gradient is the
// loss they report, however the step is split, for its clients. If your team
// needs expertise in gradient accumulation and multi-device training then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: an optimiser step that accumulates micro-batches descends the
//! per-token mean over every supervised position of the step, whatever the
//! records' lengths and however the micro-batches run. The reference is one
//! forward of both records as the two rows of one batch, whose cross-entropy
//! is by definition the mean over all of its supervised positions (the C3
//! conformance check holds that gradient to finite differences). The records
//! supervise 1 and 5 positions, so a mean of the micro-batches' own means
//! weighs the short record's token three times as much as each long one's and
//! misses the reference by far more than the tolerance.
//!
//! Both accumulation paths of the shared loop's objective are held to it: one
//! model running the micro-batches in turn, and a two-stage pipeline running
//! them overlapped across its stages (`Model::accumulate_overlapped`). The
//! stages share the ambient device, so the test runs on the CPU backend.

use std::path::Path;

use model::{Batch, Model, Objective, Pipeline, PipelineModel, Shard, IGNORE};
use qwen3::{Qwen, QwenConfig};

const BLOCK: u32 = 8;
const VOCAB: u32 = 23;

/// The two records: ids and the supervision mask, one supervised position and five.
fn records() -> [(Vec<u32>, Vec<bool>); 2] {
    [(vec![3, 4, 5], vec![false, false, true]), (vec![6, 7, 8, 9, 10, 11, 12], vec![false, false, true, true, true, true, true])]
}

fn write_dataset(dir: &Path) {
    let mut split = data::chat::EncodedSplit::default();
    for (ids, mask) in records() {
        split.push(&ids, &mask);
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &split).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(VOCAB as usize)).unwrap();
}

/// The two records as the rows of one batch: position `t` predicts id `t + 1`
/// where that id is supervised; past a record the row is padding and no target.
fn reference_batch() -> (Vec<u32>, Vec<u32>) {
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    for (ids, mask) in records() {
        for t in 0..BLOCK as usize {
            tokens.push(ids.get(t).copied().unwrap_or(0));
            targets.push(if t + 1 < ids.len() && mask[t + 1] { ids[t + 1] } else { IGNORE });
        }
    }
    assert_eq!(targets.iter().filter(|&&t| t != IGNORE).count(), 6);
    (tokens, targets)
}

/// A fresh scratch directory holding the dataset.
fn dataset(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen3-accumulation-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_dataset(&dir);
    dir
}

fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: BLOCK, max_position_embeddings: BLOCK, ..QwenConfig::tiny() }
}

/// The gradient one optimiser step of `model` applies: the micro-steps of the
/// shared causal-LM objective over both records, accumulated, times the
/// `1 / grad_accum` the loop hands AdamW.
fn accumulated<M: Model>(model: &M, dir: &Path, read: impl Fn(&M, &str) -> Vec<f32>) -> Vec<(String, Vec<f32>)> {
    let opts = model::FitOpts { batch_size: 1, block_size: BLOCK, grad_accum: 2, ..Default::default() };
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, &opts).unwrap();
    assert_eq!(train.example_count(), Some(2), "one record a row");
    let mut objective = model::causal_lm::<M>(train, val, bcfg, itos);
    // seed: any; an epoch of two draws holds both records in either order
    objective.begin(5, 0);
    model.zero_grads();
    let mut rng = data::rng::Rng::new(5);
    objective.micro_steps(model, &mut rng, 2);
    model.poll_wait();
    cfg().param_list().into_iter().map(|(name, _)| (name.clone(), read(model, &name).iter().map(|g| g * 0.5).collect())).collect()
}

fn assert_matches_reference(what: &str, got: &[(String, Vec<f32>)]) {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    let reference = Qwen::new(cfg.clone(), 2, BLOCK, &init);
    let (tokens, targets) = reference_batch();
    reference.zero_grads();
    Model::set_batch(&reference, Batch::Lm { tokens: &tokens, targets: &targets });
    Model::forward(&reference);
    Model::backward(&reference);
    reference.poll_wait();
    for (name, g) in got {
        let want = reference.read_grad(name);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-8);
        let worst = g.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
        assert!(worst < 1e-4, "{what}: {name}: the step's gradient is {worst:e} (relative) off the per-token mean of the whole step");
    }
}

#[test]
fn sequential_micro_batches_descend_the_per_token_mean_of_the_step() {
    let dir = dataset("sequential");
    let cfg = cfg();
    let model = Qwen::new(cfg.clone(), 1, BLOCK, &qwen3::init_weights(&cfg, 7));
    let got = accumulated(&model, &dir, |m, n| m.read_grad(n));
    assert_matches_reference("one model", &got);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn overlapped_pipeline_micro_batches_descend_the_per_token_mean_of_the_step() {
    let dir = dataset("pipeline");
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    let shards = vec![Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU }, Shard { start: 1, end: 2, embed: false, head: true, gpu_index: Shard::ANY_GPU }];
    let pipe = PipelineModel::new(Pipeline::<Qwen>::with_shards(cfg.clone(), 1, BLOCK, &init, shards), cfg.clone());
    let got = accumulated(&pipe, &dir, |m, n| m.pipeline().reduced_grad(n));
    assert_eq!(pipe.overlapped_steps(), 1, "the micro-batches ran overlapped across the stages");
    assert_matches_reference("two-stage pipeline", &got);
    std::fs::remove_dir_all(&dir).unwrap();
}
