// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Talker as a trainable policy: the trainer's forward equals the inference
//! path, a policy-gradient step moves log-probabilities the way the sign of the
//! advantage says, and folding the trained adapter into the inference Talker
//! gives the weights the trainer computed with.
//!
//! Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT` like the other
//! real-checkpoint tests.

use capability::CancelToken;
use qwen3tts::engine::{Placement, ResidentEngine, Rollout};
use qwen3tts::train::{Sample, TalkerTrainer, UpdateOpts};
use qwen3tts::{GenOpts, SamplingRequest, TtsPaths};

const TEXT: &str = "The people must be taught to value their liberty above their ease.";
const BLOCK: u32 = 160;

fn paths() -> Option<TtsPaths> {
    let (Ok(weights), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
        return None;
    };
    let paths = TtsPaths::new(&weights, ckpt);
    paths.require(false).is_ok().then_some(paths)
}

fn sampled(seed: u64) -> GenOpts {
    GenOpts { max_frames: 40, seed, sampling: SamplingRequest::default(), ..GenOpts::default() }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Mean log-probability of the sampled tokens given the logits at the targets.
fn mean_logprob(logits: &[f32], targets: &[u32], vocab: usize) -> f32 {
    let mut total = 0.0;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        total += row[t as usize] - lse;
    }
    total / targets.len() as f32
}

struct Setup {
    engine: ResidentEngine,
    trainer: TalkerTrainer,
    good: Rollout,
    bad: Rollout,
}

fn setup(paths: &TtsPaths) -> Setup {
    let never = CancelToken::default();
    let mut engine = ResidentEngine::load_with(paths, Placement::Device).unwrap();
    let good = engine.rollout(TEXT, "english", None, &sampled(11), &never).unwrap();
    let bad = engine.rollout(TEXT, "english", None, &sampled(12), &never).unwrap();
    let trainer = TalkerTrainer::load(&paths.talker, BLOCK, 8, 16.0, 1).unwrap();
    Setup { engine, trainer, good, bad }
}

#[test]
fn the_trainer_forward_equals_the_inference_path_at_the_start() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let s = setup(&paths);
    let rows = s.engine.policy_rows(&s.good).unwrap();
    assert!(rows.len() <= BLOCK as usize, "the rollout fits the block: {}", rows.len());
    let inference = s.engine.score(&rows).unwrap();
    let trained = s.trainer.logits(&rows);
    let worst = max_abs(&inference, &trained);
    eprintln!("trainer vs inference logits: worst difference {worst}");
    assert!(worst < 0.05, "worst difference {worst}");
}

#[test]
fn a_step_raises_the_likelihood_of_what_earned_credit_and_lowers_what_did_not() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let s = setup(&paths);
    let mut trainer = s.trainer;
    let (good_rows, bad_rows) = (s.engine.policy_rows(&s.good).unwrap(), s.engine.policy_rows(&s.bad).unwrap());
    let vocab = trainer.logits(&good_rows).len() / good_rows.target_positions().count();
    let tokens = |rows: &qwen3tts::train::PolicyRows| -> Vec<u32> { rows.target_positions().map(|p| rows.targets[p]).collect() };
    let before = (mean_logprob(&trainer.logits(&good_rows), &tokens(&good_rows), vocab), mean_logprob(&trainer.logits(&bad_rows), &tokens(&bad_rows), vocab));

    // The reference is the policy before the update.
    let reference = |rows: &qwen3tts::train::PolicyRows| -> Vec<f32> { qwen3tts::train::logprobs(&trainer.logits(rows), rows) };
    let (ref_good, ref_bad) = (reference(&good_rows), reference(&bad_rows));
    let opts = UpdateOpts { lr: 3e-4, kl_beta: 0.0, ..UpdateOpts::default() };
    for _ in 0..6 {
        let samples = [
            Sample { rows: &good_rows, advantage: 1.0, reference_logprobs: ref_good.clone() },
            Sample { rows: &bad_rows, advantage: -1.0, reference_logprobs: ref_bad.clone() },
        ];
        trainer.update(&samples, &opts);
    }
    let after = (mean_logprob(&trainer.logits(&good_rows), &tokens(&good_rows), vocab), mean_logprob(&trainer.logits(&bad_rows), &tokens(&bad_rows), vocab));
    eprintln!("mean log-probability: good {:.4} -> {:.4}, bad {:.4} -> {:.4}", before.0, after.0, before.1, after.1);
    assert!(after.0 > before.0 + 0.01, "the credited rollout became likelier");
    assert!(after.1 < before.1 - 0.01, "the penalised rollout became less likely");
}

#[test]
fn the_folded_adapter_gives_the_inference_talker_the_trainers_weights() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let mut s = setup(&paths);
    let rows = s.engine.policy_rows(&s.good).unwrap();
    let toks: Vec<u32> = rows.target_positions().map(|p| rows.targets[p]).collect();
    let vocab = s.trainer.logits(&rows).len() / toks.len();
    let reference: Vec<f32> = vec![-3.0; toks.len()];
    for _ in 0..4 {
        let samples = [Sample { rows: &rows, advantage: 1.0, reference_logprobs: reference.clone() }];
        s.trainer.update(&samples, &UpdateOpts { lr: 3e-4, kl_beta: 0.0, ..UpdateOpts::default() });
    }
    let trained = s.trainer.logits(&rows);
    let before = s.engine.score(&rows).unwrap();
    s.engine.set_talker_weights(&s.trainer.folded_weights()).unwrap();
    let after = s.engine.score(&rows).unwrap();
    let moved = max_abs(&before, &after);
    let worst = max_abs(&after, &trained);
    eprintln!("fold: inference logits moved by {moved}; differ from the trainer's by {worst} (vocab {vocab})");
    assert!(moved > 0.05, "the update changed the policy: {moved}");
    assert!(worst < 0.05, "folded inference weights equal the trained ones: {worst}");
}

#[test]
fn a_saved_adapter_folds_back_to_the_weights_it_was_saved_from() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let mut s = setup(&paths);
    let rows = s.engine.policy_rows(&s.good).unwrap();
    let reference = vec![-1.0; rows.target_positions().count()];
    for _ in 0..3 {
        let samples = [Sample { rows: &rows, advantage: 1.0, reference_logprobs: reference.clone() }];
        s.trainer.update(&samples, &UpdateOpts { lr: 3e-4, kl_beta: 0.0, ..UpdateOpts::default() });
    }
    let path = std::env::temp_dir().join(format!("talker-adapter-{}.safetensors", std::process::id()));
    s.trainer.save_adapter(&path).unwrap();
    let live = s.trainer.folded_weights();
    let reloaded = qwen3tts::train::fold_saved_adapter(&paths.talker, &path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(live.len(), reloaded.len());
    for (name, weights) in &live {
        let worst = max_abs(weights, &reloaded[name]);
        assert!(worst < 1e-6, "{name} differs by {worst}");
    }
}
