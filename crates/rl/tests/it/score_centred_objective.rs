// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable off-policy reinforcement learning of language models from
// asynchronously sampled rollouts for its clients. If your team needs expertise in policy-gradient estimators that stay
// unbiased when the sampler lags the learner then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The recorded-rollout objective with `estimator: score-centred`: it reads the records' sampler top-K into its
//! batches, enables the model's extra targets with the records' K, trains, and reports the sampler's listed mass and
//! the share of masked tokens; it refuses, by name, every run it cannot centre: records without a top-K, a sampler that
//! truncated its distribution, an importance correction, a top-K id past the vocabulary, and a model whose LM head
//! takes no extra targets.

use data::rng::Rng;
use model::training::Capability;
use model::{Model, Objective, StepVerdict};
use qwen35::model::Qwen35;
use rl::objective::external::{ExternalPolicyGradient, ImportanceCorrection, PolicyGradientConfig};
use rl::objective::pg_loss::{Estimator, PgLoss};

use super::external_policy_gradient::{answer_logprobs, lora_model, opts, write_dataset, Row, DRAW_SEED};

const K: usize = 3;

fn score_centred() -> PolicyGradientConfig {
    PolicyGradientConfig { loss: PgLoss { estimator: Estimator::ScoreCentred, kl_beta: 0.0, ..PgLoss::RECORDED }, ..Default::default() }
}

/// A rollout sampled by the model it trains (behaviour = the model's own log-probabilities), its top-K three other
/// ids at the sampled token's probability.
fn on_policy(m: &Qwen35, ids: &[u32], advantage: f32) -> Row {
    let mut row = Row::new(ids, 3, Some(advantage));
    let now = answer_logprobs(m, &row);
    let top_ids: Vec<u32> = (0..now.len()).flat_map(|t| (0..K as u32).map(move |j| (t as u32 * 5 + j * 7 + 1) % 29)).collect();
    let top_lps: Vec<f32> = now.iter().flat_map(|&lp| std::iter::repeat_n(lp, K)).collect();
    row.behavior = Some(now);
    row.top = Some((K, top_ids, top_lps));
    row
}

fn open(rows: &[Row], cfg: PolicyGradientConfig) -> std::io::Result<(ExternalPolicyGradient, tempfile::TempDir)> {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), rows);
    let obj = ExternalPolicyGradient::open(dir.path(), &opts(1, 1, 1e-2), cfg)?;
    Ok((obj, dir))
}

#[test]
fn a_score_centred_step_trains_on_the_records_top_k_and_reports_its_mass() {
    let mut m = lora_model(3);
    let rows = [on_policy(&m, &[1, 2, 3, 4, 5, 6], 1.2), on_policy(&m, &[7, 8, 9, 10, 11], -0.8)];
    let (mut obj, _dir) = open(&rows, score_centred()).expect("score-centred records open");
    Objective::<Qwen35>::prepare(&mut obj, &mut m).expect("qwen35 takes extra targets");
    Objective::<Qwen35>::check_data(&mut obj, &m).expect("every id is in the vocabulary");
    Objective::<Qwen35>::begin(&mut obj, DRAW_SEED, 0);
    m.zero_grads();
    let loss = Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut Rng::new(1), 2);
    m.poll_wait();
    assert!(matches!(Objective::<Qwen35>::step_verdict(&mut obj), StepVerdict::Apply));
    assert!(loss.is_finite());
    let grad: f32 = m.param_names().iter().map(|n| m.read_grad(n).iter().map(|g| g.abs()).sum::<f32>()).sum();
    assert!(grad > 0.0 && grad.is_finite(), "the step has a gradient: {grad}");
    let step = &obj.stats().steps[0];
    // Three ids at the sampled token's probability: the sampler's listed mass is three of its probabilities.
    let expected: f32 = rows.iter().flat_map(|r| r.behavior.clone().unwrap()).map(|lp| K as f32 * lp.exp()).sum::<f32>() / rows.iter().map(|r| r.behavior.as_ref().unwrap().len()).sum::<usize>() as f32;
    assert!((step.sampler_mass - expected).abs() < 1e-4 * expected.max(1.0), "sampler mass {} vs {expected}", step.sampler_mass);
    assert!(step.sampler_mass > 0.0 && step.sampler_mass <= 1.0 && (0.0..=1.0).contains(&step.centred_masked_fraction), "{step:?}");
    assert_eq!(step.centred_masked_fraction, 0.0, "on-policy rows have a ratio of one: {step:?}");
    let metrics: Vec<&str> = Objective::<Qwen35>::metrics(&obj).iter().map(|(n, _)| *n).collect();
    assert!(metrics.contains(&"pg_sampler_mass") && metrics.contains(&"pg_centred_masked_fraction"), "{metrics:?}");
}

#[test]
fn a_run_the_estimator_cannot_centre_is_refused_by_name() {
    let m = lora_model(5);
    let rows = [on_policy(&m, &[1, 2, 3, 4, 5, 6], 1.0), on_policy(&m, &[7, 8, 9, 10, 11], -1.0)];
    let refused = |rows: &[Row], cfg: PolicyGradientConfig| open(rows, cfg).err().expect("refused").to_string();

    let bare: Vec<Row> = rows.iter().map(|r| Row { top: None, ..r.clone() }).collect();
    let why = refused(&bare, score_centred());
    assert!(why.contains("score-centred") && why.contains("behavior_top_ids"), "{why}");

    let cut = data::chat_rollout::SamplerDecl { top_p: Some(0.9), ..Default::default() };
    let truncating: Vec<Row> = rows.iter().map(|r| Row { sampler: Some(cut), ..r.clone() }).collect();
    let why = refused(&truncating, score_centred());
    assert!(why.contains("truncated") && why.contains("top_p"), "{why}");
    let whole = data::chat_rollout::SamplerDecl { temperature: Some(1.0), ..Default::default() };
    assert!(open(&rows.iter().map(|r| Row { sampler: Some(whole), ..r.clone() }).collect::<Vec<_>>(), score_centred()).is_ok(), "a sampler that only sets a temperature does not truncate");

    let why = refused(&rows, PolicyGradientConfig { importance: ImportanceCorrection::Truncate { cap: 2.0 }, ..score_centred() });
    assert!(why.contains("score-centred") && why.contains("importance"), "{why}");

    // An id past the model's 29 rows.
    let mut past = rows.clone();
    past[0].top.as_mut().unwrap().1[0] = 29;
    let (mut obj, _dir) = open(&past, score_centred()).expect("the data crate does not know the vocabulary");
    let why = Objective::<Qwen35>::check_data(&mut obj, &m).unwrap_err().to_string();
    assert!(why.contains("behavior_top_ids") && why.contains("29"), "{why}");
}

/// gpt2's trainer has no chunked LM head, so it takes no extra targets: preparing the run refuses it by capability.
#[test]
fn a_model_without_extra_targets_in_its_head_is_refused() {
    let m = lora_model(7);
    let rows = [on_policy(&m, &[1, 2, 3, 4, 5, 6], 1.0), on_policy(&m, &[7, 8, 9, 10, 11], -1.0)];
    let (mut obj, _dir) = open(&rows, score_centred()).unwrap();
    let cfg = gpt2::GptConfig { vocab: 29, block_size: 8, n_layers: 1, d_model: 16, n_heads: 2, d_ff: 32 };
    let init = <gpt2::Gpt as Model>::init_weights(&cfg, 1);
    let mut g = gpt2::Gpt::new(cfg, 1, 8, &init);
    let refusal = Objective::<gpt2::Gpt>::prepare(&mut obj, &mut g).unwrap_err();
    assert_eq!(refusal.capability, Capability::SparseTargets, "{refusal}");
}
