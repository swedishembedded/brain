// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training a policy on the rollouts it served without the importance
// ratio drifting off the policy that sampled them, for its clients. If your team needs expertise in serving and
// training one model consistently then you can procure our services by sending an email to info@swedishembedded.com.

//! The behaviour policy of the policy-gradient objective.
//!
//! - Algorithmic: the importance ratio is `exp(new - behaviour)` against the behaviour log-probabilities the rollout
//!   recorded, held fixed for as long as the record is trained: over several optimiser steps on the same records the
//!   ratio a step reports is the one an independent forward measures against the record, and once the policy has left
//!   the clip band the clip engages. A run that rescored the behaviour every step (a mutant written here) would always
//!   read a ratio of one and never clip; the check rejects it.
//! - Numerical: the log-probabilities the serving path records for the tokens it samples (the engine behind the
//!   scheduler, at the request's temperature) are the trainer's log-probabilities of the same ids at step 0, under the
//!   trainer's `1 / T` logit scale, within the round-off of two kernel paths. A record scored at the wrong temperature,
//!   or shifted by one position, falls outside that band.

use model::Objective;
use qwen35::model::Qwen35;
use rl::objective::external::{BehaviourSource, ExternalPolicyGradient, PolicyGradientConfig};
use rl::objective::pg_loss::PgLoss;

use super::external_policy_gradient::{answer_logprobs, lora_model, no_kl, opts, write_dataset, Row, DRAW_SEED};

/// Ratios outside `[1 - 0.2, 1 + 0.2]` on the side the advantage pushes toward are clipped.
const EPS: f32 = 0.2;

/// Train two records of one group (rewards 1 and 0: advantages +0.5 and -0.5) for `steps` optimiser steps. Each step's
/// behaviour log-probabilities are the records' recorded ones, measured once before the first step, or (`rescore`, the
/// mutant) the current policy's, rewritten into the records before every step. Checks that every step's reported ratio
/// extremes and clip fraction are those an independent forward gives against the recorded behaviour, and that the clip
/// engages once the policy has left the band.
fn check_ratio_held_fixed(rescore: bool) -> Result<(), String> {
    let mut m = lora_model(3);
    let ids = [[1u32, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]];
    let recorded: Vec<Vec<f32>> = ids.iter().map(|ids| answer_logprobs(&m, &Row::new(ids, 3, None))).collect();
    let mut clipped_steps = 0;
    for step in 1..=10u32 {
        let now: Vec<Vec<f32>> = ids.iter().map(|ids| answer_logprobs(&m, &Row::new(ids, 3, None))).collect();
        let behaviour = if rescore { &now } else { &recorded };
        let rows: Vec<Row> = ids.iter().zip(behaviour).zip([1.0, 0.0]).map(|((ids, b), r)| Row { behavior: Some(b.clone()), ..Row::rewarded(ids, 3, r, "g") }).collect();
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path(), &rows);
        let cfg = PolicyGradientConfig { loss: PgLoss { clip_eps: EPS, ..no_kl().loss }, behaviour: BehaviourSource::Recorded, ..no_kl() };
        let mut obj = ExternalPolicyGradient::open(dir.path(), &opts(1, 2, 1e-2), cfg).unwrap();
        Objective::<Qwen35>::prepare(&mut obj, &mut m).unwrap();
        Objective::<Qwen35>::begin(&mut obj, DRAW_SEED, 0);
        m.zero_grads();
        // seed: unused by this objective, which draws its records by the order `begin` positioned
        Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut data::rng::Rng::new(1), 2);
        m.poll_wait();
        let got = obj.stats().steps[0].clone();

        // Independently: every answer token's ratio against the record, and whether its advantage's side is clipped.
        let (mut lo, mut hi, mut clipped, mut tokens) = (f32::INFINITY, f32::NEG_INFINITY, 0usize, 0usize);
        for ((n, b), adv) in now.iter().zip(&recorded).zip([0.5f32, -0.5]) {
            for (n, b) in n.iter().zip(b) {
                let r = (n - b).exp();
                (lo, hi, tokens) = (lo.min(r), hi.max(r), tokens + 1);
                clipped += usize::from((adv > 0.0 && r > 1.0 + EPS) || (adv < 0.0 && r < 1.0 - EPS));
            }
        }
        let want_clip = clipped as f32 / tokens as f32;
        if (got.ratio_min - lo).abs() > 1e-4 * lo || (got.ratio_max - hi).abs() > 1e-4 * hi {
            return Err(format!("step {step}: the step reports ratios [{}, {}], an independent forward against the recorded behaviour [{lo}, {hi}]", got.ratio_min, got.ratio_max));
        }
        if (got.clip_fraction - want_clip).abs() > 1e-6 {
            return Err(format!("step {step}: clip fraction {} against {want_clip} from the recorded behaviour (ratios [{lo}, {hi}])", got.clip_fraction));
        }
        clipped_steps += usize::from(want_clip > 0.0);
        m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 0.5);
        m.poll_wait();
    }
    if clipped_steps == 0 {
        return Err("ten steps at a learning rate of 5e-2 never left the clip band; the fixture does not exercise the clip".into());
    }
    Ok(())
}

#[test]
fn the_behaviour_policy_is_the_record_held_fixed_over_every_step_that_trains_it() {
    check_ratio_held_fixed(false).unwrap();
}

/// Mutation kill: a run that rescored the behaviour log-probabilities every step is rejected.
#[test]
fn rescoring_the_behaviour_every_step_is_rejected() {
    let why = check_ratio_held_fixed(true).expect_err("a rescored behaviour reads a ratio of one");
    println!("killed, behaviour recomputed each step: {why}");
}

#[cfg(feature = "qwen3")]
mod served {
    use model::serve::{Request, SampleParams, Scheduler};
    use model::{Batch, Model};
    use qwen3::serve::Engine;
    use qwen3::{Qwen, QwenConfig};

    const PROMPT: [u32; 4] = [1, 5, 9, 2];
    const NEW: usize = 8;

    /// Weights whose embeddings are scaled up, so the next-token distribution is far from uniform and a temperature
    /// error moves a log-probability by much more than round-off.
    fn weights(cfg: &QwenConfig, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
        let mut w = qwen3::init_weights(cfg, seed);
        let mut scaled = 0;
        for (_, v) in w.iter_mut().filter(|(n, _)| n.as_str() == "tok.weight") {
            v.iter_mut().for_each(|x| *x *= 5.0);
            scaled += 1;
        }
        assert!(scaled > 0, "the tiny decoder has an embedding to scale");
        w
    }

    /// The tokens the engine samples after `PROMPT` at temperature `temp`, and the log-probability it records for each.
    fn serve(cfg: &QwenConfig, w: &std::collections::HashMap<String, Vec<f32>>, temp: f32, seed: u64) -> (Vec<u32>, Vec<f32>) {
        let engine = Engine::from_map(cfg.clone(), w, 4, 32, 1, 12, 12, false, false);
        let mut sched = Scheduler::new(engine, 1);
        let id = sched.submit_sampled(Request { prompt: PROMPT.to_vec(), max_new: NEW, stop: vec![] }, SampleParams { temp, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, seed);
        loop {
            let report = sched.step_report();
            assert!(report.rejected.is_empty(), "{:?}", report.rejected);
            if let Some((_, tokens)) = report.completed.into_iter().find(|(i, _)| *i == id) {
                let lps = report.completed_logprobs.into_iter().find(|(i, _)| *i == id).expect("a sampled request records its log-probabilities").1;
                return (tokens, lps);
            }
        }
    }

    /// The trainer's log-probability of each of `tokens` after `PROMPT` under the logit scale `scale`, read `shift`
    /// positions off (0: the position that predicts the token).
    fn trainer(cfg: &QwenConfig, w: &std::collections::HashMap<String, Vec<f32>>, tokens: &[u32], scale: f32, shift: usize) -> Vec<f32> {
        let t = PROMPT.len() + NEW;
        let mut m = Qwen::new(cfg.clone(), 1, t as u32, w);
        let objectives = Model::token_objectives_mut(&mut m).expect("a token objective");
        objectives.set_logit_scale(scale);
        let row: Vec<u32> = PROMPT.iter().chain(tokens).copied().collect();
        let mut targets = vec![model::IGNORE; t];
        for i in 0..tokens.len() {
            targets[PROMPT.len() - 1 + i] = row[PROMPT.len() + i];
        }
        Model::set_batch(&m, Batch::Lm { tokens: &row, targets: &targets });
        let _ = Model::forward(&m);
        let lp = model::training::objectives(&m).batch_token_logprobs();
        (0..tokens.len()).map(|i| lp[PROMPT.len() - 1 + i - shift.min(PROMPT.len() - 1 + i)]).collect()
    }

    fn gap(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// The band two kernel paths of one fp32 policy may differ by on these log-probabilities.
    const BAND: f32 = 1e-4;

    #[test]
    fn the_served_behaviour_log_probabilities_are_the_trainers_at_step_zero() {
        let cfg = QwenConfig::tiny();
        let w = weights(&cfg, 21);
        for (temp, seed) in [(1.0f32, 3u64), (0.7, 4), (1.5, 5)] {
            let (tokens, served) = serve(&cfg, &w, temp, seed);
            assert_eq!(tokens.len(), NEW);
            let trained = trainer(&cfg, &w, &tokens, 1.0 / temp, 0);
            let spread = served.iter().fold(0.0f32, |a, &b| a.max(-b));
            println!("T {temp}: served {served:?}\n        trainer {trained:?}\n        largest gap {:e}", gap(&served, &trained));
            assert!(spread > 0.5, "the fixture's distribution is peaked enough to tell temperatures apart: {served:?}");
            assert!(gap(&served, &trained) <= BAND, "T {temp}: the serving path recorded {served:?}, the trainer scores {trained:?}");
            if temp != 1.0 {
                let wrong_temperature = trainer(&cfg, &w, &tokens, 1.0, 0);
                assert!(gap(&served, &wrong_temperature) > 10.0 * BAND, "T {temp}: scoring at T = 1 is not told apart: {}", gap(&served, &wrong_temperature));
            }
            let shifted = trainer(&cfg, &w, &tokens, 1.0 / temp, 1);
            assert!(gap(&served, &shifted) > 10.0 * BAND, "T {temp}: a record shifted by one position is not told apart: {}", gap(&served, &shifted));
        }
    }
}
