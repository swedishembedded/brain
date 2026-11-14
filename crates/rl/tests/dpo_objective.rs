// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end training test for `rl::objective::dpo::Dpo` (self-improve
//! roadmap P13's gate): a real `qwen3::Qwen`, a small synthetic pair set
//! (one prompt, one chosen and one rejected completion), trained for a
//! handful of optimizer steps and asserted to move the chosen sequence's
//! total log-probability UP and the rejected one's DOWN. This is NOT the
//! gradient-correctness gate (`tests/dpo_gradcheck.rs` is) - it proves the
//! pack -> forward -> pair_term -> weighted backward -> optimizer-step
//! pipeline actually does what DPO promises against a real `Model`.

use data::rng::{random_seed, Rng};
use model::Objective;
use qwen3::{Qwen, QwenConfig};
use rl::objective::dpo::{Dpo, DpoConfig, DpoPair};

fn gpu_disabled() -> bool {
    brain_testutil::gpu_tests_skipped()
}

/// Per-token `log pi(token)` for one `(prompt, completion)` span, on a
/// `Qwen` built at `b = 2` (the second row is an all-IGNORE pad row so this
/// probe can reuse the SAME training-shaped model instance without a
/// second construction) - `targets[t] == IGNORE` positions read back `0.0`,
/// matching `model::Model::batch_token_logprobs`'s own documented contract.
fn completion_token_logprobs(model: &mut Qwen, seq_len: usize, prompt: &[u32], completion: &[u32]) -> Vec<f32> {
    let mut tokens = vec![0u32; 2 * seq_len];
    let mut targets = vec![model::IGNORE; 2 * seq_len];
    let plen = prompt.len();
    for (i, &t) in prompt.iter().chain(completion.iter()).take(seq_len).enumerate() {
        tokens[i] = t;
    }
    for (i, &tok) in completion.iter().enumerate() {
        let tok_idx = plen + i;
        if tok_idx >= seq_len {
            break;
        }
        targets[tok_idx - 1] = tok;
    }
    model.set_batch(&tokens, &targets);
    let _ = model.forward();
    let lp = model.batch_token_logprobs();
    (0..completion.len())
        .map(|i| {
            let tok_idx = plen + i;
            if tok_idx >= seq_len {
                0.0
            } else {
                lp[tok_idx - 1]
            }
        })
        .collect()
}

fn completion_logprob_sum(model: &mut Qwen, seq_len: usize, prompt: &[u32], completion: &[u32]) -> f32 {
    completion_token_logprobs(model, seq_len, prompt, completion).iter().sum()
}

#[test]
fn dpo_training_raises_chosen_logprob_and_lowers_rejected_logprob() {
    if gpu_disabled() {
        return;
    }
    let seq_len = 8usize;
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 5);
    let mut model = Qwen::new(cfg, 2, seq_len as u32, &init);

    let prompt = vec![1u32, 2, 3];
    let chosen = vec![4u32, 5, 6];
    let rejected = vec![7u32, 8, 9];

    // Reference logprobs are the INITIAL policy's own per-token logprobs -
    // the standard DPO setup (reference = the pre-training checkpoint),
    // computed once here and cached inside the pair `Dpo` queues (see
    // `dpo.rs`'s own module doc comment on why this is never recomputed
    // per step).
    let ref_chosen = completion_token_logprobs(&mut model, seq_len, &prompt, &chosen);
    let ref_rejected = completion_token_logprobs(&mut model, seq_len, &prompt, &rejected);

    let before_chosen = completion_logprob_sum(&mut model, seq_len, &prompt, &chosen);
    let before_rejected = completion_logprob_sum(&mut model, seq_len, &prompt, &rejected);

    let pair = DpoPair { prompt: prompt.clone(), chosen: chosen.clone(), rejected: rejected.clone() };
    let dpo_cfg = DpoConfig { beta: 0.5, seq_len, nll_weight: 0.0, beta_ablation: false };
    let mut obj = Dpo::try_from_pairs(dpo_cfg, vec![pair], move |_pair: &DpoPair| (ref_chosen.clone(), ref_rejected.clone())).expect("a valid DPO configuration");

    obj.prepare(&mut model).unwrap();
    let mut rng = Rng::new(random_seed());
    for step in 1..=8u32 {
        model.zero_grads();
        let loss = obj.micro_step(&model, &mut rng);
        assert!(loss.is_finite(), "step {step}: DPO loss {loss} is not finite");
        model.adamw_step(step, 5e-3, 0.0, Default::default(), Some(1.0), 1.0);
        model.poll_wait();
    }

    let after_chosen = completion_logprob_sum(&mut model, seq_len, &prompt, &chosen);
    let after_rejected = completion_logprob_sum(&mut model, seq_len, &prompt, &rejected);

    assert!(
        after_chosen > before_chosen,
        "chosen sequence's total logprob should rise: before {before_chosen}, after {after_chosen}"
    );
    assert!(
        after_rejected < before_rejected,
        "rejected sequence's total logprob should fall: before {before_rejected}, after {after_rejected}"
    );
}

/// `log pi(supervised tokens)` of one rendered conversation, summed, from an independent forward: a `b = 1` model built
/// from `weights`, its logits read back and normalised on the host in f64. Token `t` is supervised when `mask[t]`, and
/// is predicted by position `t - 1`.
fn independent_logprob(cfg: &QwenConfig, weights: &std::collections::HashMap<String, Vec<f32>>, ids: &[u32], mask: &[bool]) -> f64 {
    let model = Qwen::new(cfg.clone(), 1, ids.len() as u32, weights);
    let logits = model.logits_all(ids);
    let vocab = cfg.vocab as usize;
    (1..ids.len())
        .filter(|&t| mask[t])
        .map(|t| {
            let row = &logits[(t - 1) * vocab..t * vocab];
            let top = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
            let lse = top + row.iter().map(|&v| (f64::from(v) - top).exp()).sum::<f64>().ln();
            f64::from(row[ids[t] as usize]) - lse
        })
        .sum()
}

fn weights_of(model: &Qwen) -> std::collections::HashMap<String, Vec<f32>> {
    model::Model::param_names(model).into_iter().map(|n| { let w = model.read_weight(&n); (n, w) }).collect()
}

/// The `dpo_margin` a step logs is the reference-normalised margin `(log pi_c - log ref_c) - (log pi_r - log ref_r)`
/// an independent forward measures: each conversation scored alone by a separately built model at the weights the
/// step saw, the reference by one at the weights the run started from, normalised in f64. The two rows differ in
/// length, so a margin that mixed the rows' positions, dropped the reference or summed unsupervised positions is off.
#[test]
fn the_logged_margin_is_the_margin_an_independent_forward_measures() {
    if gpu_disabled() {
        return;
    }
    let seq_len = 8usize;
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, random_seed());
    let mut model = Qwen::new(cfg.clone(), 2, seq_len as u32, &init);
    let chosen: (Vec<u32>, Vec<bool>) = (vec![1, 2, 3, 4, 5, 6], vec![false, false, false, true, true, true]);
    let rejected: (Vec<u32>, Vec<bool>) = (vec![1, 2, 3, 9, 8], vec![false, false, false, true, true]);
    let pair = rl::objective::dpo::PackedPair::from_masked(seq_len, (&chosen.0, &chosen.1), (&rejected.0, &rejected.1)).unwrap();
    let dpo_cfg = DpoConfig { beta: 0.5, seq_len, nll_weight: 0.0, beta_ablation: false };
    let mut obj = Dpo::try_from_dataset(dpo_cfg, vec![pair]).expect("a valid DPO configuration");
    obj.prepare(&mut model).unwrap();
    Objective::<Qwen>::check_data(&mut obj, &model).unwrap();

    let reference = (independent_logprob(&cfg, &init, &chosen.0, &chosen.1), independent_logprob(&cfg, &init, &rejected.0, &rejected.1));
    let mut rng = Rng::new(random_seed());
    let mut moved = 0.0f64;
    for step in 1..=4u32 {
        let seen = weights_of(&model);
        model.zero_grads();
        obj.micro_step(&model, &mut rng);
        let logged = Objective::<Qwen>::metrics(&obj).into_iter().find(|(k, _)| *k == "dpo_margin").expect("DPO logs its margin").1;
        let policy = (independent_logprob(&cfg, &seen, &chosen.0, &chosen.1), independent_logprob(&cfg, &seen, &rejected.0, &rejected.1));
        let want = (policy.0 - reference.0) - (policy.1 - reference.1);
        // Four f32 sums of a few log-probabilities each, against the same sums in f64.
        let scale = policy.0.abs() + policy.1.abs() + reference.0.abs() + reference.1.abs();
        let off = (f64::from(logged) - want).abs();
        assert!(off <= 1e-5 * scale, "step {step}: logged margin {logged}, independent {want} (off {off:e})");
        moved = moved.max(want.abs());
        model.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        model.poll_wait();
    }
    assert!(moved > 1e-2, "the steps moved the margin ({moved}), so the comparison is not of zeros");
}
