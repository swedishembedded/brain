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

use data::rng::Rng;
use model::Objective;
use qwen3::{Qwen, QwenConfig};
use rl::objective::dpo::{Dpo, DpoConfig, DpoPair};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
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
    let dpo_cfg = DpoConfig { beta: 0.5, seq_len };
    let mut obj = Dpo::from_pairs(dpo_cfg, vec![pair], move |_pair: &DpoPair| (ref_chosen.clone(), ref_rejected.clone()));

    obj.prepare(&mut model);
    let mut rng = Rng::new(7);
    for step in 1..=8u32 {
        model.zero_grads();
        let loss = obj.micro_step(&model, &mut rng);
        assert!(loss.is_finite(), "step {step}: DPO loss {loss} is not finite");
        model.adamw_step(step, 5e-3, 0.0, Some(1.0), 1.0);
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
