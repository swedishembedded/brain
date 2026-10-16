// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The objective-level half of the conformance suite (C7). `model::conformance`
//! checks the capability contracts every objective stands on; this checks the
//! objectives themselves on a registered model: one optimiser step of each of
//! SFT, DPO, GRPO and top-K distillation is finite and lowers that objective's
//! own loss on a two-example overfit set.
//!
//! A model that declares no token objectives is held to the declaration: every
//! rl objective must refuse it from `prepare` with the typed
//! [`model::training::Unsupported`], never panic and never train.
//!
//! Swedish Embedded AB implements solutions for verifying that every model in
//! a training stack honours the same fine-tuning contract, for its clients. If
//! your team needs expertise in conformance testing of preference optimisation
//! and reinforcement learning for language models you can procure our services
//! by sending an email to info@swedishembedded.com.

use data::rng::Rng;
use model::conformance::{batch, build, Subject, BATCH, SEQ};
use model::conformance_grad::randomize_adapters;
use model::training::{require_token_objectives, Capability, Unsupported};
use model::{Adam, Batch, Model, ModelConfig, Objective, IGNORE};

use crate::env::{Environment, Reward, Step, Task, Verifier};
use crate::objective::distill::{DistillConfig, DistillExample, DistillTopK, TopK};
use crate::objective::dpo::{weigh_pair, PackedPair};
use crate::objective::grpo::{token_term, Grpo, GrpoConfig};

/// The learning rate of the single step: large enough to move a rank-2
/// adapter measurably, small enough to stay in the loss's linear regime.
const LR: f32 = 1e-2;
const BETA: f32 = 0.5;
const CLIP_EPS: f32 = 0.2;

/// Run C7 for `subject`. `seed` draws the initial weights.
pub fn run_objectives<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    if !subject.declares.token_objectives {
        return refuses_every_objective(subject, seed);
    }
    let vocab = ModelConfig::vocab(&(subject.config)(None));
    let rank = subject.declares.adapters.then_some(model::conformance::RANK);
    let fresh = |rows: u32| -> Result<M, String> {
        let (mut m, _) = build(subject, rank, rows, seed);
        randomize_adapters(&m, seed);
        require_token_objectives(&mut m).map_err(|e| e.to_string())?.enable_weighted_loss();
        Ok(m)
    };
    let (tokens, targets) = batch(vocab);

    // SFT: plain next-token cross-entropy on the conformance batch.
    let m = fresh(BATCH)?;
    lowers(subject.name, "sft", &m, &mut || {
        m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
        let loss = m.forward();
        m.backward();
        loss
    })?;

    // DPO: row 0 is the chosen answer, row 1 the rejected one, against the
    // frozen starting model as the reference.
    let m = fresh(2)?;
    let reference = fresh(2)?;
    let row = SEQ as usize;
    let mask = vec![true; row];
    let mut packed = PackedPair::from_masked(row, (&tokens[..row], &mask), (&tokens[row..2 * row], &mask))?;
    packed.score_reference(&reference);
    lowers(subject.name, "dpo", &m, &mut || {
        let (term, _margin) = weigh_pair(BETA, 0.0, &m, &packed);
        m.backward();
        term.loss
    })?;

    // GRPO: the clipped surrogate on fixed rows with one positive and one
    // negative advantage, old log-probabilities from the starting policy.
    let m = fresh(BATCH)?;
    m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    m.forward();
    let old_lp = model::training::objectives(&m).batch_token_logprobs();
    let advantage: Vec<f32> = (0..tokens.len()).map(|i| if i < row { 1.0 } else { -1.0 }).collect();
    lowers(subject.name, "grpo", &m, &mut || {
        m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
        m.forward();
        let objectives = model::training::objectives(&m);
        let new_lp = objectives.batch_token_logprobs();
        let mut weights = vec![0.0f32; tokens.len()];
        let (mut total, mut count) = (0.0f32, 0usize);
        for i in (0..tokens.len()).filter(|&i| targets[i] != IGNORE) {
            let term = token_term(old_lp[i], new_lp[i], advantage[i], CLIP_EPS, None, 0.0);
            weights[i] = term.weight;
            total += term.loss;
            count += 1;
        }
        objectives.set_loss_weights(&weights);
        m.backward();
        total / count.max(1) as f32
    })?;

    // Distillation: a two-example set whose teacher puts all its mass on two ids.
    let m = fresh(1)?;
    let examples: Vec<DistillExample> = (0..2u32)
        .map(|e| {
            let prompt: Vec<u32> = tokens[..2].to_vec();
            let completion: Vec<u32> = (0..SEQ - 2).map(|i| (i * 3 + e + 1) % vocab).collect();
            let teacher = completion.iter().map(|&c| TopK { ids: vec![c, (c + 1) % vocab], probs: vec![0.7, 0.3] }).collect();
            DistillExample { prompt, completion, teacher }
        })
        .collect();
    let mut distill = DistillTopK::from_examples(DistillConfig { seq_len: SEQ as usize }, examples);
    let mut rng = Rng::new(seed);
    // The source cycles its two examples, so each evaluation draws both once.
    lowers(subject.name, "distill", &m, &mut || distill.micro_steps(&m, &mut rng, 2))
}

/// Zero the gradients, take the loss and its gradient from `loss_of`, apply
/// one AdamW step, and require the loss to be finite and lower afterwards.
fn lowers<M: Model>(name: &str, objective: &str, m: &M, loss_of: &mut dyn FnMut() -> f32) -> Result<(), String> {
    m.zero_grads();
    let before = loss_of();
    m.adamw_step(1, LR, 0.0, Adam::default(), None, 1.0);
    m.zero_grads();
    let after = loss_of();
    m.poll_wait();
    if !(before.is_finite() && after.is_finite()) {
        return Err(format!("{name}: {objective}: loss {before} -> {after} is not finite"));
    }
    if after >= before {
        return Err(format!("{name}: {objective}: one optimiser step did not lower the loss ({before} -> {after})"));
    }
    Ok(())
}

struct OneTask;

impl Environment for OneTask {
    fn name(&self) -> &str {
        "conformance"
    }
    fn tasks(&self, _seed: u64) -> Vec<Task> {
        vec![Task { id: "t0".into(), prompt: vec![1, 2], answer: serde_json::Value::Null }]
    }
}

struct NoReward;

impl Verifier for NoReward {
    fn verify(&self, _task: &Task, _transcript: &[Step], _completion: &[u32]) -> Reward {
        Reward { value: 0.0, parts: Default::default() }
    }
}

/// The declared-absent branch: each objective's `prepare` returns the typed
/// refusal for the token-objectives capability.
fn refuses_every_objective<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let refused = |what: &str, r: Result<(), Unsupported>| match r {
        Err(Unsupported { capability: Capability::TokenObjectives, .. }) => Ok(()),
        Err(other) => Err(format!("{}: {what} refused with {other}", subject.name)),
        Ok(()) => Err(format!("{}: declares no token objectives yet {what} accepted the model", subject.name)),
    };
    let (mut m, _) = build(subject, None, 1, seed);
    let mut distill = DistillTopK::from_examples(DistillConfig { seq_len: SEQ as usize }, vec![DistillExample { prompt: vec![0], completion: vec![1], teacher: vec![TopK::default()] }]);
    refused("distillation", Objective::<M>::prepare(&mut distill, &mut m))?;
    let grpo_cfg = GrpoConfig {
        group_size: 2,
        clip_eps: CLIP_EPS,
        kl_beta: 0.0,
        seq_len: SEQ as usize,
        rollout: model::rollout::RolloutParams { max_new: 2, sample: model::serve::SampleParams::greedy(), eos: None },
        max_attempts: 1,
    };
    let mut grpo = Grpo::new(OneTask, NoReward, grpo_cfg);
    refused("grpo", Objective::<M>::prepare(&mut grpo, &mut m))?;
    let dpo_cfg = crate::objective::dpo::DpoConfig { beta: BETA, seq_len: SEQ as usize, nll_weight: 0.0 };
    let pair = crate::objective::dpo::DpoPair { prompt: vec![0], chosen: vec![1], rejected: vec![2] };
    let mut dpo = crate::objective::dpo::Dpo::from_pairs(dpo_cfg, vec![pair], |_| (Vec::new(), Vec::new()));
    refused("dpo", Objective::<M>::prepare(&mut dpo, &mut m))?;
    Ok(())
}
