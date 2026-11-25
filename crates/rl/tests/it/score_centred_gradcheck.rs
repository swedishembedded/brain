// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable off-policy reinforcement learning of language models from
// asynchronously sampled rollouts for its clients. If your team needs expertise in policy-gradient estimators that stay
// unbiased when the sampler lags the learner then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The score-centred estimator's backward on real decoders: the LoRA adapters of a tiny Qwen3.5 (hybrid
//! Gated-DeltaNet) and a tiny Qwen3, differentiated through the weighted loss with the sampler's top-K as extra
//! targets in the LM head, match central finite differences of that loss.
//!
//! The estimator holds its coefficients constant (`f(r_a)` and the residuals are detached), so its gradient is the
//! gradient of `sum_i w_i (-log pi(y_i)) + sum_ij c_ij (-log pi(v_ij))` with `w` and `c` fixed at the point they were
//! computed: the harness computes them once with [`weigh`] at the base point, then the loss of every perturbed point
//! from the model's own read-back log-probabilities. The sampler's top-K lists the label, three other ids and an
//! unused slot, with log-probabilities offset from the trainer's so ratios sit on both sides of one and some leave the
//! band; the advantages are of both signs and one row imitates.

use std::cell::{Cell, RefCell};

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::training;
use model::{Batch, Model, IGNORE};
use rl::objective::grpo_term::ClipBand;
use rl::objective::pg_batch::{install_sampler_ids, weigh, PgBatch, PgEstimator, PgSpec, SamplerTop, TokenWeights};
use rl::objective::score_centring::CentringBand;

const T: usize = 12;
const K: usize = 5;
const DENOMINATOR: f32 = 9.0;

struct Harness<M: Model> {
    m: M,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    advantage: Vec<f32>,
    top_ids: Vec<u32>,
    top_lp: Vec<f32>,
    /// The estimator's coefficients at the base point, held constant.
    weights: RefCell<Option<TokenWeights>>,
    forwarded: Cell<bool>,
}

fn spec() -> PgSpec {
    PgSpec { band: ClipBand::new(0.2, 0.2), kl_beta: 0.0, estimator: PgEstimator::ScoreCentred(CentringBand { low: 0.6, high: 1.6 }) }
}

impl<M: Model> Harness<M> {
    /// `advantage[r]` is row `r`'s (NaN: an imitation row); `vocab` bounds the ids.
    fn new(mut m: M, vocab: u32, advantage: &[f32], seed: u64) -> Harness<M> {
        let rows = advantage.len();
        let objectives = training::require_token_objectives(&mut m).unwrap();
        objectives.enable_weighted_loss();
        objectives.enable_sparse_targets(K as u32).unwrap();
        // Off the invisible start (B = 0): every adapter tensor carries gradient.
        let mut rng = Rng::new(seed ^ 0x51);
        for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..m.read_weight(&name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            m.write_weight(&name, &w);
        }
        let row_tokens: [u32; T] = [1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8];
        let tokens: Vec<u32> = (0..rows).flat_map(|r| row_tokens.iter().map(move |&t| if r == 0 { t } else { (t + 3 * r as u32) % vocab })).collect();
        let mut targets = vec![IGNORE; rows * T];
        for r in 0..rows {
            for t in (T - 5)..(T - 1) {
                targets[r * T + t] = tokens[r * T + t + 1];
            }
        }
        let mut top_ids = vec![IGNORE; rows * T * K];
        for i in (0..rows * T).filter(|&i| targets[i] != IGNORE) {
            top_ids[i * K] = targets[i];
            for j in 1..K - 1 {
                top_ids[i * K + j] = (targets[i] + 2 * j as u32 + i as u32) % vocab;
            }
        }
        let mut h = Harness { m, tokens, targets, old_lp: vec![0.0; rows * T], advantage: vec![f32::NAN; rows * T], top_ids, top_lp: vec![f32::NAN; rows * T * K], weights: RefCell::new(None), forwarded: Cell::new(false) };
        // The sampler's log-probabilities are offsets from the trainer's, so ratios are known.
        h.load();
        let _ = h.m.forward();
        let o = training::objectives(&h.m);
        let (probe, probe_top) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
        let offsets = [0.3f32, -0.25, 0.8, -0.2, 0.15, -0.9, 0.35, -0.3];
        for (n, i) in (0..rows * T).filter(|&i| h.targets[i] != IGNORE).enumerate() {
            h.old_lp[i] = probe[i] - offsets[n % offsets.len()];
            h.advantage[i] = advantage[i / T];
            for j in 0..K - 1 {
                // The listed mass stays below one: the sampler's are the trainer's, shifted and shrunk.
                h.top_lp[i * K + j] = if j == 0 { h.old_lp[i] } else { probe_top[i * K + j] - offsets[(n + j) % offsets.len()] }.min(-0.05) - 0.2;
            }
        }
        h.load();
        let _ = h.m.forward();
        let (w, terms) = weigh(&training::objectives(&h.m).batch_token_logprobs(), Some(&training::objectives(&h.m).batch_sparse_target_logprobs().unwrap()), &h.batch(), &spec());
        assert!(terms.finite && terms.centred_masked > 0 && terms.centred_masked < terms.policy_tokens, "the base point must mask some tokens and keep others: {terms:?}");
        *h.weights.borrow_mut() = Some(w);
        h
    }

    /// Keep only the extra targets' part of the loss (every label weight zero): the part a backward that dropped the
    /// extra targets would get wrong, at a scale the check resolves (each listed id carries a few percent of a
    /// token's weight in the full loss).
    fn extra_only(self) -> Harness<M> {
        if let Some(w) = self.weights.borrow_mut().as_mut() {
            w.label.iter_mut().for_each(|w| *w = 0.0);
        }
        self
    }

    fn batch(&self) -> PgBatch<'_> {
        PgBatch { targets: &self.targets, old_lp: &self.old_lp, ref_lp: None, advantage: &self.advantage, is_weight: None, top: Some(SamplerTop { k: K, ids: &self.top_ids, logprobs: &self.top_lp }), support: None }
    }

    fn load(&self) {
        self.m.set_batch(Batch::Lm { tokens: &self.tokens, targets: &self.targets });
        training::objectives(&self.m).set_loss_denominator(DENOMINATOR);
        install_sampler_ids(&self.m, &self.batch(), &spec()).unwrap();
    }
}

impl<M: Model> CheckModel for Harness<M> {
    fn param_names(&self) -> Vec<String> {
        self.m.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.m.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.m.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.m.read_grad(name)
    }

    /// The weighted loss with extra targets at the frozen coefficients, from the forward's own log-probabilities; it
    /// installs those coefficients for the backward.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        self.forwarded.set(true);
        let o = training::objectives(&self.m);
        let weights = self.weights.borrow();
        let w = weights.as_ref().expect("computed at construction");
        let extra = w.extra.as_ref().expect("the score-centred estimator has extra targets");
        o.set_loss_weights(&w.label);
        o.set_sparse_target_weights(extra).unwrap();
        let (lp, top) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
        let label: f64 = lp.iter().zip(&w.label).map(|(l, w)| -f64::from(*l) * f64::from(*w)).sum();
        let listed: f64 = top.iter().zip(extra).zip(&self.top_ids).filter(|(_, id)| **id != IGNORE).map(|((l, c), _)| -f64::from(*l) * f64::from(*c)).sum();
        ((label + listed) / f64::from(DENOMINATOR)) as f32
    }

    fn zero_grads(&self) {
        self.m.zero_grads();
    }

    fn backward(&self) {
        let _ = self.loss();
        self.m.backward();
        self.m.poll_wait();
    }
}

fn check<M: Model>(h: &Harness<M>, what: &str) {
    let report = directional_check(h, 1.25e-3, 4, 0x6791);
    let worst = report.checks.iter().fold((0.0f32, 0.0f32), |(a, r), c| (a.max(c.abs_err), r.max(c.rel_err)));
    eprintln!("{what}: {} adapter tensors checked, worst abs error {:e}, worst rel error {:e}", report.checks.len(), worst.0, worst.1);
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "{what}: {:?}", failures.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "{what}: dead gradients {:?}", report.dead_gradients());
    assert!(!report.checks.is_empty());
}

#[test]
fn qwen35_adapters_match_finite_differences_of_the_score_centred_loss() {
    use qwen35::config::{lora_cfg, Qwen35Config};
    let cfg = Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
    let vocab = cfg.vocab;
    let init = qwen35::init::init_weights(&cfg, 29);
    let m = qwen35::model::Qwen35::new_train_on(gpu_core::Gpu::new_cpu(qwen35::model::pipelines()), cfg, 3, T as u32, &init);
    check(&Harness::new(m, vocab, &[1.3, -0.9, f32::NAN], 29), "qwen35 score-centred");
    let m = qwen35::model::Qwen35::new_train_on(gpu_core::Gpu::new_cpu(qwen35::model::pipelines()), Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() }, 3, T as u32, &init);
    check(&Harness::new(m, vocab, &[1.3, -0.9, f32::NAN], 29).extra_only(), "qwen35 score-centred, extra targets alone");
}

#[cfg(feature = "qwen3")]
#[test]
fn qwen3_adapters_match_finite_differences_of_the_score_centred_loss() {
    use qwen3::{Qwen, QwenConfig};
    let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
    let vocab = model::ModelConfig::vocab(&cfg);
    let init = <Qwen as Model>::init_weights(&cfg, 31);
    let m = Qwen::new(cfg, 3, T as u32, &init);
    check(&Harness::new(m, vocab, &[1.1, -1.2, f32::NAN], 31), "qwen3 score-centred");
    let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
    check(&Harness::new(Qwen::new(cfg, 3, T as u32, &init), vocab, &[1.1, -1.2, f32::NAN], 31).extra_only(), "qwen3 score-centred, extra targets alone");
}
