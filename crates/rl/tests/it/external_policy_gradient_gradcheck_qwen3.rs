// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that reward-driven fine-tuning of language models
// differentiates the objective it reports, for its clients. If your team needs expertise in gradient verification for
// reinforcement learning on agents then you can procure our services by sending an email to
// info@swedishembedded.com.

#![cfg(feature = "qwen3")]

//! The policy-gradient objective of externally collected rollouts ([`rl::objective::pg_batch`]), differentiated with
//! respect to the LoRA adapters of a tiny plain `qwen3` decoder, matches central finite differences of the loss the
//! host computes from the model's own log-probabilities: the same check as `external_policy_gradient_gradcheck` on
//! the hybrid decoder, because the objective is written once over `model::Model`. The base point is asserted clear
//! of the clip boundaries instead of loosening the tolerance.

use std::cell::Cell;

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::training;
use model::{Batch, Model, IGNORE};
use qwen3::{Qwen, QwenConfig};
use rl::objective::grpo_term::{near_clip_boundary, ClipBand};
use rl::objective::pg_batch::{weigh, PgBatch, PgSpec};

const T: usize = 12;
const CLIP_EPS: f32 = 0.2;
const KL_BETA: f32 = 0.3;
const MARGIN: f32 = 0.15;

struct Harness {
    m: Qwen,
    rows: usize,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
    forwarded: Cell<bool>,
}

impl Harness {
    /// `advantage[r]` is row `r`'s (NaN: an imitation row).
    fn new(seed: u64, advantage: &[f32]) -> Harness {
        let rows = advantage.len();
        let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
        let init = <Qwen as Model>::init_weights(&cfg, seed);
        let mut m = Qwen::new(cfg, rows as u32, T as u32, &init);
        training::require_token_objectives(&mut m).unwrap().enable_weighted_loss();
        // Off the invisible start (B = 0): every adapter tensor carries gradient.
        let mut rng = Rng::new(seed ^ 0x51);
        for name in Model::param_names(&m).into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..Model::read_weight(&m, &name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            Model::write_weight(&m, &name, &w);
        }
        let row_tokens: Vec<u32> = vec![1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8];
        let mut tokens = Vec::new();
        for r in 0..rows {
            tokens.extend(row_tokens.iter().map(|&t| if r == 0 { t } else { (t + 3) % 22 }));
        }
        let mut targets = vec![IGNORE; rows * T];
        for r in 0..rows {
            for t in (T - 5)..(T - 1) {
                targets[r * T + t] = tokens[r * T + t + 1];
            }
        }
        let mut h = Harness { m, rows, tokens, targets, old_lp: vec![0.0; rows * T], ref_lp: vec![0.0; rows * T], advantage: vec![f32::NAN; rows * T], forwarded: Cell::new(false) };
        // Behaviour and reference are offsets from the model's own log-probabilities, so the ratios are known.
        h.load();
        let _ = h.m.forward();
        let probe = training::objectives(&h.m).batch_token_logprobs();
        let offsets = [0.6f32, -0.55, 0.7, -0.5, 0.65, -0.6, 0.55, -0.7];
        for (k, i) in (0..rows * T).filter(|&i| h.targets[i] != IGNORE).enumerate() {
            h.old_lp[i] = probe[i] - offsets[k % offsets.len()];
            h.ref_lp[i] = probe[i] - 0.25;
            h.advantage[i] = advantage[i / T];
        }
        h
    }

    fn load(&self) {
        Model::set_batch(&self.m, Batch::Lm { tokens: &self.tokens, targets: &self.targets });
    }

    fn batch(&self) -> PgBatch<'_> {
        PgBatch { targets: &self.targets, old_lp: &self.old_lp, ref_lp: Some(&self.ref_lp), advantage: &self.advantage }
    }

    fn assert_base_point_is_clear_of_the_clip_boundaries(&self) {
        self.load();
        let _ = self.m.forward();
        let lp = training::objectives(&self.m).batch_token_logprobs();
        for i in (0..self.rows * T).filter(|&i| self.targets[i] != IGNORE && !self.advantage[i].is_nan() && self.advantage[i] != 0.0) {
            let ratio = (lp[i] - self.old_lp[i]).exp();
            assert!(!near_clip_boundary(ratio, CLIP_EPS, MARGIN), "token {i}: base ratio {ratio} sits within the FD-unsafe margin of a clip boundary");
        }
    }
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        Model::param_names(&self.m)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Model::read_weight(&self.m, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Model::write_weight(&self.m, name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Model::read_grad(&self.m, name)
    }

    /// The true objective on the host, from the forward's own log-probabilities; it also installs the token weights
    /// the backward differentiates, exactly as a training step does.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        self.forwarded.set(true);
        let objectives = training::objectives(&self.m);
        let (weights, terms) = weigh(&objectives.batch_token_logprobs(), &self.batch(), &PgSpec { band: ClipBand::symmetric(CLIP_EPS), kl_beta: KL_BETA });
        objectives.set_loss_weights(&weights);
        terms.loss / terms.supervised as f32
    }

    fn zero_grads(&self) {
        Model::zero_grads(&self.m);
    }

    fn backward(&self) {
        let _ = self.loss();
        self.m.backward();
        self.m.poll_wait();
    }
}

fn check(h: &Harness, what: &str) {
    h.assert_base_point_is_clear_of_the_clip_boundaries();
    let report = directional_check(h, 1.25e-3, 4, 0x6790);
    let worst = report.checks.iter().fold((0.0f32, 0.0f32), |(a, r), c| (a.max(c.abs_err), r.max(c.rel_err)));
    eprintln!("{what}: {} adapter tensors checked, worst abs error {:e}, worst rel error {:e}", report.checks.len(), worst.0, worst.1);
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "{what}: {:?}", failures.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "{what}: dead gradients {:?}", report.dead_gradients());
    assert!(!report.checks.is_empty());
}

#[test]
fn the_adapters_of_a_plain_qwen3_match_finite_differences_on_rows_of_positive_negative_zero_and_no_advantage() {
    check(&Harness::new(11, &[1.4, -0.9, 0.0, f32::NAN]), "qwen3 text");
}
