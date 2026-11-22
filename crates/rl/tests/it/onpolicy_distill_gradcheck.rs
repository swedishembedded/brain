// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling a large teacher model into a small student on the student's
// own samples, for its clients. If your team needs expertise in on-policy distillation of language models then you
// can procure our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "qwen3")]

//! The token-level on-policy distillation term ([`rl::objective::onpolicy_distill::token_advantage`], trained through
//! the shared policy-gradient step [`rl::objective::pg_batch::install_weights`]), differentiated with respect to the
//! LoRA adapters of a tiny plain `qwen3`, matches central finite differences of the loss the step reports. The student
//! is on-policy (its behaviour log-probabilities are its own, every ratio is one, far from a clip edge) and the teacher
//! is another policy: the same base with its adapters switched off. A teacher that is the student gives a term and a
//! gradient that are exactly zero.

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::training;
use model::{Batch, Model, IGNORE};
use qwen3::{Qwen, QwenConfig};
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;
use rl::objective::onpolicy_distill::token_advantage;
use rl::objective::pg_batch::{install_weights, with_adapters_disabled, PgBatch, PgSpec};

const T: usize = 12;
const ROWS: usize = 2;
const SPEC: PgSpec = PgSpec { band: ClipBand::new(0.2, 0.2), kl_beta: 0.0 };

struct Harness {
    m: Qwen,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    advantage: Vec<f32>,
}

/// Which policy scores the student's tokens.
#[derive(Clone, Copy)]
enum Teacher {
    /// The base with the adapters off: another policy than the student.
    Base,
    /// The student itself.
    Student,
}

impl Harness {
    fn new(seed: u64, teacher: Teacher) -> Harness {
        let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
        let init = <Qwen as Model>::init_weights(&cfg, seed);
        let mut m = Qwen::new(cfg, ROWS as u32, T as u32, &init);
        training::require_token_objectives(&mut m).unwrap().enable_weighted_loss();
        let mut rng = Rng::new(seed ^ 0x0d15);
        for name in Model::param_names(&m).into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..Model::read_weight(&m, &name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            Model::write_weight(&m, &name, &w);
        }
        let row: [u32; T] = [1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8];
        let tokens: Vec<u32> = (0..ROWS).flat_map(|r| row.iter().map(move |&t| if r == 0 { t } else { (t + 3) % 22 })).collect();
        let mut targets = vec![IGNORE; ROWS * T];
        for r in 0..ROWS {
            for t in (T - 5)..(T - 1) {
                targets[r * T + t] = tokens[r * T + t + 1];
            }
        }
        let mut h = Harness { m, tokens, targets, old_lp: Vec::new(), advantage: vec![f32::NAN; ROWS * T] };
        let student = h.logprobs();
        let teacher_lp = match teacher {
            Teacher::Base => with_adapters_disabled(&h.m, || h.logprobs()).expect("the model carries adapters"),
            Teacher::Student => student.clone(),
        };
        for i in (0..ROWS * T).filter(|&i| h.targets[i] != IGNORE) {
            h.advantage[i] = token_advantage(teacher_lp[i], student[i]);
        }
        h.old_lp = student;
        h
    }

    fn load(&self) {
        Model::set_batch(&self.m, Batch::Lm { tokens: &self.tokens, targets: &self.targets });
    }

    fn logprobs(&self) -> Vec<f32> {
        self.load();
        let _ = self.m.forward();
        training::objectives(&self.m).batch_token_logprobs()
    }

    fn denominator(&self) -> Option<f32> {
        let supervised = self.targets.iter().filter(|&&t| t != IGNORE).count();
        LossAgg::TokenSum.denominator(ROWS, supervised, 1)
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

    /// What a training step runs before its backward, divided as its gradient is.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        let batch = PgBatch { targets: &self.targets, old_lp: &self.old_lp, ref_lp: None, advantage: &self.advantage, is_weight: None };
        let denominator = self.denominator();
        let (terms, _) = install_weights(&self.m, &batch, &SPEC, denominator);
        assert!(terms.finite);
        terms.normalised_loss(denominator)
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

#[test]
fn the_token_level_distillation_term_differentiates_the_loss_it_reports() {
    let h = Harness::new(23, Teacher::Base);
    let signal: Vec<f32> = h.advantage.iter().copied().filter(|a| !a.is_nan()).collect();
    assert!(signal.iter().any(|a| *a > 0.02) && signal.iter().any(|a| *a < -0.02), "the teacher must prefer some tokens and reject others: {signal:?}");
    let report = directional_check(&h, 1.25e-3, 4, 0x0d15);
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "{:?}", failures.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "dead gradients {:?}", report.dead_gradients());
    assert!(!report.checks.is_empty());
}

#[test]
fn a_teacher_that_is_the_student_gives_a_zero_term_and_a_zero_gradient() {
    let h = Harness::new(29, Teacher::Student);
    assert!(h.advantage.iter().filter(|a| !a.is_nan()).all(|a| *a == 0.0), "{:?}", h.advantage);
    h.zero_grads();
    let loss = h.loss();
    assert_eq!(loss, 0.0);
    h.m.backward();
    h.m.poll_wait();
    let names: Vec<String> = h.param_names().into_iter().filter(|n| n.contains(".lora_")).collect();
    assert!(!names.is_empty());
    for name in names {
        assert!(h.read_grad(&name).iter().all(|g| *g == 0.0), "{name} carries a gradient");
    }
}
