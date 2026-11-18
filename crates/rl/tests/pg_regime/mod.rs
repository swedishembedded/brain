// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that every reward-driven fine-tuning regime differentiates
// the objective it reports, for its clients. If your team needs expertise in gradient verification for reinforcement
// learning on language models then you can procure our services by sending an email to info@swedishembedded.com.

//! The finite-difference harness the policy-gradient regimes share (each regime's `<name>_gradcheck.rs` holds its own
//! settings): a tiny plain `qwen3` with LoRA adapters off their zero start, two rows of one group whose advantages
//! come from rewards through [`group_advantages_with`], behaviour log-probabilities offset from the model's own so
//! every ratio is known, and a loss that is [`install_weights`] itself, the function a training step runs before its
//! backward, divided as the regime's [`LossAgg`] divides it. One token of each row sits past the clip on the side
//! its advantage pushes toward, so the check also holds that a clipped token adds nothing.

#![allow(dead_code)]


use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::training;
use model::{Batch, Model, IGNORE};
use qwen3::{Qwen, QwenConfig};
use rl::objective::advantage::{group_advantages_with, AdvantageNorm};
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;
use rl::objective::pg_batch::{install_weights, PgBatch, PgSpec};

/// Positions per row.
pub const T: usize = 12;
/// How far a base-point ratio must sit from a clip edge so no finite-difference step crosses it.
const MARGIN: f32 = 0.12;

/// What a regime sets: how rewards become advantages, the clip band, the KL weight and the aggregation.
#[derive(Clone, Copy, Debug)]
pub struct Regime {
    pub name: &'static str,
    pub norm: AdvantageNorm,
    pub band: ClipBand,
    pub kl_beta: f32,
    pub loss_agg: LossAgg,
}

pub struct Harness {
    m: Qwen,
    regime: Regime,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
}

impl Harness {
    /// Two rows, each a rollout of one group with `rewards[r]`; row `r`'s supervised tokens are its last four.
    pub fn new(seed: u64, regime: Regime, rewards: [f64; 2]) -> Harness {
        let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
        let init = <Qwen as Model>::init_weights(&cfg, seed);
        let mut m = Qwen::new(cfg, 2, T as u32, &init);
        training::require_token_objectives(&mut m).unwrap().enable_weighted_loss();
        let mut rng = Rng::new(seed ^ 0x51);
        for name in Model::param_names(&m).into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..Model::read_weight(&m, &name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            Model::write_weight(&m, &name, &w);
        }
        let row_tokens: [u32; T] = [1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8];
        let tokens: Vec<u32> = (0..2).flat_map(|r| row_tokens.iter().map(move |&t| if r == 0 { t } else { (t + 3) % 22 })).collect();
        let mut targets = vec![IGNORE; 2 * T];
        for r in 0..2 {
            for t in (T - 5)..(T - 1) {
                targets[r * T + t] = tokens[r * T + t + 1];
            }
        }
        let advantages = group_advantages_with(regime.norm, &rewards);
        let mut h = Harness { m, regime, tokens, targets, old_lp: vec![0.0; 2 * T], ref_lp: vec![0.0; 2 * T], advantage: vec![f32::NAN; 2 * T] };
        h.load();
        let _ = h.m.forward();
        let probe = training::objectives(&h.m).batch_token_logprobs();
        // log(new / old) per supervised token: inside the band on both sides, and the last of each row past the edge its
        // advantage pushes toward (the first row's advantage is positive: past the upper edge; the second's negative:
        // under the lower one), where the policy term is constant.
        let offsets = [[0.04f32, -0.05, 0.02, 0.6], [-0.03, 0.05, -0.02, -0.6]];
        for r in 0..2 {
            let supervised: Vec<usize> = (0..T).map(|t| r * T + t).filter(|&i| h.targets[i] != IGNORE).collect();
            for (k, i) in supervised.into_iter().enumerate() {
                h.old_lp[i] = probe[i] - offsets[r][k];
                h.ref_lp[i] = probe[i] - 0.25;
                h.advantage[i] = advantages[r] as f32;
            }
        }
        h
    }

    fn load(&self) {
        Model::set_batch(&self.m, Batch::Lm { tokens: &self.tokens, targets: &self.targets });
    }

    fn spec(&self) -> PgSpec {
        PgSpec { band: self.regime.band, kl_beta: self.regime.kl_beta }
    }

    fn denominator(&self) -> Option<f32> {
        let supervised = self.targets.iter().filter(|&&t| t != IGNORE).count();
        self.regime.loss_agg.denominator(2, supervised, 1)
    }

    /// The base point's ratios are clear of both clip edges, and the last token of each row is past the edge its
    /// advantage pushes toward.
    fn assert_base_point(&self) {
        self.load();
        let _ = self.m.forward();
        let lp = training::objectives(&self.m).batch_token_logprobs();
        let (low, high) = (1.0 - self.regime.band.low, 1.0 + self.regime.band.high);
        for r in 0..2 {
            let supervised: Vec<usize> = (0..T).map(|t| r * T + t).filter(|&i| self.targets[i] != IGNORE).collect();
            for &i in &supervised {
                let ratio = (lp[i] - self.old_lp[i]).exp();
                assert!((ratio - low).abs() > MARGIN && (ratio - high).abs() > MARGIN, "{}: token {i}: ratio {ratio} sits within the FD-unsafe margin of a clip edge", self.regime.name);
            }
            let last = *supervised.last().unwrap();
            let ratio = (lp[last] - self.old_lp[last]).exp();
            let past = if self.advantage[last] > 0.0 { ratio > high } else { ratio < low };
            assert!(self.advantage[last] != 0.0 && past, "{}: row {r}'s last token (ratio {ratio}, advantage {}) is not past its clip edge", self.regime.name, self.advantage[last]);
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

    /// What a training step runs before its backward ([`install_weights`]), divided as its gradient is.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        let batch = PgBatch { targets: &self.targets, old_lp: &self.old_lp, ref_lp: Some(&self.ref_lp), advantage: &self.advantage, is_weight: None };
        let denominator = self.denominator();
        let (terms, _) = install_weights(&self.m, &batch, &self.spec(), denominator);
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

/// The regime's analytic adapter gradients match central finite differences of its loss.
pub fn check(regime: Regime, rewards: [f64; 2]) {
    let h = Harness::new(11, regime, rewards);
    h.assert_base_point();
    let report = directional_check(&h, 1.25e-3, 4, 0x6790);
    let worst = report.checks.iter().fold((0.0f32, 0.0f32), |(a, r), c| (a.max(c.abs_err), r.max(c.rel_err)));
    eprintln!("{}: {} adapter tensors checked, worst abs error {:e}, worst rel error {:e}", regime.name, report.checks.len(), worst.0, worst.1);
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "{}: {:?}", regime.name, failures.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "{}: dead gradients {:?}", regime.name, report.dead_gradients());
    assert!(!report.checks.is_empty());
}
