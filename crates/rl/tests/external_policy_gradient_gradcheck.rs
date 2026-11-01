// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that reward-driven fine-tuning of vision-language models
// differentiates the objective it reports, for its clients. If your team needs expertise in gradient verification for
// reinforcement learning on multimodal agents then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The policy-gradient objective of externally collected rollouts ([`rl::objective::pg_batch`]), differentiated with
//! respect to the LoRA adapters of a tiny hybrid decoder, matches central finite differences of the loss the host
//! computes from the model's own log-probabilities: the clipped surrogate with positive and negative advantages, the
//! k3 KL to a reference, and the cross-entropy of an imitation record, on a text batch and on a batch whose rows hold
//! an image (the vision tower's rows spliced over the placeholder tokens, three-axis rotary positions).
//!
//! The harness recomputes the true objective with the same `weigh` function a training step calls, so the check is of
//! the real objective and not of a stand-in. A token whose base-point ratio sits near a clip boundary would flip
//! between clipped and unclipped under an arbitrarily small perturbation, a real discontinuity rather than noise, so
//! the base point is asserted clear of both boundaries instead of loosening the tolerance.

use std::cell::Cell;

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use gpu_core::Gpu;
use model::serve::PromptEmbeds;
use model::training;
use model::{Batch, Model, IGNORE};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use rl::objective::grpo_term::{near_clip_boundary, ClipBand};
use rl::objective::pg_batch::{weigh, PgBatch, PgSpec};

const T: usize = 12;
const IMAGE: u32 = 27;
const CLIP_EPS: f32 = 0.2;
const KL_BETA: f32 = 0.3;
const MARGIN: f32 = 0.15;

struct Harness {
    m: Qwen35,
    rows: usize,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    embeds: Option<Vec<Option<PromptEmbeds>>>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
    forwarded: Cell<bool>,
}

fn spec() -> PgSpec {
    PgSpec { band: ClipBand::symmetric(CLIP_EPS), kl_beta: KL_BETA }
}

impl Harness {
    /// `advantage[r]` is row `r`'s (NaN: an imitation row); `image` puts a 2 x 2 image in every row.
    fn new(seed: u64, advantage: &[f32], image: bool) -> Harness {
        let rows = advantage.len();
        let cfg = Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
        let d = cfg.d_model as usize;
        let init = qwen35::init::init_weights(&cfg, seed);
        let mut m = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg, rows as u32, T as u32, &init);
        training::require_token_objectives(&mut m).unwrap().enable_weighted_loss();
        // Off the invisible start (B = 0): every adapter tensor carries gradient.
        let mut rng = Rng::new(seed ^ 0x51);
        for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..m.read_weight(&name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            m.write_weight(&name, &w);
        }

        // "text, a 2 x 2 image, text" when `image`, else text throughout; the answer is the last four tokens.
        let row_tokens: Vec<u32> = if image { vec![1, 2, 3, IMAGE, IMAGE, IMAGE, IMAGE, 4, 5, 6, 7, 8] } else { vec![1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8] };
        let mut tokens = Vec::new();
        for r in 0..rows {
            tokens.extend(row_tokens.iter().map(|&t| if t == IMAGE || r == 0 { t } else { (t + 3) % 27 }));
        }
        let mut targets = vec![IGNORE; rows * T];
        for r in 0..rows {
            for t in (T - 5)..(T - 1) {
                targets[r * T + t] = tokens[r * T + t + 1];
            }
        }
        let embeds = image.then(|| {
            let positions: Vec<u32> = (3..7).collect();
            let rotary = qwen3vl::mrope::get_rope_index(&row_tokens, IMAGE, &[(1, 2, 2)]);
            (0..rows)
                .map(|r| {
                    let mut rng = Rng::new(0x1A6E + r as u64);
                    let image_rows: Vec<f32> = (0..positions.len() * d).map(|_| rng.next_f32() - 0.5).collect();
                    Some(PromptEmbeds::new(d, positions.clone(), image_rows).unwrap().with_rotary_positions(rotary.clone()))
                })
                .collect()
        });
        let mut h = Harness { m, rows, tokens, targets, embeds, old_lp: vec![0.0; rows * T], ref_lp: vec![0.0; rows * T], advantage: vec![f32::NAN; rows * T], forwarded: Cell::new(false) };

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
        match &self.embeds {
            Some(embeds) => self.m.set_batch_embeds(&self.tokens, &self.targets, embeds),
            None => Model::set_batch(&self.m, Batch::Lm { tokens: &self.tokens, targets: &self.targets }),
        }
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

    /// The true objective on the host, from the forward's own log-probabilities; it also installs the token weights
    /// the backward differentiates, exactly as a training step does.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        self.forwarded.set(true);
        let objectives = training::objectives(&self.m);
        let (weights, terms) = weigh(&objectives.batch_token_logprobs(), &self.batch(), &spec());
        objectives.set_loss_weights(&weights);
        terms.loss / terms.supervised as f32
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
fn the_adapters_match_finite_differences_on_text_rows_of_positive_negative_zero_and_no_advantage() {
    check(&Harness::new(11, &[1.4, -0.9, 0.0, f32::NAN], false), "text");
}

#[test]
fn the_adapters_match_finite_differences_on_rows_that_hold_an_image() {
    check(&Harness::new(13, &[1.3, -0.8], true), "image");
}
