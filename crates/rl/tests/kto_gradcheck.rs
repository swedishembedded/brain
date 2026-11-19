// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for aligning a model with a product's own thumbs-up and thumbs-down
// feedback, for its clients. If your team needs expertise in verifying feedback fine-tuning then you can procure our
// services by sending an email to info@swedishembedded.com.

//! L2 of KTO: the loss `rl::objective::feedback::weigh_feedback` installs (the function a micro-step runs before its
//! backward), over an example packed with its mismatched row and referenced against a separately initialised frozen
//! model, matches central finite differences of a tiny `qwen3`'s weights, for a desirable and an undesirable example.
//! The reference point is held fixed while the weights move, as the objective holds it (no gradient flows through it).

use std::cell::Cell;

use gradcheck::{directional_check, CheckModel};
use qwen3::{Qwen, QwenConfig};
use rl::objective::feedback::{weigh_feedback, FeedbackConfig, PackedFeedback, FEEDBACK_ROWS};

const SEQ_LEN: usize = 6;
const CFG: FeedbackConfig = FeedbackConfig { beta: 0.4, desirable_weight: 1.0, undesirable_weight: 1.3, seq_len: SEQ_LEN };
/// The reference point, held: a value the log-ratio of the fixture sits on neither side of exactly.
const Z: f32 = 0.3;

struct KtoHarness {
    m: Qwen,
    example: PackedFeedback,
    forwarded: Cell<bool>,
}

impl KtoHarness {
    fn new(seed: u64, desirable: bool) -> KtoHarness {
        let cfg = QwenConfig::tiny();
        let prompt = [3u32, 10, 17];
        let own: Vec<u32> = prompt.iter().copied().chain([1, 8, 15]).collect();
        let mismatched: Vec<u32> = prompt.iter().copied().chain([22, 6, 13]).collect();
        let mask = [false, false, false, true, true, true];
        let mut example = PackedFeedback::from_masked(SEQ_LEN, (&own, &mask), (&mismatched, &mask), desirable).expect("the example fits");
        // A differently initialised reference, so the log-ratio is not the degenerate zero of a policy against itself.
        let reference = Qwen::new(cfg.clone(), FEEDBACK_ROWS, SEQ_LEN as u32, &qwen3::init_weights(&cfg, seed + 1));
        example.score_reference(&reference);
        let mut m = Qwen::new(cfg.clone(), FEEDBACK_ROWS, SEQ_LEN as u32, &qwen3::init_weights(&cfg, seed));
        m.enable_weighted_loss();
        KtoHarness { m, example, forwarded: Cell::new(false) }
    }
}

impl CheckModel for KtoHarness {
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

    /// The objective's own forward: the KTO loss of the example against the held reference point, with the weights the
    /// backward differentiates written to the model.
    fn loss(&self) -> f32 {
        let (term, _) = weigh_feedback(&CFG, &self.m, &self.example, Z);
        self.forwarded.set(true);
        term.loss
    }

    fn zero_grads(&self) {
        self.m.zero_grads();
    }

    fn backward(&self) {
        if !self.forwarded.get() {
            let _ = self.loss();
        }
        self.m.backward();
        self.m.poll_wait();
    }
}

#[test]
fn kto_gradients_match_finite_differences_for_desirable_and_undesirable_examples() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for desirable in [true, false] {
        let h = KtoHarness::new(13, desirable);
        // The same step as DPO's gate on this shape: the loss is a sigmoid of a sum of log-probabilities, curved where a
        // reference from another initialisation puts it.
        let report = directional_check(&h, 2e-3, 4, 4);
        report.print();
        let failures = report.failures(4e-3, 8e-2);
        assert!(failures.is_empty(), "KTO (desirable {desirable}) gradcheck failed: {failures:?}");
        assert!(report.dead_gradients().is_empty(), "KTO (desirable {desirable}): dead gradients {:?}", report.dead_gradients());
    }
}
