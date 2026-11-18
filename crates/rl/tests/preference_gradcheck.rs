// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for preference fine-tuning of language models with the loss that suits the
// data, for its clients. If your team needs expertise in verifying preference optimization then you can procure our
// services by sending an email to info@swedishembedded.com.

//! L2 of the pair losses beside DPO (IPO, SimPO, ORPO): the loss `rl::objective::dpo::weigh_pair` installs, over a
//! pair packed and referenced as `dpo_gradcheck.rs` packs it (the reference-free losses read none of it), matches
//! central finite differences of a tiny `qwen3`'s weights, the chosen-answer anchor on.

use std::cell::Cell;

use gradcheck::{directional_check, CheckModel};
use qwen3::{Qwen, QwenConfig};
use rl::objective::dpo::{weigh_pair, PackedPair, PairLoss};

const BETA: f32 = 0.4;
/// The chosen-answer anchor is checked together with the preference term: the
/// loss and the gradient must agree with it on.
const NLL_WEIGHT: f32 = 0.3;
const SEQ_LEN: usize = 6;

/// A tiny `qwen3::Qwen` built with `b=2` (chosen row 0, rejected row 1):
/// the same prompt, then two different supervised continuations, so the
/// prompt positions are masked exactly as a rendered preference pair's are.
struct DpoHarness {
    loss: PairLoss,
    m: Qwen,
    pair: PackedPair,
    fwd_done: Cell<bool>,
}

impl DpoHarness {
    fn new(seed: u64, loss: PairLoss) -> DpoHarness {
        let cfg = QwenConfig::tiny();
        let prompt = [3u32, 10, 17];
        let chosen: Vec<u32> = prompt.iter().copied().chain([1, 8, 15]).collect();
        let rejected: Vec<u32> = prompt.iter().copied().chain([22, 6, 13]).collect();
        let mask = [false, false, false, true, true, true];
        let mut pair = PackedPair::from_masked(SEQ_LEN, (&chosen, &mask), (&rejected, &mask)).expect("the pair fits");

        // The frozen reference is a differently initialised model, so the
        // margin `u` the check runs at is not the degenerate zero of a
        // policy scored against itself.
        let reference = Qwen::new(cfg.clone(), 2, SEQ_LEN as u32, &qwen3::init_weights(&cfg, seed + 1));
        pair.score_reference(&reference);

        let mut m = Qwen::new(cfg.clone(), 2, SEQ_LEN as u32, &qwen3::init_weights(&cfg, seed));
        m.enable_weighted_loss();
        DpoHarness { loss, m, pair, fwd_done: Cell::new(false) }
    }
}

impl CheckModel for DpoHarness {
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

    /// The objective's own forward: the TRUE `-log sigma(u)` DPO loss, with
    /// the weights backward will differentiate written to the model - see
    /// this file's own header on why `Qwen::forward()`'s own return value
    /// cannot be used here.
    fn loss(&self) -> f32 {
        let (term, _margin) = weigh_pair(self.loss, BETA, NLL_WEIGHT, &self.m, &self.pair);
        self.fwd_done.set(true);
        term.loss
    }

    fn zero_grads(&self) {
        self.m.zero_grads();
    }

    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        self.m.backward();
        self.m.poll_wait();
    }
}

#[test]
fn check_ipo_simpo_orpo_qwen3() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for loss in [PairLoss::Ipo, PairLoss::SimPo { gamma: 0.2 }, PairLoss::Orpo] {
        let h = DpoHarness::new(13, loss);
        // The step DPO's gate takes on this shape (see dpo_gradcheck.rs): each loss is smooth in the rows' sums.
        let report = directional_check(&h, 2e-3, 4, 4);
        report.print();
        let failures = report.failures(4e-3, 8e-2);
        assert!(failures.is_empty(), "{}: gradcheck failed: {failures:?}", loss.name());
        assert!(report.dead_gradients().is_empty(), "{}: dead gradients {:?}", loss.name(), report.dead_gradients());
    }
}
