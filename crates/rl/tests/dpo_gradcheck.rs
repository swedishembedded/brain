// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DPO's own local `gradcheck::CheckModel` harness (self-improve roadmap
//! P13's gate, following P12's `grpo_gradcheck.rs` established pattern) -
//! NOT the blanket `impl<M: model::Model> CheckModel for M`.
//! `qwen3::Qwen::forward()`'s own return value is the model's plain
//! (default- or stale-weighted) forward loss - not the true DPO scalar
//! `-log sigma(u)` `backward()` ends up differentiating once the pair's
//! real per-row weight is known. This harness's `loss()` is the objective's
//! own forward instead: `rl::objective::dpo::weigh_pair`, the function
//! `Dpo::micro_step` runs before its backward, over a pair packed by
//! `PackedPair::from_masked` and referenced by `PackedPair::score_reference`
//! against a separately initialised frozen model - so finite differences
//! test the real objective a training run would differentiate, not a
//! stand-in for it.
//!
//! Unlike GRPO's clipped surrogate, `-log sigma(u)` is smooth everywhere in
//! `u` - there is no piecewise-constant clip boundary to guard finite
//! differences away from here.

use std::cell::Cell;

use gradcheck::{directional_check, CheckModel};
use qwen3::{Qwen, QwenConfig};
use rl::objective::dpo::{weigh_pair, PackedPair};

const BETA: f32 = 0.4;
const SEQ_LEN: usize = 6;

/// A tiny `qwen3::Qwen` built with `b=2` (chosen row 0, rejected row 1):
/// the same prompt, then two different supervised continuations, so the
/// prompt positions are masked exactly as a rendered preference pair's are.
struct DpoHarness {
    m: Qwen,
    pair: PackedPair,
    fwd_done: Cell<bool>,
}

impl DpoHarness {
    fn new(seed: u64) -> DpoHarness {
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
        DpoHarness { m, pair, fwd_done: Cell::new(false) }
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
        let (term, _margin) = weigh_pair(BETA, &self.m, &self.pair);
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
fn check_dpo_qwen3() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let h = DpoHarness::new(13);

    // `eps = 2e-3`, not the `5e-3` other gates on this shape use: against a
    // reference from a different initialisation the margin sits where
    // `-log sigma(u)` is curved, and the central difference's truncation
    // error grows with `eps`. Over direction seeds 1..=5 the worst relative
    // error measured 1.5e-2..3.2e-2 at `eps = 2e-3`, 5.9e-2..9.3e-2 at
    // `5e-3` and 1.9e-1..2.6e-1 at `1e-2` - shrinking with `eps`, which is
    // what a correct analytic gradient under FD truncation error does.
    let report = directional_check(&h, 2e-3, 4, 4);
    report.print();
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "DPO gradcheck failed: {failures:?}");
    assert!(report.dead_gradients().is_empty(), "DPO gradcheck found dead (all-zero analytic, nonzero numeric) gradients: {:?}", report.dead_gradients());
}
