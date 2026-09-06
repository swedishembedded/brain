// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DistillTopK's own local `gradcheck::CheckModel` harness (self-improve
//! roadmap P14's gate, following P12/P13's `grpo_gradcheck.rs`/
//! `dpo_gradcheck.rs` established pattern) - NOT the blanket
//! `impl<M: model::Model> CheckModel for M`. `qwen3::Qwen::forward()`'s own
//! return value, for ANY single one of the K weighted-CE passes
//! `rl::objective::distill::DistillTopK::micro_step` runs, is only that
//! ONE pass's own term - never the accumulated `KL(q_K‖p)` scalar the
//! accumulated K-pass `backward()` calls actually differentiate. This
//! harness's `loss()` instead recomputes the real accumulated
//! `Σ_k q_k·CE_k + Σ_k q_k·ln(q_k)` (`= KL(q_K‖p)`, see
//! `rl::objective::distill`'s own module doc comment for the derivation) on
//! the host, replaying the exact same `rl::objective::distill::
//! pass_arrays`/`neg_entropy` calls `DistillTopK::micro_step` calls - so
//! finite differences test the real objective a training run would
//! differentiate, not a stand-in for it.
//!
//! `KL(q_K‖p)` is smooth everywhere `p` is nonzero (which it always is for a
//! softmax) - there is no piecewise-constant clip boundary to guard finite
//! differences away from here, unlike GRPO's clipped surrogate.

use std::cell::Cell;

use gradcheck::{directional_check, CheckModel};
use qwen3::{Qwen, QwenConfig};
use rl::objective::distill::{neg_entropy, pass_arrays, TopK};

const SEQ_LEN: usize = 6;
const N: usize = SEQ_LEN;

/// A tiny `qwen3::Qwen` built with `b=1` - unlike DPO/GRPO's `b=2`/pairwise
/// packing, DistillTopK's weight formula never reads the model's own current
/// predictions (see the module doc comment on `rl::objective::distill`'s
/// `pass_arrays`: the weight is `teacher_prob·count`, pure data, no
/// current-policy readback), so there is no packed-row shape constraint
/// forcing more than one sequence per forward here.
struct DistillHarness {
    m: Qwen,
    tokens: Vec<u32>,
    /// 3 of `N`'s 6 positions carry a (small, `K=3`) teacher top-K
    /// distribution; the rest are prompt-only (`TopK::default()`, excluded
    /// from every pass) - exercising the "not every position has the full
    /// `K`" ragged-count path `pass_arrays` itself handles.
    teacher: Vec<TopK>,
    k_max: usize,
    fwd_done: Cell<bool>,
}

impl DistillHarness {
    fn new(seed: u64) -> DistillHarness {
        let cfg = QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, seed);
        let mut m = Qwen::new(cfg, 1, SEQ_LEN as u32, &init);
        m.enable_weighted_loss();

        let tokens: Vec<u32> = (0..N as u32).map(|i| (i * 5 + 2) % 23).collect();

        // Positions 0..2 are prompt-only (no teacher signal); positions 3..5
        // each carry a 3-of-23 top-K teacher distribution, renormalized to
        // sum to 1.0 (the keystone identity's own requirement).
        let mut teacher = vec![TopK::default(); N];
        let dists: [[(u32, f32); 3]; 3] = [
            [(1, 0.5), (7, 0.3), (13, 0.2)],
            [(2, 0.6), (9, 0.25), (15, 0.15)],
            [(3, 0.4), (11, 0.4), (17, 0.2)],
        ];
        for (i, dist) in dists.iter().enumerate() {
            teacher[3 + i] = TopK { ids: dist.iter().map(|&(id, _)| id).collect(), probs: dist.iter().map(|&(_, p)| p).collect() };
        }

        DistillHarness { m, tokens, teacher, k_max: 3, fwd_done: Cell::new(false) }
    }
}

impl CheckModel for DistillHarness {
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

    /// Recomputes the TRUE accumulated `KL(q_K‖p)` on the host - see this
    /// file's own header on why any single pass's `Qwen::forward()` return
    /// cannot be used here.
    fn loss(&self) -> f32 {
        let mut total = 0.0f32;
        for k in 0..self.k_max {
            let (targets, weights, _count) = pass_arrays(&self.teacher, k);
            self.m.set_batch(&self.tokens, &targets);
            self.m.write_weights(&weights);
            total += self.m.forward();
        }
        self.fwd_done.set(true);
        let entropy: f32 = self.teacher.iter().map(neg_entropy).sum();
        total + entropy
    }

    fn zero_grads(&self) {
        self.m.zero_grads();
    }

    /// Replays all `k_max` passes' forward AND backward, accumulating the
    /// gradient across passes exactly as `DistillTopK::micro_step` does (no
    /// `zero_grads` between passes) - `Model::backward`'s own contract
    /// ("Accumulate analytic gradients ... into the ParamStore") is what
    /// makes K independent backward calls sum to the keystone identity's
    /// `Σ_k q_k·(p − e_{v_k})` rather than overwriting each other.
    fn backward(&self) {
        for k in 0..self.k_max {
            let (targets, weights, _count) = pass_arrays(&self.teacher, k);
            self.m.set_batch(&self.tokens, &targets);
            self.m.write_weights(&weights);
            let _ = self.m.forward();
            self.m.backward();
        }
        self.fwd_done.set(true);
        self.m.poll_wait();
    }
}

#[test]
fn check_distill_topk_qwen3() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let h = DistillHarness::new(17);

    // Fixed seed at a verified comfortable-margin value - the same practice
    // every other gradcheck entry point in this workspace already follows
    // (see `dpo_gradcheck.rs`'s own doc comment on this exact tiny-model
    // shape's inherent FD noise floor).
    let report = directional_check(&h, 5e-3, 4, 4);
    report.print();
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "DistillTopK gradcheck failed: {failures:?}");
    assert!(report.dead_gradients().is_empty(), "DistillTopK gradcheck found dead (all-zero analytic, nonzero numeric) gradients: {:?}", report.dead_gradients());
}
