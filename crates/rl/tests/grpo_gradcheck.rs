// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GRPO's own local `gradcheck::CheckModel` harness (self-improve roadmap
//! P12's gate) - NOT the blanket `impl<M: model::Model> CheckModel for M`.
//! `qwen3::Qwen::forward()`'s own return value is the model's weighted-CE
//! forward computed with whatever loss weights already happen to be on the
//! device (default/stale) - not the scalar `backward()` ends up
//! differentiating once GRPO's true per-token weight is known. This
//! harness's `loss()` instead recomputes the TRUE clipped surrogate (plus
//! the optional k3 KL term) on the host, from the model's own freshly
//! forwarded per-token logprobs, calling the exact same
//! `rl::objective::grpo::token_term` function
//! `Grpo::micro_step` calls - so finite differences test the real objective
//! a training run would differentiate, not a stand-in for it.
//!
//! ## The clip-boundary gotcha
//!
//! `token_term`'s clip indicator is piecewise-constant in the importance
//! ratio `r = exp(new_lp - old_lp)`: a token whose BASE-point ratio sits
//! within `eps` of `1 - clip_eps` or `1 + clip_eps` can flip between
//! "clipped" (zero gradient) and "unclipped" (raw `advantage*ratio`
//! gradient) under an arbitrarily small parameter perturbation - a genuine
//! discontinuity in the objective, not finite-difference noise. Rather than
//! loosen `directional_check`'s tolerance to paper over the resulting
//! mismatch, this test asserts every sampled (non-KL-only) token's BASE
//! ratio sits outside a comfortable margin around both boundaries BEFORE
//! ever calling `directional_check`.

use std::cell::Cell;

use gradcheck::{directional_check, CheckModel};
use qwen3::{Qwen, QwenConfig};
use rl::objective::grpo::{near_clip_boundary, token_term};

const CLIP_EPS: f32 = 0.2;
const KL_BETA: f32 = 0.3;
/// How far a base-point ratio must sit from either clip boundary. Chosen
/// well above `directional_check`'s own weight-space step (`eps = 5e-3`
/// times up to a handful of dims) so no perturbation in this test can ever
/// cross a boundary the base point didn't already sit inside.
const MARGIN: f32 = 0.15;

/// Two rows of a tiny `qwen3::Qwen`, each with 5 active (non-IGNORE)
/// positions - the same tiny-model shape as every other qwen3 gradcheck
/// entry point (`gradcheck::check_qwen3_weighted` et al). `advantage`
/// spreads across positive, negative, and exact-zero (KL-only) so every
/// branch of `token_term` is exercised, not just "positive, unclipped".
struct GrpoHarness {
    m: Qwen,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
    fwd_done: Cell<bool>,
}

const N: usize = 12; // 2 rows x 6 positions

impl GrpoHarness {
    fn new(seed: u64) -> GrpoHarness {
        let cfg = QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, seed);
        let mut m = Qwen::new(cfg, 2, 6, &init);
        m.enable_weighted_loss();

        let tokens: Vec<u32> = (0..N as u32).map(|i| (i * 5 + 1) % 23).collect();
        let mut targets = vec![model::IGNORE; N];
        for row in 0..2usize {
            for t in 0..5usize {
                targets[row * 6 + t] = tokens[row * 6 + t + 1];
            }
        }

        // Probe the model's OWN initial per-token logprobs, so `old_lp`/
        // `ref_lp` are defined as offsets from a REAL base point rather than
        // hand-picked numbers that might happen to coincide with it.
        m.set_batch(&tokens, &targets);
        let _ = m.forward();
        let probe = m.batch_token_logprobs();

        // log(ratio) offsets per active position (row 0: t=0..4, row 1:
        // t=0..4) - a mix comfortably on both sides of both clip boundaries
        // (ratio = exp(log_ratio), eps = 0.2 => boundaries at 0.8 / 1.2).
        let log_ratio = [0.6_f32, -0.55, 0.7, -0.5, 0.65, 0.0, 0.6, -0.6, 0.7, -0.55];
        let advantage = [1.4_f32, -0.9, 1.6, -1.1, 0.0, 1.3, -0.8, 1.4, -1.0, 0.0];

        let mut old_lp = vec![0f32; N];
        let mut ref_lp = vec![0f32; N];
        let mut active = 0usize;
        for i in 0..N {
            if targets[i] == model::IGNORE {
                continue;
            }
            old_lp[i] = probe[i] - log_ratio[active];
            ref_lp[i] = probe[i] - 0.25; // a fixed frozen-reference offset
            active += 1;
        }
        assert_eq!(active, log_ratio.len(), "expected exactly {} active positions", log_ratio.len());

        let mut advantage_row = vec![0f32; N];
        active = 0;
        for i in 0..N {
            if targets[i] == model::IGNORE {
                continue;
            }
            advantage_row[i] = advantage[active];
            active += 1;
        }

        GrpoHarness { m, tokens, targets, old_lp, ref_lp, advantage: advantage_row, fwd_done: Cell::new(false) }
    }

    /// Base-point ratios at the CURRENT model weights - recomputed fresh
    /// (rather than reusing the construction-time `probe`) so the boundary
    /// margin check always reflects the actual point `directional_check`'s
    /// finite differences are about to perturb around.
    fn base_ratios(&self) -> Vec<f32> {
        self.m.set_batch(&self.tokens, &self.targets);
        let _ = self.m.forward();
        let new_lp = self.m.batch_token_logprobs();
        (0..N)
            .map(|i| if self.targets[i] == model::IGNORE { 1.0 } else { (new_lp[i] - self.old_lp[i]).exp() })
            .collect()
    }
}

impl CheckModel for GrpoHarness {
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

    /// Recomputes the TRUE clipped surrogate (+ k3 KL) on the host - see
    /// this file's own header on why `Qwen::forward()`'s own return value
    /// cannot be used here.
    fn loss(&self) -> f32 {
        self.m.set_batch(&self.tokens, &self.targets);
        let _ = self.m.forward();
        self.fwd_done.set(true);
        let new_lp = self.m.batch_token_logprobs();

        let mut weights = vec![0f32; N];
        let mut total = 0.0f32;
        let mut count = 0usize;
        for i in 0..N {
            if self.targets[i] == model::IGNORE {
                continue;
            }
            count += 1;
            let term = token_term(self.old_lp[i], new_lp[i], self.advantage[i], CLIP_EPS, Some(self.ref_lp[i]), KL_BETA);
            weights[i] = term.weight;
            total += term.loss;
        }
        self.m.write_weights(&weights);
        total / count as f32
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
fn grpo_analytic_grads_match_finite_differences_on_the_true_clipped_surrogate() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let h = GrpoHarness::new(11);

    // The gotcha (see this file's header): guard the base point away from
    // the clip boundary instead of loosening tolerance. Positions with
    // advantage == 0.0 are KL-only - `token_term`'s clip branch never runs
    // for them, so they need no such guard.
    let ratios = h.base_ratios();
    for (i, &ratio) in ratios.iter().enumerate() {
        if h.targets[i] == model::IGNORE || h.advantage[i] == 0.0 {
            continue;
        }
        assert!(
            !near_clip_boundary(ratio, CLIP_EPS, MARGIN),
            "token {i}: base ratio {ratio} sits within the FD-unsafe clip-boundary margin - pick a different log_ratio offset"
        );
    }

    let report = directional_check(&h, 5e-3, 4, 11 ^ 0x6790);
    report.print();
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "GRPO clipped-surrogate gradcheck failed: {failures:?}");
    assert!(report.dead_gradients().is_empty(), "GRPO clipped-surrogate gradcheck found dead (all-zero analytic, nonzero numeric) gradients: {:?}", report.dead_gradients());
}
