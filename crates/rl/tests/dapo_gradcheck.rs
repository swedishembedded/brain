// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning that keeps a policy exploring, for its
// clients. If your team needs expertise in reinforcement learning for language models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! L2 of DAPO's objective: z-scored group advantages through an asymmetric clip band (clip-higher, the upper edge at
//! 1.28) with every token of a step weighed alike ([`LossAgg::TokenSum`]) and no KL term, differentiated with respect
//! to the adapters of a tiny `qwen3`; it matches central finite differences of the loss a step reports (`pg_regime`).
//! Its truncated-rollout handling decides which rollouts train, not how a token is weighed, and is checked where the
//! advantages are computed (`rl/tests/it/policy_gradient_stability.rs`).

mod pg_regime;

use rl::objective::advantage::AdvantageNorm;
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;

#[test]
fn dapo_adapter_gradients_match_finite_differences() {
    let regime = pg_regime::Regime { name: "dapo", norm: AdvantageNorm::ZScore, band: ClipBand::new(0.2, 0.28), kl_beta: 0.0, loss_agg: LossAgg::TokenSum };
    pg_regime::check(regime, [1.0, 0.25]);
}

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before any test runs.
brain_testutil::configure_test_binary!();
