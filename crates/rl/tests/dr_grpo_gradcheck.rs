// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning whose gradient carries no length bias, for its
// clients. If your team needs expertise in reinforcement learning for language models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! L2 of Dr. GRPO: mean-centred advantages (no division by the group's spread) and every sequence's token sum divided
//! by one constant length ([`LossAgg::ConstantLength`]), differentiated with respect to the adapters of a tiny `qwen3`,
//! match central finite differences of the loss a step reports (`pg_regime`).

mod pg_regime;

use rl::objective::advantage::AdvantageNorm;
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;

#[test]
fn dr_grpo_adapter_gradients_match_finite_differences() {
    let regime = pg_regime::Regime { name: "dr-grpo", norm: AdvantageNorm::MeanCentered, band: ClipBand::symmetric(0.2), kl_beta: 0.0, loss_agg: LossAgg::ConstantLength(pg_regime::T as u32) };
    pg_regime::check(regime, [1.0, 0.25]);
}

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before any test runs.
brain_testutil::configure_test_binary!();
