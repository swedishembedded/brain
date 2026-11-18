// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning with unbiased baselines, for its clients. If
// your team needs expertise in reinforcement learning for language models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! L2 of RLOO: each rollout's advantage is its reward less the mean of the other rollouts of its group
//! ([`AdvantageNorm::LeaveOneOut`]), trained through the clipped step with the KL term, differentiated with respect to
//! the adapters of a tiny `qwen3`; it matches central finite differences of the loss a step reports (`pg_regime`).

mod pg_regime;

use rl::objective::advantage::AdvantageNorm;
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;

#[test]
fn rloo_adapter_gradients_match_finite_differences() {
    let regime = pg_regime::Regime { name: "rloo", norm: AdvantageNorm::LeaveOneOut, band: ClipBand::symmetric(0.2), kl_beta: 0.02, loss_agg: LossAgg::TokenSum };
    pg_regime::check(regime, [1.0, 0.25]);
}
