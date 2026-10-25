// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for models whose
// weights are used more than once per step, for its clients. If your team needs
// expertise in proving that a looped transformer's shared parameters accumulate
// every use's gradient then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gradient check for Ouro's training graph.
//!
//! The one thing this gate exists for: the same 2-layer stack is visited
//! `total_ut_steps` times per token, and the analytic gradient must be the sum
//! of all those uses. The batch is deliberately non-uniform in its exit
//! weights (the gate is trained, so `p_t` differs per row), which means a
//! backward that dropped one pass's contribution, or one that assigned instead
//! of accumulating into a shared parameter's gradient buffer, shows up as a
//! mismatch here rather than as slightly worse training weeks later.

use crate::{directional_check, Report};

/// Build the tiny Ouro, set a fixed causal-LM batch, and gradient-check every
/// parameter. Returns the report.
pub fn check_ouro(seed: u64) -> Report {
    use ouro::model::Ouro;
    use ouro::OuroConfig;

    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, seed);
    let model = Ouro::new_train(cfg, 2, 4, &init);
    // Causal-LM targets: next token in the row, last position masked. The two
    // rows use different residues so no row is a copy of the other.
    let x: Vec<u32> = (0..8).map(|i| (i * 7 + 1) % 29).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 7 + 2) % 29).collect();
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
    directional_check(&model, 1e-3, 8, seed ^ 0x0_0A0)
}
