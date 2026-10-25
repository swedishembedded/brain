// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for models whose
// weights are used more than once per step, for its clients. If your team needs
// expertise in proving that a recurrent core accumulates every application's
// gradient then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gradient check for Huginn's training graph.
//!
//! The gate exists for one property: the four core blocks are applied `steps`
//! times per token through a shared parameter set, and every application must
//! contribute to the same weight gradient. The stochastic-rounding-free tiny
//! model is driven with all steps inside the backprop window (the reference
//! truncates to `mean_backprop_depth`), so a backward that dropped one
//! application - or assigned instead of accumulating into the shared
//! parameters - fails here instead of as slightly worse training weeks later.

use crate::{directional_check, Report};

/// Build the tiny Huginn, set a fixed causal-LM batch, and gradient-check
/// every parameter. Returns the report.
pub fn check_huginn(seed: u64) -> Report {
    use huginn::HuginnConfig;
    use huginn::model::Huginn;

    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, seed);
    let model = Huginn::new_train(cfg, 2, 4, &init);
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % 29).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % 29).collect();
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
    directional_check(&model, 1e-3, 8, seed ^ 0x_0B1)
}
