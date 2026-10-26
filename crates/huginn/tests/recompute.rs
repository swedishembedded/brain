// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of recurrent-depth language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's `activations=` precision reaches Huginn through the generic path (`Model::set_saved_policy`): Huginn
//! keeps only the residual between block applications and rebuilds each one before its backward, so `recompute` is accepted as
//! what it already does, and keeping everything or narrower residuals are refused by name
//! (`model::saved::check_always_recomputes`).

use huginn::model::Huginn;
use huginn::HuginnConfig;

#[test]
fn huginn_answers_the_activations_key_from_its_recomputing_backward() {
    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, 3);
    let mut m = Huginn::new_train(cfg, 2, 4, &init);
    model::saved::check_always_recomputes("huginn", &mut m).unwrap();
}
