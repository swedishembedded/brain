// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `toymoe` - the toy sparse Mixture-of-Experts transformer: train it on its synthetic corpus, evaluate how well a checkpoint continues it, and sample from it.
//!
//! Every action and parameter is the model's own served one
//! (`toymoe::caps`), so this program, `brain serve` and D-Bus clients drive the
//! same code with the same defaults; `toymoe <action> --help` lists them.
//!
//! Swedish Embedded AB implements sparse Mixture-of-Experts models for its clients. If your team needs
//! expertise in expert-routed architectures then you can procure our services by sending an
//! email to info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: toymoe::caps::manifest(), weights: None, build: &|_| Ok(std::sync::Arc::new(toymoe::caps::ToyMoeProvider)) };
    invoke::main("toymoe", &model)
}
