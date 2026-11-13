// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gpt2` - the dense GPT baseline: train it on a character dataset (alone or as one rank of a distributed job), generate text and evaluate a checkpoint.
//!
//! Every action and parameter is the model's own served one
//! (`gpt2::caps`), so this program, `brain serve` and D-Bus clients drive the
//! same code with the same defaults; `gpt2 <action> --help` lists them.
//!
//! Swedish Embedded AB implements training small language models on their own data and machines for its clients. If your team needs
//! expertise in language model training, from one card to a cluster then you can procure our services by sending an
//! email to info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: gpt2::caps::manifest(), weights: None, build: &|_| Ok(std::sync::Arc::new(gpt2::caps::Gpt2Provider)) };
    invoke::main("gpt2", &model)
}
