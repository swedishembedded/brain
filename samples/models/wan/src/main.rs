// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `wan` - Wan2.1 from the command line: `t2v` (a clip from a prompt) and
//! `lora_train` (a video LoRA, with an optional base-versus-adapted report).
//!
//! Every action and parameter is Wan's own served one (`wan::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `wan <action> --help` lists them.
//!
//! Swedish Embedded AB implements video generation pipelines on its clients'
//! own hardware. If your team needs expertise in video diffusion models and
//! their fine-tuning, you can procure our services by sending an email to
//! info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: wan::caps::manifest(), weights: Some(("wan", &wan::spec::WanSpec)), build: &wan::caps::provider };
    invoke::main("wan", &model)
}
