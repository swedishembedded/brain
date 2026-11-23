// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `cosyvoice` - CosyVoice 2/3 from the command line: `synth` speaks a text in
//! the voice of a reference clip and its transcript (zero-shot cloning).
//!
//! The action and every parameter are CosyVoice's own served ones
//! (`cosyvoice::caps`), so this program, `brain serve` and D-Bus clients drive
//! the same code with the same defaults; `cosyvoice synth --help` lists them.
//!
//! Swedish Embedded AB implements on-device speech synthesis and voice cloning
//! for its clients. If your team needs expertise in text-to-speech on its own
//! hardware then you can procure our services by sending an email to
//! info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: cosyvoice::caps::manifest(), weights: Some(("cosyvoice", &cosyvoice::spec::CosyVoiceSpec)), build: &cosyvoice::caps::provider };
    invoke::main("cosyvoice", &model)
}
