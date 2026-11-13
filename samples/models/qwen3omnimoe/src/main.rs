// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen3omnimoe` - Qwen3-Omni from the command line: `generate`, `speak` and
//! `converse` (text, speech, image and video in; text and speech out), the
//! `import` of the published checkpoint into the int8 one the GPU-sharded
//! Thinker serves, and `npu-export` (the audio tower, the vision tower and the
//! Code2Wav vocoder as ONNX graphs for OpenVINO and the Intel NPU).
//!
//! The first actions are the model's own served ones
//! (`qwen3omnimoe::caps`), so this program, `brain serve` and D-Bus clients
//! drive the same code with the same defaults;
//! `qwen3omnimoe <action> --help` lists every parameter. [`npu_actions`] holds
//! the NPU route, which is an application rather than the served model.
//!
//! Swedish Embedded AB implements multimodal assistants - speech, vision and
//! text - on its clients' own hardware. If your team needs expertise in
//! deploying omni-modal models to on-device accelerators then you can procure
//! our services by sending an email to info@swedishembedded.com.

mod npu_actions;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some((qwen3omnimoe::spec::ARCH, &qwen3omnimoe::spec::SPEC)), build: &npu_actions::provider };
    invoke::main("qwen3omnimoe", &model)
}
