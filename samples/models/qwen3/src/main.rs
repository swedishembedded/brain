// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen3` - the Qwen3 dense decoder from the command line: every served
//! action (`generate`, `train`, `finetune`, `lora_train`, `lora_gate`, `eval`,
//! `calib`, `toolcall_gen`, `toolcall_eval`, `import`, `export`, `export_peft`,
//! `embed`) with every parameter, `toy` (a tiny checkpoint for tests), and the NPU route (`export-onnx`,
//! `npu-generate`, `npu-precompile`) that deploys a checkpoint to the Intel
//! NPU through OpenVINO.
//!
//! The served actions are the model's own (`qwen3::caps`), so this program,
//! `brain qwen3 <action>`, `brain serve` and D-Bus clients drive the same code
//! with the same defaults; `qwen3 <action> --help` lists every parameter. The
//! NPU actions ([`npu_actions`]) are the part that is an application rather
//! than the served model: they write ONNX for a runtime brain does not itself
//! serve through, and need the NPU crate, which depends on the model crates.
//!
//! Swedish Embedded AB implements training, fine-tuning and on-device
//! deployment of language models for its clients. If your team needs expertise
//! in adapting Qwen-family decoders and exporting them for OpenVINO and NPUs
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.

mod npu_actions;
mod toy_action;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("qwen3", &qwen3::spec::Qwen3Spec)), build: &npu_actions::provider };
    invoke::main_with("qwen3", &model, &[])
}
