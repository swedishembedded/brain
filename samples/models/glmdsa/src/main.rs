// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `glmdsa` - the GLM-5.2 decoder from the command line: every served action
//! (`train`, `finetune`, `eval`, `generate`, `import`) with every parameter,
//! and the NPU route (`export`, `npu-generate`) that deploys a checkpoint to
//! the Intel NPU through OpenVINO.
//!
//! The first five are the model's own served actions (`glmdsa::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `glmdsa <action> --help` lists every parameter. The NPU actions
//! ([`npu_actions`]) are the part that is an application rather than the
//! served model: they write ONNX for a runtime brain does not itself serve
//! through, and need the NPU crate, which depends on the model crates.
//!
//! Swedish Embedded AB implements training and on-device deployment of
//! Mixture-of-Experts language models for its clients. If your team needs
//! expertise in exporting decoders for OpenVINO and NPUs then you can
//! procure our services by sending an email to info@swedishembedded.com.

mod npu_actions;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("glmdsa", &glmdsa::spec::SPEC)), build: &npu_actions::provider };
    invoke::main_with("glmdsa", &model, &[])
}
