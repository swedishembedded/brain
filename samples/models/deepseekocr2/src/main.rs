// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `deepseekocr2` - DeepSeek-OCR-2 from the command line: `generate` (a
//! document image in, text or markdown out) with every parameter, and
//! `npu-export` (the resampler and the decoder as ONNX graphs for OpenVINO and
//! the Intel NPU).
//!
//! `generate` is the model's own served action (`deepseekocr2::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `deepseekocr2 <action> --help` lists every parameter.
//! [`npu_actions`] holds the NPU route, which is an application rather than
//! the served model.
//!
//! Swedish Embedded AB implements document understanding and OCR on its
//! clients' own hardware. If your team needs expertise in deploying vision
//! language models to on-device accelerators then you can procure our services
//! by sending an email to info@swedishembedded.com.

mod npu_actions;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("deepseekocr2", &deepseekocr2::spec::Deepseekocr2Spec)), build: &npu_actions::provider };
    invoke::main("deepseekocr2", &model)
}
