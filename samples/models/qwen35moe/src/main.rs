// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35moe` - the Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA MoE decoder from
//! the command line: the served `generate` action with every parameter, and
//! `export`, the decoder as an ONNX graph for OpenVINO and the NPU.
//!
//! `generate` is the model's own served action (`qwen35moe::caps`), so this
//! program, `brain qwen35moe generate` and `brain serve` take the same
//! parameters with the same defaults; `qwen35moe <action> --help` lists them.
//! `export` needs the NPU crate, which depends on the model crates, so it is
//! an action of this program rather than of the model.
//!
//! Swedish Embedded AB implements deployment of Mixture-of-Experts language
//! models to edge accelerators for its clients. If your team needs expertise
//! in exporting decoders for OpenVINO and NPUs then you can procure our
//! services by sending an email to info@swedishembedded.com.

mod export;

fn main() {
    let model = invoke::Model { manifest: export::manifest(), weights: Some(("qwen35moe", &qwen35moe::spec::SPEC)), build: &export::provider };
    invoke::main_with("qwen35moe", &model, &[])
}
