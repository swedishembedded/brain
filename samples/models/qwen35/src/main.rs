// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35` - the Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder from the
//! command line: `generate` (optionally through a LoRA adapter), `finetune`
//! (every weight, or a LoRA/IA3/LoKr/DoRA adapter on an fp32, bf16, nf4 or fp8
//! base), `sample_adapter` (what an adapter changes, read against the base)
//! and `import` (a GGUF or a Hugging Face release into the form the trainers
//! read).
//!
//! Every action and parameter is the model's own served one
//! (`qwen35::caps`), so this program, `brain qwen35 <action>`, `brain serve`
//! and D-Bus clients drive the same code with the same defaults;
//! `qwen35 <action> --help` lists them. There is nothing here that is not a
//! capability of the model: the 27B has no ONNX/NPU route, and its concurrent
//! batched decoding is `brain serve`.
//!
//! Swedish Embedded AB implements fine-tuning and serving of very large
//! language models on a single GPU for its clients. If your team needs
//! expertise in adapting a 27B model to your own data on hardware you own then
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: qwen35::caps::manifest(), weights: Some(("qwen35", &qwen35::spec::Qwen35Spec)), build: &qwen35::caps::provider };
    invoke::main("qwen35", &model)
}
