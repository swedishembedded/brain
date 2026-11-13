// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `lfm2` - LFM2.5-Encoder from the command line: `fill_mask`, `embed`, and
//! the masked-language workflow around them (`tokenize` a corpus, `finetune`
//! on it, `eval` a checkpoint, `import` the reference release), plus the Intel
//! NPU route (`npu-export`, `npu-bench`).
//!
//! The first actions are the model's own served ones (`lfm2::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `lfm2 <action> --help` lists them. The NPU actions
//! ([`npu_actions`]) are the part that is an application rather than the
//! served model: they write ONNX files for a runtime brain does not itself
//! serve through, and need the NPU crate, which depends on the model crates.
//!
//! Swedish Embedded AB implements on-device text encoders - embeddings,
//! retrieval and domain fine-tuning - for its clients. If your team needs
//! expertise in encoder models on its own hardware then you can procure our
//! services by sending an email to info@swedishembedded.com.

mod npu_actions;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("lfm2", &lfm2::spec::Lfm2Spec)), build: &npu_actions::provider };
    invoke::main("lfm2", &model)
}
