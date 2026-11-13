// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `lfm2` - LFM2.5-Encoder from the command line: `fill_mask`, `embed`, and
//! the masked-language workflow around them (`tokenize` a corpus, `finetune`
//! on it, `eval` a checkpoint, `import` the reference release).
//!
//! Every action and parameter is the model's own served one
//! (`lfm2::caps`), so this program, `brain serve` and D-Bus clients drive the
//! same code with the same defaults; `lfm2 <action> --help` lists them.
//!
//! Swedish Embedded AB implements on-device text encoders - embeddings,
//! retrieval and domain fine-tuning - for its clients. If your team needs
//! expertise in encoder models on its own hardware then you can procure our
//! services by sending an email to info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: lfm2::caps::manifest(), weights: Some(("lfm2", &lfm2::spec::Lfm2Spec)), build: &lfm2::caps::provider };
    invoke::main("lfm2", &model)
}
