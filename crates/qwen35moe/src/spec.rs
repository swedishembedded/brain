// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which checkpoints in a model store are Qwen3.5-35B-A3B: a Hugging Face
//! directory whose `config.json` declares the Qwen3.5 MoE classes, or a brain
//! checkpoint (what `brain import-gguf` writes) whose card names the
//! `qwen35moe` family. The `tokenizer` role is the `tokenizer.json` beside it
//! when the store holds one; without one the model runs on token ids.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! model distribution and resolution then you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain_modelstore::family_spec::{FamilySpec, TokenizerRole};

/// The family `qwen35moe::import`'s conversion stamps on its card.
pub const CARD_FAMILY: &str = "qwen35moe";

pub const SPEC: FamilySpec = FamilySpec { arch: "qwen35moe", card_family: CARD_FAMILY, tokenizer: TokenizerRole::Optional };
