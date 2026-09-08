// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device-side (`ParamStore` `.lora_a`/`.lora_b`) LoRA family's shared
//! predicate, hoisted out of qwen3/qwen35/qwen35moe/deepseek2's own
//! near-identical role-assignment code. Each of those crates independently
//! wrote `n.ends_with(".lora_a") || n.ends_with(".lora_b")` at its own
//! `Role::Trainable`-vs-`Role::Frozen` decision point; a fifth copy in
//! `crate::lora::device_adapter`'s save filter, and a sixth (as
//! `strip_suffix`) in its fold. One predicate here is what a future adapter
//! kind that needs a differently-named or differently-shaped tensor set
//! (DoRA's `.lora_m`, say) has to extend in ONE place instead of six.

/// Is `name` one of the device-side LoRA family's own trainable tensors?
/// The only two suffixes that family emits today - see
/// `crate::lora::device_adapter`.
pub fn is_adapter_param(name: &str) -> bool {
    name.ends_with(".lora_a") || name.ends_with(".lora_b")
}
