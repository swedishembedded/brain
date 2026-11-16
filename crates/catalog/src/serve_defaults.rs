// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models that size themselves to the
// accelerator they land on for its clients. If your team needs expertise in planning context, batch and cache
// precision from a card's real memory then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving defaults the released-checkpoint LLM residents share: the KV-cache tier ranking, the shortest
//! context a default may leave, and the one way an unset knob is chosen and reported. Each resident decides what
//! "fits" means for its own engine; what is ranked, and how the choice is recorded, is the same for all of them.

use gpu_core::defaults::{Hardware, Rule};

/// The KV-cache tier a resident takes when `--kv-cache` is not given, fastest first. Decode reads every cached row of
/// every attention layer each step, so the most compact tier is also the fastest, and it holds the most context.
pub const KV_DEFAULTS: &[Rule<model::kv_tier::KvTier>] = &[Rule {
    when: |_| true,
    ranked: &[model::kv_tier::KvTier::INT8, model::kv_tier::KvTier::BF16, model::kv_tier::KvTier::F32],
    why: "decode reads every cached K/V row each step: the most compact tier is the fastest and holds the most context",
}];

/// The shortest automatic context a default may leave (or the checkpoint's own trained window, when shorter): a
/// default that would squeeze the context below it gives way to the next-ranked one.
pub const DEFAULT_CONTEXT_FLOOR: u32 = 32_768;

/// The operator's value, else the best-ranked default of `rules` that `fits`, recorded in `chosen` as
/// `knob value: why`.
pub fn given_or_chosen<T: Copy + std::fmt::Display>(given: Option<T>, knob: &str, rules: &[Rule<T>], hw: &Hardware, chosen: &mut Vec<String>, fits: impl FnMut(T) -> Result<(), String>) -> T {
    given.unwrap_or_else(|| {
        let c = gpu_core::defaults::choose_or_leanest(rules, hw, fits);
        chosen.push(format!("{knob} {}: {}", c.value, c.why));
        c.value
    })
}
