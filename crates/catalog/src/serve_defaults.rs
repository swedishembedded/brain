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

/// Sequences a resident decodes together when `--max-batch` is not given, most first. A decode step's cost is the
/// weights it reads, which every row of the batch shares, so the widest batch serves the most; the sequences share one
/// KV pool, so a sequence costs its recurrent-state row and its decode activations up front and its KV only as its
/// tokens fill blocks. The widest batch that still leaves one sequence of [`DEFAULT_CONTEXT_FLOOR`] is taken. A batch
/// narrower than the clients a server has is also a queue in front of it: a request beyond it waits for a running one
/// to finish (prefill and every output token) before its prompt is read in, which on a 16-client load of 8K prompts
/// put half the requests seconds behind a full batch (knowledge 499).
pub const MAX_BATCH_DEFAULTS: &[Rule<u32>] = &[Rule {
    when: |_| true,
    ranked: &[64, 32, 16, 8, 4, 2, 1],
    why: "a decode step reads the weights once for every row it carries, and the sequences share one KV pool: the widest batch that leaves one sequence the context floor",
}];

/// The operator's value, else the best-ranked default of `rules` that `fits`, recorded in `chosen` as
/// `knob value: why`.
pub fn given_or_chosen<T: Copy + std::fmt::Display>(given: Option<T>, knob: &str, rules: &[Rule<T>], hw: &Hardware, chosen: &mut Vec<String>, fits: impl FnMut(T) -> Result<(), String>) -> T {
    given.unwrap_or_else(|| {
        let c = gpu_core::defaults::choose_or_leanest(rules, hw, fits);
        chosen.push(format!("{knob} {}: {}", c.value, c.why));
        c.value
    })
}
