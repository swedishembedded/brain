// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The adapter-serving families this catalog knows: the one place a family
//! tag (`ModelCard.family`) is tied to an architecture config. The adapter
//! registry ([`crate::adapter_registry`]) reads a family's linears from here
//! and names no architecture itself.

use model::adapter::bank::AdapterServing;

use crate::adapter_release::{QWEN35_FAMILY, QWEN_FAMILY};

/// What a served model family calls its linears: the leaves an adapter may
/// correct and the `attention` group, as the family's [`AdapterServing`]
/// config lists them. Leaf names do not depend on a config's dimensions, so a
/// small config of the family stands for all of them.
#[derive(Clone, Debug)]
pub struct ServedFamily {
    /// The `ModelCard.family` of the family's adapters.
    pub tag: &'static str,
    /// Every leaf the family has a site for.
    pub all: Vec<&'static str>,
    /// The leaves of the `attention` group.
    pub attention: Vec<&'static str>,
}

impl ServedFamily {
    /// The family of `model`, adapters tagged `tag`.
    pub fn of<M: AdapterServing + ?Sized>(tag: &'static str, model: &M) -> ServedFamily {
        ServedFamily { tag, all: model.all_leaves(), attention: model.attention_leaves() }
    }

    /// The family of adapters carrying card family `tag`, or `None` for a
    /// family this crate does not serve.
    pub fn named(tag: &str) -> Option<ServedFamily> {
        match tag {
            QWEN_FAMILY => Some(ServedFamily::of(QWEN_FAMILY, &qwen3::config::QwenConfig::tiny())),
            QWEN35_FAMILY => Some(ServedFamily::of(QWEN35_FAMILY, &qwen35::config::Qwen35Config::tiny())),
            _ => None,
        }
    }
}

/// A small Qwen3 config, for tests that size a bank without a model.
#[cfg(test)]
pub(crate) fn tiny_qwen() -> qwen3::config::QwenConfig {
    qwen3::config::QwenConfig::tiny()
}
