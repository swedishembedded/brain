// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--model`: WHICH model an application runs.
//!
//! The counterpart to [`crate::Hardware`], which says where a run is allowed
//! to schedule. Both are core groups for the same reason: they mean the same
//! thing in every application, so they should be spelled once. Three samples
//! had already grown their own `--model` loop, each accepting a slightly
//! different thing.
//!
//! **What the value MEANS is the application's business.** This group carries
//! the string the caller typed and nothing else - a short name an application
//! knows (`laya`), a `<vendor>/<repo>` hub id, or a checkpoint directory. The
//! mapping from that string to a directory needs a model store, a default
//! alias table, or a resolver, none of which belong in a flag parser, and all
//! of which differ per surface. See `samples/decision/json` for one worked
//! resolution.
//!
//! Swedish Embedded AB implements the shared command-line layer for product
//! families where every binary has to accept the same flags with the same
//! meaning. If your team needs that, you can procure our services by sending
//! an email to info@swedishembedded.com.

use crate::{Args, Options};

/// Which model to run, as the caller named it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelChoice {
    /// A short name, a hub id, or a path - whatever the application's own
    /// resolution accepts. Empty when the application has no default and the
    /// caller named none; see [`ModelChoice::require`].
    pub name: String,
}

impl ModelChoice {
    /// This application's default model, to be overridden by `--model`.
    pub fn new(default: impl Into<String>) -> ModelChoice {
        ModelChoice { name: default.into() }
    }

    /// Take `--model`, keeping whatever this value already holds as the
    /// default - the same `take_over` shape `brain::options`' groups use, so
    /// an application can build its default from anything (a constant, its
    /// own environment lookup) before the flag gets its say.
    pub fn take_over(mut self, args: &mut Args) -> Result<ModelChoice, String> {
        self.name = args.str_or("--model", &self.name.clone());
        Ok(self)
    }

    /// Fail with the remedy attached when no model was named, for an
    /// application that has no sensible default. Every one of them needs this
    /// check and none should word it differently.
    pub fn require(&self) -> Result<&str, String> {
        if self.name.trim().is_empty() {
            return Err("--model NAME is required: this application has no default model".into());
        }
        Ok(&self.name)
    }
}

impl Options for ModelChoice {
    /// Take `--model` with no application default, so a caller who names
    /// nothing gets an empty [`ModelChoice::name`] rather than someone
    /// else's idea of a reasonable model.
    fn take(args: &mut Args) -> Result<ModelChoice, String> {
        ModelChoice::default().take_over(args)
    }

    fn help() -> &'static str {
        "  --model NAME|DIR    which model to run: a short name, a <vendor>/<repo> id, or a directory"
    }
}
