// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified release of fine-tuned model
// adapters into live inference servers for its clients. If your team needs
// expertise in serving and rolling out fine-tuned language models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Which fine-tuned adapter is attached to each served model, as a fact the
//! serving surfaces can read.
//!
//! A model's [`Manifest`](capability::Manifest) is fixed at registration, but
//! an adapter can be swapped under a registered model at any time. The
//! executor therefore keeps this small, live table: whoever attaches an
//! adapter records its [`AdapterIdentity`] under the model id
//! ([`Executor::set_served_adapter`](crate::Executor::set_served_adapter)),
//! and a surface asks for it when it describes the model
//! ([`Executor::served_adapter`](crate::Executor::served_adapter)).
//! A model with no entry has no known adapter; nothing is ever defaulted.

use std::collections::HashMap;
use std::sync::RwLock;

/// How an attached adapter is named to the outside: its card's id and the
/// digest of its weight file (`sha256:<hex>`). Never a filesystem location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterIdentity {
    pub id: String,
    pub digest: String,
}

/// The model id -> attached adapter table behind [`Executor`](crate::Executor).
#[derive(Default)]
pub(crate) struct ServedAdapters(RwLock<HashMap<String, AdapterIdentity>>);

impl ServedAdapters {
    pub(crate) fn set(&self, model: &str, adapter: Option<AdapterIdentity>) {
        let mut table = self.0.write().unwrap_or_else(|e| e.into_inner());
        match adapter {
            Some(identity) => table.insert(model.to_string(), identity),
            None => table.remove(model),
        };
    }

    /// Record `adapter` for a model being registered, if it names one: a
    /// registration with no adapter leaves whatever was recorded alone, so a
    /// model that is not an adapter view can never erase a recorded identity.
    pub(crate) fn seed(&self, model: &str, adapter: Option<AdapterIdentity>) {
        if adapter.is_some() {
            self.set(model, adapter);
        }
    }

    pub(crate) fn get(&self, model: &str) -> Option<AdapterIdentity> {
        self.0.read().unwrap_or_else(|e| e.into_inner()).get(model).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(id: &str, hex: char) -> AdapterIdentity {
        AdapterIdentity { id: id.into(), digest: format!("sha256:{}", hex.to_string().repeat(64)) }
    }

    #[test]
    fn a_model_has_the_adapter_last_recorded_for_it_and_none_once_cleared() {
        let table = ServedAdapters::default();
        assert_eq!(table.get("m"), None, "nothing recorded is nothing known");

        table.set("m", Some(identity("a", '1')));
        table.set("other", Some(identity("b", '2')));
        assert_eq!(table.get("m"), Some(identity("a", '1')));

        table.set("m", Some(identity("a", '3')));
        assert_eq!(table.get("m"), Some(identity("a", '3')), "a swap replaces the identity");
        assert_eq!(table.get("other"), Some(identity("b", '2')), "other models are untouched");

        table.set("m", None);
        assert_eq!(table.get("m"), None);
    }
}
