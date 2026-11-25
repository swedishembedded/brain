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

use serde::{Deserialize, Serialize};

/// How an attached adapter is named to the outside: its card's id, the
/// digest of its weight file (`sha256:<hex>`), its policy version and, for a
/// release a training run published, the run-level identity of its weights
/// ([`PolicyLineage`]). Never a filesystem location.
///
/// The version counts the releases of one id the server has offered: it is 1
/// for the first, and each release of the same id with different weights
/// takes the next number, for as long as the process runs (an id withdrawn
/// and offered again continues its count, and the same weights offered again
/// keep their number). A caller that records the version a completion was
/// sampled under can tell how many refits behind the id's current weights it
/// is. The version is local to the serving process; the lineage is not: it is
/// read from the adapter's own card, so every server that serves the file,
/// a restarted one included, names the same run, epoch and update.
///
/// Its JSON form (`{"id", "digest", "version"[, "lineage"]}`) is what every
/// serving surface reports: a model card, a reply's `brain.adapter`, a policy
/// trace's `policy.adapter`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterIdentity {
    pub id: String,
    pub digest: String,
    pub version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<PolicyLineage>,
}

/// Which weights a published policy is, independently of the server that serves it: the training run (its resume
/// key, or the id drawn for a run without one), the job of that run (`epoch`, one more for each job that continued
/// it), the optimiser update the weights hold (`update`, counted over the whole run), the base checkpoint it trained
/// on (`base_digest`) and the adapter file (`adapter_digest`, the same as [`AdapterIdentity::digest`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLineage {
    pub run: String,
    pub epoch: u64,
    pub update: u32,
    pub base_digest: String,
    pub adapter_digest: String,
}

impl AdapterIdentity {
    /// An identity with no run-level lineage: a release no training run published.
    pub fn new(id: impl Into<String>, digest: impl Into<String>, version: u64) -> AdapterIdentity {
        AdapterIdentity { id: id.into(), digest: digest.into(), version, lineage: None }
    }

    /// The JSON form every surface reports.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("an adapter identity is plain data")
    }

    /// Read the JSON form back, refusing anything else by name.
    pub fn from_json(value: &serde_json::Value) -> Result<AdapterIdentity, String> {
        AdapterIdentity::deserialize(value).map_err(|e| format!("adapter identity: {e}"))
    }
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
        AdapterIdentity::new(id, format!("sha256:{}", hex.to_string().repeat(64)), 1)
    }

    #[test]
    fn an_identity_reads_back_from_its_json_and_a_foreign_field_is_refused() {
        let mut published = identity("a", '1');
        assert_eq!(published.to_json(), serde_json::json!({"id": "a", "digest": published.digest, "version": 1}), "no lineage, no key");
        published.lineage = Some(PolicyLineage { run: "k".into(), epoch: 2, update: 7, base_digest: "sha256:b".into(), adapter_digest: published.digest.clone() });
        assert_eq!(AdapterIdentity::from_json(&published.to_json()), Ok(published.clone()));
        let mut foreign = published.to_json();
        foreign["lineage"]["step"] = serde_json::json!(7);
        assert!(AdapterIdentity::from_json(&foreign).unwrap_err().contains("step"));
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
