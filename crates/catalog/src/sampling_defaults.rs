// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving language models with the
// sampling their authors validated, for its clients. If your team needs
// expertise in making served models behave as their checkpoints intend, then
// you can procure our services by sending an email to info@swedishembedded.com.

//! A served model that samples the way its checkpoint declares.
//!
//! [`with_checkpoint_sampling`] reads `generation_config.json` beside a model's
//! weights once, and wraps the resident so the `generate` action's `temp`,
//! `top_k`, `top_p`, `min_p` and `repetition_penalty` default to what the
//! checkpoint declares. Both a client reading the manifest and the validation
//! of a request that omits a setting see the declared value; a checkpoint that
//! declares nothing gets the chat fallback for a chat model.

use std::path::Path;
use std::sync::Arc;

use capability::{Invocation, Manifest};
use data::generation::SamplingDefaults;
use residency::{AdapterIdentity, Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// `model` served with the sampling defaults `dir`'s `generation_config.json` declares, and for a chat model the
/// chat fallback ([`SamplingDefaults::CHAT_FALLBACK`]) where it declares none. An unreadable file is logged and
/// treated as declaring nothing.
pub fn with_checkpoint_sampling(model: Arc<dyn ResidentModel>, dir: &Path) -> Arc<dyn ResidentModel> {
    let defaults = SamplingDefaults::from_generation_config(dir).unwrap_or_else(|e| {
        residency::log::warn(&format!("ignoring the checkpoint's sampling defaults: {e}"));
        SamplingDefaults::default()
    });
    Arc::new(CheckpointSampling { inner: model, defaults })
}

struct CheckpointSampling {
    inner: Arc<dyn ResidentModel>,
    defaults: SamplingDefaults,
}

impl ResidentModel for CheckpointSampling {
    fn manifest(&self) -> Manifest {
        let mut manifest = self.inner.manifest();
        let label = manifest.model.clone();
        for action in manifest.actions.iter_mut().filter(|a| a.name == "generate") {
            *action = llmserve::with_sampling_defaults(std::mem::replace(action, capability::ActionSpec::new("", "")), &label, &self.defaults);
        }
        manifest
    }
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key)
    }
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        self.inner.estimate_at(key, tier)
    }
    fn batch_limit(&self) -> Option<usize> {
        self.inner.batch_limit()
    }
    fn replicas(&self) -> usize {
        self.inner.replicas()
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        self.inner.activate(key, device)
    }
    fn stamp(&self, inv: &mut Invocation) {
        self.inner.stamp(inv)
    }
    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.inner.adapter_identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_mock::MockResident;

    fn declared(tag: &str, json: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-ckpt-sampling-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("generation_config.json"), json).unwrap();
        dir
    }

    fn default_of(model: &Arc<dyn ResidentModel>, param: &str) -> Option<serde_json::Value> {
        let manifest = model.manifest();
        let generate = manifest.actions.iter().find(|a| a.name == "generate")?;
        generate.params.iter().find(|p| p.name == param)?.default.clone()
    }

    /// Every model whose checkpoint declares sampling defaults shows them in its manifest, whatever its family.
    #[test]
    fn a_resident_is_served_with_the_defaults_its_checkpoint_declares() {
        let plain: Arc<dyn ResidentModel> = Arc::new(MockResident);
        assert_eq!(default_of(&plain, "top_p"), Some(serde_json::json!(1.0)), "the mock's own constant");
        assert_eq!(default_of(&plain, "temp"), Some(serde_json::json!(1.0)));
        let dir = declared("declared", r#"{"temperature": 0.6, "top_k": 20, "top_p": 0.95}"#);
        let served = with_checkpoint_sampling(plain.clone(), &dir);
        assert_eq!(default_of(&served, "temp"), Some(serde_json::json!(0.6)));
        assert_eq!(default_of(&served, "top_k"), Some(serde_json::json!(20)));
        assert_eq!(default_of(&served, "top_p"), Some(serde_json::json!(0.95)));
        let silent = declared("silent", r#"{"eos_token_id": 2}"#);
        let fallback = with_checkpoint_sampling(plain, &silent);
        assert_eq!((default_of(&fallback, "temp"), default_of(&fallback, "top_k"), default_of(&fallback, "top_p")), (Some(serde_json::json!(0.7)), Some(serde_json::json!(20)), Some(serde_json::json!(0.95))), "nothing declared: the chat fallback");
    }
}
