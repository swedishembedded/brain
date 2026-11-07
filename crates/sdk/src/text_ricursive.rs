// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one very large model from a single resident copy of its weights
// for its clients. If your team needs expertise in multi-tenant inference of
// fine-tuned language models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! [`crate::TextGenerationPipeline`] over a Ripple directory: the same resident the server runs
//! (`catalog::resident_ricursive`), driven in process. Its adapter is a slot of the resident's bank, so attaching,
//! switching and detaching an adapter never reloads the base, and the base's answers beside it are unchanged.

use std::path::Path;
use std::sync::Mutex;

use catalog::adapter_registry::{AdapterBankSettings, AdapterHost, AdapterTargets, AdapterView, ADAPTER_PARAM};
use catalog::adapter_release::verify_adapter;
use catalog::resident_ricursive::{RicursiveResident, RicursiveServeConfig};
use residency::Instance;

use crate::chat::{ModelIdentity, WeightsIdentity};
use crate::{Error, Result};

/// The adapter slots and rank a pipeline's bank holds: two adapters up to rank 64 (one in use, one being swapped in).
const SLOTS: u32 = 2;
const RANK_CAP: u32 = 64;

/// Whether `weights` is a directory holding a Ripple checkpoint (`config.json` declaring `model_type: ricursive`).
pub(crate) fn is_ripple(weights: &str) -> bool {
    let config = Path::new(weights).join("config.json");
    std::fs::read_to_string(config).ok().and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok()).is_some_and(|v| v.get("model_type").and_then(|t| t.as_str()) == Some("ricursive"))
}

/// A loaded Ripple and the adapter (if any) its generations run under.
pub(crate) struct RippleText {
    resident: std::sync::Arc<RicursiveResident>,
    instance: Mutex<Box<dyn Instance>>,
    /// The adapter generations run under; its lease keeps it offered.
    view: Option<AdapterView<RicursiveResident>>,
    pub(crate) identity: ModelIdentity,
}

impl RippleText {
    /// Load `weights` (a Ripple directory with its `tokenizer.json`) for `capacity` tokens on the cards `gpus`
    /// (`(index, bytes)`; none: the host tier), with `adapter` applied when given.
    pub(crate) fn load(weights: &str, tokenizer: Option<&str>, capacity: u32, gpus: &[(u32, u64)], adapter: Option<&str>) -> Result<RippleText> {
        if tokenizer.is_some_and(|t| Path::new(t) != Path::new(weights).join("tokenizer.json")) {
            return Err(Error::Backend(format!("ricursive: {weights}: the tokenizer is the checkpoint's own tokenizer.json beside it; a separate one is not read")));
        }
        let knobs = RicursiveServeConfig {
            ctx: Some(capacity),
            max_batch: Some(1),
            prefix_slots: Some(0),
            adapter_bank: Some(AdapterBankSettings { slots: SLOTS, rank_cap: RANK_CAP, targets: AdapterTargets::ALL }),
            ..Default::default()
        };
        let id = Path::new(weights).file_name().map_or_else(|| "local/ricursive".to_string(), |n| format!("local/{}", n.to_string_lossy()));
        let resident = std::sync::Arc::new(RicursiveResident::from_hf_dir(weights, &id, &knobs, gpus, 0).map_err(Error::Backend)?);
        let base = WeightsIdentity::of_path(weights, None)?;
        let tokenizer = brain_modelstore::fetch::file_digest(&Path::new(weights).join("tokenizer.json")).map_err(|e| Error::Backend(format!("{weights}: hashing the tokenizer: {e}")))?;
        let instance = resident.open_instance().map_err(Error::Backend)?;
        let mut text = RippleText { resident, instance: Mutex::new(instance), view: None, identity: ModelIdentity { base, adapter: None, tokenizer: Some(tokenizer) } };
        if let Some(path) = adapter {
            text.attach_adapter(path)?;
        }
        Ok(text)
    }

    /// Serve generations under the adapter at `path` from the next one on, replacing the one in use. The file is read
    /// once, bound to this base by the digest or id its card records and checked against the model before anything
    /// changes; a refused adapter leaves the pipeline as it was.
    pub(crate) fn attach_adapter(&mut self, path: &str) -> Result<()> {
        let loaded = verify_adapter(Path::new(path), &self.resident.served_base()).map_err(Error::Backend)?;
        let id = loaded.release.id.clone();
        let view = self.resident.adapter_view(loaded).map_err(Error::Backend)?;
        self.view = Some(view);
        self.identity.adapter = Some(WeightsIdentity::of_file(path, Some(id))?);
        Ok(())
    }

    /// Back to the base; `false` when no adapter was applied.
    pub(crate) fn detach_adapter(&mut self) -> bool {
        self.identity.adapter = None;
        self.view.take().is_some()
    }

    /// One generation of `inv`, under the adapter in use.
    pub(crate) fn run(&self, inv: capability::Invocation) -> Result<capability::Outcome> {
        let inv = match &self.view {
            Some(view) => inv.set(ADAPTER_PARAM, serde_json::json!(view.id())),
            None => inv,
        };
        let mut instance = self.instance.lock().map_err(|_| Error::Backend("ricursive: an earlier generation panicked while holding the model".to_string()))?;
        instance.run("generate", &inv, &mut |_| {}).map_err(Error::Backend)
    }
}
