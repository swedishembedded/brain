// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one large model from a single resident copy of its weights for
// its clients. If your team needs expertise in multi-tenant inference of
// hybrid-attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Adapter serving for Qwen3.5/3.8: the engines' seat on the generic bank
//! ([`model::adapter::bank`]).
//!
//! Every stage ([`Qwen35`], one per card) holds a bank over its own layers.
//! The bank's lifecycle (registration, slot residency, pins, draining) is
//! driven in lockstep over all stages here, so every stage agrees which slot an
//! adapter is in and under which id: a call that could leave stages
//! disagreeing is made on all of them or rolled back. An engine with one stage
//! (the fp32 [`crate::serve::Engine`]) is the same code over a list of one.

use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterInfo, AdapterServing, UnloadStatus, VerifiedAdapter};
use model::adapter::sites::{validate, LoraSites};
use model::dispatch::LORA_SLOT_NONE;
use model::paged::{KvNamespace, BASE_KV_NAMESPACE};
use model::serve::{AdapterAvailability, AdapterId};

use crate::config::Qwen35Config;
use crate::model::Qwen35;

impl AdapterServing for Qwen35Config {
    /// The token mixers of both layer kinds: the GDN projections and the GQA
    /// projections. (The MLP linears are `gate`, `up` and `down`.)
    fn attention_leaves(&self) -> Vec<&'static str> {
        vec!["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    }
}

/// Give every stage a bank of `cfg`, each sized for forwards of `rows` rows.
pub(crate) fn enable<'a>(stages: impl Iterator<Item = &'a Qwen35>, cfg: &AdapterBankConfig, rows: u32) -> Result<(), AdapterError> {
    for stage in stages {
        stage.enable_adapter_bank(cfg.clone(), rows)?;
    }
    Ok(())
}

/// Bytes of device memory the banks of all stages occupy.
pub(crate) fn bytes<'a>(stages: impl Iterator<Item = &'a Qwen35>) -> u64 {
    stages.filter_map(|s| s.with_adapter_bank(|bank, _| bank.bytes())).sum()
}

/// Register `adapter` on every stage. The whole adapter is checked against
/// the model's sites and the bank's targets first, so no stage can refuse what
/// another accepted.
pub(crate) fn register<'a>(stages: impl Iterator<Item = &'a Qwen35> + Clone, cfg: &Qwen35Config, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
    let sites = cfg.lora_sites();
    validate(&adapter.adapter, &sites).map_err(|e| AdapterError::Incompatible(e.to_string()))?;
    let mut ids = Vec::new();
    for stage in stages.clone() {
        match stage.with_adapter_bank(|bank, _| bank.register(adapter.clone())).ok_or(AdapterError::NoBank)? {
            Ok(id) => ids.push(id),
            Err(e) => {
                // A refusal on a later stage after earlier ones registered a
                // NEW adapter: take it back so the stages stay in step.
                for (stage, id) in stages.zip(ids) {
                    stage.with_adapter_bank(|bank, _| bank.unload(id));
                }
                return Err(e);
            }
        }
    }
    let first = ids.first().copied().ok_or(AdapterError::NoBank)?;
    debug_assert!(ids.iter().all(|&id| id == first), "stage banks disagree on an adapter's id");
    Ok(first)
}

/// Unload `id` from every stage.
pub(crate) fn unload<'a>(stages: impl Iterator<Item = &'a Qwen35>, id: AdapterId) -> Result<(UnloadStatus, Option<KvNamespace>), AdapterError> {
    let mut result = None;
    for stage in stages {
        let r = stage.with_adapter_bank(|bank, _| bank.unload(id)).ok_or(AdapterError::NoBank)??;
        result = Some(r);
    }
    Ok(match result.ok_or(AdapterError::NoBank)? {
        Some(ns) => (UnloadStatus::Unloaded, Some(ns)),
        None => (UnloadStatus::Draining, None),
    })
}

/// The registered adapters, as the first stage reports them (all stages hold
/// the same set; their `resident` flags can differ mid-upload, so a stage that
/// is not done makes the adapter not resident).
pub(crate) fn infos<'a>(stages: impl Iterator<Item = &'a Qwen35>) -> Vec<AdapterInfo> {
    let mut all: Option<Vec<AdapterInfo>> = None;
    for stage in stages {
        let Some(these) = stage.with_adapter_bank(|bank, _| bank.infos()) else { return Vec::new() };
        match &mut all {
            None => all = Some(these),
            Some(mine) => {
                for (m, t) in mine.iter_mut().zip(&these) {
                    m.resident &= t.resident;
                }
            }
        }
    }
    all.unwrap_or_default()
}

/// Is `id` usable on every stage? Drives every stage's upload (a quantum each
/// per call) and answers `Resident` only when all are.
pub(crate) fn availability<'a>(stages: impl Iterator<Item = &'a Qwen35>, id: AdapterId) -> (AdapterAvailability, Vec<KvNamespace>) {
    let mut answer = AdapterAvailability::Resident;
    let mut purge = Vec::new();
    let mut any = false;
    for stage in stages {
        let Some((a, p)) = stage.with_adapter_bank(|bank, gpu| (bank.availability(gpu, id), bank.take_purge())) else { return (AdapterAvailability::Unknown, Vec::new()) };
        any = true;
        if purge.is_empty() {
            purge = p;
        }
        answer = match (answer, a) {
            (AdapterAvailability::Unknown, _) | (_, AdapterAvailability::Unknown) => AdapterAvailability::Unknown,
            (AdapterAvailability::Loading, _) | (_, AdapterAvailability::Loading) => AdapterAvailability::Loading,
            _ => AdapterAvailability::Resident,
        };
    }
    if any {
        (answer, purge)
    } else {
        (AdapterAvailability::Unknown, Vec::new())
    }
}

/// Pin `id` on every stage.
pub(crate) fn pin<'a>(stages: impl Iterator<Item = &'a Qwen35>, id: AdapterId) {
    for stage in stages {
        stage.with_adapter_bank(|bank, _| bank.pin(id)).expect("pin_adapter on an engine without an adapter bank");
    }
}

/// Release one pin of `id` on every stage; the namespace to purge when that
/// completed a deferred unload.
pub(crate) fn unpin<'a>(stages: impl Iterator<Item = &'a Qwen35>, id: AdapterId) -> Option<KvNamespace> {
    let mut purge = None;
    for stage in stages {
        if let Some(Some(ns)) = stage.with_adapter_bank(|bank, _| bank.unpin(id)) {
            purge = Some(ns);
        }
    }
    purge
}

/// The slot a sequence in `ns` runs under, [`LORA_SLOT_NONE`] for the base.
/// Panics for an adapter that is not resident (see `AdapterBank::slot_of`).
pub(crate) fn slot_of<'a>(mut stages: impl Iterator<Item = &'a Qwen35>, ns: KvNamespace) -> u32 {
    if ns == BASE_KV_NAMESPACE {
        return LORA_SLOT_NONE;
    }
    match stages.next().and_then(|s| s.with_adapter_bank(|bank, _| bank.slot_of(ns))) {
        Some(slot) => slot,
        None => panic!("a sequence runs under adapter {ns} but this engine has no adapter bank"),
    }
}

/// Tell every stage the slot of each row of the forward about to run. A batch
/// of base rows only is passed as nothing, so it replays the base's dispatches.
pub(crate) fn set_rows<'a>(stages: impl Iterator<Item = &'a Qwen35>, slots: &[u32]) {
    let adapted = slots.iter().any(|&s| s != LORA_SLOT_NONE);
    let slots = if adapted { slots } else { &[] };
    for stage in stages {
        stage.set_adapter_rows(slots);
    }
}
