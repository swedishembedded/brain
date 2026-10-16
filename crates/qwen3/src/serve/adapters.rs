// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The qwen3 engine's seat on the generic adapter bank
//! ([`model::adapter::bank`]): which adapters a bank holds, how they are
//! registered, uploaded, pinned and evicted, and what the decode tape records
//! after each targeted linear are the model crate's; this module is the
//! engine's glue (the tape hook, the KV-namespace purge on unload) and the
//! typed names of the seven linears the tape corrects.

use model::adapter::bank::{AdapterBank, SiteLeaf};
pub use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterInfo, UnloadStatus, VerifiedAdapter};
use model::adapter::sites::LoraSites;
use model::paged::{KvNamespace, BASE_KV_NAMESPACE};
use model::serve::{AdapterAvailability, AdapterId};

use gpu_core::{DeviceBuffer, Step};

use super::Engine;

/// The seven linears of a Qwen3 block an adapter can correct, as the decode
/// tape names them. The bank itself addresses linears by site leaf
/// ([`LoraSite::leaf`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoraSite {
    /// Query projection.
    Wq,
    /// Key projection.
    Wk,
    /// Value projection.
    Wv,
    /// Attention output projection.
    Wo,
    /// MLP gate projection.
    Gate,
    /// MLP up projection.
    Up,
    /// MLP down projection.
    Down,
}

impl LoraSite {
    /// Every site, in block order.
    pub const ALL: [LoraSite; 7] = [LoraSite::Wq, LoraSite::Wk, LoraSite::Wv, LoraSite::Wo, LoraSite::Gate, LoraSite::Up, LoraSite::Down];

    /// The attention projections: `LoraCfg::attn`'s default target set.
    pub const ATTENTION: [LoraSite; 4] = [LoraSite::Wq, LoraSite::Wk, LoraSite::Wv, LoraSite::Wo];

    /// The leaf name an adapter file and `LoraCfg::targets` use (`"wq"`).
    pub fn leaf(self) -> &'static str {
        match self {
            LoraSite::Wq => "wq",
            LoraSite::Wk => "wk",
            LoraSite::Wv => "wv",
            LoraSite::Wo => "wo",
            LoraSite::Gate => "gate",
            LoraSite::Up => "up",
            LoraSite::Down => "down",
        }
    }

    /// The site named by `leaf` (`"wq"`), as `LoraCfg::targets` spells it.
    pub fn from_leaf(leaf: &str) -> Option<LoraSite> {
        LoraSite::ALL.into_iter().find(|s| s.leaf() == leaf)
    }
}

impl SiteLeaf for LoraSite {
    fn site_leaf(&self) -> &str {
        self.leaf()
    }
}

impl Engine {
    /// Give this engine an adapter bank, so sequences can run under LoRA
    /// adapters registered with [`Engine::load_adapter`] alongside base
    /// sequences, in the same batches, on this one copy of the weights. Call
    /// once, before serving: the bank's buffers are allocated here and bound
    /// into every adapter tape recorded afterwards.
    ///
    /// Device cost is [`Engine::adapter_bank_bytes`]. An engine that never
    /// calls this behaves exactly as before.
    pub fn enable_adapters(&mut self, cfg: AdapterBankConfig) -> Result<(), AdapterError> {
        if self.bank.is_some() {
            return Err(AdapterError::BadConfig("the adapter bank is already enabled".into()));
        }
        let rows = self.max_batch.max(self.max_prefill);
        self.bank = Some(AdapterBank::new(&self.gpu, &self.cfg.lora_sites(), cfg, rows)?);
        self.tape_cache.retain(|&(_, lora), _| !lora);
        self.warn_if_calib_with_adapters();
        Ok(())
    }

    /// Device bytes the adapter bank occupies; 0 without one.
    pub fn adapter_bank_bytes(&self) -> u64 {
        self.bank.as_ref().map_or(0, AdapterBank::bytes)
    }

    /// Register a verified adapter and return the id the scheduler addresses
    /// it by ([`model::serve::Scheduler::submit_for_adapter`]). The id is also
    /// its KV namespace and is never reused. Registering the digest of an
    /// adapter already registered returns that adapter's id.
    ///
    /// Host memory only: the adapter reaches a slot when the scheduler first
    /// asks about it ([`model::serve::PagedDecoder::adapter_availability`]).
    pub fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        let bank = self.bank.as_mut().ok_or(AdapterError::NoBank)?;
        let id = bank.register(adapter)?;
        self.warn_if_calib_with_adapters();
        Ok(id)
    }

    /// Unload an adapter: free its slot and purge its cached KV. While
    /// sequences still run under it, it only stops admitting new ones
    /// ([`UnloadStatus::Draining`]) and the unload completes when the last of
    /// them finishes or is cancelled.
    pub fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError> {
        let bank = self.bank.as_mut().ok_or(AdapterError::NoBank)?;
        match bank.unload(id)? {
            Some(ns) => {
                self.prefix.evict_namespace(ns, &mut self.alloc);
                Ok(UnloadStatus::Unloaded)
            }
            None => Ok(UnloadStatus::Draining),
        }
    }

    /// Every registered adapter, oldest first.
    pub fn adapters(&self) -> Vec<AdapterInfo> {
        self.bank.as_ref().map_or_else(Vec::new, AdapterBank::infos)
    }

    /// The bank slot of each row whose KV namespace is given; empty without a
    /// bank, in which case every namespace must be the base's.
    pub(super) fn row_slots(&self, namespaces: impl Iterator<Item = KvNamespace>) -> Vec<u32> {
        match &self.bank {
            Some(bank) => namespaces.map(|ns| bank.slot_of(ns)).collect(),
            None => {
                for ns in namespaces {
                    assert_eq!(ns, BASE_KV_NAMESPACE, "a sequence runs under adapter {ns} but this engine has no adapter bank");
                }
                Vec::new()
            }
        }
    }

    /// Does the bank correct a linear reading the activation that `sites` share?
    pub(super) fn lora_reads(&self, sites: &[LoraSite]) -> bool {
        let leaves: Vec<&str> = sites.iter().map(|s| s.leaf()).collect();
        self.bank.as_ref().is_some_and(|b| b.targets_any(&leaves))
    }

    /// Record the bank correction of one linear of the adapter tape.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lora_correct(&self, steps: &mut Vec<Step>, layer: usize, site: LoraSite, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, out_stride: u32, out_col0: u32) {
        if let Some(bank) = &self.bank {
            bank.correct(&self.gpu, steps, layer, site.leaf(), x, out, rows, out_stride, out_col0);
        }
    }

    pub(super) fn bank_availability(&mut self, id: AdapterId) -> AdapterAvailability {
        let Some(bank) = &mut self.bank else { return AdapterAvailability::Unknown };
        let answer = bank.availability(&self.gpu, id);
        for ns in bank.take_purge() {
            self.prefix.evict_namespace(ns, &mut self.alloc);
        }
        answer
    }

    pub(super) fn bank_pin(&mut self, id: AdapterId) {
        self.bank.as_mut().expect("pin_adapter on an engine without an adapter bank").pin(id);
    }

    pub(super) fn bank_unpin(&mut self, id: AdapterId) {
        if let Some(ns) = self.bank.as_mut().and_then(|b| b.unpin(id)) {
            self.prefix.evict_namespace(ns, &mut self.alloc);
        }
    }

    /// Say once that clip ceilings measured on the base are in use beside
    /// adapters. An adapter's K/V can exceed the base's ceilings, and the
    /// int8 append clips to them.
    pub(super) fn warn_if_calib_with_adapters(&mut self) {
        if !self.warned_calib_with_adapters && self.bank.is_some() && self.kv_calibrated() {
            self.warned_calib_with_adapters = true;
            eprintln!("serve: KV clip ceilings were calibrated on the base model and are applied to adapter sequences too; an adapter whose K/V exceed them is clipped (drop the KV calibration for adapter serving)");
        }
    }
}
