// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving engine's adapter bank: many LoRA adapters, or none, applied to
//! different rows of ONE batch on ONE resident copy of the base weights.
//!
//! ## Shape
//!
//! [`Engine::enable_adapters`] allocates, for every `(layer, linear)` the
//! bank targets, a [`model::dispatch::LoraBank`] of `slots` adapters padded to
//! `r_max`, plus one `[rows]` slot vector. A row's slot says which adapter
//! corrects it ([`model::dispatch::LORA_SLOT_NONE`]: none). The adapter
//! variant of the decode tape records two bank dispatches after each targeted
//! linear, whatever number of distinct adapters the batch holds; a batch with
//! no adapter row replays the unchanged base tape.
//!
//! ## Lifecycle
//!
//! An adapter is *registered* by [`Engine::load_adapter`] (host memory only,
//! one [`AdapterId`] = one KV namespace per distinct digest, never reused),
//! becomes *resident* in a slot when the scheduler asks about it
//! ([`PagedDecoder::adapter_availability`]: the upload is spread over calls,
//! [`AdapterBankConfig::upload_quantum_bytes`] each, so a cold load never
//! stalls decode), is *pinned* for the lifetime of each sequence using it, and
//! is evicted from its slot (least recently used, never while pinned) when
//! another adapter needs the slot. Eviction keeps the registration and the
//! cached KV: the namespace is tied to the adapter, not to the slot.
//! [`Engine::unload_adapter`] removes the registration and purges the
//! namespace's prefix-cache entries; while a sequence still uses the adapter
//! it only marks the adapter draining (no new admissions) and the last
//! unpin completes it.
//!
//! ## Verification
//!
//! This module trusts its input to have been verified against the served
//! base (card family and kind, base digest or id, content digest of the bytes
//! the tensors were parsed from). That check lives with the catalog, which
//! depends on this crate; it hands over a [`VerifiedAdapter`]. The engine
//! re-checks everything it can see itself: every site is a linear this bank
//! targets, with the base weight's own dimensions and a rank within `r_max`.

use std::collections::HashMap;
use std::fmt;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::dispatch::{lora_bank_rows, LoraBank, LoraBankKernels, LoraBankRows, LORA_SLOT_NONE};
use model::paged::{KvNamespace, NamespaceIds, BASE_KV_NAMESPACE};
use model::serve::{AdapterAvailability, AdapterId};

use super::Engine;
use crate::config::QwenConfig;
use crate::lora::DeviceAdapter;

/// The seven linears of a Qwen3 block an adapter can correct.
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

    /// `(layer, site)` of a base weight name like `blocks.3.attn.wq.weight`.
    fn parse(weight_name: &str, n_layers: u32) -> Option<(usize, LoraSite)> {
        let (layer, rest) = weight_name.strip_prefix("blocks.")?.split_once('.')?;
        let layer: u32 = layer.parse().ok()?;
        let site = match rest {
            "attn.wq.weight" => LoraSite::Wq,
            "attn.wk.weight" => LoraSite::Wk,
            "attn.wv.weight" => LoraSite::Wv,
            "attn.wo.weight" => LoraSite::Wo,
            "mlp.gate.weight" => LoraSite::Gate,
            "mlp.up.weight" => LoraSite::Up,
            "mlp.down.weight" => LoraSite::Down,
            _ => return None,
        };
        (layer < n_layers).then_some((layer as usize, site))
    }

    /// `(k, n)`: contraction and output width of this linear.
    fn dims(self, cfg: &QwenConfig) -> (u32, u32) {
        let (d, ff, hq, hkv) = (cfg.d_model, cfg.d_ff, cfg.q_dim(), cfg.kv_dim());
        match self {
            LoraSite::Wq => (d, hq),
            LoraSite::Wk | LoraSite::Wv => (d, hkv),
            LoraSite::Wo => (hq, d),
            LoraSite::Gate | LoraSite::Up => (d, ff),
            LoraSite::Down => (ff, d),
        }
    }
}

/// How big the adapter bank is and what it can correct. Fixed for the life of
/// the engine: the bank buffers are bound into the recorded tapes, so loading
/// or evicting an adapter is a buffer write and never a tape rebuild.
#[derive(Clone, Debug)]
pub struct AdapterBankConfig {
    /// Adapters resident on the device at once.
    pub slots: u32,
    /// Largest adapter rank a slot holds.
    pub r_max: u32,
    /// The linears the bank can correct; an adapter targeting any other is
    /// refused. Device cost is `slots * r_max * (k + n) * 4` bytes per
    /// targeted linear per layer.
    pub targets: Vec<LoraSite>,
    /// Adapters that may be registered at once (resident or not); their
    /// factors are held in host memory until evicted by [`Engine::unload_adapter`].
    pub max_registered: u32,
    /// Bytes of adapter factors one [`PagedDecoder::adapter_availability`]
    /// call uploads before answering `Loading`: bounds the time a cold load
    /// takes out of one scheduler iteration. At least one linear is uploaded
    /// per call.
    pub upload_quantum_bytes: u64,
}

impl AdapterBankConfig {
    /// A bank of `slots` adapters up to `r_max` rank, correcting `targets`,
    /// with room to register 64 adapters and a 4 MiB upload quantum.
    pub fn new(slots: u32, r_max: u32, targets: &[LoraSite]) -> AdapterBankConfig {
        AdapterBankConfig { slots, r_max, targets: targets.to_vec(), max_registered: 64, upload_quantum_bytes: 4 << 20 }
    }
}

/// An adapter the caller has verified against the served base, in host
/// memory, ready for [`Engine::load_adapter`].
pub struct VerifiedAdapter {
    /// The adapter's own name (its card id); reported, never interpreted.
    pub label: String,
    /// Content digest of the adapter file's bytes: the adapter's identity.
    /// Two loads with the same digest are one adapter.
    pub digest: String,
    /// The parsed factors.
    pub adapter: DeviceAdapter,
}

/// Why an adapter operation was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum AdapterError {
    /// The engine has no adapter bank ([`Engine::enable_adapters`]).
    NoBank,
    /// The bank config is unusable.
    BadConfig(String),
    /// The adapter does not fit this bank or this model.
    Incompatible(String),
    /// No adapter with this id is registered.
    Unknown(AdapterId),
    /// The adapter is being unloaded and takes no new users.
    Draining(AdapterId),
    /// `max_registered` adapters are already registered.
    Full,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterError::NoBank => write!(f, "this engine has no adapter bank"),
            AdapterError::BadConfig(why) => write!(f, "adapter bank config: {why}"),
            AdapterError::Incompatible(why) => write!(f, "adapter does not fit the bank: {why}"),
            AdapterError::Unknown(id) => write!(f, "adapter {} is not registered", id.kv_namespace()),
            AdapterError::Draining(id) => write!(f, "adapter {} is being unloaded", id.kv_namespace()),
            AdapterError::Full => write!(f, "the adapter registry is full"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// What [`Engine::unload_adapter`] did.
#[derive(Debug, PartialEq, Eq)]
pub enum UnloadStatus {
    /// Gone: slot freed, cached KV purged.
    Unloaded,
    /// Sequences still use it. It admits no new ones and is unloaded when the
    /// last one unpins.
    Draining,
}

/// A registered adapter, as [`Engine::adapters`] reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    /// The id (and KV namespace) the scheduler addresses it by.
    pub id: AdapterId,
    /// The adapter's own name.
    pub label: String,
    /// Content digest.
    pub digest: String,
    /// Its rank.
    pub rank: u32,
    /// Is it in a bank slot now?
    pub resident: bool,
    /// Sequences pinning it.
    pub pins: u32,
    /// Is it waiting for its last sequence to finish so it can be unloaded?
    pub draining: bool,
}

/// One banked linear: its adapters on the device and its shape.
struct BankSite {
    layer: usize,
    site: LoraSite,
    bank: LoraBank,
    k: u32,
    n: u32,
}

/// One linear of a registered adapter, in the bank's layout.
struct HostSite {
    bank_ix: usize,
    /// `A [rank, k]`.
    a: Vec<f32>,
    /// `B` transposed to `[rank, n]` with `alpha / rank` folded in.
    bt: Vec<f32>,
}

enum Phase {
    /// Registered, in no slot.
    Cold,
    /// Slot reserved, `sites[..next]` uploaded.
    Uploading { slot: u32, next: usize },
    /// In `slot`, usable.
    Resident { slot: u32 },
}

struct Entry {
    label: String,
    digest: String,
    rank: u32,
    sites: Vec<HostSite>,
    phase: Phase,
    pins: u32,
    draining: bool,
    last_used: u64,
}

/// The device bank plus the host-side registry that decides who is in it.
pub(super) struct AdapterBank {
    cfg: AdapterBankConfig,
    kernels: LoraBankKernels,
    sites: Vec<BankSite>,
    site_ix: HashMap<(usize, LoraSite), usize>,
    /// `[rows]` u32 slot of every row of the batch being run.
    pub(super) slot_buf: DeviceBuffer,
    /// `[rows, r_max]` projection scratch shared by every corrected linear.
    tscr: DeviceBuffer,
    entries: HashMap<AdapterId, Entry>,
    by_digest: HashMap<String, AdapterId>,
    slot_owner: Vec<Option<AdapterId>>,
    /// Namespace -> slot, for adapters that are `Resident`.
    resident_slot: HashMap<KvNamespace, u32>,
    ids: NamespaceIds,
    /// The adapter last answered `Resident` and not yet pinned: counts as
    /// pinned until the pin arrives or another adapter is asked about.
    offered: Option<AdapterId>,
    /// Namespaces of deferred unloads that completed without an unpin (an
    /// offer that was never taken up); the engine purges their cached KV.
    pending_purge: Vec<KvNamespace>,
    tick: u64,
    bytes: u64,
}

impl AdapterBank {
    fn new(gpu: &Gpu, mcfg: &QwenConfig, cfg: AdapterBankConfig, rows: u32) -> Result<AdapterBank, AdapterError> {
        if cfg.slots == 0 || cfg.r_max == 0 || cfg.targets.is_empty() || cfg.max_registered < cfg.slots {
            return Err(AdapterError::BadConfig("needs slots > 0, r_max > 0, at least one target and max_registered >= slots".into()));
        }
        let kernels = LoraBankKernels::resolve(gpu).ok_or_else(|| AdapterError::BadConfig("the device does not register the LoRA bank kernels".into()))?;
        let (s, r) = (cfg.slots as u64, cfg.r_max as u64);
        let mut sites = Vec::new();
        let mut site_ix = HashMap::new();
        let mut bytes = 0u64;
        for layer in 0..mcfg.n_layers as usize {
            for &site in &cfg.targets {
                let (k, n) = site.dims(mcfg);
                let widest = s * r * k.max(n) as u64 * 4;
                if widest > gpu.max_storage_binding_bytes() {
                    return Err(AdapterError::BadConfig(format!("{} slots of rank {} at {} x {} exceed one storage binding ({widest} bytes)", cfg.slots, cfg.r_max, k, n)));
                }
                let rank = gpu.storage(s);
                gpu.write(&rank, &vec![0u32; cfg.slots as usize]);
                let bank = LoraBank { a: gpu.storage(s * r * k as u64), bt: gpu.storage(s * r * n as u64), rank, slots: cfg.slots, r_max: cfg.r_max };
                bytes += s * r * (k + n) as u64 * 4 + s * 4;
                site_ix.insert((layer, site), sites.len());
                sites.push(BankSite { layer, site, bank, k, n });
            }
        }
        let slot_buf = gpu.storage(rows as u64);
        let tscr = gpu.storage(rows as u64 * r);
        bytes += rows as u64 * 4 + rows as u64 * r * 4;
        let slot_owner = vec![None; cfg.slots as usize];
        Ok(AdapterBank { cfg, kernels, sites, site_ix, slot_buf, tscr, entries: HashMap::new(), by_digest: HashMap::new(), slot_owner, resident_slot: HashMap::new(), ids: NamespaceIds::default(), offered: None, pending_purge: Vec::new(), tick: 0, bytes })
    }

    /// Does the bank correct any of `sites`?
    pub(super) fn targets_any(&self, sites: &[LoraSite]) -> bool {
        sites.iter().any(|s| self.cfg.targets.contains(s))
    }

    /// Record the two dispatches correcting `out` for `site` of `layer`;
    /// nothing when the bank does not target it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn correct(&self, gpu: &Gpu, steps: &mut Vec<Step>, layer: usize, site: LoraSite, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, out_stride: u32, out_col0: u32) {
        let Some(&ix) = self.site_ix.get(&(layer, site)) else { return };
        let bs = &self.sites[ix];
        debug_assert_eq!((bs.layer, bs.site), (layer, site));
        let rq = LoraBankRows { kernels: self.kernels, bank: &bs.bank, slot: &self.slot_buf, tscr: &self.tscr, x, out, row0: 0, rows, k: bs.k, n: bs.n, out_stride, out_col0 };
        steps.extend(lora_bank_rows(gpu, &rq));
    }

    /// The slot a sequence in `ns` runs under. Panics for an adapter that is
    /// not resident: answering such a row with the base would be a silent
    /// wrong answer, and the scheduler never admits one.
    fn slot_of(&self, ns: KvNamespace) -> u32 {
        if ns == BASE_KV_NAMESPACE {
            return LORA_SLOT_NONE;
        }
        *self.resident_slot.get(&ns).unwrap_or_else(|| panic!("a sequence runs under adapter {ns}, which is not resident in the bank (admit it through adapter_availability and pin_adapter)"))
    }

    fn register(&mut self, mcfg: &QwenConfig, a: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        if let Some(&id) = self.by_digest.get(&a.digest) {
            let e = &self.entries[&id];
            return if e.draining { Err(AdapterError::Draining(id)) } else { Ok(id) };
        }
        if self.entries.len() as u32 >= self.cfg.max_registered {
            return Err(AdapterError::Full);
        }
        let bad = |why: String| AdapterError::Incompatible(why);
        let rank = a.adapter.rank;
        if rank == 0 || rank > self.cfg.r_max {
            return Err(bad(format!("rank {rank} is outside 1..={}", self.cfg.r_max)));
        }
        let scale = a.adapter.scale();
        let r = rank as usize;
        let mut sites = Vec::with_capacity(a.adapter.sites.len());
        for site in &a.adapter.sites {
            let (layer, kind) = LoraSite::parse(&site.base, mcfg.n_layers).ok_or_else(|| bad(format!("{} is not a linear of this model", site.base)))?;
            let bank_ix = *self.site_ix.get(&(layer, kind)).ok_or_else(|| bad(format!("{} is not a site this bank targets", site.base)))?;
            let (k, n) = (self.sites[bank_ix].k as usize, self.sites[bank_ix].n as usize);
            if site.a.len() != r * k || site.b.len() != n * r {
                return Err(bad(format!("{} is [{n}, {k}] but the rank-{rank} pair holds A {} / B {} values", site.base, site.a.len(), site.b.len())));
            }
            // `B [n, r]` -> `[r, n]`, scaled: the layout the bank kernels read.
            let bt: Vec<f32> = (0..r).flat_map(|c| site.b.iter().skip(c).step_by(r).map(move |&v| v * scale)).collect();
            sites.push(HostSite { bank_ix, a: site.a.clone(), bt });
        }
        let id = AdapterId::new(self.ids.fresh()).expect("fresh namespaces are never the base's");
        self.by_digest.insert(a.digest.clone(), id);
        self.entries.insert(id, Entry { label: a.label, digest: a.digest, rank, sites, phase: Phase::Cold, pins: 0, draining: false, last_used: 0 });
        Ok(id)
    }

    /// Upload up to one quantum of `id`'s linears into its reserved slot;
    /// `true` once the slot is complete and published.
    fn upload_step(&mut self, gpu: &Gpu, id: AdapterId) -> bool {
        let r_max = self.cfg.r_max as u64;
        let quantum = self.cfg.upload_quantum_bytes;
        let e = self.entries.get_mut(&id).expect("registered");
        let Phase::Uploading { slot, mut next } = e.phase else { unreachable!("upload_step on a non-uploading adapter") };
        let mut sent = 0u64;
        while next < e.sites.len() && (sent == 0 || sent < quantum) {
            let hs = &e.sites[next];
            let bs = &self.sites[hs.bank_ix];
            gpu.write_f32_at(&bs.bank.a, slot as u64 * r_max * bs.k as u64, &hs.a);
            gpu.write_f32_at(&bs.bank.bt, slot as u64 * r_max * bs.n as u64, &hs.bt);
            sent += (hs.a.len() + hs.bt.len()) as u64 * 4;
            next += 1;
        }
        if next < e.sites.len() {
            e.phase = Phase::Uploading { slot, next };
            return false;
        }
        // Ranks last: a row can only name this slot once every factor is in.
        // A linear the adapter does not correct gets rank 0 (row untouched).
        let mut ranks = vec![0u32; self.sites.len()];
        for hs in &e.sites {
            ranks[hs.bank_ix] = e.rank;
        }
        for (bs, rank) in self.sites.iter().zip(ranks) {
            gpu.write_at(&bs.bank.rank, slot as u64, &[rank]);
        }
        e.phase = Phase::Resident { slot };
        let ns = id.kv_namespace();
        self.resident_slot.insert(ns, slot);
        true
    }

    fn touch(&mut self, id: AdapterId) {
        self.tick += 1;
        self.entries.get_mut(&id).expect("registered").last_used = self.tick;
    }

    /// A slot for `id`: a free one, else the least recently used resident
    /// adapter nothing pins (and that is not mid-offer), which is demoted to
    /// `Cold`. `None` when every slot is held.
    fn claim_slot(&mut self) -> Option<u32> {
        if let Some(free) = self.slot_owner.iter().position(Option::is_none) {
            return Some(free as u32);
        }
        let victim = self
            .entries
            .iter()
            .filter(|(id, e)| matches!(e.phase, Phase::Resident { .. }) && e.pins == 0 && self.offered != Some(**id))
            .min_by_key(|(_, e)| e.last_used)
            .map(|(id, _)| *id)?;
        let e = self.entries.get_mut(&victim).expect("just found");
        let Phase::Resident { slot } = e.phase else { unreachable!("filtered on Resident") };
        e.phase = Phase::Cold;
        self.resident_slot.remove(&victim.kv_namespace());
        self.slot_owner[slot as usize] = None;
        Some(slot)
    }

    fn availability(&mut self, gpu: &Gpu, id: AdapterId) -> AdapterAvailability {
        match self.entries.get(&id) {
            None => return AdapterAvailability::Unknown,
            Some(e) if e.draining => return AdapterAvailability::Unknown,
            Some(_) => {}
        }
        // A different adapter being asked about means the previous offer was
        // not taken up.
        if let Some(old) = self.offered.filter(|&o| o != id) {
            self.offered = None;
            if self.entries.get(&old).is_some_and(|e| e.draining && e.pins == 0) {
                let ns = self.remove(old);
                self.pending_purge.push(ns);
            }
        }
        if matches!(self.entries[&id].phase, Phase::Cold) {
            let Some(slot) = self.claim_slot() else { return AdapterAvailability::Loading };
            self.slot_owner[slot as usize] = Some(id);
            self.entries.get_mut(&id).expect("registered").phase = Phase::Uploading { slot, next: 0 };
        }
        if matches!(self.entries[&id].phase, Phase::Uploading { .. }) && !self.upload_step(gpu, id) {
            return AdapterAvailability::Loading;
        }
        self.touch(id);
        self.offered = Some(id);
        AdapterAvailability::Resident
    }

    fn pin(&mut self, id: AdapterId) {
        let e = self.entries.get_mut(&id).unwrap_or_else(|| panic!("pin_adapter: adapter {} is not registered", id.kv_namespace()));
        assert!(matches!(e.phase, Phase::Resident { .. }), "pin_adapter: adapter {} is not resident (it must have just answered Resident)", id.kv_namespace());
        e.pins += 1;
        if self.offered == Some(id) {
            self.offered = None;
        }
        self.touch(id);
    }

    /// Release one pin; the namespace to purge when this completed a deferred unload.
    fn unpin(&mut self, id: AdapterId) -> Option<KvNamespace> {
        let e = self.entries.get_mut(&id)?;
        e.pins = e.pins.checked_sub(1).unwrap_or_else(|| panic!("unpin_adapter: adapter {} has no pin to release", id.kv_namespace()));
        if e.pins == 0 && e.draining && self.offered != Some(id) {
            return Some(self.remove(id));
        }
        None
    }

    /// Request the unload; the namespace to purge when it happened now.
    fn unload(&mut self, id: AdapterId) -> Result<Option<KvNamespace>, AdapterError> {
        let e = self.entries.get_mut(&id).ok_or(AdapterError::Unknown(id))?;
        if e.pins > 0 || self.offered == Some(id) {
            e.draining = true;
            return Ok(None);
        }
        Ok(Some(self.remove(id)))
    }

    fn remove(&mut self, id: AdapterId) -> KvNamespace {
        let e = self.entries.remove(&id).expect("registered");
        self.by_digest.remove(&e.digest);
        match e.phase {
            Phase::Resident { slot } | Phase::Uploading { slot, .. } => self.slot_owner[slot as usize] = None,
            Phase::Cold => {}
        }
        let ns = id.kv_namespace();
        self.resident_slot.remove(&ns);
        ns
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
        self.bank = Some(AdapterBank::new(&self.gpu, &self.cfg, cfg, rows)?);
        self.tape_cache.retain(|&(_, lora), _| !lora);
        self.warn_if_calib_with_adapters();
        Ok(())
    }

    /// Device bytes the adapter bank occupies; 0 without one.
    pub fn adapter_bank_bytes(&self) -> u64 {
        self.bank.as_ref().map_or(0, |b| b.bytes)
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
        let id = bank.register(&self.cfg, adapter)?;
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
        let Some(bank) = &self.bank else { return Vec::new() };
        let mut v: Vec<AdapterInfo> = bank
            .entries
            .iter()
            .map(|(&id, e)| AdapterInfo { id, label: e.label.clone(), digest: e.digest.clone(), rank: e.rank, resident: matches!(e.phase, Phase::Resident { .. }), pins: e.pins, draining: e.draining })
            .collect();
        v.sort_by_key(|i| i.id.kv_namespace());
        v
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
        self.bank.as_ref().is_some_and(|b| b.targets_any(sites))
    }

    /// Record the bank correction of one linear of the adapter tape.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lora_correct(&self, steps: &mut Vec<Step>, layer: usize, site: LoraSite, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, out_stride: u32, out_col0: u32) {
        if let Some(bank) = &self.bank {
            bank.correct(&self.gpu, steps, layer, site, x, out, rows, out_stride, out_col0);
        }
    }

    pub(super) fn bank_availability(&mut self, id: AdapterId) -> AdapterAvailability {
        let Some(bank) = &mut self.bank else { return AdapterAvailability::Unknown };
        let answer = bank.availability(&self.gpu, id);
        for ns in std::mem::take(&mut bank.pending_purge) {
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
