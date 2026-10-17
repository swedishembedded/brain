// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The generic adapter bank: many LoRA adapters, or none, applied to
//! different rows of ONE batch on ONE resident copy of the base weights, for
//! any model that lists its adapter sites ([`LoraSites`]).
//!
//! ## Shape
//!
//! [`AdapterBank::new`] allocates, for every [`LinearSite`] the bank targets,
//! a [`LoraBank`] of `slots` adapters padded to `r_max`, plus one `[rows]`
//! slot vector. A row's slot says which adapter corrects it
//! ([`LORA_SLOT_NONE`]: none). A decode tape records
//! [`AdapterBank::correct`] after each targeted linear: two dispatches
//! whatever number of distinct adapters the batch holds. A batch with no
//! adapter row replays the unchanged base tape.
//!
//! ## What is per model
//!
//! Only the sites and the defaults. [`AdapterServing`] extends [`LoraSites`]
//! with the target groups (`attention`, `all`), the default slot count and
//! rank cap; a model overrides what differs. An operator's choice of targets
//! is an [`AdapterTargets`]: groups and site leaf names, resolved against
//! the model's own sites by [`AdapterTargets::resolve`], so a name the model
//! has no linear for is refused with the names it does have.
//!
//! ## Lifecycle
//!
//! An adapter is *registered* by [`AdapterBank::register`] (host memory only,
//! one [`AdapterId`] = one KV namespace per distinct digest, never reused),
//! becomes *resident* in a slot when the scheduler asks about it
//! ([`AdapterBank::availability`]: the upload is spread over calls,
//! [`AdapterBankConfig::upload_quantum_bytes`] each, so a cold load never
//! stalls decode), is *pinned* for the lifetime of each sequence using it, and
//! is evicted from its slot (least recently used, never while pinned) when
//! another adapter needs the slot. Eviction keeps the registration and the
//! cached KV: the namespace is tied to the adapter, not to the slot.
//! [`AdapterBank::unload`] removes the registration; while a sequence still
//! uses the adapter it only marks the adapter draining (no new admissions) and
//! the last unpin completes it. Operations that remove a registration return
//! the [`KvNamespace`] whose prefix-cache entries the engine must purge.
//!
//! ## Verification
//!
//! The bank trusts its input to have been verified against the served base
//! (card family and kind, base digest or id, content digest of the bytes the
//! tensors were parsed from); that check lives with the catalog, which hands
//! over a [`VerifiedAdapter`]. The bank re-checks everything it can see
//! itself: every pair is a site this bank targets, with that site's
//! dimensions, and a rank within `r_max`.

use std::collections::HashMap;
use std::fmt;

use gpu_core::{DeviceBuffer, Gpu, Step};

use super::sites::LoraSites;
use super::LinearSite;
use crate::dispatch::{lora_bank_group_rows, lora_bank_rows, LoraBank, LoraBankKernels, LoraBankRows, LoraGroupBank, LoraGroupOut, LORA_GROUP_MAX, LORA_SLOT_NONE};
use crate::lora::device_adapter::DeviceAdapter;
use crate::paged::{KvNamespace, NamespaceIds, BASE_KV_NAMESPACE};
use crate::serve::{AdapterAvailability, AdapterId};

/// The leaf names of the `attention` target group unless a model says
/// otherwise: the query, key, value and output projections.
pub const DEFAULT_ATTENTION_LEAVES: [&str; 4] = ["wq", "wk", "wv", "wo"];

/// A model whose adapter sites can be served from a bank. Everything has a
/// default; a model overrides only what differs from a plain transformer.
pub trait AdapterServing: LoraSites {
    /// The leaves of the `attention` group: what an adapter trained with the
    /// default LoRA configuration targets, and the default of a bank.
    fn attention_leaves(&self) -> Vec<&'static str> {
        DEFAULT_ATTENTION_LEAVES.to_vec()
    }

    /// Every leaf the model has a site for, in first-seen (block) order: the
    /// `all` group.
    fn all_leaves(&self) -> Vec<&'static str> {
        let mut leaves: Vec<&'static str> = Vec::new();
        for site in self.lora_sites() {
            if !leaves.contains(&site.leaf) {
                leaves.push(site.leaf);
            }
        }
        leaves
    }

    /// Leaves whose linears read the SAME activation, in the order a forward
    /// finishes them: a bank corrects such a group with one projection and one
    /// epilogue instead of two dispatches per linear
    /// ([`AdapterBank::correct_named_many`]). At most [`LORA_GROUP_MAX`] leaves
    /// per group. None by default: a model opts in when its forward calls the
    /// bank per group.
    fn input_groups(&self) -> Vec<Vec<&'static str>> {
        Vec::new()
    }

    /// Adapters resident in device slots at once unless the operator says.
    fn default_slots(&self) -> u32 {
        4
    }

    /// The highest adapter rank a slot holds unless the operator says.
    fn default_rank_cap(&self) -> u32 {
        16
    }
}

/// An operator's choice of the linears a bank corrects: the `attention` group,
/// every site (`all`), and/or named site leaves. Model independent, so it can
/// be parsed from a flag and carried in settings before any model is known;
/// [`AdapterTargets::resolve`] checks it against the model.
#[derive(Clone, PartialEq, Eq)]
pub struct AdapterTargets {
    attention: bool,
    all: bool,
    leaves: Vec<String>,
}

impl AdapterTargets {
    /// The model's `attention` group ([`AdapterServing::attention_leaves`]).
    pub const ATTENTION: AdapterTargets = AdapterTargets { attention: true, all: false, leaves: Vec::new() };
    /// Every linear the model has a site for.
    pub const ALL: AdapterTargets = AdapterTargets { attention: false, all: true, leaves: Vec::new() };

    /// The set a `--adapter-targets` value names: `attention`, `all`, or a
    /// comma-separated list of site leaves (`wq,down`, `in_proj_qkv,gate`).
    /// Whether a leaf exists is the model's to say ([`AdapterTargets::resolve`]).
    pub fn parse(list: &str) -> Result<AdapterTargets, String> {
        let mut targets = AdapterTargets { attention: false, all: false, leaves: Vec::new() };
        for name in list.split(',').map(str::trim) {
            match name {
                "attention" => targets.attention = true,
                "all" => targets.all = true,
                "" => return Err("an empty target name; give attention, all, or a list of site names".to_string()),
                leaf if leaf.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') => {
                    if !targets.leaves.iter().any(|l| l == leaf) {
                        targets.leaves.push(leaf.to_string());
                    }
                }
                other => return Err(format!("{other:?} is not a site name; give attention, all, or a list of site names")),
            }
        }
        Ok(targets)
    }

    /// Every target in either set.
    pub fn union(mut self, other: AdapterTargets) -> AdapterTargets {
        self.attention |= other.attention;
        self.all |= other.all;
        for leaf in other.leaves {
            if !self.leaves.contains(&leaf) {
                self.leaves.push(leaf);
            }
        }
        self
    }

    /// Does this choice cover a site named `leaf`, taking `attention` as the
    /// attention group? The check the catalog makes before an adapter reaches
    /// an engine that can resolve it exactly.
    pub fn admits(&self, leaf: &str, attention: &[&str]) -> bool {
        self.all || (self.attention && attention.contains(&leaf)) || self.leaves.iter().any(|l| l == leaf)
    }

    /// The site leaves this choice names for `model`, in block order.
    ///
    /// # Errors
    ///
    /// A named leaf the model has no site for (the message lists the ones it
    /// has), or a choice that names nothing.
    pub fn resolve<M: AdapterServing + ?Sized>(&self, model: &M) -> Result<Vec<&'static str>, String> {
        let known = model.all_leaves();
        let mut wanted: Vec<&'static str> = Vec::new();
        for name in &self.leaves {
            match known.iter().find(|k| **k == name) {
                Some(&leaf) => wanted.push(leaf),
                None => return Err(format!("{name:?} is not an adapter target of this model; give attention, all, or a list of {}", known.join(", "))),
            }
        }
        if self.attention {
            wanted.extend(model.attention_leaves());
        }
        if self.all {
            wanted.extend(known.iter().copied());
        }
        let resolved: Vec<&'static str> = known.into_iter().filter(|k| wanted.contains(k)).collect();
        if resolved.is_empty() {
            return Err("--adapter-targets names no linear".to_string());
        }
        Ok(resolved)
    }
}

impl fmt::Display for AdapterTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<&str> = Vec::new();
        if self.all {
            names.push("all");
        }
        if self.attention {
            names.push("attention");
        }
        names.extend(self.leaves.iter().map(String::as_str));
        write!(f, "{}", names.join(","))
    }
}

impl fmt::Debug for AdapterTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdapterTargets({self})")
    }
}

/// An engine with an adapter bank, as a registry drives it: adapters are
/// registered into and removed from the bank by their content digest.
/// Implemented by every serving engine that owns an [`AdapterBank`].
pub trait BankedEngine {
    /// Register a verified adapter (host memory only); see [`AdapterBank::register`].
    fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError>;
    /// Unload an adapter, or mark it draining while sequences still use it.
    fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError>;
}

/// How an operator sized a bank, before any model is known: slots, rank cap
/// and the target choice. [`AdapterBankRequest::config`] resolves it against a
/// model; a resident that places a model on cards uses [`Self::device_bytes_by_layer`]
/// to charge the bank to the stage that holds each layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterBankRequest {
    pub slots: u32,
    pub rank_cap: u32,
    pub targets: AdapterTargets,
    pub max_registered: u32,
}

impl AdapterBankRequest {
    /// The bank configuration for `model`.
    ///
    /// # Errors
    ///
    /// A target the model has no site for ([`AdapterTargets::resolve`]).
    pub fn config<M: AdapterServing + ?Sized>(&self, model: &M) -> Result<AdapterBankConfig, String> {
        let mut cfg = AdapterBankConfig::new(self.slots, self.rank_cap, &self.targets.resolve(model)?);
        cfg.max_registered = self.max_registered;
        cfg.groups = model.input_groups().into_iter().map(|g| g.into_iter().map(str::to_string).collect()).collect();
        Ok(cfg)
    }
}

impl AdapterBankConfig {
    /// Device bytes of the sites of each layer `0..n_layers` (the per-stage
    /// term of [`AdapterBankConfig::device_bytes`], without the row scratch).
    pub fn device_bytes_by_layer<M: LoraSites + ?Sized>(&self, model: &M, n_layers: usize) -> Vec<u64> {
        let sites = model.lora_sites();
        let mut by_layer = vec![0u64; n_layers];
        for unit in self.units(&sites, &(0..usize::MAX)) {
            if let Some(l) = sites[unit[0]].layer.filter(|&l| l < n_layers) {
                by_layer[l] += self.unit_bytes(&sites, &unit);
            }
        }
        by_layer
    }

    /// Device bytes of the row scratch every bank carries for forwards of at
    /// most `rows` rows.
    pub fn row_scratch_bytes(&self, rows: u32) -> u64 {
        rows as u64 * 4 + rows as u64 * self.scratch_width() * 4
    }
}

/// Something that names one linear by its site leaf, so a bank can be
/// configured from a model's own typed site names as well as from strings.
pub trait SiteLeaf {
    fn site_leaf(&self) -> &str;
}

impl SiteLeaf for &str {
    fn site_leaf(&self) -> &str {
        self
    }
}

impl SiteLeaf for String {
    fn site_leaf(&self) -> &str {
        self
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
    /// The site leaves the bank can correct; an adapter targeting any other
    /// is refused. Device cost is `slots * r_max * (k + n) * 4` bytes per
    /// targeted linear.
    pub targets: Vec<String>,
    /// Adapters that may be registered at once (resident or not); their
    /// factors are held in host memory until unloaded.
    pub max_registered: u32,
    /// Bytes of adapter factors one [`AdapterBank::availability`] call
    /// uploads before answering `Loading`: bounds the time a cold load takes
    /// out of one scheduler iteration. At least one linear is uploaded per call.
    pub upload_quantum_bytes: u64,
    /// Leaves that read one activation ([`AdapterServing::input_groups`]); the
    /// targeted members of a group in one layer share a stacked bank and are
    /// corrected together. Empty: every linear is corrected on its own.
    pub groups: Vec<Vec<String>>,
}

impl AdapterBankConfig {
    /// A bank of `slots` adapters up to `r_max` rank, correcting the linears
    /// named by `targets`, with room to register 64 adapters and a 4 MiB
    /// upload quantum. Names are checked against the model when the bank is
    /// built ([`AdapterBank::new`]).
    pub fn new<T: SiteLeaf>(slots: u32, r_max: u32, targets: &[T]) -> AdapterBankConfig {
        AdapterBankConfig { slots, r_max, targets: targets.iter().map(|t| t.site_leaf().to_string()).collect(), max_registered: 64, upload_quantum_bytes: 4 << 20, groups: Vec::new() }
    }

    /// A bank with `model`'s default size, correcting what `targets` resolves
    /// to on it.
    pub fn for_model<M: AdapterServing + ?Sized>(model: &M, targets: &AdapterTargets) -> Result<AdapterBankConfig, String> {
        let mut cfg = AdapterBankConfig::new(model.default_slots(), model.default_rank_cap(), &targets.resolve(model)?);
        cfg.groups = model.input_groups().into_iter().map(|g| g.into_iter().map(str::to_string).collect()).collect();
        Ok(cfg)
    }

    fn targets_site(&self, site: &LinearSite) -> bool {
        self.targets.iter().any(|t| t == site.leaf)
    }

    /// The bank's units over `sites` in `layers`: each a single linear or a
    /// group of two or more targeted linears of one layer that read one
    /// activation, as site indices. The one place the grouping is decided, for
    /// allocation and for the byte count alike.
    fn units(&self, sites: &[LinearSite], layers: &std::ops::Range<usize>) -> Vec<Vec<usize>> {
        let mut taken = vec![false; sites.len()];
        let mut units = Vec::new();
        for i in 0..sites.len() {
            let site = &sites[i];
            if taken[i] || site.layer.is_some_and(|l| !layers.contains(&l)) || !self.targets_site(site) {
                continue;
            }
            let members = self.groups.iter().find(|g| g.iter().any(|leaf| leaf == site.leaf)).map(|group| {
                group
                    .iter()
                    .take(LORA_GROUP_MAX)
                    .filter_map(|leaf| (0..sites.len()).find(|&j| !taken[j] && sites[j].layer == site.layer && sites[j].leaf == leaf.as_str() && sites[j].spec.inn == site.spec.inn && self.targets_site(&sites[j])))
                    .collect::<Vec<usize>>()
            });
            let unit = match members {
                Some(m) if m.len() >= 2 && m.contains(&i) => m,
                _ => vec![i],
            };
            for &j in &unit {
                taken[j] = true;
            }
            units.push(unit);
        }
        units
    }

    /// Device bytes of one unit.
    fn unit_bytes(&self, sites: &[LinearSite], unit: &[usize]) -> u64 {
        let (s, r) = (self.slots as u64, self.r_max as u64);
        let k = sites[unit[0]].spec.inn as u64;
        match unit {
            [one] => s * r * (k + sites[*one].spec.out as u64) * 4 + s * 4,
            many => {
                let n: u64 = many.iter().map(|&i| sites[i].spec.out as u64).sum();
                s * many.len() as u64 * r * k * 4 + s * r * n * 4 + s * 4 + 2 * 4
            }
        }
    }

    /// Floats per row of the projection scratch: `r_max` per site of the
    /// largest group.
    fn scratch_width(&self) -> u64 {
        self.r_max as u64 * self.groups.iter().map(|g| g.len().min(LORA_GROUP_MAX)).max().unwrap_or(1).max(1) as u64
    }

    /// Device bytes the bank allocates for `model` in an engine whose batches
    /// hold at most `rows` rows (the larger of its decode batch and prefill
    /// chunk): [`AdapterBank::bytes`] once the bank exists, known before any
    /// engine does, so a caller can budget the bank when it places the engine.
    /// Allocation sums the buffers it makes and a test holds the two equal.
    pub fn device_bytes<M: LoraSites + ?Sized>(&self, model: &M, rows: u32) -> u64 {
        let sites = model.lora_sites();
        let units: u64 = self.units(&sites, &(0..usize::MAX)).iter().map(|u| self.unit_bytes(&sites, u)).sum();
        units + self.row_scratch_bytes(rows)
    }
}

/// An adapter the caller has verified against the served base, in host
/// memory, ready for registration.
#[derive(Clone)]
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
    /// The engine has no adapter bank.
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

/// What an unload did.
#[derive(Debug, PartialEq, Eq)]
pub enum UnloadStatus {
    /// Gone: slot freed; the engine purges the cached KV.
    Unloaded,
    /// Sequences still use it. It admits no new ones and is unloaded when the
    /// last one unpins.
    Draining,
}

/// A registered adapter, as [`AdapterBank::infos`] reports it.
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

/// What a bank does with one site of the model.
#[derive(Clone, Copy)]
enum SiteState {
    /// Corrected here on its own: index into the bank's sites.
    Banked(usize),
    /// Corrected here together with the other linears of its group that read
    /// the same activation: the group's index and the linear's place in it.
    Grouped { group: usize, member: usize },
    /// A site of the model that this bank's targets leave alone: an adapter
    /// that corrects it is refused.
    Untargeted,
    /// A site of a layer another bank owns (a pipeline stage): an adapter's
    /// pair for it is that bank's to hold, so this one skips it.
    Elsewhere,
}

/// One banked linear: its adapters on the device and its shape.
struct BankSite {
    bank: LoraBank,
    k: u32,
    n: u32,
}

/// A group of linears that read one activation: one stacked bank for all of them.
struct GroupSite {
    bank: LoraGroupBank,
    k: u32,
    /// The members' output widths, in group order.
    ns: Vec<u32>,
}

/// One linear of a registered adapter, in the bank's layout.
struct HostSite {
    /// `A [rank, k]`.
    a: Vec<f32>,
    /// `B` transposed to `[rank, n]` with `alpha / rank` folded in.
    bt: Vec<f32>,
}

/// What one upload step writes: a linear on its own, or a whole group (a member
/// the adapter does not correct is `None` and is written as zeros, so no stale
/// factor of the slot's previous adapter survives in the stacked bank).
enum HostUnit {
    Single { bank_ix: usize, site: HostSite },
    Group { group: usize, members: Vec<Option<HostSite>> },
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
    sites: Vec<HostUnit>,
    phase: Phase,
    pins: u32,
    draining: bool,
    last_used: u64,
}

/// The device bank plus the host-side registry that decides who is in it.
pub struct AdapterBank {
    cfg: AdapterBankConfig,
    kernels: LoraBankKernels,
    sites: Vec<BankSite>,
    groups: Vec<GroupSite>,
    /// Base weight name of every site of the model, banked or not: tells "not
    /// a linear of this model" from "not a site this bank targets".
    model_sites: HashMap<String, SiteState>,
    /// Per layer, the bank index of each banked site by leaf.
    by_layer: Vec<HashMap<&'static str, usize>>,
    slot_buf: DeviceBuffer,
    /// `[rows, r_max · largest group]` projection scratch shared by every corrected linear.
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
    /// Allocate a bank over `sites` (the model's [`LoraSites::lora_sites`]),
    /// for batches of at most `rows` rows.
    pub fn new(gpu: &Gpu, sites: &[LinearSite], cfg: AdapterBankConfig, rows: u32) -> Result<AdapterBank, AdapterError> {
        AdapterBank::new_for_layers(gpu, sites, cfg, rows, 0..usize::MAX)
    }

    /// [`AdapterBank::new`] for one pipeline stage: only the sites of `layers`
    /// are allocated (on `gpu`, the stage's device); an adapter's pairs for
    /// other layers are skipped at registration, to be held by the bank of the
    /// stage that owns them. Every stage's bank sees the same registrations,
    /// uploads and pins, so their slots and ids agree.
    pub fn new_for_layers(gpu: &Gpu, sites: &[LinearSite], cfg: AdapterBankConfig, rows: u32, layers: std::ops::Range<usize>) -> Result<AdapterBank, AdapterError> {
        if cfg.slots == 0 || cfg.r_max == 0 || cfg.targets.is_empty() || cfg.max_registered < cfg.slots {
            return Err(AdapterError::BadConfig("needs slots > 0, r_max > 0, at least one target and max_registered >= slots".into()));
        }
        if let Some(unknown) = cfg.targets.iter().find(|t| !sites.iter().any(|s| s.leaf == t.as_str())) {
            let mut known: Vec<&str> = Vec::new();
            for s in sites {
                if !known.contains(&s.leaf) {
                    known.push(s.leaf);
                }
            }
            return Err(AdapterError::BadConfig(format!("{unknown:?} is not an adapter target of this model; it has {}", known.join(", "))));
        }
        let kernels = LoraBankKernels::resolve(gpu).ok_or_else(|| AdapterError::BadConfig("the device does not register the LoRA bank kernels".into()))?;
        let (s, r) = (cfg.slots as u64, cfg.r_max as u64);
        let mut banked = Vec::new();
        let mut grouped: Vec<GroupSite> = Vec::new();
        let mut model_sites = HashMap::new();
        let mut by_layer: Vec<HashMap<&'static str, usize>> = Vec::new();
        let mut bytes = 0u64;
        for site in sites {
            if site.layer.is_some_and(|l| !layers.contains(&l)) {
                model_sites.insert(site.name.clone(), SiteState::Elsewhere);
            } else if !cfg.targets_site(site) {
                model_sites.insert(site.name.clone(), SiteState::Untargeted);
            }
        }
        for unit in cfg.units(sites, &layers) {
            let k = sites[unit[0]].spec.inn as u32;
            let widest = unit.iter().map(|&i| s * r * (unit.len() as u64 * k as u64).max(sites[i].spec.out as u64) * 4).max().unwrap_or(0);
            if widest > gpu.max_storage_binding_bytes() {
                return Err(AdapterError::BadConfig(format!("{} slots of rank {} at {} x {} exceed one storage binding ({widest} bytes)", cfg.slots, cfg.r_max, k, sites[unit[0]].spec.out)));
            }
            let rank = gpu.storage(s);
            gpu.write(&rank, &vec![0u32; cfg.slots as usize]);
            bytes += cfg.unit_bytes(sites, &unit);
            if let [one] = unit[..] {
                let site = &sites[one];
                let n = site.spec.out as u32;
                let bank = LoraBank { a: gpu.storage(s * r * k as u64), bt: gpu.storage(s * r * n as u64), rank, slots: cfg.slots, r_max: cfg.r_max };
                model_sites.insert(site.name.clone(), SiteState::Banked(banked.len()));
                if let Some(layer) = site.layer {
                    if by_layer.len() <= layer {
                        by_layer.resize_with(layer + 1, HashMap::new);
                    }
                    by_layer[layer].insert(site.leaf, banked.len());
                }
                banked.push(BankSite { bank, k, n });
            } else {
                let ns: Vec<u32> = unit.iter().map(|&i| sites[i].spec.out as u32).collect();
                let total_n: u64 = ns.iter().map(|&n| n as u64).sum();
                let bank = LoraGroupBank {
                    a: gpu.storage(s * unit.len() as u64 * r * k as u64),
                    bt: gpu.storage(s * r * total_n),
                    rank,
                    slots: cfg.slots,
                    r_max: cfg.r_max,
                    spare: [gpu.storage(1), gpu.storage(1)],
                };
                for (member, &i) in unit.iter().enumerate() {
                    model_sites.insert(sites[i].name.clone(), SiteState::Grouped { group: grouped.len(), member });
                }
                grouped.push(GroupSite { bank, k, ns });
            }
        }
        let slot_buf = gpu.storage(rows as u64);
        let tscr = gpu.storage(rows as u64 * cfg.scratch_width());
        bytes += cfg.row_scratch_bytes(rows);
        let slot_owner = vec![None; cfg.slots as usize];
        Ok(AdapterBank { cfg, kernels, sites: banked, groups: grouped, model_sites, by_layer, slot_buf, tscr, entries: HashMap::new(), by_digest: HashMap::new(), slot_owner, resident_slot: HashMap::new(), ids: NamespaceIds::default(), offered: None, pending_purge: Vec::new(), tick: 0, bytes })
    }

    /// Device bytes the bank occupies.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The `[rows]` u32 buffer the caller writes each row's slot into before
    /// replaying a tape that holds [`AdapterBank::correct`] steps.
    pub fn slot_buf(&self) -> &DeviceBuffer {
        &self.slot_buf
    }

    /// Does the bank correct any site named by `leaves`?
    pub fn targets_any(&self, leaves: &[&str]) -> bool {
        leaves.iter().any(|l| self.cfg.targets.iter().any(|t| t == l))
    }

    /// Record the two dispatches correcting `out` for the `leaf` linear of
    /// `layer` (`x` is the activation that linear read; `out` its output, at
    /// `out_stride` floats per row, the correction landing at `out_col0`);
    /// nothing when the bank does not target it.
    #[allow(clippy::too_many_arguments)]
    pub fn correct(&self, gpu: &Gpu, steps: &mut Vec<Step>, layer: usize, leaf: &str, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, out_stride: u32, out_col0: u32) {
        let Some(&ix) = self.by_layer.get(layer).and_then(|m| m.get(leaf)) else { return };
        let bs = &self.sites[ix];
        let rq = LoraBankRows { kernels: self.kernels, bank: &bs.bank, slot: &self.slot_buf, tscr: &self.tscr, x, out, row0: 0, rows, k: bs.k, n: bs.n, out_stride, out_col0 };
        steps.extend(lora_bank_rows(gpu, &rq));
    }

    /// [`AdapterBank::correct`] for the linear whose base weight is named
    /// `base` (`blocks.3.mlp.down.weight`): the hook for a forward that knows
    /// its linears by weight name.
    #[allow(clippy::too_many_arguments)]
    pub fn correct_named(&self, gpu: &Gpu, steps: &mut Vec<Step>, base: &str, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, out_stride: u32, out_col0: u32) {
        let Some(&SiteState::Banked(ix)) = self.model_sites.get(base) else { return };
        let bs = &self.sites[ix];
        let rq = LoraBankRows { kernels: self.kernels, bank: &bs.bank, slot: &self.slot_buf, tscr: &self.tscr, x, out, row0: 0, rows, k: bs.k, n: bs.n, out_stride, out_col0 };
        steps.extend(lora_bank_rows(gpu, &rq));
    }

    /// Correct every linear in `items` (`(base weight name, its output)`, each
    /// output a dense `[rows, n]`) that reads the activation `x`, the linears of
    /// one group in one pair of dispatches and the others on their own. Names the
    /// bank does not correct are ignored. A group must be passed whole: its
    /// stacked bank is one projection, so a call naming only some of its members
    /// is a bug in the forward and panics.
    pub fn correct_named_many(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, items: &[(&str, &DeviceBuffer)]) {
        let mut by_group: Vec<(usize, Vec<(usize, &DeviceBuffer)>)> = Vec::new();
        for &(name, out) in items {
            match self.model_sites.get(name) {
                Some(&SiteState::Banked(ix)) => {
                    let bs = &self.sites[ix];
                    let rq = LoraBankRows { kernels: self.kernels, bank: &bs.bank, slot: &self.slot_buf, tscr: &self.tscr, x, out, row0: 0, rows, k: bs.k, n: bs.n, out_stride: bs.n, out_col0: 0 };
                    steps.extend(lora_bank_rows(gpu, &rq));
                }
                Some(&SiteState::Grouped { group, member }) => match by_group.iter_mut().find(|(g, _)| *g == group) {
                    Some((_, members)) => members.push((member, out)),
                    None => by_group.push((group, vec![(member, out)])),
                },
                _ => {}
            }
        }
        for (group, mut members) in by_group {
            let gs = &self.groups[group];
            members.sort_by_key(|&(member, _)| member);
            assert!(members.iter().map(|&(m, _)| m).eq(0..gs.ns.len()), "the adapter bank corrects these linears as one group of {}, but a forward named {} of them", gs.ns.len(), members.len());
            let outs: Vec<LoraGroupOut> = members.iter().map(|&(m, out)| LoraGroupOut { out, n: gs.ns[m], stride: gs.ns[m], col0: 0 }).collect();
            steps.extend(lora_bank_group_rows(gpu, self.kernels, &gs.bank, &self.slot_buf, &self.tscr, x, &outs, rows, gs.k));
        }
    }

    /// Write each row's slot ([`AdapterBank::slot_of`]) for the batch about to
    /// run; at most the `rows` the bank was built for.
    pub fn write_slots(&self, gpu: &Gpu, slots: &[u32]) {
        gpu.write(&self.slot_buf, slots);
    }

    /// The slot a sequence in `ns` runs under. Panics for an adapter that is
    /// not resident: answering such a row with the base would be a silent
    /// wrong answer, and the scheduler never admits one.
    pub fn slot_of(&self, ns: KvNamespace) -> u32 {
        if ns == BASE_KV_NAMESPACE {
            return LORA_SLOT_NONE;
        }
        *self.resident_slot.get(&ns).unwrap_or_else(|| panic!("a sequence runs under adapter {ns}, which is not resident in the bank (admit it through adapter_availability and pin_adapter)"))
    }

    /// Register a verified adapter in host memory; the id is also its KV
    /// namespace and is never reused. Registering a digest already registered
    /// returns that adapter's id.
    pub fn register(&mut self, a: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
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
        let mut units: Vec<HostUnit> = Vec::with_capacity(a.adapter.sites.len());
        let mut group_members: Vec<(usize, Vec<Option<HostSite>>)> = Vec::new();
        for site in &a.adapter.sites {
            let state = match self.model_sites.get(&site.base) {
                None => return Err(bad(format!("{} is not a linear of this model", site.base))),
                Some(SiteState::Untargeted) => return Err(bad(format!("{} is not a site this bank targets", site.base))),
                Some(SiteState::Elsewhere) => continue,
                Some(&state) => state,
            };
            let (k, n) = match state {
                SiteState::Banked(ix) => (self.sites[ix].k as usize, self.sites[ix].n as usize),
                SiteState::Grouped { group, member } => (self.groups[group].k as usize, self.groups[group].ns[member] as usize),
                _ => unreachable!("filtered above"),
            };
            if site.a.len() != r * k || site.b.len() != n * r {
                return Err(bad(format!("{} is [{n}, {k}] but the rank-{rank} pair holds A {} / B {} values", site.base, site.a.len(), site.b.len())));
            }
            // `B [n, r]` -> `[r, n]`, scaled: the layout the bank kernels read.
            let bt: Vec<f32> = (0..r).flat_map(|c| site.b.iter().skip(c).step_by(r).map(move |&v| v * scale)).collect();
            let host = HostSite { a: site.a.clone(), bt };
            match state {
                SiteState::Banked(bank_ix) => units.push(HostUnit::Single { bank_ix, site: host }),
                SiteState::Grouped { group, member } => {
                    let at = match group_members.iter().position(|(g, _)| *g == group) {
                        Some(at) => at,
                        None => {
                            group_members.push((group, (0..self.groups[group].ns.len()).map(|_| None).collect()));
                            group_members.len() - 1
                        }
                    };
                    group_members[at].1[member] = Some(host);
                }
                _ => unreachable!("filtered above"),
            }
        }
        units.extend(group_members.into_iter().map(|(group, members)| HostUnit::Group { group, members }));
        let sites = units;
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
            match &e.sites[next] {
                HostUnit::Single { bank_ix, site } => {
                    let bs = &self.sites[*bank_ix];
                    gpu.write_f32_at(&bs.bank.a, slot as u64 * r_max * bs.k as u64, &site.a);
                    gpu.write_f32_at(&bs.bank.bt, slot as u64 * r_max * bs.n as u64, &site.bt);
                    sent += (site.a.len() + site.bt.len()) as u64 * 4;
                }
                HostUnit::Group { group, members } => {
                    let gs = &self.groups[*group];
                    let (k, r) = (gs.k as usize, r_max as usize);
                    let total_n: usize = gs.ns.iter().map(|&n| n as usize).sum();
                    // Each member's `A` fills the first `rank` rows of its `r_max`-row block
                    // and its `Bt` the first `rank` rows of its block; the rest is zero.
                    let mut a = vec![0.0f32; members.len() * r * k];
                    let mut bt = vec![0.0f32; r * total_n];
                    let mut bt_off = 0usize;
                    for (m, host) in members.iter().enumerate() {
                        if let Some(h) = host {
                            a[m * r * k..m * r * k + h.a.len()].copy_from_slice(&h.a);
                            bt[bt_off..bt_off + h.bt.len()].copy_from_slice(&h.bt);
                        }
                        bt_off += r * gs.ns[m] as usize;
                    }
                    gpu.write_f32_at(&gs.bank.a, slot as u64 * a.len() as u64, &a);
                    gpu.write_f32_at(&gs.bank.bt, slot as u64 * bt.len() as u64, &bt);
                    sent += (a.len() + bt.len()) as u64 * 4;
                }
            }
            next += 1;
        }
        if next < e.sites.len() {
            e.phase = Phase::Uploading { slot, next };
            return false;
        }
        // Ranks last: a row can only name this slot once every factor is in.
        // A linear the adapter does not correct gets rank 0 (row untouched); a
        // group gets the width of its stacked rank axis when the adapter
        // corrects any member, 0 otherwise.
        let mut ranks = vec![0u32; self.sites.len()];
        let mut group_ranks = vec![0u32; self.groups.len()];
        for unit in &e.sites {
            match unit {
                HostUnit::Single { bank_ix, .. } => ranks[*bank_ix] = e.rank,
                HostUnit::Group { group, members } => group_ranks[*group] = r_max as u32 * members.len() as u32,
            }
        }
        for (bs, rank) in self.sites.iter().zip(ranks) {
            gpu.write_at(&bs.bank.rank, slot as u64, &[rank]);
        }
        for (gs, rank) in self.groups.iter().zip(group_ranks) {
            gpu.write_at(&gs.bank.rank, slot as u64, &[rank]);
        }
        e.phase = Phase::Resident { slot };
        self.resident_slot.insert(id.kv_namespace(), slot);
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

    /// Is `id` usable now? Starts (and, one quantum per call, continues) its
    /// upload into a slot. The adapter answered `Resident` is held against
    /// eviction until [`AdapterBank::pin`] or another adapter is asked about.
    pub fn availability(&mut self, gpu: &Gpu, id: AdapterId) -> AdapterAvailability {
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

    /// Namespaces whose registration completed without an unpin; the engine
    /// purges their cached KV.
    pub fn take_purge(&mut self) -> Vec<KvNamespace> {
        std::mem::take(&mut self.pending_purge)
    }

    /// Pin `id` for one more sequence.
    pub fn pin(&mut self, id: AdapterId) {
        let e = self.entries.get_mut(&id).unwrap_or_else(|| panic!("pin_adapter: adapter {} is not registered", id.kv_namespace()));
        assert!(matches!(e.phase, Phase::Resident { .. }), "pin_adapter: adapter {} is not resident (it must have just answered Resident)", id.kv_namespace());
        e.pins += 1;
        if self.offered == Some(id) {
            self.offered = None;
        }
        self.touch(id);
    }

    /// Release one pin; the namespace to purge when this completed a deferred unload.
    pub fn unpin(&mut self, id: AdapterId) -> Option<KvNamespace> {
        let e = self.entries.get_mut(&id)?;
        e.pins = e.pins.checked_sub(1).unwrap_or_else(|| panic!("unpin_adapter: adapter {} has no pin to release", id.kv_namespace()));
        if e.pins == 0 && e.draining && self.offered != Some(id) {
            return Some(self.remove(id));
        }
        None
    }

    /// Request the unload; the namespace to purge when it happened now.
    pub fn unload(&mut self, id: AdapterId) -> Result<Option<KvNamespace>, AdapterError> {
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

    /// Every registered adapter, oldest first.
    pub fn infos(&self) -> Vec<AdapterInfo> {
        let mut v: Vec<AdapterInfo> = self
            .entries
            .iter()
            .map(|(&id, e)| AdapterInfo { id, label: e.label.clone(), digest: e.digest.clone(), rank: e.rank, resident: matches!(e.phase, Phase::Resident { .. }), pins: e.pins, draining: e.draining })
            .collect();
        v.sort_by_key(|i| i.id.kv_namespace());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::TargetSpec;

    /// Two layers of a hybrid: layer 0 mixes with `conv_in`/`conv_out`, layer 1
    /// with `wq`/`wo`; both have an MLP `up`/`down`. Non-square on purpose.
    struct Hybrid;

    impl LoraSites for Hybrid {
        fn lora_sites(&self) -> Vec<LinearSite> {
            let site = |layer: usize, leaf: &'static str, tail: &str, out, inn| LinearSite { name: format!("blocks.{layer}.{tail}"), leaf, layer: Some(layer), spec: TargetSpec::whole(out, inn), save_name: None };
            vec![
                site(0, "conv_in", "mix.in.weight", 6, 5),
                site(0, "conv_out", "mix.out.weight", 5, 6),
                site(0, "up", "mlp.up.weight", 8, 5),
                site(0, "down", "mlp.down.weight", 5, 8),
                site(1, "wq", "attn.wq.weight", 4, 5),
                site(1, "wo", "attn.wo.weight", 5, 4),
                site(1, "up", "mlp.up.weight", 8, 5),
                site(1, "down", "mlp.down.weight", 5, 8),
            ]
        }
    }

    impl AdapterServing for Hybrid {
        fn attention_leaves(&self) -> Vec<&'static str> {
            vec!["conv_in", "conv_out", "wq", "wo"]
        }
        fn default_slots(&self) -> u32 {
            2
        }
    }

    /// A model that overrides nothing.
    struct Plain;

    impl LoraSites for Plain {
        fn lora_sites(&self) -> Vec<LinearSite> {
            Hybrid.lora_sites()
        }
    }

    impl AdapterServing for Plain {}

    #[test]
    fn groups_resolve_per_model_in_block_order() {
        assert_eq!(AdapterTargets::ATTENTION.resolve(&Hybrid).unwrap(), ["conv_in", "conv_out", "wq", "wo"], "the model's own attention group");
        assert_eq!(AdapterTargets::ATTENTION.resolve(&Plain).unwrap(), ["wq", "wo"], "the default group, limited to what the model has");
        assert_eq!(AdapterTargets::ALL.resolve(&Hybrid).unwrap(), ["conv_in", "conv_out", "up", "down", "wq", "wo"], "all = every leaf the model has, each once");
        let named = AdapterTargets::parse("down, conv_in").unwrap();
        assert_eq!(named.resolve(&Hybrid).unwrap(), ["conv_in", "down"], "block order, not the order given");
        let both = AdapterTargets::ATTENTION.union(AdapterTargets::parse("down").unwrap());
        assert_eq!(both.resolve(&Plain).unwrap(), ["down", "wq", "wo"]);
    }

    #[test]
    fn an_unknown_leaf_is_refused_naming_the_leaves_the_model_has() {
        let err = AdapterTargets::parse("wq,lm_head").unwrap().resolve(&Hybrid).unwrap_err();
        assert!(err.contains("lm_head") && err.contains("conv_in") && err.contains("down"), "{err}");
        assert!(AdapterTargets::parse("").is_err() && AdapterTargets::parse("wq,,wo").is_err() && AdapterTargets::parse("w q").is_err());
    }

    #[test]
    fn parse_display_union_and_admits_round_trip() {
        assert_eq!(AdapterTargets::parse("attention"), Ok(AdapterTargets::ATTENTION));
        assert_eq!(AdapterTargets::parse("all"), Ok(AdapterTargets::ALL));
        assert_eq!(AdapterTargets::parse("wq, down").unwrap().to_string(), "wq,down");
        let t = AdapterTargets::ATTENTION.union(AdapterTargets::parse("down").unwrap());
        assert_eq!(t.to_string(), "attention,down");
        assert!(t.admits("wk", &DEFAULT_ATTENTION_LEAVES) && t.admits("down", &DEFAULT_ATTENTION_LEAVES) && !t.admits("up", &DEFAULT_ATTENTION_LEAVES));
        assert!(AdapterTargets::ALL.admits("anything", &[]));
    }

    #[test]
    fn a_bank_config_takes_the_models_defaults_and_budgets_only_targeted_sites() {
        let cfg = AdapterBankConfig::for_model(&Hybrid, &AdapterTargets::parse("up").unwrap()).unwrap();
        assert_eq!((cfg.slots, cfg.r_max, cfg.targets.as_slice()), (2, 16, ["up".to_string()].as_slice()));
        // Two `up` sites of (5 + 8) values per rank row, 2 slots, rank 16, f32,
        // plus each site's per-slot rank word, plus the row vectors.
        let per_site = 2 * 16 * 13 * 4 + 2 * 4;
        assert_eq!(cfg.device_bytes(&Hybrid, 3), 2 * per_site + 3 * 4 + 3 * 16 * 4);
    }
}
