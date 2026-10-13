// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The control plane of multi-adapter serving: which LoRA adapters a served
//! Qwen3 base offers, under which model ids, and how that set reaches the
//! engine that applies them.
//!
//! ## Shape
//!
//! One [`QwenResident`](crate::resident_llm::QwenResident) owns one
//! [`AdapterRegistry`]. The registry is the set of verified adapters
//! ([`LoadedAdapter`]: the factors and the digest of the very bytes they were
//! parsed from) the operator wants served, keyed by serving id. It holds no
//! device state. The engine's adapter bank ([`qwen3::serve::Engine`]) is the
//! data plane: when the resident's instance is built, and again at the start
//! of every batch whose registry changed, [`AdapterSync`] loads what the
//! registry gained into the bank and unloads what it lost. A load is a host
//! registration (the engine uploads to a slot when the scheduler first needs
//! it), an unload of an adapter a sequence still uses completes when that
//! sequence is done, so a swap while requests run interrupts nothing.
//!
//! Each served adapter is also a model: a [`QwenAdapterView`] registered with
//! the executor under the adapter's id. A view builds nothing and costs
//! nothing; it resolves to the base's [`InstanceKey`] (so base and adapter
//! requests batch together on the one resident copy of the weights),
//! delegates placement to the base, and stamps the reserved
//! [`ADAPTER_PARAM`] so the shared instance knows which adapter a request
//! asked for. Selection by model id therefore needs no parsing in any front
//! end: the HTTP and D-Bus surfaces see one more model.
//!
//! ## Lifetime
//!
//! A view holds a lease on its registry entry. The entry is removed when the
//! last lease on it is dropped, and the executor drops a withdrawn view only
//! after its last queued or running job, so a request accepted for an adapter
//! always finds it loaded, and the adapter leaves the bank only after the
//! request is done.
//!
//! ## Sizing
//!
//! The bank's slots, rank cap and targeted linears are fixed when the engine
//! is built ([`AdapterBankSettings`]); its device cost is part of the
//! resident's footprint estimate, so placement budgets it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use capability::{Invocation, Manifest};
use brain_modelref::ModelRef;
use model::serve::AdapterId;
use qwen3::config::QwenConfig;
use qwen3::serve::{AdapterBankConfig, Engine, LoraSite};
use residency::{AdapterIdentity, Device, Executor, Instance, InstanceKey, MemCost, ResidentModel};

use crate::adapter_release::{LoadedAdapter, ServedBase};
use crate::resident_llm::QwenResident;

/// The reserved invocation param a [`QwenAdapterView`] stamps with its
/// serving id. It is the host's word, never a caller's: no manifest declares
/// it, so `ActionSpec::validate_served` refuses it from every HTTP and D-Bus
/// caller, and the base resident strips it ([`ResidentModel::stamp`]).
pub const ADAPTER_PARAM: &str = "__adapter";

/// Adapters the bank holds in device slots at once, unless `--adapter-slots`
/// says otherwise. More adapters than slots are served by swapping the least
/// recently used idle one out, so this bounds device memory, not how many
/// adapters can be offered.
pub const DEFAULT_ADAPTER_SLOTS: u32 = 4;

/// The highest adapter rank a bank slot holds unless `--adapter-rank-cap`
/// says otherwise. An adapter of a higher rank is refused when it is loaded.
pub const DEFAULT_ADAPTER_RANK_CAP: u32 = 16;

/// The linears the bank corrects unless `--adapter-targets` says otherwise:
/// the four attention projections, which is what an adapter trained with the
/// default LoRA configuration targets. An adapter that corrects any other
/// linear is refused when it is loaded, with the flag that widens the bank.
pub const DEFAULT_ADAPTER_TARGETS: AdapterTargets = AdapterTargets::ATTENTION;

/// The most adapters one resident can offer (served ids, resident in a slot or
/// not). The engine's registration table is this size.
pub const MAX_SERVED_ADAPTERS: u32 = 64;

/// A set of [`LoraSite`]s, small enough to be `Copy` so it can ride in
/// [`QwenServeConfig`](crate::resident_llm::QwenServeConfig).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AdapterTargets(u8);

impl AdapterTargets {
    /// The query, key, value and output projections.
    pub const ATTENTION: AdapterTargets = AdapterTargets(0b000_1111);
    /// All seven linears of a block.
    pub const ALL: AdapterTargets = AdapterTargets(0b111_1111);

    /// The set a `--adapter-targets` value names: `attention`, `all`, or a
    /// comma-separated list of linears (`wq,wk,wv,wo,gate,up,down`).
    pub fn parse(list: &str) -> Result<AdapterTargets, String> {
        match list.trim() {
            "attention" => return Ok(AdapterTargets::ATTENTION),
            "all" => return Ok(AdapterTargets::ALL),
            _ => {}
        }
        let mut bits = 0u8;
        for leaf in list.split(',').map(str::trim) {
            let site = LoraSite::from_leaf(leaf).ok_or_else(|| format!("{leaf:?} is not an adapter target; give attention, all, or a list of wq, wk, wv, wo, gate, up, down"))?;
            bits |= Self::bit(site);
        }
        Ok(AdapterTargets(bits))
    }

    /// Every site in either set.
    pub fn union(self, other: AdapterTargets) -> AdapterTargets {
        AdapterTargets(self.0 | other.0)
    }

    fn bit(site: LoraSite) -> u8 {
        1 << LoraSite::ALL.iter().position(|s| *s == site).expect("every site is in ALL")
    }

    /// Does the set contain `site`?
    pub fn contains(self, site: LoraSite) -> bool {
        self.0 & Self::bit(site) != 0
    }

    /// The sites, in block order.
    pub fn sites(self) -> Vec<LoraSite> {
        LoraSite::ALL.into_iter().filter(|s| self.contains(*s)).collect()
    }
}

impl fmt::Display for AdapterTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let leaves: Vec<&str> = self.sites().into_iter().map(LoraSite::leaf).collect();
        write!(f, "{}", leaves.join(","))
    }
}

impl fmt::Debug for AdapterTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdapterTargets({self})")
    }
}

/// How big the adapter bank of one served Qwen3 is. Fixed when the engine is
/// built: the bank's buffers are bound into the recorded decode tapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterBankSettings {
    /// Adapters resident in device slots at once.
    pub slots: u32,
    /// The highest adapter rank a slot holds.
    pub rank_cap: u32,
    /// The linears the bank corrects.
    pub targets: AdapterTargets,
}

impl Default for AdapterBankSettings {
    fn default() -> AdapterBankSettings {
        AdapterBankSettings { slots: DEFAULT_ADAPTER_SLOTS, rank_cap: DEFAULT_ADAPTER_RANK_CAP, targets: DEFAULT_ADAPTER_TARGETS }
    }
}

impl AdapterBankSettings {
    /// Refuse a size the engine cannot build, naming the flag that sets it.
    pub fn validate(&self) -> Result<(), String> {
        if self.slots == 0 || self.slots > MAX_SERVED_ADAPTERS {
            return Err(format!("--adapter-slots must be 1..={MAX_SERVED_ADAPTERS} (got {})", self.slots));
        }
        if self.rank_cap == 0 {
            return Err("--adapter-rank-cap must be at least 1".to_string());
        }
        if self.targets.0 == 0 {
            return Err("--adapter-targets names no linear".to_string());
        }
        Ok(())
    }

    /// The engine's bank configuration for these settings.
    pub fn bank_config(&self) -> AdapterBankConfig {
        let mut cfg = AdapterBankConfig::new(self.slots, self.rank_cap, &self.targets.sites());
        cfg.max_registered = MAX_SERVED_ADAPTERS;
        cfg
    }

    /// Device bytes the bank takes in an engine for `model` whose batches hold
    /// at most `rows` rows: the term a placement budgets on top of the base's.
    pub fn device_bytes(&self, model: &QwenConfig, rows: u32) -> u64 {
        self.bank_config().device_bytes(model, rows)
    }
}

/// One served adapter: the verified file, and how many leases keep it served.
struct Entry {
    loaded: Arc<LoadedAdapter>,
    leases: u32,
}

#[derive(Default)]
struct State {
    entries: BTreeMap<String, Entry>,
}

/// The adapters one served Qwen3 base offers. See this module's doc.
pub struct AdapterRegistry {
    /// `None`: this server was started without an adapter bank, and offers no
    /// adapter.
    settings: Option<AdapterBankSettings>,
    base: ServedBase,
    state: Mutex<State>,
    /// Bumped on every change to the set, so an instance can tell in one
    /// atomic load that nothing changed since it last looked.
    generation: AtomicU64,
}

impl AdapterRegistry {
    /// A registry for a base served as `base`, offering adapters on a bank of
    /// `settings` (or none).
    pub fn new(settings: Option<AdapterBankSettings>, base: ServedBase) -> Arc<AdapterRegistry> {
        Arc::new(AdapterRegistry { settings, base, state: Mutex::new(State::default()), generation: AtomicU64::new(0) })
    }

    /// The bank this registry's adapters run on; `None` when there is none.
    pub fn settings(&self) -> Option<AdapterBankSettings> {
        self.settings
    }

    /// The base adapters are bound to.
    pub fn base(&self) -> &ServedBase {
        &self.base
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Serving ids currently offered, in order.
    pub fn ids(&self) -> Vec<String> {
        self.lock().entries.keys().cloned().collect()
    }

    /// The identity (card id and content digest) offered under `id`.
    pub fn identity(&self, id: &str) -> Option<AdapterIdentity> {
        self.lock().entries.get(id).map(|e| e.loaded.release.identity())
    }

    /// Refuse an adapter the bank cannot hold, before it is offered: the
    /// shapes against the model are the engine's to check, but the rank and
    /// the set of linears are settings an operator can read in a message.
    fn fits(&self, loaded: &LoadedAdapter) -> Result<(), String> {
        let settings = self.settings.ok_or_else(|| "this server serves no adapters: start it with --adapter-slots (or --adapter, --adapter-manifest, --watch-adapters) to give the model an adapter bank".to_string())?;
        let id = &loaded.release.id;
        if loaded.adapter.rank > settings.rank_cap {
            return Err(format!("adapter {id} has rank {}, above this server's --adapter-rank-cap {}", loaded.adapter.rank, settings.rank_cap));
        }
        for site in &loaded.adapter.sites {
            let leaf = site.base.rsplit('.').nth(1).unwrap_or_default();
            match LoraSite::from_leaf(leaf) {
                Some(s) if settings.targets.contains(s) => {}
                Some(_) => return Err(format!("adapter {id} corrects {leaf}, which this server's --adapter-targets ({}) does not; widen the flag to serve it", settings.targets)),
                None => return Err(format!("adapter {id} holds a pair for {}, which is not a linear of a Qwen3 block", site.base)),
            }
        }
        Ok(())
    }

    /// Offer `loaded` under its card id and take a lease on it. Offering an id
    /// that is already offered replaces the adapter under it (a new release of
    /// the same name) and adds a lease. The id must be a model reference naming
    /// an adapter (`<vendor>/<repo>[-QUANT]:<owner>:<name>:<tag>`): it becomes a
    /// model id that every front end resolves, and the grammar is what keeps it
    /// from being anything else.
    pub fn attach(self: &Arc<Self>, loaded: LoadedAdapter) -> Result<AdapterLease, String> {
        let id = loaded.release.id.clone();
        check_serving_id(&id)?;
        self.fits(&loaded)?;
        let mut state = self.lock();
        if !state.entries.contains_key(&id) && state.entries.len() as u32 >= MAX_SERVED_ADAPTERS {
            return Err(format!("this server already offers {MAX_SERVED_ADAPTERS} adapters, the most it holds; retire one before adding {id}"));
        }
        match state.entries.get_mut(&id) {
            Some(entry) => {
                entry.loaded = Arc::new(loaded);
                entry.leases += 1;
            }
            None => {
                state.entries.insert(id.clone(), Entry { loaded: Arc::new(loaded), leases: 1 });
            }
        }
        drop(state);
        self.generation.fetch_add(1, Ordering::SeqCst);
        Ok(AdapterLease { registry: self.clone(), id })
    }

    /// Replace the adapter offered under `id`, which must be offered (a lease
    /// is held on it): the next release of a name already served.
    pub fn replace(&self, loaded: LoadedAdapter) -> Result<(), String> {
        let id = loaded.release.id.clone();
        self.fits(&loaded)?;
        let mut state = self.lock();
        let entry = state.entries.get_mut(&id).ok_or_else(|| format!("adapter {id} is not offered"))?;
        entry.loaded = Arc::new(loaded);
        drop(state);
        self.generation.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn release(&self, id: &str) {
        let mut state = self.lock();
        let Some(entry) = state.entries.get_mut(id) else { return };
        entry.leases -= 1;
        if entry.leases == 0 {
            state.entries.remove(id);
            drop(state);
            self.generation.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn snapshot(&self) -> (u64, Vec<(String, Arc<LoadedAdapter>)>) {
        let state = self.lock();
        // Read under the lock: a change bumps the generation after it
        // releases the lock, so a snapshot is at least as new as its number.
        let generation = self.generation.load(Ordering::SeqCst);
        (generation, state.entries.iter().map(|(id, e)| (id.clone(), e.loaded.clone())).collect())
    }
}

/// A serving id must name an adapter in the model reference grammar.
fn check_serving_id(id: &str) -> Result<(), String> {
    match ModelRef::parse(id) {
        Ok(r) if r.adapter().is_some() => Ok(()),
        Ok(_) => Err(format!("adapter id {id:?} names a model, not an adapter: an adapter card is named <vendor>/<repo>:<owner>:<name>:<tag>, as the trainer records it")),
        Err(e) => Err(format!("adapter id {id:?} is not a model reference ({e}): an adapter card is named <vendor>/<repo>:<owner>:<name>:<tag>, as the trainer records it")),
    }
}

/// A hold on one entry of an [`AdapterRegistry`]; the entry is removed when
/// the last hold is dropped.
pub struct AdapterLease {
    registry: Arc<AdapterRegistry>,
    id: String,
}

impl Drop for AdapterLease {
    fn drop(&mut self) {
        self.registry.release(&self.id);
    }
}

/// The model id of one served adapter: a thin resident over the base's.
///
/// Everything it does is the base's, except being listed under its own id,
/// stamping which adapter a request named, and reporting the adapter's
/// identity. See this module's doc.
pub struct QwenAdapterView {
    id: String,
    base: Arc<QwenResident>,
    _lease: AdapterLease,
}

impl QwenAdapterView {
    /// The view of the adapter `lease` holds, over `base`.
    pub(crate) fn new(base: Arc<QwenResident>, lease: AdapterLease) -> QwenAdapterView {
        QwenAdapterView { id: lease.id.clone(), base, _lease: lease }
    }

    /// The model id this view is listed under.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl ResidentModel for QwenAdapterView {
    fn manifest(&self) -> Manifest {
        let mut manifest = self.base.manifest();
        manifest.model = self.id.clone();
        manifest
    }
    /// The base's key, so base and adapter requests are one group and one
    /// instance.
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.base.instance_key(action, inv)
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.base.estimate(key)
    }
    fn estimate_at(&self, key: &InstanceKey, tier: residency::Tier) -> MemCost {
        self.base.estimate_at(key, tier)
    }
    /// Builds the BASE's instance (the executor may claim the view first).
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        self.base.activate(key, device)
    }
    fn stamp(&self, inv: &mut Invocation) {
        if let Some(params) = inv.params.as_object_mut() {
            params.insert(ADAPTER_PARAM.to_string(), serde_json::json!(self.id));
        }
    }
    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.base.adapters().identity(&self.id)
    }
}

/// Puts adapters in front of, and takes them out of, a RUNNING executor: the
/// one place an adapter becomes (or stops being) a model of a live server.
///
/// An adapter is offered by registering its [`QwenAdapterView`], withdrawn by
/// unregistering it; nothing is rebuilt and no instance is evicted, so a
/// request that is running or queued is never disturbed.
pub struct LiveAdapters {
    resident: Arc<QwenResident>,
    executor: Executor,
}

impl LiveAdapters {
    /// Offer and withdraw adapters of `resident` on `executor`, which must be
    /// the executor `resident` is registered with.
    pub fn new(resident: Arc<QwenResident>, executor: Executor) -> LiveAdapters {
        LiveAdapters { resident, executor }
    }

    /// Offer `adapter` under its card id as a model of its own. A card id that
    /// is already offered gets the new release in place of the old (the same
    /// model, a new digest).
    ///
    /// # Errors
    ///
    /// The id names a model this executor serves that is not one of this
    /// resident's adapters (an adapter can never shadow a model), is not an
    /// adapter reference, or the bank cannot hold the adapter
    /// ([`AdapterRegistry::attach`]).
    pub fn serve(&self, adapter: LoadedAdapter) -> Result<(), String> {
        let id = adapter.release.id.clone();
        let listed = self.executor.manifests().iter().any(|m| m.model == id);
        if !listed {
            self.executor.register(Arc::new(self.resident.adapter_view(adapter)?));
            return Ok(());
        }
        if !self.resident.adapters().ids().contains(&id) {
            return Err(format!("adapter id {id} is already a served model; an adapter does not replace a model"));
        }
        let identity = adapter.release.identity();
        self.resident.adapters().replace(adapter)?;
        self.executor.set_served_adapter(&id, Some(identity));
        Ok(())
    }

    /// Withdraw the adapter offered under `id`: it leaves the model listing at
    /// once and takes no new request, requests already accepted for it finish,
    /// and the engine unloads it when the next batch starts. An id that is not
    /// one of this resident's adapters is left alone: this cannot withdraw a
    /// model.
    pub fn retire(&self, id: &str) {
        if self.resident.adapters().ids().iter().any(|offered| offered == id) {
            self.executor.unregister(id);
        }
    }
}

/// Where a request's adapter name leads in one instance.
enum Route {
    Ready(AdapterId),
    /// The engine refused the adapter; the reason was logged when it did.
    Refused,
}

/// An instance's view of its registry: what the engine's bank holds, and the
/// route from a serving id to it. See this module's doc.
pub(crate) struct AdapterSync {
    registry: Arc<AdapterRegistry>,
    seen: Option<u64>,
    /// Engine id of every adapter the bank holds, by digest: the engine's
    /// identity for an adapter is its content.
    loaded: HashMap<String, AdapterId>,
    routes: HashMap<String, Route>,
}

impl AdapterSync {
    pub(crate) fn new(registry: Arc<AdapterRegistry>) -> AdapterSync {
        AdapterSync { registry, seen: None, loaded: HashMap::new(), routes: HashMap::new() }
    }

    /// Bring `engine`'s bank in line with the registry: register what it
    /// gained, unload what it lost. One atomic load when nothing changed.
    pub(crate) fn sync(&mut self, engine: &mut Engine) {
        if self.seen == Some(self.registry.generation.load(Ordering::SeqCst)) {
            return;
        }
        let (generation, offered) = self.registry.snapshot();
        let wanted: HashSet<&str> = offered.iter().map(|(_, a)| a.release.digest.as_str()).collect();
        self.routes.clear();
        for (id, adapter) in &offered {
            let digest = &adapter.release.digest;
            if let Some(&engine_id) = self.loaded.get(digest) {
                self.routes.insert(id.clone(), Route::Ready(engine_id));
                continue;
            }
            match engine.load_adapter(adapter.as_ref().clone().into_engine()) {
                Ok(engine_id) => {
                    self.loaded.insert(digest.clone(), engine_id);
                    self.routes.insert(id.clone(), Route::Ready(engine_id));
                }
                Err(why) => {
                    eprintln!("serve: adapter {id} ({digest}) cannot be loaded into the bank, so requests for it fail: {why}");
                    self.routes.insert(id.clone(), Route::Refused);
                }
            }
        }
        let gone: Vec<String> = self.loaded.keys().filter(|d| !wanted.contains(d.as_str())).cloned().collect();
        for digest in gone {
            let engine_id = self.loaded.remove(&digest).expect("listed from the map");
            // `Draining` finishes by itself when the last sequence under the
            // adapter is done; an unknown id was already gone.
            if let Err(why) = engine.unload_adapter(engine_id) {
                if !matches!(why, qwen3::serve::AdapterError::Unknown(_)) {
                    eprintln!("serve: unloading adapter {digest} from the bank: {why}");
                }
            }
        }
        self.seen = Some(generation);
    }

    /// The engine adapter a request for serving id `id` runs under.
    ///
    /// The error is what the caller is told: the same "no model" an unknown
    /// id gets, because an adapter that was withdrawn, or that the engine
    /// refused, is not something the caller can act on and its reason (logged
    /// where it happened) is the operator's.
    pub(crate) fn route(&self, id: &str) -> Result<AdapterId, String> {
        match self.routes.get(id) {
            Some(Route::Ready(engine_id)) => Ok(*engine_id),
            Some(Route::Refused) | None => Err(format!("no model '{id}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_release::load_adapter;
    use brain_testutil::adapters::{scratch_dir, write_adapter};

    fn registry(settings: Option<AdapterBankSettings>) -> Arc<AdapterRegistry> {
        AdapterRegistry::new(settings, ServedBase::new("unused.safetensors", Some("Qwen/Qwen3-0.6B".to_string())))
    }

    fn loaded(dir: &std::path::Path, id: &str, fill: f32) -> LoadedAdapter {
        let path = dir.join(format!("{}-{fill}.safetensors", id.replace([':', '/'], "_")));
        write_adapter(&path, id, "Qwen/Qwen3-0.6B", None, fill);
        load_adapter(&path).expect("fixture adapter loads")
    }

    const ID: &str = "Qwen/Qwen3-0.6B:lab:canary:v1";

    #[test]
    fn targets_parse_the_two_names_and_a_list_and_refuse_anything_else() {
        assert_eq!(AdapterTargets::parse("attention"), Ok(AdapterTargets::ATTENTION));
        assert_eq!(AdapterTargets::parse("all"), Ok(AdapterTargets::ALL));
        assert_eq!(AdapterTargets::parse("wq, down").unwrap().sites(), vec![LoraSite::Wq, LoraSite::Down]);
        assert_eq!(AdapterTargets::ATTENTION.to_string(), "wq,wk,wv,wo");
        assert!(AdapterTargets::parse("wq,lm_head").unwrap_err().contains("lm_head"));
        assert!(AdapterTargets::parse("").is_err(), "an empty list names no linear");
        assert_eq!(DEFAULT_ADAPTER_TARGETS, AdapterTargets::ATTENTION);
    }

    #[test]
    fn settings_name_the_flag_that_is_wrong() {
        assert_eq!(AdapterBankSettings::default().validate(), Ok(()));
        for (bad, flag) in [
            (AdapterBankSettings { slots: 0, ..Default::default() }, "--adapter-slots"),
            (AdapterBankSettings { slots: MAX_SERVED_ADAPTERS + 1, ..Default::default() }, "--adapter-slots"),
            (AdapterBankSettings { rank_cap: 0, ..Default::default() }, "--adapter-rank-cap"),
            (AdapterBankSettings { targets: AdapterTargets(0), ..Default::default() }, "--adapter-targets"),
        ] {
            assert!(bad.validate().unwrap_err().contains(flag), "{bad:?}");
        }
        let cfg = AdapterBankSettings { slots: 3, rank_cap: 8, targets: AdapterTargets::ALL }.bank_config();
        assert_eq!((cfg.slots, cfg.r_max, cfg.targets.len(), cfg.max_registered), (3, 8, 7, MAX_SERVED_ADAPTERS));
    }

    /// An adapter is offered only on a server that has a bank, only when the
    /// bank can hold it, and only under an id that is an adapter reference.
    #[test]
    fn the_registry_refuses_what_the_bank_cannot_hold_with_the_flag_to_change() {
        let dir = scratch_dir("registry-fits");
        let adapter = loaded(&dir, ID, 0.5);

        let none = registry(None);
        assert!(none.attach(adapter.clone()).err().unwrap().contains("--adapter-slots"), "no bank, no adapters");

        let small = registry(Some(AdapterBankSettings { rank_cap: 1, ..Default::default() }));
        assert!(small.attach(adapter.clone()).err().unwrap().contains("--adapter-rank-cap"), "the fixture is rank 2");

        let ok = registry(Some(AdapterBankSettings::default()));
        let wrong_sites = registry(Some(AdapterBankSettings { targets: AdapterTargets::parse("down").unwrap(), ..Default::default() }));
        assert!(wrong_sites.attach(adapter.clone()).err().unwrap().contains("--adapter-targets"), "the fixture corrects wq");

        for bad in ["pinned", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B:lab:../x:v1", "../etc:lab:x:v1"] {
            let mut a = adapter.clone();
            a.release.id = bad.to_string();
            assert!(ok.attach(a).is_err(), "{bad:?} is not an adapter reference");
        }
        assert!(ok.attach(adapter).is_ok());
    }

    /// An entry lives while any lease on it does; the generation moves on
    /// every change; a second release of the same name replaces the first.
    #[test]
    fn an_entry_lives_as_long_as_its_last_lease_and_a_release_replaces_it() {
        let dir = scratch_dir("registry-lease");
        let reg = registry(Some(AdapterBankSettings::default()));
        let g0 = reg.generation.load(Ordering::SeqCst);

        let first = reg.attach(loaded(&dir, ID, 0.5)).unwrap();
        let g1 = reg.generation.load(Ordering::SeqCst);
        assert!(g1 > g0);
        let old_digest = reg.identity(ID).unwrap().digest;

        // The next release of the same name, offered while the first is still
        // leased (a retiring view not yet dropped).
        let second = reg.attach(loaded(&dir, ID, 0.25)).unwrap();
        assert_ne!(reg.identity(ID).unwrap().digest, old_digest, "the later release is the one offered");
        assert_eq!(reg.ids(), vec![ID.to_string()]);

        drop(first);
        assert!(reg.identity(ID).is_some(), "one lease is still held");
        let g2 = reg.generation.load(Ordering::SeqCst);
        drop(second);
        assert_eq!(reg.identity(ID), None, "the last lease removes the entry");
        assert!(reg.generation.load(Ordering::SeqCst) > g2);
        assert!(reg.replace(loaded(&dir, ID, 0.1)).is_err(), "a name not offered cannot be replaced");
    }

    /// The offering is bounded.
    #[test]
    fn the_registry_offers_at_most_the_engines_table() {
        let dir = scratch_dir("registry-full");
        let reg = registry(Some(AdapterBankSettings::default()));
        let leases: Vec<_> = (0..MAX_SERVED_ADAPTERS).map(|i| reg.attach(loaded(&dir, &format!("Qwen/Qwen3-0.6B:lab:a{i}:v1"), 0.5)).unwrap()).collect();
        assert!(reg.attach(loaded(&dir, "Qwen/Qwen3-0.6B:lab:one-more:v1", 0.5)).err().unwrap().contains("retire one"));
        assert!(reg.attach(loaded(&dir, "Qwen/Qwen3-0.6B:lab:a0:v1", 0.25)).is_ok(), "a new release of an offered name is not a new adapter");
        drop(leases);
    }
}
