// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The control plane of multi-adapter serving: which LoRA adapters a served
//! base offers, under which model ids, and how that set reaches the engine
//! that applies them.
//!
//! ## Shape
//!
//! One served base (an [`AdapterHost`]: the Qwen3 resident, the released
//! Qwen3.8 GGUF, Ripple) owns one [`AdapterRegistry`]. The registry is the
//! set of verified adapters
//! ([`LoadedAdapter`]: the factors and the digest of the very bytes they were
//! parsed from) the operator wants served, keyed by serving id. It holds no
//! device state. The engine's adapter bank ([`model::adapter::bank::AdapterBank`]) is the
//! data plane: when the resident's instance is built, and again at the start
//! of every batch whose registry changed, [`AdapterSync`] loads what the
//! registry gained into the bank and unloads what it lost. A load is a host
//! registration (the engine uploads to a slot when the scheduler first needs
//! it), an unload of an adapter a sequence still uses completes when that
//! sequence is done, so a swap while requests run interrupts nothing.
//!
//! Each served adapter is also a model: an [`AdapterView`] registered with
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
//! ## A base hosted in a worker
//!
//! A base whose instance runs in a supervised worker keeps this registry in
//! the serving process, which offers it to the worker as an
//! [`AdapterSource`]: every release with its file, digest and version. The
//! worker rebuilds its resident's own registry from that list through an
//! [`AdapterMirror`], which reads each file, refuses bytes that are not the
//! offered digest, and keeps the serving process's version, so a reply names
//! a release as it would in process.
//!
//! ## Sizing
//!
//! The bank's slots, rank cap and targeted linears are fixed when the engine
//! is built ([`AdapterBankSettings`]); its device cost is part of the
//! resident's footprint estimate, so placement budgets it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use capability::{ActionResult, Invocation, Manifest, Progress};
use crate::{run_batch_scheduled, run_open_scheduled, scheduler_metrics, Frontend, TextFrontend};
use model::serve::{PagedDecoder, Scheduler};
use brain_modelref::ModelRef;
use model::serve::AdapterId;
use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterServing, BankedEngine};
pub use model::adapter::bank::AdapterTargets;

use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::supervised::{AdapterSink, AdapterSource, HostedAdapter};
use residency::{AdapterIdentity, Device, Executor, Instance, InstanceKey, MemCost, ResidentModel};

use crate::adapter_release::{LoadedAdapter, ServedBase, ServedFamily};

/// The reserved invocation param an [`AdapterView`] stamps with its
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
/// The bank's buffers are sized by the cap when the engine is built, so an
/// adapter hot-loaded into a running server can never exceed it: the default
/// is the rank a learning loop usually trains (32), not the smallest one.
pub const DEFAULT_ADAPTER_RANK_CAP: u32 = 32;

/// The linears the bank corrects unless `--adapter-targets` says otherwise:
/// every adapter site of the model, which is what brain's trainers adapt by
/// default (all seven projections of a Qwen3 block). The bank's buffers are
/// fixed when the engine is built, so a narrower default would refuse an
/// adapter trained after the server started (hot-loaded through
/// `--serve-adapters`) until a restart; an operator who wants the smaller bank
/// names its targets.
pub const DEFAULT_ADAPTER_TARGETS: AdapterTargets = AdapterTargets::ALL;

/// The most adapters one resident can offer (served ids, resident in a slot or
/// not). The engine's registration table is this size.
pub const MAX_SERVED_ADAPTERS: u32 = 64;

/// How big the adapter bank of one served base is. Fixed when the engine is
/// built: the bank's buffers are bound into the recorded decode tapes.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        Ok(())
    }

    /// These settings as a request a resident can place before it knows its
    /// model's sites.
    pub fn request(&self) -> model::adapter::bank::AdapterBankRequest {
        model::adapter::bank::AdapterBankRequest { slots: self.slots, rank_cap: self.rank_cap, targets: self.targets.clone(), max_registered: MAX_SERVED_ADAPTERS }
    }

    /// The engine's bank configuration for these settings on `model`.
    ///
    /// # Errors
    ///
    /// A target the model has no linear for ([`AdapterTargets::resolve`]).
    pub fn bank_config<M: AdapterServing + ?Sized>(&self, model: &M) -> Result<AdapterBankConfig, String> {
        let mut cfg = AdapterBankConfig::new(self.slots, self.rank_cap, &self.targets.resolve(model)?);
        cfg.max_registered = MAX_SERVED_ADAPTERS;
        cfg.merged = model.merged_sites();
        Ok(cfg)
    }

    /// Device bytes the bank takes in an engine for `model` whose batches hold
    /// at most `rows` rows: the term a placement budgets on top of the base's.
    /// Targets the model cannot resolve budget nothing; building the engine
    /// refuses them with the reason.
    pub fn device_bytes<M: AdapterServing + ?Sized>(&self, model: &M, rows: u32) -> u64 {
        self.bank_config(model).map_or(0, |cfg| cfg.device_bytes(model, rows))
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
    /// The last release offered under each id this process has served, with
    /// its version, kept after the id is withdrawn: an id offered again
    /// continues its count, and offering the same weights again keeps their
    /// number. One small record per distinct id; the ids are the operator's.
    releases: HashMap<String, (String, u64)>,
}

impl State {
    /// Stamp `loaded` with its version under its id: the next number when its
    /// weights differ from the last release of the id, that release's number
    /// when they do not.
    fn stamp(&mut self, loaded: &mut LoadedAdapter) {
        let id = &loaded.release.id;
        let digest = &loaded.release.digest;
        let version = match self.releases.get(id) {
            Some((last, version)) if last == digest => *version,
            Some((_, version)) => version + 1,
            None => 1,
        };
        loaded.version = version;
        self.releases.insert(id.clone(), (digest.clone(), version));
    }
}

/// The adapters one served base offers. See this module's doc.
pub struct AdapterRegistry {
    /// `None`: this server was started without an adapter bank, and offers no
    /// adapter.
    settings: Option<AdapterBankSettings>,
    base: ServedBase,
    family: ServedFamily,
    state: Mutex<State>,
    /// Bumped on every change to the set, so an instance can tell in one
    /// atomic load that nothing changed since it last looked.
    generation: AtomicU64,
}

impl AdapterRegistry {
    /// A registry for a base served as `base`, offering adapters on a bank of
    /// `settings` (or none).
    pub fn new(settings: Option<AdapterBankSettings>, base: ServedBase) -> Arc<AdapterRegistry> {
        let family = base.family().clone();
        base.warm_digest();
        Arc::new(AdapterRegistry { settings, base, family, state: Mutex::new(State::default()), generation: AtomicU64::new(0) })
    }

    /// The bank this registry's adapters run on; `None` when there is none.
    pub fn settings(&self) -> Option<AdapterBankSettings> {
        self.settings.clone()
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
        self.lock().entries.get(id).map(|e| e.loaded.identity())
    }

    /// Refuse an adapter the bank cannot hold, before it is offered: the
    /// shapes against the model are the engine's to check, but the rank and
    /// the set of linears are settings an operator can read in a message.
    fn fits(&self, loaded: &LoadedAdapter) -> Result<(), String> {
        let settings = self.settings.as_ref().ok_or_else(|| "this server serves no adapters: start it with --adapter-slots (or --adapter, --adapter-manifest, --watch-adapters) to give the model an adapter bank".to_string())?;
        let id = &loaded.release.id;
        if loaded.family != self.family.tag {
            return Err(format!("adapter {id} is a {} adapter; the base it would ride on takes {} adapters", loaded.family, self.family.tag));
        }
        if loaded.adapter.rank > settings.rank_cap {
            return Err(format!(
                "adapter {id} has rank {}, above this server's --adapter-rank-cap {}; the cap sizes the bank's device buffers when the server starts, so restart with --adapter-rank-cap {} (or more) to serve it",
                loaded.adapter.rank, settings.rank_cap, loaded.adapter.rank
            ));
        }
        let mut missing: Vec<&str> = Vec::new();
        let a = &loaded.adapter;
        for base in a.sites.iter().map(|s| s.base.as_str()).chain(a.experts.iter().map(|p| p.base.as_str())) {
            let leaf = a.leaf(base);
            if !self.family.all.contains(&leaf) {
                return Err(format!("adapter {id} holds a pair for {base}, which is not an adapter site of a {} model (it has {})", self.family.tag, self.family.all.join(", ")));
            }
            if !settings.targets.admits(leaf, &self.family.attention) && !missing.contains(&leaf) {
                missing.push(leaf);
            }
        }
        if !missing.is_empty() {
            // Name every linear the bank lacks and the flag value that would
            // serve this adapter, so one restart fixes it.
            let widened = AdapterTargets::parse(&missing.join(",")).map(|m| settings.targets.clone().union(m))?;
            return Err(format!(
                "adapter {id} corrects {}, which this server's --adapter-targets ({}) does not; restart with --adapter-targets {widened} (or all) to serve it",
                missing.join(","),
                settings.targets
            ));
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
        self.offer(loaded, true)
    }

    /// [`Self::attach`] for a registry that mirrors another one (a worker's, rebuilt from what its serving process
    /// offers): `loaded` keeps the version the registry it mirrors numbered it, which replies name.
    fn attach_numbered(self: &Arc<Self>, loaded: LoadedAdapter) -> Result<AdapterLease, String> {
        self.offer(loaded, false)
    }

    /// Offer `loaded` and take a lease on it, numbering its release here when `number` (else it keeps its own).
    fn offer(self: &Arc<Self>, mut loaded: LoadedAdapter, number: bool) -> Result<AdapterLease, String> {
        let id = loaded.release.id.clone();
        check_serving_id(&id)?;
        self.fits(&loaded)?;
        let mut state = self.lock();
        if !state.entries.contains_key(&id) && state.entries.len() as u32 >= MAX_SERVED_ADAPTERS {
            return Err(format!("this server already offers {MAX_SERVED_ADAPTERS} adapters, the most it holds; retire one before adding {id}"));
        }
        if number {
            state.stamp(&mut loaded);
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
    /// is held on it): the next release of a name already served. Returns the
    /// identity now offered, with its version.
    pub fn replace(&self, mut loaded: LoadedAdapter) -> Result<AdapterIdentity, String> {
        let id = loaded.release.id.clone();
        self.fits(&loaded)?;
        let mut state = self.lock();
        if !state.entries.contains_key(&id) {
            return Err(format!("adapter {id} is not offered"));
        }
        state.stamp(&mut loaded);
        let identity = loaded.identity();
        state.entries.get_mut(&id).expect("checked present").loaded = Arc::new(loaded);
        drop(state);
        self.generation.fetch_add(1, Ordering::SeqCst);
        Ok(identity)
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

/// The serving process's side of a hosted base's adapters: what the registry offers, each release with the file a
/// worker reads it from and the version it is numbered here ([`residency::supervised`]).
impl AdapterSource for AdapterRegistry {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn offered(&self) -> (u64, Vec<HostedAdapter>) {
        let (generation, offered) = self.snapshot();
        let adapters = offered.into_iter().map(|(id, a)| HostedAdapter { id, digest: a.release.digest.clone(), path: a.release.path.clone(), version: a.version }).collect();
        (generation, adapters)
    }
}

/// A worker's side of a hosted base's adapters: the hosted resident's own registry, rebuilt from what the serving
/// process offers. Each attached id holds one lease, so the registry offers exactly what the worker was told, and the
/// resident's instance brings its engine's bank in line from it as it does in process ([`AdapterSync`]).
pub struct AdapterMirror {
    registry: Arc<AdapterRegistry>,
    leases: Mutex<HashMap<String, AdapterLease>>,
}

impl AdapterMirror {
    /// Mirror into `registry`, the registry of the resident the worker hosts.
    pub fn new(registry: Arc<AdapterRegistry>) -> AdapterMirror {
        AdapterMirror { registry, leases: Mutex::new(HashMap::new()) }
    }
}

impl AdapterSink for AdapterMirror {
    /// Read the release's file, refuse it unless its bytes are the offered digest and its card the offered id, and
    /// offer it under the serving process's version in place of what the id offered before.
    fn attach(&self, adapter: &HostedAdapter) -> Result<(), String> {
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        // Whatever happens below, the id no longer offers what it offered: a refused release is unoffered.
        let previous = leases.remove(&adapter.id);
        let mut loaded = crate::adapter_release::load_adapter(&adapter.path)?;
        if loaded.release.digest != adapter.digest {
            return Err(format!("{} holds {}, not the offered release {}", adapter.path.display(), loaded.release.digest, adapter.digest));
        }
        if loaded.release.id != adapter.id {
            return Err(format!("{} is adapter {}, not the offered {}", adapter.path.display(), loaded.release.id, adapter.id));
        }
        loaded.version = adapter.version;
        let lease = self.registry.attach_numbered(loaded)?;
        leases.insert(adapter.id.clone(), lease);
        drop(previous);
        Ok(())
    }

    fn detach(&self, id: &str) {
        let lease = self.leases.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
        drop(lease);
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

/// A resident that carries an [`AdapterRegistry`] and can have adapters offered
/// as models of their own beside it: the Qwen3 resident, the released Qwen3.8
/// GGUF and Ripple each implement it.
pub trait AdapterHost: ResidentModel + Send + Sync + Sized + 'static {
    /// The adapters this base offers.
    fn adapters(&self) -> &Arc<AdapterRegistry>;

    /// Offer `adapter` as a model of its own on `executor`, which must be the
    /// executor `self` is registered with: leases the adapter in the
    /// registry and registers the view the way this resident's kind is
    /// registered (single-device or multi-device).
    fn register_view(self: &Arc<Self>, executor: &Executor, adapter: LoadedAdapter) -> Result<(), String>;

    /// The base adapters are bound to, for [`crate::adapter_release::verify_adapter`].
    fn served_base(&self) -> ServedBase {
        self.adapters().base().clone()
    }

    /// The view of `adapter` over this resident, without registering it.
    fn adapter_view(self: &Arc<Self>, adapter: LoadedAdapter) -> Result<AdapterView<Self>, String> {
        let lease = self.adapters().attach(adapter)?;
        Ok(AdapterView::new(self.clone(), lease))
    }
}

/// The model id of one served adapter: a thin resident over the base's.
///
/// Everything it does is the base's, except being listed under its own id,
/// stamping which adapter a request named, and reporting the adapter's
/// identity. See this module's doc.
pub struct AdapterView<R: AdapterHost> {
    id: String,
    base: Arc<R>,
    _lease: AdapterLease,
}

impl<R: AdapterHost> AdapterView<R> {
    /// The view of the adapter `lease` holds, over `base`.
    pub fn new(base: Arc<R>, lease: AdapterLease) -> AdapterView<R> {
        AdapterView { id: lease.id.clone(), base, _lease: lease }
    }

    /// The model id this view is listed under.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl<R: AdapterHost> ResidentModel for AdapterView<R> {
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
    fn batch_limit(&self) -> Option<usize> {
        self.base.batch_limit()
    }
    /// The base's replicas: an adapter is served by every instance of its base, each with the adapter in its own bank.
    fn replicas(&self) -> usize {
        self.base.replicas()
    }
    fn replica_routing(&self) -> Option<residency::affinity::ReplicaRouting> {
        self.base.replica_routing()
    }
    fn prefix_chain(&self, inv: &Invocation) -> residency::affinity::PrefixChain {
        self.base.prefix_chain(inv)
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
    /// The base's plan, which the view shares.
    fn replan(&self, key: &InstanceKey, usable: &[(Device, u64)]) -> Option<String> {
        self.base.replan(key, usable)
    }
}

impl<R: AdapterHost + MultiDeviceResidentModel> MultiDeviceResidentModel for AdapterView<R> {
    fn unavailable_reason(&self, key: &InstanceKey) -> Option<String> {
        self.base.unavailable_reason(key)
    }
    fn estimate_multi(&self, key: &InstanceKey) -> MultiDeviceCost {
        self.base.estimate_multi(key)
    }
    /// Builds the BASE's instance, which every view of it shares.
    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        self.base.activate_multi(key, devices)
    }
}

/// Puts adapters in front of, and takes them out of, a RUNNING executor: the
/// one place an adapter becomes (or stops being) a model of a live server.
///
/// An adapter is offered by registering its [`AdapterView`], withdrawn by
/// unregistering it; nothing is rebuilt and no instance is evicted, so a
/// request that is running or queued is never disturbed.
pub struct LiveAdapters {
    target: Box<dyn AdapterTarget>,
    executor: Executor,
}

/// An [`AdapterHost`] with its type erased, so a watcher and a CLI can drive
/// whichever resident the server put adapters on.
trait AdapterTarget: Send + Sync {
    fn registry(&self) -> &Arc<AdapterRegistry>;
    fn offer(&self, executor: &Executor, adapter: LoadedAdapter) -> Result<(), String>;
}

impl<R: AdapterHost> AdapterTarget for Arc<R> {
    fn registry(&self) -> &Arc<AdapterRegistry> {
        self.adapters()
    }
    fn offer(&self, executor: &Executor, adapter: LoadedAdapter) -> Result<(), String> {
        self.register_view(executor, adapter)
    }
}

impl LiveAdapters {
    /// Offer and withdraw adapters of `resident` on `executor`, which must be
    /// the executor `resident` is registered with.
    pub fn new<R: AdapterHost>(resident: Arc<R>, executor: Executor) -> LiveAdapters {
        LiveAdapters { target: Box::new(resident), executor }
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
            return self.target.offer(&self.executor, adapter);
        }
        if !self.target.registry().ids().contains(&id) {
            return Err(format!("adapter id {id} is already a served model; an adapter does not replace a model"));
        }
        let identity = self.target.registry().replace(adapter)?;
        self.executor.set_served_adapter(&id, Some(identity));
        Ok(())
    }

    /// The release offered under `id` (its digest and policy version), when this resident offers one.
    pub fn identity(&self, id: &str) -> Option<AdapterIdentity> {
        self.target.registry().identity(id)
    }

    /// Withdraw the adapter offered under `id`: it leaves the model listing at
    /// once and takes no new request, requests already accepted for it finish,
    /// and the engine unloads it when the next batch starts. An id that is not
    /// one of this resident's adapters is left alone: this cannot withdraw a
    /// model.
    pub fn retire(&self, id: &str) {
        if self.target.registry().ids().iter().any(|offered| offered == id) {
            self.executor.unregister(id);
        }
    }
}

/// Where a request's adapter name leads in one instance.
enum Route {
    /// The engine's id for the adapter and the identity of the release it was
    /// loaded from, kept together so a reply names the weights that decoded it
    /// even if the registry has swapped the id since.
    Ready(AdapterId, AdapterIdentity),
    /// The engine refused the adapter; the reason was logged when it did.
    Refused,
}

/// An instance's view of its registry: what the engine's bank holds, and the
/// route from a serving id to it. See this module's doc.
///
/// [`Self::sync`] takes `&self`: a request front end reads the routes while an
/// open group runs, and the group refreshes them between steps through the same
/// shared view ([`crate::run_open_scheduled`]).
pub struct AdapterSync {
    registry: Arc<AdapterRegistry>,
    state: Mutex<SyncState>,
}

/// What an [`AdapterSync`] has made of the registry so far.
struct SyncState {
    seen: Option<u64>,
    /// Engine id of every adapter the bank holds, by digest: the engine's
    /// identity for an adapter is its content.
    loaded: HashMap<String, AdapterId>,
    routes: HashMap<String, Route>,
}

impl AdapterSync {
    /// The served base's digest, once it has been hashed.
    pub fn base_digest(&self) -> Option<String> {
        self.registry.base().peek_digest()
    }

    pub fn new(registry: Arc<AdapterRegistry>) -> AdapterSync {
        AdapterSync { registry, state: Mutex::new(SyncState { seen: None, loaded: HashMap::new(), routes: HashMap::new() }) }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, SyncState> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether the registry has changed since the last [`Self::sync`]: the
    /// routes may name a release that has been replaced.
    pub fn stale(&self) -> bool {
        self.state().seen != Some(self.registry.generation.load(Ordering::SeqCst))
    }

    /// Bring `engine`'s bank in line with the registry: register what it
    /// gained, unload what it lost. Cheap when nothing changed. Safe while
    /// sequences run: a release that is replaced drains (its running sequences
    /// finish on the weights they began on, the engine unloads it after the last)
    /// and a new one is registered beside it.
    pub fn sync<E: BankedEngine>(&self, engine: &mut E) {
        let mut state = self.state();
        if state.seen == Some(self.registry.generation.load(Ordering::SeqCst)) {
            return;
        }
        let (generation, offered) = self.registry.snapshot();
        let wanted: HashSet<&str> = offered.iter().map(|(_, a)| a.release.digest.as_str()).collect();
        state.routes.clear();
        for (id, adapter) in &offered {
            let digest = &adapter.release.digest;
            if let Some(&engine_id) = state.loaded.get(digest) {
                state.routes.insert(id.clone(), Route::Ready(engine_id, adapter.identity()));
                continue;
            }
            match engine.load_adapter(adapter.as_ref().clone().into_engine()) {
                Ok(engine_id) => {
                    state.loaded.insert(digest.clone(), engine_id);
                    state.routes.insert(id.clone(), Route::Ready(engine_id, adapter.identity()));
                }
                Err(why) => {
                    eprintln!("serve: adapter {id} ({digest}) cannot be loaded into the bank, so requests for it fail: {why}");
                    state.routes.insert(id.clone(), Route::Refused);
                }
            }
        }
        let gone: Vec<String> = state.loaded.keys().filter(|d| !wanted.contains(d.as_str())).cloned().collect();
        for digest in gone {
            let engine_id = state.loaded.remove(&digest).expect("listed from the map");
            // `Draining` finishes by itself when the last sequence under the
            // adapter is done; an unknown id was already gone.
            if let Err(why) = engine.unload_adapter(engine_id) {
                if !matches!(why, AdapterError::Unknown(_)) {
                    eprintln!("serve: unloading adapter {digest} from the bank: {why}");
                }
            }
        }
        state.seen = Some(generation);
    }

    /// The engine adapter a request for serving id `id` runs under, with the
    /// identity of the release behind it.
    ///
    /// The error is what the caller is told: the same "no model" an unknown
    /// id gets, because an adapter that was withdrawn, or that the engine
    /// refused, is not something the caller can act on and its reason (logged
    /// where it happened) is the operator's.
    pub fn route(&self, id: &str) -> Result<(AdapterId, AdapterIdentity), String> {
        match self.state().routes.get(id) {
            Some(Route::Ready(engine_id, identity)) => Ok((*engine_id, identity.clone())),
            Some(Route::Refused) | None => Err(format!("no model '{id}'")),
        }
    }
}

/// A scheduler-backed instance of a text family whose engine has an adapter
/// bank: every request names its adapter by the reserved param its model id
/// stamped ([`ADAPTER_PARAM`]), and the group runs base and adapter rows in
/// the same batched steps. The engine's bank is brought in line with the
/// registry when a group starts and again, between steps, before an open group
/// admits a request after the registry moved, so an id re-published under load
/// routes every later request to its new release while running sequences keep
/// the release they began on (a replaced release drains in the bank, and a
/// reply names the identity captured at admission). A request naming an
/// adapter before it was offered is answered "no model".
pub struct AdaptedInstance<D: PagedDecoder + BankedEngine, F: Frontend = TextFrontend> {
    front: F,
    sched: Scheduler<D>,
    capacity: usize,
    adapters: AdapterSync,
}

impl<D: PagedDecoder + BankedEngine, F: Frontend> AdaptedInstance<D, F> {
    /// `capacity` is the number of requests the scheduler runs at once.
    pub fn new(front: F, sched: Scheduler<D>, capacity: usize, registry: Arc<AdapterRegistry>) -> AdaptedInstance<D, F> {
        let mut instance = AdaptedInstance { front, sched, capacity, adapters: AdapterSync::new(registry) };
        // Every adapter offered so far is registered with the engine before
        // the first request, so a cold instance answers an adapter request
        // at once.
        instance.adapters.sync(instance.sched.decoder_mut());
        instance
    }
}

impl<D: PagedDecoder + BankedEngine + Send, F: Frontend + Send> Instance for AdaptedInstance<D, F> {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.adapters.sync(self.sched.decoder_mut());
        if action == crate::SCORE_ACTION {
            return crate::score_batch_scheduled(&mut self.sched, &AdaptedFrontend { front: &self.front, adapters: &self.adapters }, invs);
        }
        run_batch_scheduled(&mut self.sched, &AdaptedFrontend { front: &self.front, adapters: &self.adapters }, invs, progress)
    }

    fn run_open(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn residency::JobFeed) -> bool {
        // A scoring group is closed: it is answered by `run_batch`.
        if action == crate::SCORE_ACTION {
            return false;
        }
        self.adapters.sync(self.sched.decoder_mut());
        // The bank and routes are refreshed whenever the registry has moved as a request is admitted, running
        // sequences or not: a re-published id never waits for its lane to drain.
        let adapters = &self.adapters;
        run_open_scheduled(&mut self.sched, &AdaptedFrontend { front: &self.front, adapters }, invs, self.capacity, progress, feed, &mut |engine| adapters.sync(engine));
        true
    }

    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        scheduler_metrics(&self.sched)
    }
}

/// The request front end of a family, plus the adapter a request's model id
/// stamped.
struct AdaptedFrontend<'a, F: Frontend> {
    front: &'a F,
    adapters: &'a AdapterSync,
}

impl<F: Frontend> Frontend for AdaptedFrontend<'_, F> {
    fn tok(&self) -> &data::qwen_tokenizer::QwenBpe {
        self.front.tok()
    }
    fn stop(&self) -> &[u32] {
        self.front.stop()
    }
    fn label(&self) -> &str {
        self.front.label()
    }
    fn parse(&self, inv: &Invocation) -> Result<crate::ParsedRequest, String> {
        self.front.parse(inv)
    }
    fn parse_embedded(&self, inv: &Invocation) -> Result<(crate::ParsedRequest, Option<model::serve::PromptEmbeds>), String> {
        self.front.parse_embedded(inv)
    }
    fn parse_in_background(&self, inv: &Invocation) -> Option<crate::Encoding> {
        self.front.parse_in_background(inv)
    }
    fn encodes_inputs(&self) -> bool {
        self.front.encodes_inputs()
    }
    fn route(&self, inv: &Invocation) -> Result<Option<(AdapterId, AdapterIdentity)>, String> {
        inv.get_str(ADAPTER_PARAM).map(|name| self.adapters.route(&name)).transpose()
    }
    fn stale(&self) -> bool {
        self.adapters.stale()
    }
    fn base_digest(&self) -> Option<String> {
        self.adapters.base_digest()
    }
    fn rejection(&self, inv: &Invocation, reason: &model::serve::RejectReason) -> String {
        match (reason, inv.get_str(ADAPTER_PARAM)) {
            // An adapter the engine unloaded between routing and admission:
            // as an unknown model, with no engine detail.
            (model::serve::RejectReason::AdapterUnavailable { .. }, Some(name)) => format!("no model '{name}'"),
            _ => self.front.rejection(inv, reason),
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

    /// The policy version of an id counts its releases with different weights:
    /// it starts at 1, moves up on each replacement, stays when the same
    /// weights are offered again, and a withdrawn id continues its count.
    #[test]
    fn an_adapters_policy_version_counts_its_releases_and_survives_withdrawal() {
        let dir = scratch_dir("registry-versions");
        let reg = registry(Some(AdapterBankSettings::default()));
        let (one, two) = (loaded(&dir, ID, 0.5), loaded(&dir, ID, 0.25));
        assert_ne!(one.release.digest, two.release.digest, "the fixtures differ");
        assert_eq!(one.version, 0, "a release read from disk has no version until it is offered");

        let lease = reg.attach(one.clone()).unwrap();
        assert_eq!(reg.identity(ID).unwrap().version, 1);
        assert_eq!(reg.replace(two.clone()).unwrap().version, 2, "different weights under the id are the next release");
        assert_eq!(reg.replace(two.clone()).unwrap().version, 2, "the same weights offered again keep their number");
        assert_eq!(reg.identity(ID).unwrap().version, 2);

        drop(lease);
        assert!(reg.identity(ID).is_none(), "withdrawn");
        let _again = reg.attach(one).unwrap();
        assert_eq!(reg.identity(ID).unwrap().version, 3, "an id offered again continues its count, so a version never repeats");

        let other = loaded(&dir, "Qwen/Qwen3-0.6B:lab:other:v1", 0.5);
        let _other = reg.attach(other).unwrap();
        assert_eq!(reg.identity("Qwen/Qwen3-0.6B:lab:other:v1").unwrap().version, 1, "each id counts for itself");
    }

    /// A learning loop trains rank 32; a server that sizes its bank by default
    /// must not refuse it, and the cap above it stays a flag.
    #[test]
    fn the_default_rank_cap_holds_rank_32() {
        assert_eq!(AdapterBankSettings::default().rank_cap, 32);
        assert_eq!(DEFAULT_ADAPTER_RANK_CAP, 32);
    }

    /// A brain trainer adapts every projection by default, and a bank is fixed when the server starts: the
    /// default bank must hold an adapter trained afterwards.
    #[test]
    fn the_default_targets_are_every_site_the_trainers_adapt() {
        assert_eq!(DEFAULT_ADAPTER_TARGETS, AdapterTargets::ALL);
        let model = qwen3::config::QwenConfig::tiny();
        let sites = AdapterBankSettings::default().bank_config(&model).unwrap().targets;
        assert_eq!(sites, qwen3::finetune::LORA_TARGETS, "the default bank corrects exactly what served training adapts by default");
    }

    #[test]
    fn settings_name_the_flag_that_is_wrong() {
        assert_eq!(AdapterBankSettings::default().validate(), Ok(()));
        for (bad, flag) in [
            (AdapterBankSettings { slots: 0, ..Default::default() }, "--adapter-slots"),
            (AdapterBankSettings { slots: MAX_SERVED_ADAPTERS + 1, ..Default::default() }, "--adapter-slots"),
            (AdapterBankSettings { rank_cap: 0, ..Default::default() }, "--adapter-rank-cap"),
        ] {
            assert!(bad.validate().unwrap_err().contains(flag), "{bad:?}");
        }
        let model = qwen3::config::QwenConfig::tiny();
        let cfg = AdapterBankSettings { slots: 3, rank_cap: 8, targets: AdapterTargets::ALL }.bank_config(&model).unwrap();
        assert_eq!((cfg.slots, cfg.r_max, cfg.targets.len(), cfg.max_registered), (3, 8, 7, MAX_SERVED_ADAPTERS), "`all` is every site the model has");
        let attention = AdapterBankSettings { targets: AdapterTargets::ATTENTION, ..Default::default() }.bank_config(&model).unwrap();
        assert_eq!(attention.targets, ["wq", "wk", "wv", "wo"]);
        let refused = AdapterBankSettings { targets: AdapterTargets::parse("wq,lm_head").unwrap(), ..Default::default() };
        assert!(refused.bank_config(&model).unwrap_err().contains("lm_head"), "a leaf the model lacks is refused when the bank is sized");
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
        let refusal = small.attach(adapter.clone()).err().unwrap();
        assert!(refusal.contains("--adapter-rank-cap 1;") && refusal.contains("restart with --adapter-rank-cap 2"), "the fixture is rank 2, and the refusal names the value that serves it: {refusal}");

        let ok = registry(Some(AdapterBankSettings::default()));
        let wrong_sites = registry(Some(AdapterBankSettings { targets: AdapterTargets::parse("down").unwrap(), ..Default::default() }));
        let refusal = wrong_sites.attach(adapter.clone()).err().unwrap();
        assert!(refusal.contains("corrects wq") && refusal.contains("--adapter-targets down,wq"), "the refusal names the missing linear and the flag value that serves it: {refusal}");

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

    /// What a serving process offers reaches a worker's registry as the same releases: the source lists each
    /// offered adapter with its file, digest and version, and the worker's mirror, rebuilt from that list alone,
    /// offers the same identities (a worker that never saw the first release still numbers the second as the
    /// serving process did). A file whose bytes are not the offered digest is refused and leaves the id unoffered;
    /// a detach withdraws it; each change moves the mirror's generation.
    #[test]
    fn a_worker_mirror_offers_what_the_serving_registry_offers_under_its_versions() {
        use residency::supervised::{AdapterSink, AdapterSource};
        let dir = scratch_dir("registry-mirror");
        let served = registry(Some(AdapterBankSettings::default()));
        let (one, two) = (loaded(&dir, ID, 0.5), loaded(&dir, ID, 0.25));
        let _lease = served.attach(one).unwrap();
        served.replace(two).unwrap();
        const OTHER: &str = "Qwen/Qwen3-0.6B:lab:other:v1";
        let _other = served.attach(loaded(&dir, OTHER, 0.125)).unwrap();
        let (generation, offered) = served.offered();
        assert_eq!(generation, served.generation(), "the list is as new as the generation it names");
        assert_eq!(offered.iter().map(|a| (a.id.as_str(), a.version)).collect::<Vec<_>>(), [(ID, 2), (OTHER, 1)]);

        let worker = registry(Some(AdapterBankSettings::default()));
        let mirror = AdapterMirror::new(worker.clone());
        for adapter in &offered {
            mirror.attach(adapter).unwrap();
        }
        for id in [ID, OTHER] {
            assert_eq!(worker.identity(id), served.identity(id), "{id} is the same release in the worker");
        }
        let before = worker.generation();
        let mut forged = offered[1].clone();
        forged.digest = "sha256:00".to_string();
        let refused = mirror.attach(&forged).unwrap_err();
        assert!(refused.contains("sha256:00"), "{refused}");
        assert_eq!(worker.identity(OTHER), None, "a refused release leaves its id unoffered");
        mirror.detach(ID);
        assert_eq!(worker.ids(), Vec::<String>::new(), "detached");
        assert!(worker.generation() > before, "each change moves the generation an instance syncs by");
        mirror.detach("Qwen/Qwen3-0.6B:lab:never:v1");
    }
}
