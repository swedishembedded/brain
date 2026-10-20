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
//! device state. The engine's adapter bank ([`model::adapter::bank::AdapterBank`]) is the
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use capability::{ActionResult, Invocation, Manifest, Progress};
use llmserve::{run_batch_scheduled, run_open_scheduled, OpenGroup, scheduler_metrics, Frontend, TextFrontend};
use model::serve::{PagedDecoder, Scheduler};
use brain_modelref::ModelRef;
use model::serve::AdapterId;
use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterServing, BankedEngine};
pub use model::adapter::bank::AdapterTargets;

use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{AdapterIdentity, Device, Executor, Instance, InstanceKey, MemCost, ResidentModel};

use crate::adapter_release::{LoadedAdapter, ServedBase};
use crate::resident_llm::QwenResident;
pub use crate::served_family::ServedFamily;

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

/// How big the adapter bank of one served Qwen3 is. Fixed when the engine is
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
}

/// The adapters one served Qwen3 base offers. See this module's doc.
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
        let family = ServedFamily::named(base.family()).unwrap_or_else(|| panic!("no adapter family '{}' is served", base.family()));
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
        self.lock().entries.get(id).map(|e| e.loaded.release.identity())
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
            return Err(format!("adapter {id} has rank {}, above this server's --adapter-rank-cap {}", loaded.adapter.rank, settings.rank_cap));
        }
        let mut missing: Vec<&str> = Vec::new();
        for site in &loaded.adapter.sites {
            let leaf = site.base.rsplit('.').nth(1).unwrap_or_default();
            if !self.family.all.contains(&leaf) {
                return Err(format!("adapter {id} holds a pair for {}, which is not a linear of a {} block (it has {})", site.base, self.family.tag, self.family.all.join(", ")));
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

/// A resident that carries an [`AdapterRegistry`] and can have adapters offered
/// as models of their own beside it: Qwen3 ([`QwenResident`]) and the released
/// Qwen3.8 GGUF ([`crate::resident_qwen35::Qwen35GgufServed`]).
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

/// A Qwen3 adapter's view.
pub type QwenAdapterView = AdapterView<QwenResident>;

impl<R: AdapterHost> AdapterView<R> {
    /// The view of the adapter `lease` holds, over `base`.
    pub(crate) fn new(base: Arc<R>, lease: AdapterLease) -> AdapterView<R> {
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
/// An adapter is offered by registering its [`QwenAdapterView`], withdrawn by
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
        let identity = adapter.release.identity();
        self.target.registry().replace(adapter)?;
        self.executor.set_served_adapter(&id, Some(identity));
        Ok(())
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
pub(crate) struct AdapterSync {
    registry: Arc<AdapterRegistry>,
    seen: Option<u64>,
    /// Engine id of every adapter the bank holds, by digest: the engine's
    /// identity for an adapter is its content.
    loaded: HashMap<String, AdapterId>,
    routes: HashMap<String, Route>,
}

impl AdapterSync {
    /// The served base's digest, once it has been hashed.
    pub(crate) fn base_digest(&self) -> Option<String> {
        self.registry.base().peek_digest()
    }

    pub(crate) fn new(registry: Arc<AdapterRegistry>) -> AdapterSync {
        AdapterSync { registry, seen: None, loaded: HashMap::new(), routes: HashMap::new() }
    }

    /// Whether the registry has changed since the last [`Self::sync`]: the
    /// routes may name a release that has been replaced.
    pub(crate) fn stale(&self) -> bool {
        self.seen != Some(self.registry.generation.load(Ordering::SeqCst))
    }

    /// Bring `engine`'s bank in line with the registry: register what it
    /// gained, unload what it lost. One atomic load when nothing changed.
    pub(crate) fn sync<E: BankedEngine>(&mut self, engine: &mut E) {
        if !self.stale() {
            return;
        }
        let (generation, offered) = self.registry.snapshot();
        let wanted: HashSet<&str> = offered.iter().map(|(_, a)| a.release.digest.as_str()).collect();
        self.routes.clear();
        for (id, adapter) in &offered {
            let digest = &adapter.release.digest;
            if let Some(&engine_id) = self.loaded.get(digest) {
                self.routes.insert(id.clone(), Route::Ready(engine_id, adapter.release.identity()));
                continue;
            }
            match engine.load_adapter(adapter.as_ref().clone().into_engine()) {
                Ok(engine_id) => {
                    self.loaded.insert(digest.clone(), engine_id);
                    self.routes.insert(id.clone(), Route::Ready(engine_id, adapter.release.identity()));
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
                if !matches!(why, AdapterError::Unknown(_)) {
                    eprintln!("serve: unloading adapter {digest} from the bank: {why}");
                }
            }
        }
        self.seen = Some(generation);
    }

    /// The engine adapter a request for serving id `id` runs under, with the
    /// identity of the release behind it.
    ///
    /// The error is what the caller is told: the same "no model" an unknown
    /// id gets, because an adapter that was withdrawn, or that the engine
    /// refused, is not something the caller can act on and its reason (logged
    /// where it happened) is the operator's.
    pub(crate) fn route(&self, id: &str) -> Result<(AdapterId, AdapterIdentity), String> {
        match self.routes.get(id) {
            Some(Route::Ready(engine_id, identity)) => Ok((*engine_id, identity.clone())),
            Some(Route::Refused) | None => Err(format!("no model '{id}'")),
        }
    }
}

/// A scheduler-backed instance of a text family whose engine has an adapter
/// bank: every request names its adapter by the reserved param its model id
/// stamped ([`ADAPTER_PARAM`]), and the group runs base and adapter rows in
/// the same batched steps. The engine's bank is brought in line with the
/// registry first, between groups, when no sequence is running, so a hot swap
/// never changes what a running sequence reads. An adapter offered while a
/// group runs is routable from the next group; a request naming it earlier is
/// answered "no model".
pub struct AdaptedInstance<D: PagedDecoder + BankedEngine> {
    front: TextFrontend,
    sched: Scheduler<D>,
    capacity: usize,
    adapters: AdapterSync,
}

impl<D: PagedDecoder + BankedEngine> AdaptedInstance<D> {
    /// `capacity` is the number of requests the scheduler runs at once.
    pub fn new(front: TextFrontend, sched: Scheduler<D>, capacity: usize, registry: Arc<AdapterRegistry>) -> AdaptedInstance<D> {
        let mut instance = AdaptedInstance { front, sched, capacity, adapters: AdapterSync::new(registry) };
        // Every adapter offered so far is registered with the engine before
        // the first request, so a cold instance answers an adapter request
        // at once.
        instance.adapters.sync(instance.sched.decoder_mut());
        instance
    }
}

impl<D: PagedDecoder + BankedEngine + Send> Instance for AdaptedInstance<D> {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.adapters.sync(self.sched.decoder_mut());
        run_batch_scheduled(&mut self.sched, &AdaptedFrontend { front: &self.front, adapters: &self.adapters }, invs, progress)
    }

    fn run_open(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn residency::JobFeed) -> bool {
        self.adapters.sync(self.sched.decoder_mut());
        let mut invs = invs;
        // A group that idles while the registry has moved hands back: refresh the
        // bank and routes, then go on with the jobs still queued on the feed.
        let mut numbered = 0;
        while let OpenGroup::Refresh { numbered: n } = run_open_scheduled(&mut self.sched, &AdaptedFrontend { front: &self.front, adapters: &self.adapters }, invs, self.capacity, numbered, progress, feed) {
            self.adapters.sync(self.sched.decoder_mut());
            (invs, numbered) = (&[], n);
        }
        true
    }

    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        scheduler_metrics(&self.sched)
    }
}

/// The request front end of a family, plus the adapter a request's model id
/// stamped.
struct AdaptedFrontend<'a> {
    front: &'a TextFrontend,
    adapters: &'a AdapterSync,
}

impl Frontend for AdaptedFrontend<'_> {
    fn tok(&self) -> &data::qwen_tokenizer::QwenBpe {
        self.front.tok()
    }
    fn stop(&self) -> &[u32] {
        self.front.stop()
    }
    fn label(&self) -> &str {
        self.front.label()
    }
    fn parse(&self, inv: &Invocation) -> Result<llmserve::ParsedRequest, String> {
        self.front.parse(inv)
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
    use crate::adapter_release::{load_adapter, QWEN35_FAMILY};
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
    fn the_default_targets_are_the_attention_group() {
        assert_eq!(DEFAULT_ADAPTER_TARGETS, AdapterTargets::ATTENTION);
    }

    const Q35_ID: &str = "unsloth/Qwen3.8-27B-GGUF:lab:facts:v1";

    fn qwen35_registry(settings: Option<AdapterBankSettings>) -> Arc<AdapterRegistry> {
        AdapterRegistry::new(settings, ServedBase::of_family(QWEN35_FAMILY, "unused.gguf", Some("unsloth/Qwen3.8-27B-GGUF".to_string())).quantizing(&["Qwen/Qwen3.8-27B-FP8"]))
    }

    /// A Qwen3.8 adapter over the named linears of layer 0 (shapes are the
    /// engine's to check, not the registry's).
    fn qwen35_adapter(dir: &std::path::Path, bases: &[&str]) -> LoadedAdapter {
        use brain_testutil::adapters::{write_lora_adapter_of, Linear};
        let linears: Vec<Linear> = bases.iter().map(|b| Linear { base: format!("blocks.0.{b}.weight"), out: 4, inn: 3 }).collect();
        let path = dir.join(format!("q35-{}.safetensors", bases.join("-").replace('.', "_")));
        write_lora_adapter_of("qwen35", &path, Q35_ID, "Qwen/Qwen3.8-27B-FP8", 2, &linears, 5);
        load_adapter(&path).expect("a trainer-shaped qwen35 adapter loads")
    }

    /// One registry implementation serves both families: the pre-attach check
    /// resolves against the family's real sites, not the Qwen3 ones, names the
    /// flag that widens the bank, and refuses an adapter of the other family.
    #[test]
    fn the_pre_attach_check_resolves_against_the_families_own_sites() {
        let dir = scratch_dir("registry-qwen35");
        let mixer = qwen35_adapter(&dir, &["linear_attn.in_proj_qkv", "self_attn.q_proj"]);
        let mlp = qwen35_adapter(&dir, &["mlp.gate"]);
        assert_eq!(mixer.family, "qwen35");

        let attention = qwen35_registry(Some(AdapterBankSettings::default()));
        assert!(attention.attach(mixer.clone()).is_ok(), "the default group of a Qwen3.8 is its mixers (in_proj_*, out_proj, q/k/v/o_proj)");
        let why = attention.attach(mlp.clone()).err().unwrap();
        assert!(why.contains("gate") && why.contains("--adapter-targets"), "{why}");

        let all = qwen35_registry(Some(AdapterBankSettings { targets: AdapterTargets::ALL, ..Default::default() }));
        assert!(all.attach(mlp.clone()).is_ok(), "`all` is every site the model has");
        let down = qwen35_registry(Some(AdapterBankSettings { targets: AdapterTargets::parse("gate,up,down").unwrap(), ..Default::default() }));
        assert!(down.attach(mlp).is_ok() && down.attach(mixer.clone()).is_err(), "named leaves are the family's own names");

        // The other family's adapter and the other family's leaves.
        let qwen3 = registry(Some(AdapterBankSettings::default()));
        let why = qwen3.attach(mixer).err().unwrap();
        assert!(why.contains("qwen35") && why.contains("qwen"), "an adapter of another family is refused by family: {why}");
        let q3_adapter = loaded(&dir, "unsloth/Qwen3.8-27B-GGUF:lab:wq:v1", 0.5);
        assert!(attention.attach(q3_adapter).err().unwrap().contains("qwen"), "a Qwen3 adapter does not ride on a Qwen3.8");
    }

    #[test]
    fn a_quantisation_is_bound_by_the_release_it_was_made_from() {
        let dir = scratch_dir("registry-quantizing");
        let adapter = qwen35_adapter(&dir, &["self_attn.q_proj"]);
        assert_eq!(adapter.release.id, Q35_ID);
        // The adapter's card names `Qwen/Qwen3.8-27B-FP8`, which the served
        // GGUF declares it quantises, so no file digest of the GGUF is needed.
        let base = ServedBase::of_family(QWEN35_FAMILY, "unused.gguf", Some("unsloth/Qwen3.8-27B-GGUF".to_string())).quantizing(&["Qwen/Qwen3.8-27B-FP8"]);
        assert!(crate::adapter_release::verify_adapter(&dir.join("q35-self_attn_q_proj.safetensors"), &base).is_ok());
        let file = dir.join("other.gguf");
        brain_testutil::adapters::write_base(&file, 1.0);
        let other = ServedBase::of_family(QWEN35_FAMILY, &file, Some("unsloth/Other".to_string()));
        let why = crate::adapter_release::verify_adapter(&dir.join("q35-self_attn_q_proj.safetensors"), &other).err().unwrap();
        assert!(why.contains("Qwen/Qwen3.8-27B-FP8") && why.contains("unsloth/Other"), "a base that is not the release's quantisation is refused by name: {why}");
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
        let model = crate::served_family::tiny_qwen();
        let cfg = AdapterBankSettings { slots: 3, rank_cap: 8, targets: AdapterTargets::ALL }.bank_config(&model).unwrap();
        assert_eq!((cfg.slots, cfg.r_max, cfg.targets.len(), cfg.max_registered), (3, 8, 7, MAX_SERVED_ADAPTERS), "`all` is every site the model has");
        let attention = AdapterBankSettings::default().bank_config(&model).unwrap();
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
        assert!(small.attach(adapter.clone()).err().unwrap().contains("--adapter-rank-cap"), "the fixture is rank 2");

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
}
