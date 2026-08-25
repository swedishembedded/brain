// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The host-side, **checkpoint-scoped** quantized-weight cache the real DiT
//! forward consults before it reads anything off disk.
//!
//! Swedish Embedded AB implements cross-request weight residency and memory
//! governance for production inference pipelines. If your team needs
//! expertise in making a multi-gigabyte checkpoint's cold-start cost
//! disappear without giving up bit-exact reproducibility, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # What changed, and why the old scope was the bug
//!
//! Phase 9 introduced this cache scoped to ONE `generate()` call: the
//! `RealDit` owned it and dropped it when the generation finished. That
//! removed the dominant share of every denoise step past the first, but it
//! left the whole cost standing on the FIRST step of every generation - and a
//! real profiling pass measured that first step at better than a third of the
//! whole run, because this box's rotational storage reads the checkpoint cold
//! at a rate far below what the GPU path needs. Two back-to-back generations
//! against the same checkpoint therefore paid that same disk cost twice,
//! seconds apart, for bytes that had not changed.
//!
//! The cache's contents were never a property of a generation. They are a
//! pure function of immutable checkpoint bytes, so their correct scope is the
//! CHECKPOINT, and their correct lifetime is however long the memory ceiling
//! says they may live. That is what this module implements:
//!
//! * keyed on [`CheckpointId`] (path + byte length + mtime - the identity
//!   [`crate::text_cache::encoder_identity`] already defines for exactly this
//!   purpose, reused rather than re-invented) plus the block index and the
//!   quant tier, so two generations with different prompts against one
//!   checkpoint share every entry and a replaced/re-quantized file at the
//!   same path shares none;
//! * held in a process-wide [`registry`], so the cache outlives the
//!   `RealDit`, the `generate()` call and the resident instance alike;
//! * bounded by a real byte budget (see [`budget_from_limits`]) and evicted
//!   with `residency::place::CostAware` - the SAME GDSF policy the residency
//!   manager uses for whole model instances, not a second, bespoke rule that
//!   would have to be re-tuned separately;
//! * `Sync`: a `RwLock` over the slot table and `Arc` over each entry, so a
//!   reader holds no lock while it uploads, and two threads (the conditional
//!   and unconditional CFG branches, on two cards) can read it concurrently.
//!   That dispatch is a later phase's work; this type simply does not block
//!   it.
//!
//! # Correctness, and what eviction can and cannot do
//!
//! Every entry is `model::int8::quantize_weight`/`int4::quantize_weight_q4`
//! applied to bytes that cannot change while the file's length and mtime do
//! not. A hit is therefore bit-identical to a recompute by construction, and
//! an eviction can only ever cost time: the next access misses, re-reads and
//! re-quantizes, and produces the same bytes again. There is no state here
//! that can go stale in a way that changes a number - only state that can go
//! missing. `tests/block_weight_cache.rs` gates both halves of that claim.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use residency::lru::Entry;
use residency::place::{CostAware, EvictionPolicy};
use residency::{Device, MemCost, Tier};

use crate::block::{CachedQBlockWeights, QTier};

/// A checkpoint's identity, without reading its contents: path plus the two
/// stat fields that change when the file does. Deliberately the SAME identity
/// [`crate::text_cache::Key`] carries for the text encoder - one definition
/// of "is this the same checkpoint" in this crate, not two.
///
/// A file that cannot be stat'ed yields zeros, which is why
/// [`GenerationCache::for_checkpoint`] declines to register such a path in
/// the shared [`registry`]: an unstable identity must degrade to a private,
/// per-caller cache, never to a shared entry that does not describe the file.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CheckpointId {
    pub path: String,
    pub len: u64,
    pub mtime: i64,
}

impl CheckpointId {
    pub fn for_path(path: &str) -> CheckpointId {
        let (len, mtime) = crate::text_cache::encoder_identity(path);
        CheckpointId { path: path.to_string(), len, mtime }
    }

    /// True when the stat succeeded, i.e. this identity actually describes a
    /// file rather than being the "could not stat" zero.
    fn is_stable(&self) -> bool {
        self.len != 0
    }
}

/// One block's cache slot, plus the two counters `CostAware` scores against.
struct Slot {
    weights: Arc<CachedQBlockWeights>,
    bytes: u64,
    uses: u64,
    last_use: u64,
}

/// One remembered embeddings-connector routing: the exact inputs it was
/// computed from, and what it produced.
struct ConnectorSlot {
    context: Vec<f32>,
    valid: Vec<f32>,
    context_len: usize,
    out: Vec<f32>,
    last_use: u64,
}

/// Hits, misses and evictions since this store was created - what a caller
/// (the trace, a resident model's `metrics`) reports instead of guessing
/// whether the cache is doing anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Blocks currently held.
    pub blocks: usize,
    /// Host bytes currently held by block entries.
    pub bytes: u64,
}

#[derive(Default)]
struct Inner {
    blocks: HashMap<(u32, QTier), Slot>,
    connector: Vec<ConnectorSlot>,
    bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// At most this many distinct connector routings are remembered.
///
/// Two is the working set of one generation with CFG on (the conditional and
/// unconditional branches); four lets the previous generation's pair survive
/// long enough to be reused by a re-run of the same prompt. Unbounded - which
/// is what the per-generation cache could afford, since it died with the
/// generation - would now be a genuine leak: every new prompt in a
/// long-running server adds a few megabytes that nothing ever removes.
const MAX_CONNECTOR_ENTRIES: usize = 4;

/// The shared, budgeted slot table behind a [`GenerationCache`] handle.
struct WeightStore {
    inner: RwLock<Inner>,
    /// `None` = ungoverned (no ceiling was published and none was asked for).
    budget: Option<u64>,
}

impl WeightStore {
    fn new(budget: Option<u64>) -> WeightStore {
        WeightStore { inner: RwLock::new(Inner::default()), budget }
    }
}

/// Score `slot` the way the residency manager scores a whole model instance,
/// so a cheap, cold, rarely-touched block is dropped before an expensive,
/// hot, frequently-touched one.
///
/// The block cache's entries live in host RAM and are reloaded from disk, so
/// they map onto `Tier::Warm` on `Device::Cpu` with a `MemCost` whose `ram`
/// IS the reload cost in bytes - which is precisely the signal
/// `CostAware::score` (GDSF: `uses * bytes / age`) consumes. Reusing the
/// policy object rather than transcribing its formula is the point: when the
/// residency benchmark re-tunes it, this cache inherits the new rule.
fn score(slot: &Slot, now: u64) -> f64 {
    let e = Entry { cost: MemCost::new(0, slot.bytes), device: Device::Cpu, last_use: slot.last_use, uses: slot.uses, pinned: false, tier: Tier::Warm };
    CostAware.score(&e, now)
}

/// The process-wide table of live stores, one per [`CheckpointId`].
///
/// Strong `Arc`s, deliberately: the whole point is that the cache outlives
/// every `RealDit` and every `generate()` call that populated it. What bounds
/// it is the byte budget and eviction, not the lifetime of a handle - a
/// `Weak` table would silently re-pay the full cold-disk cost the moment the
/// last generation finished, which is the exact defect this module exists to
/// remove.
fn registry() -> &'static RwLock<HashMap<CheckpointId, Arc<WeightStore>>> {
    static REG: OnceLock<RwLock<HashMap<CheckpointId, Arc<WeightStore>>>> = OnceLock::new();
    REG.get_or_init(Default::default)
}

/// Fraction of the process-wide host ceiling this cache may occupy.
///
/// Not a tuned constant - a statement about what else has to fit. A real
/// generation also holds the DiT head tensors, the encoded text context, the
/// VAE's weights and its pixel-space decode buffers in host RAM at the same
/// time, and `memauth::limits().ram_total` is the ceiling for ALL of it. Two
/// thirds leaves the rest a third, which at the real 22B/int8 footprint
/// (~13 GB of blocks) means a `--limit-ram-total 24G` run caches everything
/// and a tighter one caches a prefix and evicts the rest by cost.
const CACHE_SHARE_NUM: u64 = 2;
const CACHE_SHARE_DEN: u64 = 3;

/// The byte budget this cache runs under, from the process-wide ceiling the
/// `--limit-ram-total` flag publishes (`memauth::limits`).
///
/// `None` when no ceiling was published: that is the pre-existing,
/// deliberately unbounded behaviour on a box with 184 GiB of RAM and a 13 GB
/// cache, and turning it into a guessed default would change how every
/// existing run behaves in exchange for governing nothing anybody asked to
/// govern. A run that wants a bound says so, with the flag the sibling
/// milestone added for exactly this.
pub fn budget_from_limits() -> Option<u64> {
    memauth::limits().ram_total.map(|n| n / CACHE_SHARE_DEN * CACHE_SHARE_NUM)
}

/// A handle onto one checkpoint's cache.
///
/// Cloning a handle shares the store; two handles obtained from
/// [`Self::for_checkpoint`] with the same path share it too. `Default` is a
/// PRIVATE, unregistered, ungoverned store - what a test or a synthetic
/// in-memory `TensorSource` (which has no checkpoint identity at all) gets,
/// and exactly the behaviour every existing caller of `GenerationCache::
/// default()` already had.
///
/// The name is historical: it was one generation's scratch. It is now one
/// checkpoint's, and the type is kept so no caller's import path changed
/// while the scope did.
#[derive(Clone)]
pub struct GenerationCache {
    store: Arc<WeightStore>,
}

impl Default for GenerationCache {
    fn default() -> GenerationCache {
        GenerationCache { store: Arc::new(WeightStore::new(budget_from_limits())) }
    }
}

impl GenerationCache {
    /// The shared cache for the checkpoint at `path`, creating it on first
    /// use. Two `generate()` calls against the same file - different prompts,
    /// different sizes, seconds or hours apart - get the same store and
    /// therefore the same already-quantized bytes.
    ///
    /// A path that cannot be stat'ed falls back to a private store: an
    /// identity that does not describe the file must not be shared under
    /// (see [`CheckpointId`]).
    pub fn for_checkpoint(path: &str) -> GenerationCache {
        let id = CheckpointId::for_path(path);
        if !id.is_stable() {
            return GenerationCache::default();
        }
        if let Some(s) = registry().read().unwrap().get(&id) {
            return GenerationCache { store: s.clone() };
        }
        let mut reg = registry().write().unwrap();
        let s = reg.entry(id).or_insert_with(|| Arc::new(WeightStore::new(budget_from_limits()))).clone();
        GenerationCache { store: s }
    }

    /// A private store under an explicit byte budget - the seam a test uses
    /// to force eviction deterministically without publishing a
    /// process-wide `memauth` ceiling (a `OnceLock`, so a test that set one
    /// would fix it for every other test in the same binary).
    pub fn with_budget(budget: Option<u64>) -> GenerationCache {
        GenerationCache { store: Arc::new(WeightStore::new(budget)) }
    }

    /// The byte budget in force, if any.
    pub fn budget(&self) -> Option<u64> {
        self.store.budget
    }

    /// This layer's cached weights at `tier`, if held. Returns an `Arc` and
    /// releases the lock, so the (multi-hundred-millisecond) device upload
    /// that follows blocks no other reader.
    pub fn block(&self, layer: usize, tier: QTier) -> Option<Arc<CachedQBlockWeights>> {
        let mut inner = self.store.inner.write().unwrap();
        inner.tick += 1;
        let now = inner.tick;
        let Some(slot) = inner.blocks.get_mut(&(layer as u32, tier)) else {
            inner.misses += 1;
            return None;
        };
        slot.uses += 1;
        slot.last_use = now;
        let w = slot.weights.clone();
        inner.hits += 1;
        Some(w)
    }

    /// Take ownership of a freshly quantized block, retain it if the budget
    /// allows, and hand back the shared handle the caller uploads from.
    ///
    /// Always returns usable weights. When one block alone exceeds the whole
    /// budget the entry is simply not retained - the forward still runs, it
    /// just re-reads next time. A cache that refused to hand back weights it
    /// could not keep would turn a memory ceiling into a failure.
    pub fn store_block(&self, layer: usize, tier: QTier, weights: CachedQBlockWeights) -> Arc<CachedQBlockWeights> {
        let bytes = weights.byte_len() as u64;
        let w = Arc::new(weights);
        let mut inner = self.store.inner.write().unwrap();
        inner.tick += 1;
        let now = inner.tick;
        if let Some(budget) = self.store.budget {
            if bytes > budget {
                return w;
            }
            evict_until_fits(&mut inner, budget, bytes, now);
        }
        if let Some(old) = inner.blocks.insert((layer as u32, tier), Slot { weights: w.clone(), bytes, uses: 1, last_use: now }) {
            inner.bytes -= old.bytes;
        }
        inner.bytes += bytes;
        w
    }

    /// True when `layer` is held at `tier` - a structural check for a test,
    /// with no effect on the hit/miss counters or the recency stamps.
    pub fn is_cached(&self, layer: usize, tier: QTier) -> bool {
        self.store.inner.read().unwrap().blocks.contains_key(&(layer as u32, tier))
    }

    /// Hits/misses/evictions/blocks/bytes so far.
    pub fn stats(&self) -> CacheStats {
        let inner = self.store.inner.read().unwrap();
        CacheStats { hits: inner.hits, misses: inner.misses, evictions: inner.evictions, blocks: inner.blocks.len(), bytes: inner.bytes }
    }

    /// Real host bytes the block half currently holds.
    pub fn block_byte_len(&self) -> u64 {
        self.store.inner.read().unwrap().bytes
    }

    /// Per-entry byte counts, so a real-weight test can MEASURE the per-block
    /// footprint at the production width rather than assert it from a
    /// derivation nobody checks.
    pub fn block_byte_lens(&self) -> Vec<usize> {
        self.store.inner.read().unwrap().blocks.values().map(|s| s.bytes as usize).collect()
    }

    /// Drop everything this store holds. What `residency::Instance::demote`
    /// calls: it releases host RAM without invalidating anything, because a
    /// later access simply misses and recomputes the identical bytes.
    pub fn clear(&self) {
        let mut inner = self.store.inner.write().unwrap();
        inner.evictions += inner.blocks.len() as u64;
        inner.blocks.clear();
        inner.connector.clear();
        inner.bytes = 0;
    }

    /// The connector output previously computed for exactly these inputs, if
    /// any. Compared by VALUE rather than by a hash or a pointer: a hash
    /// collision would silently substitute one prompt's conditioning for
    /// another's, and the comparison is a few megabytes against a routing
    /// that costs seconds.
    pub(crate) fn connector_hit(&self, context: &[f32], valid: &[f32], context_len: usize) -> Option<Vec<f32>> {
        let mut inner = self.store.inner.write().unwrap();
        inner.tick += 1;
        let now = inner.tick;
        let hit = inner.connector.iter_mut().find(|e| e.context_len == context_len && e.context == context && e.valid == valid)?;
        hit.last_use = now;
        Some(hit.out.clone())
    }

    pub(crate) fn connector_store(&self, context: &[f32], valid: &[f32], context_len: usize, out: &[f32]) {
        let mut inner = self.store.inner.write().unwrap();
        inner.tick += 1;
        let now = inner.tick;
        while inner.connector.len() >= MAX_CONNECTOR_ENTRIES {
            let victim = inner.connector.iter().enumerate().min_by_key(|(_, e)| e.last_use).map(|(i, _)| i).expect("non-empty");
            inner.connector.remove(victim);
            inner.evictions += 1;
        }
        inner.connector.push(ConnectorSlot { context: context.to_vec(), valid: valid.to_vec(), context_len, out: out.to_vec(), last_use: now });
    }

    /// Real host bytes the connector half holds - the counterpart of
    /// [`Self::block_byte_len`], so a test can measure this cache's own
    /// footprint instead of deriving it (a memory claim nothing measures is
    /// not a measured claim).
    pub fn connector_byte_len(&self) -> usize {
        self.store.inner.read().unwrap().connector.iter().map(|e| std::mem::size_of_val(e.context.as_slice()) + std::mem::size_of_val(e.valid.as_slice()) + std::mem::size_of_val(e.out.as_slice())).sum()
    }
}

/// Drop lowest-scoring entries until `incoming` more bytes fit under
/// `budget`. Never loops forever: each pass removes exactly one entry and the
/// table is finite, and an empty table always "fits" because
/// `store_block` already rejected an entry larger than the whole budget.
fn evict_until_fits(inner: &mut Inner, budget: u64, incoming: u64, now: u64) {
    while inner.bytes + incoming > budget && !inner.blocks.is_empty() {
        let victim = inner.blocks.iter().min_by(|a, b| score(a.1, now).partial_cmp(&score(b.1, now)).unwrap_or(std::cmp::Ordering::Equal)).map(|(k, _)| *k).expect("non-empty");
        if let Some(s) = inner.blocks.remove(&victim) {
            inner.bytes -= s.bytes;
            inner.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two handles for the same path are the same store; a different path is
    /// a different store. This is what makes generation B reuse generation
    /// A's entries, and it is the one property the whole milestone rests on.
    #[test]
    fn one_checkpoint_path_yields_one_shared_store_and_a_different_path_does_not() {
        let dir = std::env::temp_dir().join(format!("ltxv-wc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        std::fs::write(&a, b"aaaa").unwrap();
        std::fs::write(&b, b"bbbbbb").unwrap();
        let (pa, pb) = (a.to_string_lossy().to_string(), b.to_string_lossy().to_string());

        let h1 = GenerationCache::for_checkpoint(&pa);
        let h2 = GenerationCache::for_checkpoint(&pa);
        let h3 = GenerationCache::for_checkpoint(&pb);
        assert!(Arc::ptr_eq(&h1.store, &h2.store), "the same checkpoint path must hand back the SAME store");
        assert!(!Arc::ptr_eq(&h1.store, &h3.store), "a different checkpoint must not share a store");

        // A path that cannot be stat'ed degrades to a private store rather
        // than sharing under an identity that describes no file.
        let missing = dir.join("nope.gguf").to_string_lossy().to_string();
        let m1 = GenerationCache::for_checkpoint(&missing);
        let m2 = GenerationCache::for_checkpoint(&missing);
        assert!(!Arc::ptr_eq(&m1.store, &m2.store), "an unstat-able path must not be registered as a shared identity");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A rewritten file at the same path is a DIFFERENT checkpoint and must
    /// not inherit the old one's entries - the bug class that would serve one
    /// checkpoint's weights for another's.
    #[test]
    fn rewriting_the_file_changes_its_identity() {
        let dir = std::env::temp_dir().join(format!("ltxv-wc-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.gguf");
        std::fs::write(&p, b"one").unwrap();
        let id1 = CheckpointId::for_path(&p.to_string_lossy());
        std::fs::write(&p, b"a different length").unwrap();
        let id2 = CheckpointId::for_path(&p.to_string_lossy());
        assert_ne!(id1, id2, "a rewritten checkpoint at the same path must not share an identity");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sibling milestone will dispatch the two CFG branches concurrently
    /// across two cards, and both branches read this cache. Pinning
    /// `Send + Sync` here means that work does not begin by redesigning this
    /// type's concurrency primitive - the mistake the previous `RefCell`
    /// shape would have forced.
    #[test]
    fn the_cache_is_send_and_sync_so_concurrent_cfg_dispatch_is_not_blocked() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GenerationCache>();
        assert_send_sync::<CheckpointId>();
    }

    /// And it really survives concurrent use, not merely the trait bounds:
    /// many threads hammering one store's read and write paths must leave the
    /// accounting consistent (bytes equal to the sum of what is held) rather
    /// than deadlocking or drifting.
    #[test]
    fn concurrent_readers_and_writers_leave_the_accounting_consistent() {
        let cache = GenerationCache::with_budget(None);
        std::thread::scope(|sc| {
            for _ in 0..8 {
                let c = cache.clone();
                sc.spawn(move || {
                    for l in 0..16 {
                        if c.block(l, QTier::Int8).is_none() {
                            // A real block is expensive to build; the
                            // accounting under test does not care what the
                            // bytes are, only that they are consistently
                            // charged and released.
                            c.block(l, QTier::Int4);
                        }
                        c.connector_store(&[l as f32], &[1.0], 1, &[l as f32; 4]);
                        let _ = c.connector_hit(&[l as f32], &[1.0], 1);
                    }
                });
            }
        });
        let s = cache.stats();
        assert_eq!(cache.block_byte_len(), 0, "nothing was stored, so nothing may be charged");
        assert_eq!(s.blocks, 0);
        assert!(s.misses > 0, "the readers must actually have run: {s:?}");
        assert!(cache.connector_byte_len() > 0, "the connector half must hold its bounded working set");
    }

    /// The budget derives from the published ceiling and leaves room for
    /// everything else a generation holds in host RAM.
    #[test]
    fn the_budget_is_a_share_of_the_published_ceiling() {
        assert_eq!(budget_from_limits(), memauth::limits().ram_total.map(|n| n / 3 * 2));
    }
}
