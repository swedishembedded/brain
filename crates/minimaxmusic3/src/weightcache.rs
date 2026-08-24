// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The process-wide, **checkpoint-scoped** host-weight cache the pipeline
//! consults before it reads anything off disk.
//!
//! Swedish Embedded AB implements cross-request weight residency and memory
//! governance for production inference pipelines. If your team needs
//! expertise in making a multi-gigabyte checkpoint's cold-start cost
//! disappear without giving up bit-exact reproducibility, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! # Why this exists now, and why it did not before
//!
//! `crate::caps`, `crate::generate` and the residency adapter all used to
//! say the same thing: nothing here is held warm, because "the whole
//! checkpoint does not fit in RAM even once on the machine this port was
//! built on". That machine had ~26 GB of RAM and an integrated GPU. The
//! validation machine has **184 GB and two 24 GB Tesla P40s** - a fact this
//! crate's roadmap ledger records at Phase 12, along with the instruction to
//! "read every 'on this machine' claim above as historical". The
//! justification for statelessness expired with the hardware; the four
//! host-side weight sets are 12-13 GB together, which is 7% of this box's
//! RAM.
//!
//! What that costs today is a re-read of `transformer/` (9.1 GB on disk),
//! `rvq_depth_decoder/` (1.3 GB), `vocoder/` (207 MB) and
//! `condition_encoder/` (97 MB) on **every** `generate` call, for bytes that
//! did not change between calls.
//!
//! # What is cached, and what deliberately is not
//!
//! Cached: the four components whose imported form is a plain tree of host
//! `Vec<f32>` - [`crate::dit::DitWeights`],
//! [`crate::depth_decoder::DepthDecoderWeights`],
//! [`crate::vocoder::VocoderWeights`] and
//! [`crate::condition_encoder::ConditionEncoderWeights`]. They own no
//! device buffer, borrow nothing, and are `Send + Sync`, so an `Arc` of one
//! is safe to hand to any stage on any card.
//!
//! **Not** cached: the Global LLM. `global_llm::import` returns a
//! `qwen3::Qwen`, which owns its own `Gpu` (`qwen3::model::Qwen::gpu`), is
//! `Send` but **not** `Sync`, and is built against a KV capacity
//! `t = prompt_len + max_frames + 8` that is a function of the REQUEST
//! (`crate::generate::generate`), not of the checkpoint. A cache entry keyed
//! on the checkpoint would therefore be wrong for the next request's
//! capacity, and one keyed on the capacity too would be a device-resident
//! 13.5 GB object pinned to a card the scheduler has not been told about.
//! Warming it is real work with a real design (a capacity-max policy plus a
//! KV-reset seam `qwen3` does not expose today) and is named in the roadmap
//! rather than half-done here. Its import is already streamed - peak host
//! RAM during it is one tensor, not the 16 GB checkpoint - so what a cache
//! would save there is disk and quantize time, not a memory cliff.
//!
//! # Correctness: what an eviction can and cannot cost
//!
//! Every entry is `<component>::import` applied to bytes that cannot change
//! while the checkpoint directory's total length and newest mtime do not. A
//! hit is therefore bit-identical to a re-import by construction, and an
//! eviction can only ever cost TIME: the next access misses, re-reads, and
//! produces the same numbers again. There is no state here that can go
//! stale in a way that changes an output - only state that can go missing.
//! That is what makes it safe for `crates/residency` to drop this cache at
//! any moment, which is exactly what
//! `crates/cli/src/resident_minimaxmusic3.rs`'s `Instance::demote` does.
//!
//! # Who decides how big it gets
//!
//! Two governors, in this order:
//!
//! 1. **The residency manager**, through `Instance::demote`/`promote` on the
//!    resident adapter - the intended one. `estimate`/`estimate_at` report
//!    what this cache holds, so the manager budgets against a real number
//!    and evicts a real number.
//! 2. **A local byte ceiling**, [`budget_from_limits`], for the in-process
//!    callers that have no residency manager at all (`brain minimaxmusic3
//!    generate`, a bench, a test). It evicts with
//!    `residency::place::CostAware` - the SAME GDSF policy the residency
//!    manager scores whole model instances with, reused rather than
//!    transcribed, so a future re-tuning of that policy reaches here too.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use residency::place::{CostAware, EvictionPolicy};
use residency::{Device, MemCost, Tier};

use crate::condition_encoder::ConditionEncoderWeights;
use crate::config::{DepthDecoderConfig, DitConfig, VocoderConfig};
use crate::depth_decoder::DepthDecoderWeights;
use crate::dit::DitWeights;
use crate::vocoder::VocoderWeights;

/// Which component a cache entry holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Role {
    Dit,
    DepthDecoder,
    Vocoder,
    ConditionEncoder,
}

/// A checkpoint DIRECTORY's identity, without reading its contents: the
/// path, the summed length of every file under it, and the newest mtime
/// among them - all three recursively, because a nested layout is a real
/// released shape.
///
/// A directory's OWN mtime is not enough: it changes when an entry is added
/// or removed but not when a file's contents are rewritten in place, so an
/// identity built on it would serve stale weights after a re-download. The
/// summed length plus newest mtime moves on either.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CheckpointId {
    pub path: String,
    pub bytes: u64,
    pub mtime: i64,
}

impl CheckpointId {
    pub fn for_dir(dir: &str) -> CheckpointId {
        let (bytes, mtime) = scan(std::path::Path::new(dir));
        CheckpointId { path: dir.to_string(), bytes, mtime }
    }

    /// True when the scan actually saw a file. A path that cannot be read
    /// yields `(0, 0)`, which is an identity that describes nothing - the
    /// loaders below refuse to register such an entry rather than sharing a
    /// slot that could not be invalidated.
    fn is_stable(&self) -> bool {
        self.bytes != 0
    }
}

/// `(summed file length, newest mtime)` under `dir`, RECURSIVELY. A
/// non-recursive walk silently returns 0 for a nested checkpoint layout,
/// which would make every such checkpoint's identity unstable and disable
/// the cache for it without saying so.
fn scan(dir: &std::path::Path) -> (u64, i64) {
    let Ok(rd) = std::fs::read_dir(dir) else { return (0, 0) };
    let mut bytes = 0u64;
    let mut mtime = 0i64;
    for e in rd.flatten() {
        match e.file_type() {
            Ok(t) if t.is_dir() => {
                let (b, m) = scan(&e.path());
                bytes += b;
                mtime = mtime.max(m);
            }
            _ => {
                if let Ok(md) = e.metadata() {
                    bytes += md.len();
                    let secs = md
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    mtime = mtime.max(secs);
                }
            }
        }
    }
    (bytes, mtime)
}

/// The total on-disk size of a checkpoint directory, recursively - what a
/// caller with no closed form for a component's host footprint sizes it
/// from. Exposed because `crates/cli/src/resident_minimaxmusic3.rs` needs
/// exactly this number and re-deriving it there would be a second walk that
/// could disagree with the one the identity above is built on.
pub fn checkpoint_bytes(dir: &str) -> u64 {
    scan(std::path::Path::new(dir)).0
}

/// One cached component.
enum Held {
    Dit(Arc<DitWeights>),
    DepthDecoder(Arc<DepthDecoderWeights>),
    Vocoder(Arc<VocoderWeights>),
    ConditionEncoder(Arc<ConditionEncoderWeights>),
}

struct Slot {
    held: Held,
    bytes: u64,
    uses: u64,
    last_use: u64,
}

/// Hits, misses, evictions and the current footprint - what the resident
/// adapter's `Instance::metrics` reports instead of an operator having to
/// infer from a trace whether residency is doing anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Components currently held.
    pub entries: usize,
    /// Host bytes currently held.
    pub bytes: u64,
}

#[derive(Default)]
struct Inner {
    slots: HashMap<(Role, CheckpointId), Slot>,
    bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

fn store() -> &'static RwLock<Inner> {
    static STORE: OnceLock<RwLock<Inner>> = OnceLock::new();
    STORE.get_or_init(Default::default)
}

/// Fraction of the process-wide host ceiling this cache may occupy when one
/// is published at all.
///
/// Not a tuned constant - a statement about what else has to fit. A real
/// generation ALSO holds, in host RAM and at the same time, the Global LLM's
/// `lm_head` read back for the sampling head (3.28 GB), the whole song's
/// per-frame hidden states (`num_frames * 8 * 4096 * 4` bytes - ~786 MB for
/// four minutes) and every chunk's latents across the denoise/vocoder
/// boundary. Two thirds leaves the rest a third.
const CACHE_SHARE_NUM: u64 = 2;
const CACHE_SHARE_DEN: u64 = 3;

/// The local byte ceiling, or `None` when no process-wide host limit was
/// published (`--limit-ram-total` / `BRAIN_LIMIT_RAM_TOTAL`) - in which case
/// the residency manager's `demote` is the only governor, which is the
/// intended arrangement under `brain serve`.
pub fn budget_from_limits() -> Option<u64> {
    memauth::limits().ram_total.map(|n| n / CACHE_SHARE_DEN * CACHE_SHARE_NUM)
}

/// Score a slot the way the residency manager scores a whole model
/// instance. These entries live in host RAM and are reloaded from disk, so
/// they map onto `Tier::Warm` on `Device::Cpu` with a `MemCost` whose `ram`
/// IS the reload cost in bytes - precisely the signal `CostAware::score`
/// (GDSF: `uses * bytes / age`) consumes.
fn score(slot: &Slot, now: u64) -> f64 {
    let e = residency::lru::Entry { cost: MemCost::new(0, slot.bytes), device: Device::Cpu, last_use: slot.last_use, uses: slot.uses, pinned: false, tier: Tier::Warm };
    CostAware.score(&e, now)
}

/// Drop lowest-scoring slots until the footprint fits `budget`. `keep` is
/// the entry the caller just inserted, which must never be the victim of
/// its own insertion.
fn evict_to(inner: &mut Inner, budget: u64, keep: &(Role, CheckpointId)) {
    if inner.bytes <= budget {
        return;
    }
    let now = inner.tick;
    let mut order: Vec<((Role, CheckpointId), f64)> = inner.slots.iter().filter(|(k, _)| *k != keep).map(|(k, s)| (k.clone(), score(s, now))).collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (key, _) in order {
        if inner.bytes <= budget {
            break;
        }
        if let Some(s) = inner.slots.remove(&key) {
            inner.bytes = inner.bytes.saturating_sub(s.bytes);
            inner.evictions += 1;
        }
    }
}

/// Look `key` up; on a hit bump its counters and return the slot's contents.
fn take_hit(key: &(Role, CheckpointId)) -> Option<Held> {
    let mut inner = store().write().ok()?;
    inner.tick += 1;
    let now = inner.tick;
    let slot = inner.slots.get_mut(key)?;
    slot.uses += 1;
    slot.last_use = now;
    let held = match &slot.held {
        Held::Dit(w) => Held::Dit(w.clone()),
        Held::DepthDecoder(w) => Held::DepthDecoder(w.clone()),
        Held::Vocoder(w) => Held::Vocoder(w.clone()),
        Held::ConditionEncoder(w) => Held::ConditionEncoder(w.clone()),
    };
    inner.hits += 1;
    Some(held)
}

fn insert(key: (Role, CheckpointId), held: Held, bytes: u64) {
    let Ok(mut inner) = store().write() else { return };
    inner.tick += 1;
    let now = inner.tick;
    inner.misses += 1;
    if let Some(old) = inner.slots.insert(key.clone(), Slot { held, bytes, uses: 1, last_use: now }) {
        inner.bytes = inner.bytes.saturating_sub(old.bytes);
    }
    inner.bytes += bytes;
    if let Some(b) = budget_from_limits() {
        evict_to(&mut inner, b, &key);
    }
}

/// The single load path: on a hit return the `Arc`, on a miss run `load`
/// **outside** the lock and register the result.
///
/// The load runs unlocked deliberately: it is a multi-second, multi-GB disk
/// read, and holding the table across it would serialise every other
/// component's lookup behind it. Two threads racing the same miss both do
/// the read and the second one's insert replaces the first - wasteful once,
/// never wrong, and the alternative (a per-key in-flight latch) buys nothing
/// for a pipeline whose stages are sequential by construction.
fn get_or_load<T>(role: Role, dir: &str, bytes: u64, load: impl FnOnce() -> Result<T, String>, wrap: impl FnOnce(Arc<T>) -> Held, unwrap: impl Fn(&Held) -> Option<Arc<T>>) -> Result<Arc<T>, String> {
    let id = CheckpointId::for_dir(dir);
    if !id.is_stable() {
        // Nothing to key on - a directory that cannot be scanned cannot be
        // invalidated either, so it is loaded fresh and never registered.
        return load().map(Arc::new);
    }
    let key = (role, id);
    if let Some(held) = take_hit(&key) {
        if let Some(w) = unwrap(&held) {
            return Ok(w);
        }
    }
    let w = Arc::new(load()?);
    insert(key, wrap(w.clone()), bytes);
    Ok(w)
}

/// The fp32-materialization factor for a component with no closed form for
/// its host footprint.
///
/// This repo's safetensors reader materialises every tensor as f32 on read,
/// so a bf16/fp16 checkpoint doubles going into host memory. Charging 2x
/// over-charges a checkpoint that is already fp32 on disk (the DiT's is),
/// which is the safe direction for a memory budget - and the two components
/// this applies to are 207 MB and 97 MB on disk, so the worst case is
/// ~300 MB of over-budgeting, not a placement decision.
const FP32_MATERIALIZATION_FACTOR: u64 = 2;

/// [`crate::dit::import`], warm. Charged at the closed form
/// [`crate::memory::dit_weight_bytes`] or the on-disk size, whichever is
/// larger - the closed form omits the non-block tensors and the on-disk
/// figure omits nothing but is already fp32 here, so the max covers both.
pub fn dit(dir: &str, cfg: &DitConfig) -> Result<Arc<DitWeights>, String> {
    let bytes = crate::memory::dit_weight_bytes(cfg).max(checkpoint_bytes(dir));
    get_or_load(
        Role::Dit,
        dir,
        bytes,
        || crate::dit::import(dir, cfg),
        Held::Dit,
        |h| match h {
            Held::Dit(w) => Some(w.clone()),
            _ => None,
        },
    )
}

/// [`crate::depth_decoder::import`], warm. Charged at
/// [`crate::memory::depth_decoder_weight_bytes`], which is the closed form
/// over every tensor this component has (the checkpoint is bf16 on disk, so
/// the on-disk figure would under-charge by 2x).
pub fn depth_decoder(dir: &str, cfg: &DepthDecoderConfig) -> Result<Arc<DepthDecoderWeights>, String> {
    let bytes = crate::memory::depth_decoder_weight_bytes(cfg);
    get_or_load(
        Role::DepthDecoder,
        dir,
        bytes,
        || crate::depth_decoder::import(dir, cfg),
        Held::DepthDecoder,
        |h| match h {
            Held::DepthDecoder(w) => Some(w.clone()),
            _ => None,
        },
    )
}

/// [`crate::vocoder::import`], warm. No closed form exists for this
/// component's host footprint (its shapes come out of a conv stack this
/// crate does not enumerate), so it is charged from its checkpoint - see
/// [`FP32_MATERIALIZATION_FACTOR`].
pub fn vocoder(dir: &str, cfg: &VocoderConfig) -> Result<Arc<VocoderWeights>, String> {
    let bytes = checkpoint_bytes(dir) * FP32_MATERIALIZATION_FACTOR;
    get_or_load(
        Role::Vocoder,
        dir,
        bytes,
        || crate::vocoder::import(dir, cfg),
        Held::Vocoder,
        |h| match h {
            Held::Vocoder(w) => Some(w.clone()),
            _ => None,
        },
    )
}

/// [`crate::condition_encoder::import`], warm. Charged like the vocoder.
pub fn condition_encoder(dir: &str) -> Result<Arc<ConditionEncoderWeights>, String> {
    let bytes = checkpoint_bytes(dir) * FP32_MATERIALIZATION_FACTOR;
    get_or_load(
        Role::ConditionEncoder,
        dir,
        bytes,
        || crate::condition_encoder::import(dir),
        Held::ConditionEncoder,
        |h| match h {
            Held::ConditionEncoder(w) => Some(w.clone()),
            _ => None,
        },
    )
}

/// Hits, misses, evictions and the current footprint.
pub fn stats() -> CacheStats {
    let Ok(inner) = store().read() else { return CacheStats::default() };
    CacheStats { hits: inner.hits, misses: inner.misses, evictions: inner.evictions, entries: inner.slots.len(), bytes: inner.bytes }
}

/// Host bytes currently held.
pub fn bytes() -> u64 {
    stats().bytes
}

/// Release everything. Safe at any moment - see this module's own
/// correctness note - and the operation
/// `crates/cli/src/resident_minimaxmusic3.rs`'s `Instance::demote` performs.
/// Callers still holding an `Arc` from a previous load keep their copy alive
/// until they drop it, which is what makes this safe to call while a
/// generation is running.
pub fn clear() {
    let Ok(mut inner) = store().write() else { return };
    let n = inner.slots.len() as u64;
    inner.slots.clear();
    inner.bytes = 0;
    inner.evictions += n;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory whose contents change must not be served from the cache.
    /// Length-only would miss an in-place rewrite of the same size; mtime
    /// alone would miss a same-second replacement of a different size. Both
    /// together is the claim.
    #[test]
    fn the_identity_moves_when_the_checkpoint_does() {
        let dir = std::env::temp_dir().join(format!("mm3-wc-id-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.to_string_lossy().into_owned();
        std::fs::write(dir.join("a.safetensors"), b"one").unwrap();
        let a = CheckpointId::for_dir(&path);
        assert!(a.is_stable());
        std::fs::write(dir.join("a.safetensors"), b"one plus more").unwrap();
        assert_ne!(a, CheckpointId::for_dir(&path), "a longer file must change the identity");

        // A NESTED file must count: a non-recursive scan would report the
        // same identity for a nested layout no matter what changed in it.
        let nested = dir.join("sub");
        std::fs::create_dir_all(&nested).unwrap();
        let before = CheckpointId::for_dir(&path);
        std::fs::write(nested.join("b.safetensors"), b"nested").unwrap();
        assert_ne!(before, CheckpointId::for_dir(&path), "a nested file must be seen");
        assert_eq!(checkpoint_bytes(&path), 13 + 6);

        // An unreadable path is UNSTABLE, so nothing is ever registered
        // under it (rather than every such path sharing one wrong slot).
        assert!(!CheckpointId::for_dir("/nonexistent-mm3-checkpoint").is_stable());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A miss loads, a hit does not - proven by a loader that counts its own
    /// calls, so this gates the CACHE rather than the loader.
    #[test]
    fn a_second_load_of_the_same_checkpoint_hits() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = std::env::temp_dir().join(format!("mm3-wc-hit-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.safetensors"), b"stand-in checkpoint bytes").unwrap();
        let path = dir.to_string_lossy().into_owned();

        let loads = AtomicUsize::new(0);
        let load = || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(ConditionEncoderWeights { layer_weight_logits: vec![1.0], layer_scale: 1.0, proj_weight: vec![2.0; 8], proj_bias: vec![0.0] })
        };
        let unwrap = |h: &Held| match h {
            Held::ConditionEncoder(w) => Some(w.clone()),
            _ => None,
        };
        let first = get_or_load(Role::ConditionEncoder, &path, 64, load, Held::ConditionEncoder, unwrap).unwrap();
        let second = get_or_load(Role::ConditionEncoder, &path, 64, load, Held::ConditionEncoder, unwrap).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1, "the second call must not have re-read the checkpoint");
        assert!(Arc::ptr_eq(&first, &second), "a hit must hand back the SAME allocation, not an equal copy");

        // Clearing releases it, and the next call re-loads - the property
        // that makes `demote` safe: an eviction costs time, never a number.
        clear();
        let third = get_or_load(Role::ConditionEncoder, &path, 64, load, Held::ConditionEncoder, unwrap).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert_eq!(third.proj_weight, first.proj_weight, "a re-load must reproduce the same weights");
        clear();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The local ceiling evicts by `CostAware` and never evicts the entry
    /// that was just inserted (which would make the insert a no-op that
    /// still counted as a miss).
    #[test]
    fn the_local_budget_evicts_the_cheap_cold_entry_and_never_the_new_one() {
        let mut inner = Inner { tick: 100, ..Inner::default() };
        let key = |r: Role| (r, CheckpointId { path: format!("{r:?}"), bytes: 1, mtime: 1 });
        let slot = |bytes: u64, uses: u64, last_use: u64| Slot { held: Held::ConditionEncoder(Arc::new(ConditionEncoderWeights { layer_weight_logits: Vec::new(), layer_scale: 0.0, proj_weight: Vec::new(), proj_bias: Vec::new() })), bytes, uses, last_use };
        // A small, cold, once-used entry and a large, hot, recent one.
        inner.slots.insert(key(Role::Vocoder), slot(200 << 20, 1, 1));
        inner.slots.insert(key(Role::Dit), slot(9 << 30, 50, 99));
        inner.slots.insert(key(Role::DepthDecoder), slot(2 << 30, 5, 100));
        inner.bytes = (200 << 20) + (9u64 << 30) + (2u64 << 30);

        evict_to(&mut inner, (11u64 << 30) + (100 << 20), &key(Role::DepthDecoder));
        assert!(!inner.slots.contains_key(&key(Role::Vocoder)), "the cheap cold entry must go first");
        assert!(inner.slots.contains_key(&key(Role::Dit)), "the expensive hot entry must survive");
        assert!(inner.slots.contains_key(&key(Role::DepthDecoder)), "the just-inserted entry is never its own victim");
        assert_eq!(inner.evictions, 1);
        assert_eq!(inner.bytes, (9u64 << 30) + (2u64 << 30));
    }
}
