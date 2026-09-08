// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Memoized dispatch records: record a `Step` once, re-submit it forever.**
//!
//! Every backend's `step()` builds a per-dispatch object before anything runs
//! - on wgpu a fresh uniform buffer plus a fresh bind group, both host-side
//! driver calls costing tens of microseconds each. That is free for a model
//! that records its tape ONCE at construction (most of this workspace), and it
//! is the dominant cost for one that re-records per step: an incremental
//! decoder rebuilds its whole tape every token, and a sparse-MoE decoder's
//! tape is thousands of dispatches wide because it dispatches every expert
//! whether or not the router picked it.
//!
//! `backend-wgpu::WgpuBackend::step`'s own doc already names the remedy -
//! "for hot loops prefer `uniform_dynamic` + `step_buf` so the uniform/bind
//! group are allocated once and reused" - but taking it means every shared
//! builder in `model::block`/`model::moe`/`model::vit` growing a second,
//! caller-owned-uniform form, and every model rewriting its loop against it.
//! This module gets the same result underneath all of them: a dispatch is a
//! pure function of `(kernel, buffers, params, threads)`, so the second time a
//! caller asks for one it can have the first one back.
//!
//! ## Why the answer is exact, not approximate
//!
//! The key is every input `Gpu::step` has. Two calls with the same kernel
//! slot, the same buffers in the same order, the same `params` words and the
//! same thread count cannot produce different dispatches - the backend has
//! nothing else to read. So a hit returns the dispatch the miss would have
//! built, bit for bit, and a cached run's OUTPUT is identical to an uncached
//! one rather than merely close. Values the dispatch reads from those buffers
//! are free to change between submits: a bind group names buffers, never
//! their contents.
//!
//! ## Why an entry pins its buffers
//!
//! Buffer identity is [`backend_api::DeviceBuffer::alloc_id`], the `Arc`
//! address - which a later allocation may reuse once the last handle drops.
//! An entry therefore holds a strong handle to every buffer in its own key, so
//! no id in the map can name freed memory, and eviction drops the key and the
//! handles together. The consequence is the reason this is OPT-IN and capped
//! (see [`super::Gpu::enable_step_cache`]): a caller that churns fresh
//! buffers would keep them all alive.

use std::collections::HashMap;

use backend_api::DeviceBuffer;

use crate::Step;

/// Everything [`super::Gpu::step`] passes the backend, in one hashable value.
///
/// `bufs` holds `alloc_id`s as `usize` rather than the pointers themselves:
/// the value is an identity to compare, never dereferenced, and a raw pointer
/// would make the map neither `Send` nor `Sync`.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    kind: usize,
    threads: u32,
    bufs: Box<[usize]>,
    params: Box<[u32]>,
}

struct Entry {
    step: Step,
    /// Whether this dispatch was ever asked for a second time - what
    /// [`StepCache::put`]'s eviction keeps.
    reused: bool,
    /// Strong handles to exactly the buffers [`Key::bufs`] names - see this
    /// module's doc. Never read; held so an `alloc_id` cannot be recycled
    /// under a live key.
    _pinned: Box<[DeviceBuffer]>,
}

/// A bounded memo of recorded dispatches. Not public: reached through
/// [`super::Gpu`], which owns the arming flag and the lock.
pub(crate) struct StepCache {
    cap: usize,
    map: HashMap<Key, Entry>,
    hits: u64,
    misses: u64,
}

impl StepCache {
    pub(crate) fn new(cap: usize) -> StepCache {
        StepCache { cap: cap.max(1), map: HashMap::new(), hits: 0, misses: 0 }
    }

    fn key(kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Key {
        Key {
            kind,
            threads,
            bufs: bufs.iter().map(|b| b.alloc_id() as usize).collect(),
            params: params.into(),
        }
    }

    /// The dispatch recorded for this exact call before, if there was one.
    pub(crate) fn get(&mut self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Option<Step> {
        match self.map.get_mut(&Self::key(kind, bufs, params, threads)) {
            Some(e) => {
                e.reused = true;
                self.hits += 1;
                Some(e.step.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Remember `step` for this call.
    ///
    /// At capacity, entries that have never been asked for twice are dropped
    /// and the ones that have are kept. That is the whole eviction policy, and
    /// it is self-tuning for the shape this exists for: a decode step's tape
    /// is overwhelmingly dispatches that repeat verbatim every step (they get
    /// hit, so they stay) plus a thin seam that carries the position and can
    /// never repeat (never hit, so it is what gets dropped). If keeping the
    /// reused ones still leaves no room, the map is emptied - a working set
    /// larger than the cap is not a cache workload, and refilling costs
    /// exactly what running uncached would have.
    ///
    /// Either way a key and the strong handles pinning its buffers die
    /// together, which is what keeps the `alloc_id` key sound.
    pub(crate) fn put(&mut self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32, step: &Step) {
        if self.map.len() >= self.cap {
            self.map.retain(|_, e| e.reused);
            if self.map.len() >= self.cap {
                self.map.clear();
            }
        }
        self.map.insert(
            Self::key(kind, bufs, params, threads),
            Entry { step: step.clone(), reused: false, _pinned: bufs.iter().map(|b| (*b).clone()).collect() },
        );
    }

    /// `(hits, misses, live entries)` - what a test asserts against to show a
    /// replayed tape is actually being replayed.
    pub(crate) fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.map.len())
    }
}
