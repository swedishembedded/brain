// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A byte-bounded LRU of decoder memories, so that describing and then
//! extracting the same image runs the vision half once.
//!
//! The key names everything the memory depends on: the image's content, the
//! preprocessing version, the model revision and the proposal options. Two
//! requests that differ in any of them never share an entry. A cache belongs
//! to one served instance and is never handed to another, so entries cannot
//! cross adapters or tenants that were loaded separately.
//!
//! Swedish Embedded AB implements solutions for avoiding repeated work in
//! perception pipelines for its clients. If your team needs expertise in
//! caching intermediate features of vision models then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::device::DecoderMemory;

/// What a decoder memory is a function of.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryKey {
    /// SHA-256 of the image dimensions and pixels.
    content: [u8; 32],
    /// Version of the preprocessing that produced the memory.
    preprocessing: u32,
    /// Revision of the weights (their SHA-256).
    revision: Arc<str>,
    /// Bits of `conf` and `text_conf`, then `max_objects` and `max_text_lines` (`usize::MAX` for unset).
    proposals: [u64; 4],
}

impl MemoryKey {
    /// The key of an `h x w` RGB image under the proposal options of a request.
    pub fn new(rgb: &[u8], h: usize, w: usize, preprocessing: u32, revision: &Arc<str>, conf: f32, text_conf: f32, max_objects: usize, max_text: usize) -> MemoryKey {
        let mut hasher = Sha256::new();
        hasher.update((h as u64).to_le_bytes());
        hasher.update((w as u64).to_le_bytes());
        hasher.update(rgb);
        MemoryKey { content: hasher.finalize().into(), preprocessing, revision: revision.clone(), proposals: [conf.to_bits() as u64, text_conf.to_bits() as u64, max_objects as u64, max_text as u64] }
    }
}

/// Least-recently-used entries, evicted until the held bytes fit the bound.
pub struct MemoryCache {
    capacity: usize,
    held: usize,
    tick: u64,
    entries: HashMap<MemoryKey, (Arc<DecoderMemory>, u64)>,
    /// Use time to key, oldest first.
    order: BTreeMap<u64, MemoryKey>,
    hits: u64,
    misses: u64,
}

impl MemoryCache {
    /// A cache holding at most `capacity` bytes of memory rows; `0` stores nothing.
    pub fn new(capacity: usize) -> MemoryCache {
        MemoryCache { capacity, held: 0, tick: 0, entries: HashMap::new(), order: BTreeMap::new(), hits: 0, misses: 0 }
    }

    /// The entry for `key`, marked as just used.
    pub fn get(&mut self, key: &MemoryKey) -> Option<Arc<DecoderMemory>> {
        let Some((memory, used)) = self.entries.get_mut(key) else {
            self.misses += 1;
            return None;
        };
        self.hits += 1;
        self.order.remove(used);
        self.tick += 1;
        *used = self.tick;
        self.order.insert(self.tick, key.clone());
        Some(memory.clone())
    }

    /// Store `memory` under `key`, evicting the least recently used entries to stay within the bound. An entry
    /// larger than the whole bound is not stored.
    pub fn put(&mut self, key: MemoryKey, memory: Arc<DecoderMemory>) {
        let size = memory.bytes();
        if size > self.capacity {
            return;
        }
        if let Some((old, used)) = self.entries.remove(&key) {
            self.order.remove(&used);
            self.held -= old.bytes();
        }
        while self.held + size > self.capacity {
            let Some((_, oldest)) = self.order.pop_first() else { break };
            if let Some((evicted, _)) = self.entries.remove(&oldest) {
                self.held -= evicted.bytes();
            }
        }
        self.tick += 1;
        self.order.insert(self.tick, key.clone());
        self.entries.insert(key, (memory, self.tick));
        self.held += size;
    }

    /// Bytes of memory rows held.
    pub fn held_bytes(&self) -> usize {
        self.held
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Lookups answered from the cache, and lookups that were not.
    pub fn hits_and_misses(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(floats: usize) -> Arc<DecoderMemory> {
        Arc::new(DecoderMemory { rows: vec![0.5; floats], dim: 1 })
    }

    fn key(pixel: u8, conf: f32) -> MemoryKey {
        MemoryKey::new(&[pixel; 12], 2, 2, 1, &Arc::from("rev"), conf, 0.3, usize::MAX, usize::MAX)
    }

    #[test]
    fn the_key_names_content_revision_version_and_proposals() {
        let rev: Arc<str> = Arc::from("a");
        let base = MemoryKey::new(&[1; 12], 2, 2, 1, &rev, 0.3, 0.3, 64, 32);
        assert_eq!(base, MemoryKey::new(&[1; 12], 2, 2, 1, &rev, 0.3, 0.3, 64, 32));
        assert_ne!(base, MemoryKey::new(&[2; 12], 2, 2, 1, &rev, 0.3, 0.3, 64, 32), "content");
        assert_ne!(base, MemoryKey::new(&[1; 12], 2, 2, 2, &rev, 0.3, 0.3, 64, 32), "preprocessing version");
        assert_ne!(base, MemoryKey::new(&[1; 12], 2, 2, 1, &Arc::from("b"), 0.3, 0.3, 64, 32), "model revision");
        assert_ne!(base, MemoryKey::new(&[1; 12], 2, 2, 1, &rev, 0.4, 0.3, 64, 32), "conf");
        assert_ne!(base, MemoryKey::new(&[1; 12], 2, 2, 1, &rev, 0.3, 0.3, 8, 32), "max_objects");
        assert_ne!(MemoryKey::new(&[1; 12], 1, 4, 1, &rev, 0.3, 0.3, 64, 32), MemoryKey::new(&[1; 12], 4, 1, 1, &rev, 0.3, 0.3, 64, 32), "shape");
    }

    #[test]
    fn the_least_recently_used_entry_goes_first_and_the_bound_holds() {
        let mut cache = MemoryCache::new(3 * 4 * 10);
        for p in 0..3u8 {
            cache.put(key(p, 0.3), memory(10));
        }
        assert_eq!((cache.len(), cache.held_bytes()), (3, 120));
        assert!(cache.get(&key(0, 0.3)).is_some(), "touch the oldest");
        cache.put(key(3, 0.3), memory(10));
        assert_eq!(cache.len(), 3);
        assert!(cache.get(&key(1, 0.3)).is_none(), "entry 1 was the least recently used");
        assert!(cache.get(&key(0, 0.3)).is_some() && cache.get(&key(2, 0.3)).is_some() && cache.get(&key(3, 0.3)).is_some());
        assert!(cache.held_bytes() <= 120);
    }

    #[test]
    fn an_entry_larger_than_the_bound_is_not_stored_and_a_zero_bound_stores_nothing() {
        let mut cache = MemoryCache::new(39);
        cache.put(key(0, 0.3), memory(10));
        assert!(cache.is_empty());
        let mut off = MemoryCache::new(0);
        off.put(key(0, 0.3), memory(1));
        assert!(off.is_empty());
    }

    #[test]
    fn replacing_an_entry_does_not_double_count_its_bytes() {
        let mut cache = MemoryCache::new(1000);
        cache.put(key(0, 0.3), memory(10));
        cache.put(key(0, 0.3), memory(20));
        assert_eq!((cache.len(), cache.held_bytes()), (1, 80));
        assert_eq!(cache.hits_and_misses(), (0, 0));
        assert!(cache.get(&key(9, 0.3)).is_none());
        assert_eq!(cache.hits_and_misses(), (0, 1));
    }
}
