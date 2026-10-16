// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for agent workloads that resend long
// conversation histories to large language models for its clients. If your team
// needs expertise in prefix reuse for hybrid recurrent/attention models then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The prompt-prefix cache of the released-GGUF engine: the bookkeeping only -
//! which prompts are cached, which snapshot row and KV block each owns, who is
//! evicted - with no device in it, so it is unit-tested directly.
//!
//! # What a cache entry is
//!
//! An agent resends the same long history every turn; prefilling it again costs
//! tens of seconds on the 27B. A full-attention model reuses KV rows for that.
//! This model is 3:1 Gated-DeltaNet, whose layers keep a recurrent STATE, not
//! rows: a prefix can be resumed only from the state exactly AFTER the prefix.
//! So an entry is the pair that makes a prefix resumable:
//!
//! * the **KV rows** `0..L` of the sequence that prefilled it. Decoding only
//!   appends rows past the prompt, so these stay valid for as long as the
//!   sequence's physical block is kept - the entry simply retains the block
//!   (`table`) when the sequence ends instead of freeing it;
//! * a **snapshot** of every GDN layer's state (and conv history) taken the
//!   moment the prefill ended, before the first decode step advanced it: a row
//!   of the engine's snapshot pool (`snap_row`);
//! * the last prompt token's **hidden row**, so a prompt that is exactly the
//!   cached one needs no compute at all.
//!
//! A new prompt that starts with a cached prompt copies the KV rows into its own
//! block, restores the snapshot and prefills only the rest. Entries are
//! exact-token prefixes (no hashing, no false sharing between prompts that
//! merely look alike), at most `capacity` of them, evicted least recently used.

use model::paged::BlockTable;

/// Prompts shorter than this are not worth a snapshot row and a retained block.
pub(super) const MIN_CACHED_TOKENS: usize = 64;

/// One cached prompt.
pub(super) struct Entry {
    pub tokens: Vec<u32>,
    /// Owns the physical block holding KV rows `0..tokens.len()`.
    pub table: BlockTable,
    pub snap_row: u32,
    /// The last prompt token's `[d_model]` hidden row.
    pub last_hidden: Vec<f32>,
    stamp: u64,
}

impl Entry {
    pub fn new(tokens: Vec<u32>, table: BlockTable, snap_row: u32, last_hidden: Vec<f32>) -> Entry {
        Entry { tokens, table, snap_row, last_hidden, stamp: 0 }
    }
}

/// The cache: entries plus the pool of snapshot rows they and in-flight
/// prefills draw from.
pub(super) struct PrefixCache {
    entries: Vec<Entry>,
    free_rows: Vec<u32>,
    tick: u64,
    hit_tokens: u64,
    lookup_tokens: u64,
}

impl PrefixCache {
    /// A cache over `snapshot_rows` snapshot rows (0 disables it).
    pub fn new(snapshot_rows: u32) -> PrefixCache {
        PrefixCache { entries: Vec::new(), free_rows: (0..snapshot_rows).rev().collect(), tick: 0, hit_tokens: 0, lookup_tokens: 0 }
    }

    pub fn enabled(&self) -> bool {
        !self.entries.is_empty() || !self.free_rows.is_empty()
    }

    /// The entry holding the longest cached prefix of `prompt` (an entry equal
    /// to the whole prompt qualifies), refreshed as most recently used.
    /// Counts the lookup for [`Self::stats`].
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<usize> {
        self.lookup_tokens += prompt.len() as u64;
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tokens.len() <= prompt.len() && prompt[..e.tokens.len()] == e.tokens[..])
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)?;
        self.tick += 1;
        self.entries[best].stamp = self.tick;
        self.hit_tokens += self.entries[best].tokens.len() as u64;
        Some(best)
    }

    pub fn entry(&self, i: usize) -> &Entry {
        &self.entries[i]
    }

    /// A snapshot row for a prefill that is about to finish, evicting the least
    /// recently used entry when none is free. The evicted entry's block table
    /// is returned for the caller to release; `None` row when every row is
    /// held by a prefill still in flight.
    pub fn take_row(&mut self) -> (Option<u32>, Option<BlockTable>) {
        if let Some(row) = self.free_rows.pop() {
            return (Some(row), None);
        }
        match self.entries.iter().enumerate().min_by_key(|(_, e)| e.stamp).map(|(i, _)| i) {
            Some(i) => {
                let e = self.entries.remove(i);
                (Some(e.snap_row), Some(e.table))
            }
            None => (None, None),
        }
    }

    /// Add `entry`. Entries it supersedes - a cached prompt that is a prefix of
    /// it, which an agent conversation growing turn by turn always leaves
    /// behind - are removed, and an identical prompt keeps only the newer one.
    /// Returns the retired entries' tables and the rows they free.
    pub fn insert(&mut self, mut entry: Entry) -> Vec<BlockTable> {
        let mut retired = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            if e.tokens.len() <= entry.tokens.len() && entry.tokens[..e.tokens.len()] == e.tokens[..] {
                let e = self.entries.remove(i);
                self.free_rows.push(e.snap_row);
                retired.push(e.table);
            } else {
                i += 1;
            }
        }
        self.tick += 1;
        entry.stamp = self.tick;
        self.entries.push(entry);
        retired
    }

    /// Evict the least recently used entry, returning its table (to free a
    /// block). `None` when the cache is empty.
    pub fn evict_lru(&mut self) -> Option<BlockTable> {
        let i = self.entries.iter().enumerate().min_by_key(|(_, e)| e.stamp).map(|(i, _)| i)?;
        let e = self.entries.remove(i);
        self.free_rows.push(e.snap_row);
        Some(e.table)
    }

    /// `(hit tokens, looked-up tokens, cached prompts)`, the shape
    /// `PagedDecoder::prefix_stats` reports.
    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hit_tokens, self.lookup_tokens, self.entries.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tokens: &[u32], row: u32) -> Entry {
        Entry::new(tokens.to_vec(), BlockTable::new(), row, vec![0.0])
    }

    /// The longest cached prefix wins, an exact-length match counts, and a
    /// prompt that merely shares a start with an entry (not all of it) does not.
    #[test]
    fn lookup_finds_the_longest_cached_prefix_only() {
        let mut c = PrefixCache::new(4);
        let r1 = c.take_row().0.unwrap();
        assert!(c.insert(entry(&[1, 2, 3], r1)).is_empty());
        let r2 = c.take_row().0.unwrap();
        // Not a prefix of the first: a different conversation.
        c.insert(entry(&[1, 2, 9, 9], r2));
        assert_eq!(c.lookup(&[1, 2, 3, 4, 5]).map(|i| c.entry(i).tokens.len()), Some(3));
        assert_eq!(c.lookup(&[1, 2, 3]).map(|i| c.entry(i).tokens.len()), Some(3), "an identical prompt hits");
        assert!(c.lookup(&[1, 2]).is_none(), "a prompt shorter than the entry cannot reuse it");
        assert!(c.lookup(&[7, 7, 7]).is_none());
    }

    /// A conversation that grows replaces its own earlier entry, freeing the
    /// snapshot row and handing back the block to release.
    #[test]
    fn a_longer_prompt_supersedes_the_prefix_it_extends() {
        let mut c = PrefixCache::new(2);
        let r1 = c.take_row().0.unwrap();
        c.insert(entry(&[1, 2], r1));
        let r2 = c.take_row().0.unwrap();
        let retired = c.insert(entry(&[1, 2, 3, 4], r2));
        assert_eq!(retired.len(), 1, "the shorter entry's block is returned for release");
        assert_eq!(c.stats().2, 1);
        assert_eq!(c.take_row().0, Some(r1), "its snapshot row is free again");
    }

    /// With every row in use the least recently used entry is evicted, and a
    /// lookup counts as a use.
    #[test]
    fn the_least_recently_used_entry_is_evicted_for_a_row() {
        let mut c = PrefixCache::new(2);
        for (tokens, _) in [([1u32, 1], 0), ([2, 2], 1)] {
            let r = c.take_row().0.unwrap();
            c.insert(entry(&tokens, r));
        }
        c.lookup(&[1, 1, 5]); // [1,1] is now the most recently used
        let (row, evicted) = c.take_row();
        assert!(row.is_some() && evicted.is_some());
        assert!(c.lookup(&[2, 2, 5]).is_none(), "[2,2] was the least recently used and is gone");
        assert!(c.lookup(&[1, 1, 5]).is_some());
    }

    #[test]
    fn a_disabled_cache_hands_out_no_rows() {
        let mut c = PrefixCache::new(0);
        assert!(!c.enabled());
        let (row, evicted) = c.take_row();
        assert!(row.is_none() && evicted.is_none());
    }

    #[test]
    fn stats_count_hit_and_looked_up_tokens() {
        let mut c = PrefixCache::new(1);
        let r = c.take_row().0.unwrap();
        c.insert(entry(&[1, 2, 3], r));
        c.lookup(&[1, 2, 3, 4]);
        c.lookup(&[9, 9]);
        assert_eq!(c.stats(), (3, 6, 1));
    }
}
