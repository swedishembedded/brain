// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for agent workloads that resend long
// conversation histories to large language models for its clients. If your team
// needs expertise in prefix reuse for hybrid recurrent/attention models then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The snapshot prompt-prefix cache: the bookkeeping only -
//! which prompts are cached, which snapshot row and KV block each owns, who is
//! evicted - with no device in it, so it is unit-tested directly.
//!
//! # What a cache entry is
//!
//! Shared by every engine whose layers keep a recurrent state (Gated-DeltaNet
//! and the like): the entry is keyed like the block cache ([`crate::paged::PrefixCache`]) -
//! namespace (adapter identity) plus exact token prefix - and evicted by the same
//! recency rule, but resumes from a state snapshot instead of by sharing blocks
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
//! A new prompt that starts with a cached prompt gets the KV rows (copied into its
//! own block, or - with paged KV - by sharing the entry's blocks), restores the snapshot and prefills only the rest. Entries are
//! exact-token prefixes (no hashing, no false sharing between prompts that
//! merely look alike), at most `capacity` of them, evicted least recently used.
//!
//! An entry belongs to the KV namespace of the sequence that prefilled it (its
//! `table`'s): K/V computed under one adapter are not the base's or another
//! adapter's, so a prompt only ever resumes from an entry of its own namespace.

use crate::paged::{BlockTable, KvNamespace};

/// One cached prompt.
pub struct Entry<P> {
    pub tokens: Vec<u32>,
    /// Owns the physical block holding KV rows `0..tokens.len()`.
    pub table: BlockTable,
    pub snap_row: u32,
    /// What the engine needs to resume from this entry besides the KV rows and
    /// the snapshot (the last prompt token's hidden row, which drafter rows are
    /// valid, ...): opaque to the cache.
    pub payload: P,
    stamp: u64,
    /// A branch point ([`SnapshotPrefixCache::branch_point`]): a prefix
    /// different prompts continue, kept while they extend it.
    branch: bool,
}

impl<P> Entry<P> {
    pub fn new(tokens: Vec<u32>, table: BlockTable, snap_row: u32, payload: P) -> Entry<P> {
        Entry { tokens, table, snap_row, payload, stamp: 0, branch: false }
    }

    /// This entry as a branch point: the prompts that extend it are other
    /// continuations of a shared prefix, not later turns replacing it, so it is
    /// neither superseded by them nor taken as their earlier turn's row; it
    /// leaves the cache only by recency.
    pub fn branch(mut self) -> Entry<P> {
        self.branch = true;
        self
    }

    /// Whether `self` is a cached prefix of `prompt` in `namespace` that a turn
    /// extending it replaces. A branch point is replaced only by the same prefix
    /// filed again (two prompts that left one document at once both file it).
    fn superseded_by(&self, namespace: KvNamespace, prompt: &[u32]) -> bool {
        let prefix = self.table.kv_namespace() == namespace && self.tokens.len() <= prompt.len() && prompt[..self.tokens.len()] == self.tokens[..];
        prefix && (!self.branch || self.tokens.len() == prompt.len())
    }
}

/// The cache: entries plus the pool of snapshot rows they and in-flight
/// prefills draw from.
pub struct SnapshotPrefixCache<P> {
    entries: Vec<Entry<P>>,
    free_rows: Vec<u32>,
    tick: u64,
    hit_tokens: u64,
    lookup_tokens: u64,
}

impl<P> SnapshotPrefixCache<P> {
    /// A cache over `snapshot_rows` snapshot rows (0 disables it).
    pub fn new(snapshot_rows: u32) -> SnapshotPrefixCache<P> {
        SnapshotPrefixCache { entries: Vec::new(), free_rows: (0..snapshot_rows).rev().collect(), tick: 0, hit_tokens: 0, lookup_tokens: 0 }
    }

    pub fn enabled(&self) -> bool {
        !self.entries.is_empty() || !self.free_rows.is_empty()
    }

    /// The entry [`Self::lookup`] would pick, without counting the lookup or refreshing the entry.
    pub fn peek(&self, namespace: KvNamespace, prompt: &[u32]) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.table.kv_namespace() == namespace && e.tokens.len() <= prompt.len() && prompt[..e.tokens.len()] == e.tokens[..])
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
    }

    /// The entry holding the longest cached prefix of `prompt` (an entry equal
    /// to the whole prompt qualifies), refreshed as most recently used.
    /// Counts the lookup for [`Self::stats`].
    pub fn lookup(&mut self, namespace: KvNamespace, prompt: &[u32]) -> Option<usize> {
        self.lookup_tokens += prompt.len() as u64;
        let best = self.peek(namespace, prompt)?;
        self.tick += 1;
        self.entries[best].stamp = self.tick;
        self.hit_tokens += self.entries[best].tokens.len() as u64;
        Some(best)
    }

    pub fn entry(&self, i: usize) -> &Entry<P> {
        &self.entries[i]
    }

    /// The payload of the entry for exactly `tokens` in `namespace`, to update
    /// what became true of it after it was filed (`None` once it is gone).
    pub fn payload_mut(&mut self, namespace: KvNamespace, tokens: &[u32]) -> Option<&mut P> {
        self.entries.iter_mut().find(|e| e.table.kv_namespace() == namespace && e.tokens == tokens).map(|e| &mut e.payload)
    }

    /// Where `prompt` leaves the cached prompt it shares the most with, floored
    /// to a multiple of `align`, when that is at least `min` tokens and longer
    /// than any cached prefix of `prompt`; `None` otherwise.
    ///
    /// A recurrent layer resumes only from the state exactly after a cached
    /// prefix, so two prompts that share a long document and then diverge leave
    /// nothing the next prompt sharing that document can resume from: each
    /// entry ends inside its own continuation. A prefill that files the state
    /// at the branch point as an entry ([`Entry::branch`]) turns the document
    /// into a prefix every later such prompt resumes from.
    pub fn branch_point(&self, namespace: KvNamespace, prompt: &[u32], align: usize, min: usize) -> Option<usize> {
        let resumable = self.peek(namespace, prompt).map_or(0, |i| self.entries[i].tokens.len());
        let shared = self
            .entries
            .iter()
            .filter(|e| e.table.kv_namespace() == namespace)
            .map(|e| e.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count())
            .max()?;
        let at = shared / align.max(1) * align.max(1);
        (at >= min && at > resumable && at < prompt.len()).then_some(at)
    }

    /// A snapshot row for a prefill of `prompt` (in `namespace`) that is about
    /// to finish: a free one, else the row of an entry `prompt` supersedes (a
    /// cached prefix of it: the earlier turn of the same conversation, which
    /// [`Self::insert`] retires anyway), else the least recently used entry's.
    ///
    /// Taking the superseded entry before an unrelated one matters when the
    /// concurrent conversations fill the cache: every conversation holds its
    /// previous turn's entry while the next turn runs, so evicting the least
    /// recently used one drops a conversation whose next turn is about to
    /// arrive, and that turn then prefills its whole history again (round-robin
    /// turns defeat LRU exactly when conversations = rows).
    ///
    /// The evicted entry's block table is returned for the caller to release;
    /// `None` row when every row is held by a prefill still in flight.
    pub fn take_row(&mut self, namespace: KvNamespace, prompt: &[u32]) -> (Option<u32>, Option<BlockTable>) {
        if let Some(row) = self.free_rows.pop() {
            return (Some(row), None);
        }
        let superseded = self.entries.iter().position(|e| e.superseded_by(namespace, prompt));
        let victim = superseded.or_else(|| self.entries.iter().enumerate().min_by_key(|(_, e)| e.stamp).map(|(i, _)| i));
        match victim {
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
    pub fn insert(&mut self, mut entry: Entry<P>) -> Vec<BlockTable> {
        let mut retired = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            if self.entries[i].superseded_by(entry.table.kv_namespace(), &entry.tokens) {
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

    /// Drop every entry of `namespace` (its adapter was unloaded), returning
    /// their tables for the caller to release.
    pub fn evict_namespace(&mut self, namespace: KvNamespace) -> Vec<BlockTable> {
        let (gone, kept): (Vec<Entry<P>>, Vec<Entry<P>>) = std::mem::take(&mut self.entries).into_iter().partition(|e| e.table.kv_namespace() == namespace);
        self.entries = kept;
        self.free_rows.extend(gone.iter().map(|e| e.snap_row));
        gone.into_iter().map(|e| e.table).collect()
    }

    /// Evict the least recently used entry, returning its table (to free a
    /// block). `None` when the cache is empty.
    pub fn evict_lru(&mut self) -> Option<BlockTable> {
        self.evict_lru_sparing(None)
    }

    /// [`Self::evict_lru`] sparing the entry [`Self::lookup`] would pick for `prompt` in `namespace`: the
    /// admission of that prompt is about to resume from it. `None` when nothing else is cached.
    pub fn evict_lru_keeping(&mut self, namespace: KvNamespace, prompt: &[u32]) -> Option<BlockTable> {
        let keep = self.peek(namespace, prompt);
        self.evict_lru_sparing(keep)
    }

    fn evict_lru_sparing(&mut self, keep: Option<usize>) -> Option<BlockTable> {
        let i = self.entries.iter().enumerate().filter(|(i, _)| Some(*i) != keep).min_by_key(|(_, e)| e.stamp).map(|(i, _)| i)?;
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
    use crate::paged::BASE_KV_NAMESPACE;

    fn entry(tokens: &[u32], row: u32) -> Entry<u32> {
        Entry::new(tokens.to_vec(), BlockTable::new(), row, 0)
    }

    /// The longest cached prefix wins, an exact-length match counts, and a
    /// prompt that merely shares a start with an entry (not all of it) does not.
    #[test]
    fn lookup_finds_the_longest_cached_prefix_only() {
        let mut c = SnapshotPrefixCache::<u32>::new(4);
        let r1 = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        assert!(c.insert(entry(&[1, 2, 3], r1)).is_empty());
        let r2 = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        // Not a prefix of the first: a different conversation.
        c.insert(entry(&[1, 2, 9, 9], r2));
        assert_eq!(c.lookup(BASE_KV_NAMESPACE, &[1, 2, 3, 4, 5]).map(|i| c.entry(i).tokens.len()), Some(3));
        assert_eq!(c.lookup(BASE_KV_NAMESPACE, &[1, 2, 3]).map(|i| c.entry(i).tokens.len()), Some(3), "an identical prompt hits");
        assert!(c.lookup(BASE_KV_NAMESPACE, &[1, 2]).is_none(), "a prompt shorter than the entry cannot reuse it");
        assert!(c.lookup(BASE_KV_NAMESPACE, &[7, 7, 7]).is_none());
    }

    /// Two prompts that share a document and then diverge leave no entry the next such prompt can resume from:
    /// every entry ends past the shared part. The branch point is where they part, on the alignment, and only when
    /// it is a prefix the cache cannot already resume from.
    #[test]
    fn the_branch_point_is_where_a_prompt_leaves_a_cached_one() {
        let mut c = SnapshotPrefixCache::<u32>::new(4);
        let doc: Vec<u32> = (0..10).collect();
        let first: Vec<u32> = doc.iter().copied().chain([50, 51, 52]).collect();
        let r = c.take_row(BASE_KV_NAMESPACE, &first).0.unwrap();
        c.insert(entry(&first, r));
        let second: Vec<u32> = doc.iter().copied().chain([60, 61, 62]).collect();
        assert_eq!(c.branch_point(BASE_KV_NAMESPACE, &second, 4, 4), Some(8), "10 shared tokens, floored to the alignment");
        assert_eq!(c.branch_point(BASE_KV_NAMESPACE, &second, 4, 12), None, "shorter than the minimum is not worth a row");
        assert_eq!(c.branch_point(7, &second, 4, 4), None, "only entries of the prompt's own namespace count");
        let extended: Vec<u32> = first.iter().copied().chain([53]).collect();
        assert_eq!(c.branch_point(BASE_KV_NAMESPACE, &extended, 4, 4), None, "a prompt the entry is a prefix of resumes from it instead");
        assert_eq!(c.branch_point(BASE_KV_NAMESPACE, &[9, 9, 9, 9, 9, 9], 4, 4), None);

        // Filed, the branch entry serves every later prompt that shares the document.
        let r = c.take_row(BASE_KV_NAMESPACE, &second[..8]).0.unwrap();
        c.insert(entry(&second[..8], r).branch());
        let third: Vec<u32> = doc.iter().copied().chain([70]).collect();
        assert_eq!(c.peek(BASE_KV_NAMESPACE, &third).map(|i| c.entry(i).tokens.len()), Some(8));
        assert_eq!(c.branch_point(BASE_KV_NAMESPACE, &third, 4, 4), None, "the cache already resumes from 8");
        *c.payload_mut(BASE_KV_NAMESPACE, &second[..8]).expect("filed") = 5;
        assert_eq!(c.entry(c.peek(BASE_KV_NAMESPACE, &third).unwrap()).payload, 5);
        assert!(c.payload_mut(BASE_KV_NAMESPACE, &second[..7]).is_none(), "only the exact prefix names an entry");
    }

    /// A branch entry is shared by different continuations: a longer prompt it is a prefix of does not supersede it,
    /// and a finishing prefill does not take its row as its own earlier turn's.
    #[test]
    fn a_branch_entry_outlives_the_prompts_that_extend_it() {
        let mut c = SnapshotPrefixCache::<u32>::new(2);
        let r = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        c.insert(entry(&[1, 2, 3, 4], r).branch());
        let r = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        assert!(c.insert(entry(&[1, 2, 3, 4, 5, 6], r)).is_empty(), "the branch entry is not retired");
        assert_eq!(c.stats().2, 2);
        let (row, evicted) = c.take_row(BASE_KV_NAMESPACE, &[1, 2, 3, 4, 5, 6, 7]);
        assert!(row.is_some() && evicted.is_some());
        assert_eq!(c.peek(BASE_KV_NAMESPACE, &[1, 2, 3, 4, 9]).map(|i| c.entry(i).tokens.len()), Some(4), "the conversation's own entry went, not the branch");
        assert_eq!(c.insert(entry(&[1, 2, 3, 4], row.unwrap()).branch()).len(), 1, "the same branch point filed twice is kept once");
    }

    /// A conversation that grows replaces its own earlier entry, freeing the
    /// snapshot row and handing back the block to release.
    #[test]
    fn a_longer_prompt_supersedes_the_prefix_it_extends() {
        let mut c = SnapshotPrefixCache::<u32>::new(2);
        let r1 = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        c.insert(entry(&[1, 2], r1));
        let r2 = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        let retired = c.insert(entry(&[1, 2, 3, 4], r2));
        assert_eq!(retired.len(), 1, "the shorter entry's block is returned for release");
        assert_eq!(c.stats().2, 1);
        assert_eq!(c.take_row(BASE_KV_NAMESPACE, &[]).0, Some(r1), "its snapshot row is free again");
    }

    /// With every row in use the least recently used entry is evicted, and a
    /// lookup counts as a use.
    #[test]
    fn the_least_recently_used_entry_is_evicted_for_a_row() {
        let mut c = SnapshotPrefixCache::<u32>::new(2);
        for (tokens, _) in [([1u32, 1], 0), ([2, 2], 1)] {
            let r = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
            c.insert(entry(&tokens, r));
        }
        c.lookup(BASE_KV_NAMESPACE, &[1, 1, 5]); // [1,1] is now the most recently used
        let (row, evicted) = c.take_row(BASE_KV_NAMESPACE, &[]);
        assert!(row.is_some() && evicted.is_some());
        assert!(c.lookup(BASE_KV_NAMESPACE, &[2, 2, 5]).is_none(), "[2,2] was the least recently used and is gone");
        assert!(c.lookup(BASE_KV_NAMESPACE, &[1, 1, 5]).is_some());
    }

    /// Round-robin turns of as many conversations as there are rows: a finishing
    /// turn takes the row of its own earlier turn, not the least recently used
    /// entry, which belongs to the conversation whose turn comes next.
    #[test]
    fn a_turn_takes_its_own_earlier_entry_before_another_conversations() {
        let mut c = SnapshotPrefixCache::<u32>::new(2);
        let (a1, b1) = ([1u32, 1, 1], [2u32, 2, 2]);
        for tokens in [a1, b1] {
            let r = c.take_row(BASE_KV_NAMESPACE, &tokens).0.unwrap();
            c.insert(entry(&tokens, r));
        }
        for turn in 2..6u32 {
            for conv in [1u32, 2] {
                let prompt: Vec<u32> = std::iter::repeat_n(conv, 3).chain(std::iter::repeat_n(9, turn as usize)).collect();
                assert!(c.lookup(BASE_KV_NAMESPACE, &prompt).is_some(), "turn {turn} of conversation {conv} resumes from its previous turn");
                let (row, evicted) = c.take_row(BASE_KV_NAMESPACE, &prompt);
                assert!(row.is_some() && evicted.is_some());
                c.insert(entry(&prompt, row.unwrap()));
            }
        }
        assert_eq!(c.stats().2, 2, "one entry per conversation");
    }

    /// K/V computed under an adapter are not the base's: a prompt resumes only
    /// from an entry of its own namespace, and unloading an adapter drops
    /// exactly its entries and frees their rows.
    #[test]
    fn entries_belong_to_their_adapters_namespace() {
        let mut c = SnapshotPrefixCache::<u32>::new(3);
        let prompt = [1u32, 2, 3];
        for ns in [BASE_KV_NAMESPACE, 1, 2] {
            let r = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
            c.insert(Entry::new(prompt.to_vec(), BlockTable::with_namespace(ns), r, 0));
        }
        assert_eq!(c.stats().2, 3, "the same prompt in three namespaces is three entries, none superseding another");
        for ns in [BASE_KV_NAMESPACE, 1, 2] {
            let hit = c.lookup(ns, &[1, 2, 3, 4]).expect("own namespace hits");
            assert_eq!(c.entry(hit).table.kv_namespace(), ns);
        }
        assert!(c.lookup(7, &[1, 2, 3, 4]).is_none(), "a namespace with no entry misses");
        assert_eq!(c.evict_namespace(1).len(), 1);
        assert!(c.lookup(1, &[1, 2, 3]).is_none() && c.lookup(2, &[1, 2, 3]).is_some());
        assert!(c.take_row(BASE_KV_NAMESPACE, &[]).0.is_some(), "the dropped entry's snapshot row is free again");
    }

    /// Reclaiming for a waiting prompt spares the entry that prompt resumes from, however old it is.
    #[test]
    fn eviction_for_a_waiting_prompt_spares_the_entry_it_resumes_from() {
        let mut c = SnapshotPrefixCache::<u32>::new(2);
        for tokens in [[1u32, 1, 1], [2, 2, 2]] {
            let r = c.take_row(BASE_KV_NAMESPACE, &tokens).0.unwrap();
            c.insert(entry(&tokens, r));
        }
        assert!(c.evict_lru_keeping(BASE_KV_NAMESPACE, &[1, 1, 1, 9]).is_some(), "the other entry goes");
        assert!(c.peek(BASE_KV_NAMESPACE, &[1, 1, 1, 9]).is_some());
        assert!(c.evict_lru_keeping(BASE_KV_NAMESPACE, &[1, 1, 1, 9]).is_none(), "nothing else is left to give");
        assert_eq!(c.stats(), (0, 0, 1), "peeking counts no lookup");
    }

    #[test]
    fn a_disabled_cache_hands_out_no_rows() {
        let mut c = SnapshotPrefixCache::<u32>::new(0);
        assert!(!c.enabled());
        let (row, evicted) = c.take_row(BASE_KV_NAMESPACE, &[]);
        assert!(row.is_none() && evicted.is_none());
    }

    #[test]
    fn stats_count_hit_and_looked_up_tokens() {
        let mut c = SnapshotPrefixCache::<u32>::new(1);
        let r = c.take_row(BASE_KV_NAMESPACE, &[]).0.unwrap();
        c.insert(entry(&[1, 2, 3], r));
        c.lookup(BASE_KV_NAMESPACE, &[1, 2, 3, 4]);
        c.lookup(BASE_KV_NAMESPACE, &[9, 9]);
        assert_eq!(c.stats(), (3, 6, 1));
    }
}
