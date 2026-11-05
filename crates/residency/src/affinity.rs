// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving agent workloads on every
// card of a multi-GPU host for its clients: conversations that resend a growing
// history each turn keep their prefix-cache hits while all cards stay busy. If
// your team needs expertise in multi-replica LLM serving then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Choosing which replica of a model runs a request.
//!
//! A model that can run on several cards ([`ResidentModel::replicas`](crate::ResidentModel::replicas))
//! keeps one instance per card, and each instance has its own prefix cache. A
//! conversation that resends a growing history every turn pays its whole
//! prefill again unless it lands on the replica that served its earlier turns,
//! so the choice is made in this order:
//!
//! 1. **A replica holding a useful prefix of the prompt**, among the replicas
//!    with a free slot: the one holding the longest.
//! 2. **The least-loaded active replica** with a free slot (one that is built or
//!    already has work), so a light load is served by as few cards as possible
//!    and shares one batch.
//! 3. **The next replica that is not built yet**, when every active replica's
//!    slots are taken: the instance is built on its card because demand needs it.
//! 4. **The best-matching or least-loaded replica** when every replica is full:
//!    the request queues behind running work.
//!
//! A replica's prefix cache lives inside its instance, on the lane that is
//! decoding, where the dispatcher cannot read it. [`PrefixDirectory`] is the
//! dispatcher's own record of what it routed to each replica, sized like the
//! cache it mirrors (the number of prompts the cache keeps) and dropped when the
//! replica is no longer built, because its cache went with it. It is a routing
//! hint: a stale entry costs one prefill, never a wrong answer.
//!
//! A request is keyed once, by the model ([`ResidentModel::prefix_chain`](crate::ResidentModel::prefix_chain)):
//! a [`PrefixChain`] of cumulative digests, one per segment of the prompt, so a
//! lookup is one pass over the chain and a prompt is never hashed again.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hasher;

/// How a model that runs several replicas wants requests spread over them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicaRouting {
    /// Requests one replica runs at once (its decode slots). A further replica is built only once every active
    /// replica holds this many.
    pub slots: usize,
    /// Prompts one replica's prefix cache keeps: the bound of the dispatcher's record of it.
    pub cached_prompts: usize,
}

/// The shortest shared prefix, in bytes of prompt, that is worth routing for: under it the prefill a hit saves is
/// smaller than the balance a free replica gives.
pub const MIN_USEFUL_PREFIX: u64 = 2048;

/// The most links one prompt's chain keeps, bounding the directory's memory for a very long raw prompt.
const MAX_LINKS: usize = 4096;

/// A request's prompt as cumulative digests: link `k` identifies the first `k` segments, and carries the bytes they hold.
/// Two prompts share a prefix of `k` segments exactly when their links `k` are equal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixChain {
    /// `(digest of segments 0..=k, bytes of segments 0..=k)`.
    links: Vec<(u64, u64)>,
}

impl PrefixChain {
    /// A chain whose first link is `root`: what every request of the same namespace shares (the adapter, the tools,
    /// the system prompt). Empty when `root` is.
    pub fn rooted(root: &[u8]) -> PrefixChain {
        let mut chain = PrefixChain::default();
        chain.push(root);
        chain
    }

    /// Extend the chain by one segment.
    pub fn push(&mut self, segment: &[u8]) {
        if self.links.len() >= MAX_LINKS {
            return;
        }
        let (before, bytes) = self.links.last().copied().unwrap_or((0, 0));
        let mut h = DefaultHasher::new();
        h.write_u64(before);
        h.write(segment);
        self.links.push((h.finish(), bytes + segment.len() as u64));
    }

    /// Extend the chain by `bytes` split into `chunk`-sized segments (a raw prompt has no message boundaries).
    pub fn push_chunked(&mut self, bytes: &[u8], chunk: usize) {
        for part in bytes.chunks(chunk.max(1)) {
            self.push(part);
        }
    }

    /// No segment at all: nothing to route by.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }
}

struct Entry {
    digests: Vec<u64>,
    stamp: u64,
}

/// What the dispatcher has routed to one replica, as the prompts its prefix cache holds.
pub struct PrefixDirectory {
    capacity: usize,
    entries: HashMap<u64, Entry>,
    /// Each digest, to the newest entry holding it.
    index: HashMap<u64, u64>,
    /// Recency: stamp to entry id, oldest first.
    recency: BTreeMap<u64, u64>,
    clock: u64,
}

impl PrefixDirectory {
    /// A directory of at most `capacity` prompts; 0 records nothing (a replica without a prefix cache).
    pub fn new(capacity: usize) -> PrefixDirectory {
        PrefixDirectory { capacity, entries: HashMap::new(), index: HashMap::new(), recency: BTreeMap::new(), clock: 0 }
    }

    /// Bytes of `chain`'s longest prefix some held prompt shares; 0 when none. One pass from the end of the chain.
    pub fn matched(&self, chain: &PrefixChain) -> u64 {
        chain.links.iter().rev().find(|(digest, _)| self.index.contains_key(digest)).map_or(0, |&(_, bytes)| bytes)
    }

    /// Note that the replica now holds `chain`'s prompt. A held prompt this one extends is replaced by it, as the
    /// cache replaces an earlier turn with the later one; the oldest prompt leaves when the directory is full.
    pub fn record(&mut self, chain: &PrefixChain) {
        if self.capacity == 0 || chain.is_empty() {
            return;
        }
        if let Some(&(digest, _)) = chain.links.iter().rev().find(|(d, _)| self.index.contains_key(d)) {
            let owner = self.index[&digest];
            if self.entries.get(&owner).and_then(|e| e.digests.last()) == Some(&digest) {
                self.remove(owner);
            }
        }
        while self.entries.len() >= self.capacity {
            let Some((_, oldest)) = self.recency.pop_first() else { break };
            self.remove(oldest);
        }
        self.clock += 1;
        let id = self.clock;
        let digests: Vec<u64> = chain.links.iter().map(|&(d, _)| d).collect();
        for &d in &digests {
            self.index.insert(d, id);
        }
        self.recency.insert(id, id);
        self.entries.insert(id, Entry { digests, stamp: id });
    }

    /// Forget everything: the replica's cache is gone.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.recency.clear();
    }

    /// Prompts currently held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn remove(&mut self, id: u64) {
        let Some(entry) = self.entries.remove(&id) else { return };
        self.recency.remove(&entry.stamp);
        for d in entry.digests {
            if self.index.get(&d) == Some(&id) {
                self.index.remove(&d);
            }
        }
    }
}

/// One replica as [`choose_replica`] sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicaView {
    /// Jobs queued or running on the replica.
    pub load: usize,
    /// Built, or has work: running on its card now.
    pub active: bool,
    /// Its devices have the budget for it.
    pub can_run: bool,
    /// Bytes of the request's prompt its prefix cache holds (0 when none or not useful).
    pub matched: u64,
}

/// The replica that runs the next request (see the module doc for the order). Replicas that cannot run are passed over
/// unless none can, so the refusal is the budget's own.
pub fn choose_replica(views: &[ReplicaView], slots: usize) -> usize {
    let usable: Vec<usize> = match (0..views.len()).filter(|&i| views[i].can_run).collect::<Vec<_>>() {
        none if none.is_empty() => (0..views.len()).collect(),
        some => some,
    };
    let least_loaded = |of: &mut dyn Iterator<Item = usize>| of.min_by_key(|&i| (views[i].load, i));
    let free: Vec<usize> = usable.iter().copied().filter(|&i| views[i].load < slots).collect();
    let best_match = free.iter().map(|&i| views[i].matched).max().unwrap_or(0);
    if best_match >= MIN_USEFUL_PREFIX {
        return least_loaded(&mut free.iter().copied().filter(|&i| views[i].matched == best_match)).unwrap_or(0);
    }
    if let Some(i) = least_loaded(&mut free.iter().copied().filter(|&i| views[i].active)) {
        return i;
    }
    if let Some(&i) = free.first() {
        return i;
    }
    // Every replica is full: wait where the prompt is cached, else where the queue is shortest.
    let best_match = usable.iter().map(|&i| views[i].matched).max().unwrap_or(0);
    let pool: Vec<usize> = usable.into_iter().filter(|&i| best_match < MIN_USEFUL_PREFIX || views[i].matched == best_match).collect();
    least_loaded(&mut pool.into_iter()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(segments: &[&str]) -> PrefixChain {
        let mut c = PrefixChain::default();
        for s in segments {
            c.push(s.as_bytes());
        }
        c
    }

    /// A segment big enough that a prefix holding it is worth routing for.
    fn big(tag: &str) -> String {
        tag.repeat(MIN_USEFUL_PREFIX as usize)
    }

    fn view(load: usize, active: bool, matched: u64) -> ReplicaView {
        ReplicaView { load, active, can_run: true, matched }
    }

    #[test]
    fn a_chain_shares_exactly_the_segments_two_prompts_share() {
        let a = chain(&["system", "hello", "answer"]);
        let b = chain(&["system", "hello", "other"]);
        let mut dir = PrefixDirectory::new(4);
        dir.record(&a);
        assert_eq!(dir.matched(&a), 6 + 5 + 6);
        assert_eq!(dir.matched(&b), 6 + 5, "the two shared segments, not the third");
        assert_eq!(dir.matched(&chain(&["other system", "hello"])), 0, "a different root shares nothing, whatever follows");
    }

    #[test]
    fn a_later_turn_replaces_the_earlier_one_and_the_oldest_prompt_leaves_when_full() {
        let mut dir = PrefixDirectory::new(2);
        dir.record(&chain(&["s", "a"]));
        dir.record(&chain(&["s", "a", "b"]));
        assert_eq!(dir.len(), 1, "turn two extends turn one: one held prompt");
        dir.record(&chain(&["t", "x"]));
        dir.record(&chain(&["u", "y"]));
        assert_eq!(dir.len(), 2);
        assert_eq!(dir.matched(&chain(&["s", "a", "b"])), 0, "the oldest prompt left");
        assert!(dir.matched(&chain(&["u", "y"])) > 0);
        dir.clear();
        assert!(dir.is_empty() && dir.matched(&chain(&["u", "y"])) == 0);
    }

    #[test]
    fn a_zero_capacity_directory_holds_nothing() {
        let mut dir = PrefixDirectory::new(0);
        dir.record(&chain(&["s"]));
        assert_eq!(dir.matched(&chain(&["s"])), 0);
    }

    #[test]
    fn a_chain_is_bounded() {
        let mut c = PrefixChain::default();
        c.push_chunked(&vec![7u8; 10 * MAX_LINKS], 5);
        assert_eq!(c.len(), MAX_LINKS);
    }

    #[test]
    fn a_light_load_fills_the_active_replica_before_building_another() {
        let views = [view(3, true, 0), view(0, false, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views, 8), 0);
    }

    #[test]
    fn the_next_replica_is_built_once_the_active_ones_are_full() {
        let views = [view(8, true, 0), view(0, false, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views, 8), 1, "demand exceeds the slots of replica 0");
        let views = [view(8, true, 0), view(8, true, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views, 8), 2);
    }

    #[test]
    fn active_replicas_with_room_are_balanced_by_load() {
        let views = [view(5, true, 0), view(2, true, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views, 8), 1);
    }

    #[test]
    fn a_cached_prefix_beats_the_least_loaded_replica() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(1, true, 0), view(6, true, hit)];
        assert_eq!(choose_replica(&views, 8), 1, "the replica holding the conversation, however busy, while it has a slot");
        let views = [view(1, true, hit / 2), view(6, true, hit), view(0, true, 0)];
        assert_eq!(choose_replica(&views, 8), 1, "the longest prefix wins");
    }

    #[test]
    fn a_prefix_too_short_to_matter_does_not_override_balance() {
        let views = [view(6, true, 16), view(1, true, 0)];
        assert_eq!(choose_replica(&views, 8), 1);
    }

    #[test]
    fn a_full_replica_holding_the_prefix_gives_way_to_one_with_room() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(8, true, hit), view(0, false, 0)];
        assert_eq!(choose_replica(&views, 8), 1, "a shared system prompt must not pin every conversation to one card");
    }

    #[test]
    fn when_every_replica_is_full_the_request_waits_where_its_prompt_is_cached() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(9, true, 0), view(12, true, hit)];
        assert_eq!(choose_replica(&views, 8), 1);
        let views = [view(12, true, 0), view(9, true, 0)];
        assert_eq!(choose_replica(&views, 8), 1, "no useful prefix: the shortest queue");
    }

    #[test]
    fn a_replica_that_cannot_run_is_passed_over_unless_none_can() {
        let mut held = view(0, true, 0);
        held.can_run = false;
        assert_eq!(choose_replica(&[held, view(5, true, 0)], 8), 1);
        let mut other = view(1, true, 0);
        other.can_run = false;
        assert_eq!(choose_replica(&[held, other], 8), 0, "none can run: the first, so the refusal is the budget's");
    }

    #[test]
    fn a_big_shared_prefix_is_a_useful_one() {
        let sys = big("s");
        let mut dir = PrefixDirectory::new(4);
        dir.record(&chain(&[&sys, "question one"]));
        assert!(dir.matched(&chain(&[&sys, "question two"])) >= MIN_USEFUL_PREFIX);
    }
}
