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
//! A replica is whatever holds its own prefix cache and runs requests: an
//! instance on one card of this process, or a whole `brain serve` endpoint behind
//! a router (`apiserve::route`). Both choose with [`choose_replica`] over
//! [`ReplicaView`]s and remember what they sent where in a [`PrefixDirectory`],
//! and both key a text request with [`text_request_chain`], so the policy exists
//! once whatever transport carries the request.
//!
//! A model that can run on several cards ([`ResidentModel::replicas`](crate::ResidentModel::replicas))
//! keeps one instance per card, and each instance has its own prefix cache. A
//! conversation that resends a growing history every turn pays its whole
//! prefill again unless it lands on the replica that served its earlier turns,
//! so the choice is made in this order:
//!
//! 1. **A replica holding a useful prefix of the prompt**, among the replicas
//!    with a free slot that carry at most [`balance_slack`] more requests than
//!    the least loaded one: the one holding the longest.
//! 2. **The least-loaded active replica** with a free slot (one that is built or
//!    already has work), so a light load is served by as few cards as possible
//!    and shares one batch. Load is relative to the replica's own slots, so a
//!    replica with twice the slots takes twice the requests.
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

use std::borrow::Cow;
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
    /// Requests the replica runs at once: it has a free slot while `load` is below this.
    pub slots: usize,
    /// Built, or has work: running on its card now.
    pub active: bool,
    /// Its devices have the budget for it.
    pub can_run: bool,
    /// Bytes of the request's prompt its prefix cache holds (0 when none or not useful).
    pub matched: u64,
}

/// The replica that runs the next request (see the module doc for the order). Replicas that cannot run are passed over
/// unless none can, so the refusal is the budget's own.
pub fn choose_replica(views: &[ReplicaView]) -> usize {
    let usable: Vec<usize> = match (0..views.len()).filter(|&i| views[i].can_run).collect::<Vec<_>>() {
        none if none.is_empty() => (0..views.len()).collect(),
        some => some,
    };
    // Load as a share of the replica's slots, compared without division; the lower index on a tie.
    let lighter = |&a: &usize, &b: &usize| {
        let (va, vb) = (views[a], views[b]);
        ((va.load as u128) * (vb.slots.max(1) as u128)).cmp(&((vb.load as u128) * (va.slots.max(1) as u128))).then(a.cmp(&b))
    };
    let least_loaded = |of: &mut dyn Iterator<Item = usize>| of.min_by(lighter);
    let free: Vec<usize> = usable.iter().copied().filter(|&i| views[i].load < views[i].slots).collect();
    // A hit is worth its prefill only on a replica not far busier than the least loaded one: within
    // [`balance_slack`] requests of the lowest load share among the free replicas.
    let lowest_share = free.iter().map(|&i| views[i].load as f64 / views[i].slots.max(1) as f64).fold(f64::INFINITY, f64::min);
    let balanced: Vec<usize> = free.iter().copied().filter(|&i| views[i].load as f64 <= lowest_share * views[i].slots.max(1) as f64 + balance_slack(views[i].slots) as f64).collect();
    let best_match = balanced.iter().map(|&i| views[i].matched).max().unwrap_or(0);
    if best_match >= MIN_USEFUL_PREFIX {
        return least_loaded(&mut balanced.iter().copied().filter(|&i| views[i].matched == best_match)).unwrap_or(0);
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

/// How many requests more than the least-loaded free replica (at its share of slots) a replica holding a useful
/// prefix may carry and still be chosen for it: an eighth of its slots, at least 8. Below that the cached prefill wins;
/// past it balance does, so a prefix every request shares (one system prompt) spreads over the replicas instead of
/// piling onto the one that served it first.
pub fn balance_slack(slots: usize) -> usize {
    (slots / 8).max(8)
}

/// The request parameters that shape the start of a rendered chat prompt, before the first message: everything a
/// request of one conversation repeats ahead of its history.
pub const PROMPT_SHAPING: [&str; 8] = ["chat", "tools", "tool_choice", "system", "enable_thinking", "reasoning_effort", "preserve_thinking", "template_flavor"];

/// A raw prompt (no message boundaries) is cut into links of this many bytes.
pub const RAW_PROMPT_LINK_BYTES: usize = 1024;

/// The prompt of a text request, as its transport carries it.
#[derive(Clone, Copy, Debug)]
pub enum TextPrompt<'a> {
    /// A chat: the messages as the text of one JSON array.
    Messages(&'a str),
    /// A raw prompt to continue.
    Raw(&'a str),
}

/// A text request's prompt as a [`PrefixChain`]: one link for everything that shapes the start of the rendered prompt
/// (`namespace`, then each `(key, value)` of `shaping` in [`PROMPT_SHAPING`] order as the transport spells it), then one
/// link per chat message, or [`RAW_PROMPT_LINK_BYTES`] chunks of a raw prompt. Turn `n + 1` of a conversation shares
/// `n + 1` links with turn `n`. `namespace` separates prompts whose cached state is not interchangeable (another model,
/// another adapter). The text is hashed once, in one pass: nothing is rendered or tokenized.
pub fn text_request_chain<'a>(namespace: &str, shaping: impl IntoIterator<Item = (&'a str, Cow<'a, str>)>, prompt: TextPrompt<'_>) -> PrefixChain {
    let mut root = String::from(namespace);
    for (key, value) in shaping {
        root.push('\u{1f}');
        root.push_str(key);
        root.push('=');
        root.push_str(&value);
    }
    let mut chain = PrefixChain::rooted(root.as_bytes());
    match prompt {
        TextPrompt::Messages(messages) => json_array_elements(messages).into_iter().for_each(|message| chain.push(message.as_bytes())),
        TextPrompt::Raw(text) => chain.push_chunked(text.as_bytes(), RAW_PROMPT_LINK_BYTES),
    }
    chain
}

/// The top-level elements of the JSON array `text`, as slices of it, found in one scan without building a value. Text
/// that is not an array, or is cut short, yields what the scan found: a routing hint never refuses a request (whatever
/// admits it reports the parse error).
pub fn json_array_elements(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut elements = Vec::new();
    let (mut depth, mut in_string, mut escaped, mut start) = (0usize, false, false, None::<usize>);
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                start.get_or_insert(i);
            }
            b'[' | b'{' => {
                depth += 1;
                if depth > 1 {
                    start.get_or_insert(i);
                }
            }
            b']' | b'}' => {
                if depth == 1 {
                    elements.extend(start.take().map(|s| text[s..i].trim_end()));
                    break;
                }
                depth = depth.saturating_sub(1);
            }
            b',' if depth == 1 => elements.extend(start.take().map(|s| text[s..i].trim_end())),
            b if depth >= 1 && !b.is_ascii_whitespace() => {
                start.get_or_insert(i);
            }
            _ => {}
        }
    }
    elements
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
        ReplicaView { load, slots: 8, active, can_run: true, matched }
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
        assert_eq!(choose_replica(&views), 0);
    }

    #[test]
    fn the_next_replica_is_built_once_the_active_ones_are_full() {
        let views = [view(8, true, 0), view(0, false, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views), 1, "demand exceeds the slots of replica 0");
        let views = [view(8, true, 0), view(8, true, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views), 2);
    }

    #[test]
    fn active_replicas_with_room_are_balanced_by_load() {
        let views = [view(5, true, 0), view(2, true, 0), view(0, false, 0)];
        assert_eq!(choose_replica(&views), 1);
    }

    #[test]
    fn a_cached_prefix_beats_the_least_loaded_replica() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(1, true, 0), view(6, true, hit)];
        assert_eq!(choose_replica(&views), 1, "the replica holding the conversation, however busy, while it has a slot");
        let views = [view(1, true, hit / 2), view(6, true, hit), view(0, true, 0)];
        assert_eq!(choose_replica(&views), 1, "the longest prefix wins");
    }

    #[test]
    fn a_prefix_too_short_to_matter_does_not_override_balance() {
        let views = [view(6, true, 16), view(1, true, 0)];
        assert_eq!(choose_replica(&views), 1);
    }

    #[test]
    fn a_full_replica_holding_the_prefix_gives_way_to_one_with_room() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(8, true, hit), view(0, false, 0)];
        assert_eq!(choose_replica(&views), 1, "a shared system prompt must not pin every conversation to one card");
    }

    #[test]
    fn when_every_replica_is_full_the_request_waits_where_its_prompt_is_cached() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(9, true, 0), view(12, true, hit)];
        assert_eq!(choose_replica(&views), 1);
        let views = [view(12, true, 0), view(9, true, 0)];
        assert_eq!(choose_replica(&views), 1, "no useful prefix: the shortest queue");
    }

    #[test]
    fn a_replica_that_cannot_run_is_passed_over_unless_none_can() {
        let mut held = view(0, true, 0);
        held.can_run = false;
        assert_eq!(choose_replica(&[held, view(5, true, 0)]), 1);
        let mut other = view(1, true, 0);
        other.can_run = false;
        assert_eq!(choose_replica(&[held, other]), 0, "none can run: the first, so the refusal is the budget's");
    }

    #[test]
    fn a_big_shared_prefix_is_a_useful_one() {
        let sys = big("s");
        let mut dir = PrefixDirectory::new(4);
        dir.record(&chain(&[&sys, "question one"]));
        assert!(dir.matched(&chain(&[&sys, "question two"])) >= MIN_USEFUL_PREFIX);
    }

    #[test]
    fn load_is_weighed_against_each_replicas_own_slots() {
        let small = ReplicaView { load: 4, slots: 8, active: true, can_run: true, matched: 0 };
        let large = ReplicaView { load: 8, slots: 32, active: true, can_run: true, matched: 0 };
        assert_eq!(choose_replica(&[small, large]), 1, "a quarter of 32 slots is lighter than half of 8");
        let full_small = ReplicaView { load: 8, ..small };
        let full_large = ReplicaView { load: 32, ..large };
        let over_small = ReplicaView { load: 12, ..small };
        assert_eq!(choose_replica(&[over_small, full_large]), 1, "every replica full: the shortest queue relative to its slots");
        assert_eq!(choose_replica(&[full_small, ReplicaView { load: 31, ..large }]), 1, "the one with a free slot");
    }

    fn chat_chain(namespace: &str, system: &str, messages: &[(&str, &str)]) -> PrefixChain {
        let array = format!("[{}]", messages.iter().map(|(r, c)| format!(r#"{{"role":"{r}","content":"{c}"}}"#)).collect::<Vec<_>>().join(","));
        text_request_chain(namespace, [("system", Cow::Borrowed(system))], TextPrompt::Messages(&array))
    }

    #[test]
    fn a_text_chain_shares_the_turns_two_requests_share() {
        let mut dir = PrefixDirectory::new(4);
        let turn1 = chat_chain("m", "sys", &[("user", "a")]);
        dir.record(&turn1);
        let turn2 = chat_chain("m", "sys", &[("user", "a"), ("assistant", "b"), ("user", "c")]);
        assert_eq!(dir.matched(&turn2), dir.matched(&turn1), "every link of turn one is shared by turn two");
        assert_eq!(dir.matched(&chat_chain("other", "sys", &[("user", "a")])), 0, "another namespace shares nothing");
        assert_eq!(dir.matched(&chat_chain("m", "other", &[("user", "a")])), 0, "another prompt start shares nothing");
        let raw = "x".repeat(3 * RAW_PROMPT_LINK_BYTES);
        let held = text_request_chain("m", [], TextPrompt::Raw(&raw));
        assert_eq!(held.len(), 1 + 3, "the root and three chunks");
    }

    #[test]
    fn the_array_scan_splits_only_at_the_top_level_and_survives_hostile_text() {
        assert_eq!(json_array_elements(r#" [ {"a":[1,2]} , "x,]" , 3 ] "#), vec![r#"{"a":[1,2]}"#, r#""x,]""#, "3"]);
        assert!(json_array_elements("[]").is_empty() && json_array_elements("").is_empty());
        let _ = json_array_elements(r#"[{"a": "unterminated"#);
        let _ = json_array_elements("]]]}}}[[[{{{");
        let _ = json_array_elements(r#"["\"#);
    }

    #[test]
    fn a_prefix_every_request_shares_does_not_pin_a_wide_replica_while_others_idle() {
        // Replicas with hundreds of slots are never full: the hit must give way once its replica carries much more
        // than the least loaded one, or one shared system prompt sends every request to one replica.
        let hit = MIN_USEFUL_PREFIX * 4;
        let wide = |load, matched| ReplicaView { load, slots: 256, active: true, can_run: true, matched };
        assert_eq!(choose_replica(&[wide(20, hit), wide(0, 0)]), 0, "a little more load is worth the cached prefill");
        assert_eq!(choose_replica(&[wide(40, hit), wide(0, 0)]), 1, "far more load is not");
        assert_eq!(choose_replica(&[wide(40, hit), wide(10, hit / 2), wide(0, 0)]), 1, "the best hit among the balanced ones");
    }
}
