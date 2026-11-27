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
//! 1. **The cheapest active replica with a free slot** (one that is built or
//!    already has work), by [`replica_cost`]: the prefill it still has to do
//!    (the prompt less what its cache holds), the load it already carries
//!    (each request weighed at the prompt over its [`balance_slack`]),
//!    and how close the request would bring its KV pool to full. A
//!    conversation returns to the replica holding its history unless that one
//!    is far busier; a light load is served by as few cards as possible and
//!    shares one batch.
//! 2. **The next replica that is not built yet**, when every active replica's
//!    slots are taken: the instance is built on its card because demand needs it.
//! 3. **The best-matching or least-loaded replica** when every replica is full:
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

    /// Extend the chain by one segment of `units` units (tokens) whose identity is `segment`.
    pub fn push_units(&mut self, segment: &[u8], units: u64) {
        if self.links.len() >= MAX_LINKS {
            return;
        }
        let (before, total) = self.links.last().copied().unwrap_or((0, 0));
        let mut h = DefaultHasher::new();
        h.write_u64(before);
        h.write(segment);
        self.links.push((h.finish(), total + units));
    }

    /// The units (bytes or tokens) the whole chain covers.
    pub fn units(&self) -> u64 {
        self.links.last().map_or(0, |&(_, units)| units)
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
    /// Requests waiting on the replica beyond `load` (another client's, as the replica reports them).
    pub queued: usize,
    /// Free KV the replica reports, in the request's units (tokens); `None` when it reports none.
    pub kv_free: Option<u64>,
    /// Built, or has work: running on its card now.
    pub active: bool,
    /// Its devices have the budget for it.
    pub can_run: bool,
    /// Bytes of the request's prompt its prefix cache holds (0 when none or not useful).
    pub matched: u64,
}

/// The prompt of the request being placed, in the units of its [`PrefixChain`] (tokens for a token chain, bytes for a
/// text one), and the output it may add to the KV.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestSize {
    pub prompt: u64,
    pub output: u64,
}

/// What running a request of `size` on `view` is expected to cost, in prompt units: the prefill still to do, plus
/// `size.prompt / slack` for every request it already carries or queues (so a replica holding the whole prompt is
/// worth up to `slack` more requests than an idle one; `slack` is its [`balance_slack`]), plus a KV penalty growing
/// with the square of the share of its free KV the request would take, and prohibitive when it would not fit.
pub fn replica_cost(view: &ReplicaView, size: RequestSize, slack: f64) -> f64 {
    let prompt = size.prompt.max(1) as f64;
    let uncached = size.prompt.saturating_sub(view.matched) as f64;
    let busy = (view.load + view.queued) as f64 * prompt / slack.max(1.0);
    let kv = match view.kv_free {
        None => 0.0,
        Some(free) => {
            let need = uncached + size.output as f64;
            if need > free as f64 {
                prompt * KV_FULL_PENALTY
            } else {
                prompt * (need / free.max(1) as f64).powi(2)
            }
        }
    };
    uncached + busy + kv
}

/// The cost, in prompts, of a replica whose KV the request would not fit: it would preempt or wait, so any replica
/// where it fits is chosen first.
const KV_FULL_PENALTY: f64 = 1000.0;

/// The replica that runs the next request of `size` (see the module doc for the order). Replicas that cannot run are
/// passed over unless none can, so the refusal is the budget's own.
pub fn choose_replica(views: &[ReplicaView], size: RequestSize) -> usize {
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
    let share = |v: &ReplicaView| (v.load + v.queued) as f64 / v.slots.max(1) as f64;
    let lightest = free.iter().filter(|&&i| views[i].active).map(|&i| share(&views[i])).fold(f64::INFINITY, f64::min);
    let cheapest = free
        .iter()
        .copied()
        .filter(|&i| views[i].active)
        .map(|i| (replica_cost(&views[i], size, balance_slack(views[i].slots, lightest)), i))
        .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| lighter(&a.1, &b.1)));
    if let Some((_, i)) = cheapest {
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

/// How many requests a replica of `slots` slots may carry beyond the lightest one before holding the whole prompt in
/// cache stops paying for it ([`replica_cost`]), when the lightest active replica with room carries `lightest` of its
/// slots: a quarter of that load, at least [`MIN_BALANCE_SLACK`] requests. Below that the cached prefill wins; past it
/// balance does, so a prefix every request shares (one system prompt) or a conversation's history spreads over the
/// replicas instead of piling onto the one that served it first.
///
/// The bound follows the load, not the slots: a replica's slots are what it admits, which can be far more than its
/// card decodes at speed (a server advertising thousands), so a bound in slots let a cached prompt hold hundreds of
/// requests on one card while others idled (knowledge #607). Against an idle replica the bound is
/// [`MIN_BALANCE_SLACK`]: a cached prompt saves one prefill, an idle card serves every request after it. Under an even
/// load it grows with the load, so a quarter's imbalance is the most a cache hit buys and conversations stay put while
/// the fleet is balanced; moving one changes the difference by one request, far less than the bound, so it does not
/// swing back. Load counts as a share of the slots, so replicas of different widths balance by share.
pub fn balance_slack(slots: usize, lightest: f64) -> f64 {
    let lightest = if lightest.is_finite() { lightest.max(0.0) } else { 0.0 };
    (lightest * slots as f64 / 4.0).max(MIN_BALANCE_SLACK as f64)
}

/// The fewest requests a replica may carry beyond an idle one before a cached prompt stops keeping work there: a small
/// batch, which decodes about as fast as one request, so piling it onto the cached replica costs little; past it an
/// idle card is worth more than the prefill a hit saves.
pub const MIN_BALANCE_SLACK: usize = 8;

/// The request parameters that shape the start of a rendered chat prompt, before the first message: everything a
/// request of one conversation repeats ahead of its history.
pub const PROMPT_SHAPING: [&str; 7] = ["chat", "tools", "tool_choice", "system", "enable_thinking", "reasoning_effort", "template_flavor"];

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
    // The root counts only the prompt it stands for (the shaping fields), not the namespace naming it: two requests
    // that share nothing but the model share no work.
    let mut chain = PrefixChain::default();
    chain.push_units(root.as_bytes(), (root.len() - namespace.len()) as u64);
    match prompt {
        TextPrompt::Messages(messages) => json_array_elements(messages).into_iter().for_each(|message| chain.push(message.as_bytes())),
        TextPrompt::Raw(text) => chain.push_chunked(text.as_bytes(), RAW_PROMPT_LINK_BYTES),
    }
    chain
}

/// Tokens per link of a token chain: the granularity a paged KV cache shares prefixes at.
pub const TOKEN_LINK: usize = 16;

/// A request's prompt as a token [`PrefixChain`]: a root link for `root` (the namespace: model, its weights and adapter
/// version, and the prompt-shaping fields), counted as `root_tokens`, then one link per [`TOKEN_LINK`] tokens of
/// `prompt`. Two prompts share a prefix of `k` links exactly when they share the root and their first `k - 1` token
/// blocks, so a match is counted in the tokens a cache would actually reuse, and never across namespaces.
pub fn token_request_chain(root: &[u8], root_tokens: u64, prompt: &[u32]) -> PrefixChain {
    let mut chain = PrefixChain::default();
    chain.push_units(root, root_tokens);
    for block in prompt.chunks(TOKEN_LINK) {
        let bytes: Vec<u8> = block.iter().flat_map(|t| t.to_le_bytes()).collect();
        chain.push_units(&bytes, block.len() as u64);
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
        ReplicaView { load, slots: 8, queued: 0, kv_free: None, active, can_run: true, matched }
    }

    /// The request the policy tests place: a prompt four useful prefixes long.
    const SIZE: RequestSize = RequestSize { prompt: MIN_USEFUL_PREFIX * 4, output: 0 };

    fn choose(views: &[ReplicaView]) -> usize {
        choose_replica(views, SIZE)
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
        assert_eq!(choose(&views), 0);
    }

    #[test]
    fn the_next_replica_is_built_once_the_active_ones_are_full() {
        let views = [view(8, true, 0), view(0, false, 0), view(0, false, 0)];
        assert_eq!(choose(&views), 1, "demand exceeds the slots of replica 0");
        let views = [view(8, true, 0), view(8, true, 0), view(0, false, 0)];
        assert_eq!(choose(&views), 2);
    }

    #[test]
    fn active_replicas_with_room_are_balanced_by_load() {
        let views = [view(5, true, 0), view(2, true, 0), view(0, false, 0)];
        assert_eq!(choose(&views), 1);
    }

    #[test]
    fn a_cached_prefix_beats_the_least_loaded_replica() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(1, true, 0), view(6, true, hit)];
        assert_eq!(choose(&views), 1, "the replica holding the conversation, however busy, while it has a slot");
        let views = [view(3, true, hit / 2), view(3, true, hit), view(3, true, 0)];
        assert_eq!(choose(&views), 1, "at equal load the longest prefix wins");
    }

    #[test]
    fn a_prefix_too_short_to_matter_does_not_override_balance() {
        let views = [view(6, true, 16), view(1, true, 0)];
        assert_eq!(choose(&views), 1);
    }

    #[test]
    fn a_full_replica_holding_the_prefix_gives_way_to_one_with_room() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(8, true, hit), view(0, false, 0)];
        assert_eq!(choose(&views), 1, "a shared system prompt must not pin every conversation to one card");
    }

    #[test]
    fn when_every_replica_is_full_the_request_waits_where_its_prompt_is_cached() {
        let hit = MIN_USEFUL_PREFIX * 4;
        let views = [view(9, true, 0), view(12, true, hit)];
        assert_eq!(choose(&views), 1);
        let views = [view(12, true, 0), view(9, true, 0)];
        assert_eq!(choose(&views), 1, "no useful prefix: the shortest queue");
    }

    #[test]
    fn a_replica_that_cannot_run_is_passed_over_unless_none_can() {
        let mut held = view(0, true, 0);
        held.can_run = false;
        assert_eq!(choose(&[held, view(5, true, 0)]), 1);
        let mut other = view(1, true, 0);
        other.can_run = false;
        assert_eq!(choose(&[held, other]), 0, "none can run: the first, so the refusal is the budget's");
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
        let small = ReplicaView { load: 32, slots: 64, queued: 0, kv_free: None, active: true, can_run: true, matched: 0 };
        let large = ReplicaView { load: 64, slots: 256, ..small };
        assert_eq!(choose(&[small, large]), 1, "a quarter of 256 slots is lighter than half of 64");
        let full_small = ReplicaView { load: 64, ..small };
        let full_large = ReplicaView { load: 256, ..large };
        let over_small = ReplicaView { load: 96, ..small };
        assert_eq!(choose(&[over_small, full_large]), 1, "every replica full: the shortest queue relative to its slots");
        assert_eq!(choose(&[full_small, ReplicaView { load: 255, ..large }]), 1, "the one with a free slot");
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
        // However many slots they advertise: the bound is in requests, a quarter of the lightest load, at least 8.
        let hit = MIN_USEFUL_PREFIX * 4;
        for slots in [256, 4096] {
            let wide = |load, matched| ReplicaView { load, slots, queued: 0, kv_free: None, active: true, can_run: true, matched };
            assert_eq!(choose(&[wide(6, hit), wide(0, 0)]), 0, "a small batch is worth the cached prefill");
            assert_eq!(choose(&[wide(20, hit), wide(0, 0)]), 1, "a real batch is not worth an idle card");
            assert_eq!(choose(&[wide(64, hit), wide(2, hit / 2), wide(0, 0)]), 1, "half the prompt cached on a lightly loaded replica is cheapest");
            assert_eq!(choose(&[wide(70, hit), wide(60, 0)]), 0, "under an even load a conversation stays where it is cached");
            assert_eq!(choose(&[wide(80, hit), wide(60, 0)]), 1, "past a quarter's imbalance it moves");
        }
    }

    #[test]
    fn a_replica_whose_kv_the_request_would_fill_loses_and_waiting_work_counts_as_load() {
        let wide = |load, matched, kv_free| ReplicaView { load, slots: 256, queued: 0, kv_free, active: true, can_run: true, matched };
        let hit = SIZE.prompt;
        let size = RequestSize { prompt: SIZE.prompt, output: 1024 };
        assert_eq!(choose_replica(&[wide(0, hit, Some(512)), wide(0, 0, Some(1 << 20))], size), 1, "the output does not fit beside the cached prompt");
        assert_eq!(choose_replica(&[wide(0, hit, Some(1 << 20)), wide(0, 0, Some(1 << 20))], size), 0, "plenty of KV: the cache decides");
        assert_eq!(choose_replica(&[wide(0, 0, Some(size.prompt + 1100)), wide(0, 0, Some(1 << 20))], size), 1, "nearly full KV costs more than an idle replica with room");
        let mut busy = wide(0, 0, None);
        busy.queued = 64;
        assert_eq!(choose(&[busy, wide(10, 0, None)]), 1, "work queued on a replica is load too");
    }
}
