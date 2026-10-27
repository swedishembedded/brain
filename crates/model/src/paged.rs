// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Paged KV-cache bookkeeping: a fixed pool of physical KV blocks, a free-list
//! allocator with per-block reference counts (so sequences can *share* blocks —
//! a common prompt prefix, parallel samples, speculative branches — and free them
//! independently), and per-sequence **block tables** mapping logical token
//! positions to physical blocks.
//!
//! This is the pure host-side bookkeeping, unit-tested without a device. The GPU
//! block pool (`[num_blocks * block_size * kv_dim]` per layer) and the paged
//! attention kernels that follow a block table live next to the model decode; a
//! sequence's KV grows one block at a time instead of reserving its worst-case
//! length up front, and any free block can back any sequence's next logical block.

/// A pool of physical KV blocks handed out by id. Reference-counted: `alloc`
/// yields a fresh block (refcount 1), `incref` shares one, `decref` releases a
/// reference and returns the block to the free list when the last one drops.
#[derive(Clone, Debug)]
pub struct BlockAllocator {
    block_size: u32,
    num_blocks: u32,
    free: Vec<u32>,     // stack of free physical block ids
    refcount: Vec<u16>, // per physical block
}

impl BlockAllocator {
    /// A pool of `num_blocks` blocks of `block_size` tokens each.
    pub fn new(num_blocks: u32, block_size: u32) -> BlockAllocator {
        assert!(num_blocks > 0 && block_size > 0);
        // Hand out low ids first (nicer locality) — push high..low so pop yields low.
        let free: Vec<u32> = (0..num_blocks).rev().collect();
        BlockAllocator { block_size, num_blocks, free, refcount: vec![0; num_blocks as usize] }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }
    /// Physical blocks currently free.
    pub fn free_blocks(&self) -> u32 {
        self.free.len() as u32
    }
    /// Tokens that could still be admitted (free blocks × block size).
    pub fn free_tokens(&self) -> u32 {
        self.free_blocks() * self.block_size
    }
    pub fn refcount(&self, block: u32) -> u16 {
        self.refcount[block as usize]
    }

    /// Allocate a fresh block (refcount 1), or `None` if the pool is exhausted.
    pub fn alloc(&mut self) -> Option<u32> {
        let b = self.free.pop()?;
        debug_assert_eq!(self.refcount[b as usize], 0, "alloc of a live block {b}");
        self.refcount[b as usize] = 1;
        Some(b)
    }

    /// Share an existing block (refcount += 1) — prefix sharing / copy-on-write.
    pub fn incref(&mut self, block: u32) {
        self.refcount[block as usize] += 1;
    }

    /// Release a reference; frees the block when the last reference drops.
    pub fn decref(&mut self, block: u32) {
        let rc = &mut self.refcount[block as usize];
        assert!(*rc > 0, "double free of block {block}");
        *rc -= 1;
        if *rc == 0 {
            self.free.push(block);
        }
    }
}

/// Which K/V computation a block belongs to. K/V depend on every weight that
/// produced them, so blocks computed under one runtime adapter are wrong under
/// another adapter or under the base model: a sequence's blocks, and the
/// prefix-cache entries pointing at them, live in exactly one namespace.
pub type KvNamespace = u32;

/// The base model's namespace: the only one a decoder without adapters uses.
pub const BASE_KV_NAMESPACE: KvNamespace = 0;

/// Hands out adapter namespaces, strictly increasing and never repeated for
/// the life of the allocator, so an id retired by an unload can never name a
/// later adapter's blocks. Owned by whatever assigns adapter identities (one
/// per serving engine); [`BASE_KV_NAMESPACE`] is never handed out.
#[derive(Clone, Debug, Default)]
pub struct NamespaceIds {
    issued: KvNamespace,
}

impl NamespaceIds {
    /// The next unused namespace. Panics after `u32::MAX` namespaces, which
    /// would take billions of adapter loads in one process lifetime.
    pub fn fresh(&mut self) -> KvNamespace {
        self.issued = self.issued.checked_add(1).expect("KV namespace ids exhausted");
        self.issued
    }
}

/// A sequence's logical→physical block map. `len` is the number of tokens whose
/// K/V are stored; the cache grows a block at a time as tokens are appended.
///
/// The table also carries the sequence's [`KvNamespace`], so the adapter a
/// sequence's K/V were computed under travels with the blocks through prefill,
/// batched decode, fork and host-RAM offload instead of being a side channel
/// that can disagree with them.
#[derive(Clone, Debug, Default)]
pub struct BlockTable {
    blocks: Vec<u32>, // logical block index -> physical block id
    len: u32,         // tokens stored
    kv_namespace: KvNamespace,
    state_row: Option<u32>, // the sequence's row in a hybrid decoder's recurrent-state pool
    /// Prompt tokens a decoder served from its prefix cache when the sequence began (see [`Self::cached_rows`]).
    cached_rows: u32,
}

impl BlockTable {
    /// An empty table in the base namespace.
    pub fn new() -> BlockTable {
        BlockTable::default()
    }
    /// An empty table whose K/V will be computed in `kv_namespace`.
    pub fn with_namespace(kv_namespace: KvNamespace) -> BlockTable {
        BlockTable { kv_namespace, ..BlockTable::default() }
    }
    /// The namespace this sequence's K/V belong to; fixed for its lifetime,
    /// including across [`Self::release`].
    pub fn kv_namespace(&self) -> KvNamespace {
        self.kv_namespace
    }
    /// Tokens stored so far (== the next token's absolute position).
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The physical block ids (logical order) — uploaded to the GPU for the kernel.
    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// The row of the decoder's recurrent-state pool this sequence owns, for a
    /// hybrid decoder (linear-attention layers keep a fixed-size state per
    /// SEQUENCE, not per token, so it lives beside the paged KV blocks rather
    /// than in them). `None` until the decoder assigns one. Not shared by
    /// [`Self::fork`] or restored by [`Self::restore`]: a row belongs to
    /// exactly one live sequence.
    pub fn state_row(&self) -> Option<u32> {
        self.state_row
    }
    /// Record the recurrent-state row the decoder assigned to this sequence.
    pub fn set_state_row(&mut self, row: u32) {
        self.state_row = Some(row);
    }
    /// Hand the recurrent-state row back (to the pool it came from).
    pub fn take_state_row(&mut self) -> Option<u32> {
        self.state_row.take()
    }

    /// How many leading prompt tokens this sequence did not have to compute because a prefix cache served them
    /// (0 for a cold prompt). Set by the decoder when it starts the sequence from a cached prefix
    /// ([`Self::adopt_prefix`], [`Self::adopt_rows`], or [`Self::mark_cached`] for a decoder that copies the
    /// rows) and kept for the sequence's life, so the scheduler can report it with the outcome.
    pub fn cached_rows(&self) -> u32 {
        self.cached_rows
    }
    /// Record that the first `rows` tokens of this sequence came from a prefix cache.
    pub fn mark_cached(&mut self, rows: u32) {
        self.cached_rows = rows;
    }

    /// The `(physical_block, offset)` holding logical token `tok` (must be < len).
    pub fn locate(&self, tok: u32, block_size: u32) -> (u32, u32) {
        (self.blocks[(tok / block_size) as usize], tok % block_size)
    }

    /// Reserve room for one more token, allocating a block if the current one is
    /// full, and return the `(physical_block, offset)` the caller writes K/V into.
    /// The stored length advances by one. Errors if the pool is exhausted.
    pub fn append(&mut self, alloc: &mut BlockAllocator) -> Result<(u32, u32), String> {
        let offset = self.len % alloc.block_size();
        if offset == 0 {
            let b = alloc.alloc().ok_or("paged KV pool exhausted")?;
            self.blocks.push(b);
        }
        let block = *self.blocks.last().unwrap();
        self.len += 1;
        Ok((block, offset))
    }

    /// Ensure capacity for `n` more tokens by pre-allocating blocks (used by
    /// prefill so a whole chunk of positions is backed before the kernel writes).
    /// Returns the base absolute position the caller should start writing at.
    pub fn reserve(&mut self, n: u32, alloc: &mut BlockAllocator) -> Result<u32, String> {
        let start = self.len;
        let bs = alloc.block_size();
        let end = self.len + n;
        // Blocks needed to cover [0, end) that we don't already have.
        let need = end.div_ceil(bs);
        while (self.blocks.len() as u32) < need {
            let b = alloc.alloc().ok_or("paged KV pool exhausted")?;
            self.blocks.push(b);
        }
        self.len = end;
        Ok(start)
    }

    /// Shrink to `new_len` tokens, freeing (decref) any blocks that fall entirely
    /// beyond it — used to roll back rejected speculative tokens. `new_len` must
    /// not exceed the current length.
    pub fn truncate(&mut self, new_len: u32, alloc: &mut BlockAllocator) {
        assert!(new_len <= self.len, "truncate {new_len} > len {}", self.len);
        let keep = new_len.div_ceil(alloc.block_size()) as usize;
        for &b in &self.blocks[keep..] {
            alloc.decref(b);
        }
        self.blocks.truncate(keep);
        self.len = new_len;
    }

    /// Release every block (decref) — call when the sequence completes/evicts.
    pub fn release(&mut self, alloc: &mut BlockAllocator) {
        for &b in &self.blocks {
            alloc.decref(b);
        }
        self.blocks.clear();
        self.len = 0;
        self.cached_rows = 0;
    }

    /// Fork a sequence that **shares** all of this table's blocks (incref each) —
    /// the basis for prefix sharing, parallel sampling, and speculative branches.
    /// The child sees the same `len`; a copy-on-write [`Self::unshare_tail`] makes
    /// the last block private before the child writes a diverging token.
    pub fn fork(&self, alloc: &mut BlockAllocator) -> BlockTable {
        for &b in &self.blocks {
            alloc.incref(b);
        }
        BlockTable { blocks: self.blocks.clone(), len: self.len, kv_namespace: self.kv_namespace, state_row: None, cached_rows: 0 }
    }

    /// Copy-on-write the last (writable) block if it is shared: allocate a private
    /// block, returning `Some((old, new))` so the caller can copy the block's live
    /// contents old→new on the device. No-op (returns `None`) when the tail block
    /// is already private or the table is empty.
    pub fn unshare_tail(&mut self, alloc: &mut BlockAllocator) -> Option<(u32, u32)> {
        let last = *self.blocks.last()?;
        if alloc.refcount(last) <= 1 {
            return None;
        }
        let fresh = alloc.alloc().expect("COW needs a free block");
        *self.blocks.last_mut().unwrap() = fresh;
        alloc.decref(last);
        Some((last, fresh))
    }

    /// Rebuild a table over blocks the caller ALREADY owns one reference on
    /// each of - the swap-in half of host-RAM KV offload
    /// ([`crate::kv_offload`]): [`crate::kv_offload::KvOffload::promote_kv`]
    /// allocates fresh blocks, restores their bytes, and hands the ownership
    /// of those references to the table it builds here.
    ///
    /// Deliberately NOT [`Self::adopt_prefix`], which increfs (it *shares*
    /// blocks that stay owned by the prefix cache) and forces the length to a
    /// whole number of blocks. A restored sequence owns its blocks outright
    /// and its length is whatever it was when it was demoted - usually a
    /// partially-filled tail block. `kv_namespace` is the one the sequence was
    /// demoted from: the restored bytes are only valid under it.
    pub fn restore(kv_namespace: KvNamespace, blocks: Vec<u32>, len: u32) -> BlockTable {
        assert!(!blocks.is_empty() || len == 0, "restore: {len} tokens cannot live in zero blocks");
        BlockTable { blocks, len, kv_namespace, state_row: None, cached_rows: 0 }
    }

    /// Start a FRESH table from shared full prefix blocks (incref each): the
    /// prefix-cache hit path. The table's length becomes `blocks * block_size`
    /// — adopted blocks are always full — and the next append/reserve writes
    /// after them.
    pub fn adopt_prefix(&mut self, blocks: &[u32], alloc: &mut BlockAllocator) {
        assert!(self.is_empty(), "adopt_prefix expects a fresh sequence");
        for &b in blocks {
            alloc.incref(b);
            self.blocks.push(b);
        }
        self.len = blocks.len() as u32 * alloc.block_size();
        self.cached_rows = self.len;
    }
}

impl BlockTable {
    /// Start a FRESH table that shares the blocks holding the first `len` tokens of a cached sequence (incref each).
    /// Unlike [`Self::adopt_prefix`] the last shared block may be partly filled; the caller makes it private with
    /// [`Self::unshare_tail`] before writing past `len`.
    pub fn adopt_rows(&mut self, blocks: &[u32], len: u32, alloc: &mut BlockAllocator) {
        assert!(self.is_empty(), "adopt_rows expects a fresh sequence");
        assert_eq!(blocks.len() as u32, len.div_ceil(alloc.block_size()), "adopt_rows: {len} tokens live in exactly {} blocks", len.div_ceil(alloc.block_size()));
        for &b in blocks {
            alloc.incref(b);
            self.blocks.push(b);
        }
        self.len = len;
        self.cached_rows = len;
    }
}

/// Content-addressed index of FULL, immutable KV blocks for prompt-prefix
/// reuse. An entry maps `(namespace, parent physical block, this block's
/// token ids)` to the physical block holding that prefix's K/V.
///
/// The namespace is part of the key at the root of every chain, so identical
/// prompts under the base model and under different adapters index disjoint
/// chains: a lookup in one namespace can never reach another's blocks (and
/// below the root, chaining by physical parent keeps it that way).
///
/// Chaining by the parent's *physical identity* rather than by a rolling hash
/// makes a hit exact **by construction**: two different prefixes can never
/// alias, because reaching an entry requires holding its actual parent block —
/// there is no hash to collide. This is the invariant the warm-vs-cold
/// identity test pins: a cache hit MUST produce byte-identical KV.
///
/// The cache holds one reference on every indexed block, so a cached block
/// never returns to the free list while indexed; [`PrefixCache::evict`]
/// releases least-recently-used chain tails under pool pressure. Only entries
/// whose block is cache-only (refcount 1) are evictable — a block still backing
/// a live sequence stays.
#[derive(Debug, Default)]
pub struct PrefixCache {
    map: std::collections::HashMap<PrefixKey, PrefixEntry>,
    /// Reverse index for eviction bookkeeping.
    by_block: std::collections::HashMap<u32, PrefixKey>,
    tick: u64,
}

/// `(namespace, parent physical block or [`PREFIX_ROOT`], token ids)`.
type PrefixKey = (KvNamespace, u32, Vec<u32>);

#[derive(Debug)]
struct PrefixEntry {
    block: u32,
    last_use: u64,
}

/// The "no parent" sentinel: the first block of a sequence chains off this.
pub const PREFIX_ROOT: u32 = u32::MAX;

impl PrefixCache {
    pub fn new() -> PrefixCache {
        PrefixCache::default()
    }

    /// Cached full blocks currently indexed.
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The longest cached chain matching `prompt`'s leading full blocks,
    /// covering at most `max_tokens` tokens. Returns the physical blocks to
    /// adopt (possibly empty). Touches each hit's LRU stamp.
    pub fn lookup(&mut self, prompt: &[u32], block_size: u32, max_tokens: usize) -> Vec<u32> {
        self.lookup_in(BASE_KV_NAMESPACE, prompt, block_size, max_tokens)
    }

    /// [`Self::lookup`] within `namespace`: only chains cached under that
    /// namespace are visible.
    pub fn lookup_in(&mut self, namespace: KvNamespace, prompt: &[u32], block_size: u32, max_tokens: usize) -> Vec<u32> {
        let bs = block_size as usize;
        let mut parent = PREFIX_ROOT;
        let mut out = Vec::new();
        for chunk in prompt.chunks_exact(bs) {
            if (out.len() + 1) * bs > max_tokens {
                break;
            }
            match self.map.get_mut(&(namespace, parent, chunk.to_vec())) {
                Some(e) => {
                    self.tick += 1;
                    e.last_use = self.tick;
                    parent = e.block;
                    out.push(e.block);
                }
                None => break,
            }
        }
        out
    }

    /// Index `prompt`'s full blocks from block index `from` onward, whose K/V
    /// now live in `blocks[from..]` (the sequence's own freshly-prefilled
    /// blocks). Takes a cache reference on each newly indexed block. Blocks
    /// `0..from` must be the chain a preceding [`Self::lookup`] returned.
    pub fn insert_chain(&mut self, prompt: &[u32], blocks: &[u32], from: usize, alloc: &mut BlockAllocator) {
        self.insert_chain_in(BASE_KV_NAMESPACE, prompt, blocks, from, alloc)
    }

    /// [`Self::insert_chain`] within `namespace`: `blocks` must hold K/V
    /// computed in that namespace, and the preceding lookup must have been
    /// made in it too.
    pub fn insert_chain_in(
        &mut self,
        namespace: KvNamespace,
        prompt: &[u32],
        blocks: &[u32],
        from: usize,
        alloc: &mut BlockAllocator,
    ) {
        let bs = alloc.block_size() as usize;
        let full = (prompt.len() / bs).min(blocks.len());
        let mut parent = if from == 0 { PREFIX_ROOT } else { blocks[from - 1] };
        for i in from..full {
            let key = (namespace, parent, prompt[i * bs..(i + 1) * bs].to_vec());
            if let Some(e) = self.map.get(&key) {
                // An identical prefix was cached by another sequence; keep the
                // canonical chain (later lookups follow the cached block).
                parent = e.block;
                continue;
            }
            let b = blocks[i];
            if self.by_block.contains_key(&b) {
                // Already indexed under a different key — cannot happen for a
                // freshly prefilled block, but never double-index.
                parent = b;
                continue;
            }
            alloc.incref(b);
            self.tick += 1;
            self.map.insert(key.clone(), PrefixEntry { block: b, last_use: self.tick });
            self.by_block.insert(b, key);
            parent = b;
        }
    }

    /// [`Self::lookup_in`] without refreshing recency: how much of `prompt`
    /// would hit, for a caller that only needs to know (admission).
    pub fn peek_in(&self, namespace: KvNamespace, prompt: &[u32], block_size: u32, max_tokens: usize) -> Vec<u32> {
        let bs = block_size as usize;
        let mut parent = PREFIX_ROOT;
        let mut out = Vec::new();
        for chunk in prompt.chunks_exact(bs) {
            if (out.len() + 1) * bs > max_tokens {
                break;
            }
            match self.map.get(&(namespace, parent, chunk.to_vec())) {
                Some(e) => {
                    parent = e.block;
                    out.push(e.block);
                }
                None => break,
            }
        }
        out
    }

    /// Release cache-only entries until `want` blocks have been freed or
    /// nothing more is evictable; returns how many were freed. See
    /// [`Self::evict_keeping`].
    pub fn evict(&mut self, want: u32, alloc: &mut BlockAllocator) -> u32 {
        self.evict_keeping(want, &[], alloc)
    }

    /// [`Self::evict`] that never frees a block of `keep` (the chain the
    /// caller is about to adopt).
    ///
    /// Only CACHE-ONLY entries (block refcount 1) are evictable, and only
    /// LEAVES: an entry other cached blocks chain off stays until they are
    /// gone, because freeing a chain's head strands every later block of it
    /// (unreachable by lookup, still occupying the pool) while freeing its
    /// tail merely shortens the prefix a later turn can reuse. Among leaves
    /// the least recently used goes first; a chain whose last child was just
    /// evicted becomes a leaf itself, so a cold conversation shrinks from the
    /// tail, one block at a time.
    pub fn evict_keeping(&mut self, want: u32, keep: &[u32], alloc: &mut BlockAllocator) -> u32 {
        use std::cmp::Reverse;
        let mut children: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for key in self.map.keys() {
            *children.entry(key.1).or_insert(0) += 1;
        }
        let evictable = |e: &PrefixEntry, children: &std::collections::HashMap<u32, u32>, alloc: &BlockAllocator| {
            alloc.refcount(e.block) == 1 && !keep.contains(&e.block) && !children.contains_key(&e.block)
        };
        let mut leaves: std::collections::BinaryHeap<Reverse<(u64, u32)>> =
            self.map.values().filter(|e| evictable(e, &children, alloc)).map(|e| Reverse((e.last_use, e.block))).collect();
        let mut freed = 0u32;
        while freed < want {
            let Some(Reverse((_, block))) = leaves.pop() else { break };
            let key = self.by_block.remove(&block).expect("a leaf is indexed");
            self.map.remove(&key);
            alloc.decref(block);
            freed += 1;
            let parent = key.1;
            if parent == PREFIX_ROOT {
                continue;
            }
            let remaining = children.get_mut(&parent).expect("the evicted entry counted its parent");
            *remaining -= 1;
            if *remaining == 0 {
                children.remove(&parent);
                if let Some(pe) = self.by_block.get(&parent).and_then(|k| self.map.get(k)) {
                    if evictable(pe, &children, alloc) {
                        leaves.push(Reverse((pe.last_use, pe.block)));
                    }
                }
            }
        }
        freed
    }

    /// Drop every entry indexed under `namespace` - an adapter unload. The
    /// cache releases its reference on each block; a block no sequence still
    /// reads returns to the pool, one a live sequence reads stays allocated
    /// until that sequence releases it (it is merely no longer reachable by
    /// lookup). Returns how many blocks went back to the pool. Idempotent,
    /// and other namespaces are untouched.
    pub fn evict_namespace(&mut self, namespace: KvNamespace, alloc: &mut BlockAllocator) -> u32 {
        let free_before = alloc.free_blocks();
        let doomed: Vec<PrefixKey> = self.map.keys().filter(|k| k.0 == namespace).cloned().collect();
        for key in doomed {
            let entry = self.map.remove(&key).expect("key just listed");
            self.by_block.remove(&entry.block);
            alloc.decref(entry.block);
        }
        alloc.free_blocks() - free_before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_reuse() {
        let mut a = BlockAllocator::new(4, 16);
        assert_eq!(a.free_blocks(), 4);
        assert_eq!(a.free_tokens(), 64);
        let b0 = a.alloc().unwrap();
        let b1 = a.alloc().unwrap();
        assert_ne!(b0, b1);
        assert_eq!(a.free_blocks(), 2);
        a.decref(b0);
        assert_eq!(a.free_blocks(), 3);
        // b0 is reusable now.
        let b2 = a.alloc().unwrap();
        assert_eq!(b2, b0);
    }

    #[test]
    fn pool_exhaustion() {
        let mut a = BlockAllocator::new(2, 16);
        assert!(a.alloc().is_some());
        assert!(a.alloc().is_some());
        assert!(a.alloc().is_none());
    }

    #[test]
    fn block_table_grows_one_block_per_block_size_tokens() {
        let mut a = BlockAllocator::new(8, 4); // block_size 4
        let mut t = BlockTable::new();
        // 5 tokens → 2 blocks (4 + 1).
        let mut located = Vec::new();
        for _ in 0..5 {
            located.push(t.append(&mut a).unwrap());
        }
        assert_eq!(t.len(), 5);
        assert_eq!(t.blocks().len(), 2);
        // token 0..3 in block[0] offsets 0..3, token 4 in block[1] offset 0.
        assert_eq!(located[0], (t.blocks()[0], 0));
        assert_eq!(located[3], (t.blocks()[0], 3));
        assert_eq!(located[4], (t.blocks()[1], 0));
        // locate agrees.
        assert_eq!(t.locate(4, 4), (t.blocks()[1], 0));
        assert_eq!(a.free_blocks(), 6);
        t.release(&mut a);
        assert_eq!(a.free_blocks(), 8);
    }

    #[test]
    fn truncate_frees_tail_blocks() {
        let mut a = BlockAllocator::new(8, 4); // block_size 4
        let mut t = BlockTable::new();
        for _ in 0..10 {
            t.append(&mut a).unwrap(); // 10 tokens -> 3 blocks
        }
        assert_eq!(t.blocks().len(), 3);
        let free_before = a.free_blocks();
        // Roll back to 5 tokens -> 2 blocks; the 3rd block is freed.
        t.truncate(5, &mut a);
        assert_eq!(t.len(), 5);
        assert_eq!(t.blocks().len(), 2);
        assert_eq!(a.free_blocks(), free_before + 1);
        // Truncate within the last kept block frees nothing more.
        t.truncate(4, &mut a);
        assert_eq!(t.blocks().len(), 1);
    }

    #[test]
    fn reserve_prefill_chunk() {
        let mut a = BlockAllocator::new(8, 4);
        let mut t = BlockTable::new();
        let start = t.reserve(6, &mut a).unwrap(); // 6 tokens → 2 blocks
        assert_eq!(start, 0);
        assert_eq!(t.len(), 6);
        assert_eq!(t.blocks().len(), 2);
        // append continues from there into the partially-filled 2nd block.
        let (blk, off) = t.append(&mut a).unwrap();
        assert_eq!((blk, off), (t.blocks()[1], 2));
    }

    /// A prefix hit must be exact: same tokens under a DIFFERENT parent are a
    /// different prefix and must not alias — the chain is keyed by physical
    /// parent identity, so there is no hash to collide.
    #[test]
    fn prefix_cache_is_exact_by_construction() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();

        // Sequence A: prompt [1..8] -> two full blocks, cached.
        let pa: Vec<u32> = (1..=8).collect();
        let mut ta = BlockTable::new();
        ta.reserve(8, &mut a).unwrap();
        cache.insert_chain(&pa, ta.blocks(), 0, &mut a);
        assert_eq!(cache.len(), 2);

        // Same first block -> hit; the second block diverges -> chain stops.
        let pb: Vec<u32> = vec![1, 2, 3, 4, 9, 9, 9, 9];
        assert_eq!(cache.lookup(&pb, 4, 8), vec![ta.blocks()[0]]);

        // The SECOND block of A has tokens [5,6,7,8]; a prompt STARTING with
        // those tokens shares no parent, so it must miss entirely.
        let pc: Vec<u32> = vec![5, 6, 7, 8];
        assert!(cache.lookup(&pc, 4, 4).is_empty(), "same tokens, different prefix: no hit");

        // Full match walks the whole chain, bounded by max_tokens.
        assert_eq!(cache.lookup(&pa, 4, 8), ta.blocks().to_vec());
        assert_eq!(cache.lookup(&pa, 4, 7).len(), 1, "max_tokens caps the chain");
    }

    /// The cache's reference keeps a block alive after its sequence releases
    /// it; adoption increfs; eviction only touches cache-only blocks, LRU
    /// first.
    #[test]
    fn prefix_cache_refcounts_and_lru_eviction() {
        let mut a = BlockAllocator::new(8, 4);
        let mut cache = PrefixCache::new();
        let prompt: Vec<u32> = (0..8).collect();
        let mut t = BlockTable::new();
        t.reserve(8, &mut a).unwrap();
        let blocks = t.blocks().to_vec();
        cache.insert_chain(&prompt, t.blocks(), 0, &mut a);
        assert_eq!(a.refcount(blocks[0]), 2, "sequence + cache");

        // The sequence finishes: blocks stay alive on the cache's reference.
        t.release(&mut a);
        assert_eq!(a.refcount(blocks[0]), 1);
        assert_eq!(a.free_blocks(), 6);

        // A new sequence adopts the cached chain.
        let mut t2 = BlockTable::new();
        let hits = cache.lookup(&prompt, 4, 8);
        t2.adopt_prefix(&hits, &mut a);
        assert_eq!(t2.len(), 8);
        assert_eq!(a.refcount(blocks[0]), 2);

        // Under pressure, only the cache-only block is evictable... and a
        // block still backing t2 stays even when asked for more.
        t2.truncate(4, &mut a); // t2 keeps only block 0
        let freed = cache.evict(2, &mut a);
        assert_eq!(freed, 1, "block 0 is live in t2; only block 1 was evictable");
        // Release t2; now block 0 (still cached) is evictable.
        t2.release(&mut a);
        assert_eq!(cache.evict(2, &mut a), 1);
        assert_eq!(a.free_blocks(), 8, "everything returned to the pool");
    }

    /// Cache `prompt` under `ns` the way an engine does after a cold prefill,
    /// returning the sequence's blocks.
    fn prefill_into(
        cache: &mut PrefixCache,
        a: &mut BlockAllocator,
        ns: KvNamespace,
        prompt: &[u32],
    ) -> BlockTable {
        let mut t = BlockTable::with_namespace(ns);
        t.reserve(prompt.len() as u32, a).unwrap();
        cache.insert_chain_in(ns, prompt, t.blocks(), 0, a);
        t
    }

    /// Under pressure the cache gives up the TAIL of the coldest conversation,
    /// not its head: evicting a chain's first block strands every later block
    /// (unreachable by lookup yet still occupying the pool), whereas dropping
    /// the tail keeps a shorter prefix that the conversation's next turn still
    /// reuses.
    #[test]
    fn eviction_takes_a_cold_chains_tail_and_keeps_its_head_reachable() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();
        let prompt: Vec<u32> = (1..=12).collect();
        let mut t = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &prompt);
        let blocks = t.blocks().to_vec();
        t.release(&mut a);
        assert_eq!(cache.evict(1, &mut a), 1);
        assert_eq!(cache.lookup(&prompt, 4, 12), blocks[..2].to_vec(), "the first two blocks are still a hit");
    }

    /// A request about to run must not lose its own earlier turn to the
    /// reclaim its admission triggers: with as many conversations as the pool
    /// holds, plain least-recently-used order evicts exactly the conversation
    /// whose next turn arrives soonest.
    #[test]
    fn eviction_keeping_spares_the_named_chain_even_when_it_is_the_coldest() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();
        let (mine, other): (Vec<u32>, Vec<u32>) = ((1..=8).collect(), (101..=108).collect());
        let mut t = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &mine);
        let mine_blocks = t.blocks().to_vec();
        t.release(&mut a);
        let mut t = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &other);
        t.release(&mut a);
        assert_eq!(cache.evict_keeping(2, &mine_blocks, &mut a), 2);
        assert_eq!(cache.lookup(&mine, 4, 8), mine_blocks, "my chain survived");
        assert!(cache.lookup(&other, 4, 8).is_empty(), "the other conversation paid");
        assert_eq!(cache.evict_keeping(2, &mine_blocks, &mut a), 0, "kept blocks are never freed");
    }

    /// Looking a chain up must not refresh it when only asking how much of a
    /// prompt would hit.
    #[test]
    fn peek_reports_the_hit_without_touching_recency() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();
        let (first, second): (Vec<u32>, Vec<u32>) = ((1..=4).collect(), (101..=104).collect());
        let mut t = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &first);
        let first_blocks = t.blocks().to_vec();
        t.release(&mut a);
        let mut t = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &second);
        t.release(&mut a);
        assert_eq!(cache.peek_in(BASE_KV_NAMESPACE, &first, 4, 4), first_blocks);
        assert_eq!(cache.evict(1, &mut a), 1);
        assert!(cache.peek_in(BASE_KV_NAMESPACE, &first, 4, 4).is_empty(), "the peeked chain stayed the coldest and was the one evicted");
    }

    /// The same prompt prefilled under the base, adapter A and adapter B is
    /// three different K/V computations: no namespace may adopt another's
    /// blocks, in either direction, and each still hits its own.
    #[test]
    fn prefix_cache_isolates_namespaces() {
        let mut a = BlockAllocator::new(32, 4);
        let mut cache = PrefixCache::new();
        let mut ids = NamespaceIds::default();
        let (adapter_a, adapter_b) = (ids.fresh(), ids.fresh());
        let prompt: Vec<u32> = (1..=8).collect();

        let tb = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &prompt);
        let ta = prefill_into(&mut cache, &mut a, adapter_a, &prompt);
        let tc = prefill_into(&mut cache, &mut a, adapter_b, &prompt);
        assert_eq!(cache.len(), 6, "each namespace indexes its own two blocks");

        for (ns, t) in [(BASE_KV_NAMESPACE, &tb), (adapter_a, &ta), (adapter_b, &tc)] {
            assert_eq!(cache.lookup_in(ns, &prompt, 4, 8), t.blocks().to_vec(), "ns {ns} hits itself");
        }
        let all = [tb.blocks(), ta.blocks(), tc.blocks()];
        for (i, x) in all.iter().enumerate() {
            for y in &all[i + 1..] {
                assert!(x.iter().all(|b| !y.contains(b)), "no physical block is shared across namespaces");
            }
        }

        // A namespace that cached nothing sees nothing, whatever the others hold.
        assert!(cache.lookup_in(ids.fresh(), &prompt, 4, 8).is_empty());
    }

    /// Everything in namespace 0 behaves as before namespaces existed: the
    /// namespace-free entry points ARE the base namespace.
    #[test]
    fn namespace_free_entry_points_are_the_base_namespace() {
        let mut a = BlockAllocator::new(8, 4);
        let mut cache = PrefixCache::new();
        let prompt: Vec<u32> = (1..=8).collect();
        let mut t = BlockTable::new();
        t.reserve(8, &mut a).unwrap();
        assert_eq!(t.kv_namespace(), BASE_KV_NAMESPACE);
        cache.insert_chain(&prompt, t.blocks(), 0, &mut a);
        assert_eq!(cache.lookup_in(BASE_KV_NAMESPACE, &prompt, 4, 8), t.blocks().to_vec());
        let mut u = BlockTable::new();
        u.reserve(8, &mut a).unwrap();
        cache.insert_chain_in(BASE_KV_NAMESPACE, &(9..=16).collect::<Vec<u32>>(), u.blocks(), 0, &mut a);
        assert_eq!(cache.lookup(&(9..=16).collect::<Vec<u32>>(), 4, 8), u.blocks().to_vec());
    }

    /// Unloading an adapter frees the blocks only it was keeping alive,
    /// leaves blocks a live sequence still reads until that sequence
    /// finishes, and never touches another namespace.
    #[test]
    fn evict_namespace_frees_that_namespaces_cached_blocks_only() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();
        let mut ids = NamespaceIds::default();
        let adapter = ids.fresh();
        let prompt: Vec<u32> = (1..=8).collect();
        let mut base = prefill_into(&mut cache, &mut a, BASE_KV_NAMESPACE, &prompt);
        let mut live = prefill_into(&mut cache, &mut a, adapter, &prompt);
        let mut done = prefill_into(&mut cache, &mut a, adapter, &(11..=18).collect::<Vec<u32>>());
        done.release(&mut a);
        assert_eq!(cache.len(), 6);
        let free_before = a.free_blocks();

        // `done` was cache-only (2 blocks); `live` still backs a sequence.
        assert_eq!(cache.evict_namespace(adapter, &mut a), 2);
        assert_eq!(a.free_blocks(), free_before + 2);
        assert_eq!(cache.len(), 2, "only the base entries remain");
        assert!(cache.lookup_in(adapter, &prompt, 4, 8).is_empty());
        assert_eq!(cache.lookup_in(BASE_KV_NAMESPACE, &prompt, 4, 8), base.blocks().to_vec());

        // The live sequence keeps its blocks; releasing it returns them.
        live.release(&mut a);
        base.release(&mut a);
        assert_eq!(cache.evict(8, &mut a), 2);
        assert_eq!(a.free_blocks(), 16, "nothing leaked");
        assert_eq!(cache.evict_namespace(adapter, &mut a), 0, "idempotent");
    }

    /// A namespace id handed out after an unload never equals an earlier one,
    /// so a stale id can never reach a newer adapter's blocks.
    #[test]
    fn fresh_namespace_never_aliases_an_evicted_one() {
        let mut a = BlockAllocator::new(16, 4);
        let mut cache = PrefixCache::new();
        let mut ids = NamespaceIds::default();
        let old = ids.fresh();
        assert_ne!(old, BASE_KV_NAMESPACE);
        let prompt: Vec<u32> = (1..=8).collect();
        let mut t = prefill_into(&mut cache, &mut a, old, &prompt);
        t.release(&mut a);
        cache.evict_namespace(old, &mut a);
        let new = ids.fresh();
        assert!(new > old);
        // Even without the eviction the new id would not see the old chain.
        let mut t = prefill_into(&mut cache, &mut a, old, &prompt);
        assert!(cache.lookup_in(new, &prompt, 4, 8).is_empty());
        t.release(&mut a);
    }

    /// The namespace is part of a sequence's identity: it survives fork,
    /// truncate and release (a released table is reused for nothing, but a
    /// forked child must read the parent's blocks under the same adapter) and
    /// a KV swap-in restores it.
    #[test]
    fn block_table_namespace_travels_with_the_sequence() {
        let mut a = BlockAllocator::new(8, 4);
        let mut t = BlockTable::with_namespace(7);
        t.reserve(6, &mut a).unwrap();
        let child = t.fork(&mut a);
        assert_eq!(child.kv_namespace(), 7);
        t.release(&mut a);
        assert_eq!(t.kv_namespace(), 7);
        let blocks = child.blocks().to_vec();
        assert_eq!(BlockTable::restore(7, blocks, 6).kv_namespace(), 7);
    }

    #[test]
    fn fork_shares_blocks_and_cow_unshares_tail() {
        let mut a = BlockAllocator::new(8, 4);
        let mut parent = BlockTable::new();
        for _ in 0..6 {
            parent.append(&mut a).unwrap();
        }
        let free_before = a.free_blocks();
        let mut child = parent.fork(&mut a);
        // Sharing frees nothing but bumps refcounts.
        assert_eq!(a.free_blocks(), free_before);
        assert_eq!(child.blocks(), parent.blocks());
        assert_eq!(a.refcount(parent.blocks()[0]), 2);
        // Child diverges: COW the tail block (shared, rc 2).
        let cow = child.unshare_tail(&mut a);
        let (old, new) = cow.expect("tail was shared");
        assert_eq!(old, parent.blocks()[1]);
        assert_ne!(new, old);
        assert_eq!(*child.blocks().last().unwrap(), new);
        // The parent's tail is now private again.
        assert_eq!(a.refcount(old), 1);
        // Full prefix block[0] is still shared by both.
        assert_eq!(a.refcount(parent.blocks()[0]), 2);
    }

    /// A recurrent-state row belongs to one live sequence: a fork shares its
    /// KV blocks but must not share (and then overwrite) its state row.
    #[test]
    fn a_state_row_is_owned_by_one_sequence_and_not_forked() {
        let mut a = BlockAllocator::new(8, 4);
        let mut parent = BlockTable::new();
        assert_eq!(parent.state_row(), None);
        parent.set_state_row(3);
        parent.append(&mut a).unwrap();
        let child = parent.fork(&mut a);
        assert_eq!(parent.state_row(), Some(3));
        assert_eq!(child.state_row(), None);
        assert_eq!(parent.take_state_row(), Some(3));
        assert_eq!(parent.state_row(), None);
    }
}

#[cfg(test)]
mod gpu_tests {
    use data::rng::Rng;

    const SCORES: usize = 0;
    const SOFTMAX: usize = 1;
    const APPLY: usize = 2;
    const P_SCORES: usize = 3;
    const P_APPLY: usize = 4;

    static PIPES: &[(&str, &str)] = &[
        ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
        ("decode_softmax", kernels::DECODE_SOFTMAX),
        ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
        ("paged_decode_scores", kernels::PAGED_DECODE_SCORES),
        ("paged_decode_apply", kernels::PAGED_DECODE_APPLY),
    ];

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    /// The paged decode attention must reproduce the contiguous decode attention
    /// bit-for-bit on identical K/V — the block-table indirection only changes
    /// *where* keys/values live, not the math. K/V are scattered into a SCRAMBLED
    /// physical block order to genuinely exercise the mapping.
    #[test]
    fn paged_attention_matches_contiguous() {
        let g = gpu_core::testgpu::dev(PIPES);
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let hkv = nkv * hd;
        let hq = nh * hd;
        let (t, cap, bs, num_blocks) = (20u32, 64u32, 4u32, 16u32);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let mut rng = Rng::new(data::rng::random_seed());
        let q: Vec<f32> = (0..hq).map(|_| rng.next_gaussian() as f32).collect();
        let kk: Vec<f32> = (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect();
        let vv: Vec<f32> = (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect();

        // --- contiguous reference ---
        let qb = g.storage_init("q", &q);
        let kc = g.storage_init("kc", &kk);
        let vc = g.storage_init("vc", &vv);
        let sc = g.storage((nh * cap) as u64);
        let pr = g.storage((nh * cap) as u64);
        let ctxc = g.storage(hq as u64);
        let steps = vec![
            g.step(SCORES, &[&qb, &kc, &sc], &[nh, group, hd, t, cap, hkv, fb(scale)], nh * t),
            g.step(SOFTMAX, &[&sc, &pr], &[nh, t, cap], nh),
            g.step(APPLY, &[&pr, &vc, &ctxc], &[nh, group, hd, t, cap, hkv], nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_contig = g.read(&ctxc, hq as usize);

        // --- paged: scatter the SAME K/V into a pool via a reversed block table ---
        let nblk = t.div_ceil(bs);
        let block_table: Vec<u32> = (0..nblk).map(|i| num_blocks - 1 - i).collect();
        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for tok in 0..t {
            let phys = block_table[(tok / bs) as usize];
            let dst = ((phys * bs + tok % bs) * hkv) as usize;
            let src = (tok * hkv) as usize;
            pk[dst..dst + hkv as usize].copy_from_slice(&kk[src..src + hkv as usize]);
            pv[dst..dst + hkv as usize].copy_from_slice(&vv[src..src + hkv as usize]);
        }
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let bt = g.storage(nblk as u64);
        g.write(&bt, &block_table);
        let scp = g.storage((nh * cap) as u64);
        let prp = g.storage((nh * cap) as u64);
        let ctxp = g.storage(hq as u64);
        let psteps = vec![
            g.step(P_SCORES, &[&qb, &poolk, &bt, &scp], &[nh, group, hd, t, bs, hkv, cap, fb(scale)], nh * t),
            g.step(SOFTMAX, &[&scp, &prp], &[nh, t, cap], nh),
            g.step(P_APPLY, &[&prp, &poolv, &bt, &ctxp], &[nh, group, hd, t, bs, hkv, cap], nh * hd),
        ];
        g.submit(&[], &psteps);
        let ctx_paged = g.read(&ctxp, hq as usize);

        let err = ctx_contig.iter().zip(&ctx_paged).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged vs contiguous attention: maxabs = {err:e}");
        assert!(err < 1e-6, "paged vs contiguous attention maxabs={err}");
    }
}

#[cfg(test)]
mod batched_tests {
    use data::rng::Rng;

    static PIPES: &[(&str, &str)] = &[
        ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
        ("decode_softmax", kernels::DECODE_SOFTMAX),
        ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
        ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
        ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ];

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    /// Batched paged decode over sequences of DIFFERENT lengths must equal each
    /// sequence decoded on its own (bit-exact). Each sequence has its own query,
    /// length, and an interleaved (scrambled) block table sharing one pool.
    #[test]
    fn batched_paged_matches_per_sequence() {
        // Contiguous (ref) at 0..2, batched paged at 3..5.
        let g = gpu_core::testgpu::dev(PIPES);
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lens = [5u32, 12, 20];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 64u32;
        // REGRESSION guard for a dispatch-width bug in attention scratch
        // sizing: `max_bt` deliberately exceeds `ceil(max(lens)/bs)` (5) so `cap` is
        // strictly greater than every sequence's real length -- exactly the
        // shape (a buffer sized for the engine's served context, actual
        // sequences much shorter) the dispatch-width question was about.
        // Before this, `cap == max(lens)` exactly, so a kernel that
        // mistakenly used `cap` as a per-thread COMPUTE bound instead of a
        // pure addressing stride could never be caught here — the two values
        // were indistinguishable at cap==max(lens).
        let max_bt = 10u32; // 2x ceil(20/4) -- cap(40) > max(lens)(20)
        let cap = max_bt * bs; // 40

        let mut rng = Rng::new(data::rng::random_seed());
        // Per-seq queries + K/V.
        let qs: Vec<Vec<f32>> = (0..batch).map(|_| (0..hq).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let ks: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let vs: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

        // Interleaved (scrambled) physical block tables: seq b uses blocks b, b+3, b+6...
        let tables: Vec<Vec<u32>> = (0..batch).map(|b| (0..max_bt).map(|i| b + i * batch).collect()).collect();

        // Fill one shared pool from the per-seq K/V through the block tables.
        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for b in 0..batch as usize {
            for tok in 0..lens[b] {
                let phys = tables[b][(tok / bs) as usize];
                let dst = ((phys * bs + tok % bs) * hkv) as usize;
                let src = (tok * hkv) as usize;
                pk[dst..dst + hkv as usize].copy_from_slice(&ks[b][src..src + hkv as usize]);
                pv[dst..dst + hkv as usize].copy_from_slice(&vs[b][src..src + hkv as usize]);
            }
        }

        // --- batched paged ---
        let qflat: Vec<f32> = qs.iter().flatten().copied().collect();
        let qb = g.storage_init("q", &qflat);
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let btflat: Vec<u32> = (0..batch as usize).flat_map(|b| tables[b].clone()).collect();
        let bt = g.storage((batch * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &lens);
        let sc = g.storage((batch * nh * cap) as u64);
        let pr = g.storage((batch * nh * cap) as u64);
        let ctxb = g.storage((batch * hq) as u64);
        let steps = vec![
            g.step(3, &[&qb, &poolk, &bt, &sl, &sc], &[batch, nh, group, hd, bs, hkv, cap, max_bt, fb(scale)], batch * nh * cap),
            g.step(4, &[&sc, &sl, &pr], &[batch, nh, cap], batch * nh),
            g.step(5, &[&pr, &poolv, &bt, &sl, &ctxb], &[batch, nh, group, hd, bs, hkv, cap, max_bt], batch * nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_batched = g.read(&ctxb, (batch * hq) as usize);

        // --- reference: each seq contiguously on its own K/V ---
        let mut worst = 0f32;
        for b in 0..batch as usize {
            let t = lens[b];
            let qc = g.storage_init("qc", &qs[b]);
            let kc = g.storage_init("kc", &ks[b]);
            let vc = g.storage_init("vc", &vs[b]);
            let rcap = t.max(1);
            let scc = g.storage((nh * rcap) as u64);
            let prc = g.storage((nh * rcap) as u64);
            let ctxc = g.storage(hq as u64);
            let rs = vec![
                g.step(0, &[&qc, &kc, &scc], &[nh, group, hd, t, rcap, hkv, fb(scale)], nh * t),
                g.step(1, &[&scc, &prc], &[nh, t, rcap], nh),
                g.step(2, &[&prc, &vc, &ctxc], &[nh, group, hd, t, rcap, hkv], nh * hd),
            ];
            g.submit(&[], &rs);
            let refb = g.read(&ctxc, hq as usize);
            let got = &ctx_batched[b * hq as usize..(b + 1) * hq as usize];
            let e = refb.iter().zip(got).fold(0f32, |m, (a, c)| m.max((a - c).abs()));
            worst = worst.max(e);
        }
        println!("batched paged vs per-sequence: worst maxabs = {worst:e}");
        assert!(worst < 1e-6, "batched paged vs per-sequence maxabs={worst}");
    }
}

#[cfg(test)]
mod flash_tests {
    use data::rng::Rng;

    static PIPES: &[(&str, &str)] = &[
        ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
        ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        ("paged_flash_decode", kernels::PAGED_FLASH_DECODE),
        ("paged_flash_decode_i8", kernels::PAGED_FLASH_DECODE_I8),
        ("paged_flash_decode_split", kernels::PAGED_FLASH_DECODE_SPLIT),
        ("paged_flash_decode_combine", kernels::PAGED_FLASH_DECODE_COMBINE),
        ("paged_decode_scores_i8_batched", kernels::PAGED_DECODE_SCORES_I8_BATCHED),
        ("paged_decode_apply_i8_batched", kernels::PAGED_DECODE_APPLY_I8_BATCHED),
        ("paged_flash_decode_split_i8", kernels::PAGED_FLASH_DECODE_SPLIT_I8),
        ("paged_flash_decode_combine_i8", kernels::PAGED_FLASH_DECODE_COMBINE_I8),
    ];

    /// The shared test device for `pipes`, or `None` - reported as a skip with its reason - on a backend that does not
    /// run workgroup-barrier kernels (the CPU JIT, `BRAIN_BACKEND=cpu`): every kernel these tests gate is `@cpu no`.
    fn barrier_gpu(pipes: &'static [(&'static str, &'static str)]) -> Option<gpu_core::Gpu> {
        let g = gpu_core::testgpu::dev(pipes);
        if g.caps().workgroup_reductions {
            Some(g)
        } else {
            brain_testutil::skip_unavailable("paged flash kernels: this backend does not run workgroup-barrier kernels (they are GPU-only)");
            None
        }
    }

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    /// `paged_flash_decode` (one dispatch, online softmax, no materialised
    /// scores/probs) must numerically agree with the three-stage reference
    /// triad (`paged_decode_scores_batched` -> `decode_softmax_batched` ->
    /// `paged_decode_apply_batched`) it sits beside, never replaces, behind
    /// the M1.1 selector.
    ///
    /// NOT bit-identical - checked against source, not assumed: the fused
    /// kernel rescales its running sum/accumulator once per 16-key tile
    /// (textbook online softmax), while the triad computes one exact max over
    /// the WHOLE row before a single un-rescaled exp/sum pass, so the two
    /// reduction orders agree to within float error, not bit for bit - the
    /// same reason `flash_attn_causal_gqa` is checked against its
    /// materialized reference with an absolute-error gate, not `assert_eq`,
    /// in this crate's `block.rs` (`flash_causal_gqa_matches_materialized_
    /// gqa_fwd`), whose `t=100` choice this test also borrows.
    ///
    /// Run once as-is (wgpu, `gpu_core::testgpu::dev`) and once more with
    /// `BRAIN_DEVICE=vulkan` - the kernel's two GPU backends, `@cpu no`
    /// deliberately excluding the CPU JIT (see the kernel's own header).
    ///
    /// Sequence lengths deliberately straddle the kernel's own `BC=8` tile
    /// size (one key, under one tile, exactly one tile, one tile + 1, several
    /// tiles, and `t=100`). GQA (`n_kv_heads < n_heads`) exercises
    /// `hkv = h / group` the same way every other paged-attention test in
    /// this file does. Block tables are interleaved (scrambled) across ONE
    /// shared pool, `batched_paged_matches_per_sequence`'s own convention,
    /// so the paged indirection is genuinely exercised rather than
    /// degenerating to a contiguous layout.
    #[test]
    fn paged_flash_decode_matches_batched_triad() {
        let Some(g) = barrier_gpu(PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32; // >= ceil(100/4)
        let cap = max_bt * bs;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..batch * hq).map(|_| rng.next_gaussian() as f32).collect();
        let ks: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let vs: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

        // Interleaved (scrambled) physical block tables sharing one pool.
        let tables: Vec<Vec<u32>> = (0..batch).map(|b| (0..max_bt).map(|i| b + i * batch).collect()).collect();

        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for b in 0..batch as usize {
            for tok in 0..lens[b] {
                let phys = tables[b][(tok / bs) as usize];
                let dst = ((phys * bs + tok % bs) * hkv) as usize;
                let src = (tok * hkv) as usize;
                pk[dst..dst + hkv as usize].copy_from_slice(&ks[b][src..src + hkv as usize]);
                pv[dst..dst + hkv as usize].copy_from_slice(&vs[b][src..src + hkv as usize]);
            }
        }

        let qb = g.storage_init("q", &qflat);
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let btflat: Vec<u32> = (0..batch as usize).flat_map(|b| tables[b].clone()).collect();
        let bt = g.storage((batch * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &lens);

        // --- reference: the three-stage batched triad ---
        let sc = g.storage((batch * nh * cap) as u64);
        let pr = g.storage((batch * nh * cap) as u64);
        let ctx_ref_buf = g.storage((batch * hq) as u64);
        let steps = vec![
            g.step(0, &[&qb, &poolk, &bt, &sl, &sc], &[batch, nh, group, hd, bs, hkv, cap, max_bt, fb(scale)], batch * nh * cap),
            g.step(1, &[&sc, &sl, &pr], &[batch, nh, cap], batch * nh),
            g.step(2, &[&pr, &poolv, &bt, &sl, &ctx_ref_buf], &[batch, nh, group, hd, bs, hkv, cap, max_bt], batch * nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_ref = g.read(&ctx_ref_buf, (batch * hq) as usize);

        // --- one fused dispatch, no scores/probs buffer at all ---
        let ctx_flash_buf = g.storage((batch * hq) as u64);
        let fsteps = vec![g.dispatch(
            3,
            &[&qb, &poolk, &poolv, &bt, &sl, &ctx_flash_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(batch * nh),
        )];
        g.submit(&[], &fsteps);
        let ctx_flash = g.read(&ctx_flash_buf, (batch * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_flash).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_decode vs batched triad: worst maxabs = {worst:e}");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_decode vs batched triad maxabs={worst}");
    }

    /// M2.7's split-key two-pass FlashDecode (`paged_flash_decode_split` ->
    /// `paged_flash_decode_combine`) against the SAME three-stage batched
    /// triad reference `paged_flash_decode_matches_batched_triad` gates
    /// against - same scenario (shapes, GQA group, scrambled shared-pool
    /// block tables, the `[1, 7, 8, 9, 41, 100]` lengths straddling the
    /// kernel's own `BC=8` tile boundary), so any divergence is attributable
    /// to the split/combine change, not a different fixture.
    ///
    /// `tiles_per_split = 2` (16 tokens/split) is deliberately SMALL relative
    /// to every length in `lens` - `len=100` needs `ceil(13/2) = 7` splits,
    /// the largest in the batch, so `n_splits = 7` is dispatched for EVERY
    /// sequence in the batch; `len=1` (`ntiles=1`) only fills split 0, so
    /// splits 1..6 exercise `paged_flash_decode_split`'s own documented
    /// degenerate path (`kt0 >= ntiles`, the tile loop runs zero times) for
    /// six of this test's fourteen (2 short-length x 6 excess-split)
    /// combinations - not just the "one real split" case a larger
    /// `tiles_per_split` would collapse every length in this fixture down
    /// to.
    ///
    /// Same `1e-3` absolute-error bound every other fused-vs-triad gate in
    /// this file uses (`paged_flash_decode_matches_batched_triad`'s own
    /// comment names why: a reassociated online-softmax reduction is never
    /// bit-exact against the triad's exact-max-then-single-pass reference) -
    /// the split/combine two-pass merge is YET ANOTHER reassociation of the
    /// identical reduction (per-split instead of per-tile), so the same
    /// tolerance class applies, not a tighter or looser one.
    #[test]
    fn paged_flash_decode_split_matches_batched_triad() {
        let Some(g) = barrier_gpu(PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32; // >= ceil(100/4)
        let cap = max_bt * bs;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..batch * hq).map(|_| rng.next_gaussian() as f32).collect();
        let ks: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let vs: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

        let tables: Vec<Vec<u32>> = (0..batch).map(|b| (0..max_bt).map(|i| b + i * batch).collect()).collect();

        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for b in 0..batch as usize {
            for tok in 0..lens[b] {
                let phys = tables[b][(tok / bs) as usize];
                let dst = ((phys * bs + tok % bs) * hkv) as usize;
                let src = (tok * hkv) as usize;
                pk[dst..dst + hkv as usize].copy_from_slice(&ks[b][src..src + hkv as usize]);
                pv[dst..dst + hkv as usize].copy_from_slice(&vs[b][src..src + hkv as usize]);
            }
        }

        let qb = g.storage_init("q", &qflat);
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let btflat: Vec<u32> = (0..batch as usize).flat_map(|b| tables[b].clone()).collect();
        let bt = g.storage((batch * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &lens);

        // --- reference: the three-stage batched triad ---
        let sc = g.storage((batch * nh * cap) as u64);
        let pr = g.storage((batch * nh * cap) as u64);
        let ctx_ref_buf = g.storage((batch * hq) as u64);
        let steps = vec![
            g.step(0, &[&qb, &poolk, &bt, &sl, &sc], &[batch, nh, group, hd, bs, hkv, cap, max_bt, fb(scale)], batch * nh * cap),
            g.step(1, &[&sc, &sl, &pr], &[batch, nh, cap], batch * nh),
            g.step(2, &[&pr, &poolv, &bt, &sl, &ctx_ref_buf], &[batch, nh, group, hd, bs, hkv, cap, max_bt], batch * nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_ref = g.read(&ctx_ref_buf, (batch * hq) as usize);

        // --- split phase: partial (m, l, o) per (sequence, head, split) ---
        let tiles_per_split = 2u32;
        let bc = 8u32; // paged_flash_decode_split's own const BC
        let max_ntiles = lens.iter().map(|&t| t.div_ceil(bc)).max().unwrap();
        let n_splits = max_ntiles.div_ceil(tiles_per_split);

        let part_m = g.storage((batch * nh * n_splits) as u64);
        let part_l = g.storage((batch * nh * n_splits) as u64);
        let part_o = g.storage((batch * nh * n_splits * hd) as u64);
        let split_steps = vec![g.dispatch(
            5,
            &[&qb, &poolk, &poolv, &bt, &sl, &part_m, &part_l, &part_o],
            &[batch, nh, nkv, hd, group, bs, max_bt, n_splits, tiles_per_split],
            gpu_core::Dispatch::Workgroups(batch * nh * n_splits),
        )];
        g.submit(&[], &split_steps);

        // --- combine phase: merge the n_splits partials into one context ---
        let ctx_split_buf = g.storage((batch * hq) as u64);
        let combine_steps = vec![g.step(
            6,
            &[&part_m, &part_l, &part_o, &ctx_split_buf],
            &[batch, nh, hd, n_splits],
            batch * nh * 128, // 128 = paged_flash_decode_combine's own @workgroup_size
        )];
        g.submit(&[], &combine_steps);
        let ctx_split = g.read(&ctx_split_buf, (batch * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_split).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_decode_split+combine vs batched triad: worst maxabs = {worst:e} (n_splits={n_splits})");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_decode_split+combine vs batched triad maxabs={worst}");
    }

    /// One batched, GQA-aware int8 decode case: `lens` sequences of ragged
    /// length in a pool whose block tables span `max_bt * bs` tokens (far
    /// more than any live length), `kv_heads` x `group` heads of width `hd`.
    struct SplitI8Case {
        lens: Vec<u32>,
        kv_heads: u32,
        group: u32,
        hd: u32,
        max_bt: u32,
        n_splits: u32,
    }

    /// Runs `paged_flash_decode_split_i8` -> `paged_flash_decode_combine_i8`
    /// and the int8 triad (`paged_decode_scores_i8_batched` ->
    /// `decode_softmax_batched` -> `paged_decode_apply_i8_batched`) over the
    /// same randomly filled, really quantized (`quantize_pool_i8`) pool with
    /// scrambled block tables; returns the worst absolute difference of the
    /// two contexts. Both read the live length from `seq_lens`, so the
    /// per-sequence lengths below `cap` are what is being exercised.
    fn split_i8_vs_triad(c: &SplitI8Case) -> f32 {
        let g = gpu_core::testgpu::dev(PIPES);
        let (nkv, group, hd) = (c.kv_heads, c.group, c.hd);
        let nh = nkv * group;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let batch = c.lens.len() as u32;
        let bs = 16u32;
        let max_bt = c.max_bt;
        let cap = max_bt * bs;
        let max_len = *c.lens.iter().max().unwrap();
        assert!(max_len <= cap);
        let blocks_per_seq = max_len.div_ceil(bs);
        let num_blocks = batch * blocks_per_seq;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..batch * hq).map(|_| rng.next_gaussian() as f32).collect();
        // Blocks interleave across sequences so the paged indirection is
        // genuinely exercised. Entries past a sequence's live blocks stay 0
        // and must never be read.
        let tables: Vec<Vec<u32>> =
            (0..batch).map(|b| (0..max_bt).map(|i| if i < blocks_per_seq { b + i * batch } else { 0 }).collect()).collect();
        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for (table, &len) in tables.iter().zip(&c.lens) {
            for tok in 0..len {
                let dst = ((table[(tok / bs) as usize] * bs + tok % bs) * hkv) as usize;
                for e in 0..hkv as usize {
                    pk[dst + e] = rng.next_gaussian() as f32;
                    pv[dst + e] = rng.next_gaussian() as f32;
                }
            }
        }

        let qb = g.storage_init("q", &qflat);
        let btflat: Vec<u32> = tables.concat();
        let bt = g.storage(btflat.len() as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &c.lens);
        let (pk_words, sk) = quantize_pool_i8(&pk, nkv, hd);
        let (pv_words, sv) = quantize_pool_i8(&pv, nkv, hd);
        let poolk = g.storage(pk_words.len() as u64);
        g.write(&poolk, &pk_words);
        let poolv = g.storage(pv_words.len() as u64);
        g.write(&poolv, &pv_words);
        let scales_k = g.storage_init("sk", &sk);
        let scales_v = g.storage_init("sv", &sv);

        let sc = g.storage((batch * nh * cap) as u64);
        let pr = g.storage((batch * nh * cap) as u64);
        let ctx_ref_buf = g.storage((batch * hq) as u64);
        g.submit(
            &[],
            &[
                g.step(7, &[&qb, &poolk, &bt, &sl, &scales_k, &sc], &[batch, nh, group, hd, bs, hkv, cap, max_bt, fb(scale)], batch * nh * cap),
                g.step(1, &[&sc, &sl, &pr], &[batch, nh, cap], batch * nh),
                g.step(8, &[&pr, &poolv, &bt, &sl, &scales_v, &ctx_ref_buf], &[batch, nh, group, hd, bs, hkv, cap, max_bt], batch * nh * hd),
            ],
        );
        let ctx_ref = g.read(&ctx_ref_buf, (batch * hq) as usize);

        // One workgroup per (sequence, kv head, 4-head slab, split); the
        // partials are `[m, l, o_0..o_127]` per (sequence, head, split).
        let slabs = group.div_ceil(4);
        let part = g.storage((batch * nh * c.n_splits * (2 + 128)) as u64);
        let ctx_split_buf = g.storage((batch * hq) as u64);
        g.submit(
            &[],
            &[
                g.dispatch(
                    9,
                    &[&qb, &poolk, &poolv, &scales_k, &scales_v, &bt, &sl, &part],
                    &[batch, nh, nkv, hd, group, bs, max_bt, c.n_splits],
                    gpu_core::Dispatch::Workgroups(batch * nkv * slabs * c.n_splits),
                ),
                g.step(10, &[&part, &sl, &ctx_split_buf], &[batch, nh, hd, c.n_splits], batch * nh * 128),
            ],
        );
        let ctx_split = g.read(&ctx_split_buf, (batch * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_split).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(ctx_ref.iter().any(|v| v.abs() > 1e-3), "sanity: the reference context is not all zero");
        worst
    }

    /// The batched GQA split-key int8 decode kernel pair must agree with the
    /// int8 triad `qwen3::serve` falls back to. Covers the batch sizes a
    /// server presents (1, 4, 37, 128), ragged lengths including 1, tile and
    /// block boundaries (127/128/129, 15/16/17) and lengths far below `cap`,
    /// a split count that is one, a few and far more than any sequence has
    /// chunks (the surplus splits must exit without writing), GQA group 4 at
    /// head_dim 128, plus groups 1, 2 and 8 (one and two head slabs) and
    /// head_dim 64.
    #[test]
    fn paged_flash_decode_split_i8_matches_int8_triad() {
        if barrier_gpu(PIPES).is_none() {
            return;
        }
        let ragged = |n: u32| -> Vec<u32> {
            let fixed = [1u32, 2, 15, 16, 17, 127, 128, 129, 255, 256, 257, 700];
            (0..n).map(|i| fixed[(i as usize * 7 + 3) % fixed.len()]).collect()
        };
        let mut cases = Vec::new();
        for n_splits in [1u32, 3, 64] {
            cases.push(SplitI8Case { lens: vec![1], kv_heads: 2, group: 4, hd: 128, max_bt: 64, n_splits });
            cases.push(SplitI8Case { lens: vec![129], kv_heads: 2, group: 4, hd: 128, max_bt: 64, n_splits });
            cases.push(SplitI8Case { lens: vec![1, 127, 128, 129], kv_heads: 2, group: 4, hd: 128, max_bt: 64, n_splits });
            cases.push(SplitI8Case { lens: ragged(37), kv_heads: 2, group: 4, hd: 128, max_bt: 64, n_splits });
            cases.push(SplitI8Case { lens: ragged(128), kv_heads: 2, group: 4, hd: 128, max_bt: 64, n_splits });
        }
        for (group, hd) in [(1u32, 128u32), (2, 128), (8, 128), (4, 64)] {
            cases.push(SplitI8Case { lens: vec![1, 33, 300, 700], kv_heads: 2, group, hd, max_bt: 64, n_splits: 3 });
        }
        for c in &cases {
            let worst = split_i8_vs_triad(c);
            println!(
                "split_i8+combine vs int8 triad: bsz={} group={} hd={} n_splits={} worst maxabs={worst:e}",
                c.lens.len(),
                c.group,
                c.hd,
                c.n_splits
            );
            assert!(
                worst < 1e-3,
                "split_i8+combine vs int8 triad: bsz={} lens={:?} group={} hd={} n_splits={} maxabs={worst}",
                c.lens.len(),
                c.lens,
                c.group,
                c.hd,
                c.n_splits
            );
        }
    }

    /// Whole-tensor `rel_l2` (f64-accumulated sum-of-squares ratio) - the
    /// SAME formula `qwen3::serve`'s own `int8_kv_scale_and_bytes_match_a_
    /// host_oracle` gates at `< 0.01` ("the serving tolerance already used by
    /// kv_int8", M2.2's own gate wording), reused here rather than a fresh
    /// hand-fitted bound.
    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut sq_err, mut sq_mag) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            sq_err += (*x as f64 - *y as f64).powi(2);
            sq_mag += *x as f64 * *x as f64;
        }
        (sq_err / sq_mag.max(1e-12)).sqrt()
    }

    /// Per-`(physical slot, kv head)` symmetric int8 quantization of a flat
    /// `[num_slots, n_kv*head_dim]` pool - the identical `absmax/127` scale
    /// and round-clamp `qwen3::serve`'s real `paged_kv_append_i8_clipped_
    /// batched` path uses (pinned by `int8_kv_scale_and_bytes_match_a_host_
    /// oracle`), packed 4-per-`u32` the same way `paged_decode_scores_i8_
    /// batched`'s `pool` binding reads it. A never-written slot's row is all
    /// zero, so `absmax == 0` -> `scale = 1.0`, byte `0` -> dequants back to
    /// exactly `0.0`, matching the masked-key path unaffected either way.
    fn quantize_pool_i8(pool: &[f32], n_kv: u32, hd: u32) -> (Vec<u32>, Vec<f32>) {
        let hkv = (n_kv * hd) as usize;
        assert_eq!(pool.len() % hkv, 0, "pool length must be a whole number of {hkv}-wide rows");
        let num_slots = pool.len() / hkv;
        let mut bytes = vec![0u8; pool.len()];
        let mut scales = vec![0f32; num_slots * n_kv as usize];
        for slot in 0..num_slots {
            for h in 0..n_kv as usize {
                let base = slot * hkv + h * hd as usize;
                let row = &pool[base..base + hd as usize];
                let absmax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
                scales[slot * n_kv as usize + h] = scale;
                for (d, &v) in row.iter().enumerate() {
                    let q = (v / scale).round().clamp(-127.0, 127.0) as i32;
                    bytes[base + d] = (q as i8) as u8;
                }
            }
        }
        let words = bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, &b)| w | (u32::from(b) << (8 * i)))).collect();
        (words, scales)
    }

    /// `paged_flash_decode_i8` (M2.2's int8-KV twin) against `paged_flash_
    /// decode` itself - the fp32 FUSED kernel, per the milestone's own gate
    /// wording ("cosine/rel_l2 vs the fp32 fused kernel"), not the three-stage
    /// triad. Same shapes/scrambled block tables as `paged_flash_decode_
    /// matches_batched_triad` above; the pool is quantized with
    /// [`quantize_pool_i8`] (the real production scale/round/clamp scheme),
    /// so the only source of disagreement is genuine int8 quantization noise,
    /// not a synthetic one.
    #[test]
    fn paged_flash_decode_int8_matches_fp32_fused_kernel() {
        let Some(g) = barrier_gpu(PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32; // >= ceil(100/4)

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..batch * hq).map(|_| rng.next_gaussian() as f32).collect();
        let ks: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let vs: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

        let tables: Vec<Vec<u32>> = (0..batch).map(|b| (0..max_bt).map(|i| b + i * batch).collect()).collect();

        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for b in 0..batch as usize {
            for tok in 0..lens[b] {
                let phys = tables[b][(tok / bs) as usize];
                let dst = ((phys * bs + tok % bs) * hkv) as usize;
                let src = (tok * hkv) as usize;
                pk[dst..dst + hkv as usize].copy_from_slice(&ks[b][src..src + hkv as usize]);
                pv[dst..dst + hkv as usize].copy_from_slice(&vs[b][src..src + hkv as usize]);
            }
        }

        let qb = g.storage_init("q", &qflat);
        let poolk_f32 = g.storage_init("pk", &pk);
        let poolv_f32 = g.storage_init("pv", &pv);
        let btflat: Vec<u32> = (0..batch as usize).flat_map(|b| tables[b].clone()).collect();
        let bt = g.storage((batch * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &lens);

        // --- ground truth: the fp32 fused kernel ---
        let ctx_fp32_buf = g.storage((batch * hq) as u64);
        let fp32_steps = vec![g.dispatch(
            3,
            &[&qb, &poolk_f32, &poolv_f32, &bt, &sl, &ctx_fp32_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(batch * nh),
        )];
        g.submit(&[], &fp32_steps);
        let ctx_fp32 = g.read(&ctx_fp32_buf, (batch * hq) as usize);

        // --- the int8-KV twin, over a quantized copy of the SAME pool ---
        let (pk_words, sk) = quantize_pool_i8(&pk, nkv, hd);
        let (pv_words, sv) = quantize_pool_i8(&pv, nkv, hd);
        let poolk_i8 = g.storage(pk_words.len() as u64);
        g.write(&poolk_i8, &pk_words);
        let poolv_i8 = g.storage(pv_words.len() as u64);
        g.write(&poolv_i8, &pv_words);
        let scales_k = g.storage_init("sk", &sk);
        let scales_v = g.storage_init("sv", &sv);

        let ctx_i8_buf = g.storage((batch * hq) as u64);
        let i8_steps = vec![g.dispatch(
            4,
            &[&qb, &poolk_i8, &poolv_i8, &scales_k, &scales_v, &bt, &sl, &ctx_i8_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(batch * nh),
        )];
        g.submit(&[], &i8_steps);
        let ctx_i8 = g.read(&ctx_i8_buf, (batch * hq) as usize);

        let l2 = rel_l2(&ctx_fp32, &ctx_i8);
        let cos = crate::hostmath::cosine(&ctx_fp32, &ctx_i8);
        println!("paged_flash_decode_i8 vs fp32 fused: rel_l2={l2:.6} cosine={cos:.8}");
        assert!(l2 < 0.01, "paged_flash_decode_i8 vs fp32 fused: rel_l2={l2} too high");
        assert!((1.0 - cos) < 0.01, "paged_flash_decode_i8 vs fp32 fused: cosine={cos} too low");
    }

    /// `paged_flash_decode`'s `dtype_variant`-templated bf16 storage tier
    /// (M2.2) against the plain fp32 fused kernel - chaining `dtype_variant`
    /// twice over `pool_k` then `pool_v`, since (unlike the split scores/apply
    /// pair) this kernel reads both pools in one dispatch. Not registered in
    /// [`PIPES`] since the templated source only exists once `dtype_variant`
    /// runs (it is not a `.wgsl` file on disk); this test builds its own tiny
    /// pipeline list instead, the same "leak a runtime `Vec` to `'static`"
    /// pattern `dtype_variant`'s own cache already uses internally.
    #[test]
    fn paged_flash_decode_bf16_matches_fp32_fused_kernel() {
        use gpu_core::select::Dtype;

        let (n1, s1) =
            kernels::template::dtype_variant("paged_flash_decode", kernels::PAGED_FLASH_DECODE, "pool_k", Dtype::BF16).unwrap();
        let (bf16_name, bf16_src) = kernels::template::dtype_variant(n1, s1, "pool_v", Dtype::BF16).unwrap();

        let pipes: Vec<(&str, &str)> = vec![("paged_flash_decode", kernels::PAGED_FLASH_DECODE), (bf16_name, bf16_src)];
        let pipes: &'static [(&str, &str)] = Box::leak(pipes.into_boxed_slice());
        let Some(g) = barrier_gpu(pipes) else { return };

        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..batch * hq).map(|_| rng.next_gaussian() as f32).collect();
        let ks: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let vs: Vec<Vec<f32>> = lens.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

        let tables: Vec<Vec<u32>> = (0..batch).map(|b| (0..max_bt).map(|i| b + i * batch).collect()).collect();

        let mut pk = vec![0f32; (num_blocks * bs * hkv) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * hkv) as usize];
        for b in 0..batch as usize {
            for tok in 0..lens[b] {
                let phys = tables[b][(tok / bs) as usize];
                let dst = ((phys * bs + tok % bs) * hkv) as usize;
                let src = (tok * hkv) as usize;
                pk[dst..dst + hkv as usize].copy_from_slice(&ks[b][src..src + hkv as usize]);
                pv[dst..dst + hkv as usize].copy_from_slice(&vs[b][src..src + hkv as usize]);
            }
        }

        let qb = g.storage_init("q", &qflat);
        let poolk_f32 = g.storage_init("pk", &pk);
        let poolv_f32 = g.storage_init("pv", &pv);
        let btflat: Vec<u32> = (0..batch as usize).flat_map(|b| tables[b].clone()).collect();
        let bt = g.storage((batch * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(batch as u64);
        g.write(&sl, &lens);

        let ctx_fp32_buf = g.storage((batch * hq) as u64);
        let fp32_steps = vec![g.dispatch(
            0,
            &[&qb, &poolk_f32, &poolv_f32, &bt, &sl, &ctx_fp32_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(batch * nh),
        )];
        g.submit(&[], &fp32_steps);
        let ctx_fp32 = g.read(&ctx_fp32_buf, (batch * hq) as usize);

        // Same flat-index packing `model::half::pack_bf16`'s own convention
        // and `dtype_variant`'s decode agree on: element `2i` low half,
        // `2i+1` high half of word `i`.
        let pk_bf16 = crate::half::pack_bf16(&pk);
        let pv_bf16 = crate::half::pack_bf16(&pv);
        let poolk_bf16 = g.storage(pk_bf16.len() as u64);
        g.write(&poolk_bf16, &pk_bf16);
        let poolv_bf16 = g.storage(pv_bf16.len() as u64);
        g.write(&poolv_bf16, &pv_bf16);

        let ctx_bf16_buf = g.storage((batch * hq) as u64);
        let bf16_steps = vec![g.dispatch(
            1,
            &[&qb, &poolk_bf16, &poolv_bf16, &bt, &sl, &ctx_bf16_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(batch * nh),
        )];
        g.submit(&[], &bf16_steps);
        let ctx_bf16 = g.read(&ctx_bf16_buf, (batch * hq) as usize);

        let l2 = rel_l2(&ctx_fp32, &ctx_bf16);
        let cos = crate::hostmath::cosine(&ctx_fp32, &ctx_bf16);
        println!("paged_flash_decode#pool_k=bf16#pool_v=bf16 vs fp32 fused: rel_l2={l2:.6} cosine={cos:.8}");
        assert!(l2 < 0.01, "paged_flash_decode bf16 vs fp32 fused: rel_l2={l2} too high");
        assert!((1.0 - cos) < 0.01, "paged_flash_decode bf16 vs fp32 fused: cosine={cos} too low");
    }

    /// `paged_flash_prefill` (M2.3, one dispatch, `BR=64` query rows/workgroup,
    /// online softmax) must numerically agree with the SAME three-stage
    /// reference triad `paged_flash_decode_matches_batched_triad` above gates
    /// against - checked against `qwen3::serve::prefill`'s own source, not
    /// assumed: that function calls `run_batched` (hence the triad) once per
    /// CHUNK with `bsz` = chunk length and `seqlens[i] = start+i+1`, `bt`
    /// duplicated identically across every row of the chunk (one prefill
    /// dispatch is always ONE sequence). This test builds exactly that shape:
    /// a `start > 0` prefix (as if earlier chunks/`KV_APPEND` already wrote
    /// it) plus a `cc` = 130 row chunk spanning THREE `BR=64` query tiles, so
    /// the fused kernel's tile-boundary and causal-early-exit logic are both
    /// exercised, not just a single full tile.
    ///
    /// Own separate pipeline list (not [`PIPES`]): this is a fresh kernel set
    /// this test needs no other test's [`PIPES`] entries to spin up, and
    /// `gpu_core::testgpu::dev` shares the underlying device across distinct
    /// kernel sets anyway (keyed by the slice's own address, `new_like`
    /// parented onto whichever device is already live).
    ///
    /// Same `1e-3` absolute-error bound `paged_flash_decode_matches_batched_
    /// triad` uses, for the identical reason: the fused kernel rescales its
    /// running sum once per `BC=8`-key tile (online softmax) where the triad
    /// takes one exact max over the whole row first - not bit-identical, per
    /// this file's own repeated precedent.
    #[test]
    fn paged_flash_prefill_matches_batched_triad() {
        static PREFILL_PIPES: &[(&str, &str)] = &[
            ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
            ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
            ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
            ("paged_flash_prefill", kernels::PAGED_FLASH_PREFILL),
        ];
        let Some(g) = barrier_gpu(PREFILL_PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (kv_stride, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let start = 17u32; // tokens already cached by an earlier chunk
        let cc = 130u32; // this chunk's own rows - spans 3 BR=64 query tiles
        let total = start + cc; // live keys once this chunk's own K/V land
        let bs = 4u32;
        let num_blocks = 64u32;
        let max_bt = (total.div_ceil(bs) + 4).max(8); // headroom past what's used
        let cap = total;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..cc * hq).map(|_| rng.next_gaussian() as f32).collect();
        let kflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();
        let vflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();

        // A scrambled (reversed) logical->physical block mapping, shared by
        // every row of the chunk - `qwen3::serve::prefill`'s own
        // construction duplicates the SAME table across every row since a
        // chunk is always one sequence.
        let table: Vec<u32> = (0..max_bt).map(|lb| num_blocks - 1 - lb).collect();

        let mut pk = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        for tok in 0..total {
            let phys = table[(tok / bs) as usize];
            let dst = ((phys * bs + tok % bs) * kv_stride) as usize;
            let src = (tok * kv_stride) as usize;
            pk[dst..dst + kv_stride as usize].copy_from_slice(&kflat[src..src + kv_stride as usize]);
            pv[dst..dst + kv_stride as usize].copy_from_slice(&vflat[src..src + kv_stride as usize]);
        }

        let seqlens: Vec<u32> = (0..cc).map(|i| start + i + 1).collect(); // causal: row i attends 0..=start+i
        let btflat: Vec<u32> = (0..cc as usize).flat_map(|_| table.clone()).collect();

        let qb = g.storage_init("q", &qflat);
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let bt = g.storage((cc * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(cc as u64);
        g.write(&sl, &seqlens);

        // --- reference: the three-stage batched triad, `bsz = cc` rows ---
        let sc = g.storage((cc * nh * cap) as u64);
        let pr = g.storage((cc * nh * cap) as u64);
        let ctx_ref_buf = g.storage((cc * hq) as u64);
        let steps = vec![
            g.step(0, &[&qb, &poolk, &bt, &sl, &sc], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt, fb(scale)], cc * nh * cap),
            g.step(1, &[&sc, &sl, &pr], &[cc, nh, cap], cc * nh),
            g.step(2, &[&pr, &poolv, &bt, &sl, &ctx_ref_buf], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt], cc * nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_ref = g.read(&ctx_ref_buf, (cc * hq) as usize);

        // --- one fused dispatch per (head, query-tile), no scores/probs at all ---
        let ctx_flash_buf = g.storage((cc * hq) as u64);
        let ntiles_q = cc.div_ceil(64); // BR = paged_flash_prefill's own tile size
        let fsteps = vec![g.dispatch(
            3,
            &[&qb, &poolk, &poolv, &bt, &sl, &ctx_flash_buf],
            &[cc, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(nh * ntiles_q),
        )];
        g.submit(&[], &fsteps);
        let ctx_flash = g.read(&ctx_flash_buf, (cc * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_flash).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_prefill vs batched triad: worst maxabs = {worst:e}");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_prefill vs batched triad maxabs={worst}");
    }

    /// `paged_flash_prefill_i8` - the INT8-KV twin of `paged_flash_prefill`,
    /// gated against BOTH references that matter, in one test at one shape so
    /// the two bounds describe the same scenario:
    ///
    /// 1. **the int8 triad it actually replaces** (`paged_decode_scores_i8_
    ///    batched` -> `decode_softmax_batched` -> `paged_decode_apply_i8_
    ///    batched`, which is what `qwen3::serve`'s `kv_int8` causal-chunk
    ///    prefill dispatches today). Both sides read the IDENTICAL quantized
    ///    pool, so quantization is common-mode and the only residual is the
    ///    online softmax's reassociation - the same `1e-3` absolute bound
    ///    `paged_flash_prefill_matches_batched_triad` uses against the fp32
    ///    triad, for the identical reason.
    /// 2. **the fp32 fused kernel** over the unquantized pool - the
    ///    `rel_l2`/`cosine < 0.01` form `paged_flash_decode_int8_matches_fp32_
    ///    fused_kernel` establishes for the int8 decode twin, which bounds the
    ///    genuine int8 quantization error rather than the reassociation.
    ///
    /// Same scenario as `paged_flash_prefill_matches_batched_triad` above -
    /// a `start = 17` prefix already in the pool, a `cc = 130` row chunk
    /// spanning THREE `BR=64` query tiles, a scrambled block table shared by
    /// every row - so a divergence here is attributable to the int8 staging
    /// path and nothing else. The pool is quantized with [`quantize_pool_i8`]
    /// (the real production scale/round/clamp scheme), never a synthetic one.
    #[test]
    fn paged_flash_prefill_int8_matches_the_int8_triad_and_the_fp32_fused_kernel() {
        static PREFILL_I8_PIPES: &[(&str, &str)] = &[
            ("paged_decode_scores_i8_batched", kernels::PAGED_DECODE_SCORES_I8_BATCHED),
            ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
            ("paged_decode_apply_i8_batched", kernels::PAGED_DECODE_APPLY_I8_BATCHED),
            ("paged_flash_prefill_i8", kernels::PAGED_FLASH_PREFILL_I8),
            ("paged_flash_prefill", kernels::PAGED_FLASH_PREFILL),
        ];
        let Some(g) = barrier_gpu(PREFILL_I8_PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 8u32); // head_dim % 4 == 0: the packed-word contract
        let group = nh / nkv;
        let (kv_stride, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let start = 17u32;
        let cc = 130u32; // spans 3 BR=64 query tiles
        let total = start + cc;
        let bs = 4u32;
        let num_blocks = 64u32;
        let max_bt = (total.div_ceil(bs) + 4).max(8);
        let cap = total;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..cc * hq).map(|_| rng.next_gaussian() as f32).collect();
        let kflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();
        let vflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();

        let table: Vec<u32> = (0..max_bt).map(|lb| num_blocks - 1 - lb).collect();
        let mut pk = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        for tok in 0..total {
            let phys = table[(tok / bs) as usize];
            let dst = ((phys * bs + tok % bs) * kv_stride) as usize;
            let src = (tok * kv_stride) as usize;
            pk[dst..dst + kv_stride as usize].copy_from_slice(&kflat[src..src + kv_stride as usize]);
            pv[dst..dst + kv_stride as usize].copy_from_slice(&vflat[src..src + kv_stride as usize]);
        }

        let seqlens: Vec<u32> = (0..cc).map(|i| start + i + 1).collect();
        let btflat: Vec<u32> = (0..cc as usize).flat_map(|_| table.clone()).collect();

        let qb = g.storage_init("q", &qflat);
        let bt = g.storage((cc * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(cc as u64);
        g.write(&sl, &seqlens);

        let (pk_words, sk) = quantize_pool_i8(&pk, nkv, hd);
        let (pv_words, sv) = quantize_pool_i8(&pv, nkv, hd);
        let poolk_i8 = g.storage(pk_words.len() as u64);
        g.write(&poolk_i8, &pk_words);
        let poolv_i8 = g.storage(pv_words.len() as u64);
        g.write(&poolv_i8, &pv_words);
        let scales_k = g.storage_init("sk", &sk);
        let scales_v = g.storage_init("sv", &sv);

        // --- reference 1: the int8 triad, over the SAME quantized pool ---
        let sc = g.storage((cc * nh * cap) as u64);
        let pr = g.storage((cc * nh * cap) as u64);
        let ctx_triad_buf = g.storage((cc * hq) as u64);
        let triad = vec![
            g.step(0, &[&qb, &poolk_i8, &bt, &sl, &scales_k, &sc], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt, fb(scale)], cc * nh * cap),
            g.step(1, &[&sc, &sl, &pr], &[cc, nh, cap], cc * nh),
            g.step(2, &[&pr, &poolv_i8, &bt, &sl, &scales_v, &ctx_triad_buf], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt], cc * nh * hd),
        ];
        g.submit(&[], &triad);
        let ctx_triad = g.read(&ctx_triad_buf, (cc * hq) as usize);

        // --- the fused int8 dispatch, no scores/probs at all ---
        let ctx_i8_buf = g.storage((cc * hq) as u64);
        let ntiles_q = cc.div_ceil(64); // BR = paged_flash_prefill_i8's own tile size
        let fused = vec![g.dispatch(
            3,
            &[&qb, &poolk_i8, &poolv_i8, &scales_k, &scales_v, &bt, &sl, &ctx_i8_buf],
            &[cc, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(nh * ntiles_q),
        )];
        g.submit(&[], &fused);
        let ctx_i8 = g.read(&ctx_i8_buf, (cc * hq) as usize);

        let worst = ctx_triad.iter().zip(&ctx_i8).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_prefill_i8 vs int8 triad: worst maxabs = {worst:e}");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_prefill_i8 vs int8 triad maxabs={worst}");

        // --- reference 2: the fp32 fused kernel over the UNQUANTIZED pool ---
        let poolk_f32 = g.storage_init("pk", &pk);
        let poolv_f32 = g.storage_init("pv", &pv);
        let ctx_fp32_buf = g.storage((cc * hq) as u64);
        let fp32 = vec![g.dispatch(
            4,
            &[&qb, &poolk_f32, &poolv_f32, &bt, &sl, &ctx_fp32_buf],
            &[cc, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(nh * ntiles_q),
        )];
        g.submit(&[], &fp32);
        let ctx_fp32 = g.read(&ctx_fp32_buf, (cc * hq) as usize);

        let l2 = rel_l2(&ctx_fp32, &ctx_i8);
        let cos = crate::hostmath::cosine(&ctx_fp32, &ctx_i8);
        println!("paged_flash_prefill_i8 vs fp32 fused: rel_l2={l2:.6} cosine={cos:.8}");
        assert!(l2 < 0.01, "paged_flash_prefill_i8 vs fp32 fused: rel_l2={l2} too high");
        assert!((1.0 - cos) < 0.01, "paged_flash_prefill_i8 vs fp32 fused: cosine={cos} too low");
    }

    /// M2.5: `paged_flash_prefill_hd256` (a SEPARATE kernel file from
    /// `paged_flash_prefill`, see its own header for why) streams two
    /// `HD0=128`-wide head_dim fragments through the same `ksh`/`vsh`
    /// budget that kernel already uses, instead of doubling it - this test
    /// is the numerical proof that the streamed two-pass score/output
    /// accumulation is equivalent to a single full-head_dim pass, at the
    /// real `head_dim=256` shape `qwen35::config::Qwen35Config::
    /// qwen38_27b()` actually ships (checked against that constructor, not
    /// assumed). Otherwise an exact clone of `paged_flash_prefill_matches_
    /// batched_triad` above - same GQA ratio, same `start`/`cc` shape
    /// spanning three `BR=64` query tiles, same scrambled block table - only
    /// `hd` and the dispatched kernel differ, so any divergence here is
    /// attributable to the head_dim split, not a different scenario.
    ///
    /// Same `1e-3` absolute-error bound as every other fused-vs-triad gate
    /// in this file, for the identical reason: online softmax reassociates
    /// the reduction, so this is never bit-exact against the triad's exact-
    /// max-then-single-pass reference.
    #[test]
    fn paged_flash_prefill_hd256_matches_batched_triad_at_head_dim_256() {
        static PREFILL_HD256_PIPES: &[(&str, &str)] = &[
            ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
            ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
            ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
            ("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256),
        ];
        let Some(g) = barrier_gpu(PREFILL_HD256_PIPES) else { return };
        let (nh, nkv, hd) = (4u32, 2u32, 256u32); // qwen38_27b's own head_dim
        let group = nh / nkv;
        let (kv_stride, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let start = 17u32; // tokens already cached by an earlier chunk
        let cc = 130u32; // this chunk's own rows - spans 3 BR=64 query tiles
        let total = start + cc; // live keys once this chunk's own K/V land
        let bs = 4u32;
        let num_blocks = 64u32;
        let max_bt = (total.div_ceil(bs) + 4).max(8); // headroom past what's used
        let cap = total;

        let mut rng = Rng::new(data::rng::random_seed());
        let qflat: Vec<f32> = (0..cc * hq).map(|_| rng.next_gaussian() as f32).collect();
        let kflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();
        let vflat: Vec<f32> = (0..total * kv_stride).map(|_| rng.next_gaussian() as f32).collect();

        // A scrambled (reversed) logical->physical block mapping, shared by
        // every row of the chunk - same convention `paged_flash_prefill_
        // matches_batched_triad` already uses.
        let table: Vec<u32> = (0..max_bt).map(|lb| num_blocks - 1 - lb).collect();

        let mut pk = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        let mut pv = vec![0f32; (num_blocks * bs * kv_stride) as usize];
        for tok in 0..total {
            let phys = table[(tok / bs) as usize];
            let dst = ((phys * bs + tok % bs) * kv_stride) as usize;
            let src = (tok * kv_stride) as usize;
            pk[dst..dst + kv_stride as usize].copy_from_slice(&kflat[src..src + kv_stride as usize]);
            pv[dst..dst + kv_stride as usize].copy_from_slice(&vflat[src..src + kv_stride as usize]);
        }

        let seqlens: Vec<u32> = (0..cc).map(|i| start + i + 1).collect(); // causal: row i attends 0..=start+i
        let btflat: Vec<u32> = (0..cc as usize).flat_map(|_| table.clone()).collect();

        let qb = g.storage_init("q", &qflat);
        let poolk = g.storage_init("pk", &pk);
        let poolv = g.storage_init("pv", &pv);
        let bt = g.storage((cc * max_bt) as u64);
        g.write(&bt, &btflat);
        let sl = g.storage(cc as u64);
        g.write(&sl, &seqlens);

        // --- reference: the three-stage batched triad, `bsz = cc` rows ---
        let sc = g.storage((cc * nh * cap) as u64);
        let pr = g.storage((cc * nh * cap) as u64);
        let ctx_ref_buf = g.storage((cc * hq) as u64);
        let steps = vec![
            g.step(0, &[&qb, &poolk, &bt, &sl, &sc], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt, fb(scale)], cc * nh * cap),
            g.step(1, &[&sc, &sl, &pr], &[cc, nh, cap], cc * nh),
            g.step(2, &[&pr, &poolv, &bt, &sl, &ctx_ref_buf], &[cc, nh, group, hd, bs, kv_stride, cap, max_bt], cc * nh * hd),
        ];
        g.submit(&[], &steps);
        let ctx_ref = g.read(&ctx_ref_buf, (cc * hq) as usize);

        // --- one fused dispatch per (head, query-tile), head_dim streamed
        // as two 128-wide fragments, no scores/probs at all ---
        let ctx_flash_buf = g.storage((cc * hq) as u64);
        let ntiles_q = cc.div_ceil(64); // BR = paged_flash_prefill_hd256's own tile size
        let fsteps = vec![g.dispatch(
            3,
            &[&qb, &poolk, &poolv, &bt, &sl, &ctx_flash_buf],
            &[cc, nh, nkv, hd, group, bs, max_bt],
            gpu_core::Dispatch::Workgroups(nh * ntiles_q),
        )];
        g.submit(&[], &fsteps);
        let ctx_flash = g.read(&ctx_flash_buf, (cc * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_flash).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_prefill_hd256 vs batched triad: worst maxabs = {worst:e}");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_prefill_hd256 vs batched triad maxabs={worst}");
    }
}
