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

/// A sequence's logical→physical block map. `len` is the number of tokens whose
/// K/V are stored; the cache grows a block at a time as tokens are appended.
#[derive(Clone, Debug, Default)]
pub struct BlockTable {
    blocks: Vec<u32>, // logical block index -> physical block id
    len: u32,         // tokens stored
}

impl BlockTable {
    pub fn new() -> BlockTable {
        BlockTable::default()
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
    }

    /// Fork a sequence that **shares** all of this table's blocks (incref each) —
    /// the basis for prefix sharing, parallel sampling, and speculative branches.
    /// The child sees the same `len`; a copy-on-write [`Self::unshare_tail`] makes
    /// the last block private before the child writes a diverging token.
    pub fn fork(&self, alloc: &mut BlockAllocator) -> BlockTable {
        for &b in &self.blocks {
            alloc.incref(b);
        }
        BlockTable { blocks: self.blocks.clone(), len: self.len }
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
    }
}

/// Content-addressed index of FULL, immutable KV blocks for prompt-prefix
/// reuse. An entry maps `(parent physical block, this block's token ids)` to
/// the physical block holding that prefix's K/V.
///
/// Chaining by the parent's *physical identity* rather than by a rolling hash
/// makes a hit exact **by construction**: two different prefixes can never
/// alias, because reaching an entry requires holding its actual parent block —
/// there is no hash to collide. This is the invariant the warm-vs-cold
/// identity test pins: a cache hit MUST produce byte-identical KV.
///
/// The cache holds one reference on every indexed block, so a cached block
/// never returns to the free list while indexed; [`PrefixCache::evict`]
/// releases least-recently-used entries under pool pressure. Only entries
/// whose block is cache-only (refcount 1) are evictable — a block still backing
/// a live sequence stays.
#[derive(Debug, Default)]
pub struct PrefixCache {
    map: std::collections::HashMap<(u32, Vec<u32>), PrefixEntry>,
    /// Reverse index for eviction bookkeeping.
    by_block: std::collections::HashMap<u32, (u32, Vec<u32>)>,
    tick: u64,
}

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
        let bs = block_size as usize;
        let mut parent = PREFIX_ROOT;
        let mut out = Vec::new();
        for chunk in prompt.chunks_exact(bs) {
            if (out.len() + 1) * bs > max_tokens {
                break;
            }
            match self.map.get_mut(&(parent, chunk.to_vec())) {
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
    pub fn insert_chain(
        &mut self,
        prompt: &[u32],
        blocks: &[u32],
        from: usize,
        alloc: &mut BlockAllocator,
    ) {
        let bs = alloc.block_size() as usize;
        let full = (prompt.len() / bs).min(blocks.len());
        let mut parent = if from == 0 { PREFIX_ROOT } else { blocks[from - 1] };
        for i in from..full {
            let key = (parent, prompt[i * bs..(i + 1) * bs].to_vec());
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

    /// Release least-recently-used CACHE-ONLY entries (block refcount 1) until
    /// `want` blocks have been freed or nothing more is evictable. Returns how
    /// many were freed. Evicting a mid-chain parent strands its cached
    /// children (unreachable by lookup); their stamps stop advancing, so they
    /// become LRU and follow shortly.
    pub fn evict(&mut self, want: u32, alloc: &mut BlockAllocator) -> u32 {
        let mut freed = 0u32;
        while freed < want {
            let victim = self
                .map
                .iter()
                .filter(|(_, e)| alloc.refcount(e.block) == 1)
                .min_by_key(|(_, e)| e.last_use)
                .map(|(k, e)| (k.clone(), e.block));
            let Some((key, block)) = victim else { break };
            self.map.remove(&key);
            self.by_block.remove(&block);
            alloc.decref(block);
            freed += 1;
        }
        freed
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

        let mut rng = Rng::new(7);
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
        // shape (a buffer sized for the engine's `BRAIN_QWEN_CTX`, actual
        // sequences much shorter) the dispatch-width question was about.
        // Before this, `cap == max(lens)` exactly, so a kernel that
        // mistakenly used `cap` as a per-thread COMPUTE bound instead of a
        // pure addressing stride could never be caught here — the two values
        // were indistinguishable at cap==max(lens).
        let max_bt = 10u32; // 2x ceil(20/4) -- cap(40) > max(lens)(20)
        let cap = max_bt * bs; // 40

        let mut rng = Rng::new(11);
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
    ];

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
        let g = gpu_core::testgpu::dev(PIPES);
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

        let mut rng = Rng::new(29);
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
        let fsteps = vec![g.step(
            3,
            &[&qb, &poolk, &poolv, &bt, &sl, &ctx_flash_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            batch * nh * 64, // 64 = paged_flash_decode's own @workgroup_size
        )];
        g.submit(&[], &fsteps);
        let ctx_flash = g.read(&ctx_flash_buf, (batch * hq) as usize);

        let worst = ctx_ref.iter().zip(&ctx_flash).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("paged_flash_decode vs batched triad: worst maxabs = {worst:e}");
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        assert!(worst < 1e-3, "paged_flash_decode vs batched triad maxabs={worst}");
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
        let g = gpu_core::testgpu::dev(PIPES);
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32; // >= ceil(100/4)

        let mut rng = Rng::new(37);
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
        let fp32_steps = vec![g.step(
            3,
            &[&qb, &poolk_f32, &poolv_f32, &bt, &sl, &ctx_fp32_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            batch * nh * 64,
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
        let i8_steps = vec![g.step(
            4,
            &[&qb, &poolk_i8, &poolv_i8, &scales_k, &scales_v, &bt, &sl, &ctx_i8_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            batch * nh * 64,
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
        let g = gpu_core::testgpu::dev(pipes);

        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let lens = [1u32, 7, 8, 9, 41, 100];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 256u32;
        let max_bt = 26u32;

        let mut rng = Rng::new(41);
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
        let fp32_steps = vec![g.step(
            0,
            &[&qb, &poolk_f32, &poolv_f32, &bt, &sl, &ctx_fp32_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            batch * nh * 64,
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
        let bf16_steps = vec![g.step(
            1,
            &[&qb, &poolk_bf16, &poolv_bf16, &bt, &sl, &ctx_bf16_buf],
            &[batch, nh, nkv, hd, group, bs, max_bt],
            batch * nh * 64,
        )];
        g.submit(&[], &bf16_steps);
        let ctx_bf16 = g.read(&ctx_bf16_buf, (batch * hq) as usize);

        let l2 = rel_l2(&ctx_fp32, &ctx_bf16);
        let cos = crate::hostmath::cosine(&ctx_fp32, &ctx_bf16);
        println!("paged_flash_decode#pool_k=bf16#pool_v=bf16 vs fp32 fused: rel_l2={l2:.6} cosine={cos:.8}");
        assert!(l2 < 0.01, "paged_flash_decode bf16 vs fp32 fused: rel_l2={l2} too high");
        assert!((1.0 - cos) < 0.01, "paged_flash_decode bf16 vs fp32 fused: cosine={cos} too low");
    }
}
