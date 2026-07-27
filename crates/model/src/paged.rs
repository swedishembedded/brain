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
    use super::*;
    use data::rng::Rng;
    use gpu_core::Gpu;

    const SCORES: usize = 0;
    const SOFTMAX: usize = 1;
    const APPLY: usize = 2;
    const P_SCORES: usize = 3;
    const P_APPLY: usize = 4;

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    /// The paged decode attention must reproduce the contiguous decode attention
    /// bit-for-bit on identical K/V — the block-table indirection only changes
    /// *where* keys/values live, not the math. K/V are scattered into a SCRAMBLED
    /// physical block order to genuinely exercise the mapping.
    #[test]
    fn paged_attention_matches_contiguous() {
        let pipes: &[(&str, &str)] = &[
            ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
            ("decode_softmax", kernels::DECODE_SOFTMAX),
            ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
            ("paged_decode_scores", kernels::PAGED_DECODE_SCORES),
            ("paged_decode_apply", kernels::PAGED_DECODE_APPLY),
        ];
        let g = Gpu::new(pipes);
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
    use super::*;
    use data::rng::Rng;
    use gpu_core::Gpu;

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    /// Batched paged decode over sequences of DIFFERENT lengths must equal each
    /// sequence decoded on its own (bit-exact). Each sequence has its own query,
    /// length, and an interleaved (scrambled) block table sharing one pool.
    #[test]
    fn batched_paged_matches_per_sequence() {
        // Contiguous (ref) at 0..2, batched paged at 3..5.
        let pipes: &[(&str, &str)] = &[
            ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
            ("decode_softmax", kernels::DECODE_SOFTMAX),
            ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
            ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
            ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
            ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        ];
        let g = Gpu::new(pipes);
        let (nh, nkv, hd) = (4u32, 2u32, 8u32);
        let group = nh / nkv;
        let (hkv, hq) = (nkv * hd, nh * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lens = [5u32, 12, 20];
        let batch = lens.len() as u32;
        let bs = 4u32;
        let num_blocks = 32u32;
        let max_bt = 5u32; // ceil(20/4)
        let cap = max_bt * bs; // 20

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
