// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-RAM KV-cache offload for paged serving - architecture-agnostic, like
//! the rest of this crate ([`crate::paged`]'s allocator/block tables,
//! [`crate::serve`]'s `PagedDecoder` + `Scheduler<D>`). Nothing here names a
//! model: it is a byte-level swap of whole sequences' KV blocks between the
//! device pool and host memory, driven through one small device seam
//! ([`KvOffload`]) that any paged-KV engine can implement over its own pool
//! buffers. `crates/qwen3/src/serve.rs`'s `Engine` is the first adopter.
//!
//! Swedish Embedded AB implements memory-tiered long-context LLM serving for
//! its clients. If your team needs expertise in fitting more concurrent
//! sequences (or a longer context) onto the VRAM you already own, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Why WHOLE SEQUENCES, between scheduler turns - and never blocks inside
//! the decode loop
//!
//! The obvious design (an LRU over blocks, demote the ones nobody touched
//! recently) does not fit standard causal attention at all: **every** decode
//! step attends over the sequence's ENTIRE cached history, so no block of a
//! sequence being decoded is ever cold. A per-block scheme therefore does not
//! reduce the resident set of an active sequence - it streams that sequence's
//! whole KV across the host bus once per token.
//!
//! That is not a close call, on measured numbers from the box this was built
//! on (2x Tesla P40, `crates/gpu-core/tests/pcie_handoff.rs` for the bus,
//! `gpu_core::roof` for the card):
//!
//! | path | measured |
//! |---|---|
//! | device DRAM (`roof::Roofs::gbs`) | ~287 GB/s |
//! | host -> device (`Gpu::write_f32_chunked`, 216 MiB) | ~4.3 GB/s |
//! | device -> host (`Gpu::read`, 216 MiB) | ~1.2 GB/s |
//!
//! Reading a token's KV from VRAM is ~67x cheaper than fetching it across the
//! bus and ~230x cheaper than evicting it there. A Qwen3-8B sequence at its
//! full 40,960-token window holds 2.8 GiB of int8 KV; the attention that
//! reads it from VRAM costs ~10 ms, streaming it in would cost ~0.7 s - per
//! token. Per-block swap during active decode is decisively net-negative and
//! is deliberately not implemented.
//!
//! What DOES pay is the transfer whose cost is paid once per *scheduling
//! transition* rather than once per token: a sequence the scheduler is not
//! advancing this round has its whole KV moved to host RAM and its device
//! blocks freed for sequences that are running. The one-time swap cost is
//! amortised over every token the other sequences produce meanwhile, and host
//! RAM is large enough (184 GiB on this box against 2x24 GiB of VRAM) that
//! the number of ADMITTED sequences stops being bounded by VRAM at all. This
//! is the same shape production systems settled on, and it is what
//! [`crate::serve::Scheduler`]'s admission model already assumed: several
//! sequences admitted, not all of them necessarily dispatched every step.
//!
//! The alternative to swapping a preempted sequence out is dropping its KV
//! and re-prefilling it on resume. Swap wins whenever the prompt is long
//! enough that re-running its prefill costs more than moving its bytes, which
//! is exactly the long-context regime this exists for; recompute is not
//! implemented here (it needs no new mechanism - the scheduler could always
//! cancel and resubmit).

use std::collections::HashMap;

use crate::paged::{BlockAllocator, BlockTable};

/// Why a demote/promote could not be done. Every variant leaves the caller's
/// state exactly as it was - a refused swap is never a half-swapped sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvOffloadError {
    /// The host pool's byte budget cannot hold this sequence.
    HostPoolFull { need_bytes: u64, free_bytes: u64 },
    /// No demoted record under this key.
    NotOffloaded { key: u64 },
    /// The device pool has no room to bring this sequence back yet. The
    /// record is left in the host pool; the caller retries when blocks free.
    DevicePoolExhausted { need_blocks: u32, free_blocks: u32 },
}

impl std::fmt::Display for KvOffloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvOffloadError::HostPoolFull { need_bytes, free_bytes } => {
                write!(f, "host KV pool full: needs {need_bytes} bytes, {free_bytes} free")
            }
            KvOffloadError::NotOffloaded { key } => write!(f, "sequence {key} is not offloaded"),
            KvOffloadError::DevicePoolExhausted { need_blocks, free_blocks } => {
                write!(f, "device KV pool exhausted: needs {need_blocks} blocks, {free_blocks} free")
            }
        }
    }
}

/// Swap accounting - what a server reports and what a benchmark records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OffloadStats {
    /// Sequences demoted to host RAM.
    pub demotions: u64,
    /// Sequences promoted back to the device.
    pub promotions: u64,
    /// KV blocks whose bytes were copied device -> host.
    pub blocks_out: u64,
    /// KV blocks whose bytes were copied host -> device.
    pub blocks_in: u64,
    /// Demotes refused for want of host budget.
    pub refused: u64,
    /// Host bytes currently held by demoted sequences.
    pub bytes_resident: u64,
    /// High-water mark of [`Self::bytes_resident`].
    pub peak_bytes: u64,
    /// Sequences currently demoted.
    pub resident: u64,
}

/// One demoted sequence's KV, verbatim.
struct HostRecord {
    /// Tokens the sequence had cached - restored onto the new block table so
    /// a partially-filled tail block resumes exactly where it left off.
    len: u32,
    /// The device words of every block, in block order, `block_words` each.
    /// Raw `u32`, never `f32`: the pool's dtype is the engine's business
    /// (fp32, packed int8 plus its dequant scales, bf16 …) and a swap must
    /// round-trip the BYTES, not a numeric interpretation of them.
    words: Vec<u32>,
}

/// Host RAM holding demoted sequences' KV blocks, under a byte budget.
///
/// Keyed by whatever id the scheduler already has for a sequence - this pool
/// never invents its own handle, so a caller cannot end up with two names for
/// one sequence's cache.
pub struct HostKvPool {
    block_words: usize,
    capacity_bytes: u64,
    bytes: u64,
    records: HashMap<u64, HostRecord>,
    stats: OffloadStats,
}

impl HostKvPool {
    /// A pool holding at most `capacity_bytes` of demoted KV, whose blocks are
    /// `block_words` device words each (see
    /// [`KvOffload::kv_block_words`]). `capacity_bytes` of 0 disables
    /// offload - every demote is refused, which is what a decoder that does
    /// not want it (or a box with no RAM to spare) installs.
    pub fn new(block_words: usize, capacity_bytes: u64) -> HostKvPool {
        assert!(block_words > 0, "a KV block cannot be zero words wide");
        HostKvPool { block_words, capacity_bytes, bytes: 0, records: HashMap::new(), stats: OffloadStats::default() }
    }

    /// Bytes one block costs in host RAM - identical to what it costs on the
    /// device, since a swap is a verbatim copy.
    pub fn block_bytes(&self) -> u64 {
        self.block_words as u64 * 4
    }
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    pub fn bytes_resident(&self) -> u64 {
        self.bytes
    }
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.bytes)
    }
    /// Whether `blocks` more blocks would fit right now.
    pub fn admits(&self, blocks: usize) -> bool {
        blocks as u64 * self.block_bytes() <= self.free_bytes()
    }
    /// Is this sequence currently demoted?
    pub fn holds(&self, key: u64) -> bool {
        self.records.contains_key(&key)
    }
    /// Device blocks a demoted sequence needs to come back, `None` if it is
    /// not demoted - the scheduler's admission check before a promote.
    pub fn blocks_of(&self, key: u64) -> Option<u32> {
        self.records.get(&key).map(|r| (r.words.len() / self.block_words) as u32)
    }
    /// Sequences currently demoted.
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn stats(&self) -> OffloadStats {
        OffloadStats { bytes_resident: self.bytes, resident: self.records.len() as u64, ..self.stats }
    }

    /// Drop a demoted sequence's KV without restoring it - cancellation.
    /// `true` if there was one.
    pub fn discard(&mut self, key: u64) -> bool {
        match self.records.remove(&key) {
            Some(r) => {
                self.bytes -= (r.words.len() * 4) as u64;
                true
            }
            None => false,
        }
    }
}

/// The device seam a paged-KV engine implements to gain host-RAM offload.
///
/// Only the four `required` methods are engine-specific - how to move a set
/// of physical blocks' words between the pool buffers and the host, and where
/// this engine keeps its allocator and its host pool. [`Self::demote_kv`] /
/// [`Self::promote_kv`] / [`Self::discard_kv`] are then provided, so the
/// swap protocol (capacity checks, ordering, refcounts, rollback) has exactly
/// one implementation for every adopter.
///
/// The methods take `&mut self` one at a time rather than handing out
/// simultaneous `&mut` to the allocator, the host pool and the pool buffers -
/// which is what lets an engine hold all three as its own plain fields.
pub trait KvOffload {
    /// Device words one physical block costs, summed over everything the
    /// engine stores per block: every layer, K and V, plus any per-slot
    /// metadata (int8 dequant scales). Must be exactly what
    /// [`Self::read_kv_blocks`] emits per block.
    fn kv_block_words(&self) -> usize;

    /// The engine's block allocator.
    fn kv_alloc_mut(&mut self) -> &mut BlockAllocator;

    /// The engine's host pool.
    fn host_kv_mut(&mut self) -> &mut HostKvPool;

    /// Append the device words of every block in `blocks` (in that order,
    /// [`Self::kv_block_words`] each) to `out`.
    fn read_kv_blocks(&mut self, blocks: &[u32], out: &mut Vec<u32>);

    /// The inverse: write `words` (`blocks.len() * kv_block_words()` of them)
    /// back into those physical blocks, verbatim.
    fn write_kv_blocks(&mut self, blocks: &[u32], words: &[u32]);

    /// Copy `table`'s whole KV to host RAM and release its device blocks,
    /// under `key`. Returns how many blocks the device pool actually got back
    /// (a block still shared with the prefix cache or another sequence stays
    /// live - its bytes are copied all the same, so the restore is exact
    /// either way).
    ///
    /// `table` is left empty on success and untouched on every error.
    fn demote_kv(&mut self, key: u64, table: &mut BlockTable) -> Result<u32, KvOffloadError> {
        let blocks = table.blocks().to_vec();
        let len = table.len();
        if blocks.is_empty() {
            return Ok(0);
        }
        // Budget FIRST: refusing after the readback would have paid the
        // expensive half of the swap for nothing.
        if !self.host_kv_mut().admits(blocks.len()) {
            let pool = self.host_kv_mut();
            let (need_bytes, free_bytes) = (blocks.len() as u64 * pool.block_bytes(), pool.free_bytes());
            pool.stats.refused += 1;
            return Err(KvOffloadError::HostPoolFull { need_bytes, free_bytes });
        }
        let mut words = Vec::with_capacity(blocks.len() * self.kv_block_words());
        self.read_kv_blocks(&blocks, &mut words);
        debug_assert_eq!(words.len(), blocks.len() * self.kv_block_words(), "read_kv_blocks emitted the wrong width");

        let free_before = self.kv_alloc_mut().free_blocks();
        table.release(self.kv_alloc_mut());
        let reclaimed = self.kv_alloc_mut().free_blocks() - free_before;

        let bytes = (words.len() * 4) as u64;
        let pool = self.host_kv_mut();
        pool.records.insert(key, HostRecord { len, words });
        pool.bytes += bytes;
        pool.stats.peak_bytes = pool.stats.peak_bytes.max(pool.bytes);
        pool.stats.demotions += 1;
        pool.stats.blocks_out += blocks.len() as u64;
        Ok(reclaimed)
    }

    /// Bring a demoted sequence back: fresh device blocks, the exact bytes it
    /// had, and a block table resuming at the same length. The physical block
    /// ids are whatever the pool hands out now and are deliberately allowed to
    /// differ from the ones it left on - a paged cache is addressed through
    /// the block table, so identity of the *contents* is the invariant, not
    /// identity of the slots.
    ///
    /// On [`KvOffloadError::DevicePoolExhausted`] the record stays in the host
    /// pool and every block taken on the way is given back, so a failed
    /// promote costs nothing but the attempt.
    fn promote_kv(&mut self, key: u64) -> Result<BlockTable, KvOffloadError> {
        let Some(rec) = self.host_kv_mut().records.remove(&key) else {
            return Err(KvOffloadError::NotOffloaded { key });
        };
        let wpb = self.kv_block_words();
        let n = rec.words.len() / wpb;
        let mut blocks: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            match self.kv_alloc_mut().alloc() {
                Some(b) => blocks.push(b),
                None => {
                    // Roll back completely: give the blocks back, put the
                    // record back, report what was missing.
                    let free_blocks = self.kv_alloc_mut().free_blocks() + blocks.len() as u32;
                    for &b in &blocks {
                        self.kv_alloc_mut().decref(b);
                    }
                    self.host_kv_mut().records.insert(key, rec);
                    return Err(KvOffloadError::DevicePoolExhausted { need_blocks: n as u32, free_blocks });
                }
            }
        }
        self.write_kv_blocks(&blocks, &rec.words);
        let pool = self.host_kv_mut();
        pool.bytes -= (rec.words.len() * 4) as u64;
        pool.stats.promotions += 1;
        pool.stats.blocks_in += n as u64;
        Ok(BlockTable::restore(blocks, rec.len))
    }

    /// Drop a demoted sequence's KV without restoring it (cancellation).
    fn discard_kv(&mut self, key: u64) -> bool {
        self.host_kv_mut().discard(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device with no device: one flat `Vec<u32>` standing in for the
    /// engine's pool buffers, addressed exactly as a real one is
    /// (`block * block_words`). Enough to pin the swap protocol - refcounts,
    /// budget, rollback, and byte-exactness - with no GPU at all; the real
    /// device path is gated separately in `qwen3::serve`'s own tests.
    struct MockEngine {
        pool: Vec<u32>,
        block_words: usize,
        alloc: BlockAllocator,
        host: HostKvPool,
    }

    impl KvOffload for MockEngine {
        fn kv_block_words(&self) -> usize {
            self.block_words
        }
        fn kv_alloc_mut(&mut self) -> &mut BlockAllocator {
            &mut self.alloc
        }
        fn host_kv_mut(&mut self) -> &mut HostKvPool {
            &mut self.host
        }
        fn read_kv_blocks(&mut self, blocks: &[u32], out: &mut Vec<u32>) {
            for &b in blocks {
                let at = b as usize * self.block_words;
                out.extend_from_slice(&self.pool[at..at + self.block_words]);
            }
        }
        fn write_kv_blocks(&mut self, blocks: &[u32], words: &[u32]) {
            for (i, &b) in blocks.iter().enumerate() {
                let at = b as usize * self.block_words;
                let src = i * self.block_words;
                self.pool[at..at + self.block_words].copy_from_slice(&words[src..src + self.block_words]);
            }
        }
    }

    impl MockEngine {
        fn new(num_blocks: u32, block_size: u32, block_words: usize, host_bytes: u64) -> MockEngine {
            MockEngine {
                pool: vec![0; num_blocks as usize * block_words],
                block_words,
                alloc: BlockAllocator::new(num_blocks, block_size),
                host: HostKvPool::new(block_words, host_bytes),
            }
        }

        /// Fill a table's blocks with a per-(block, word) pattern that is
        /// deliberately NOT a valid float in places (high bit patterns,
        /// 0xFFFF_FFFF): a swap must move BYTES, so a path that ever routed
        /// them through an f32 (quieting a NaN, flushing a denormal) has to
        /// fail this.
        fn paint(&mut self, table: &BlockTable, seed: u32) {
            for (lb, &b) in table.blocks().iter().enumerate() {
                for w in 0..self.block_words {
                    let at = b as usize * self.block_words + w;
                    self.pool[at] = seed
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add((lb as u32).wrapping_mul(0x8000_0001))
                        .wrapping_add((w as u32).wrapping_mul(0x7FFF_FFFF));
                }
            }
        }

        fn snapshot(&self, table: &BlockTable) -> Vec<u32> {
            let mut out = Vec::new();
            for &b in table.blocks() {
                let at = b as usize * self.block_words;
                out.extend_from_slice(&self.pool[at..at + self.block_words]);
            }
            out
        }
    }

    /// **The correctness gate.** A demoted-and-restored sequence's KV must be
    /// byte-identical to what it had, the device blocks must really have gone
    /// back to the pool in between (that is the entire point), and the restore
    /// must not depend on landing on the same physical slots.
    #[test]
    fn demote_promote_round_trips_the_exact_bytes() {
        let (block_size, block_words) = (4u32, 7usize);
        let mut e = MockEngine::new(8, block_size, block_words, 1 << 20);

        let mut t = BlockTable::new();
        for _ in 0..10 {
            t.append(&mut e.alloc).unwrap(); // 10 tokens -> 3 blocks
        }
        e.paint(&t, 1234);
        let want = e.snapshot(&t);
        let (want_len, want_blocks) = (t.len(), t.blocks().to_vec());
        let free_before = e.alloc.free_blocks();

        let reclaimed = e.demote_kv(7, &mut t).unwrap();
        assert_eq!(reclaimed, 3, "every block was private, so all three come back");
        assert_eq!(e.alloc.free_blocks(), free_before + 3);
        assert!(t.is_empty() && t.blocks().is_empty(), "a demoted table holds nothing on the device");
        assert_eq!(e.host.bytes_resident(), 3 * (block_words as u64) * 4);
        assert!(e.host.holds(7));
        assert_eq!(e.host.blocks_of(7), Some(3));

        // Another sequence takes the freed blocks and scribbles over them, so
        // a restore that "worked" by leaving the old contents in place cannot
        // pass by accident. One block is PINNED by a third sequence that never
        // goes away, so the promote below is forced onto a different (and
        // differently-ordered) set of physical slots than the sequence left on.
        let mut pin = BlockTable::new();
        pin.append(&mut e.alloc).unwrap();
        let mut other = BlockTable::new();
        for _ in 0..10 {
            other.append(&mut e.alloc).unwrap();
        }
        assert!(other.blocks().iter().any(|b| want_blocks.contains(b)), "the freed blocks really were reusable");
        e.paint(&other, 999);
        other.release(&mut e.alloc);

        let back = e.promote_kv(7).unwrap();
        assert_eq!(back.len(), want_len, "a restored sequence resumes at its own length");
        assert_ne!(
            back.blocks(),
            want_blocks.as_slice(),
            "the pool's free list is a stack, so this restore genuinely lands on PERMUTED slots - \
             which is the case that would break if anything assumed slot identity"
        );
        assert_eq!(e.snapshot(&back), want, "restored KV must be byte-identical");
        assert_eq!(e.host.bytes_resident(), 0);
        assert!(!e.host.holds(7));

        let s = e.host.stats();
        assert_eq!((s.demotions, s.promotions, s.blocks_out, s.blocks_in), (1, 1, 3, 3));
    }

    /// A refused demote (host budget) must leave the sequence exactly as it
    /// was - still on the device, still decodable - not half-swapped.
    #[test]
    fn a_refused_demote_leaves_the_sequence_untouched() {
        let (block_size, block_words) = (4u32, 7usize);
        // Room for two blocks; the sequence needs three.
        let mut e = MockEngine::new(8, block_size, block_words, 2 * block_words as u64 * 4);
        let mut t = BlockTable::new();
        for _ in 0..10 {
            t.append(&mut e.alloc).unwrap();
        }
        e.paint(&t, 5);
        let want = e.snapshot(&t);
        let free_before = e.alloc.free_blocks();

        let err = e.demote_kv(1, &mut t).unwrap_err();
        assert_eq!(err, KvOffloadError::HostPoolFull { need_bytes: 3 * block_words as u64 * 4, free_bytes: 2 * block_words as u64 * 4 });
        assert_eq!(t.len(), 10, "the table must be untouched");
        assert_eq!(e.snapshot(&t), want, "and its KV must still be there");
        assert_eq!(e.alloc.free_blocks(), free_before);
        assert_eq!(e.host.stats().refused, 1);
    }

    /// A promote with no room on the device must fail cleanly and stay
    /// demoted - the scheduler retries it when blocks free up, so a partial
    /// allocation leaked here would strand pool capacity forever.
    #[test]
    fn a_promote_with_no_device_room_rolls_back_completely() {
        let (block_size, block_words) = (4u32, 5usize);
        let mut e = MockEngine::new(4, block_size, block_words, 1 << 20);
        let mut t = BlockTable::new();
        for _ in 0..9 {
            t.append(&mut e.alloc).unwrap(); // 3 blocks
        }
        e.paint(&t, 77);
        let want = e.snapshot(&t);
        e.demote_kv(3, &mut t).unwrap();

        // Occupy the pool down to two free blocks; the record needs three.
        let mut hog = BlockTable::new();
        for _ in 0..8 {
            hog.append(&mut e.alloc).unwrap();
        }
        assert_eq!(e.alloc.free_blocks(), 2);
        let err = e.promote_kv(3).unwrap_err();
        assert_eq!(err, KvOffloadError::DevicePoolExhausted { need_blocks: 3, free_blocks: 2 });
        assert_eq!(e.alloc.free_blocks(), 2, "a failed promote must give every block back");
        assert!(e.host.holds(3), "and must leave the sequence demoted, not lost");

        // Once the hog goes away the same promote succeeds, bytes intact.
        hog.release(&mut e.alloc);
        let back = e.promote_kv(3).unwrap();
        assert_eq!(e.snapshot(&back), want);
    }

    /// Demoting a sequence whose prefix blocks are SHARED (the prefix cache
    /// holds a reference) must still round-trip exactly: the shared blocks are
    /// copied like any other, and only the references this sequence held are
    /// dropped.
    #[test]
    fn a_shared_prefix_block_is_copied_and_only_its_own_reference_dropped() {
        let (block_size, block_words) = (4u32, 5usize);
        let mut e = MockEngine::new(8, block_size, block_words, 1 << 20);
        let mut t = BlockTable::new();
        for _ in 0..8 {
            t.append(&mut e.alloc).unwrap(); // 2 blocks
        }
        e.paint(&t, 42);
        let want = e.snapshot(&t);
        let shared = t.blocks()[0];
        e.alloc.incref(shared); // stands in for the prefix cache's reference

        let reclaimed = e.demote_kv(9, &mut t).unwrap();
        assert_eq!(reclaimed, 1, "only the private block returns to the pool");
        assert_eq!(e.alloc.refcount(shared), 1, "the cache's reference survives");

        let back = e.promote_kv(9).unwrap();
        assert_eq!(e.snapshot(&back), want, "a shared block's bytes round-trip like any other");
    }

    /// Cancellation: a demoted sequence can be dropped outright, and its host
    /// bytes must come back.
    #[test]
    fn discard_frees_the_host_bytes() {
        let (block_size, block_words) = (4u32, 5usize);
        let mut e = MockEngine::new(8, block_size, block_words, 1 << 20);
        let mut t = BlockTable::new();
        for _ in 0..8 {
            t.append(&mut e.alloc).unwrap();
        }
        e.demote_kv(2, &mut t).unwrap();
        assert!(e.host.bytes_resident() > 0);
        assert!(e.discard_kv(2));
        assert_eq!(e.host.bytes_resident(), 0);
        assert!(!e.discard_kv(2), "discarding twice is not an error, just false");
        assert_eq!(e.promote_kv(2).unwrap_err(), KvOffloadError::NotOffloaded { key: 2 });
    }

    /// A zero-byte budget is the "offload disabled" configuration: every
    /// demote is refused, nothing else changes.
    #[test]
    fn a_zero_capacity_pool_refuses_every_demote() {
        let (block_size, block_words) = (4u32, 5usize);
        let mut e = MockEngine::new(8, block_size, block_words, 0);
        let mut t = BlockTable::new();
        t.append(&mut e.alloc).unwrap();
        assert!(matches!(e.demote_kv(0, &mut t).unwrap_err(), KvOffloadError::HostPoolFull { .. }));
        assert_eq!(t.len(), 1);
    }
}
