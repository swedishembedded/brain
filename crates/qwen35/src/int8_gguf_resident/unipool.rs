// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One device pool for the KV blocks of the full-attention layers and the recurrent-state rows of the Gated-DeltaNet
//! layers.
//!
//! Swedish Embedded AB implements memory management for LLM serving for its clients, where state that belongs to
//! sequences that are not running takes the room of sequences that are. If your team needs expertise in sharing one
//! device memory budget between caches, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! # The mechanism
//!
//! The KV planes and the recurrent-state pools are sparse buffers ([`gpu_core::Gpu::try_alloc_sparse`]): address ranges
//! reserved at load with memory attached to 2 MiB granules of them on demand. Addresses never change, so no kernel, recorded
//! graph or `u32` element index is touched, and a row's arithmetic is what it was.
//!
//! * KV block `b` is granule `b` of every K and V data plane ([`KvTier::block_rows_for_granule`] sizes a block to a
//!   granule). The block allocator backs a block when it first hands it out ([`BlockBacking`]) and a freed block keeps its
//!   memory, so a warm pool costs nothing.
//! * State row `r` is the byte range `r * row_bytes ..` of every recurrent layer's pool. Admitting the sequence attaches
//!   memory to the granules of that range, shared (reference counted) with the neighbouring rows when a row is not a whole
//!   number of granules. A released row stays backed for the next sequence; when the scheduler runs short of blocks it
//!   reclaims released rows first ([`StateLedger::reclaim`]), which takes back the granules no row uses.
//!
//! # One budget
//!
//! The pool is planned as `blocks` blocks of memory: the memory every block could back. Backed blocks plus backed state
//! granules must never exceed it, and the engine keeps that true by *lending* the block allocator's coldest free blocks
//! ([`model::paged::BlockAllocator::lend`]) to cover the state it backs: blocks that are lent hold no memory and cannot be
//! handed to a sequence, so the memory they would have backed is the state's. The scheduler therefore sees the state as
//! used blocks, and its admission, preemption and speculation accounting are right without knowing about it, except that
//! a sequence that has not been given its state yet owes it ([`UnifiedPool::sequence_blocks`]).

use std::sync::{Arc, Mutex};

use gpu_core::{DeviceBuffer, Gpu};
use model::paged::{BlockAllocator, BlockBacking};

use crate::config::Qwen35Config;

/// The granule of a CUDA sparse buffer: the driver's minimum allocation granularity on every card to date. The pool checks the
/// device's own figure at load ([`UnifiedPool::new`]) and refuses a device that differs, since a block is sized to it.
pub(crate) const GRANULE_BYTES: u64 = 2 << 20;

/// A card shares its memory between KV and state in whole blocks only if one sequence's state row is at most this fraction of
/// it (the 27B's is 192 MiB of a 96 GiB card: 1/512); on a card small against the block the pool's granularity is too coarse and
/// the rows stay reserved.
// perf-number: a block is a granule of every plane; 16 sequences' rows would then take a whole card
pub(crate) const MAX_SEQUENCE_STATE_SHARE: u64 = 16;

/// The most rows a block may have for a model to share its card's memory between KV and state in blocks: a pool, a context and a
/// reused prefix are whole blocks. The 27B's are 1024 rows (bf16) and 2048 (int8, fp8).
pub(crate) const MAX_BLOCK_ROWS: u32 = 4096;

/// What one recurrent-state row takes of the pool, in granules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StateGeometry {
    /// Bytes of one row in one recurrent layer's pool (the f32 state, or half of it when the pools are bf16).
    pub row_bytes: u64,
    /// Recurrent layers, each with a pool of its own.
    pub planes: u64,
    /// Rows a pool holds (`--max-batch`).
    pub rows: u32,
}

impl StateGeometry {
    /// The geometry of `cfg`'s state pools at `rows` rows, bf16 or f32.
    pub(crate) fn of(cfg: &Qwen35Config, bf16: bool, rows: u32) -> StateGeometry {
        let planes = cfg.layer_types().iter().filter(|&&t| t == crate::config::LayerType::Linear).count() as u64;
        StateGeometry { row_bytes: cfg.gdn_state_len() * if bf16 { 2 } else { 4 }, planes, rows }
    }

    /// The granules of one pool that row `row` touches.
    fn span(&self, row: u32) -> std::ops::Range<u64> {
        let (start, end) = (row as u64 * self.row_bytes, (row as u64 + 1) * self.row_bytes);
        start / GRANULE_BYTES..end.div_ceil(GRANULE_BYTES)
    }

    /// The most granules of one pool any single row touches: how many a sequence's admission can newly back, per pool.
    fn widest_span(&self) -> u64 {
        (0..self.rows.max(1)).map(|r| self.span(r).count() as u64).max().unwrap_or(0)
    }

    /// Granules of a pool that `rows` backed rows touch.
    #[cfg(test)]
    fn granules_of(&self, rows: &[u32]) -> u64 {
        let mut touched: Vec<u64> = rows.iter().flat_map(|&r| self.span(r)).collect();
        touched.sort_unstable();
        touched.dedup();
        touched.len() as u64
    }
}

/// Blocks a sequence's admission can newly take from the pool for its state row: the most granules any row can newly back, in
/// every recurrent layer's pool, in units of a block's `kv_planes` granules. An upper bound the scheduler charges a sequence
/// that has no state row yet; the blocks actually lent are exact ([`UnifiedPool::state_blocks`]).
pub(crate) fn sequence_blocks(geo: &StateGeometry, kv_planes: u64) -> u32 {
    (geo.widest_span() * geo.planes).div_ceil(kv_planes.max(1)) as u32
}

/// The pool: the KV data planes and recurrent-state pools of one card, which block ids and state rows draw on together.
pub(crate) struct UnifiedPool {
    gpu: Gpu,
    /// Every K and V data plane (the MTP head's included): block `b` is granule `b` of each.
    kv_planes: Vec<DeviceBuffer>,
    /// Every recurrent layer's state pool.
    state_planes: Vec<DeviceBuffer>,
    geo: StateGeometry,
    /// Per block id: whether its planes hold memory.
    block_backed: Vec<bool>,
    /// Per granule of a state pool: the backed rows that touch it.
    users: Vec<u32>,
    /// Granules of a state pool with at least one user.
    state_granules: u64,
    /// Blocks backed now, for the report.
    backed_blocks: u32,
}

impl UnifiedPool {
    /// A pool over `kv_planes` (`blocks` granules each) and `state_planes`. Refuses a device whose granule is not
    /// [`GRANULE_BYTES`], or planes too short for what they must hold.
    pub(crate) fn new(gpu: &Gpu, kv_planes: Vec<DeviceBuffer>, state_planes: Vec<DeviceBuffer>, geo: StateGeometry, blocks: u32) -> Result<UnifiedPool, String> {
        match gpu.sparse_granule() {
            Some(GRANULE_BYTES) => {}
            Some(other) => return Err(format!("the device's sparse granule is {other} bytes, not the {GRANULE_BYTES} a KV block is sized to")),
            None => return Err("the device has no sparse buffers".to_string()),
        }
        if state_planes.len() as u64 != geo.planes {
            return Err(format!("{} state pools for a geometry of {} recurrent layers", state_planes.len(), geo.planes));
        }
        let granules = (geo.rows as u64 * geo.row_bytes).div_ceil(GRANULE_BYTES) as usize;
        Ok(UnifiedPool { gpu: gpu.share(), kv_planes, state_planes, geo, block_backed: vec![false; blocks as usize], users: vec![0; granules], state_granules: 0, backed_blocks: 0 })
    }

    /// Granules of memory one block holds: the K and V data planes.
    pub(crate) fn block_granules(&self) -> u64 {
        self.kv_planes.len() as u64
    }

    /// Blocks a sequence that has no state row yet owes the pool ([`sequence_blocks`]).
    pub(crate) fn sequence_blocks(&self) -> u32 {
        sequence_blocks(&self.geo, self.block_granules())
    }

    /// Blocks of memory the backed state rows take: the lend the allocator must hold.
    pub(crate) fn state_blocks(&self) -> u32 {
        (self.state_granules * self.geo.planes).div_ceil(self.block_granules().max(1)) as u32
    }

    /// [`Self::state_blocks`] once row `row` is backed too.
    pub(crate) fn state_blocks_with(&self, row: u32) -> u32 {
        let new = self.geo.span(row).filter(|&g| self.users[g as usize] == 0).count() as u64;
        ((self.state_granules + new) * self.geo.planes).div_ceil(self.block_granules().max(1)) as u32
    }

    /// Blocks and state granules backed now: `(blocks, granules per state pool)`.
    pub(crate) fn backed(&self) -> (u32, u64) {
        (self.backed_blocks, self.state_granules)
    }

    /// Attach memory to every plane of block `block`, if it has none. All or nothing.
    pub(crate) fn back_block(&mut self, block: u32) -> Result<(), String> {
        let b = block as usize;
        if *self.block_backed.get(b).ok_or_else(|| format!("block {block} is outside the pool's {}", self.block_backed.len()))? {
            return Ok(());
        }
        for (i, plane) in self.kv_planes.iter().enumerate() {
            if let Err(e) = self.gpu.back_sparse(plane, block as u64, 1) {
                // Return what this block already took, so a refused block leaves the pool as it was.
                let taken: Vec<_> = self.kv_planes[..i].iter().map(|p| (p, block as u64, 1)).collect();
                let _ = self.gpu.unback_sparse_many(&taken);
                return Err(e);
            }
        }
        self.block_backed[b] = true;
        self.backed_blocks += 1;
        Ok(())
    }

    /// Attach memory to every block that has none: one backing call a plane over the whole range, so the driver grants
    /// access once a plane instead of once a block (see `back_sparse`).
    pub(crate) fn back_all_blocks(&mut self) -> Result<(), String> {
        let blocks = self.block_backed.len() as u64;
        for plane in &self.kv_planes {
            self.gpu.back_sparse(plane, 0, blocks)?;
        }
        for backed in &mut self.block_backed {
            if !*backed {
                *backed = true;
                self.backed_blocks += 1;
            }
        }
        Ok(())
    }

    /// Take the memory back from block `block`, once nothing reaches it: it must be free and not about to be handed out.
    pub(crate) fn unback_block(&mut self, block: u32) -> Result<(), String> {
        if !self.block_backed[block as usize] {
            return Ok(());
        }
        let ranges: Vec<_> = self.kv_planes.iter().map(|p| (p, block as u64, 1)).collect();
        self.gpu.unback_sparse_many(&ranges)?;
        self.block_backed[block as usize] = false;
        self.backed_blocks -= 1;
        Ok(())
    }

    /// Attach memory to the granules of state row `row` in every recurrent layer's pool. The caller has lent the allocator's
    /// blocks to cover it ([`Self::state_blocks_with`]) first.
    pub(crate) fn back_state_row(&mut self, row: u32) -> Result<(), String> {
        let span = self.geo.span(row);
        let fresh: Vec<u64> = span.clone().filter(|&g| self.users[g as usize] == 0).collect();
        for (done, &g) in fresh.iter().enumerate() {
            if let Err(e) = self.state_planes.iter().try_for_each(|p| self.gpu.back_sparse(p, g, 1).map(|_| ())) {
                // Return what this row already took, so a refused row leaves the pool as it was.
                let taken: Vec<_> = fresh[..=done].iter().flat_map(|&g| self.state_planes.iter().map(move |p| (p, g, 1))).collect();
                let _ = self.gpu.unback_sparse_many(&taken);
                return Err(e);
            }
        }
        self.state_granules += fresh.len() as u64;
        for g in span {
            self.users[g as usize] += 1;
        }
        Ok(())
    }

    /// Take the memory back from the granules of row `row` that no other backed row touches.
    pub(crate) fn release_state_row(&mut self, row: u32) -> Result<(), String> {
        let mut freed = Vec::new();
        for g in self.geo.span(row) {
            let users = &mut self.users[g as usize];
            assert!(*users > 0, "state row {row} released twice: granule {g} has no user");
            *users -= 1;
            if *users == 0 {
                freed.push(g);
            }
        }
        let ranges: Vec<_> = freed.iter().flat_map(|&g| self.state_planes.iter().map(move |p| (p, g, 1))).collect();
        self.state_granules -= freed.len() as u64;
        self.gpu.unback_sparse_many(&ranges).map(|_| ())
    }
}

impl BlockBacking for UnifiedPool {
    fn back(&mut self, block: u32) -> Result<(), String> {
        self.back_block(block)
    }
}

/// The engine's side of the pool: the shared pool the allocator backs blocks through, and the blocks lent to cover the state
/// rows backed now.
///
/// A released row stays backed until the scheduler needs its blocks ([`Self::reclaim`]): backing a row is one `cuMemCreate` and
/// one `cuMemSetAccess` per granule per recurrent layer (~140 us each) on the serving thread, ~30 ms a row, and the blocks lent to
/// cover it are blocks of a pool that is backed at load anyway, so keeping it costs no device memory. Dropping released rows
/// eagerly (beyond eight while others ran, all of them once the server went idle) made every burst of requests after an idle
/// spell, and every admission past the eighth at a concurrency limit, pay that cost before its first token.
pub(crate) struct StateLedger {
    pool: Arc<Mutex<UnifiedPool>>,
    lent: Vec<u32>,
    /// Rows released but still backed, oldest first.
    retained: std::collections::VecDeque<u32>,
}

impl StateLedger {
    pub(crate) fn new(pool: UnifiedPool) -> StateLedger {
        StateLedger { pool: Arc::new(Mutex::new(pool)), lent: Vec::new(), retained: Default::default() }
    }

    /// Back every block now, so that no block is backed on the serving thread the first time a sequence takes it: one block is
    /// 32 `cuMemCreate` and `cuMemSetAccess` pairs (~4.7 ms), and the ~700 blocks of 64 streams of 8K were 3.3 s of host time in
    /// the first minute of load. A block a state row displaces gives its memory back (`admit`), as it does from any state.
    pub(crate) fn back_all_blocks(&self) -> Result<(), String> {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).back_all_blocks()
    }

    /// The pool, for the allocator that backs blocks through it.
    pub(crate) fn backing(&self) -> Arc<Mutex<dyn BlockBacking>> {
        self.pool.clone()
    }

    pub(crate) fn sequence_blocks(&self) -> u32 {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).sequence_blocks()
    }

    /// The released row that is still backed and was released last: the row to give the next sequence, which then costs no
    /// memory operation.
    pub(crate) fn retained_row(&self) -> Option<u32> {
        self.retained.back().copied()
    }

    /// Back state row `row`: a retained row is backed already; otherwise lend the allocator's coldest free blocks to cover the
    /// memory it takes (taking back the memory those blocks held, and the retained rows' if the allocator has too few), then
    /// attach memory to the row's granules. `Err` when the allocator has no blocks to lend: the scheduler's admission should
    /// have kept that from happening.
    pub(crate) fn admit(&mut self, alloc: &mut BlockAllocator, row: u32) -> Result<(), String> {
        if let Some(at) = self.retained.iter().position(|&r| r == row) {
            self.retained.remove(at);
            return Ok(());
        }
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let mut extra = pool.state_blocks_with(row).saturating_sub(self.lent.len() as u32);
        while extra > 0 && alloc.free_blocks() < extra && !self.retained.is_empty() {
            // Memory kept for the next sequence goes before a sequence is refused.
            Self::drop_oldest(&mut self.retained, &mut self.lent, &mut pool, alloc)?;
            extra = pool.state_blocks_with(row).saturating_sub(self.lent.len() as u32);
        }
        if extra > 0 {
            let blocks = alloc.lend(extra).ok_or_else(|| format!("the KV pool has {} free blocks and the state row needs {extra} more of them", alloc.free_blocks()))?;
            for &b in &blocks {
                pool.unback_block(b)?;
            }
            self.lent.extend(blocks);
        }
        pool.back_state_row(row)
    }

    /// Back `rows` at load and hold them as released rows, as long as the allocator keeps `keep_free` blocks free after
    /// lending for them. Backing a row is ~30 ms of driver calls on the serving thread, so without this every new high of
    /// concurrency paid it at admission (the first 64 sequences of a server, ~2 s before their prompts ran). Released rows
    /// are reclaimable capacity, so the pool loses nothing a sequence could use. Returns the rows backed.
    pub(crate) fn warm(&mut self, alloc: &mut BlockAllocator, rows: impl IntoIterator<Item = u32>, keep_free: u32) -> Result<u32, String> {
        let mut backed = 0;
        for row in rows {
            let need = {
                let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
                pool.state_blocks_with(row).saturating_sub(self.lent.len() as u32)
            };
            if alloc.free_blocks() < need.saturating_add(keep_free) {
                break;
            }
            self.admit(alloc, row)?;
            self.release(row);
            backed += 1;
        }
        Ok(backed)
    }

    /// Release state row `row`. It stays backed for the next sequence until [`Self::reclaim`] wants its blocks.
    pub(crate) fn release(&mut self, row: u32) {
        self.retained.push_back(row);
    }

    /// Give the allocator back up to `want` blocks from the oldest retained rows: the capacity the scheduler reclaims before it
    /// refuses or preempts a sequence. Returns the blocks repaid.
    pub(crate) fn reclaim(&mut self, alloc: &mut BlockAllocator, want: u32) -> Result<u32, String> {
        let before = alloc.free_blocks();
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        while alloc.free_blocks() - before < want && !self.retained.is_empty() {
            Self::drop_oldest(&mut self.retained, &mut self.lent, &mut pool, alloc)?;
        }
        Ok(alloc.free_blocks() - before)
    }

    /// Take the memory back from the oldest retained row and repay the blocks no longer needed.
    fn drop_oldest(retained: &mut std::collections::VecDeque<u32>, lent: &mut Vec<u32>, pool: &mut UnifiedPool, alloc: &mut BlockAllocator) -> Result<(), String> {
        let row = retained.pop_front().expect("a retained row");
        pool.release_state_row(row)?;
        let needed = pool.state_blocks() as usize;
        if lent.len() > needed {
            alloc.repay(lent.split_off(needed));
        }
        Ok(())
    }

    /// `(blocks backed, state granules per pool backed, blocks lent)`, for the report.
    pub(crate) fn report(&self) -> (u32, u64, usize) {
        let (blocks, granules) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).backed();
        (blocks, granules, self.lent.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 27B's state: 48 heads x 128 x 128 f32 = 3 MiB a row, in 48 recurrent layers.
    fn f32_rows(rows: u32) -> StateGeometry {
        StateGeometry { row_bytes: 3 << 20, planes: 48, rows }
    }

    #[test]
    fn a_row_that_is_not_a_whole_number_of_granules_shares_the_granule_at_its_edge_with_its_neighbour() {
        let geo = f32_rows(64);
        assert_eq!(geo.span(0), 0..2);
        assert_eq!(geo.span(1), 1..3, "row 1 starts in the middle of granule 1");
        assert_eq!(geo.widest_span(), 2);
        assert_eq!(geo.granules_of(&[0]), 2);
        assert_eq!(geo.granules_of(&[0, 1]), 3, "two rows touch three granules, not four");
        assert_eq!(geo.granules_of(&[0, 1, 2, 3]), 6, "four rows are 12 MiB: six granules, none wasted");
        assert_eq!(geo.granules_of(&[0, 2]), 4, "rows 0 and 2 share no granule");
    }

    #[test]
    fn a_sequence_owes_the_pool_the_widest_row_in_blocks_of_every_kv_plane() {
        // 48 pools x 2 granules = 96 granules; a block of the 27B at 16 attention layers is 32 granules.
        assert_eq!(sequence_blocks(&f32_rows(64), 32), 3);
        // bf16 pools: 1.5 MiB a row; the widest row touches two granules too.
        let bf16 = StateGeometry { row_bytes: 3 << 19, planes: 48, rows: 64 };
        assert_eq!(bf16.widest_span(), 2);
        assert_eq!(sequence_blocks(&bf16, 32), 3);
        // With the MTP head's two planes a block is 34 granules.
        assert_eq!(sequence_blocks(&f32_rows(64), 34), 3);
        // A single row is the only one there is.
        assert_eq!(StateGeometry { rows: 1, ..f32_rows(1) }.widest_span(), 2);
    }

    #[test]
    fn the_blocks_lent_follow_the_granules_backed_not_the_rows() {
        let geo = f32_rows(64);
        // Four rows back six granules a pool: 288 granules = 9 blocks of 32; the same rows at 3 blocks each would be 12.
        let backed = geo.granules_of(&[0, 1, 2, 3]) * geo.planes;
        assert_eq!(backed.div_ceil(32), 9);
        assert!(backed.div_ceil(32) < 4 * sequence_blocks(&geo, 32) as u64);
    }
}
