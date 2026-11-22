// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving engines for looped and hybrid decoders for its clients: models that run
// some layers several times per token and mix full attention, sliding windows and convolution state. If your team
// needs expertise in the per-sequence memory of such models under continuous batching then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The per-sequence state of a looped decoder, laid out from its [`VisitPlan`], as a building block a
//! [`super::PagedDecoder`] engine composes.
//!
//! A looped decoder runs `visits` logical layers over fewer stored ones, and every visit keeps its own state per
//! sequence (its state slot, [`crate::looped::Visit::slot`]). What a slot keeps follows from its attention:
//!
//! * a **full** visit keeps K/V that grow with the sequence: a paged plane per slot, all planes addressed by the SAME
//!   block ids. One [`BlockTable`] per sequence (the scheduler's) therefore covers every full visit at once, a block
//!   costs `full slots x 2 x block_size` rows ([`VisitLayout::block_bytes`]), and the scheduler's
//!   `blocks_for(tokens)` admission charges the true per-token cost of every full visit with no knowledge of visits;
//! * a **windowed** visit (`Window(n)`) keeps a ring of exactly `n` K/V rows per sequence (position `p` in ring row
//!   `p % n`, the [`crate::gqa_mixer::ring_row`] contract), however long the sequence runs;
//! * every slot keeps the declared convolution states ([`ConvState`], `taps - 1` f32 inputs per channel).
//!
//! The rings and convolution rows of a sequence are fixed in size: together they are its **state row**
//! ([`VisitLayout::row_bytes`], recorded as [`BlockTable::state_row`]). Rows `0..live_rows` serve running sequences
//! (the scheduler never runs more than [`VisitPool::max_batch`]); rows after them are the prefix cache's snapshots.
//! The rows are a store of their own beside the block pool, reserved for the engine's life: the pool's KV blocks are
//! the only memory that is shared, so `PagedDecoder::sequence_state_blocks` stays 0 (the state does not come out of
//! the block pool) and [`VisitPool::pool_bytes`] reports both, the device cost of the whole layout.
//! [`VisitLayout::sizing_for`] splits a byte budget between the two.
//!
//! # Prefill, decode and exactness
//!
//! [`VisitPool::attend_chunk`] runs one prompt chunk of one sequence through one visit's attention: a full visit
//! appends the chunk's K/V to the sequence's blocks and attends through its block table; a windowed visit gathers the
//! ring's live rows, attends ([`crate::gqa_mixer::ring_chunk_attend`]) and leaves the chunk's last `n` rows in the
//! ring. [`VisitPool::attend_decode`] runs one new token of each of a batch of sequences: the row is written at its
//! position (its block for a full visit, its ring row for a windowed one) and every sequence attends through its own
//! table. Both read keys in position order with distances from absolute positions ([`crate::flash_ext`]), so a
//! chunked prefill, a whole prefill and token-by-token decode give the same rows. Convolution state moves between
//! the rows and the kernels' batch-ordered layout with [`VisitPool::conv_rows`] / [`VisitPool::store_conv_rows`]
//! (the [`crate::short_conv`] step and prefill kernels read `[n, channels, taps - 1]`).
//!
//! # Prefix reuse
//!
//! One [`SnapshotPrefixCache`] entry spans every visit: its table holds the blocks of all full visits (a fork of the
//! sequence's table, shared by reference count; a prompt that resumes from it copies a partly filled tail block
//! before writing into it), and its snapshot row holds every ring and every convolution state as they stood after the
//! cached prompt. A prompt that extends a cached one adopts the blocks, copies the snapshot over its own state row and
//! prefills only the rest; a repeat of the cached prompt computes nothing ([`Resumed::exact`]). Entries are keyed by
//! KV namespace (adapter) and the exact token prefix, as every entry of that cache is, and for a prompt carrying
//! embedding rows by their content ([`Prompt`] with a [`ContentKey`]; [`VisitPool::file_prefix_content`]), so two
//! prompts with the same placeholders and other images never share state.

use gpu_core::{DeviceBuffer, Gpu};

use crate::flash_ext::{flash_ext_paged_fwd, paged_index, AttnInputs, AttnOptions, FlashExtIds, HeadShape, KvPlanes, PagedKv, PagedSeq};
use crate::gqa_mixer::{ring_chunk_attend, ring_copy_pipelines, RingChunk, RingCopyIds};
use crate::kv_tier::{copy_words_within, KvLayer, KvTier};
use crate::looped::{Attention, VisitPlan};
use crate::paged::{BlockAllocator, BlockTable, KvNamespace};
use crate::serve::ContentKey;
use crate::snapshot_prefix::{Entry, Prompt, SnapshotPrefixCache};

/// The shortest prompt worth a prefix entry: copying a sequence's whole state row for a handful of tokens costs more
/// than prefilling them again. [`VisitPool::with_min_cached_tokens`] changes it.
pub const MIN_CACHED_TOKENS: usize = 16;

/// One convolution state a slot keeps per sequence: the last `taps - 1` inputs of each of `channels` channels, f32,
/// laid out `[channels, taps - 1]` (the [`crate::short_conv`] history).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvState {
    pub channels: u32,
    pub taps: u32,
}

impl ConvState {
    /// f32 words of one sequence's state.
    #[must_use]
    pub fn words(self) -> u64 {
        u64::from(self.channels) * u64::from(self.taps - 1)
    }
}

/// What every visit's state looks like: its K/V rows (`n_kv_heads * head_dim` wide, in `kv_tier`) and the
/// convolution states each slot carries.
#[derive(Clone, Debug, PartialEq)]
pub struct VisitStateShape {
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// The tier of every K/V plane, paged and ring: f32 or bf16 (what [`crate::flash_ext`] reads).
    pub kv_tier: KvTier,
    /// The convolution states of every slot, in the order the engine names them by index.
    pub conv: Vec<ConvState>,
}

impl VisitStateShape {
    /// Elements of one K (or V) row.
    #[must_use]
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    fn validate(&self) -> Result<(), String> {
        if self.n_kv_heads == 0 || self.head_dim == 0 {
            return Err(format!("visit state: {} kv heads of width {} hold no K/V", self.n_kv_heads, self.head_dim));
        }
        if self.kv_tier != KvTier::F32 && self.kv_tier != KvTier::BF16 {
            return Err(format!("visit state: K/V tier {} is not one the extended attention reads (f32, bf16)", self.kv_tier));
        }
        if self.kv_tier == KvTier::BF16 && !self.kv_dim().is_multiple_of(2) {
            return Err(format!("visit state: a bf16 row packs element pairs, so {} elements do not fit", self.kv_dim()));
        }
        if let Some(c) = self.conv.iter().find(|c| c.channels == 0 || c.taps < 2) {
            return Err(format!("visit state: a convolution state of {} channels and {} taps keeps nothing", c.channels, c.taps));
        }
        Ok(())
    }
}

/// What one state slot keeps of K/V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKv {
    /// Growing K/V in paged plane `plane` (block ids shared with every other paged plane).
    Paged { plane: usize },
    /// A ring of `rows` rows per sequence in ring plane `ring`.
    Ring { ring: usize, rows: u32 },
}

/// The visit plan with what each of its slots keeps: pure arithmetic, no device.
#[derive(Clone, Debug, PartialEq)]
pub struct VisitLayout {
    plan: VisitPlan,
    shape: VisitStateShape,
    slots: Vec<SlotKv>,
    paged: usize,
    rings: Vec<u32>,
}

impl VisitLayout {
    /// The layout of `plan`: a paged plane for each full slot, a ring of the window's rows for each windowed one, the
    /// convolution states of `shape` for every slot. Refuses a slot whose visits disagree on their attention (they
    /// would read one state two ways) and a shape the kernels cannot hold.
    pub fn new(plan: VisitPlan, shape: VisitStateShape) -> Result<VisitLayout, String> {
        shape.validate()?;
        let owners = plan.slot_owners();
        if let Some(v) = plan.visits().iter().find(|v| v.attention != owners[v.slot].attention) {
            return Err(format!("visit state: visit {} runs {:?} but shares slot {} with visit {}, which runs {:?}", v.index, v.attention, v.slot, owners[v.slot].index, owners[v.slot].attention));
        }
        let (mut paged, mut rings) = (0, Vec::new());
        let slots = owners
            .iter()
            .map(|owner| match owner.attention {
                Attention::Full => {
                    paged += 1;
                    SlotKv::Paged { plane: paged - 1 }
                }
                Attention::Window(n) => {
                    rings.push(n);
                    SlotKv::Ring { ring: rings.len() - 1, rows: n }
                }
            })
            .collect();
        Ok(VisitLayout { plan, shape, slots, paged, rings })
    }

    pub fn plan(&self) -> &VisitPlan {
        &self.plan
    }

    pub fn shape(&self) -> &VisitStateShape {
        &self.shape
    }

    /// What visit `visit`'s slot keeps.
    #[must_use]
    pub fn slot_kv(&self, visit: usize) -> SlotKv {
        self.slots[self.plan.visits()[visit].slot]
    }

    /// Paged planes (full slots).
    #[must_use]
    pub fn paged_planes(&self) -> usize {
        self.paged
    }

    /// Ring rows of each windowed slot, in ring order.
    #[must_use]
    pub fn ring_rows(&self) -> &[u32] {
        &self.rings
    }

    /// Device bytes of one block of `block_size` rows: the K and V rows of every full slot.
    #[must_use]
    pub fn block_bytes(&self, block_size: u32) -> u64 {
        let (kvd, hd) = (u64::from(self.shape.kv_dim()), u64::from(self.shape.head_dim));
        2 * self.paged as u64 * self.shape.kv_tier.plane_bytes(u64::from(block_size), kvd, hd)
    }

    /// Device bytes of one state row: every windowed slot's ring (K and V) and every slot's convolution states.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        let (kvd, hd) = (u64::from(self.shape.kv_dim()), u64::from(self.shape.head_dim));
        let rings: u64 = self.rings.iter().map(|&n| 2 * self.shape.kv_tier.plane_bytes(u64::from(n), kvd, hd)).sum();
        rings + self.slots.len() as u64 * self.conv_row_words() * 4
    }

    /// f32 words of one slot's convolution states.
    fn conv_row_words(&self) -> u64 {
        self.shape.conv.iter().map(|c| c.words()).sum()
    }

    /// Device bytes of the zero convolution row a pool keeps for fresh sequences to start from.
    #[must_use]
    pub fn zero_row_bytes(&self) -> u64 {
        self.slots.len() as u64 * self.conv_row_words() * 4
    }

    /// A pool for `budget` device bytes: `live_rows` running sequences and `snapshot_rows` prefix entries take their
    /// state rows (and the zero row) first, the rest is blocks of `block_size`. Refuses a budget that leaves the blocks fewer than
    /// `min_tokens` tokens (one sequence's longest context, say).
    pub fn sizing_for(&self, budget: u64, block_size: u32, live_rows: u32, snapshot_rows: u32, min_tokens: u32) -> Result<VisitPoolSizing, String> {
        if block_size == 0 || live_rows == 0 {
            return Err("visit state: a pool needs a block size and at least one live row".into());
        }
        let rows = u64::from(live_rows) + u64::from(snapshot_rows);
        let state = rows * self.row_bytes() + self.zero_row_bytes();
        let left = budget.checked_sub(state).ok_or_else(|| format!("visit state: {rows} state rows take {state} bytes, more than the budget of {budget}"))?;
        let per_block = self.block_bytes(block_size).max(1);
        let num_blocks = u32::try_from(left / per_block).unwrap_or(u32::MAX);
        let need = min_tokens.div_ceil(block_size);
        if num_blocks < need.max(1) {
            return Err(format!("visit state: {left} bytes after the state rows hold {num_blocks} blocks of {block_size} tokens, fewer than the {need} one {min_tokens}-token sequence needs"));
        }
        Ok(VisitPoolSizing { block_size, num_blocks, live_rows, snapshot_rows, undo_rows: false })
    }
}

/// How many of everything a [`VisitPool`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisitPoolSizing {
    /// Rows of a KV block.
    pub block_size: u32,
    /// KV blocks of the full visits' pool (the scheduler's block pool).
    pub num_blocks: u32,
    /// State rows of running sequences: the most sequences one step carries.
    pub live_rows: u32,
    /// State rows of prefix-cache entries (0 turns prefix reuse off).
    pub snapshot_rows: u32,
    /// One more state row per live row, holding a sequence's state while a speculative verify runs ahead of what it
    /// may keep ([`VisitPool::checkpoint`] / [`VisitPool::rollback`]). Off for an engine that does not speculate.
    pub undo_rows: bool,
}

impl VisitPoolSizing {
    /// State rows the pool reserves: live, snapshot and (when kept) undo rows.
    #[must_use]
    pub fn state_rows(&self) -> u64 {
        u64::from(self.live_rows) * (1 + u64::from(self.undo_rows)) + u64::from(self.snapshot_rows)
    }
}

/// What [`VisitPool::begin`] found in the prefix cache.
#[derive(Clone, Debug, PartialEq)]
pub struct Resumed<P> {
    /// Leading prompt tokens whose state was restored: prefill continues at this position.
    pub rows: usize,
    /// The payload filed with the prompt, when the cached prompt is the whole prompt (nothing to compute).
    pub exact: Option<P>,
    /// The partly filled tail block the sequence copied before writing into it, `(from, to, rows)`: an engine that keeps
    /// per-block data of its own beside the pool (hidden-state taps) copies the same rows.
    pub copied_tail: Option<(u32, u32, u32)>,
}

/// Where the rows of one decode step go for one visit: the K/V layer, its block geometry (a ring's block is one
/// sequence's whole ring), each row's `(block, offset)` and each sequence's block table.
struct DecodeTarget<'a> {
    layer: &'a KvLayer,
    block_size: u32,
    /// Ring rows, 0 for a growing cache ([`PagedKv::ring`]).
    ring: u32,
    rows: Vec<(u32, u32)>,
    tables: Vec<Vec<u32>>,
}

/// Kernel indices the pool dispatches, resolved from [`VisitPool::pipelines`].
#[derive(Clone, Copy)]
struct PoolIds {
    copy_within: usize,
    gather: usize,
    scatter: usize,
    append_f32: usize,
    append_bf16: usize,
    flash: FlashExtIds,
    ring_copy: RingCopyIds,
    ring_attn: Option<usize>,
}

/// The bf16 store variant of the paged append (`Ops::kv_append_batched`'s bf16 kernel).
fn append_bf16_variant() -> (&'static str, &'static str) {
    kernels::template::dtype_variant_store("paged_kv_append_batched_word", kernels::PAGED_KV_APPEND_BATCHED_WORD, "pool", gpu_core::select::Dtype::BF16).expect("brain-kernels ships the paged-KV bf16 store variant")
}

/// A count or offset as a kernel's `u32` word index.
fn word(x: u64, what: &str) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| panic!("visit state: {what} = {x} exceeds the kernels' u32 element index"))
}

/// The device state of every sequence of a looped decoder: the full visits' block pool and the state rows (rings and
/// convolution states), plus the prefix cache over both. See the module doc.
pub struct VisitPool<P> {
    layout: VisitLayout,
    sizing: VisitPoolSizing,
    /// One per full slot: `num_blocks * block_size` rows.
    paged: Vec<KvLayer>,
    /// One per windowed slot: [`VisitPoolSizing::state_rows`] `* ring rows` rows (live, then snapshot, then undo rows).
    rings: Vec<KvLayer>,
    /// `[slot][state]`: `(state_rows + 1) * words` f32, the last row all zeros (a fresh sequence's state).
    conv: Vec<Vec<DeviceBuffer>>,
    alloc: BlockAllocator,
    free_rows: Vec<u32>,
    cache: SnapshotPrefixCache<P>,
    min_cached: usize,
    ids: PoolIds,
}

impl<P: Clone> VisitPool<P> {
    /// The kernels the pool dispatches (paged appends, row copies, the extended attention and the ring copies, the
    /// short-convolution family the engine runs over the rows), for the engine to register next to its own list.
    #[must_use]
    pub fn pipelines() -> Vec<(&'static str, &'static str)> {
        let mut v = vec![
            ("copy_within", kernels::COPY_WITHIN),
            ("kv_block_gather", kernels::KV_BLOCK_GATHER),
            ("kv_block_scatter", kernels::KV_BLOCK_SCATTER),
            ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
            append_bf16_variant(),
        ];
        crate::ops::extend_kernel_list(&mut v, &crate::flash_ext::pipelines());
        crate::ops::extend_kernel_list(&mut v, &ring_copy_pipelines());
        crate::ops::extend_kernel_list(&mut v, &crate::short_conv::pipelines());
        v
    }

    /// The pool of `layout` at `sizing` on `g` (which registered [`Self::pipelines`]).
    pub fn new(g: &Gpu, layout: VisitLayout, sizing: VisitPoolSizing) -> Result<VisitPool<P>, String> {
        let VisitPoolSizing { block_size, num_blocks, live_rows, snapshot_rows, .. } = sizing;
        if block_size == 0 || num_blocks == 0 || live_rows == 0 {
            return Err(format!("visit state: a pool of {num_blocks} blocks of {block_size} rows and {live_rows} live rows holds no sequence"));
        }
        let at = |name: &str| g.kernel_index(name).ok_or_else(|| format!("visit state: kernel {name} is not registered on this Gpu (see VisitPool::pipelines)"));
        let ids = PoolIds {
            copy_within: at("copy_within")?,
            gather: at("kv_block_gather")?,
            scatter: at("kv_block_scatter")?,
            append_f32: at("paged_kv_append_batched")?,
            append_bf16: at(append_bf16_variant().0)?,
            flash: FlashExtIds::resolve(g),
            ring_copy: RingCopyIds::resolve(g),
            ring_attn: g.kernel_index("gqa_ring_chunk_attn"),
        };
        let shape = layout.shape.clone();
        let (kvd, hd) = (u64::from(shape.kv_dim()), u64::from(shape.head_dim));
        let rows = sizing.state_rows();
        let pool_rows = u64::from(num_blocks) * u64::from(block_size);
        for (what, r) in std::iter::once(("paged plane", pool_rows)).chain(layout.rings.iter().map(|&n| ("ring plane", rows * u64::from(n)))) {
            if !KvTier::fits_addressing(r, kvd) {
                return Err(format!("visit state: a {what} of {r} rows of {kvd} elements exceeds the kernels' u32 element index"));
            }
        }
        let paged = (0..layout.paged).map(|_| KvLayer::new(g, shape.kv_tier, pool_rows, kvd, hd)).collect();
        let rings = layout.rings.iter().map(|&n| KvLayer::new(g, shape.kv_tier, rows * u64::from(n), kvd, hd)).collect();
        let conv = (0..layout.slots.len()).map(|_| shape.conv.iter().map(|c| g.storage((rows + 1) * c.words())).collect()).collect();
        Ok(VisitPool {
            layout,
            sizing,
            paged,
            rings,
            conv,
            alloc: BlockAllocator::new(num_blocks, block_size),
            free_rows: (0..live_rows).rev().collect(),
            cache: SnapshotPrefixCache::new(snapshot_rows),
            min_cached: MIN_CACHED_TOKENS,
            ids,
        })
    }

    /// Prompts shorter than `tokens` are neither filed nor looked up (at least 1).
    #[must_use]
    pub fn with_min_cached_tokens(mut self, tokens: usize) -> VisitPool<P> {
        self.min_cached = tokens.max(1);
        self
    }

    pub fn layout(&self) -> &VisitLayout {
        &self.layout
    }

    pub fn sizing(&self) -> VisitPoolSizing {
        self.sizing
    }

    // ---- what the scheduler reads (`PagedDecoder`'s accounting) ----

    /// The block pool every full visit draws on (`PagedDecoder::alloc_mut`).
    pub fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }

    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// Blocks holding `tokens` positions of every full visit.
    #[must_use]
    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.sizing.block_size)
    }

    /// The most sequences at once: one state row each (`PagedDecoder::max_batch`).
    #[must_use]
    pub fn max_batch(&self) -> usize {
        self.sizing.live_rows as usize
    }

    /// Device bytes of the whole layout: the block pool, every state row, live, snapshot and undo, and the zero
    /// convolution row a fresh sequence starts from (`PagedDecoder::kv_pool_bytes`).
    #[must_use]
    pub fn pool_bytes(&self) -> u64 {
        let rows = self.sizing.state_rows();
        u64::from(self.sizing.num_blocks) * self.layout.block_bytes(self.sizing.block_size) + rows * self.layout.row_bytes() + self.layout.zero_row_bytes()
    }

    /// Positions the block pool holds (`PagedDecoder::kv_pool_capacity_tokens`).
    #[must_use]
    pub fn capacity_tokens(&self) -> u64 {
        u64::from(self.sizing.num_blocks) * u64::from(self.sizing.block_size)
    }

    // ---- sequences ----

    /// Start the sequence of `table` (fresh: no blocks, no state row) on `prompt`: give it a state row with every
    /// convolution state zeroed, and resume it from the longest cached prefix of `prompt` in its namespace. Panics
    /// when every live row is taken (more running sequences than [`Self::max_batch`]).
    pub fn begin<'p>(&mut self, g: &Gpu, table: &mut BlockTable, prompt: impl Into<Prompt<'p>>) -> Resumed<P> {
        let prompt: Prompt<'p> = prompt.into();
        assert!(table.is_empty() && table.state_row().is_none(), "visit state: begin expects a fresh sequence");
        let row = self.free_rows.pop().unwrap_or_else(|| panic!("visit state: no free state row: more running sequences than the {} live rows", self.sizing.live_rows));
        table.set_state_row(row);
        let zero = self.zero_row();
        let mut steps = Vec::new();
        for (bufs, c) in self.conv.iter().flat_map(|slot| slot.iter().zip(&self.layout.shape.conv)) {
            steps.push(copy_words_within(g, self.ids.copy_within, bufs, zero * c.words(), u64::from(row) * c.words(), c.words()));
        }
        g.submit(&[], &steps);
        let fresh = Resumed { rows: 0, exact: None, copied_tail: None };
        if prompt.tokens.len() < self.min_cached || !self.cache.enabled() {
            return fresh;
        }
        let Some(hit) = self.cache.lookup(table.kv_namespace(), prompt) else { return fresh };
        let entry = self.cache.entry(hit);
        let bs = self.sizing.block_size;
        let rows = entry.tokens.len() as u32;
        let (snap, shared) = (entry.snap_row, entry.table.blocks()[..rows.div_ceil(bs) as usize].to_vec());
        let exact = (rows as usize == prompt.tokens.len()).then(|| entry.payload.clone());
        table.adopt_rows(&shared, rows, &mut self.alloc);
        let mut copied_tail = None;
        if !rows.is_multiple_of(bs) {
            // The entry keeps serving later prompts from this block, so the one this sequence writes next is a copy.
            let (from, to) = table.unshare_tail(&mut self.alloc).expect("visit state: a block shared with the prefix cache has a single owner");
            self.copy_block_rows(g, from, to, rows % bs);
            copied_tail = Some((from, to, rows % bs));
        }
        self.copy_state(g, self.snapshot_row(snap), row);
        Resumed { rows: rows as usize, exact, copied_tail }
    }

    /// Back `n` more positions of the sequence (every full visit's rows); returns the first one's position. A pool
    /// that cannot back them unwinds as a full device ([`BlockTable::reserve_or_raise`]).
    pub fn reserve(&mut self, table: &mut BlockTable, n: u32) -> u32 {
        table.reserve_or_raise(n, &mut self.alloc)
    }

    /// Back one more position (a decode step's token); returns its position.
    pub fn append(&mut self, table: &mut BlockTable) -> u32 {
        table.append_or_raise(&mut self.alloc);
        table.len() - 1
    }

    /// Release a finished, cancelled or preempted sequence: its blocks (the prefix cache keeps what it shares) and its
    /// state row.
    pub fn release(&mut self, table: &mut BlockTable) {
        if let Some(row) = table.take_state_row() {
            debug_assert!(row < self.sizing.live_rows && !self.free_rows.contains(&row), "visit state: state row {row} released twice");
            self.free_rows.push(row);
        }
        table.release(&mut self.alloc);
    }

    /// File the sequence of `table`, whose state is exactly `prompt`'s (it holds `prompt.len()` positions and nothing
    /// past them), as a prefix entry carrying `payload`: its blocks are shared with the entry, its state row is
    /// snapshotted. Entries it supersedes or evicts give their blocks back.
    pub fn file_prefix(&mut self, g: &Gpu, table: &BlockTable, prompt: &[u32], payload: P) {
        self.file_prefix_content(g, table, prompt, None, payload);
    }

    /// [`Self::file_prefix`] for a prompt carrying embedding rows: `content` is the key of the prompt's rows (of the
    /// whole request; only the part inside `prompt` is compared), so only a prompt with the same rows resumes from it.
    pub fn file_prefix_content(&mut self, g: &Gpu, table: &BlockTable, prompt: &[u32], content: Option<std::sync::Arc<ContentKey>>, payload: P) {
        if prompt.len() < self.min_cached || !self.cache.enabled() {
            return;
        }
        assert_eq!(table.len() as usize, prompt.len(), "visit state: a prefix entry is the state after exactly its prompt");
        let row = table.state_row().expect("visit state: a sequence files a prefix while it holds its state row");
        let ns = table.kv_namespace();
        let key = content.clone().unwrap_or_else(|| std::sync::Arc::new(ContentKey::none()));
        let (snap, evicted) = self.cache.take_row(ns, Prompt::new(prompt, &key));
        if let Some(mut gone) = evicted {
            gone.release(&mut self.alloc);
        }
        let Some(snap) = snap else { return };
        self.copy_state(g, row, self.snapshot_row(snap));
        let shared = table.fork(&mut self.alloc);
        let entry = Entry::new(prompt.to_vec(), shared, snap, payload);
        let entry = match content {
            Some(c) => entry.with_content(c),
            None => entry,
        };
        for mut gone in self.cache.insert(entry) {
            gone.release(&mut self.alloc);
        }
    }

    /// Blocks a fresh prefill of `prompt` in `namespace` adopts from the cache as it stands: the whole blocks of the
    /// longest cached prefix (a partly filled tail is copied, not adopted).
    #[must_use]
    pub fn cached_prefix_blocks<'p>(&self, namespace: KvNamespace, prompt: impl Into<Prompt<'p>>) -> u32 {
        let prompt: Prompt<'p> = prompt.into();
        if prompt.tokens.len() < self.min_cached {
            return 0;
        }
        self.cache.peek(namespace, prompt).map_or(0, |i| self.cache.entry(i).tokens.len() as u32 / self.sizing.block_size)
    }

    /// Drop least recently used entries until `want` blocks came back to the pool or nothing is cached; returns the
    /// blocks freed (an entry whose blocks a running sequence shares frees none of them, only its snapshot row).
    pub fn reclaim(&mut self, want: u32) -> u32 {
        self.reclaim_with(want, |c| c.evict_lru())
    }

    /// [`Self::reclaim`] sparing the entry `prompt` (in `namespace`) is about to resume from.
    pub fn reclaim_keeping<'p>(&mut self, want: u32, namespace: KvNamespace, prompt: impl Into<Prompt<'p>>) -> u32 {
        let prompt: Prompt<'p> = prompt.into();
        self.reclaim_with(want, |c| c.evict_lru_keeping(namespace, prompt))
    }

    fn reclaim_with(&mut self, want: u32, mut evict: impl FnMut(&mut SnapshotPrefixCache<P>) -> Option<BlockTable>) -> u32 {
        let before = self.alloc.free_blocks();
        while self.alloc.free_blocks() - before < want {
            let Some(mut gone) = evict(&mut self.cache) else { break };
            gone.release(&mut self.alloc);
        }
        self.alloc.free_blocks() - before
    }

    /// Drop every entry of `namespace` (its adapter was unloaded); returns how many.
    pub fn evict_namespace(&mut self, namespace: KvNamespace) -> usize {
        let gone = self.cache.evict_namespace(namespace);
        let n = gone.len();
        for mut t in gone {
            t.release(&mut self.alloc);
        }
        n
    }

    /// `(tokens served from the cache, tokens looked up, blocks the entries hold)` (`PagedDecoder::prefix_stats`).
    #[must_use]
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        let (hit, looked, _) = self.cache.stats();
        (hit, looked, self.cache.held_blocks())
    }

    // ---- device work ----

    /// The context `[n, n_heads * head_dim]` of one chunk of `n` query rows of the sequence of `table` at positions
    /// `start ..` through visit `visit`'s attention under `opts` (whose window must be the visit's: none for a full
    /// visit, at most the ring for a windowed one). `k`/`v` are the chunk's `[n, kv_dim]` f32 rows as the cache stores
    /// them; they are written to the visit's state. The positions must be backed ([`Self::reserve`]).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_chunk(&self, g: &Gpu, visit: usize, table: &BlockTable, start: u32, heads: &HeadShape, opts: &AttnOptions, inputs: &AttnInputs, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, n: u32) -> DeviceBuffer {
        self.check_heads(heads);
        assert!(n >= 1 && table.len() >= start + n, "visit state: positions {start}..{} are not backed (the table holds {})", start + n, table.len());
        match self.layout.slot_kv(visit) {
            SlotKv::Paged { plane } => {
                assert!(opts.window.is_none(), "visit state: visit {visit} attends its whole history; it takes no window");
                let rows: Vec<(u32, u32)> = (start..start + n).map(|p| table.locate(p, self.sizing.block_size)).collect();
                let mut steps = self.append_rows(g, &self.paged[plane], &rows, self.sizing.block_size, k, v);
                let seq = PagedSeq { q_row0: 0, n_q: n, q_pos0: start, kv_pos0: 0 };
                let tables = [table.blocks()];
                let max_bt = table.blocks().len() as u32;
                let (ctx, attend) = self.attend(g, &self.paged[plane], &[seq], &tables, max_bt, self.sizing.block_size, 0, heads, opts, inputs, q, n);
                steps.extend(attend);
                g.submit(&[], &steps);
                ctx
            }
            SlotKv::Ring { ring, rows } => {
                let row = table.state_row().expect("visit state: a sequence attends while it holds its state row");
                let chunk = RingChunk { layer: &self.rings[ring], row0: row * rows, window: rows, start, flash: self.ids.flash, copy: self.ids.ring_copy, ring_attn: self.ids.ring_attn };
                ring_chunk_attend(g, &chunk, heads, opts, inputs, q, k, v, n)
            }
        }
    }

    /// The context `[b, n_heads * head_dim]` of one new token of each of `b` sequences through visit `visit`'s
    /// attention: row `i` of `q`/`k`/`v` is the token at the last backed position of `tables[i]` ([`Self::append`]).
    /// The tokens' K/V are written first (each at its block, or its ring row), then every row attends through its own
    /// sequence's table.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_decode(&self, g: &Gpu, visit: usize, tables: &[&BlockTable], heads: &HeadShape, opts: &AttnOptions, inputs: &AttnInputs, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer) -> DeviceBuffer {
        self.check_heads(heads);
        let b = tables.len() as u32;
        assert!(b >= 1, "visit state: a decode step of no sequence");
        let positions: Vec<u32> = tables
            .iter()
            .map(|t| {
                assert!(!t.is_empty(), "visit state: a decode row needs its position backed");
                t.len() - 1
            })
            .collect();
        let seqs: Vec<PagedSeq> = positions.iter().enumerate().map(|(i, &p)| PagedSeq { q_row0: i as u32, n_q: 1, q_pos0: p, kv_pos0: 0 }).collect();
        let target = match self.layout.slot_kv(visit) {
            SlotKv::Paged { plane } => {
                assert!(opts.window.is_none(), "visit state: visit {visit} attends its whole history; it takes no window");
                let bs = self.sizing.block_size;
                DecodeTarget { layer: &self.paged[plane], block_size: bs, ring: 0, rows: tables.iter().zip(&positions).map(|(t, &p)| t.locate(p, bs)).collect(), tables: tables.iter().map(|t| t.blocks().to_vec()).collect() }
            }
            SlotKv::Ring { ring, rows } => {
                assert!(opts.window.is_some_and(|w| w <= rows), "visit state: visit {visit} keeps a ring of {rows} rows, so it attends a window of at most {rows} keys, not {:?}", opts.window);
                let state: Vec<u32> = tables.iter().map(|t| t.state_row().expect("visit state: a decoding sequence holds its state row")).collect();
                DecodeTarget { layer: &self.rings[ring], block_size: rows, ring: rows, rows: state.iter().zip(&positions).map(|(&r, &p)| (r, crate::gqa_mixer::ring_row(p, rows))).collect(), tables: state.iter().map(|&r| vec![r]).collect() }
            }
        };
        let mut steps = self.append_rows(g, target.layer, &target.rows, target.block_size, k, v);
        let tables: Vec<&[u32]> = target.tables.iter().map(Vec::as_slice).collect();
        let max_bt = tables.iter().map(|t| t.len() as u32).max().unwrap_or(1).max(1);
        let (ctx, attend) = self.attend(g, target.layer, &seqs, &tables, max_bt, target.block_size, target.ring, heads, opts, inputs, q, b);
        steps.extend(attend);
        g.submit(&[], &steps);
        ctx
    }

    /// Convolution state `state` of visit `visit` for each sequence of `tables`, gathered into a fresh
    /// `[tables.len(), channels, taps - 1]` buffer (the layout [`crate::short_conv::short_conv_step`] and, for one
    /// sequence, the prefill's history read).
    pub fn conv_rows(&self, g: &Gpu, visit: usize, state: usize, tables: &[&BlockTable]) -> DeviceBuffer {
        let (buf, words) = self.conv_buffer(visit, state);
        let ids = self.state_rows(g, tables);
        let n = tables.len() as u32;
        let out = g.storage_uninit(u64::from(n) * words);
        let w = word(words, "convolution state words");
        g.submit(&[], &[g.step(self.ids.gather, &[&ids, buf, &out], &[n, w, 0, w], n * w)]);
        out
    }

    /// Write `src` (`[tables.len(), channels, taps - 1]`) back as the convolution state `state` of visit `visit` of each
    /// sequence of `tables`.
    pub fn store_conv_rows(&self, g: &Gpu, visit: usize, state: usize, tables: &[&BlockTable], src: &DeviceBuffer) {
        let (buf, words) = self.conv_buffer(visit, state);
        let ids = self.state_rows(g, tables);
        let n = tables.len() as u32;
        let w = word(words, "convolution state words");
        g.submit(&[], &[g.step(self.ids.scatter, &[&ids, src, buf], &[n, w, 0, w], n * w)]);
    }

    // ---- speculative verify ----

    /// Keep each sequence's state row (every ring and every convolution state) in its undo row before a verify writes
    /// rows past what it may keep. The full visits' rows need no copy: a rejected position's block-table entry is
    /// truncated and its rows are rewritten when the position is next computed. A ring row, by contrast, is shared
    /// with the position one window earlier, and a convolution state holds the last inputs, so both are restored by
    /// [`Self::rollback`].
    pub fn checkpoint(&self, g: &Gpu, tables: &[&BlockTable]) {
        for t in tables {
            let row = t.state_row().expect("visit state: a verifying sequence holds its state row");
            self.copy_state(g, row, self.undo_row(row));
        }
    }

    /// Restore the state row of the sequence of `table` to what [`Self::checkpoint`] kept: the rings and convolution
    /// states as they stood before the verify. The engine then recomputes the rows it keeps.
    pub fn rollback(&self, g: &Gpu, table: &BlockTable) {
        let row = table.state_row().expect("visit state: a verifying sequence holds its state row");
        self.copy_state(g, self.undo_row(row), row);
    }

    // ---- internals ----

    fn check_heads(&self, heads: &HeadShape) {
        let shape = &self.layout.shape;
        assert!(heads.n_kv_heads == shape.n_kv_heads && heads.head_dim == shape.head_dim, "visit state: attention of {} kv heads of {} against a state of {} of {}", heads.n_kv_heads, heads.head_dim, shape.n_kv_heads, shape.head_dim);
    }

    /// Row `zero_row` of every convolution buffer is never written: a fresh sequence's state.
    fn zero_row(&self) -> u64 {
        self.sizing.state_rows()
    }

    /// The undo row of live state row `row`.
    fn undo_row(&self, row: u32) -> u32 {
        assert!(self.sizing.undo_rows, "visit state: the pool was sized without undo rows (VisitPoolSizing::undo_rows), so it cannot undo a verify");
        self.sizing.live_rows + self.sizing.snapshot_rows + row
    }

    fn snapshot_row(&self, snap: u32) -> u32 {
        self.sizing.live_rows + snap
    }

    fn conv_buffer(&self, visit: usize, state: usize) -> (&DeviceBuffer, u64) {
        let slot = self.layout.plan.visits()[visit].slot;
        let c = self.layout.shape.conv.get(state).unwrap_or_else(|| panic!("visit state: no convolution state {state} (the shape declares {})", self.layout.shape.conv.len()));
        (&self.conv[slot][state], c.words())
    }

    fn state_rows(&self, g: &Gpu, tables: &[&BlockTable]) -> DeviceBuffer {
        assert!(!tables.is_empty(), "visit state: no sequence to move state for");
        let rows: Vec<u32> = tables.iter().map(|t| t.state_row().expect("visit state: the sequence holds its state row")).collect();
        let ids = g.storage(rows.len() as u64);
        g.write(&ids, &rows);
        ids
    }

    /// Steps writing row `i` of `k`/`v` at `rows[i] = (block, offset)` of `layer`, whose blocks are `block_size` rows (a
    /// ring's block is one sequence's whole ring).
    fn append_rows(&self, g: &Gpu, layer: &KvLayer, rows: &[(u32, u32)], block_size: u32, k: &DeviceBuffer, v: &DeviceBuffer) -> Vec<gpu_core::Step> {
        let n = rows.len() as u32;
        let (blocks, offsets) = (g.storage(u64::from(n)), g.storage(u64::from(n)));
        g.write(&blocks, &rows.iter().map(|r| r.0).collect::<Vec<_>>());
        g.write(&offsets, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        let kvd = self.layout.shape.kv_dim();
        let (kind, threads) = if self.layout.shape.kv_tier == KvTier::BF16 { (self.ids.append_bf16, crate::kv_tier::append_word_threads(n, kvd)) } else { (self.ids.append_f32, n * kvd) };
        [(&layer.k, k), (&layer.v, v)].into_iter().map(|(plane, src)| g.step(kind, &[src, &blocks, &offsets, plane.data()], &[n, kvd, block_size], threads)).collect()
    }

    /// The extended attention of `seqs` (rows of `q`) through `tables` into `layer`.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, g: &Gpu, layer: &KvLayer, seqs: &[PagedSeq], tables: &[&[u32]], max_bt: u32, block_size: u32, ring: u32, heads: &HeadShape, opts: &AttnOptions, inputs: &AttnInputs, q: &DeviceBuffer, n_rows: u32) -> (DeviceBuffer, Vec<gpu_core::Step>) {
        let words = paged_index(seqs, tables, max_bt);
        let index = g.storage(words.len() as u64);
        g.write(&index, &words);
        let max_q = seqs.iter().map(|s| s.n_q).max().unwrap_or(1);
        let paged = PagedKv { index: &index, n_seqs: seqs.len() as u32, max_q, n_rows, block_size, max_bt, ring };
        let ctx = g.storage_uninit(u64::from(n_rows) * u64::from(heads.n_heads) * u64::from(heads.head_dim));
        let lse = g.storage_uninit(u64::from(n_rows) * u64::from(heads.n_heads));
        let steps = flash_ext_paged_fwd(g, &self.ids.flash, heads, opts, inputs, &paged, q, KvPlanes::of(layer), &ctx, &lse);
        (ctx, steps)
    }

    /// Copy the first `rows` rows of block `from` over block `to` in every paged plane.
    fn copy_block_rows(&self, g: &Gpu, from: u32, to: u32, rows: u32) {
        let bs = u64::from(self.sizing.block_size);
        let (kvd, hd) = (u64::from(self.layout.shape.kv_dim()), u64::from(self.layout.shape.head_dim));
        let steps: Vec<_> = self.paged.iter().flat_map(|l| [&l.k, &l.v]).flat_map(|p| p.copy_rows(g, self.ids.copy_within, u64::from(from) * bs, u64::from(to) * bs, u64::from(rows), kvd, hd)).collect();
        g.submit(&[], &steps);
    }

    /// Copy state row `from` over state row `to`: every ring and every convolution state.
    fn copy_state(&self, g: &Gpu, from: u32, to: u32) {
        let (kvd, hd) = (u64::from(self.layout.shape.kv_dim()), u64::from(self.layout.shape.head_dim));
        let mut steps = Vec::new();
        for (layer, &n) in self.rings.iter().zip(&self.layout.rings) {
            let n = u64::from(n);
            for plane in [&layer.k, &layer.v] {
                steps.extend(plane.copy_rows(g, self.ids.copy_within, u64::from(from) * n, u64::from(to) * n, n, kvd, hd));
            }
        }
        for (buf, c) in self.conv.iter().flat_map(|slot| slot.iter().zip(&self.layout.shape.conv)) {
            steps.push(copy_words_within(g, self.ids.copy_within, buf, u64::from(from) * c.words(), u64::from(to) * c.words(), c.words()));
        }
        g.submit(&[], &steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::looped::LoopShape;

    fn layout(window: u32) -> VisitLayout {
        // 2 prelude + 2 core x 2 passes + 1 coda; layers 1 and 3 attend their whole history.
        let plan = VisitPlan::new(LoopShape { prelude: 2, core: 2, coda: 1, passes: 2 }, |l| if l % 2 == 1 { Attention::Full } else { Attention::Window(window) }).unwrap();
        VisitLayout::new(plan, VisitStateShape { n_kv_heads: 2, head_dim: 4, kv_tier: KvTier::F32, conv: vec![ConvState { channels: 8, taps: 4 }, ConvState { channels: 16, taps: 4 }] }).unwrap()
    }

    /// Spec: a block holds every full visit's K and V rows for its positions, a state row every windowed visit's ring
    /// (exactly the window's rows, whatever the context) and every visit's convolution states; the budget split
    /// charges the rows first and refuses a budget that cannot hold one context of blocks.
    #[test]
    fn a_block_costs_every_full_visit_and_a_row_every_ring_and_convolution() {
        let l = layout(5);
        // Visits: W F W F W F W (prelude 0,1; core 2,3 twice; coda 4): 3 full and 4 windowed slots.
        assert_eq!((l.paged_planes(), l.ring_rows()), (3, &[5u32, 5, 5, 5][..]));
        assert_eq!(l.slot_kv(1), SlotKv::Paged { plane: 0 });
        assert_eq!(l.slot_kv(4), SlotKv::Ring { ring: 2, rows: 5 });
        let kv_row = 8 * 4; // 2 heads of 4, f32
        assert_eq!(l.block_bytes(16), 3 * 2 * 16 * kv_row);
        assert_eq!(l.row_bytes(), 4 * 2 * 5 * kv_row + 7 * (8 * 3 + 16 * 3) * 4);
        assert_eq!(layout(9).row_bytes() - l.row_bytes(), 4 * 2 * 4 * kv_row, "a ring grows with its window only");

        let budget = 10 * l.row_bytes() + l.zero_row_bytes() + 20 * l.block_bytes(16) + 7;
        let s = l.sizing_for(budget, 16, 6, 4, 64).unwrap();
        assert_eq!(s, VisitPoolSizing { block_size: 16, num_blocks: 20, live_rows: 6, snapshot_rows: 4, undo_rows: false });
        assert!(l.sizing_for(budget, 16, 6, 4, 400).unwrap_err().contains("fewer than"));
        assert!(l.sizing_for(5 * l.row_bytes(), 16, 6, 4, 1).unwrap_err().contains("more than the budget"));
    }

    /// Spec: visits sharing a slot read one state, so they must run the same attention; tiers the attention cannot
    /// read and empty convolution states are refused.
    #[test]
    fn a_layout_refuses_states_it_cannot_serve() {
        let shape = LoopShape { prelude: 0, core: 2, coda: 0, passes: 2 };
        let shared = VisitPlan::new(shape, |l| if l == 0 { Attention::Full } else { Attention::Window(4) }).unwrap().share_core_slots(1).unwrap();
        let ok = VisitStateShape { n_kv_heads: 1, head_dim: 4, kv_tier: KvTier::F32, conv: vec![] };
        assert!(VisitLayout::new(shared, ok.clone()).unwrap_err().contains("shares slot"));
        let plan = VisitPlan::uniform(shape, Attention::Full).unwrap();
        assert!(VisitLayout::new(plan.clone(), VisitStateShape { kv_tier: KvTier::INT8, ..ok.clone() }).unwrap_err().contains("tier"));
        assert!(VisitLayout::new(plan.clone(), VisitStateShape { conv: vec![ConvState { channels: 4, taps: 1 }], ..ok.clone() }).unwrap_err().contains("keeps nothing"));
        assert!(VisitLayout::new(plan, ok).is_ok());
    }
}
