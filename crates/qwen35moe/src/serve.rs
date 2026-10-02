// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-GPU, correctness-first `model::serve::PagedDecoder` for Qwen3.5-35B-A3B.
//!
//! Builds on `Qwen35::run_decode_batch` (`crates/qwen35moe/src/model.rs`) --
//! one decode token for each of several sequences, in one set of dispatches.
//!
//! Scope: steady-state decode is genuinely MULTI-SEQUENCE - one iteration's
//! dispatch set carries one row per resident sequence, so the weight reads
//! that dominate a decode step are paid once for the whole batch instead of
//! once per request. PREFILL is still one sequence, one token at a time (see
//! below). This is still not `qwen3::serve::Engine`'s full production feature
//! set -- see "Deliberately deferred" at the end of this doc for the exact
//! list.
//!
//! # The one real design problem: two kinds of per-sequence state, one trait slot
//!
//! [`model::serve::PagedDecoder`]'s methods thread a `&mut BlockTable` (paged
//! KV bookkeeping) per sequence -- that covers the 10 GQA layers. The 30 GDN
//! layers need a SECOND per-sequence resource, a fixed-size recurrent `state`
//! plus a causal-conv `hist` buffer pair per GDN layer
//! (`model::gdn::gdn_recurrent_step`/`gdn_causal_conv1d_step`'s own docs), and
//! the trait has no parameter for it. `model::serve::PagedDecoder` and
//! `model::paged::{BlockTable, BlockAllocator}` are NOT modified to add one:
//! that trait/those types are shared by every `PagedDecoder` (today just
//! `qwen3::serve::Engine`, which has no GDN layers at all), so adding a
//! GDN-shaped parameter to a generic interface for one caller is unjustified.
//!
//! Resolved like this: [`BlockTable::blocks`]'s FIRST entry
//! (`table.blocks()[0]`) is a stable per-sequence key. Concretely, in this
//! engine `block_size == max_seq_len` (see below) -- a sequence's ENTIRE
//! lifetime (prompt + every generated token) fits in exactly one physical
//! block, allocated once by the first `reserve`/`append` call in
//! [`Engine::prefill`] and never touched again until [`Engine::release_table`]
//! frees it (`BlockTable::append`/`reserve` only ever grow `self.blocks` when
//! `offset == 0` on a FULL block, which at `block_size == max_seq_len` can
//! only happen once, at the very first call -- verified against
//! `model::paged::BlockTable`'s own source, not assumed). So
//! `table.blocks()[0]` is exactly the stable identity this module needs, and
//! it is used as the key into a PRIVATE `HashMap<u32, GdnSlot>`
//! ([`Engine::gdn_slots`]) this `Engine` owns -- allocated (zeroed) the first
//! time a table is seen in [`Engine::prefill`], removed in
//! [`Engine::release_table`] (which still calls `BlockTable::release` for the
//! GQA side -- the GDN map is an ADDITION, not a replacement).
//!
//! # The GQA side: one shared pool per layer, block-table addressed
//!
//! `Qwen35::step` decodes exactly one persistent sequence, into
//! `self.gqa_kcache`/`self.gqa_vcache` -- fields that exist once per `Qwen35`
//! instance, not once per admitted request. A paged multi-request engine
//! needs that same per-layer KV cache to be addressable PER SEQUENCE, so
//! `Qwen35::run_decode_batch`/`run_decode_step` take the GQA pool + GDN
//! state/hist as an explicit parameter (`crate::model::BatchDecodeCaches`/
//! `DecodeCaches`, see their own docs) instead of reading
//! `self.gqa_kcache`/`self.gdn_state` -- `Qwen35::step` itself is a thin
//! wrapper passing its OWN fields as a `DecodeCaches`, so its behaviour (and
//! its `decode_step.rs` test) is unchanged.
//!
//! With that seam in place, this `Engine` preallocates, at construction, ONE
//! `[num_blocks*block_size, kv_dim]` pool per GQA layer
//! ([`Engine::gqa_k`]/[`Engine::gqa_v`], indexed `[layer]`) -- real and
//! resident from construction, not lazily grown. Physical block `p` owns rows
//! `p*block_size .. +block_size`, so every attention dispatch reaches a
//! sequence's history through the paged kernels' own block table rather than
//! through which buffer it bound.
//!
//! That indirection is what makes cross-sequence batching possible at all: a
//! batched decode dispatch carries every sequence's query row against ONE
//! bound pool, and the `[batch]`/`[batch, max_bt]` index buffers are the only
//! thing that says whose keys are whose. `block_size == max_seq_len` means a
//! sequence's entire lifetime fits one physical block, so `max_bt == 1` and a
//! block table is a single entry per row - the simplest non-degenerate case of
//! `model::block::gqa_decode_batched_step`'s general contract, not a special
//! case of it. The device byte cost is what it always was; see
//! [`Engine::kv_pool_bytes`].
//!
//! # `prefill`
//!
//! Loops over the prompt ONE TOKEN AT A TIME, calling
//! `Qwen35::run_decode_step` per token against this sequence's own
//! `DecodeCaches` (the whole pool, plus the base row that selects its physical
//! block, and its `GdnSlot`). This is NOT `qwen3::serve::Engine`'s chunked, multi-token-per-
//! dispatch prefill -- a per-token loop is the explicitly-sanctioned
//! correctness-first shape for this pass
//! ("correct-then-freeze" -- the same
//! principle already applied to this model's int8 GEMM tiling and MoE decode
//! dispatch). The performance gap (one submit+readback per PROMPT token,
//! instead of one batched whole-prompt forward) is real and is intentionally
//! left for later work, exactly the way `crate::sample::generate_kv`'s own
//! doc already names its identical per-token-prefill gap.
//!
//! # `forward_batched_greedy`/`_window`/`forward_batched_topk`
//!
//! One [`Engine::decode_batch`] call -- one set of GPU dispatches carrying
//! every `(table, input)` pair's row -- and then greedy/top-k sampling ON THE
//! HOST from the returned logits (the head itself is still a host matvec here,
//! see [`Engine::head`]). Nothing loops over sequences on the device side: the
//! whole batch's projections are single `[bsz, d] x [d, *]` GEMMs, its
//! full-attention layers one pooled
//! `model::block::gqa_decode_batched_step`, its Gated-DeltaNet layers one
//! `model::gdn_mixer::gdn_mixer_decode_fwd` over the `b*h` axis those kernels
//! already have, and its MoE sublayer one grouped-GEMM dispatch over all
//! `bsz*top_k` routed rows. `bsz == 1` is that same path at one row, so a
//! solitary request is not on a different code path from a busy one.
//!
//! These are the SAME shared primitives `qwen35::serve::Engine` drives, which
//! is the point: a caller batching this model gets the same behaviour and the
//! same contract as one batching the dense model.
//!
//! What the batch buys is throughput, not latency: a decode step is dominated
//! by reading every weight the step touches, and those reads are what the
//! batch shares.
//!
//! `forward_batched_greedy_window` is still host-orchestrated ACROSS window
//! positions (one batched step per position), since this engine has no
//! on-device multi-step schedule -- see "Deliberately deferred".
//!
//! # What this engine is, and is not (see the sections above for the design)
//!
//! - **Weights**: any [`model::ops::TierPolicy`] this model implements (fp32,
//!   int8), from any [`checkpoint::TensorSource`] - in particular straight from
//!   the released Q8_0 GGUF through [`crate::gguf_load::source`], so the 35B
//!   model is resident at ~38 GB with no fp32 copy anywhere.
//! - **KV**: the GQA layers' K/V pool is stored f32, bf16 or per-row int8
//!   ([`model::kv_tier::KvTier`], chosen in [`EngineOptions`]) and decoded
//!   through the fused split-key attention in every tier.
//! - **Prefill**: chunked - one dispatch shape per layer per round of
//!   [`EngineOptions::prefill_chunk`] tokens (`Qwen35::run_prefill_chunk`).
//! - **Head**: on the device. Greedy and top-k decode read back token ids, never
//!   a `[vocab]` logits row; only admission's [`Engine::logits`] reads one.
//! - **Not built**: prefix-cache reuse ([`Engine::reclaim_prefix`] returns 0),
//!   speculative decode, multi-GPU layer sharding, vision/DeepStack (text only).

use std::collections::HashMap;

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::gdn::{RecurrentSlot, RecurrentSlotShape};
use model::kv_tier::{KvLayer, KvTier};
use model::ops::TierPolicy;
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use crate::config::{LayerType, Qwen35Config};
use crate::model::{pipelines, BatchDecodeCaches, BatchSeq, DecodeCaches, Qwen35};

/// [`Engine::forward_batched_greedy_window`]'s window cap. No on-device
/// multi-step schedule is built (see module doc) - 1 keeps
/// `model::serve::Scheduler`'s window logic exercised (it always calls this with
/// `k <= decode_window_capacity()`) without pretending there is real per-window
/// batching underneath.
const DECODE_WINDOW_CAPACITY: usize = 1;

/// [`Engine::forward_batched_topk`]'s candidate-list cap: how many rounds of
/// device argmax-and-mask the top-k head runs (`Qwen35::head_topk_rows_dev`).
const TOPK_CAPACITY: usize = 32;

/// Prompt tokens pushed through the layer stack per prefill round when
/// [`EngineOptions`] does not say otherwise. A round's attention scratch grows
/// as `chunk * n_heads * (pos + chunk)`, the one cost that does not shrink with
/// the dispatch count, so the round is bounded rather than the whole prompt.
pub const DEFAULT_PREFILL_CHUNK: u32 = 256;

/// The `t` a quantized engine builds its `Qwen35` at: only the single-sequence
/// buffers (`logits`, `res`, the training tape's tokens) are sized by it, and a
/// serving engine uses none of them, so the smallest legal value is the right
/// one. An fp32 engine's grouped-expert scratch is sized by it too - see
/// `Engine::from_source_on`.
const SERVING_T: u32 = 64;

/// How an [`Engine`] is built: capacity, weight tier, KV tier, prefill round.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// The hard cap on `prompt + max_new` for any ONE sequence (this engine's
    /// `block_size`, see module doc).
    pub max_seq_len: u32,
    /// How many sequences may be resident at once (`num_blocks`).
    pub max_concurrent: u32,
    pub tier: TierPolicy,
    pub kv_tier: KvTier,
    /// Prompt tokens per prefill round.
    pub prefill_chunk: u32,
}

impl EngineOptions {
    /// fp32 weights and KV - the engine's original, reference configuration.
    pub fn new(max_seq_len: u32, max_concurrent: u32) -> EngineOptions {
        EngineOptions { max_seq_len, max_concurrent, tier: TierPolicy::uniform(Dtype::F32), kv_tier: KvTier::F32, prefill_chunk: DEFAULT_PREFILL_CHUNK }
    }
    pub fn with_tier(mut self, tier: TierPolicy) -> EngineOptions {
        self.tier = tier;
        self
    }
    pub fn with_kv_tier(mut self, kv_tier: KvTier) -> EngineOptions {
        self.kv_tier = kv_tier;
        self
    }
    pub fn with_prefill_chunk(mut self, rows: u32) -> EngineOptions {
        self.prefill_chunk = rows.max(1);
        self
    }
}

/// One admitted sequence's persistent Gated-DeltaNet resources: recurrent
/// `state` + causal-conv `hist`, one pair per layer (a size-1 dummy at
/// GQA-layer indices - the same "every layer index has a plain buffer, dummy
/// where irrelevant" convention `Qwen35`'s own `gdn_state`/`gdn_hist` fields
/// use). See this module's doc for why this lives in a private `HashMap` keyed
/// by `BlockTable::blocks()[0]` rather than a `PagedDecoder`-carried parameter.
/// The struct itself (allocation, zero-init, byte-cost accounting) is
/// [`model::gdn::RecurrentSlot`], shared with `qwen35::serve`.
type GdnSlot = RecurrentSlot;

/// [`GdnSlot::new`]/[`GdnSlot::bytes`]'s shape, read off this crate's own
/// [`Qwen35Config`] - see [`RecurrentSlotShape`]'s own doc for why
/// `crates/model` takes plain dims instead of this config type directly.
fn gdn_slot_shape(cfg: &Qwen35Config) -> RecurrentSlotShape {
    let bh = cfg.linear_num_value_heads as u64;
    let state_len = bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
    let hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
    let is_recurrent = cfg.layer_types().iter().map(|t| *t == LayerType::Linear).collect();
    RecurrentSlotShape { state_len, hist_len, is_recurrent }
}

/// Device bytes of the paged GQA pool for `num_blocks` blocks of `block_size`
/// rows in `kv_tier`, plus the Gated-DeltaNet slot of every block - a
/// prediction made before any device allocation, as `PagedDecoder::kv_pool_bytes`
/// requires.
pub fn kv_pool_bytes(cfg: &Qwen35Config, kv_tier: KvTier, num_blocks: u32, block_size: u32) -> u64 {
    let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
    let rows = num_blocks as u64 * block_size as u64;
    let gqa = n_full * 2 * kv_tier.plane_bytes(rows, cfg.kv_dim() as u64, cfg.head_dim as u64);
    gqa + num_blocks as u64 * GdnSlot::bytes(&gdn_slot_shape(cfg))
}

/// Single-GPU `PagedDecoder` for Qwen3.5/3.6-35B-A3B. See this module's doc for
/// the full design (why `block_size == max_seq_len`, the `GdnSlot` map, and
/// what is and is not built).
pub struct Engine {
    /// Owns the device handle, the weights, and the decode/prefill primitives
    /// (`Qwen35::run_decode_batch`, `Qwen35::run_prefill_chunk`) this whole engine
    /// is built on. Its OWN single-sequence decode state (`gqa_kv`/`gdn_state`) is
    /// never touched by `Engine` - every step here supplies its own
    /// `DecodeCaches`/`BatchDecodeCaches` - so it is built at the smallest `t`
    /// the weight tier allows.
    model: Qwen35,
    alloc: BlockAllocator,
    /// `== max_seq_len` (the hard per-sequence `prompt + max_new` cap) - see
    /// module doc for why this makes each physical block a whole sequence's
    /// entire KV history rather than a fixed-size page of it.
    block_size: u32,
    /// `[layer]`: ONE real, preallocated `[num_blocks*block_size, kv_dim]` GQA KV
    /// pool per full-attention layer (K and V planes in `kv_tier`), a
    /// placeholder at GDN-layer indices. Physical block `p` owns rows
    /// `p*block_size .. +block_size`. See module doc "The GQA side".
    gqa_kv: Vec<KvLayer>,
    kv_tier: KvTier,
    prefill_chunk: u32,
    /// GDN recurrent state / conv history, keyed by `BlockTable::blocks()[0]` -
    /// see module doc. Populated in [`Engine::prefill`] (or
    /// [`Engine::admit_synthetic`]), removed in [`Engine::release_table`].
    gdn_slots: HashMap<u32, GdnSlot>,
}

impl Engine {
    /// Build from an in-memory fp32 weight map - the reference configuration
    /// (`EngineOptions::new`). See [`Engine::from_source`] for every other.
    pub fn from_map(cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        Self::from_source_on(Gpu::new(pipelines()), cfg, weights, EngineOptions::new(max_seq_len, max_concurrent))
    }

    /// [`Engine::from_map`] on an EXISTING device handle (warm start): the
    /// caller's `Gpu` parents this engine via `Gpu::new_like`, so building
    /// another engine on the same device costs pipeline compilation only.
    pub fn from_map_on(parent: &Gpu, cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        Self::from_source_on(parent.new_like(pipelines()), cfg, weights, EngineOptions::new(max_seq_len, max_concurrent))
    }

    /// Build from any [`TensorSource`] - a GGUF through [`crate::gguf_load::source`]
    /// streams straight onto the device one tensor at a time.
    pub fn from_source(cfg: Qwen35Config, src: &dyn TensorSource, opts: EngineOptions) -> Engine {
        Self::from_source_on(Gpu::new(pipelines()), cfg, src, opts)
    }

    /// [`Engine::from_source`] on an existing device handle.
    pub fn from_source_on_parent(parent: &Gpu, cfg: Qwen35Config, src: &dyn TensorSource, opts: EngineOptions) -> Engine {
        Self::from_source_on(parent.new_like(pipelines()), cfg, src, opts)
    }

    fn from_source_on(gpu: Gpu, cfg: Qwen35Config, src: &dyn TensorSource, opts: EngineOptions) -> Engine {
        assert!(opts.max_seq_len > 0, "max_seq_len must be > 0");
        assert!(opts.max_concurrent > 0, "max_concurrent must be > 0");
        let kv_dim = cfg.kv_dim() as u64;
        let pool_rows = opts.max_concurrent as u64 * opts.max_seq_len as u64;
        assert!(
            KvTier::fits_addressing(pool_rows, kv_dim),
            "qwen35moe::serve::Engine: {} sequences of {} tokens is {pool_rows} KV rows of {kv_dim} elements, past what the paged kernels' u32 element offsets address",
            opts.max_concurrent,
            opts.max_seq_len
        );
        // `t` sizes this instance's fixed scratch. A quantized build has no
        // `rows`-bound scratch (its expert dispatch allocates per call), so the
        // smallest `t` serves; an fp32 build's grouped-GEMM expert scratch is
        // asserted against `rows * top_k` at dispatch, and the widest row count
        // this engine submits is a prefill round or one decode token per resident
        // sequence.
        let t = if opts.tier.quantizes_anything() { SERVING_T } else { opts.max_concurrent.max(opts.prefill_chunk) };
        let model = Qwen35::new_on_tier_src(gpu, cfg.clone(), 1, t, src, &opts.tier);
        let n_layers = cfg.n_layers as usize;
        let mut gqa_kv: Vec<KvLayer> = Vec::with_capacity(n_layers);
        for ty in cfg.layer_types() {
            gqa_kv.push(match ty {
                LayerType::Full => KvLayer::new(&model.gpu, opts.kv_tier, pool_rows, kv_dim, cfg.head_dim as u64),
                LayerType::Linear => KvLayer::placeholder(&model.gpu),
            });
        }
        Engine {
            model,
            alloc: BlockAllocator::new(opts.max_concurrent, opts.max_seq_len),
            block_size: opts.max_seq_len,
            gqa_kv,
            kv_tier: opts.kv_tier,
            prefill_chunk: opts.prefill_chunk,
            gdn_slots: HashMap::new(),
        }
    }

    /// This sequence's `DecodeCaches` view: the GQA pool for its physical block
    /// id, and its `GdnSlot`. Panics if no slot exists - every live
    /// `BlockTable` this engine handed back from [`Engine::prefill`] has one, by
    /// construction; a caller passing a table this engine never prefilled (or
    /// one already released) is a caller bug, not a runtime condition to
    /// degrade gracefully from.
    fn caches_for(&self, phys: u32) -> DecodeCaches<'_> {
        let slot = self.gdn_slot(phys);
        DecodeCaches { gqa_kv: &self.gqa_kv, gqa_cap: self.block_size, gqa_base_row: phys * self.block_size, gdn_state: &slot.state, gdn_hist: &slot.hist }
    }

    fn gdn_slot(&self, phys: u32) -> &GdnSlot {
        self.gdn_slots.get(&phys).unwrap_or_else(|| {
            panic!("qwen35moe::serve::Engine: no GdnSlot for physical block {phys} (table not prefilled by this engine, or already released)")
        })
    }

    /// ONE decode step for every `(table, input)` pair at once - the engine's
    /// whole steady-state decode path, and a single set of GPU dispatches
    /// whatever the batch size (`Qwen35::run_decode_batch`). Returns the
    /// `[bsz, d_model]` final-norm hidden block, unread, for a device head.
    ///
    /// Each table gets its own position appended first (never a second physical
    /// block, see module doc - `block_size == max_seq_len` means a sequence's
    /// total length can never cross a block boundary). `offset == pos`: since
    /// there is exactly one block per sequence, the position WITHIN that block
    /// already IS the absolute decode position.
    fn decode_hidden(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> gpu_core::DeviceBuffer {
        assert_eq!(tables.len(), inputs.len(), "qwen35moe::serve::Engine::decode_hidden: tables/inputs length mismatch");
        assert!(!tables.is_empty(), "qwen35moe::serve::Engine::decode_hidden: empty batch");
        let mut coords: Vec<(u32, u32)> = Vec::with_capacity(tables.len());
        for t in tables.iter_mut() {
            let (_block, offset) = t.append(&mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted mid-decode");
            coords.push((t.blocks()[0], offset));
        }
        let slots: Vec<&GdnSlot> = coords.iter().map(|&(phys, _)| self.gdn_slot(phys)).collect();
        let seqs: Vec<BatchSeq> = coords
            .iter()
            .zip(&slots)
            .map(|(&(phys, offset), slot)| BatchSeq { phys, pos: offset, gdn_state: &slot.state, gdn_hist: &slot.hist })
            .collect();
        let caches = BatchDecodeCaches { gqa_kv: &self.gqa_kv, gqa_cap: self.block_size, seqs: &seqs };
        self.model.run_decode_batch(inputs, &caches)
    }

    /// `logits = hidden @ head^T` for one `[d_model]` hidden row, on the device
    /// (admission's first-token sampling is the only caller; steady-state decode
    /// never reads a logits row).
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let g = &self.model.gpu;
        let h = g.storage_init("qwen35moe.admit.hidden", hidden);
        let logits = self.model.head_logits_rows_dev(&h, 1);
        g.read(&logits, self.model.cfg.vocab as usize)
    }

    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// The hard `prompt + max_new` cap for one sequence - `block_size` (see
    /// module doc).
    pub fn max_seq_len(&self) -> usize {
        self.block_size as usize
    }

    pub fn vocab(&self) -> usize {
        self.model.cfg.vocab as usize
    }

    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }

    /// Prompt tokens per prefill round - what the scheduler budgets an
    /// admission iteration in.
    pub fn max_prefill_tokens(&self) -> u32 {
        self.prefill_chunk
    }

    /// No prefix cache (see module doc) - always 0 blocks reclaimed.
    pub fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }

    /// No prefix cache - always `(0, 0, 0)`, matching [`Engine::reclaim_prefix`]'s
    /// own no-op.
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }

    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.model.gpu.stats()
    }

    /// The model's device handle - for a profiler arming kernel timing.
    pub fn gpu(&self) -> &Gpu {
        &self.model.gpu
    }

    /// Block the host until every dispatch this engine recorded has finished.
    pub fn poll_wait(&self) {
        self.model.gpu.poll_wait();
    }

    /// Real, combined device footprint of the per-sequence state: the GQA pool
    /// (every physical block's dedicated `[block_size, kv_dim]` K and V planes,
    /// in the pool's KV tier, every GQA layer) PLUS the GDN slot pool's own real
    /// cost, reported at its worst-case ceiling (`num_blocks` slots) as
    /// `PagedDecoder::kv_pool_bytes`'s contract asks. The same arithmetic as the
    /// free [`kv_pool_bytes`], which a planner calls before building anything.
    pub fn kv_pool_bytes(&self) -> u64 {
        kv_pool_bytes(&self.model.cfg, self.kv_tier, self.alloc.num_blocks(), self.block_size)
    }

    /// `num_blocks * block_size` - see [`PagedDecoder::kv_pool_capacity_tokens`]'s
    /// doc. Independent of the GDN side (whose state is O(1) per sequence, not
    /// O(tokens)).
    pub fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * self.block_size as u64
    }

    pub fn decode_window_capacity(&self) -> usize {
        DECODE_WINDOW_CAPACITY
    }

    pub fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY
    }

    /// Release a finished/cancelled sequence: free its GDN slot (if any - note
    /// the key comes from `blocks()[0]` BEFORE `BlockTable::release` clears it)
    /// THEN run the ordinary KV release. The GDN map is an ADDITION to the
    /// trait's default block-release behaviour, not a replacement for it - both
    /// must run, or the GQA pool's physical block (and the underlying
    /// `BlockAllocator` accounting) would leak.
    pub fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(&phys) = t.blocks().first() {
            self.gdn_slots.remove(&phys);
        }
        t.release(&mut self.alloc);
    }

    /// Prefill a fresh prompt into `table` in rounds of [`Self::max_prefill_tokens`],
    /// each one dispatch shape per layer (`Qwen35::run_prefill_chunk`). Returns the
    /// last token's final-norm hidden state.
    pub fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "prefill expects a fresh sequence");
        assert!(!prompt.is_empty(), "qwen35moe::serve::Engine::prefill: empty prompt (no token to produce a hidden state from)");
        assert!(
            prompt.len() <= self.max_seq_len(),
            "prompt of {} tokens exceeds this engine's per-sequence capacity of {} tokens",
            prompt.len(),
            self.max_seq_len()
        );
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.model.cfg.vocab) {
            panic!("prompt token {bad} is outside the model vocabulary ({})", self.model.cfg.vocab);
        }
        // One `reserve` for the whole prompt: since `block_size == max_seq_len`
        // this allocates EXACTLY the sequence's one physical block (see module
        // doc) - every later `append` (decode) reuses it.
        table.reserve(prompt.len() as u32, &mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted");
        let phys = table.blocks()[0];
        self.gdn_slots.entry(phys).or_insert_with(|| GdnSlot::new(&self.model.gpu, &gdn_slot_shape(&self.model.cfg)));

        let d = self.model.cfg.d_model as usize;
        let mut hidden = None;
        let mut pos = 0u32;
        for round in prompt.chunks(self.prefill_chunk as usize) {
            let h = self.model.run_prefill_chunk(round, pos, &self.caches_for(phys));
            pos += round.len() as u32;
            hidden = Some(h);
        }
        self.model.gpu.read(&hidden.expect("a non-empty prompt has at least one round"), d)
    }

    /// Bring `table` to `position` cached tokens WITHOUT computing them: the
    /// KV block is reserved and a zeroed Gated-DeltaNet slot is created, so the
    /// next decode step does the work (and moves the bytes) of a step at that
    /// position over whatever the cache rows hold. This is how a long-context
    /// decode step is priced when prefilling the context for real would take
    /// hours (`brain perf run longctx` says so in its artifact); the logits it
    /// produces mean nothing. A table already at `position` is left alone, one
    /// shorter is extended and one longer truncated.
    pub fn admit_synthetic(&mut self, table: &mut BlockTable, position: u32) -> Result<(), String> {
        if position == 0 || position >= self.block_size {
            return Err(format!("a synthetic context needs 1 <= position < {} (got {position})", self.block_size));
        }
        if table.is_empty() {
            table.reserve(position, &mut self.alloc)?;
            let phys = table.blocks()[0];
            self.gdn_slots.entry(phys).or_insert_with(|| GdnSlot::new(&self.model.gpu, &gdn_slot_shape(&self.model.cfg)));
        } else if table.len() < position {
            table.reserve(position - table.len(), &mut self.alloc)?;
        } else if table.len() > position {
            table.truncate(position, &mut self.alloc);
        }
        Ok(())
    }

    /// One greedy decode step for the WHOLE batch in one set of dispatches, the
    /// head on the device: only the `bsz` winning token ids are read back.
    pub fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_greedy: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        let hidden = self.decode_hidden(tables, inputs);
        self.model.head_argmax_rows_dev(&hidden, inputs.len() as u32)
    }

    /// [`Engine::forward_batched_greedy`], repeated `k` times per sequence,
    /// feeding each step's own greedy output back as the next input - there is no
    /// real on-device window (see [`Engine::decode_window_capacity`]'s doc), so
    /// this is host-orchestrated one dispatch set at a time.
    pub fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k), "window {k} exceeds this engine's decode_window_capacity {}", self.decode_window_capacity());
        let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(k); tables.len()];
        let mut cur: Vec<u32> = inputs.to_vec();
        for _ in 0..k {
            let next = self.forward_batched_greedy(tables, &cur);
            for (o, &n) in out.iter_mut().zip(&next) {
                o.push(n);
            }
            cur = next;
        }
        out
    }

    /// One batched decode step, returning each row's top-`k` (token id, logit)
    /// candidates, best first, extracted on the device (`k` clamped to
    /// [`Engine::topk_capacity`]).
    pub fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        assert_eq!(tables.len(), inputs.len(), "forward_batched_topk: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        let hidden = self.decode_hidden(tables, inputs);
        self.model.head_topk_rows_dev(&hidden, inputs.len() as u32, k as u32)
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        Engine::max_prefill_tokens(self)
    }
    fn free_blocks(&self) -> u32 {
        Engine::free_blocks(self)
    }
    fn max_seq_len(&self) -> usize {
        Engine::max_seq_len(self)
    }
    fn vocab(&self) -> usize {
        Engine::vocab(self)
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        Engine::blocks_for(self, tokens)
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        Engine::reclaim_prefix(self, want)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        Engine::release_table(self, t)
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        Engine::prefill(self, table, prompt)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        Engine::logits(self, hidden)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        Engine::forward_batched_greedy(self, tables, inputs)
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        Engine::forward_batched_greedy_window(self, tables, inputs, k)
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        Engine::prefix_stats(self)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        Engine::device_stats(self)
    }
    fn kv_pool_bytes(&self) -> u64 {
        Engine::kv_pool_bytes(self)
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        Engine::kv_pool_capacity_tokens(self)
    }
    fn decode_window_capacity(&self) -> usize {
        Engine::decode_window_capacity(self)
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        Engine::forward_batched_topk(self, tables, inputs, k)
    }
    fn topk_capacity(&self) -> usize {
        Engine::topk_capacity(self)
    }
}

/// `model::serve::Scheduler<Engine>` -- the continuous-batching scheduler
/// specialised to this engine, the same `Scheduler` type alias convention
/// `qwen3::serve::Scheduler` uses.
pub type Scheduler = model::serve::Scheduler<Engine>;
