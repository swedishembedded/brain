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
//! # Deliberately deferred (not built in this pass)
//!
//! - **Prefix-cache reuse**: none. [`Engine::reclaim_prefix`] is a no-op
//!   returning 0, [`Engine::prefix_stats`] always reports `(0, 0, 0)`.
//! - **Chunked / batched prefill**: prompts are replayed one token at a time
//!   (see above).
//! - **Batched PREFILL**: prompts are replayed one token at a time, one
//!   sequence at a time (decode is batched; prefill is not).
//! - **int8/int4 paged KV, weight quantization, speculative decode**: not
//!   implemented; this `Engine` only ever builds a plain fp32 `Qwen35`
//!   (`Qwen35::new_on`, never `new_on_i8`).
//! - **Multi-GPU layer sharding**: single GPU only.
//! - **Vision / DeepStack**: text-only, matching `Qwen35::step`'s own scope.
//! - **On-device decode window / top-K extraction**: [`Engine::decode_window_capacity`]
//!   and [`Engine::topk_capacity`] are small fixed host-side constants (see
//!   their own docs), not real on-device scratch.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::gdn::{RecurrentSlot, RecurrentSlotShape};
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use crate::config::{LayerType, Qwen35Config};
use crate::model::{pipelines, BatchDecodeCaches, BatchSeq, DecodeCaches, Qwen35};

/// [`Engine::forward_batched_greedy_window`]'s host-side window cap. No
/// on-device windowing is built in this pass (see module doc) -- 1 keeps
/// `model::serve::Scheduler`'s window logic exercised (it always calls this
/// with `k <= decode_window_capacity()`) without ever pretending there is
/// real per-window batching underneath.
const DECODE_WINDOW_CAPACITY: usize = 1;

/// [`Engine::forward_batched_topk`]'s host-side candidate-list cap. Real
/// (non-greedy) sampling still works fully correctly at this width -- it
/// only bounds how far into the vocabulary top-p's nucleus can reach (the
/// same documented ceiling `qwen3::serve`'s own `TOPK_CAPACITY` describes) --
/// chosen small because nothing here extracts it on-device (this engine
/// sorts the WHOLE host-side logits vector and truncates, see
/// [`Engine::forward_batched_topk`]), so a caller doing more than the default
/// `top_k` pays a bigger host sort, not a device-scratch limit.
const TOPK_CAPACITY: usize = 32;

/// One admitted sequence's persistent Gated-DeltaNet resources: recurrent
/// `state` + causal-conv `hist`, one pair per layer (a size-1 dummy at
/// GQA-layer indices -- the same "every layer index has a plain buffer,
/// dummy where irrelevant" convention `Qwen35`'s own `gdn_state`/`gdn_hist`
/// fields use). See this module's doc for why this lives in a private
/// `HashMap` keyed by `BlockTable::blocks()[0]` rather than a
/// `PagedDecoder`-carried parameter. The struct itself (allocation, zero-init,
/// byte-cost accounting) is [`model::gdn::RecurrentSlot`] -- hoisted there
/// because `qwen35::serve::GdnSlot` built the identical struct byte-for-byte;
/// this alias is the only trace of the old per-crate type left at call sites
/// below.
type GdnSlot = RecurrentSlot;

/// [`GdnSlot::new`]/[`GdnSlot::bytes`]'s shape, read off this crate's own
/// [`Qwen35Config`] -- see [`RecurrentSlotShape`]'s own doc for why
/// `crates/model` takes plain dims instead of this config type directly.
fn gdn_slot_shape(cfg: &Qwen35Config) -> RecurrentSlotShape {
    let bh = cfg.linear_num_value_heads as u64;
    let state_len = bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
    let hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
    let is_recurrent = cfg.layer_types().iter().map(|t| *t == LayerType::Linear).collect();
    RecurrentSlotShape { state_len, hist_len, is_recurrent }
}

/// Single-GPU, correctness-first `PagedDecoder` for Qwen3.5-35B-A3B. See this
/// module's doc for the full design (why `block_size == max_seq_len`, the
/// `GdnSlot` map, and the complete list of deferred production features).
pub struct Engine {
    /// Owns the device handle, weights (`ParamStore`), and the per-token
    /// decode-step primitives (`Qwen35::run_decode_step`) this whole engine
    /// is built on. Constructed with `b=1, t=1`: this instance's OWN
    /// `res`/`tokens`/`logits`/`gqa_kcache`/`gdn_state` fields (P11b's
    /// single-sequence decode state) are never touched by `Engine` -- every
    /// decode step here supplies its own `DecodeCaches` -- so they are sized
    /// to the smallest legal value rather than wasting a second copy of the
    /// per-sequence state this engine already manages itself.
    model: Qwen35,
    alloc: BlockAllocator,
    /// `== max_seq_len` (the hard per-sequence `prompt + max_new` cap) --
    /// see module doc for why this makes each physical block a whole
    /// sequence's entire KV history rather than a fixed-size page of it.
    block_size: u32,
    /// `[layer]`: ONE real, preallocated `[num_blocks*block_size, kv_dim]` GQA
    /// KV pool per full-attention layer, a size-1 dummy at GDN-layer indices.
    /// Physical block `p` owns rows `p*block_size .. +block_size`. See module
    /// doc "The GQA side".
    gqa_k: Vec<DeviceBuffer>,
    gqa_v: Vec<DeviceBuffer>,
    /// GDN recurrent state / conv history, keyed by `BlockTable::blocks()[0]`
    /// -- see module doc for why this is a private map rather than a trait
    /// parameter. Populated in [`Engine::prefill`], removed in
    /// [`Engine::release_table`].
    gdn_slots: HashMap<u32, GdnSlot>,
    /// `[vocab, d_model]` host head weight -- the same "host matvec, not a
    /// device dispatch" admission-time head `qwen3::serve::Engine::logits`
    /// uses, reused here for EVERY decode step too (not just admission),
    /// since this pass never builds an on-device greedy/top-K head at all.
    head: Vec<f32>,
}

impl Engine {
    /// Build from an in-memory weight map. `max_seq_len` is the hard cap on
    /// `prompt + max_new` for any ONE sequence (this engine's `block_size`,
    /// see module doc); `max_concurrent` is how many sequences may be
    /// resident at once (`num_blocks`) -- together they size the real,
    /// upfront-allocated GQA pool ([`Engine::kv_pool_bytes`]).
    pub fn from_map(cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        Self::from_map_with_gpu(Gpu::new(pipelines()), cfg, weights, max_seq_len, max_concurrent)
    }

    /// [`Engine::from_map`] on an EXISTING device handle (warm start): the
    /// caller's `Gpu` parents this engine via `Gpu::new_like`, so building
    /// another engine on the same device costs pipeline compilation only.
    pub fn from_map_on(parent: &Gpu, cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        Self::from_map_with_gpu(parent.new_like(pipelines()), cfg, weights, max_seq_len, max_concurrent)
    }

    fn from_map_with_gpu(gpu: Gpu, cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert!(max_concurrent > 0, "max_concurrent must be > 0");
        // `b=1, t=max_concurrent`: this instance's own decode-state fields are
        // dead weight for `Engine` (see `Engine::model`'s own doc), so `t`
        // exists here only as the row count the instance's FIXED-SIZE scratch
        // is built for - and the widest row count this engine can ever submit
        // is one decode token per resident sequence. The grouped-GEMM MoE
        // scratch (`model::moe::GroupedExpertScratch`, sized `b*t*top_k` and
        // asserted against at dispatch) is what makes that a hard requirement
        // rather than a preference: a batched decode step of `max_concurrent`
        // rows routes `max_concurrent*top_k` compacted rows through it.
        // Everything else `t` sizes is O(t*d_model) and so negligible next to
        // the KV pool below.
        let model = Qwen35::new_on(gpu, cfg.clone(), 1, max_concurrent, weights);
        let kv_dim = cfg.kv_dim() as u64;
        let n_layers = cfg.n_layers as usize;
        let types = cfg.layer_types();
        let pool_rows = max_concurrent as u64 * max_seq_len as u64;
        let mut gqa_k: Vec<DeviceBuffer> = Vec::with_capacity(n_layers);
        let mut gqa_v: Vec<DeviceBuffer> = Vec::with_capacity(n_layers);
        for ty in &types {
            match ty {
                LayerType::Full => {
                    gqa_k.push(model.gpu.storage(pool_rows * kv_dim));
                    gqa_v.push(model.gpu.storage(pool_rows * kv_dim));
                }
                LayerType::Linear => {
                    gqa_k.push(model.gpu.storage(1));
                    gqa_v.push(model.gpu.storage(1));
                }
            }
        }
        let head = weights
            .get(cfg.head_weight())
            .cloned()
            .unwrap_or_else(|| weights.get("tok.weight").cloned().expect("head weight"));
        Engine {
            model,
            alloc: BlockAllocator::new(max_concurrent, max_seq_len),
            block_size: max_seq_len,
            gqa_k,
            gqa_v,
            gdn_slots: HashMap::new(),
            head,
        }
    }

    /// This sequence's `DecodeCaches` view: the GQA pool slice for its
    /// physical block id, and its `GdnSlot`. Panics if no slot exists --
    /// every live `BlockTable` this engine handed back from [`Engine::prefill`]
    /// has one, by construction; a caller passing a table this engine never
    /// prefilled (or one already released) is a caller bug, not a runtime
    /// condition to degrade gracefully from.
    fn caches_for(&self, phys: u32) -> DecodeCaches<'_> {
        let slot = self.gdn_slot(phys);
        DecodeCaches {
            gqa_kcache: &self.gqa_k,
            gqa_vcache: &self.gqa_v,
            gqa_cap: self.block_size,
            gqa_base_row: phys * self.block_size,
            gdn_state: &slot.state,
            gdn_hist: &slot.hist,
        }
    }

    fn gdn_slot(&self, phys: u32) -> &GdnSlot {
        self.gdn_slots.get(&phys).unwrap_or_else(|| {
            panic!("qwen35moe::serve::Engine: no GdnSlot for physical block {phys} (table not prefilled by this engine, or already released)")
        })
    }

    /// ONE decode step for every `(table, input)` pair at once - the engine's
    /// whole steady-state decode path, and a single set of GPU dispatches
    /// whatever the batch size (`Qwen35::run_decode_batch`).
    ///
    /// Each table gets its own position appended first (never a second
    /// physical block, see module doc -- `block_size == max_seq_len` means a
    /// sequence's total length can never cross a block boundary). `offset ==
    /// pos`: since `block_size == max_seq_len` there is exactly one block per
    /// sequence, so the position WITHIN that block already IS the absolute
    /// decode position.
    ///
    /// Returns one `[d_model]` hidden row per sequence, in batch order.
    fn decode_batch(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(tables.len(), inputs.len(), "qwen35moe::serve::Engine::decode_batch: tables/inputs length mismatch");
        assert!(!tables.is_empty(), "qwen35moe::serve::Engine::decode_batch: empty batch");
        let mut coords: Vec<(u32, u32)> = Vec::with_capacity(tables.len());
        for t in tables.iter_mut() {
            let (_block, offset) = t.append(&mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted mid-decode");
            coords.push((t.blocks()[0], offset));
        }
        let d = self.model.cfg.d_model as usize;
        let slots: Vec<&GdnSlot> = coords.iter().map(|&(phys, _)| self.gdn_slot(phys)).collect();
        let seqs: Vec<BatchSeq> = coords
            .iter()
            .zip(&slots)
            .map(|(&(phys, offset), slot)| BatchSeq { phys, pos: offset, gdn_state: &slot.state, gdn_hist: &slot.hist })
            .collect();
        let caches = BatchDecodeCaches { gqa_kpool: &self.gqa_k, gqa_vpool: &self.gqa_v, gqa_cap: self.block_size, seqs: &seqs };
        let hidden = self.model.run_decode_batch(inputs, &caches);
        let flat = self.model.gpu.read(&hidden, inputs.len() * d);
        flat.chunks(d).map(|r| r.to_vec()).collect()
    }

    /// `logits = hidden @ head^T` on the host -- see [`Engine::head`]'s doc
    /// for why this is the SAME path used for both admission and steady-state
    /// decode in this pass (no on-device head at all).
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        model::hostmath::matvec_par(&self.head, hidden, self.model.cfg.vocab as usize, self.model.cfg.d_model as usize)
    }

    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// The hard `prompt + max_new` cap for one sequence -- `block_size`
    /// (see module doc).
    pub fn max_seq_len(&self) -> usize {
        self.block_size as usize
    }

    pub fn vocab(&self) -> usize {
        self.model.cfg.vocab as usize
    }

    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }

    /// Prefill is un-chunked (one per-token loop over the WHOLE prompt every
    /// admission, see module doc) -- there is no internal chunk size to
    /// report, so this returns the engine's own hard per-sequence capacity,
    /// the same size a single whole-prompt "chunk" would be.
    pub fn max_prefill_tokens(&self) -> u32 {
        self.block_size
    }

    /// No prefix cache in this pass (see module doc) -- always 0 blocks
    /// reclaimed.
    pub fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }

    /// No prefix cache in this pass -- always `(0, 0, 0)`, matching
    /// [`Engine::reclaim_prefix`]'s own no-op.
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }

    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.model.gpu.stats()
    }

    /// Real, combined device footprint: the GQA pool (every physical block's
    /// dedicated `[block_size, kv_dim]` K + V buffer, every GQA layer) PLUS
    /// the GDN slot pool's own real cost. The GDN side is reported at its
    /// WORST-CASE ceiling (`num_blocks` slots -- this engine's own
    /// `max_concurrent`, since `block_size == max_seq_len` makes "physical
    /// block" and "concurrently-resident sequence" the same count), even
    /// though slots are allocated lazily (`GdnSlot`s are created in
    /// `prefill`, one per never-before-seen physical block id, and removed in
    /// `release_table`) and so may not ALL be resident at any one instant --
    /// this matches `PagedDecoder::kv_pool_bytes`'s own documented contract
    /// ("computed before any device allocation happens... a prediction, not
    /// a postmortem"), and since an unmeasured memory claim is worse than
    /// none, reports the GDN cost at all rather than
    /// silently counting only the paged-KV half.
    pub fn kv_pool_bytes(&self) -> u64 {
        let n_full = self.model.cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        let num_blocks = self.alloc.num_blocks() as u64;
        let gqa_bytes = n_full * num_blocks * 2 * self.block_size as u64 * self.model.cfg.kv_dim() as u64 * 4;
        let gdn_ceiling = num_blocks * GdnSlot::bytes(&gdn_slot_shape(&self.model.cfg));
        gqa_bytes + gdn_ceiling
    }

    /// `num_blocks * block_size` -- see [`PagedDecoder::kv_pool_capacity_tokens`]'s
    /// doc. Independent of the GDN side (which has no "cached token count" --
    /// its state is O(1) per sequence, not O(tokens)).
    pub fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * self.block_size as u64
    }

    pub fn decode_window_capacity(&self) -> usize {
        DECODE_WINDOW_CAPACITY
    }

    pub fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY
    }

    /// Release a finished/cancelled sequence: free its GDN slot (if any --
    /// note the key comes from `blocks()[0]` BEFORE `BlockTable::release`
    /// clears it) THEN run the ordinary KV release. The GDN map is an
    /// ADDITION to the trait's default block-release behaviour, not a
    /// replacement for it -- both must run, or the GQA pool's physical block
    /// (and the underlying `BlockAllocator` accounting) would leak.
    pub fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(&phys) = t.blocks().first() {
            self.gdn_slots.remove(&phys);
        }
        t.release(&mut self.alloc);
    }

    /// Prefill a fresh prompt into `table`, one token at a time -- see module
    /// doc "`prefill`" for why this is a per-token loop rather than a
    /// batched/chunked forward.
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
        // One `reserve` call for the whole prompt: since `block_size ==
        // max_seq_len` this allocates EXACTLY the sequence's one physical
        // block (see module doc) -- every later `append` (decode) call
        // reuses it, since the sequence's total length can never cross a
        // block boundary (enforced by the Scheduler's own admission check
        // against `max_seq_len`).
        table.reserve(prompt.len() as u32, &mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted");
        let phys = table.blocks()[0];
        self.gdn_slots.entry(phys).or_insert_with(|| GdnSlot::new(&self.model.gpu, &gdn_slot_shape(&self.model.cfg)));

        let d = self.model.cfg.d_model as usize;
        let mut hidden = vec![0.0f32; d];
        for (i, &tok) in prompt.iter().enumerate() {
            let pos = i as u32;
            let h = {
                let caches = self.caches_for(phys);
                self.model.run_decode_step(tok, pos, &caches)
            };
            hidden = self.model.gpu.read(&h, d);
        }
        hidden
    }

    /// One greedy decode step for the WHOLE batch in one set of dispatches
    /// (`Qwen35::run_decode_batch`); the head itself stays host-side in this
    /// pass, see [`Engine::head`]'s doc.
    pub fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_greedy: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs).iter().map(|h| argmax(&self.logits(h))).collect()
    }

    /// [`Engine::forward_batched_greedy`], repeated `k` times per sequence,
    /// feeding each step's own greedy output back as the next input -- there
    /// is no real on-device window in this pass (see
    /// [`Engine::decode_window_capacity`]'s doc), so this is host-orchestrated
    /// one dispatch at a time; `k` is asserted against this engine's own
    /// (tiny, fixed) capacity, matching every other `PagedDecoder`'s contract
    /// that the scheduler never requests more than that.
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
    /// candidates -- sorted host-side from the FULL logits vector (no
    /// on-device top-K extraction in this pass, see [`Engine::topk_capacity`]'s
    /// doc), `k` clamped to this engine's own capacity.
    pub fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        assert_eq!(tables.len(), inputs.len(), "forward_batched_topk: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs)
            .iter()
            .map(|hidden| {
                let logits = self.logits(hidden);
                let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
                cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                cand.truncate(k);
                cand
            })
            .collect()
    }
}

/// Greedy argmax -- pure host math, no `Engine` dependency (mirrors
/// `model::serve`'s own free `argmax` for the identical reason).
fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in s.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
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
