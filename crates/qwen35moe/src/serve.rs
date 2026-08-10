// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-GPU, correctness-first `model::serve::PagedDecoder` for Qwen3.5-35B-A3B.
//!
//! Builds on `Qwen35::step`/`reset_decode_cache`/`decode_pos` (P11b,
//! `crates/qwen35moe/src/model.rs`) -- the single-sequence incremental decode
//! primitive this whole module is a thin multi-request wrapper around.
//! Scope, deliberately: **one truly-active sequence at a time** on the GPU
//! (every dispatch here processes exactly one sequence's one token; several
//! sequences can be RESIDENT and interleaved by the [`model::serve::Scheduler`]
//! across iterations, but never batched together into one GPU dispatch). This
//! is explicitly NOT `qwen3::serve::Engine`'s production feature set --
//! see "Deliberately deferred" at the end of this doc for the exact list.
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
//! # The GQA side: a real per-block-id pool, not `Qwen35::step`'s single toy cache
//!
//! `Qwen35::step` (P11b) decodes exactly one persistent sequence, into
//! `self.gqa_kcache`/`self.gqa_vcache` -- fields that exist once per `Qwen35`
//! instance, not once per admitted request. A paged multi-request engine
//! needs that same per-layer KV cache to be addressable PER SEQUENCE. This
//! module's `Engine` refactors `Qwen35::run_decode_step`'s decode-step
//! plumbing to take the GQA cache + GDN state/hist as an explicit parameter
//! (`crate::model::DecodeCaches`, see its own doc) instead of reading
//! `self.gqa_kcache`/`self.gdn_state` -- `Qwen35::step` itself is now a thin
//! wrapper passing its OWN fields as a `DecodeCaches`, so P11b's behaviour
//! (and its `decode_step.rs` test) is unchanged bit-for-bit.
//!
//! With that seam in place, this `Engine` preallocates, at construction, a
//! REAL pool: `num_blocks` dedicated `[block_size, kv_dim]` buffers per GQA
//! layer ([`Engine::gqa_k`]/[`Engine::gqa_v`], indexed `[physical block
//! id][layer]`) -- `num_blocks * n_full_layers * 2` device buffers, real and
//! resident from construction, not lazily grown. `block_size == max_seq_len`
//! is the choice that makes this a genuine "one physical block backs one
//! sequence's whole KV history" pool rather than needing block-indirect
//! (scatter/gather) attention kernels: `model::block::gqa_decode_step` --
//! this pass's decode primitive, reused UNCHANGED, not modified (`block.rs`
//! is out of scope for this task) -- takes a flat `[cap, kv_dim]`
//! `kcache`/`vcache` with no block-table indirection at all, so pointing it
//! at "sequence S's cache" only works if that sequence's positions live in
//! ONE contiguous buffer. A single big `[num_blocks*block_size, kv_dim]`
//! buffer with per-call byte-offset addressing (`Gpu::step_sliced`) was
//! considered and rejected for this pass: `gqa_decode_step` itself always
//! binds its buffers whole (`Gpu::step`, not `step_sliced`) and cannot be
//! changed to do otherwise without editing `model::block` (frozen by this
//! task's constraints) -- so "the pool" here is `num_blocks` separate
//! same-shape allocations rather than one large buffer sliced per sequence.
//! The device byte cost is identical either way; see [`Engine::kv_pool_bytes`].
//!
//! # `prefill`
//!
//! Loops over the prompt ONE TOKEN AT A TIME, calling
//! `Qwen35::run_decode_step` per token against this sequence's own
//! `DecodeCaches` (the pool slice for its physical block id, and its
//! `GdnSlot`). This is NOT `qwen3::serve::Engine`'s chunked, multi-token-per-
//! dispatch prefill -- a per-token loop is the explicitly-sanctioned
//! correctness-first shape for this pass
//! (`docs/porting-playbook.md` §10, "correct-then-freeze"; the same
//! principle already applied to this model's int8 GEMM tiling and MoE decode
//! dispatch). The performance gap (one submit+readback per PROMPT token,
//! instead of one batched whole-prompt forward) is real and is intentionally
//! left for later work, exactly the way `crate::sample::generate_kv`'s own
//! doc already names its identical per-token-prefill gap.
//!
//! # `forward_batched_greedy`/`_window`/`forward_batched_topk`
//!
//! Given the "one truly-active sequence at a time" scope, these are thin
//! loops over the (in practice length-1, but handled generally)
//! `tables`/`inputs` slices, each iteration calling the same per-token decode
//! step and sampling greedily/top-k ON THE HOST from the returned logits. No
//! real multi-sequence GPU-batched dispatch is built (that is
//! `qwen3::serve::Engine`'s own, much larger, achievement) -- if the
//! `Scheduler` has several requests running concurrently, each iteration
//! still serves them with N independent sequential dispatches, not one
//! batched one.
//!
//! # Deliberately deferred (not built in this pass)
//!
//! - **Prefix-cache reuse**: none. [`Engine::reclaim_prefix`] is a no-op
//!   returning 0, [`Engine::prefix_stats`] always reports `(0, 0, 0)`.
//! - **Chunked / batched prefill**: prompts are replayed one token at a time
//!   (see above).
//! - **Multi-sequence GPU batching**: `forward_batched_*` loop sequentially
//!   (see above).
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
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use crate::config::{LayerType, Qwen35Config};
use crate::model::{DecodeCaches, Qwen35, PIPELINES};

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
/// `PagedDecoder`-carried parameter.
struct GdnSlot {
    state: Vec<DeviceBuffer>,
    hist: Vec<DeviceBuffer>,
}

impl GdnSlot {
    /// Allocate and explicitly ZERO every layer's buffers for a fresh
    /// sequence. `Gpu::storage` does not guarantee zero-initialised memory
    /// (see `Qwen35::reset_decode_cache`'s own explicit clear, for the exact
    /// same reason), and a fresh sequence's recurrent state / conv history
    /// MUST start at zero (`model::gdn::gdn_recurrent_step`/
    /// `gdn_causal_conv1d_step`'s own docs) -- a stale slot handed to a new
    /// sequence (this never happens here, since a slot is only created once
    /// per never-before-seen physical block id, but the buffers themselves
    /// come from the allocator's own reuse of freed blocks) would otherwise
    /// leak a PREVIOUS sequence's recurrent state into a new one's first token.
    fn new(gpu: &Gpu, cfg: &Qwen35Config) -> GdnSlot {
        let bh = cfg.linear_num_value_heads as u64;
        let state_len = bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        let mut state = Vec::with_capacity(cfg.n_layers as usize);
        let mut hist = Vec::with_capacity(cfg.n_layers as usize);
        for ty in cfg.layer_types() {
            match ty {
                LayerType::Linear => {
                    state.push(gpu.storage(state_len));
                    hist.push(gpu.storage(hist_len));
                }
                LayerType::Full => {
                    state.push(gpu.storage(1));
                    hist.push(gpu.storage(1));
                }
            }
        }
        let clears: Vec<&DeviceBuffer> = state.iter().chain(hist.iter()).collect();
        gpu.submit(&clears, &[]);
        GdnSlot { state, hist }
    }

    /// Device bytes one slot costs: every GDN layer's `state` + `hist`, fp32.
    /// The one place this arithmetic lives -- [`GdnSlot::new`]'s allocation
    /// loop and [`Engine::kv_pool_bytes`] both depend on agreeing with it.
    fn bytes(cfg: &Qwen35Config) -> u64 {
        let bh = cfg.linear_num_value_heads as u64;
        let state_len = bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        let n_gdn = cfg.layer_types().iter().filter(|t| **t == LayerType::Linear).count() as u64;
        n_gdn * (state_len + hist_len) * 4
    }
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
    /// `[physical block id][layer]`: real, preallocated GQA KV cache, a
    /// size-1 dummy at GDN-layer indices. See module doc "The GQA side".
    gqa_k: Vec<Vec<DeviceBuffer>>,
    gqa_v: Vec<Vec<DeviceBuffer>>,
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
        Self::from_map_with_gpu(Gpu::new(PIPELINES), cfg, weights, max_seq_len, max_concurrent)
    }

    /// [`Engine::from_map`] on an EXISTING device handle (warm start): the
    /// caller's `Gpu` parents this engine via `Gpu::new_like`, so building
    /// another engine on the same device costs pipeline compilation only.
    pub fn from_map_on(parent: &Gpu, cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        Self::from_map_with_gpu(parent.new_like(PIPELINES), cfg, weights, max_seq_len, max_concurrent)
    }

    fn from_map_with_gpu(gpu: Gpu, cfg: Qwen35Config, weights: &HashMap<String, Vec<f32>>, max_seq_len: u32, max_concurrent: u32) -> Engine {
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert!(max_concurrent > 0, "max_concurrent must be > 0");
        // b=1, t=1: this instance's own P11b decode-state fields are dead
        // weight for `Engine` (see `Engine::model`'s own doc) -- t=1 is the
        // smallest legal prefill length (`gdn_chunk_size(1) == 1`, so
        // `t % chunk == 0` holds for every config).
        let model = Qwen35::new_on(gpu, cfg.clone(), 1, 1, weights);
        let kv_dim = cfg.kv_dim() as u64;
        let n_layers = cfg.n_layers as usize;
        let types = cfg.layer_types();
        let mut gqa_k: Vec<Vec<DeviceBuffer>> = Vec::with_capacity(max_concurrent as usize);
        let mut gqa_v: Vec<Vec<DeviceBuffer>> = Vec::with_capacity(max_concurrent as usize);
        for _ in 0..max_concurrent {
            let mut kl = Vec::with_capacity(n_layers);
            let mut vl = Vec::with_capacity(n_layers);
            for ty in &types {
                match ty {
                    LayerType::Full => {
                        kl.push(model.gpu.storage(max_seq_len as u64 * kv_dim));
                        vl.push(model.gpu.storage(max_seq_len as u64 * kv_dim));
                    }
                    LayerType::Linear => {
                        kl.push(model.gpu.storage(1));
                        vl.push(model.gpu.storage(1));
                    }
                }
            }
            gqa_k.push(kl);
            gqa_v.push(vl);
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
        let slot = self.gdn_slots.get(&phys).unwrap_or_else(|| {
            panic!("qwen35moe::serve::Engine: no GdnSlot for physical block {phys} (table not prefilled by this engine, or already released)")
        });
        DecodeCaches {
            gqa_kcache: &self.gqa_k[phys as usize],
            gqa_vcache: &self.gqa_v[phys as usize],
            gqa_cap: self.block_size,
            gdn_state: &slot.state,
            gdn_hist: &slot.hist,
        }
    }

    /// One decode step for an ALREADY-PREFILLED sequence: append one token
    /// position to `table` (never allocates a second physical block, see
    /// module doc -- `block_size == max_seq_len` means a sequence's total
    /// length can never cross a block boundary) and run
    /// `Qwen35::run_decode_step` against its own `DecodeCaches`. `offset ==
    /// pos`: since `block_size == max_seq_len` there is exactly one block per
    /// sequence, so the position WITHIN that block already IS the absolute
    /// decode position.
    fn decode_one(&mut self, table: &mut BlockTable, input: u32) -> Vec<f32> {
        let (_block, offset) = table.append(&mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted mid-decode");
        let phys = table.blocks()[0];
        let hidden = {
            let caches = self.caches_for(phys);
            self.model.run_decode_step(input, offset, &caches)
        };
        self.model.gpu.read(&hidden, self.model.cfg.d_model as usize)
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
    /// a postmortem"), and per `docs/lessons.md` #34 ("an unmeasured memory
    /// claim is worse than none") reports the GDN cost at all rather than
    /// silently counting only the paged-KV half.
    pub fn kv_pool_bytes(&self) -> u64 {
        let n_full = self.model.cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        let num_blocks = self.alloc.num_blocks() as u64;
        let gqa_bytes = n_full * num_blocks * 2 * self.block_size as u64 * self.model.cfg.kv_dim() as u64 * 4;
        let gdn_ceiling = num_blocks * GdnSlot::bytes(&self.model.cfg);
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
        self.gdn_slots.entry(phys).or_insert_with(|| GdnSlot::new(&self.model.gpu, &self.model.cfg));

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

    /// One greedy decode step per (table, input) pair, sequentially -- see
    /// module doc for why this is a loop, not a batched GPU dispatch.
    pub fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_greedy: tables/inputs length mismatch");
        let mut out = Vec::with_capacity(tables.len());
        for (t, &inp) in tables.iter_mut().zip(inputs) {
            let hidden = self.decode_one(t, inp);
            out.push(argmax(&self.logits(&hidden)));
        }
        out
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

    /// One decode step per (table, input) pair, returning each row's
    /// top-`k` (token id, logit) candidates -- sorted host-side from the
    /// FULL logits vector (no on-device top-K extraction in this pass, see
    /// [`Engine::topk_capacity`]'s doc), `k` clamped to this engine's own
    /// capacity.
    pub fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        assert_eq!(tables.len(), inputs.len(), "forward_batched_topk: tables/inputs length mismatch");
        let mut out = Vec::with_capacity(tables.len());
        for (t, &inp) in tables.iter_mut().zip(inputs) {
            let hidden = self.decode_one(t, inp);
            let logits = self.logits(&hidden);
            let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
            cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            cand.truncate(k);
            out.push(cand);
        }
        out
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
