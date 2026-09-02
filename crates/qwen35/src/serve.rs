// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-GPU, correctness-first `model::serve::PagedDecoder` for
//! Qwen3.8-27B. Mirrors `qwen35moe::serve` exactly (same design, same
//! deferred-scope list) - the GDN/GQA decode-step orchestration this module
//! wraps is architecture-identical between the two crates (this model's
//! only structural difference from qwen35moe is a dense MLP instead of MoE,
//! which `Qwen35::run_decode_step` already handles via the row-count-
//! agnostic `mlp_fwd`, with no serving-layer consequence at all).
//!
//! Builds on `Qwen35::step`/`reset_decode_cache`/`decode_pos`
//! (`crates/qwen35/src/model.rs`) - the single-sequence incremental decode
//! primitive this whole module is a thin multi-request wrapper around.
//! Scope, deliberately: **one truly-active sequence at a time** on the GPU
//! (every dispatch here processes exactly one sequence's one token; several
//! sequences can be RESIDENT and interleaved by the [`model::serve::Scheduler`]
//! across iterations, but never batched together into one GPU dispatch).
//! This is explicitly NOT `qwen3::serve::Engine`'s production feature set -
//! see "Deliberately deferred" at the end of this doc for the exact list.
//!
//! # The one real design problem: two kinds of per-sequence state, one trait slot
//!
//! [`model::serve::PagedDecoder`]'s methods thread a `&mut BlockTable` (paged
//! KV bookkeeping) per sequence - that covers the GQA layers. The GDN layers
//! need a SECOND per-sequence resource, a fixed-size recurrent `state` plus
//! a causal-conv `hist` buffer pair per GDN layer
//! (`model::gdn::gdn_recurrent_step`/`gdn_causal_conv1d_step`'s own docs), and
//! the trait has no parameter for it. `model::serve::PagedDecoder` and
//! `model::paged::{BlockTable, BlockAllocator}` are NOT modified to add one:
//! that trait/those types are shared by every `PagedDecoder`, so adding a
//! GDN-shaped parameter to a generic interface for this one family is
//! unjustified.
//!
//! Resolved like this: [`BlockTable::blocks`]'s FIRST entry
//! (`table.blocks()[0]`) is a stable per-sequence key. Concretely, in this
//! engine `block_size == max_seq_len` (see below) - a sequence's ENTIRE
//! lifetime (prompt + every generated token) fits in exactly one physical
//! block, allocated once by the first `reserve`/`append` call in
//! [`Engine::prefill`] and never touched again until [`Engine::release_table`]
//! frees it. So `table.blocks()[0]` is exactly the stable identity this
//! module needs, and it is used as the key into a PRIVATE `HashMap<u32,
//! GdnSlot>` ([`Engine::gdn_slots`]) this `Engine` owns - allocated (zeroed)
//! the first time a table is seen in [`Engine::prefill`], removed in
//! [`Engine::release_table`] (which still calls `BlockTable::release` for the
//! GQA side - the GDN map is an ADDITION, not a replacement).
//!
//! # The GQA side: a real per-block-id pool, not `Qwen35::step`'s single toy cache
//!
//! `Qwen35::step` decodes exactly one persistent sequence, into
//! `self.gqa_kcache`/`self.gqa_vcache` - fields that exist once per `Qwen35`
//! instance, not once per admitted request. A paged multi-request engine
//! needs that same per-layer KV cache to be addressable PER SEQUENCE. This
//! module's `Engine` uses `Qwen35::run_decode_step`'s own `DecodeCaches`
//! parameter (see its own doc) instead of reading `self.gqa_kcache`/
//! `self.gdn_state` - `Qwen35::step` itself is a thin wrapper passing its OWN
//! fields as a `DecodeCaches`, so its behaviour (and its `decode_step.rs`
//! test) is unchanged bit-for-bit.
//!
//! With that seam in place, this `Engine` preallocates, at construction, a
//! REAL pool: `num_blocks` dedicated `[block_size, kv_dim]` buffers per GQA
//! layer ([`Engine::gqa_k`]/[`Engine::gqa_v`], indexed `[physical block
//! id][layer]`) - real and resident from construction, not lazily grown.
//! `block_size == max_seq_len` is the choice that makes this a genuine "one
//! physical block backs one sequence's whole KV history" pool rather than
//! needing block-indirect (scatter/gather) attention kernels:
//! `model::block::gqa_decode_step` - this pass's decode primitive, reused
//! UNCHANGED - takes a flat `[cap, kv_dim]` `kcache`/`vcache` with no
//! block-table indirection at all, so "the pool" here is `num_blocks`
//! separate same-shape allocations rather than one large buffer sliced per
//! sequence.
//!
//! # `prefill`
//!
//! Loops over the prompt ONE TOKEN AT A TIME, calling
//! `Qwen35::run_decode_step` per token against this sequence's own
//! `DecodeCaches` (the pool slice for its physical block id, and its
//! `GdnSlot`). This is NOT `qwen3::serve::Engine`'s chunked, multi-token-per-
//! dispatch prefill - a per-token DISPATCH loop is the explicitly-sanctioned
//! correctness-first shape for this pass, and that performance gap (one
//! submit per prompt token instead of one batched whole-prompt forward) is
//! real and is intentionally left for later work, exactly the way
//! `crate::sample::generate_kv`'s own doc already names its identical
//! per-token-prefill gap. What M3.4 DID fix: every token used to also pay a
//! full host `gpu.read` of the hidden state, discarded on every iteration but
//! the last - that per-token READ is gone (the device buffer is chained
//! token-to-token; the loop reads back exactly once, after it ends).
//!
//! # `forward_batched_greedy`/`_window`/`forward_batched_topk`
//!
//! Given the "one truly-active sequence at a time" scope, these are thin
//! loops over the (in practice length-1, but handled generally)
//! `tables`/`inputs` slices, each iteration calling the same per-token decode
//! step. No real multi-sequence GPU-batched dispatch is built - if the
//! `Scheduler` has several requests running concurrently, each iteration
//! still serves them with N independent sequential dispatches, not one
//! batched one. What M3.4 DID fix: each iteration's own head projection +
//! sampling pick stays entirely on the device (`Qwen35::head_argmax_dev`/
//! `head_topk_dev`) - the `[d_model]` hidden state and the `[vocab]` logits
//! block are never shipped to the host, only the final token id / top-k
//! pairs are.
//!
//! # Deliberately deferred (not built in this pass)
//!
//! - **Prefix-cache reuse**: none. [`Engine::reclaim_prefix`] is a no-op
//!   returning 0, [`Engine::prefix_stats`] always reports `(0, 0, 0)`.
//! - **Chunked / batched prefill**: prompts are replayed one token at a time
//!   (one dispatch per token - see "`prefill`" above for what M3.4 did and
//!   did not change here).
//! - **Multi-sequence GPU batching**: `forward_batched_*` loop sequentially,
//!   one sequence's dispatch at a time.
//! - **int8/int4 paged KV, weight quantization, speculative decode**: not
//!   implemented; this `Engine` only ever builds a plain fp32 `Qwen35`.
//! - **Multi-GPU layer sharding**: single GPU only.
//! - **Vision / MTP**: text-only, matching `Qwen35::step`'s own scope.
//! - **On-device decode WINDOW**: [`Engine::decode_window_capacity`] is a
//!   small fixed host-side constant (`DECODE_WINDOW_CAPACITY == 1`) - each
//!   window step is still its own host round trip, unlike `qwen3::serve::
//!   Engine`'s multi-step device schedule. Top-K EXTRACTION itself (picking
//!   the best `k` of one step's own logits) IS on-device as of M3.4 - see
//!   above.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use crate::config::{LayerType, Qwen35Config};
use crate::model::{pipelines, DecodeCaches, Qwen35};

/// [`Engine::forward_batched_greedy_window`]'s host-side window cap. No
/// on-device windowing is built in this pass - 1 keeps
/// `model::serve::Scheduler`'s window logic exercised without ever
/// pretending there is real per-window batching underneath.
const DECODE_WINDOW_CAPACITY: usize = 1;

/// [`Engine::forward_batched_topk`]'s host-side candidate-list cap. Real
/// (non-greedy) sampling still works fully correctly at this width - it
/// only bounds how far into the vocabulary top-p's nucleus can reach.
const TOPK_CAPACITY: usize = 32;

/// One admitted sequence's persistent Gated-DeltaNet resources: recurrent
/// `state` + causal-conv `hist`, one pair per layer (a size-1 dummy at
/// GQA-layer indices). See this module's doc for why this lives in a
/// private `HashMap` keyed by `BlockTable::blocks()[0]` rather than a
/// `PagedDecoder`-carried parameter.
struct GdnSlot {
    state: Vec<DeviceBuffer>,
    hist: Vec<DeviceBuffer>,
}

impl GdnSlot {
    /// Allocate and explicitly ZERO every layer's buffers for a fresh
    /// sequence. `Gpu::storage` does not guarantee zero-initialised memory,
    /// and a fresh sequence's recurrent state / conv history MUST start at
    /// zero.
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
    fn bytes(cfg: &Qwen35Config) -> u64 {
        let bh = cfg.linear_num_value_heads as u64;
        let state_len = bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        let n_gdn = cfg.layer_types().iter().filter(|t| **t == LayerType::Linear).count() as u64;
        n_gdn * (state_len + hist_len) * 4
    }
}

/// Single-GPU, correctness-first `PagedDecoder` for Qwen3.8-27B. See this
/// module's doc for the full design.
pub struct Engine {
    /// Owns the device handle, weights (`ParamStore`), and the per-token
    /// decode-step primitives (`Qwen35::run_decode_step`) this whole engine
    /// is built on. Constructed with `b=1, t=1`: this instance's OWN
    /// `res`/`tokens`/`logits`/`gqa_kcache`/`gdn_state` fields (single-
    /// sequence decode state) are never touched by `Engine` - every decode
    /// step here supplies its own `DecodeCaches` - so they are sized to the
    /// smallest legal value.
    model: Qwen35,
    alloc: BlockAllocator,
    /// `== max_seq_len` (the hard per-sequence `prompt + max_new` cap) - see
    /// module doc for why this makes each physical block a whole sequence's
    /// entire KV history rather than a fixed-size page of it.
    block_size: u32,
    /// `[physical block id][layer]`: real, preallocated GQA KV cache, a
    /// size-1 dummy at GDN-layer indices. See module doc "The GQA side".
    gqa_k: Vec<Vec<DeviceBuffer>>,
    gqa_v: Vec<Vec<DeviceBuffer>>,
    /// GDN recurrent state / conv history, keyed by `BlockTable::blocks()[0]`,
    /// see module doc for why this is a private map rather than a trait
    /// parameter; populated in `Engine::prefill`, removed in
    /// `Engine::release_table`.
    gdn_slots: HashMap<u32, GdnSlot>,
}

impl Engine {
    /// Build from an in-memory weight map. `max_seq_len` is the hard cap on
    /// `prompt + max_new` for any ONE sequence (this engine's `block_size`,
    /// see module doc); `max_concurrent` is how many sequences may be
    /// resident at once (`num_blocks`) - together they size the real,
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
        // b=1, t=1: this instance's own decode-state fields are dead weight
        // for `Engine` (see `Engine::model`'s own doc) - t=1 is the smallest
        // legal prefill length (`gdn_chunk_size(1) == 1`, so `t % chunk ==
        // 0` holds for every config).
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
        Engine { model, alloc: BlockAllocator::new(max_concurrent, max_seq_len), block_size: max_seq_len, gqa_k, gqa_v, gdn_slots: HashMap::new() }
    }

    /// This sequence's `DecodeCaches` view: the GQA pool slice for its
    /// physical block id, and its `GdnSlot`. Panics if no slot exists -
    /// every live `BlockTable` this engine handed back from [`Engine::prefill`]
    /// has one, by construction.
    fn caches_for(&self, phys: u32) -> DecodeCaches<'_> {
        let slot = self
            .gdn_slots
            .get(&phys)
            .unwrap_or_else(|| panic!("qwen35::serve::Engine: no GdnSlot for physical block {phys} (table not prefilled by this engine, or already released)"));
        DecodeCaches { gqa_kcache: &self.gqa_k[phys as usize], gqa_vcache: &self.gqa_v[phys as usize], gqa_cap: self.block_size, gdn_state: &slot.state, gdn_hist: &slot.hist }
    }

    /// One decode step for an ALREADY-PREFILLED sequence: append one token
    /// position to `table` (never allocates a second physical block, see
    /// module doc) and run `Qwen35::run_decode_step` against its own
    /// `DecodeCaches`. `offset == pos`: since `block_size == max_seq_len`
    /// there is exactly one block per sequence, so the position WITHIN that
    /// block already IS the absolute decode position.
    ///
    /// Returns the DEVICE buffer, unread: [`Engine::forward_batched_greedy`]/
    /// [`Engine::forward_batched_topk`] chain it straight into
    /// `Qwen35::head_argmax_dev`/`head_topk_dev` without ever shipping the
    /// `[d_model]` hidden state to the host, matching qwen3's own device-head
    /// contract (M3.2/M3.4).
    fn decode_one(&mut self, table: &mut BlockTable, input: u32) -> DeviceBuffer {
        let (_block, offset) = table.append(&mut self.alloc).expect("qwen35::serve::Engine: KV pool exhausted mid-decode");
        let phys = table.blocks()[0];
        let caches = self.caches_for(phys);
        self.model.run_decode_step(input, offset, &caches, None)
    }

    /// `logits = hidden @ head^T`, projected on the DEVICE via
    /// `Qwen35::head_logits_dev` (the same resident head weight
    /// `run_forward`'s own head epilogue uses - no separate host copy) - only
    /// the `[vocab]` result comes back to the host. Required by
    /// `PagedDecoder::logits`'s trait contract (a caller that genuinely wants
    /// the raw vector); admission and steady-state decode never call this -
    /// see [`Engine::admit_greedy`]/[`Engine::admit_topk`] and
    /// [`Engine::forward_batched_greedy`]/[`Engine::forward_batched_topk`],
    /// which stay on the device end to end instead.
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "qwen35::serve::Engine::logits: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        let x = self.model.gpu.storage_init("qwen35.serve.logits.hidden", hidden);
        let logits = self.model.head_logits_dev(&x);
        self.model.gpu.read(&logits, self.model.cfg.vocab as usize)
    }

    /// Admission's greedy pick, entirely on the device: uploads `hidden`
    /// (ONE row) and reuses [`crate::model::Qwen35::head_argmax_dev`] - the
    /// same head matmul + argmax reduction [`Engine::forward_batched_greedy`]
    /// dispatches - so admission never ships a `[vocab]` block to the host
    /// either. Mirrors `qwen3::serve::Engine::admit_greedy` (M3.2).
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "admit_greedy: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        let x = self.model.gpu.storage_init("qwen35.serve.admit.hidden", hidden);
        self.model.head_argmax_dev(&x)
    }

    /// Admission's non-greedy candidates, entirely on the device: uploads
    /// `hidden` (ONE row) and reuses
    /// [`crate::model::Qwen35::head_topk_dev`] - the same top-k extraction
    /// [`Engine::forward_batched_topk`] uses - so admission never sorts a
    /// `[vocab]` vector on the host either. Mirrors `qwen3::serve::Engine::
    /// admit_topk` (M3.2).
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "admit_topk: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        let k = k.clamp(1, self.topk_capacity()) as u32;
        let x = self.model.gpu.storage_init("qwen35.serve.admit.hidden", hidden);
        self.model.head_topk_dev(&x, k)
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

    /// Prefill is un-chunked (one per-token loop over the WHOLE prompt every
    /// admission, see module doc) - there is no internal chunk size to
    /// report, so this returns the engine's own hard per-sequence capacity,
    /// the same size a single whole-prompt "chunk" would be.
    pub fn max_prefill_tokens(&self) -> u32 {
        self.block_size
    }

    /// No prefix cache in this pass (see module doc) - always 0 blocks
    /// reclaimed.
    pub fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }

    /// No prefix cache in this pass - always `(0, 0, 0)`, matching
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
    /// WORST-CASE ceiling (`num_blocks` slots), even though slots are
    /// allocated lazily and so may not ALL be resident at any one instant -
    /// this matches `PagedDecoder::kv_pool_bytes`'s own documented contract.
    pub fn kv_pool_bytes(&self) -> u64 {
        let n_full = self.model.cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        let num_blocks = self.alloc.num_blocks() as u64;
        let gqa_bytes = n_full * num_blocks * 2 * self.block_size as u64 * self.model.cfg.kv_dim() as u64 * 4;
        let gdn_ceiling = num_blocks * GdnSlot::bytes(&self.model.cfg);
        gqa_bytes + gdn_ceiling
    }

    /// `num_blocks * block_size` - see [`PagedDecoder::kv_pool_capacity_tokens`]'s
    /// doc. Independent of the GDN side (which has no "cached token count" -
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

    /// Release a finished/cancelled sequence: free its GDN slot (if any -
    /// note the key comes from `blocks()[0]` BEFORE `BlockTable::release`
    /// clears it) THEN run the ordinary KV release. The GDN map is an
    /// ADDITION to the trait's default block-release behaviour, not a
    /// replacement for it.
    pub fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(&phys) = t.blocks().first() {
            self.gdn_slots.remove(&phys);
        }
        t.release(&mut self.alloc);
    }

    /// Prefill a fresh prompt into `table`, one token at a time - see module
    /// doc "`prefill`" for why this is a per-token DISPATCH loop rather than
    /// a batched/chunked forward (that part is still deliberately deferred).
    ///
    /// M3.4: every token but the last used to pay a full `gpu.read(&h, d)` -
    /// a host-synchronising readback the caller immediately overwrote on the
    /// next iteration, since only the FINAL token's hidden state is ever
    /// wanted. `run_decode_step` is now chained token-to-token as a device
    /// buffer with no readback in the loop at all; the one real readback
    /// happens exactly once, after the loop, off the last token's hidden
    /// state - the same "submit every step, read back once" shape `qwen3::
    /// serve::Engine::prefill`'s own chunk loop uses (M3.1), ported here at
    /// token granularity instead of chunk granularity since this engine has
    /// no multi-token batched dispatch to chunk over.
    pub fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "prefill expects a fresh sequence");
        assert!(!prompt.is_empty(), "qwen35::serve::Engine::prefill: empty prompt (no token to produce a hidden state from)");
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
        // block (see module doc) - every later `append` (decode) call
        // reuses it.
        table.reserve(prompt.len() as u32, &mut self.alloc).expect("qwen35::serve::Engine: KV pool exhausted");
        let phys = table.blocks()[0];
        self.gdn_slots.entry(phys).or_insert_with(|| GdnSlot::new(&self.model.gpu, &self.model.cfg));

        let d = self.model.cfg.d_model as usize;
        let mut h: Option<DeviceBuffer> = None;
        for (i, &tok) in prompt.iter().enumerate() {
            let pos = i as u32;
            let caches = self.caches_for(phys);
            h = Some(self.model.run_decode_step(tok, pos, &caches, None));
        }
        let h = h.expect("prefill: prompt is non-empty (asserted above), so the loop ran at least once");
        self.model.gpu.read(&h, d)
    }

    /// One greedy decode step per (table, input) pair, sequentially (see
    /// module doc for why this is a loop, not a batched GPU dispatch) - but
    /// each step's head projection + argmax stays entirely on the device
    /// (M3.4): `decode_one`'s hidden state is never read back to the host,
    /// and `Qwen35::head_argmax_dev` reads back only the winning index.
    pub fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_greedy: tables/inputs length mismatch");
        let mut out = Vec::with_capacity(tables.len());
        for (t, &inp) in tables.iter_mut().zip(inputs) {
            let hidden = self.decode_one(t, inp);
            out.push(self.model.head_argmax_dev(&hidden));
        }
        out
    }

    /// [`Engine::forward_batched_greedy`], repeated `k` times per sequence,
    /// feeding each step's own greedy output back as the next input - there
    /// is no real on-device window in this pass, so this is
    /// host-orchestrated one dispatch at a time.
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
    /// top-`k` (token id, logit) candidates - extracted entirely on the
    /// device via `Qwen35::head_topk_dev` (M3.4: no `[vocab]` logits vector
    /// is ever shipped to the host to sort), `k` clamped to this engine's own
    /// capacity.
    pub fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity()) as u32;
        assert_eq!(tables.len(), inputs.len(), "forward_batched_topk: tables/inputs length mismatch");
        let mut out = Vec::with_capacity(tables.len());
        for (t, &inp) in tables.iter_mut().zip(inputs) {
            let hidden = self.decode_one(t, inp);
            out.push(self.model.head_topk_dev(&hidden, k));
        }
        out
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
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        Engine::admit_greedy(self, hidden)
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        Engine::admit_topk(self, hidden, k)
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

/// `model::serve::Scheduler<Engine>` - the continuous-batching scheduler
/// specialised to this engine, the same `Scheduler` type alias convention
/// `qwen3::serve::Scheduler` uses.
pub type Scheduler = model::serve::Scheduler<Engine>;
