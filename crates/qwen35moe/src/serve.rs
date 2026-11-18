// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-GPU `model::serve::PagedDecoder` for Qwen3.5-35B-A3B.
//!
//! Builds on `Qwen35::run_decode_batch` (`crates/qwen35moe/src/model.rs`) --
//! one decode token for each of several sequences, in one set of dispatches --
//! and `Qwen35::run_prefill_chunk` (one sequence's prompt in rounds). Steady
//! state decode is genuinely MULTI-SEQUENCE: one iteration's dispatch set
//! carries one row per resident sequence, so the weight reads that dominate a
//! decode step are paid once for the whole batch instead of once per request.
//! See "What this engine is, and is not" at the end for the feature set.
//!
//! # Two kinds of per-sequence state
//!
//! The 10 GQA layers keep K/V rows that grow with the context; the 30 GDN
//! layers keep a fixed-size recurrent `state` plus a causal-conv `hist` per
//! layer (`model::gdn::gdn_recurrent_step`/`gdn_causal_conv1d_step`'s own docs)
//! that does not. The engine holds them the way the dense 27B engine's paged
//! layout does (`qwen35::int8_gguf_resident`'s `KvLayout::Paged`):
//!
//! - **KV: one shared paged pool.** ONE `[pool_rows, kv_dim]` pool per GQA
//!   layer ([`Engine::gqa_kv`]), cut into blocks of
//!   [`EngineOptions::kv_block_rows`] rows handed out by a
//!   `model::paged::BlockAllocator`. A sequence's `BlockTable` takes a block as
//!   its context crosses into it, and every attention dispatch reaches a
//!   sequence's history through the paged kernels' block tables
//!   (`[batch, max_bt]` for a decode step, `[n, max_bt]` for a prefill round).
//!   The pool's size is a row count the placement plans
//!   ([`EngineOptions::kv_pool_rows`]), not `max_concurrent * max_seq_len`:
//!   short sequences leave blocks for others, and the scheduler admits on the
//!   blocks free.
//! - **Recurrent state: one pool row per sequence.** [`GdnPool`] holds
//!   `max_concurrent` rows of state and conv history per GDN layer; a sequence
//!   is given a free row when its prefill starts (recorded as its
//!   `BlockTable::state_row`) and gives it back in [`Engine::release_table`].
//!   A decode step stages any batch with one gather and one scatter per layer
//!   (`pool_rows_*2.wgsl`) by those rows. `max_concurrent` is therefore the
//!   most sequences the engine runs at once (`PagedDecoder::max_batch`).
//!
//! `Qwen35::run_decode_batch`/`run_prefill_chunk` take the pool and the
//! sequence's blocks and state row as explicit parameters
//! (`crate::model::BatchDecodeCaches`/`DecodeCaches`) instead of reading the
//! instance's own `gqa_kv`/`gdn_state`; `Qwen35::step` passes its OWN fields as
//! a one-block `DecodeCaches`, so its behaviour (and its `decode_step.rs` test)
//! is unchanged.
//!
//! # `prefill`
//!
//! Reserves the prompt's blocks, takes a state row, and runs the prompt in
//! rounds of [`EngineOptions::prefill_chunk`] tokens (`Qwen35::run_prefill_chunk`,
//! one dispatch shape per layer per round) over a private recurrent state,
//! which then becomes the sequence's pool row.
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
//! on-device multi-step schedule (see "What this engine is, and is not").
//!
//! # What this engine is, and is not (see the sections above for the design)
//!
//! - **Weights**: any [`model::ops::TierPolicy`] this model implements (fp32,
//!   int8), from any [`checkpoint::TensorSource`] - in particular straight from
//!   the released Q8_0 GGUF through [`crate::gguf_load::source`], so the 35B
//!   model is resident at ~38 GB with no fp32 copy anywhere.
//! - **KV**: one paged pool shared by every sequence, stored f32, bf16, per-row
//!   int8 or per-row fp8 ([`model::kv_tier::KvTier`], chosen in
//!   [`EngineOptions`]) and decoded through the fused split-key attention in
//!   every tier.
//! - **Prefill**: chunked - one dispatch shape per layer per round of
//!   [`EngineOptions::prefill_chunk`] tokens (`Qwen35::run_prefill_chunk`).
//! - **Head**: on the device. Greedy and top-k decode read back token ids, never
//!   a `[vocab]` logits row; only admission's [`Engine::logits`] reads one.
//! - **Not built**: prefix-cache reuse ([`Engine::reclaim_prefix`] returns 0: a
//!   recurrent layer resumes a prefix only from the state snapshot taken at its
//!   end, which this engine does not keep), speculative decode, multi-GPU layer
//!   sharding, vision/DeepStack (text only).

use std::collections::{BTreeSet, HashMap};

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu};
use model::gdn::{RecurrentSlot, RecurrentSlotShape};
use model::kv_tier::{KvLayer, KvTier};
use model::ops::TierPolicy;
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use crate::config::{LayerType, Qwen35Config};
use crate::q8::Qwen35Q8;
use crate::model::{pipelines, MOE_GROUPED_MIN_ROWS, BatchDecodeCaches, BatchSeq, DecodeCaches, DecodeHead, DecodeOut, DecodeTape, GdnStore, Hidden, Qwen35};

/// [`Engine::forward_batched_greedy_window`]'s window cap. No on-device
/// multi-step schedule is built (see module doc) - 1 keeps
/// `model::serve::Scheduler`'s window logic exercised (it always calls this with
/// `k <= decode_window_capacity()`) without pretending there is real per-window
/// batching underneath.
const DECODE_WINDOW_CAPACITY: usize = 1;

/// [`Engine::forward_batched_topk`]'s candidate-list cap: how many rounds of
/// device argmax-and-mask the top-k head runs (`Qwen35::head_topk_rows_dev`).
const TOPK_CAPACITY: usize = 32;

/// The `t` a quantized engine builds its `Qwen35` at: only the single-sequence
/// buffers (`logits`, `res`, the training tape's tokens) are sized by it, and a
/// serving engine uses none of them, so the smallest legal value is the right
/// one. An fp32 engine's grouped-expert scratch is sized by it too - see
/// `Engine::from_source_on`.
const SERVING_T: u32 = 64;

/// The activation bytes an engine built with `opts` allocates beyond its weights, KV pool and recurrent state: the
/// working set of a decode step over every resident sequence, which the recorded steps hold between steps
/// ([`decode_step_bytes`]), and beside it one prefill round. What a placement keeps free for it
/// (`catalog::resident_qwen35moe`), and what the engine's handles keep of their freed blocks.
///
/// The device is not known when a placement asks, so a round size left to the device is charged at the widest the
/// policy chooses ([`model::serve::TENSOR_CORE_PREFILL_ROUND_ROWS`]).
pub fn activation_bytes(cfg: &Qwen35Config, opts: &EngineOptions) -> u64 {
    let rows = opts.prefill_chunk.unwrap_or(model::serve::TENSOR_CORE_PREFILL_ROUND_ROWS);
    decode_step_bytes(cfg, opts.max_concurrent, opts.kv_tier, opts.max_seq_len) + prefill_round_bytes(cfg, rows)
}

/// A recorded decode step keeps every temporary of every layer (nothing inside one recording is reused), as a ratio of
/// a layer's words a row ([`prefill_round_bytes`]). Measured on gpu3 (GH200, int8 35B-A3B, int8 KV, the shared
/// recording scratch): 0.021 / 0.218 / 0.892 GiB at 1 / 16 / 64 rows, 1.37 of the layer's words a row and layer beyond
/// the head's logits. Charged as 3/2.
const DECODE_STEP_LAYER_WORDS: (u64, u64) = (3, 2);

/// Fixed device bytes of a recorded decode step whatever its rows (the routing tables, the per-layer metadata): the
/// one-row step measured above, with a margin.
const DECODE_STEP_FIXED_BYTES: u64 = 32 << 20;

/// Device bytes the working set of one decode step of `rows` rows holds: every layer's temporaries, the head's logits
/// and candidates, and every attention layer's split records for sequences of up to `max_seq_len` positions in a
/// `kv_tier` pool ([`model::kv_tier::flash_part_words_bound`]). The records grow with the context the step is sized
/// for, not with the rows' words: a step sized for a 256K context holds them for every attention layer, which the
/// layer ratio, measured at a 74K context, does not cover. The recorded steps of every batch size share one such set,
/// the widest step's ([`gpu_core::tape::TapeScratch`]), so this at the engine's widest batch is what they hold.
pub fn decode_step_bytes(cfg: &Qwen35Config, rows: u32, kv_tier: KvTier, max_seq_len: u32) -> u64 {
    let (num, den) = DECODE_STEP_LAYER_WORDS;
    let layers = u64::from(cfg.n_layers) * layer_words(cfg) * 4 * num / den;
    let head = 2 * u64::from(cfg.vocab) * 4;
    let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
    let splits = n_full * 4 * model::kv_tier::flash_part_words_bound(kv_tier, rows, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, max_seq_len);
    DECODE_STEP_FIXED_BYTES + u64::from(rows) * (layers + head) + splits
}

/// f32 words of one layer's intermediates a row: the residual pair, the larger of the recurrent and the attention
/// mixer's projections, and the expert sublayer's per-slot gate, up and product (the routed experts and the shared one).
fn layer_words(cfg: &Qwen35Config) -> u64 {
    let w = u64::from;
    let d = w(cfg.d_model);
    let recurrent = d + 3 * w(cfg.linear_conv_dim()) + 3 * w(cfg.linear_value_dim()) + d;
    let attention = d + w(cfg.q_proj_dim()) + 2 * w(cfg.kv_dim()) + 2 * w(cfg.q_dim()) + d;
    let experts = d + 3 * w(cfg.top_k + 1) * w(cfg.moe_intermediate_size) + d;
    2 * d + recurrent.max(attention) + experts
}

/// Layers' worth of one layer's intermediates a prefill round holds at its peak, as a ratio. Measured on gpu3 (GH200,
/// int8 35B-A3B, bf16 KV, `prefill_round_census::the_activations_of_a_prefill_round_of_the_real_checkpoint_fit_their_
/// charge`): 0.31 GiB above the resident at 256-row rounds and 3.42 GiB at 4096, 4.1 and 3.5 of the layer's words a
/// row past the private recurrent slot. Not a count of live layers: the f32 Gated-DeltaNet chunk form (a device without
/// the segment kernels) allocates scratch of its own beyond the projections the layer's words count; on the segment
/// kernels the 4096-row peak is 2.15 GiB (2.1). Charged as 9/2 so the bound holds for both forms with a margin.
const PREFILL_LIVE_LAYERS: (u64, u64) = (9, 2);

/// Device bytes one prefill round of `rows` rows allocates beyond the weights, the KV pool and the state pool: the
/// sequence's private recurrent slot (the prompt runs over it before it becomes the pool row) and
/// [`PREFILL_LIVE_LAYERS`] of a layer's f32 intermediates a row ([`layer_words`]).
pub fn prefill_round_bytes(cfg: &Qwen35Config, rows: u32) -> u64 {
    let w = u64::from;
    let layer_words = layer_words(cfg);
    let (num, den) = PREFILL_LIVE_LAYERS;
    let slot = gdn_slot_shape(cfg);
    let slot_bytes = 4 * slot.is_recurrent.iter().filter(|&&r| r).count() as u64 * (slot.state_len + slot.hist_len);
    slot_bytes + w(rows) * layer_words * 4 * num / den
}

/// How an [`Engine`] is built: capacity, KV pool, weight tier, KV tier, prefill round.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// The hard cap on `prompt + max_new` for any ONE sequence.
    pub max_seq_len: u32,
    /// How many sequences may be resident at once: the recurrent-state rows ([`GdnPool`]).
    pub max_concurrent: u32,
    /// Rows of one KV block ([`Self::block_rows`] clamps it to `max_seq_len`).
    pub kv_block_rows: u32,
    /// Rows of the shared KV pool, in whole blocks; `None` gives every one of `max_concurrent` sequences a whole
    /// context ([`Self::pool_rows`]). A placement sizes it to what the card holds beside everything else.
    pub kv_pool_rows: Option<u32>,
    pub tier: TierPolicy,
    pub kv_tier: KvTier,
    /// Prompt tokens per prefill round; `None` takes the device's ([`model::serve::prefill_round_rows_for`]: wider on
    /// tensor cores, where a round's per-dispatch cost is a large share of a narrow one).
    pub prefill_chunk: Option<u32>,
    /// Record a decode step once per batch shape and replay it (`Qwen35::
    /// record_decode`) instead of building it every token. Only the int8 expert
    /// tier can be recorded; for an fp32 engine this is ignored.
    pub decode_tapes: bool,
    /// Rows from which the int8 expert GEMMs group each expert's slots
    /// ([`Qwen35::set_moe_grouped_min_rows`]).
    pub moe_grouped_min_rows: u32,
    /// Take the fused native decode kernels where the device is offered them
    /// ([`Qwen35::set_decode_fusion`]).
    pub decode_fusion: bool,
    /// The arithmetic of the fp32-weight products ([`Qwen35::set_gemm_precision`]).
    pub gemm_precision: GemmPrecision,
}

impl EngineOptions {
    /// fp32 weights and KV - the engine's original, reference configuration.
    pub fn new(max_seq_len: u32, max_concurrent: u32) -> EngineOptions {
        EngineOptions {
            max_seq_len,
            max_concurrent,
            kv_block_rows: model::paged::KV_BLOCK_ROWS,
            kv_pool_rows: None,
            tier: TierPolicy::uniform(Dtype::F32),
            kv_tier: KvTier::F32,
            prefill_chunk: None,
            decode_tapes: true,
            moe_grouped_min_rows: MOE_GROUPED_MIN_ROWS,
            decode_fusion: true,
            gemm_precision: GemmPrecision::Auto,
        }
    }
    /// Rows of one KV block: [`Self::kv_block_rows`], at most one whole sequence (an engine whose sequences are shorter
    /// than a block gives each one block of its context).
    pub fn block_rows(&self) -> u32 {
        self.kv_block_rows.clamp(1, self.max_seq_len.max(1))
    }
    /// Rows of the shared KV pool: [`Self::kv_pool_rows`] in whole blocks, or room for `max_concurrent` whole contexts.
    pub fn pool_rows(&self) -> u64 {
        let block = self.block_rows() as u64;
        match self.kv_pool_rows {
            Some(rows) => rows as u64 / block * block,
            None => self.max_concurrent as u64 * (self.max_seq_len as u64).div_ceil(block) * block,
        }
    }
    pub fn with_kv_block_rows(mut self, rows: u32) -> EngineOptions {
        self.kv_block_rows = rows.max(1);
        self
    }
    pub fn with_kv_pool_rows(mut self, rows: u32) -> EngineOptions {
        self.kv_pool_rows = Some(rows);
        self
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
        self.prefill_chunk = Some(rows.max(1));
        self
    }
    pub fn with_moe_grouped_min_rows(mut self, rows: u32) -> EngineOptions {
        self.moe_grouped_min_rows = rows;
        self
    }
    pub fn with_decode_fusion(mut self, on: bool) -> EngineOptions {
        self.decode_fusion = on;
        self
    }
    pub fn with_decode_tapes(mut self, on: bool) -> EngineOptions {
        self.decode_tapes = on;
        self
    }
    pub fn with_gemm_precision(mut self, precision: GemmPrecision) -> EngineOptions {
        self.gemm_precision = precision;
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

/// Every resident sequence's Gated-DeltaNet recurrent state: per layer one pool
/// of `rows` rows of state and one of conv history, row `r` belonging to the
/// sequence whose `BlockTable::state_row` is `r`. Zeroed once at construction; a
/// row is overwritten wholesale when its sequence is admitted.
struct GdnPool {
    /// `[layer]`: `[rows, state_len]`, a size-1 placeholder at GQA layers.
    state: Vec<DeviceBuffer>,
    hist: Vec<DeviceBuffer>,
    shape: RecurrentSlotShape,
    /// Rows given to no sequence, the lowest handed out first.
    free: BTreeSet<u32>,
    /// Rows in all: the most sequences resident at once.
    rows: u32,
    scatter: usize,
}

impl GdnPool {
    fn new(gpu: &Gpu, shape: RecurrentSlotShape, rows: u32) -> GdnPool {
        // The row kernels address a pool through u32 element offsets.
        assert!(
            (rows as u64) * shape.state_len.max(shape.hist_len) <= u32::MAX as u64,
            "qwen35moe::serve::Engine: {rows} recurrent-state rows of {} elements are past a u32 element offset",
            shape.state_len
        );
        let (mut state, mut hist) = (Vec::new(), Vec::new());
        for &recurrent in &shape.is_recurrent {
            let (s, h) = if recurrent { (rows as u64 * shape.state_len, rows as u64 * shape.hist_len) } else { (1, 1) };
            state.push(gpu.storage(s.max(1)));
            hist.push(gpu.storage(h.max(1)));
        }
        let clears: Vec<&DeviceBuffer> = state.iter().chain(hist.iter()).collect();
        gpu.submit(&clears, &[]);
        let scatter = gpu.kernel_index("pool_rows_scatter2").expect("pool_rows_scatter2 is in this model's kernel set");
        GdnPool { state, hist, shape, free: (0..rows).collect(), rows, scatter }
    }

    /// A row no sequence holds, or `None` when every row is given.
    fn take(&mut self) -> Option<u32> {
        self.free.pop_first()
    }

    /// Make `slot` the state of the sequence on `row`: one scatter per
    /// recurrent layer, overwriting the whole row.
    fn store(&mut self, gpu: &Gpu, row: u32, slot: &GdnSlot) {
        let ids = gpu.storage(1);
        gpu.write(&ids, &[row]);
        let (sl, hl) = (self.shape.state_len as u32, self.shape.hist_len as u32);
        let steps: Vec<_> = (0..self.shape.is_recurrent.len())
            .filter(|&l| self.shape.is_recurrent[l])
            .map(|l| gpu.step(self.scatter, &[&slot.state[l], &slot.hist[l], &ids, &self.state[l], &self.hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
        // The slot is about to be freed and its scatters must have read it first: in submission order on a
        // backend that runs it so (a freed block is only handed to work queued behind them), by a drain elsewhere.
        if !gpu.submissions_ordered() {
            gpu.poll_wait();
        }
    }

    /// Give `row` back: its state is no sequence's any more.
    fn give_back(&mut self, row: u32) {
        assert!(self.free.insert(row), "qwen35moe::serve::Engine: recurrent-state row {row} given back twice");
    }

    /// Whether `row` is a sequence's.
    fn is_live(&self, row: u32) -> bool {
        row < self.rows && !self.free.contains(&row)
    }
}

/// Device bytes of the shared GQA pool of `pool_rows` rows in `kv_tier`, plus
/// `state_rows` Gated-DeltaNet state rows (one per sequence that may run) - a
/// prediction made before any device allocation, as `PagedDecoder::kv_pool_bytes`
/// requires, and what a placement charges ([`EngineOptions::pool_rows`],
/// [`EngineOptions::max_concurrent`]).
pub fn kv_pool_bytes(cfg: &Qwen35Config, kv_tier: KvTier, pool_rows: u64, state_rows: u32) -> u64 {
    gqa_pool_bytes(cfg, kv_tier, pool_rows) + state_rows as u64 * GdnSlot::bytes(&gdn_slot_shape(cfg))
}

/// Device bytes of a shared GQA pool of `pool_rows` rows in `kv_tier`, every full-attention layer's K and V planes.
pub fn gqa_pool_bytes(cfg: &Qwen35Config, kv_tier: KvTier, pool_rows: u64) -> u64 {
    let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
    n_full * 2 * kv_tier.plane_bytes(pool_rows, cfg.kv_dim() as u64, cfg.head_dim as u64)
}

/// Device bytes the weights occupy for `cfg` under `tier`, predicted before any
/// allocation: an int8 linear is packed words plus one f32 scale per 32
/// (`9/8` byte per weight), everything else - norms, the router, the GDN
/// scalars, the embedding table - fp32. The leaves that are int8 are exactly
/// those `Qwen35::new_on_tier_src` quantises: the mixer linears, the routed
/// experts, the shared expert (it rides in the expert banks when it has their
/// shape) and the untied head.
pub fn weight_bytes(cfg: &Qwen35Config, tier: &TierPolicy) -> u64 {
    let shared_in_banks = Qwen35Q8::shared_fits_bank(cfg) && tier.want("blocks.0.mlp.experts.0.gate.weight") == Dtype::I8;
    cfg.param_list()
        .into_iter()
        .map(|(name, count)| {
            let linear = Qwen35Q8::is_i8_linear(&name)
                || name == "lm_head.weight"
                || (name == "tok.weight" && crate::model::embed_table_i8(cfg, tier, false))
                || (shared_in_banks && Qwen35Q8::is_shared_expert_linear(&name));
            let bytes_per_weight_x8 = if linear && tier.want(&name) == Dtype::I8 { 9 } else { 32 };
            count as u64 * bytes_per_weight_x8 / 8
        })
        .sum()
}

/// Recorded decode steps kept per engine. Their temporaries are one shared set (the widest step's,
/// [`Engine::tape_scratch_bytes`]), so a tape costs its small per-step inputs and its graph; the cache is still bounded
/// and the least recently used shape is dropped first.
const MAX_DECODE_TAPES: usize = 8;

/// What a recorded decode step is bound to: how many rows it carries, what its
/// head returns, and the context its attention's split count is sized for
/// ([`Engine::decode_cap`]). Which sequences those rows are - their blocks,
/// positions, tokens - is data the step reads from its metadata buffers, so one
/// recording serves every batch of that size whose sequences are no longer.
#[derive(Clone, PartialEq, Eq, Hash)]
struct TapeKey {
    rows: u32,
    head: HeadKey,
    cap: u32,
}

/// The head a recorded step ends in: a greedy pick, top-`k` candidates, or the logits themselves (one recording for
/// every step whose rows the host reads or draws from after the step).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum HeadKey {
    Greedy,
    TopK(u32),
    Logits,
}

impl HeadKey {
    fn of(head: DecodeHead) -> HeadKey {
        match head {
            DecodeHead::Greedy => HeadKey::Greedy,
            DecodeHead::TopK(k) => HeadKey::TopK(k),
            DecodeHead::Rows | DecodeHead::Drawn(_) => HeadKey::Logits,
        }
    }
}

/// The shortest context a recorded step's attention is sized for: steps over short sequences share one recording.
const DECODE_CAP_FLOOR: u32 = 4096;

/// Where a profiled stretch of decode or prefill went: whole-pass wall time, and
/// per-kernel DEVICE time (the table that ranks the work - a loaded host
/// stretches the wall clock, never a kernel's own timestamp pair).
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Passes profiled in each region.
    pub passes: u32,
    /// Production-path wall seconds over `passes` passes.
    pub wall_s: f64,
    /// The fastest single production-path pass, seconds: on a shared host the
    /// mean is what OTHER tenants' load makes it, the minimum is the cost of
    /// the work itself.
    pub best_pass_s: f64,
    /// The median production-path pass, seconds.
    pub median_pass_s: f64,
    /// Seconds per pass the launching thread spent RUNNING (its own CPU time:
    /// the host cost of recording and launching the pass), and seconds it spent
    /// runnable-but-not-running because other tenants held the cores. Their
    /// difference from `wall_s / passes` is the device wait. `(0, 0)` where the
    /// kernel exposes no per-thread scheduler statistics.
    pub host_run_s: f64,
    pub host_wait_s: f64,
    /// Per production-path pass: `(submissions, dispatches, host writes, uniform
    /// blocks allocated)` - what the host paid to enqueue it. `None` where the
    /// backend keeps no counters.
    pub host_ops_per_pass: Option<(f64, f64, f64, f64)>,
    /// Wall seconds of the same passes with timestamp queries armed (inflated;
    /// for the table only).
    pub timed_wall_s: f64,
    /// `(kernel, device ms over the timed passes, dispatch count)`, slowest
    /// first. Empty on a backend with no kernel timing.
    pub rows: Vec<(String, f64, u64)>,
}

impl KernelProfile {
    /// Summed kernel device time over the timed passes, ms.
    pub fn device_ms(&self) -> f64 {
        self.rows.iter().map(|r| r.1).sum()
    }
    /// Device milliseconds per pass.
    pub fn device_ms_per_pass(&self) -> f64 {
        self.device_ms() / self.passes.max(1) as f64
    }
}

/// The calling thread's `(running, runnable-waiting)` nanoseconds so far, from the
/// kernel's per-thread scheduler statistics; `None` where there are none.
fn thread_sched_ns() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let mut f = text.split_whitespace().map(|v| v.parse::<u64>().ok());
    Some((f.next()??, f.next()??))
}

/// Single-GPU `PagedDecoder` for Qwen3.5/3.6-35B-A3B. See this module's doc for
/// the full design (the shared paged KV pool, the recurrent-state rows, and
/// what is and is not built).
pub struct Engine {
    /// Recorded decode steps by shape, least recently used first. Declared
    /// before `model` so the buffers a tape pins are released before the device
    /// handle that owns them.
    /// Each with the [`gpu_core::tape::TapeScratch::replacements`] count it was recorded at.
    tapes: Vec<(TapeKey, DecodeTape, u64)>,
    /// The temporaries every recorded step binds, shared: steps never run at once.
    tape_scratch: gpu_core::tape::TapeScratch,
    decode_tapes: bool,
    /// Owns the device handle, the weights, and the decode/prefill primitives
    /// (`Qwen35::run_decode_batch`, `Qwen35::run_prefill_chunk`) this whole engine
    /// is built on. Its OWN single-sequence decode state (`gqa_kv`/`gdn_state`) is
    /// never touched by `Engine` - every step here supplies its own
    /// `DecodeCaches`/`BatchDecodeCaches` - so it is built at the smallest `t`
    /// the weight tier allows.
    model: Qwen35,
    /// The shared pool's blocks of [`Self::block_size`] rows.
    alloc: BlockAllocator,
    /// Rows of one KV block ([`EngineOptions::block_rows`]).
    block_size: u32,
    /// The hard per-sequence `prompt + max_new` cap.
    max_seq_len: u32,
    /// Block-table entries of a decode step's rows: the blocks of one whole sequence, so a recorded step serves
    /// every batch of its size whatever its sequences' lengths.
    max_bt: u32,
    /// `[layer]`: ONE real, preallocated `[num_blocks*block_size, kv_dim]` GQA KV
    /// pool per full-attention layer (K and V planes in `kv_tier`), a
    /// placeholder at GDN-layer indices. Physical block `p` owns rows
    /// `p*block_size .. +block_size`. See module doc.
    gqa_kv: Vec<KvLayer>,
    kv_tier: KvTier,
    prefill_chunk: u32,
    /// GDN recurrent state / conv history, one pool row per sequence - see
    /// module doc. Given and written in [`Engine::prefill`] (or
    /// [`Engine::admit_synthetic`]), given back in [`Engine::release_table`].
    gdn: GdnPool,
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
        let (block_size, pool_rows) = (opts.block_rows(), opts.pool_rows());
        let max_bt = opts.max_seq_len.div_ceil(block_size);
        assert!(
            pool_rows >= max_bt as u64 * block_size as u64,
            "qwen35moe::serve::Engine: a KV pool of {pool_rows} rows cannot hold one sequence of {} tokens in blocks of {block_size}",
            opts.max_seq_len
        );
        assert!(
            KvTier::fits_addressing(pool_rows, kv_dim),
            "qwen35moe::serve::Engine: a KV pool of {pool_rows} rows of {kv_dim} elements is past what the paged kernels' u32 element offsets address"
        );
        // `t` sizes this instance's fixed scratch. A quantized build's only
        // `rows`-bound scratch is the expert sublayer's activation packing, and a
        // pass wider than it packs into room of its own, so the smallest `t`
        // serves; an fp32 build's grouped-GEMM expert scratch is
        // asserted against `rows * top_k` at dispatch, and the widest row count
        // this engine submits is a prefill round or one decode token per resident
        // sequence. An int8 request is an fp32 build where the device has no
        // packed int8 dot (`Qwen35::new_on_tier_src` holds it at fp32), so the
        // build that is sized is the one the device runs, not the one asked for.
        let prefill_chunk = opts.prefill_chunk.unwrap_or_else(|| model::serve::prefill_round_rows_for(gpu_core::provider::cuda::tensor_core_kernels_enabled(&gpu)));
        let quantized = opts.tier.quantizes_anything() && gpu.caps().numeric.int8_dot;
        let t = if quantized { SERVING_T } else { opts.max_concurrent.max(prefill_chunk) };
        let mut model = Qwen35::new_on_tier_src(gpu, cfg.clone(), 1, t, src, &opts.tier);
        model.set_gemm_precision(opts.gemm_precision);
        model.set_moe_grouped_min_rows(opts.moe_grouped_min_rows);
        model.set_decode_fusion(opts.decode_fusion);
        // The activations the placement charges this engine (`catalog::resident_qwen35moe`): the larger of a decode
        // step over every resident sequence and one prefill round. Their freed blocks stay with the handles.
        model.retain_freed_blocks(activation_bytes(&cfg, &EngineOptions { prefill_chunk: Some(prefill_chunk), ..opts.clone() }));
        let n_layers = cfg.n_layers as usize;
        let mut gqa_kv: Vec<KvLayer> = Vec::with_capacity(n_layers);
        for ty in cfg.layer_types() {
            gqa_kv.push(match ty {
                LayerType::Full => KvLayer::new(&model.gpu, opts.kv_tier, pool_rows, kv_dim, cfg.head_dim as u64),
                LayerType::Linear => KvLayer::placeholder(&model.gpu),
            });
        }
        let decode_tapes = opts.decode_tapes && model.moe_device_routed();
        let gdn = GdnPool::new(&model.gpu, gdn_slot_shape(&cfg), opts.max_concurrent);
        Engine {
            tapes: Vec::new(),
            tape_scratch: gpu_core::tape::TapeScratch::new(),
            decode_tapes,
            model,
            alloc: BlockAllocator::new((pool_rows / block_size as u64) as u32, block_size),
            block_size,
            max_seq_len: opts.max_seq_len,
            max_bt,
            gqa_kv,
            kv_tier: opts.kv_tier,
            prefill_chunk,
            gdn,
        }
    }

    /// The `DecodeCaches` of a sequence being prefilled into the pool blocks
    /// `table`: those rows of the GQA pool and the private recurrent state `slot`
    /// that [`GdnPool::store`] moves into the pool once the prompt is in.
    fn prefill_caches<'a>(&'a self, table: &'a [u32], slot: &'a GdnSlot) -> DecodeCaches<'a> {
        DecodeCaches { gqa_kv: &self.gqa_kv, gqa_cap: self.max_seq_len, block_size: self.block_size, table, gdn_state: &slot.state, gdn_hist: &slot.hist }
    }

    /// The recurrent-state row of `table`, given now if it has none. The scheduler never runs more sequences than there
    /// are rows ([`PagedDecoder::max_batch`]), so a sequence it admits always finds one.
    fn assign_state_row(&mut self, table: &mut BlockTable) -> Result<u32, String> {
        if let Some(row) = table.state_row() {
            return Ok(row);
        }
        let row = self.gdn.take().ok_or_else(|| format!("qwen35moe::serve::Engine: every one of the {} recurrent-state rows is in use", self.gdn.rows))?;
        table.set_state_row(row);
        Ok(row)
    }

    /// A zeroed private recurrent state for a sequence about to be admitted.
    fn fresh_slot(&self) -> GdnSlot {
        GdnSlot::new(&self.model.gpu, &gdn_slot_shape(&self.model.cfg))
    }

    /// Append one position to every table - a block taken from the pool when the
    /// sequence crosses into one; the scheduler made room for it
    /// (`model::serve::Scheduler::ensure_decode_room`) - and return each
    /// sequence's `(state row, absolute position)`. A table this engine never
    /// admitted (or already released) has no recurrent state to step: a caller
    /// bug, not a condition to degrade from.
    fn advance_tables(&mut self, tables: &mut [&mut BlockTable]) -> Vec<(u32, u32)> {
        assert!(!tables.is_empty(), "qwen35moe::serve::Engine: empty batch");
        tables
            .iter_mut()
            .map(|t| {
                let row = t.state_row().filter(|&r| self.gdn.is_live(r));
                let row = row.unwrap_or_else(|| panic!("qwen35moe::serve::Engine: a sequence with no recurrent state (table not prefilled by this engine, or already released)"));
                let pos = t.len();
                t.append(&mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted mid-decode");
                (row, pos)
            })
            .collect()
    }

    /// The `BatchSeq` rows of `tables` at `coords` ([`Self::advance_tables`]).
    fn seqs_for<'a>(tables: &'a [&mut BlockTable], coords: &[(u32, u32)]) -> Vec<BatchSeq<'a>> {
        tables.iter().zip(coords).map(|(t, &(state_row, pos))| BatchSeq { state_row, pos, table: t.blocks() }).collect()
    }

    /// The context a decode step over `coords` is sized for: the longest sequence's next length rounded up to a power of
    /// two, at least [`DECODE_CAP_FLOOR`] and at most the engine's context. A recorded step's attention splits its keys
    /// and holds a record per split by this (`model::kv_tier::flash_part_words_bound`), so a step sized for the whole
    /// context would pay for it in every attention layer whatever lengths it carries; rounded, a growing batch is
    /// recorded again only when its longest sequence doubles.
    fn decode_cap(&self, coords: &[(u32, u32)]) -> u32 {
        let live = coords.iter().map(|&(_, pos)| pos + 1).max().unwrap_or(1);
        live.next_power_of_two().max(DECODE_CAP_FLOOR).min(self.max_seq_len)
    }

    /// The caches of a decode step over `seqs`, sized for sequences of up to `cap` positions.
    fn batch_caches<'a>(&'a self, seqs: &'a [BatchSeq<'a>], cap: u32) -> BatchDecodeCaches<'a> {
        BatchDecodeCaches {
            gqa_kv: &self.gqa_kv,
            gqa_cap: cap,
            block_size: self.block_size,
            max_bt: self.max_bt,
            seqs,
            gdn: GdnStore::Pool { state: &self.gdn.state, hist: &self.gdn.hist },
        }
    }

    /// ONE decode step for every `(sequence, input)` pair at once - the engine's
    /// whole steady-state decode path, and a single set of GPU dispatches
    /// whatever the batch size - ending in the device head `head` selects.
    ///
    /// On the int8 tier the step is recorded once per batch shape
    /// ([`Qwen35::record_decode`]) and every later token replays it: the host
    /// writes a few small buffers and makes one submission instead of building
    /// ~1600 dispatches. Otherwise the step is built each time, its scratch drawn
    /// from the replay arena so the backend can still replay what repeats.
    fn decode(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], head: DecodeHead) -> DecodeOut {
        // A sequence's tokens must not change with which sequences share its step through an fp32 product: the router
        // (a hard top-k) on the library's TF32 GEMM rounds a row differently at two rows than at one.
        model::ops::fp32_kernels(|| self.decode_rows(tables, inputs, head))
    }

    fn decode_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], head: DecodeHead) -> DecodeOut {
        assert_eq!(tables.len(), inputs.len(), "qwen35moe::serve::Engine: tables/inputs length mismatch");
        let coords = self.advance_tables(tables);
        // A recording holds no vendor-library product, so the NVFP4 tier's stored-format GEMVs (up to GEMV_MAX_ROWS rows)
        // are what a recorded step runs; a taller batch decodes unrecorded, where the tile product serves it.
        let unrecordable =
            self.model.moe_nvfp4_active() && inputs.len() as u32 > model::ops::GEMV_MAX_ROWS;
        let tables: &[&mut BlockTable] = tables;
        if !self.decode_tapes || unrecordable {
            let rows = inputs.len() as u32;
            let seqs = Self::seqs_for(tables, &coords);
            let caches = self.batch_caches(&seqs, self.max_seq_len);
            // The readback that ends the step is the drain the arena's contract asks for.
            let _scope = self.model.gpu.scratch_scope();
            let hidden = self.model.run_decode_batch(inputs, &caches);
            return match head {
                DecodeHead::Greedy => DecodeOut::Greedy(self.model.head_argmax_rows_dev(&hidden, rows)),
                DecodeHead::TopK(k) => DecodeOut::TopK(self.model.head_topk_rows_dev(&hidden, rows, k)),
                DecodeHead::Rows | DecodeHead::Drawn(_) => self.model.read_logits(&self.model.head_logits_rows_of(&hidden, rows), rows, head),
            };
        }
        let cap = self.decode_cap(&coords);
        let key = TapeKey { rows: inputs.len() as u32, head: HeadKey::of(head), cap };
        match self.tapes.iter().position(|(k, _, _)| *k == key) {
            // Most recently used last.
            Some(i) => {
                let hit = self.tapes.remove(i);
                self.tapes.push(hit);
            }
            None => {
                let tape = {
                    let seqs = Self::seqs_for(tables, &coords);
                    let caches = self.batch_caches(&seqs, cap);
                    self.model.record_decode(inputs, &caches, head, &self.tape_scratch)
                };
                if self.tapes.len() >= MAX_DECODE_TAPES {
                    self.tapes.remove(0);
                }
                // A step wider than any before grew a shared slot: the tapes recorded earlier bind the old buffer and
                // would hold it beside the set, so they go (and are recorded again against the grown set when used).
                let generation = self.tape_scratch.replacements();
                self.tapes.retain(|(_, _, g)| *g == generation);
                self.tapes.push((key, tape, generation));
            }
        }
        let (_, tape, _) = self.tapes.last().expect("the tape for this shape was just found or recorded");
        let seqs = Self::seqs_for(tables, &coords);
        self.model.replay_decode(tape, inputs, &seqs, head)
    }

    /// `logits = hidden @ head^T` for one `[d_model]` hidden row, on the device
    /// (admission's first-token sampling is the only caller; steady-state decode
    /// never reads a logits row).
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let g = &self.model.gpu;
        let h = self.admit_hidden(hidden);
        let logits = self.model.head_logits_rows_dev(&h, 1);
        g.read(&logits, self.model.cfg.vocab as usize)
    }

    /// Admission's greedy pick on the device: the head and the argmax reduction a greedy decode step runs, so only
    /// the token comes back, not a `[vocab]` row for the host to scan.
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        let h = self.admit_hidden(hidden);
        self.model.head_argmax_rows_dev(&Hidden::plain(h), 1)[0]
    }

    /// Admission's top-`k` candidates on the device ([`Engine::forward_batched_topk`]'s extraction), instead of the
    /// host sorting the whole vocabulary.
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let h = self.admit_hidden(hidden);
        let k = k.clamp(1, self.topk_capacity()) as u32;
        self.model.head_topk_rows_dev(&Hidden::plain(h), 1, k).swap_remove(0)
    }

    /// One prefill's `[d_model]` hidden row on the device, for admission's head.
    fn admit_hidden(&self, hidden: &[f32]) -> DeviceBuffer {
        assert_eq!(hidden.len(), self.model.cfg.d_model as usize, "qwen35moe admission: the hidden state is one row of {} floats, not {}", self.model.cfg.d_model, hidden.len());
        self.model.gpu.storage_init("qwen35moe.admit.hidden", hidden)
    }

    /// Turn the fused native decode kernels on or off for the steps that follow
    /// ([`Qwen35::set_decode_fusion`]). Recorded steps keep the kernels they were
    /// recorded with, so they are dropped.
    pub fn set_decode_fusion(&mut self, on: bool) {
        self.tapes.clear();
        self.model.set_decode_fusion(on);
    }

    /// Record decode steps and replay them, or build every step: the same answer,
    /// at a very different host cost. Drops what was recorded.
    pub fn set_decode_tapes(&mut self, on: bool) {
        self.tapes.clear();
        self.decode_tapes = on && self.model.moe_device_routed();
    }

    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// Device bytes the recorded decode steps' shared temporaries hold: the widest recorded step's working set.
    pub fn tape_scratch_bytes(&self) -> u64 {
        self.tape_scratch.held_bytes()
    }

    /// The hard `prompt + max_new` cap for one sequence.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len as usize
    }

    /// The most sequences resident at once: the recurrent-state rows.
    pub fn max_batch(&self) -> usize {
        self.gdn.rows as usize
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

    /// Real, combined device footprint of the per-sequence state: the shared GQA
    /// pool (every block's `[block_size, kv_dim]` K and V planes, in the pool's
    /// KV tier, every GQA layer) PLUS the GDN state pool's own real cost (every
    /// row), as `PagedDecoder::kv_pool_bytes`'s contract asks. The same
    /// arithmetic as the free [`kv_pool_bytes`], which a planner calls before
    /// building anything.
    pub fn kv_pool_bytes(&self) -> u64 {
        kv_pool_bytes(&self.model.cfg, self.kv_tier, self.kv_pool_capacity_tokens(), self.gdn.rows)
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

    /// Release a finished/cancelled sequence: give its recurrent-state row back
    /// (if it was given one) and its KV blocks to the pool. Both must run, or the
    /// row or the blocks would leak.
    pub fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(row) = t.take_state_row() {
            self.gdn.give_back(row);
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
        // One `reserve` for the whole prompt: the blocks its tokens cover. Decode
        // takes the next as the sequence crosses into it.
        table.reserve(prompt.len() as u32, &mut self.alloc).expect("qwen35moe::serve::Engine: KV pool exhausted");
        let row = self.assign_state_row(table).unwrap_or_else(|e| panic!("{e}"));
        // The prompt runs over a private state, which then becomes this
        // sequence's pool row: the prefill kernels bind one sequence's buffers.
        let slot = self.fresh_slot();
        let d = self.model.cfg.d_model as usize;
        let mut hidden = None;
        let mut pos = 0u32;
        for round in prompt.chunks(self.prefill_chunk as usize) {
            let h = self.model.run_prefill_chunk(round, pos, &self.prefill_caches(table.blocks(), &slot));
            pos += round.len() as u32;
            hidden = Some(h);
        }
        let hidden = self.model.gpu.read(&hidden.expect("a non-empty prompt has at least one round"), d);
        self.gdn.store(&self.model.gpu, row, &slot);
        hidden
    }

    /// Bring `table` to `position` cached tokens WITHOUT computing them: the
    /// KV blocks are reserved and a zeroed Gated-DeltaNet row is given, so the
    /// next decode step does the work (and moves the bytes) of a step at that
    /// position over whatever the cache rows hold. This is how a long-context
    /// decode step is priced when prefilling the context for real would take
    /// hours (`brain perf run longctx` says so in its artifact); the logits it
    /// produces mean nothing. A table already at `position` is left alone, one
    /// shorter is extended and one longer truncated.
    pub fn admit_synthetic(&mut self, table: &mut BlockTable, position: u32) -> Result<(), String> {
        if position == 0 || position >= self.max_seq_len {
            return Err(format!("a synthetic context needs 1 <= position < {} (got {position})", self.max_seq_len));
        }
        if table.is_empty() {
            table.reserve(position, &mut self.alloc)?;
            let row = self.assign_state_row(table)?;
            let zero = self.fresh_slot();
            self.gdn.store(&self.model.gpu, row, &zero);
        } else if table.len() < position {
            table.reserve(position - table.len(), &mut self.alloc)?;
        } else if table.len() > position {
            table.truncate(position, &mut self.alloc);
        }
        Ok(())
    }

    /// The per-kernel profile of `passes` batched decode steps of `batch`
    /// sequences at a synthetic context of `position` tokens
    /// ([`Engine::admit_synthetic`]): the production path timed on the wall,
    /// then the same passes with per-dispatch timestamps armed. Each sequence
    /// advances one position per pass, so the context must leave room for `2 *
    /// passes + 1` more tokens.
    pub fn profile_decode_at(&mut self, batch: u32, position: u32, passes: u32) -> Result<KernelProfile, String> {
        if batch == 0 || passes == 0 {
            return Err(format!("profile_decode_at needs a batch and a pass count (got {batch} x {passes})"));
        }
        if position as u64 + 2 * passes as u64 + 1 >= self.max_seq_len as u64 {
            return Err(format!("profiling {passes} passes from position {position} runs past this engine's capacity {}", self.max_seq_len));
        }
        let mut tables: Vec<BlockTable> = (0..batch).map(|_| BlockTable::new()).collect();
        let outcome = (|| {
            for t in tables.iter_mut() {
                self.admit_synthetic(t, position)?;
            }
            let inputs = vec![1u32; batch as usize];
            let step = |e: &mut Engine, tables: &mut Vec<BlockTable>| {
                let mut rows: Vec<&mut BlockTable> = tables.iter_mut().collect();
                e.forward_batched_greedy(&mut rows, &inputs);
            };
            // One unrecorded pass first: pipeline compilation is not decode time
            // (and, on the tape path, it is the pass that records the tape).
            step(self, &mut tables);
            self.poll_wait();
            let mut profile = self.profile_regions(passes, |e| step(e, &mut tables));
            // A recorded step is priced as it runs - launched whole, back to
            // back - not dispatch by dispatch: per-dispatch timing of eager
            // launches carries the host's launch latency, which the tape removes.
            if self.decode_tapes {
                if let Some((_, tape, _)) = self.tapes.iter().rev().find(|(k, _, _)| k.rows == batch && k.head == HeadKey::Greedy) {
                    if let Some(rows) = self.model.gpu.profile_tape(tape.tape(), passes) {
                        profile.rows = rows.into_iter().map(|(n, ms, calls)| (n, ms * passes as f64, calls * passes as u64)).collect();
                    }
                }
            }
            Ok(profile)
        })();
        for mut t in tables {
            self.release_table(&mut t);
        }
        outcome
    }

    /// The per-kernel profile of `passes` cold prefills of `tokens` tokens each
    /// (rounds of [`Engine::max_prefill_tokens`]), same two regions.
    pub fn profile_prefill(&mut self, tokens: u32, passes: u32) -> Result<KernelProfile, String> {
        if tokens == 0 || tokens > self.max_seq_len || passes == 0 {
            return Err(format!("profile_prefill needs 1 <= tokens <= {} and a pass count (got {tokens} x {passes})", self.max_seq_len));
        }
        let vocab = self.vocab() as u64;
        let prompt: Vec<u32> = (0..tokens as u64).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let once = |e: &mut Engine| {
            let mut t = BlockTable::new();
            e.prefill(&mut t, &prompt);
            e.release_table(&mut t);
        };
        once(self);
        self.poll_wait();
        Ok(self.profile_regions(passes, once))
    }

    /// The two regions every profile shares: `passes` production-path passes
    /// timed on the wall, then `passes` more with kernel timing armed.
    fn profile_regions(&mut self, passes: u32, mut pass: impl FnMut(&mut Engine)) -> KernelProfile {
        let sched0 = thread_sched_ns();
        let ops0 = self.model.gpu.stats();
        let t0 = std::time::Instant::now();
        let mut each = Vec::with_capacity(passes as usize);
        for _ in 0..passes {
            let t = std::time::Instant::now();
            pass(self);
            self.poll_wait();
            each.push(t.elapsed().as_secs_f64());
        }
        let wall_s = t0.elapsed().as_secs_f64();
        let sched1 = thread_sched_ns();
        let host_ops_per_pass = match (ops0, self.model.gpu.stats()) {
            (Some(a), Some(b)) => {
                let per = |x: u64, y: u64| (y - x) as f64 / passes as f64;
                Some((per(a.submits, b.submits), per(a.dispatches, b.dispatches), per(a.writes, b.writes), per(a.uniform_allocs, b.uniform_allocs)))
            }
            _ => None,
        };
        let (host_run_s, host_wait_s) = match (sched0, sched1) {
            (Some(a), Some(b)) => ((b.0 - a.0) as f64 * 1e-9 / passes as f64, (b.1 - a.1) as f64 * 1e-9 / passes as f64),
            _ => (0.0, 0.0),
        };
        each.sort_by(|a, b| a.total_cmp(b));
        let (best_pass_s, median_pass_s) = (each[0], each[each.len() / 2]);

        // A backend that cannot time kernels reports no rows rather than zeros.
        let timed = self.model.gpu.set_kernel_timing(true);
        self.model.gpu.reset_kernel_times();
        let t1 = std::time::Instant::now();
        for _ in 0..passes {
            pass(self);
        }
        self.poll_wait();
        let timed_wall_s = t1.elapsed().as_secs_f64();
        let mut rows: Vec<(String, f64, u64)> = if timed { self.model.gpu.kernel_times().unwrap_or_default() } else { Vec::new() };
        self.model.gpu.set_kernel_timing(false);
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        KernelProfile { passes, wall_s, best_pass_s, median_pass_s, host_run_s, host_wait_s, host_ops_per_pass, timed_wall_s, rows }
    }

    /// One greedy decode step for the WHOLE batch in one set of dispatches, the
    /// head on the device: only the `bsz` winning token ids are read back.
    pub fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_greedy: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        match self.decode(tables, inputs, DecodeHead::Greedy) {
            DecodeOut::Greedy(ids) => ids,
            _ => unreachable!("a greedy decode returns greedy picks"),
        }
    }

    /// One batched decode step returning each row's whole `[vocab]` logits: the step of
    /// [`Engine::forward_batched_topk`] with the rows read back instead of their best `k`.
    pub fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_logit_rows: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        match self.decode(tables, inputs, DecodeHead::Rows) {
            DecodeOut::Rows(rows) => rows,
            _ => unreachable!("a logit-row decode returns rows"),
        }
    }

    /// [`Engine::forward_batched_logit_rows`] with each keyed row drawn where the logits are
    /// (`model::serve::PagedDecoder::forward_batched_drawn`): only the drawn tokens and the rows whose draw is `None`
    /// are read back.
    pub fn forward_batched_drawn(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], draws: &[Option<model::serve::RowDraw>]) -> Vec<model::serve::Drawn> {
        assert_eq!(tables.len(), inputs.len(), "forward_batched_drawn: tables/inputs length mismatch");
        assert_eq!(draws.len(), inputs.len(), "forward_batched_drawn: draws/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        match self.decode(tables, inputs, DecodeHead::Drawn(draws)) {
            DecodeOut::Drawn(drawn) => drawn,
            _ => unreachable!("a drawn decode returns draws"),
        }
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
        match self.decode(tables, inputs, DecodeHead::TopK(k as u32)) {
            DecodeOut::TopK(c) => c,
            _ => unreachable!("a top-k decode returns candidates"),
        }
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        Engine::max_prefill_tokens(self)
    }
    fn max_batch(&self) -> Option<usize> {
        Some(Engine::max_batch(self))
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
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        Engine::forward_batched_logit_rows(self, tables, inputs)
    }
    fn forward_batched_drawn(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], draws: &[Option<model::serve::RowDraw>]) -> Vec<model::serve::Drawn> {
        Engine::forward_batched_drawn(self, tables, inputs, draws)
    }
}

/// `model::serve::Scheduler<Engine>` -- the continuous-batching scheduler
/// specialised to this engine, the same `Scheduler` type alias convention
/// `qwen3::serve::Scheduler` uses.
pub type Scheduler = model::serve::Scheduler<Engine>;
