// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.8-27B served INT8 and GPU-RESIDENT across as many cards as its real
//! per-layer bytes need, loaded **directly from a released GGUF** (Q8_0,
//! Q4_K_M, ...) - no fp32 intermediate file anywhere on the path.
//!
//! That last clause is the whole reason this module exists next to
//! [`crate::serve`]'s single-GPU `Engine`. `Engine` loads a brain-native
//! `.safetensors` through `checkpoint::load`, which means the deployment
//! story for a real checkpoint is "first convert 29 GB of Q8_0 into ~108 GB
//! of fp32 on disk, then load that" - a conversion most boxes that can RUN
//! this model still cannot STORE. Here the released `.gguf` IS the load
//! format: `checkpoint::gguf::MmapGguf` is a `checkpoint::TensorSource` that
//! dequantizes one tensor at a time out of the mapping, and
//! [`Qwen35::new_i8_shard`] consumes exactly that trait, re-quantizing each
//! leaf to brain's own group-wise INT8 as it uploads. Peak host use is one
//! tensor, never one model.
//!
//! # What this composes (and deliberately does not reimplement)
//!
//! * **Placement** - `model::shard::plan_fewest_devices`, the capacity-aware
//!   exact-DP contiguous layer partitioner. Nothing here knows how many cards
//!   the box has or whether they match: it supplies real per-layer byte costs
//!   ([`layer_cost`]) and takes the plan it is given. One card, two P40s, or a
//!   mixed 24/8 GB pair all work, and a model that genuinely does not fit is
//!   reported as unplaceable rather than OOMing partway through a multi-minute
//!   load.
//! * **Name mapping** - `checkpoint::remap::RemapSource` over a
//!   [`Fetch::Whole`] plan built from [`crate::gguf_import::classify`], the
//!   SAME llama.cpp-name classifier the offline `brain import` converter
//!   drives. There is no second name table here, and no dequantize/copy step
//!   of this module's own: `RemapSource` streams straight through to the
//!   inner `MmapGguf`.
//! * **Stage construction** - [`Qwen35::new_i8_shard`], one instance per card,
//!   each holding only its own `shard.start..shard.end` layers.
//! * **Decode** - `Qwen35::run_decode_step` per stage per token, with the
//!   boundary residual handed from card to card device to device
//!   (`Gpu::copy_to`: over the card link when the pair has one, `d_model`
//!   floats, 20 KiB at this shape). Gated at tiny scale by
//!   `crate::model`'s own `two_shard_int8_decode_matches_the_whole_shard_model`,
//!   which proves the two-stage composition is bit-equal to the whole-shard
//!   model.
//! * **Prefill** - `Qwen35::run_prefill_chunk_stage` per stage per ROUND, the same
//!   seam widened to a whole round's `[n, d_model]` boundary block (5.2 MB at
//!   this resident's `n = 256`). Gated at tiny scale by `crate::model`'s own
//!   `two_shard_chunked_prefill_matches_token_by_token_replay`, which proves a
//!   chunked replay leaves both stages in exactly the decode state the
//!   per-token replay leaves.
//! * **Head epilogue** - `crate::stream::head_logits_on`, the same final-norm
//!   + int8-`lm_head` projection the streaming real-weight path already uses.
//!
//! # The endpoints do not live in a shard, and cannot
//!
//! Both `[vocab, d_model]` tables are 5_085_593_600 bytes as fp32 at this
//! shape. On a 24 GB Tesla P40 that is over `max_buffer_size` (~4.09 GiB) AND
// perf-number: arithmetic: the fp32 table size above over the 2047 MiB binding limit
//! 2.4x `max_storage_buffer_binding_size` (2047 MiB, which wgpu clamps to
//! `i32::MAX` on every backend), so an fp32 endpoint is not a thing that can
//! be allocated OR bound here - it is not a size question. Measured, not
//! assumed: the first real load of this resident died in
//! `paramstore::upload` with "needs a single 5085593600-byte buffer but this
//! device's queried max_buffer_size is 4292870144 bytes".
//!
//! So every stage is built with `embed: false, head: false` and this module
//! owns both ends:
//!
//! * the **embedding** is never uploaded as fp32 and never materialized on the host - decode needs
//!   one ROW per token, and `checkpoint::gguf::MmapGguf::tensor_range`
//!   dequantizes exactly the quant blocks that row touches, straight out of
//!   the mapping. It enters stage 0 through `run_decode_step`'s own
//!   `input_override` seam, the same seam that carries the residual between
//!   cards, so no new path exists for it. (A plain 16-bit table on a card with room for it is the exception:
//!   the first stage holds it as stored and gathers the step's tokens itself, [`ServingShape::embed`].)
//! * the **head** is quantized to INT8 (1.42 GB, inside both limits) by
//!   `model::int8::upload_quantized` straight from the mapping, and lives on
//!   the last stage's card.
//!
//! Both choices follow `crate::stream`'s real-weight path rather than
//! inventing anything: that module reached the identical conclusion about the
//! same two tensors on the same hardware.
//!
//! # Scope, deliberately
//!
//! * **No MTP.** The GGUF's `blk.64.*` + `nextn.*` MTP block is not imported
//!   and `cfg.mtp` is forced `false`: `Qwen35::new_impl_on` asserts that MTP
//!   implies a whole shard (the head needs `res[n_layers]` and the shared
//!   `lm_head`), which a multi-card split is not by construction.
//!   Self-speculative decode stays a single-GPU concern.
//! * **No flash-attention prefill.** A prefill round's GQA scratch is
//!   `[chunk, n_heads, pos+chunk]` twice over, the one cost that does not
//!   shrink with the dispatch count, and it is why [`MAX_PREFILL_TOKENS`] is a
//!   bounded round size rather than "the whole prompt at once". Growing this
//!   resident into genuinely long contexts needs a flash-attention prefill
//!   kernel, not a bigger constant - `paged_flash_prefill.wgsl` is that
//!   kernel, but its tiles cap `head_dim` at 128 and this model's is 256.
//!
//!   This bullet used to read "**Per-token prefill.** The prompt is replayed
//!   one token at a time through the same decode path... [that] needs the
//!   seam widened to a whole round's `[n, d_model]` boundary residual...
//!   Worth doing and not done here." It is done: `Qwen35::
//!   run_prefill_chunk_stage` is that widened seam (`run_prefill_chunk`, the
//!   whole-model M25 primitive `crate::serve::Engine` drives, is now a thin
//!   wrapper over it), and [`Qwen35GgufInstance::stack_prefill_chunk`] drives
// perf-number: hardware count, plus measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   it round by round across every card. Measured on 2x Tesla P40 at a real
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   1731-token prompt: 262.8 s (6.6 tok/s) before, 26.5 s (65.4 tok/s)
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   after - 9.9x. What it did NOT need, contrary to that note, was `t`-sized
//!   per-stage buffers on every card: every buffer a round touches is
//!   allocated per call from `n`, so the stages stay built at `b = t = 1`.
//!   What it DID need, and that note did not predict, was bounding how many
//!   layers' worth of those per-call buffers can be in flight at once - see
//!   `Qwen35::run_prefill_chunk_stage`'s `DRAIN_EVERY_N_LAYERS`.
//!   This bullet used to also carry its own sub-note, "Per-token prefill at
//!   the Q4 tier, and only there": a round was only worth issuing if the
//!   tier's `m > 1` kernel was a real tiled GEMM, `Q4`'s was not
//!   (`matmul_q4_dyn.wgsl`, naive by its own header), and on real weights
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   that made a chunked replay 7.3x SLOWER than the per-token one, so
//!   [`Qwen35GgufInstance::replay_prompt`] picked the tape by tier. Fixed by
//!   wiring `matmul_q4_dyn_reg` (already proven bit-identical, kernel-
//!   performance ledger M5.5) into `model::ops::Ops::bind` - `Qwen35::
//!   chunked_prefill_is_profitable` no longer excludes Q4, and the real
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   1555-token measurement moved from 152.1 s (10.2 tok/s) per-token /
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
//!   1108.3 s (1.4 tok/s) chunked to 22.9 s (68.0 tok/s) chunked, a ~6.6x
//!   win. See [`MAX_PREFILL_TOKENS`] for the full before/after.
//! * **Batched PREFILL.** Decode is genuinely multi-sequence as of
//!   [`Qwen35GgufInstance::generate_batch`] - one dispatch set per step
//!   carrying every live request's row - but a prompt is still replayed into
//!   its own slot alone. That is the right split, not an omission: a prefill
//!   round's rows already fill the machine, so a batch has nothing to
//!   amortize there, while a decode step is dominated by weight reads the
//!   batch shares.
//! * **Continuous batching.** [`Qwen35GgufInstance::generate_batch`] runs a
//!   FIXED group: a finished row is frozen rather than replaced, so the group
//!   costs its longest member. Admitting a waiting request into a freed slot
//!   is `crate::serve::Scheduler`'s job.
//! * **Text only.** `crate::vl`'s vision front-end is not spliced in here.

mod background;
mod engine;
mod mtp;
mod pace;
mod speculate;
pub use speculate::{SpecCost, SpecServing, SpeculativeMode};
pub use engine::{checkpoint_len, PrefillPolicy, Qwen35GgufEngine, SpecTimes};

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::OnceLock;

use capability::{ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType};
use checkpoint::gguf::MmapGguf;
use checkpoint::remap::{Fetch, RemapSource};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use data::rng::Rng;
use gguf::import::Mapped;
use gguf::GdnFixSource;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::adapter::bank::AdapterBankRequest;
use crate::decode_tape::{DecodePlan, DecodePlans, HeadKind, PlanKey, PlanOut};
use model::dispatch::LORA_SLOT_NONE;
use model::kv_tier::{KvLayer, KvTier};
use model::ops::TierPolicy;
use model::shard::{LayerBytes, Shard};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use crate::adapter_serving as adapters;
use crate::config::{LayerType, Qwen35Config};
use crate::stream::HeadRows;
use crate::resident_checkpoint::ResidentCheckpoint;
pub(crate) use mtp::{checkpoint_has_mtp, MTP_PREFIX};
use crate::model::{BatchDecodeCaches, BatchSeq, DecodeCaches, GdnPoolStore, KvWindow, PooledCaches, PrefillSeg, Qwen35, SpecDecodeStats, StageRows};

/// Catalog id. Names the real upstream release this resident loads, exactly
/// (`https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`, the `Q8_0` file) -
/// per AGENTS.md every served model is `<vendor>/<repo>[-<QUANT>]` matching
/// its upstream URL, and `brain/` is reserved for built-ins with no upstream
/// provenance (which is what `crate::caps::MODEL`, the fp32 brain-checkpoint
/// path, is).
pub const MODEL: &str = "unsloth/Qwen3.8-27B-Q8_0";

/// How many prompt tokens [`Qwen35GgufInstance::generate`] pushes through the
/// whole pipeline per round.
///
/// The upside is `crate::serve`'s and applies here twice over: a round
/// amortises each stage's per-layer dispatch overhead over `n` rows AND
/// replaces `n` per-token stage hand-offs (there are `n_stages` of those per
/// token) with one.
///
/// The same 256 `crate::serve::Engine` uses - but RE-MEASURED on this model
/// rather than inherited, because the first honest attempt at it did not run
// perf-number: hardware count (2x P40), not a rate
/// at all. Measured on 2x Tesla P40, real Q4_K_M checkpoint at the INT8 tier,
/// 1731-token prompt; `prefill_seconds` from the instance's own metrics, peak
/// card occupancy sampled from `nvidia-smi`:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | (per token) | 262.8 s | 6.6 | - |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 64 | 36.5 s | 47.4 | 16.2 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 128 | 26.6 s | 65.0 | 17.8 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 192 | 26.4 s | 65.4 | 20.3 GiB |
/// | 256 | OOM | - | (> 24 GiB) |
///
/// A round's transients are not just its GQA scratch: EVERY intermediate a
/// layer allocates stays alive until something drains the queue, and a round
/// used to drain only at its terminal readback - so a stage held all ~32 of
/// its layers' worth at once, at the real 27B widths (`d_model` 5120,
/// `intermediate_size` 17408, `conv_dim` 10240). That is a DEPTH problem, not
/// a chunk-size problem, and capping the chunk is the wrong lever for it: a
/// single-card plan (the Q4 tier fits one) is 64 layers deep and overruns the
/// same budget at half the chunk.
///
/// So the fix went where the defect was - `Qwen35::run_prefill_chunk_stage`'s
/// `DRAIN_EVERY_N_LAYERS` - and the table became:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 256 | 26.5 s | 65.4 | 15.3 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 512 | 27.6 s | 62.8 | 16.6 GiB |
///
/// 256 is now both the fastest measured round size AND cheaper in memory than
/// 64 was before the drain, and 512 is slower - so this is a real optimum
/// rather than a memory-driven compromise.
///
/// One thing that is NOT a consideration here and is in `crate::serve`: the
/// per-stage buffers a round needs are all allocated per call from `n`
/// (`Qwen35::run_prefill_chunk_stage` sizes every one of them), so a stage
/// built at `b = t = 1` carries no FIXED cost for this at all.
///
/// **This constant now applies to a Q4 build too** - see
/// [`Qwen35GgufInstance::replay_prompt`] and
/// `Qwen35::chunked_prefill_is_profitable`. It did not always: the Q4 tier's
/// `m > 1` GEMM (`matmul_q4_dyn.wgsl`) used to resolve to the naive
/// one-thread-per-output kernel its own header describes, and a round
/// through it cost far MORE than the same tokens through `matmul_q4_gemv`'s
/// coalesced `m = 1` path - measured on the real checkpoint, uniform Q4 on
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// ONE P40, 1555-token prompt: 152.1 s (10.2 tok/s) per token, 1108.3 s
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// (1.4 tok/s) in rounds of 256, a 7.3x REGRESSION, where the same change on
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// the INT8 two-card stack was a 9.9x win. Same host code, same round size;
/// the only difference was which GEMM the weight tier bound.
///
/// Fixed by wiring `matmul_q4_dyn_reg` (128x128 register-tiled, already
/// proven bit-identical to `matmul_q4_dyn`, kernel-performance ledger M5.5:
// perf-number: recorded in kernel-performance ledger M5.5 (matmul_q4_dyn_reg vs matmul_q4_dyn)
/// 2.02x at `m = 32` rising to 12.56x at `m = 2048`) into `model::ops::
/// Ops::bind`. Re-measured on the same real checkpoint, same one P40, the
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// same 1555-token prompt, same round size: **22.9 s (68.0 tok/s)** - a
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// ~6.6x win over the per-token baseline, and slightly FASTER than the
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// INT8 stack's own chunked result (64.6-65.4 tok/s) despite streaming
/// fewer bytes per round, not more.
const MAX_PREFILL_TOKENS: u32 = 256;

/// The round size on a stack whose every stage runs the chunk round on tensor
/// cores (`gpu_core::provider::cuda::tensor_core_kernels_enabled`).
///
/// 256 is the P40-era optimum above: there the round is arithmetic-bound and
/// a bigger one only costs memory. On tensor cores the arithmetic is cheap
/// enough that what a round pays per DISPATCH - every step is a launch plus a
/// parameter upload, a few thousand of them per round whatever its width - is
/// a real fraction of it, so a wider round amortises that. Measured on a
/// GH200 (`qwen35_gguf_prefill_profile`, production rounds at depth 256, the
/// fastest of repeated rounds on a shared card):
///
/// | rows | ms/round | tok/s |
/// |------|----------|-------|
/// | 256  | 188      | 1363  |
/// | 512  | 309      | 1659  |
/// | 768  | 434      | 1770  |
/// | 1024 | 610      | 1679  |
/// | 2048 | 1028     | 1993  |
///
/// Re-measured with the prefill GEMM chosen per shape class (the Hopper
/// warpgroup candidate wins from about 512 rows): the 2048-row round is
/// faster than 1024 on `qwen35_gguf_prefill_profile` and on `brain perf run
/// longctx` prompts of 1K/8K/32K, and 4096 rows buy only a little more for
/// twice the scratch (the cuda.md ledger has the numbers). 2048 rows of
/// scratch is still about a gigabyte per stage, which every device that can
/// hold this model's weights has to spare.
const TENSOR_CORE_PREFILL_TOKENS: u32 = 2048;

/// Hidden rows one head readback of [`Qwen35GgufInstance::teacher_forced_scores`]
/// projects at once. The readback is `[rows, vocab]` on the host, which at this
/// model's vocabulary is 993 KB per row: eight rows keep it under 8 MB while
/// amortising the head GEMM's weight stream across them.
const SCORE_HEAD_ROWS: usize = 8;

// ------------------------------------------------------------ byte accounting
//
// The per-layer decode-state formula (`Qwen35Config::layer_decode_state_bytes`)
// now lives on the config itself - shared with `Qwen35Config::
// infer_footprint_bytes`'s single-device pre-flight estimate, see that
// method's own doc for why the two must not each keep a private copy.

/// Device bytes the INT8 `lm_head` occupies: `model::ops::Weight::I8`'s
/// `[n, k/4]` packed words plus its `[n, k/GROUP]` f32 scales, both 4 bytes
/// an element. 1.42 GB at the real shape - against 5.09 GB for the fp32 table
/// this replaces, which is not merely large but IMPOSSIBLE on a 24 GB P40
/// (past `max_buffer_size`, and 2.4x the 2047 MiB storage-binding limit).
/// The embedding table is held on the first stage's card when it is at most this fraction (the reciprocal) of the
/// smallest budgeted card: a 2.5 GB table on a 96 GB card, not on a 24 GB one.
const EMBED_RESIDENT_MAX_SHARE: u64 = 16;

fn head_bytes(cfg: &Qwen35Config, tier: Dtype) -> u64 {
    model::ops::Weight::bytes_for(tier, cfg.vocab as usize, cfg.d_model as usize)
}

/// The byte-exact per-stage cost model this resident hands to
/// `model::shard::plan_fewest_devices`, at `cap` decode positions.
///
/// * `per_layer` - the layer's weights at `tier` (`Qwen35Config::
///   layer_weight_bytes`, itself gated against what `model::ops::Weight::
///   upload` really places) plus that layer's own decode state, its KV window
///   stored in `kv` (`Qwen35Config::layer_decode_state_bytes`).
/// * `embed` - **zero** unless [`ServingShape::embed`] names a tier. The `[vocab, d_model]` embedding is
///   not uploaded as fp32: decode only ever needs one ROW per token, so [`Qwen35GgufInstance`] reads
///   it straight out of the mapping ([`MmapGguf::tensor_range`], which decodes
///   only the quant blocks that row touches). 5.09 GB that exists neither in
///   VRAM nor in host RAM. A checkpoint that stores the table as plain 16-bit floats on a card with room for it
///   ([`Qwen35GgufResident::embed_tier`]) holds it on the first stage at that tier instead, so the step's
///   tokens are the only input that crosses the link, and the charge is the table's bytes.
/// * `head` - `norm.weight` plus the INT8 `lm_head` ([`head_bytes`]), both
///   held by this resident on the last stage's card rather than by the shard
///   (the shard is built `head: false`; see [`Qwen35GgufInstance::head`]).
///
/// Charging the endpoints truthfully matters more here than anywhere else: at
/// this vocab a mis-charged endpoint is several GB, i.e. the difference
/// between a plan that fits and a card that OOMs mid-load.
pub fn layer_cost(cfg: &Qwen35Config, cap: u32, tier: &TierPolicy, kv: KvTier, max_batch: u32) -> LayerBytes {
    serving_cost(cfg, &ServingShape::new(cap, tier.clone(), kv, max_batch))
}

/// Everything a serving configuration holds on the cards besides the model's
/// own weights - the one request the cost model ([`serving_cost`]), the sizing
/// check ([`sizing_refusal`]) and the automatic context ([`auto_cap`]) answer.
#[derive(Clone, Debug, PartialEq)]
pub struct ServingShape {
    /// Per-sequence `prompt + max_new` capacity.
    pub cap: u32,
    /// The decoder body's weight tier.
    pub tier: TierPolicy,
    /// How the GQA layers' K/V are stored.
    pub kv: KvTier,
    /// Sequences decoding at once.
    pub max_batch: u32,
    /// Cached prompts kept for prefix reuse.
    pub prefix_slots: u32,
    /// Rows of the shared KV pool of the paged layout ([`KvLayout::Paged`]);
    /// 0 for one `cap`-row window per slot.
    pub kv_pool_rows: u32,
    /// What speculative decoding keeps on the cards.
    pub spec: SpecCost,
    /// Rows of the largest prefill forward a stage runs: the transient
    /// activations of one ([`prefill_workspace_bytes`]) are memory the cards
    /// must keep free beside everything this shape places.
    pub prefill_rows: u32,
    /// What an adapter bank holds on the cards, when one is served: part of the
    /// shape so the automatic context and pool size leave room for it, rather
    /// than filling the cards and refusing the bank afterwards.
    pub adapter_bank: Option<BankBytes>,
    /// The tier the lm_head is held at ([`ResidentCheckpoint::head_tier`]): int8 for a GGUF, as stored for an HF release.
    pub head: Dtype,
    /// The tier the first stage holds the embedding table at, or `None` when it stays in the mapping and the host
    /// decodes the rows of every step ([`ResidentCheckpoint::embed_tier`]). On the card the step's token ids are
    /// all that crosses the link, and a decode step or mixed forward needs no host work in front of its first kernel.
    pub embed: Option<Dtype>,
}

/// An adapter bank's device bytes: beside each layer it corrects, and the row
/// scratch, charged once to the head.
#[derive(Clone, Debug, PartialEq)]
pub struct BankBytes {
    pub per_layer: Vec<u64>,
    pub head: u64,
}

impl ServingShape {
    /// `max_batch` sequences of `cap` tokens, no prompt cache, no speculation.
    pub fn new(cap: u32, tier: TierPolicy, kv: KvTier, max_batch: u32) -> ServingShape {
        ServingShape { cap, tier, kv, max_batch, prefix_slots: 0, kv_pool_rows: 0, spec: SpecCost::NONE, prefill_rows: TENSOR_CORE_PREFILL_TOKENS, adapter_bank: None, head: Dtype::I8, embed: None }
    }

    /// The same shape for a stack whose prefill forward may carry `rows` rows (the
    /// batched-prefill setting): never fewer than the widest single-sequence round,
    /// which a stack with tensor cores runs whatever the setting.
    pub fn with_prefill_rows(&self, rows: u32) -> ServingShape {
        ServingShape { prefill_rows: rows.max(TENSOR_CORE_PREFILL_TOKENS), ..self.clone() }
    }

    /// Device bytes every stage must keep free beside what this shape places: the
    /// activations of a prefill round and of a decode (or verify) step, which run
    /// at the same time on separate streams. Decode rows are the larger of the
    /// sequences and the verify rows speculation batches.
    pub fn stage_workspace(&self, cfg: &Qwen35Config) -> u64 {
        let rows = self.max_batch.max(self.spec.scratch_rows).max(1);
        prefill_workspace_bytes(cfg, self.prefill_rows) + decode_scratch_bytes(cfg, rows, self.cap)
    }

    /// The same shape at another context.
    pub fn at(&self, cap: u32) -> ServingShape {
        ServingShape { cap, ..self.clone() }
    }

    /// Sequence slots the one-window layout gives every GQA layer: running
    /// sequences and cached prompts alike.
    fn window_slots(&self) -> u32 {
        self.max_batch.max(1) + if self.kv_pool_rows > 0 { 0 } else { self.prefix_slots }
    }
}

/// Device bytes one layer of type `ty` holds for decode state: the GQA layers'
/// K/V (a `cap`-row window per slot, or the shared pool of `kv_pool_rows` rows
/// when that is non-zero) and the GDN layers' recurrent state (per running
/// sequence, plus one snapshot per cached prompt, plus the speculative verify
/// scratch). The one place the placement planner and the engine's pool
/// accounting get the figure from.
pub(crate) fn layer_state_bytes(cfg: &Qwen35Config, ty: LayerType, shape: &ServingShape) -> u64 {
    let slots = shape.window_slots() as u64;
    match ty {
        LayerType::Full if shape.kv_pool_rows > 0 => 2 * shape.kv.plane_bytes(shape.kv_pool_rows as u64, cfg.kv_dim() as u64, cfg.head_dim as u64),
        LayerType::Full => slots * cfg.layer_decode_state_bytes(ty, shape.cap, shape.kv),
        LayerType::Linear => {
            (slots + shape.prefix_slots as u64) * cfg.layer_decode_state_bytes(ty, shape.cap, shape.kv)
                + shape.spec.scratch_rows as u64 * (cfg.gdn_state_len() + cfg.gdn_hist_len()) * 4
        }
    }
}

/// The per-stage cost the planner places for `shape` ([`layer_cost`]'s full
/// form): every layer's weights and decode state, and the head with, when
/// speculation loads it, the MTP head on the same card.
///
/// A cached prompt of the one-window layout holds a whole sequence slot (the KV
/// window of the prompt and the GDN pool row that goes with the block) plus a
/// snapshot row of every GDN layer's state. In the paged layout the GQA layers
/// hold one pool of `kv_pool_rows` rows shared by all sequences and cached
/// prompts, and a cached prompt keeps only blocks and its snapshots.
///
/// Speculation is charged where it lives: the verify scratch beside each
/// recurrent layer, on that layer's card, and the MTP head (one full-attention
/// layer, its `eh_proj`, and a KV window per slot) beside the head.
pub fn serving_cost(cfg: &Qwen35Config, shape: &ServingShape) -> LayerBytes {
    // Decode state is per SEQUENCE; weights are shared by the whole batch and
    // counted once, which is the entire point of batching.
    let bank = shape.adapter_bank.as_ref();
    let per_layer = cfg
        .layer_types()
        .into_iter()
        .enumerate()
        .map(|(l, ty)| cfg.layer_weight_bytes_at(&format!("blocks.{l}."), ty, &shape.tier) + layer_state_bytes(cfg, ty, shape) + bank.and_then(|b| b.per_layer.get(l)).copied().unwrap_or(0))
        .collect();
    let mtp = if shape.spec.mtp_head {
        let d = cfg.d_model as u64;
        cfg.layer_weight_bytes_at(mtp::MTP_PREFIX, LayerType::Full, &shape.tier) + d * 2 * d * 2 + shape.window_slots() as u64 * cfg.layer_decode_state_bytes(LayerType::Full, shape.cap, shape.kv)
    } else {
        0
    };
    let embed = shape.embed.map_or(0, |tier| model::ops::Weight::bytes_for(tier, cfg.vocab as usize, cfg.d_model as usize));
    LayerBytes { per_layer, embed, head: head_bytes(cfg, shape.head) + cfg.d_model as u64 * 4 + mtp + bank.map_or(0, |b| b.head) }
}

/// Whether `max_batch` sequences of `cap` tokens make KV planes the kernels
/// can address: they index a plane with `u32` element offsets
/// ([`model::kv_tier::MAX_PLANE_ELEMS`]), so a pool past that would alias its
/// own start, however many bytes the cards could hold. A sizing that fails
/// this is planned as unplaceable.
pub fn kv_pool_addressable(cfg: &Qwen35Config, cap: u32, max_batch: u32) -> bool {
    KvTier::fits_addressing(max_batch as u64 * cap as u64, cfg.kv_dim() as u64)
}

/// [`kv_pool_addressable`] for a paged pool of `rows` KV rows.
fn kv_rows_addressable(cfg: &Qwen35Config, rows: u32) -> bool {
    KvTier::fits_addressing(rows as u64, cfg.kv_dim() as u64)
}

/// Why a serving configuration cannot be placed, in the operator's terms - or
/// `None` when it fits. `usable` is each budgeted device's usable bytes.
///
/// Weights are shared by the whole batch and counted once; the decode state
/// (KV windows and recurrent state) is per sequence, so `max_batch` and `cap`
/// are what a refusal tells the operator to lower, and a more compact KV tier
/// what it suggests instead.
pub fn sizing_refusal(cfg: &Qwen35Config, shape: &ServingShape, usable: &[u64]) -> Option<String> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let ServingShape { cap, kv, max_batch, prefix_slots, kv_pool_rows, .. } = *shape;
    if kv_pool_rows > 0 {
        if kv_pool_rows < cap {
            return Some(format!("a KV pool of {kv_pool_rows} rows cannot hold one sequence of {cap} tokens; raise the pool or lower --ctx"));
        }
        if !kv_rows_addressable(cfg, kv_pool_rows) {
            return Some(format!("a KV pool of {kv_pool_rows} rows of {} elements per layer is past the kernels' u32 element index; lower the pool", cfg.kv_dim()));
        }
    } else if !kv_pool_addressable(cfg, cap, max_batch + prefix_slots) {
        return Some(format!(
            "{max_batch} sequences and {prefix_slots} cached prompts of {cap} tokens is {} KV rows of {} elements per layer, past the kernels' u32 element index; lower --max-batch, the prefix-slot setting or --ctx",
            (max_batch + prefix_slots) as u64 * cap as u64,
            cfg.kv_dim()
        ));
    }
    // Every card also keeps its prefill round's activations free.
    let workspace = shape.stage_workspace(cfg);
    let caps: Vec<(usize, u64)> = usable.iter().map(|&u| u.saturating_sub(workspace)).enumerate().collect();
    if model::shard::plan_fewest_devices(&serving_cost(cfg, shape), &caps).is_some() {
        return None;
    }
    // Placement said no: the global planner states the arithmetic per category and finds the
    // smallest KV tier change (the only lever a serving configuration may pull for itself).
    let demands = serving_demands(cfg, shape);
    let total_usable: u64 = caps.iter().map(|&(_, c)| c).sum();
    let capacity = model::memplan::Capacity { device_bytes: total_usable, host_bytes: 0 };
    let (report, fits_with) = match model::memplan::plan(&demands, capacity, &model::memplan::Authorized::only(&[model::memplan::Lever::CompactKv])) {
        Ok(r) => (r, true),
        Err(model::memplan::PlanError::DoesNotFit { best, .. }) => (best, false),
        Err(e) => return Some(format!("serving sizing is inconsistent: {e}")),
    };
    // The aggregate says a tier fits; placement across the real cards has the last word.
    let suggestion = report.decisions.iter().rev().find(|d| d.category == model::memplan::Category::Kv).map(|d| d.to.clone()).filter(|to| {
        fits_with && KvTier::parse(to).is_ok_and(|t| model::shard::plan_fewest_devices(&serving_cost(cfg, &ServingShape { kv: t, ..shape.clone() }), &caps).is_some())
    });
    let bytes_of = |c: model::memplan::Category| {
        // The request's own numbers, not the post-decision ones.
        demands.iter().find(|d| d.category == c).map_or(0, |d| d.variants[0].device_bytes)
    };
    let (weights, kv_bytes, state) = (bytes_of(model::memplan::Category::Weights), bytes_of(model::memplan::Category::Kv), bytes_of(model::memplan::Category::DecodeState));
    let advice = match suggestion {
        Some(t) => format!("a more compact --kv-cache ({t}) would fit these cards"),
        None => "lower --max-batch or --ctx, or choose a more compact --kv-cache".to_string(),
    };
    Some(format!(
        "--max-batch {max_batch} x --ctx {cap} with a {} KV cache needs {:.1} GiB ({:.1} GiB weights + {:.1} GiB KV cache + {:.1} GiB recurrent state for {max_batch} sequences) but the {} budgeted GPU(s) offer {:.1} GiB after keeping {:.1} GiB free on each for prefill and decode activations; {advice}",
        kv.as_str(),
        gib(weights + kv_bytes + state),
        gib(weights),
        gib(kv_bytes),
        gib(state),
        usable.len(),
        gib(total_usable),
        gib(workspace)
    ))
}

/// The serving configuration as the global planner's categories: the stored weights (with the MTP
/// head when speculation loads it), the KV cache (offering each smaller tier), and the GDN recurrent
/// state with the verify scratch, each with the byte count [`serving_cost`] charges for it.
pub fn serving_demands(cfg: &Qwen35Config, shape: &ServingShape) -> Vec<model::memplan::Demand> {
    use model::memplan::{kv_demand_with, Category, Demand};
    let types = cfg.layer_types();
    let by_type = |want: LayerType, kv: KvTier| -> u64 { types.iter().filter(|&&t| t == want).map(|&t| layer_state_bytes(cfg, t, &ServingShape { kv, ..shape.clone() })).sum() };
    let cost = serving_cost(cfg, shape);
    let state: u64 = by_type(LayerType::Full, shape.kv) + by_type(LayerType::Linear, shape.kv);
    let weights = cost.total() - state;
    let kv_demand = kv_demand_with(shape.kv, &KvTier::ALL, |t| by_type(LayerType::Full, t)).expect("a KV demand over supported tiers is consistent");
    vec![Demand::fixed(Category::Weights, "as stored", weights), kv_demand, Demand::fixed(Category::DecodeState, "recurrent state", by_type(LayerType::Linear, shape.kv))]
}

/// `(index into devices, capacity)` for every GPU in `devices`, each less
/// `headroom` - the shape `model::shard::plan_fewest_devices` plans over. A
/// non-GPU device is skipped: this model has no CPU or NPU path.
fn gpu_capacities(devices: &[(Device, u64)], headroom: u64) -> Vec<(usize, u64)> {
    devices.iter().enumerate().filter(|(_, (d, _))| matches!(d, Device::Gpu(_))).map(|(i, &(_, cap))| (i, cap.saturating_sub(headroom))).collect()
}

/// Rows of one prefill round, on a stack whose stages all run it on tensor cores or not.
pub fn prefill_round_rows_for(tensor_cores: bool) -> u32 {
    if tensor_cores {
        TENSOR_CORE_PREFILL_TOKENS
    } else {
        MAX_PREFILL_TOKENS
    }
}

/// How many layers' worth of intermediate activations a prefill round holds at
/// once, as a ratio. A round's buffers are returned to the allocator only when
/// the device has finished with them, so the host, which queues layers ahead,
/// sees the buffers of several layers live together. Measured on a GH200 at the
/// real 27B shape (`tests/gguf_resident_real.rs`, 1024 rows): 2.09 layers;
/// charged as 5/2 so the bound holds with a margin.
const PREFILL_LIVE_LAYERS: (u64, u64) = (5, 2);

/// Device bytes one stage allocates, on top of its weights and decode state,
/// while it prefills a round of `rows` rows: the f32 activations of a layer
/// ([`PREFILL_LIVE_LAYERS`] of them at once). The placement must leave this
/// free on every card, or the first long prompt dies on an allocation after the
/// model loaded; it is linear in `rows` and independent of the context.
///
/// A layer's intermediates are the residual pair, its mixer's projections
/// (the larger of the recurrent and the attention mixer) and the three
/// feed-forward buffers.
pub fn prefill_workspace_bytes(cfg: &Qwen35Config, rows: u32) -> u64 {
    let d = cfg.d_model as u64;
    let recurrent = d + 3 * cfg.linear_conv_dim() as u64 + 3 * cfg.linear_value_dim() as u64 + d;
    let attention = d + cfg.q_proj_dim() as u64 + 2 * cfg.kv_dim() as u64 + 2 * cfg.q_dim() as u64 + d;
    let feed_forward = d + 3 * cfg.intermediate_size as u64 + d;
    let layer_words = 2 * d + recurrent.max(attention) + feed_forward;
    let (num, den) = PREFILL_LIVE_LAYERS;
    rows as u64 * layer_words * 4 * num / den
}

/// Device bytes one decode (or verify) step of `rows` rows allocates at a
/// context of `cap`: per row the attention's score and probability strips of
/// two layers in flight (`n_heads * cap` f32 each), which is what grows with the
/// context, and a fixed share for the step's other activations and the replay
/// arena it fills on first use. Measured on a GH200 at the real 27B shape
/// (`tests/gguf_resident_real.rs`, 8 rows, first step): 268 MiB at a context
/// of 8192 and 540 MiB at 100000; charged with a margin.
pub fn decode_scratch_bytes(cfg: &Qwen35Config, rows: u32, cap: u32) -> u64 {
    const FIXED_PER_ROW: u64 = 40 << 20;
    const STRIPS_IN_FLIGHT: u64 = 4;
    rows as u64 * (FIXED_PER_ROW + STRIPS_IN_FLIGHT * cfg.n_heads as u64 * cap as u64 * 4)
}

/// Granularity of [`auto_cap`]'s answer, so the advertised context is a round
/// number an operator can read in a log line rather than a byte-exact sweep
/// result.
const AUTO_CAP_GRANULE: u32 = 1024;

/// The longest per-sequence context `shape` can have on `devices` (its own
/// `cap` is ignored): the largest cap whose [`serving_cost`] the planner still
/// places, with the step workspace ([`ServingShape::stage_workspace`]) kept free on every card,
/// rounded down to [`AUTO_CAP_GRANULE`]. Nothing but memory (and the kernels'
/// `u32` KV index, [`kv_pool_addressable`]) bounds it - not the checkpoint's
/// trained context, which a longer cap extends with YaRN.
///
/// `None` when even a one-token context does not fit, i.e. the weights alone
/// exceed the cards.
pub fn auto_cap(cfg: &Qwen35Config, devices: &[(Device, u64)], shape: &ServingShape) -> Option<u32> {
    let fits = |cap: u32| {
        let at = shape.at(cap);
        let caps = gpu_capacities(devices, at.stage_workspace(cfg));
        kv_pool_addressable(cfg, cap, at.max_batch + at.prefix_slots) && model::shard::plan_fewest_devices(&serving_cost(cfg, &at), &caps).is_some()
    };
    if gpu_capacities(devices, 0).is_empty() {
        return None;
    }
    largest_fitting(fits).map(|fit| if fit >= AUTO_CAP_GRANULE { fit / AUTO_CAP_GRANULE * AUTO_CAP_GRANULE } else { fit })
}

/// The largest `n` for which `fits(n)` holds, or `None` when not even 1 does.
/// Feasibility is monotone in `n` (cost only grows), so the boundary is bracketed
/// by doubling and bisected. An addressing bound inside `fits` makes it fail
/// before `u32` overflows, so the doubling ends.
fn largest_fitting(fits: impl Fn(u32) -> bool) -> Option<u32> {
    if !fits(1) {
        return None;
    }
    let (mut fit, mut over) = (1u32, 2u32);
    while fits(over) {
        fit = over;
        over = over.checked_mul(2)?;
    }
    while over - fit > 1 {
        let mid = fit + (over - fit) / 2;
        if fits(mid) {
            fit = mid;
        } else {
            over = mid;
        }
    }
    Some(fit)
}

/// Rows of one KV block in the paged layout `brain serve` uses for this model:
/// large enough that a million-token sequence's block table stays a few thousand
/// entries (it is uploaded every decode step), and a multiple of the decode
/// kernel's key tile.
pub const KV_BLOCK_ROWS: u32 = 256;

/// Granularity of [`auto_pool_rows`]'s answer.
const AUTO_POOL_GRANULE: u32 = 16 * KV_BLOCK_ROWS;

/// The largest shared KV pool, in rows, that `shape` leaves room for on `devices`
/// once the weights, the recurrent state and the step workspace are placed
/// ([`auto_cap`]'s counterpart for the paged layout: sequences hold the blocks
/// their tokens need, so what memory bounds is the pool, not a window per slot).
/// With `cap` the per-sequence context is fixed and the workspace is sized for
/// it; without, one sequence may take the whole pool. Rounded down to
/// [`AUTO_POOL_GRANULE`]; `None` when not even one block fits, or when a fixed
/// `cap` does not fit the pool.
pub fn auto_pool_rows(cfg: &Qwen35Config, devices: &[(Device, u64)], shape: &ServingShape, cap: Option<u32>) -> Option<u32> {
    if gpu_capacities(devices, 0).is_empty() {
        return None;
    }
    let fits = |rows: u32| {
        let at = ServingShape { kv_pool_rows: rows, cap: cap.unwrap_or(rows), ..shape.clone() };
        let caps = gpu_capacities(devices, at.stage_workspace(cfg));
        kv_rows_addressable(cfg, rows) && model::shard::plan_fewest_devices(&serving_cost(cfg, &at), &caps).is_some()
    };
    let fit = largest_fitting(fits)?;
    let rows = if fit >= AUTO_POOL_GRANULE { fit / AUTO_POOL_GRANULE * AUTO_POOL_GRANULE } else { fit / KV_BLOCK_ROWS * KV_BLOCK_ROWS };
    (rows > 0 && rows >= cap.unwrap_or(0)).then_some(rows)
}

// ------------------------------------------------------------ the fetch plan

/// Every brain-canonical name this GGUF offers for the MAIN decoder stack,
/// mapped to the GGUF tensor it comes from - built by running
/// [`crate::gguf_import::classify`] (the offline converter's OWN llama.cpp
/// name classifier, not a second copy) over `mg.names()`.
///
/// `Mapped::Dropped` names are skipped, which is exactly how the MTP block
/// (`blk.{n_layers}.*` and its `nextn.*` extras) excludes itself: `classify`
/// already drops it wholesale, so this resident never has to special-case it.
///
/// [`Mapped::Transformed`] is a 1:1 rename too and belongs in this map -
/// the VALUE transform it carries is applied on read by [`gguf::GdnFixSource`], which
/// checks the destination name, so the two mechanisms agree by construction
/// as long as this function does not silently drop the transformed leaf.
/// (It did, once: every `A_log` vanished from the plan the moment `classify`
/// stopped returning `Simple` for it, and the load failed by name - loudly,
/// which is the only reason that was a five-minute fix rather than a silent
/// hole.) `Mapped::Split` and `Mapped::Permuted` genuinely have no place
/// here: one produces SEVERAL destinations from one source, the other
/// reorders rows, and a plain rename can express neither; this
/// architecture's map emits neither.
pub(crate) fn gguf_name_map(mg: &MmapGguf, cfg: &Qwen35Config) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for name in mg.names() {
        match crate::gguf_import::classify(name, cfg) {
            Mapped::Simple(brain) | Mapped::Transformed { into: brain, .. } => {
                out.insert(brain, name.clone());
            }
            Mapped::Split { .. } | Mapped::Permuted { .. } | Mapped::Dropped(_) => {}
        }
    }
    out
}

/// The GGUF tensor names of the three ENDPOINT tensors this resident holds
/// itself rather than inside a shard, resolved through the same
/// [`gguf_name_map`] so no llama.cpp spelling is written down twice:
/// `(embedding table, final norm, lm_head)`.
///
/// They are outside the shards because neither `[vocab, d_model]` table can
/// be an fp32 device buffer at this scale (5.09 GB, past a 24 GB P40's
/// `max_buffer_size`): the embedding is read a row at a time from the
/// mapping, and the head is quantized to INT8. See [`layer_cost`].
pub fn endpoint_names(mg: &MmapGguf, cfg: &Qwen35Config) -> Result<(String, String, String), String> {
    let available = gguf_name_map(mg, cfg);
    let get = |brain: &str| available.get(brain).cloned().ok_or_else(|| format!("{MODEL}: this GGUF offers no tensor for '{brain}'"));
    Ok((get("tok.weight")?, get("norm.weight")?, get(cfg.head_weight())?))
}

/// The `checkpoint::remap::RemapSource` fetch plan for ONE stage: exactly the
/// tensors `Qwen35::new_i8_shard` will ask for on `shard` - its own
/// `blocks.{l}.*` for `l` in `shard.start..shard.end`, plus whichever
/// endpoints `shard` declares it owns - each a 1:1 [`Fetch::Whole`] rename of
/// the GGUF tensor [`crate::gguf_import::classify`] maps to it.
///
/// This resident always builds its stages with `embed: false, head: false`
/// (see [`endpoint_names`]), so in practice the plan is layers only; the
/// function stays honest about `shard` regardless, which is what lets a test
/// pin the endpoint behaviour too.
///
/// The wanted set is [`crate::model::shard_param_list`] itself, not a
/// re-derivation of it, so a plan that satisfies this function is by
/// construction a plan `new_i8_shard` cannot find a hole in.
pub fn shard_fetch_plan(mg: &MmapGguf, cfg: &Qwen35Config, shard: &Shard) -> Result<HashMap<String, Fetch>, String> {
    let available = gguf_name_map(mg, cfg);
    let want = crate::model::shard_param_list(cfg, shard);
    let mut plan = HashMap::with_capacity(want.len());
    let mut missing = Vec::new();
    for (name, _) in &want {
        match available.get(name) {
            Some(src) => {
                plan.insert(name.clone(), Fetch::Whole(src.clone()));
            }
            None => missing.push(name.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(format!("{MODEL}: this GGUF offers no tensor for {} required name(s): {}", missing.len(), missing.join(", ")));
    }
    Ok(plan)
}

/// [`shard_fetch_plan`] wrapped as a live, transform-applying source over
/// `mg`, with the plan checked against every wanted tensor's declared element
/// count BEFORE a single byte is uploaded (`RemapSource::validate` reads
/// shapes only).
///
/// Validating up front is what turns a config-vs-checkpoint mismatch into one
/// named error instead of a panic dozens of gigabytes into a load.
pub fn shard_source<'a>(mg: &'a MmapGguf, cfg: &Qwen35Config, shard: &Shard) -> Result<GdnFixSource<'a>, String> {
    let plan = shard_fetch_plan(mg, cfg, shard)?;
    let src = RemapSource::new(mg, plan);
    src.validate(&crate::model::shard_param_list(cfg, shard))?;
    Ok(GdnFixSource::new(src, cfg.gdn_head_order()))
}

/// The lm_head on `gpu` in the tier `ckpt` holds it at ([`ResidentCheckpoint::head_tier`]): int8 quantized on the way in for a
/// GGUF (never the whole 5 GB fp32 table on the host), bf16 as stored for an HF release.
fn upload_head(ckpt: &ResidentCheckpoint, ops: &model::ops::Ops, gpu: &Gpu, name: &str, v: usize, d: usize) -> Result<model::ops::Weight, String> {
    let src = ckpt.endpoint_source();
    match ckpt.head_tier() {
        Dtype::I8 => {
            let (w, s) = model::int8::upload_quantized(&mut paramstore::upload::Uploader::new(gpu), src, name, v, d).map_err(|e| format!("{MODEL}: {e}"))?;
            Ok(model::ops::Weight::I8 { w, s, n: v as u32, k: d as u32 })
        }
        tier => {
            let staging = checkpoint::packed::BoundedWorkspace::new(checkpoint::packed::MemoryBudget::new(crate::fp8_source::WEIGHT_STAGING_BYTES, 0));
            model::ops::upload_from_source(ops, src, name, v, d, tier, &staging).ok_or_else(|| format!("{MODEL}: '{name}' is not in the checkpoint"))
        }
    }
}

/// The weight-tier policy that holds every linear of the checkpoint as it is
/// stored ([`gguf::held_tier_of`]): a Q4_K tensor as a K-quant, anything else
/// at int8 - the main layers under `blocks.{l}.`, the MTP layer's under
/// [`mtp::MTP_PREFIX`]. What the resident serves when no weight format is asked for.
pub fn held_tiers(mg: &MmapGguf, cfg: &Qwen35Config) -> Result<TierPolicy, String> {
    let all = Shard { start: 0, end: cfg.n_layers as usize, embed: false, head: false, gpu_index: 0 };
    let src = shard_source(mg, cfg, &all)?;
    let mut leaves = Vec::new();
    for (l, ty) in cfg.layer_types().into_iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(ty) {
            let name = format!("blocks.{l}.{suffix}");
            leaves.push((name.clone(), gguf::held_tier_of(&src, &name), n as u64 * k as u64));
        }
    }
    if mtp::checkpoint_has_mtp(mg, cfg) {
        let mtp::MtpSource { shard, src, .. } = mtp::mtp_source(mg, cfg, 0)?;
        for (suffix, n, k) in cfg.layer_leaves(LayerType::Full) {
            let tier = gguf::held_tier_of(&src, &format!("blocks.{}.{suffix}", shard.start));
            leaves.push((format!("{}{suffix}", mtp::MTP_PREFIX), tier, n as u64 * k as u64));
        }
    }
    Ok(TierPolicy::per_leaf(leaves))
}

/// [`held_tiers`] of the checkpoint at `path`: a GGUF file, or an HF directory ([`ResidentCheckpoint::held_tiers`]).
pub fn held_tiers_at(path: &str) -> Result<TierPolicy, String> {
    let ckpt = ResidentCheckpoint::open(path)?;
    let cfg = ckpt.config(1)?;
    ckpt.held_tiers(&cfg)
}

/// [`crate::gguf_import::config_from_gguf`] with the two adjustments a
/// multi-card resident needs, in one place so the resident and its tests
/// cannot disagree:
///
/// * `mtp = false` - the importer sets `true` because IT imports the MTP
///   block; this path must not (see this module's doc).
/// * `block_size = cap` - purely descriptive here (every stage is built at
///   `t = 1`), kept truthful so a `to_json` dump of this config says what the
///   resident actually serves.
pub fn resident_config(mg: &MmapGguf, cap: u32) -> Result<Qwen35Config, String> {
    let mut cfg = crate::gguf_import::config_from_gguf(mg)?;
    cfg.mtp = false;
    cfg.block_size = cap;
    Ok(cfg)
}

/// The cards a resident may be planned on, as `(device, usable bytes)`: only those in `allowed` (the
/// `--device` set the process was given), each less `reserve` bytes kept free. A tool that planned on
/// every physical card regardless would place the model on a card another job holds and fail the load
/// there with an out-of-memory on a card it was told to leave alone.
pub fn budgeted_gpus(allowed: &[u32], reserve: u64) -> Vec<(Device, u64)> {
    budget_cards(gpu_core::devices::gpus().iter().map(|d| (d.index, d.identity.vram_bytes)), allowed, reserve)
}

/// [`budgeted_gpus`] over an explicit list of `(index, total bytes)` cards.
pub fn budget_cards(cards: impl IntoIterator<Item = (u32, u64)>, allowed: &[u32], reserve: u64) -> Vec<(Device, u64)> {
    cards.into_iter().filter(|(index, _)| allowed.contains(index)).map(|(index, total)| (Device::Gpu(index), total.saturating_sub(reserve))).filter(|&(_, usable)| usable > 0).collect()
}

// ------------------------------------------------------------ per-shard state

/// How a stage lays out its sequences' GQA K/V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvLayout {
    /// Every slot owns one `cap`-row window: slot `s` is block `s`.
    OneBlock,
    /// One pool of `pool_rows` rows in `block_rows`-row blocks, shared by every
    /// sequence and cached prompt. A sequence's blocks are whatever its
    /// [`model::paged::BlockTable`] holds, published with
    /// [`ShardCaches::set_blocks`] before the sequence runs; a slot is only its
    /// recurrent-state row.
    Paged { block_rows: u32, pool_rows: u32 },
}

impl KvLayout {
    /// Recurrent-state rows a stage holds: one per sequence that can run. The
    /// one-block layout also gives each cached prompt its whole slot; a paged
    /// layout keeps a cached prompt as blocks and a snapshot, so it takes none.
    fn state_rows(self, max_batch: u32, prefix_slots: u32) -> u32 {
        match self {
            KvLayout::OneBlock => max_batch + prefix_slots,
            KvLayout::Paged { .. } => max_batch,
        }
    }
}

/// One STAGE's decode state for up to `slots` concurrent sequences, indexed by
/// ABSOLUTE layer index with a size-1 dummy everywhere the stage/layer-type
/// does not apply - `crate::model::DecodeCaches`' own documented convention,
/// and the same shapes `crate::serve::Engine` allocates.
///
/// Both kinds of per-sequence state are POOLS addressed by slot, so a batched
/// decode dispatch carries every sequence's row and the slot table is the only
/// thing that says whose state is whose:
///
/// * **GQA**: ONE `[kv_rows, kv_dim]` pool per layer, reached through the paged
///   kernels' block table. In the [`KvLayout::OneBlock`] layout (`kv_rows =
///   slots*cap`) sequence `i` owns rows `i*cap .. +cap` (`slot` IS the physical
///   block id); in [`KvLayout::Paged`] the sequence's blocks are whatever the
///   engine published for its slot.
/// * **GDN**: per layer a `[slots, state_len]` pool and a `[slots, hist_len]`
///   pool, row `i` belonging to slot `i`. The batched step runs on those rows
///   IN PLACE (one native launch per layer where the device offers it, one
///   gather and one scatter otherwise) instead of staging a per-sequence
///   buffer through a copy per sequence, which at 64 sequences cost half the
///   step.
///
/// The single-sequence kernels (prefill rounds, a lone decode token,
/// speculative rounds) bind whole buffers, not pool rows, so one sequence at a
/// time works on a private `work` copy of the GDN state: a prefill runs there
/// and [`Self::store_work`] writes the finished state into the sequence's pool
/// row. The single-sequence APIs that never touch the pool (`generate`, the
/// speculative loop) simply live on `work`.
struct ShardCaches {
    /// `[layer]` -> K and V planes of `[kv_rows, kv_dim]` (`slots*cap` rows, or
    /// the pool of a paged layout) in the cache's KV tier (placeholder at
    /// non-owned / GQA layers).
    gqa_kv: Vec<KvLayer>,
    /// `[layer]`: `[slots, state_len]` / `[slots, hist_len]` pools (dummy at
    /// non-owned / GQA layers).
    pool_state: Vec<DeviceBuffer>,
    pool_hist: Vec<DeviceBuffer>,
    /// `[layer]`: the one working copy a single sequence is prefilled or
    /// stepped alone on.
    work_state: Vec<DeviceBuffer>,
    work_hist: Vec<DeviceBuffer>,
    /// `[layer]`: `[snapshot_rows, state_len]` / `[snapshot_rows, hist_len]`
    /// snapshots of the working copy taken at the end of a prefill, for prefix
    /// reuse (dummy at non-owned / GQA layers).
    snap_state: Vec<DeviceBuffer>,
    snap_hist: Vec<DeviceBuffer>,
    /// `[layer]`: whether this stage owns the layer and it is a Gated-DeltaNet one.
    recurrent: Vec<bool>,
    /// `[layer]`: whether this stage owns the layer and it is a full-attention one.
    full: Vec<bool>,
    slots: u32,
    cap: u32,
    /// Rows per physical KV block: the pool's block size when `paged`, `cap`
    /// when one block holds a whole sequence.
    block_rows: u32,
    /// Whether the sequences' KV is addressed through block tables.
    paged: bool,
    /// `[slot]`: the blocks of that slot's KV window in logical order (paged
    /// layouts only). Starts as the contiguous run the one-block layout would
    /// give the slot, which is what callers that never publish blocks (the
    /// single-sequence APIs) see; the serving engine replaces it per sequence.
    slot_tables: Vec<Vec<u32>>,
    /// `[layer]`: where a speculative verify step keeps the recurrent state
    /// between a sequence's rows (see [`speculate`]); empty until speculation
    /// is enabled.
    verify_snaps: Vec<model::gdn_verify::GdnRowSnapshots>,
    /// Kernel indices of `pool_rows_scatter2`, `pool_rows_gather2` and `copy_within`
    /// on this stage's device.
    scatter: usize,
    gather: usize,
    copy_within: usize,
    /// One recurrent layer's zero state and zero conv history (a fresh sequence's starting row), cleared once.
    zero_state: DeviceBuffer,
    zero_hist: DeviceBuffer,
}

impl ShardCaches {
    fn new(gpu: &Gpu, cfg: &Qwen35Config, shard: &Shard, cap: u32, slots: u32, snapshot_rows: u32, kv_tier: KvTier, layout: KvLayout) -> ShardCaches {
        assert!(slots > 0, "{MODEL}: ShardCaches needs at least one slot");
        let kv = cfg.kv_dim() as u64;
        let (state, hist) = (cfg.gdn_state_len(), cfg.gdn_hist_len());
        // The row kernels address a pool through u32 element offsets.
        assert!(slots as u64 * state.max(hist) <= u32::MAX as u64, "{MODEL}: {slots} recurrent-state rows of {state} elements are past a u32 element offset");
        let n = cfg.n_layers as usize;
        let types = cfg.layer_types();
        let mut gqa_kv = Vec::with_capacity(n);
        let (mut pool_state, mut pool_hist, mut work_state, mut work_hist, mut snap_state, mut snap_hist, mut recurrent, mut full_layers) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (l, ty) in types.iter().enumerate() {
            let full = shard.owns(l) && *ty == LayerType::Full;
            full_layers.push(full);
            let kv_rows = match layout {
                KvLayout::OneBlock => slots as u64 * cap as u64,
                KvLayout::Paged { pool_rows, .. } => pool_rows as u64,
            };
            gqa_kv.push(if full { KvLayer::new(gpu, kv_tier, kv_rows, kv, cfg.head_dim as u64) } else { KvLayer::placeholder(gpu) });
            let lin = shard.owns(l) && *ty == LayerType::Linear;
            recurrent.push(lin);
            let (ps, ph, ws, wh) = if lin { (slots as u64 * state, slots as u64 * hist, state, hist) } else { (1, 1, 1, 1) };
            let (ss, sh) = if lin && snapshot_rows > 0 { (snapshot_rows as u64 * state, snapshot_rows as u64 * hist) } else { (1, 1) };
            pool_state.push(gpu.storage(ps));
            pool_hist.push(gpu.storage(ph));
            work_state.push(gpu.storage(ws));
            work_hist.push(gpu.storage(wh));
            snap_state.push(gpu.storage(ss));
            snap_hist.push(gpu.storage(sh));
        }
        let scatter = gpu.kernel_index("pool_rows_scatter2").expect("pool_rows_scatter2 is in the qwen35 kernel set");
        let gather = gpu.kernel_index("pool_rows_gather2").expect("pool_rows_gather2 is in the qwen35 kernel set");
        let copy_within = gpu.kernel_index("copy_within").expect("copy_within is in the qwen35 kernel set");
        let (block_rows, paged, slot_tables) = match layout {
            KvLayout::OneBlock => (cap, false, Vec::new()),
            KvLayout::Paged { block_rows, pool_rows } => {
                assert!(block_rows > 0 && pool_rows >= block_rows, "{MODEL}: a paged KV pool needs whole blocks ({pool_rows} rows in blocks of {block_rows})");
                let per_slot = cap.div_ceil(block_rows);
                let pool_blocks = pool_rows / block_rows;
                (block_rows, true, (0..slots).map(|s| (s * per_slot..(s + 1) * per_slot).take_while(|&b| b < pool_blocks).collect()).collect())
            }
        };
        let (zero_state, zero_hist) = (gpu.storage(state), gpu.storage(hist));
        let caches = ShardCaches { gqa_kv, pool_state, pool_hist, work_state, work_hist, snap_state, snap_hist, recurrent, full: full_layers, slots, cap, block_rows, paged, slot_tables, verify_snaps: Vec::new(), scatter, gather, copy_within, zero_state, zero_hist };
        gpu.submit(&[&caches.zero_state, &caches.zero_hist], &[]);
        caches.reset(gpu);
        caches
    }

    /// Zero every GDN recurrent state / conv history (pool and work) for fresh
    /// sequences. `Gpu::storage` does not guarantee zeroed memory and a fresh
    /// sequence's recurrent state MUST start at zero. The GQA pool is
    /// deliberately left alone - a decode step's attention only ever reads rows
    /// `0..=pos` of a slot's own window, so a stale row past the new sequence's
    /// length is never read (the same argument `Qwen35::reset_decode_cache`
    /// makes).
    fn reset(&self, gpu: &Gpu) {
        let clears: Vec<&DeviceBuffer> = self.pool_state.iter().chain(&self.pool_hist).chain(&self.work_state).chain(&self.work_hist).collect();
        gpu.submit(&clears, &[]);
    }

    /// Zero the working copy: a sequence about to be prefilled starts from zero
    /// state without touching any pool row another sequence is decoding on.
    fn reset_work(&self, gpu: &Gpu) {
        let clears: Vec<&DeviceBuffer> = self.work_state.iter().chain(&self.work_hist).collect();
        gpu.submit(&clears, &[]);
    }

    /// Start `slot`'s pool row from zero state: one row scatter per owned recurrent layer from the shared zero row.
    /// Unlike [`Self::reset_work`] followed by [`Self::store_work`] it touches no working copy (so nothing waits for
    /// a copy to be read) and the stream order alone keeps the row ahead of the forward that reads it.
    fn zero_pool_row(&self, gpu: &Gpu, cfg: &Qwen35Config, slot: u32) {
        assert!(slot < self.slots, "{MODEL}: slot {slot} is outside this stage's {} slots", self.slots);
        let ids = gpu.storage_uninit(1);
        gpu.write(&ids, &[slot]);
        let (sl, hl) = (cfg.gdn_state_len() as u32, cfg.gdn_hist_len() as u32);
        let steps: Vec<_> = (0..self.pool_state.len())
            .filter(|&l| self.recurrent[l])
            .map(|l| gpu.step(self.scatter, &[&self.zero_state, &self.zero_hist, &ids, &self.pool_state[l], &self.pool_hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
    }

    /// Make the working copy the GDN state of `slot`: one row scatter per
    /// owned recurrent layer, overwriting the whole row.
    fn store_work(&self, gpu: &Gpu, cfg: &Qwen35Config, slot: u32) {
        assert!(slot < self.slots, "{MODEL}: slot {slot} is outside this stage's {} slots", self.slots);
        let ids = gpu.storage(1);
        gpu.write(&ids, &[slot]);
        let (sl, hl) = (cfg.gdn_state_len() as u32, cfg.gdn_hist_len() as u32);
        let steps: Vec<_> = (0..self.pool_state.len())
            .filter(|&l| self.recurrent[l])
            .map(|l| gpu.step(self.scatter, &[&self.work_state[l], &self.work_hist[l], &ids, &self.pool_state[l], &self.pool_hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
        // The scatters must have read the working copy before it is reused.
        gpu.poll_wait();
    }

    /// Make slot `slot`'s pool row the working GDN state (one row gather per
    /// owned recurrent layer): how a prompt prefilled over several calls picks
    /// up where its last call left off.
    fn load_work(&self, gpu: &Gpu, cfg: &Qwen35Config, slot: u32) {
        assert!(slot < self.slots, "{MODEL}: slot {slot} is outside this stage's {} slots", self.slots);
        let ids = gpu.storage(1);
        gpu.write(&ids, &[slot]);
        let (sl, hl) = (cfg.gdn_state_len() as u32, cfg.gdn_hist_len() as u32);
        let steps: Vec<_> = (0..self.pool_state.len())
            .filter(|&l| self.recurrent[l])
            .map(|l| gpu.step(self.gather, &[&self.pool_state[l], &self.pool_hist[l], &ids, &self.work_state[l], &self.work_hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
    }

    /// Copy the working GDN state into snapshot row `row` (one row scatter per
    /// owned recurrent layer) - the state a later prompt that starts with this
    /// one resumes from.
    fn snapshot_work(&self, gpu: &Gpu, cfg: &Qwen35Config, row: u32) {
        let ids = gpu.storage(1);
        gpu.write(&ids, &[row]);
        let (sl, hl) = (cfg.gdn_state_len() as u32, cfg.gdn_hist_len() as u32);
        let steps: Vec<_> = (0..self.snap_state.len())
            .filter(|&l| self.recurrent[l])
            .map(|l| gpu.step(self.scatter, &[&self.work_state[l], &self.work_hist[l], &ids, &self.snap_state[l], &self.snap_hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
    }

    /// Make snapshot row `row` the working GDN state (one row gather per owned
    /// recurrent layer).
    fn restore_work(&self, gpu: &Gpu, cfg: &Qwen35Config, row: u32) {
        let ids = gpu.storage(1);
        gpu.write(&ids, &[row]);
        let (sl, hl) = (cfg.gdn_state_len() as u32, cfg.gdn_hist_len() as u32);
        let steps: Vec<_> = (0..self.snap_state.len())
            .filter(|&l| self.recurrent[l])
            .map(|l| gpu.step(self.gather, &[&self.snap_state[l], &self.snap_hist[l], &ids, &self.work_state[l], &self.work_hist[l]], &[1, sl, hl], sl + hl))
            .collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
    }

    /// Copy KV rows `0..rows` of the window `from` over the same rows of the
    /// window `to`, in every owned full-attention layer. Windows of one layout
    /// only: whole slots, or block tables (copied block by block).
    fn copy_kv_rows(&self, gpu: &Gpu, cfg: &Qwen35Config, from: KvWindow<'_>, to: KvWindow<'_>, rows: u32) {
        let (kv, hd) = (cfg.kv_dim() as u64, cfg.head_dim as u64);
        // (source row, destination row, rows) of each contiguous run.
        let runs: Vec<(u64, u64, u64)> = match (from, to) {
            (KvWindow::Block(a), KvWindow::Block(b)) => vec![(a as u64 * self.cap as u64, b as u64 * self.cap as u64, rows as u64)],
            (KvWindow::Table(a), KvWindow::Table(b)) => {
                let bs = self.block_rows as u64;
                (0..(rows as u64).div_ceil(bs)).map(|i| (a[i as usize] as u64 * bs, b[i as usize] as u64 * bs, (rows as u64 - i * bs).min(bs))).collect()
            }
            _ => panic!("{MODEL}: KV rows are copied between windows of one layout"),
        };
        let mut steps = Vec::new();
        for (layer, _) in self.gqa_kv.iter().zip(&self.full).filter(|(_, owned)| **owned) {
            for plane in [&layer.k, &layer.v] {
                for &(src, dst, n) in &runs {
                    steps.extend(plane.copy_rows(gpu, self.copy_within, src, dst, n, kv, hd));
                }
            }
        }
        gpu.submit(&[], &steps);
        gpu.poll_wait();
    }

    /// Rows per physical KV block: what the attention kernels' `block_size` is.
    fn kv_block(&self) -> u32 {
        self.block_rows
    }

    /// The physical KV blocks of `slot`'s sequence.
    fn window(&self, slot: u32) -> KvWindow<'_> {
        if self.paged {
            KvWindow::Table(&self.slot_tables[slot as usize])
        } else {
            KvWindow::Block(slot)
        }
    }

    /// Publish the physical blocks of the sequence running in `slot`, in logical
    /// order. Every kernel of the sequence addresses its KV through them, so the
    /// engine calls this whenever the sequence's table changes, before the next
    /// forward. A no-op for the one-block layout, where the slot is the block.
    fn set_blocks(&mut self, slot: u32, blocks: &[u32]) {
        if self.paged && self.slot_tables[slot as usize] != blocks {
            self.slot_tables[slot as usize] = blocks.to_vec();
        }
    }

    /// The single-sequence view (prefill, and any single-token step): GQA
    /// window of `slot`, GDN state on the working copy.
    fn view(&self, slot: u32) -> DecodeCaches<'_> {
        DecodeCaches { gqa_kv: &self.gqa_kv, gqa_cap: self.kv_block(), gqa_window: self.window(slot), gdn_state: &self.work_state, gdn_hist: &self.work_hist, slot: 0, memory: &[] }
    }

    /// The pools as a multi-sequence prefill round works on them
    /// ([`Qwen35::run_prefill_multi_stage`]): every sequence by its slot.
    fn pooled(&self) -> PooledCaches<'_> {
        PooledCaches {
            gqa_kv: &self.gqa_kv,
            gqa_cap: self.kv_block(),
            gdn_state: &self.pool_state,
            gdn_hist: &self.pool_hist,
            work_state: &self.work_state,
            work_hist: &self.work_hist,
            gather: self.gather,
            scatter: self.scatter,
        }
    }

    /// The batch's rows: row `i` decodes slot `slot[i]` at absolute position
    /// `positions[i]`. Rows name their slots, so a batch is any subset of the
    /// resident sequences in any order - what lets sequences join and leave
    /// between steps without renumbering the ones that stay. The GDN state is
    /// the pool ([`Self::gdn_pool`]), found by slot.
    fn batch(&self, slots: &[u32], positions: &[u32]) -> Vec<BatchSeq<'_>> {
        slots.iter().zip(positions).map(|(&slot, &pos)| BatchSeq { window: self.window(slot), slot, pos, gdn_state: &[], gdn_hist: &[] }).collect()
    }

    fn gdn_pool(&self) -> GdnPoolStore<'_> {
        GdnPoolStore { state: &self.pool_state, hist: &self.pool_hist, bf16: false }
    }
}

/// One sequence's share of a [`Qwen35GgufInstance::prefill_hidden_multi`] forward.
struct MultiPrefillJob<'a> {
    prompt: &'a [u32],
    /// Prompt tokens `start..end` go in; `start` rows are already in the slot's KV window.
    start: usize,
    end: usize,
    slot: u32,
}

/// What a multi-sequence forward hands back for one job.
struct MultiPrefillRows {
    /// The last row's `[d_model]` pre-norm hidden state.
    last: Vec<f32>,
    /// Every row's, in order, when asked for (the MTP head's input); else empty.
    all: Vec<f32>,
}

/// The same generated positions scored by both of this crate's decode tapes -
/// what [`Qwen35GgufInstance::tape_comparison_trace`] returns. One `[vocab]`
/// row per position in each of `decode` and `chunk`, and the token sequence
/// they were taken along.
pub struct TapeTrace {
    /// Logits from the one-token-per-dispatch decode tape (`run_decode_batch`).
    pub decode: Vec<Vec<f32>>,
    /// Logits from the chunk tape (`run_prefill_chunk_stage`) at one row.
    pub chunk: Vec<Vec<f32>>,
    /// The greedy continuation both traces followed.
    pub ids: Vec<u32>,
}

/// What the first stage of a chain starts from.
enum FirstRows<'a> {
    /// `[rows, d_model]` embedding rows decoded on the host.
    Host(&'a [f32]),
    /// The stage gathers the step's tokens from the embedding table it holds; `words` is the size of the rows it makes.
    Gather { words: usize },
}

impl<'a> FirstRows<'a> {
    fn words(&self) -> usize {
        match self {
            FirstRows::Host(x) => x.len(),
            FirstRows::Gather { words } => *words,
        }
    }

    fn stage_rows(&self) -> StageRows<'a> {
        match self {
            FirstRows::Host(x) => StageRows::Host(x),
            FirstRows::Gather { .. } => StageRows::Gather,
        }
    }
}

/// One card's stage: the [`Qwen35`] instance holding that card's layer range,
/// plus this sequence's decode state on the SAME card.
///
/// The stage's `Shard` is always `embed: false, head: false` - the endpoints
/// are the instance's ([`EmbedTable`], [`Head`]), not the shard's, because
/// neither `[vocab, d_model]` table can be an fp32 device buffer at this
/// scale (see [`layer_cost`]).
struct DeviceShard {
    qwen35: Qwen35,
    caches: ShardCaches,
}

/// How this instance reads one embedding row.
///
/// The table is `[vocab, d_model]` - 5.09 GB as f32 - and decode only ever
/// needs a per-token ROW, never a GEMM. So it is neither uploaded nor
/// materialized on the host: [`MmapGguf::tensor_range`] dequantizes exactly
/// the quant blocks that row touches, straight out of the mapping. Peak
/// allocation is one row (20 KiB).
struct EmbedTable {
    /// The GGUF tensor name (`token_embd.weight` on a llama.cpp file, but
    /// resolved via [`endpoint_names`], never spelled here).
    name: String,
    d: usize,
}

/// The head epilogue this resident owns, on the LAST stage's card: the final
/// `norm.weight` and an INT8 `lm_head`, projected with
/// `crate::stream::head_logits_on` (the same implementation the streaming
/// real-weight path uses - there is exactly one "final norm then project to
/// vocab logits" in this crate).
///
/// It lives here rather than inside the last `Qwen35` shard because
/// `Qwen35::new_impl_on` would hold the head as plain fp32, which at
/// `[248320, 5120]` is 5.09 GB - past a 24 GB P40's `max_buffer_size`, so
/// such a shard cannot be built at all. Quantized it is 1.42 GB, inside both
/// that and the 2047 MiB storage-binding limit.
struct Head {
    ops: model::ops::Ops,
    norm: DeviceBuffer,
    w: model::ops::Weight,
}

// ------------------------------------------------------------ the instance

/// **The target's own internals, kept for a cross-attending drafter.**
///
/// A DFlash2-style draft model does not read the target's OUTPUT; it reads
/// the residual leaving a handful of the target's layers and cross-attends to
/// it. That is a number no caller can reconstruct from logits, so the stack
/// has to hand it out, and the cheapest place to keep it is one host-side
/// slab indexed by ABSOLUTE token position - the same index the KV cache uses,
/// so a speculative round that rewinds and re-commits its accepted prefix
/// overwrites exactly the rows it invalidated, with no bookkeeping.
///
/// Off unless [`Qwen35GgufInstance::enable_hidden_taps`] has run; the slab is
/// `cap * layers.len() * d_model` f32 (104 MB at five layers and a 1024-token
/// cap), which is why it is allocated then and not at construction.
struct HiddenTaps {
    /// ABSOLUTE decoder-layer indices, in the order a consumer wants them
    /// CONCATENATED - the draft model's `fc` consumes one wide row, and which
    /// 5120-wide band means which layer is fixed by the checkpoint.
    layers: Vec<usize>,
    /// `[cap, layers.len() * d_model]`, by absolute position.
    rows: RefCell<Vec<f32>>,
    /// How far the current sequence has been written. Read past this and the
    /// rows are a previous generation's, which is a wrong answer rather than
    /// a missing one - so it is checked, not trusted.
    filled: Cell<u32>,
    /// Bumped by every [`Qwen35GgufInstance::reset`]. A consumer that CACHES
    /// anything derived from these rows - which a drafter must, or it would
    /// re-project the whole context every round - cannot detect a new
    /// sequence from the context length alone: a second generation with a
    /// longer prompt looks exactly like the first one continuing. This is the
    /// number that distinguishes them.
    generation: Cell<u64>,
}

/// A built, multi-card, GGUF-resident Qwen3.8-27B.
pub struct Qwen35GgufInstance {
    cfg: Qwen35Config,
    shards: Vec<DeviceShard>,
    /// Kept open for the instance's lifetime so embedding rows can be read on
    /// demand. This is the header + mapping handle, not data.
    ckpt: ResidentCheckpoint,
    embed: EmbedTable,
    /// Whether the first stage holds the embedding table on its card ([`ServingShape::embed`]); otherwise `embed`
    /// rows are decoded on the host.
    embed_resident: bool,
    head: Head,
    tok: QwenBpe,
    /// Stop ids from the GGUF's own embedded tokenizer metadata plus the chat
    /// markup this model's template emits.
    eos: Vec<u32>,
    /// `prompt + max_new` ceiling for one sequence (the cache capacity every
    /// stage was built with).
    cap: u32,
    /// The exact-decode regime (`--exact-decode`): every stage groups its
    /// decode rows and sizes their attention from [`Self::cap`], and the head
    /// projects the batch in the same groups - see
    /// [`Qwen35::set_exact_decode`](crate::model::Qwen35::set_exact_decode).
    exact_decode: bool,
    /// How many sequences one [`Self::generate_batch`] call may carry - every
    /// stage's [`ShardCaches`] was built with exactly this many slots, and the
    /// placement was charged for them ([`layer_cost`]). `1` is the default and
    /// leaves this instance byte-identical to an unbatched one.
    max_batch: u32,
    /// Cached prompts the stages hold room for ([`Qwen35GgufResident::with_prefix_slots`]).
    prefix_slots: u32,
    /// How every stage stores its GQA layers' K/V ([`KvTier`]).
    kv_tier: KvTier,
    /// The decoder body's weight tier the stages were loaded at.
    tier: TierPolicy,
    /// How the stages address a sequence's K/V rows.
    kv_layout: KvLayout,
    /// The per-layer residual tap a cross-attending drafter needs - `None`
    /// (and free) until [`Self::enable_hidden_taps`] turns it on.
    taps: RefCell<Option<HiddenTaps>>,
    /// The adapter bank slot every row of the prefill now running belongs to
    /// ([`LORA_SLOT_NONE`]: the base). Set by the engine around
    /// [`Self::prefill_hidden`]; a prefill is one sequence, so one slot.
    prefill_adapter_slot: Cell<u32>,
    /// Packed rows a speculative verify step may carry (0: this instance does
    /// not speculate) - see [`speculate`].
    spec_rows: u32,
    /// The checkpoint's multi-token-prediction head, when loaded - see [`mtp`].
    mtp: Option<mtp::MtpDrafter>,
    /// Recorded decode steps ([`Self::decode_planned`]).
    plans: RefCell<DecodePlans>,
    /// Whether a plain decode step may replay a recording (on by default).
    plans_on: Cell<bool>,
}

/// What a decode step's head hands the host.
pub(crate) enum HeadOut {
    Ids(Vec<u32>),
    TopK(Vec<Vec<(u32, f32)>>),
}

impl HeadOut {
    pub(crate) fn into_ids(self) -> Vec<u32> {
        match self {
            HeadOut::Ids(ids) => ids,
            HeadOut::TopK(_) => unreachable!("a greedy head answered with candidates"),
        }
    }

    pub(crate) fn into_topk(self) -> Vec<Vec<(u32, f32)>> {
        match self {
            HeadOut::TopK(top) => top,
            HeadOut::Ids(_) => unreachable!("a top-k head answered with ids"),
        }
    }
}

/// Row groups `[start, end)` of an exact-decode head projection over `rows`
/// rows, at most `step` rows each.
fn head_groups(rows: u32, step: u32) -> impl Iterator<Item = (usize, usize)> {
    let rows = rows as usize;
    let step = step as usize;
    (0..rows).step_by(step).map(move |start| (start, (start + step).min(rows)))
}

impl Qwen35GgufInstance {
    /// The model config the resident was built from.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Every stage's model, in layer order.
    fn stage_models(&self) -> impl Iterator<Item = &Qwen35> + Clone {
        self.shards.iter().map(|s| &s.qwen35)
    }

    /// Run the rows of the forward about to start under the adapter slots
    /// `slots` (empty: the base's dispatches, unchanged).
    fn set_adapter_rows(&self, slots: &[u32]) {
        adapters::set_rows(self.stage_models(), slots);
    }

    /// Token `t`'s embedding row, `[d_model]`, read from the mapping.
    fn embed_row(&self, t: u32) -> Result<Vec<f32>, String> {
        self.ckpt
            .embed_rows(&self.embed.name, &[t as usize], self.embed.d)
            .ok_or_else(|| format!("{MODEL}: token id {t} is outside '{}'", self.embed.name))?
    }

    /// A whole prefill round's embeddings, `[n, d_model]` in token order -
    /// [`Self::embed_row`]'s chunked form, and stage 0's `input_override` for
    /// [`Qwen35::prefill_chunk_stage`].
    ///
    /// The rows are gathered by [`MmapGguf::tensor_rows`]: an arbitrary set of token ids names an arbitrary set of
    /// ROWS, which are not contiguous in the `[vocab, d_model]` table, so each is its own range of quant blocks
    /// (never the 5.09 GB table), and the rows decode in parallel with one hand-off to the CPU pool. One hand-off per
    /// row, as this was, kept the device idle at the start of every 512-row prompt round.
    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let rows: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        self.ckpt.embed_rows(&self.embed.name, &rows, self.embed.d).ok_or_else(|| {
            let bad = tokens.iter().find(|&&t| self.ckpt.embed_rows(&self.embed.name, &[t as usize], self.embed.d).is_none()).copied().unwrap_or(0);
            format!("{MODEL}: token id {bad} is outside '{}'", self.embed.name)
        })?
    }

    /// `input` through EVERY stage in order, returning the last stage's output on the host. `run` is one
    /// stage's forward over its input rows; `observe` sees each stage's output (and the handle that owns it)
    /// before the next stage starts.
    ///
    /// The residual crosses each card boundary device to device ([`StageRows::Device`]): the next stage's
    /// forward copies it straight from the previous stage's output over the card link when the pair has
    /// one, ordered on the two streams, so the host neither waits for a stage nor carries its output. Only
    /// the first stage's input (the embedding rows, which live in the mapped file) and the last stage's output
    /// (which the head projects) cross the host. A stage whose layer range is empty passes its input through.
    fn chain_stages<'s>(&'s self, input: &[f32], run: impl FnMut(&'s DeviceShard, StageRows<'_>) -> DeviceBuffer, observe: impl FnMut(&Gpu, &DeviceBuffer)) -> Vec<f32> {
        let words = input.len();
        self.chain_stages_with(FirstRows::Host(input), run, observe, |g, out| g.read(out, words))
    }

    /// [`Self::chain_stages`] with the first stage's input given as [`FirstRows`] and the last stage's output handed to
    /// `finish` instead of read whole: a caller that needs a few rows of a large block (a mixed round's decode rows
    /// and each prompt's last row) reads only those.
    fn chain_stages_with<'s, R>(
        &'s self,
        first: FirstRows<'_>,
        mut run: impl FnMut(&'s DeviceShard, StageRows<'_>) -> DeviceBuffer,
        mut observe: impl FnMut(&Gpu, &DeviceBuffer),
        finish: impl FnOnce(&Gpu, &DeviceBuffer) -> R,
    ) -> R {
        let words = first.words();
        let mut prev: Option<(&Gpu, DeviceBuffer)> = None;
        for s in &self.shards {
            let rows = match &prev {
                None => first.stage_rows(),
                Some((from, out)) => StageRows::device(from, out, words),
            };
            let out = run(s, rows);
            observe(&s.qwen35.gpu, &out);
            prev = Some((&s.qwen35.gpu, out));
        }
        let (g, out) = prev.expect("a plan always has at least one stage");
        finish(g, &out)
    }

    /// The input of the first stage for a step of `tokens`: its embedding rows read from the checkpoint on the host,
    /// or - when that stage holds the table ([`Self::embeds_on_device`]) - the step's token ids alone, which the
    /// card gathers itself (no host decode of the rows, no upload of them).
    fn first_rows<'a>(&self, tokens: &[u32], host_rows: &'a mut Vec<f32>) -> Result<FirstRows<'a>, String> {
        if self.embeds_on_device() {
            if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
                return Err(format!("{MODEL}: token id {bad} is outside '{}'", self.embed.name));
            }
            return Ok(FirstRows::Gather { words: tokens.len() * self.cfg.d_model as usize });
        }
        *host_rows = self.embed_rows(tokens)?;
        Ok(FirstRows::Host(host_rows))
    }

    /// Whether the first stage holds the embedding table on its card (see [`ServingShape::embed`]).
    fn embeds_on_device(&self) -> bool {
        self.embed_resident
    }

    /// One token through EVERY stage in order, then the head - returning
    /// `[vocab]` logits. The residual stream crosses each card boundary
    /// device to device ([`Self::chain_stages`]); the embedding enters stage 0
    /// from the host and the final hidden state reaches the head on the host.
    ///
    /// A stage whose layer range is empty (which a capacity-driven plan may
    /// legitimately produce) needs no special case: `run_decode_step` passes
    /// its input straight through.
    fn stack_step(&self, token_id: u32, pos: u32, slot: u32) -> Result<Vec<f32>, String> {
        let embedded = self.embed_row(token_id)?;
        let debug = std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_some();
        let mut per_stage_rms = Vec::new();
        let carry = self.chain_stages(
            &embedded,
            |s, rows| s.qwen35.run_decode_step(token_id, pos, &s.caches.view(slot), Some(rows)),
            |g, out| {
                if debug {
                    let carry = g.read(out, embedded.len());
                    per_stage_rms.push((carry.iter().map(|v| v * v).sum::<f32>() / carry.len() as f32).sqrt());
                }
            },
        );
        let last = self.shards.last().expect("a plan always has at least one stage");
        let logits = crate::stream::head_logits_on(&last.qwen35.gpu, &self.head.ops, &self.cfg, &self.head.norm, &self.head.w, &carry);
        self.debug_step(pos, &per_stage_rms, &logits);
        Ok(logits)
    }

    /// **One PREFILL ROUND through every stage in order**, then the head -
    /// [`Self::stack_step`]'s multi-token sibling, and the reason a real
    /// prompt no longer costs one whole pipeline pass per token.
    ///
    /// `tokens` are `n` consecutive prompt tokens starting at absolute
    /// position `pos_start`. The boundary residual crosses each card boundary
    /// device to device exactly as in `stack_step`, but as the round's whole
    /// `[n, d_model]` block rather than one `[d_model]` row - so the
    /// hand-offs per round are `n_stages`, not `n_stages * n`, and each
    /// stage issues one dispatch shape per layer instead of `n` of them.
    ///
    /// Only the LAST row is projected to logits: the rest of the block exists
    /// to have been computed (its K/V is in the cache, its GDN state is
    /// threaded), not to be sampled from. Returns that last token's `[vocab]`
    /// logits, so a caller can treat a round exactly like the last
    /// `stack_step` of the tokens it consumed.
    fn stack_prefill_chunk(&self, tokens: &[u32], pos_start: u32, slot: u32) -> Result<Vec<f32>, String> {
        let d = self.cfg.d_model as usize;
        let n = tokens.len();
        let (carry, per_stage_rms) = self.stack_chunk_carry(tokens, pos_start, slot)?;
        let last = self.shards.last().expect("a plan always has at least one stage");
        let logits =
            crate::stream::head_logits_on(&last.qwen35.gpu, &self.head.ops, &self.cfg, &self.head.norm, &self.head.w, &carry[(n - 1) * d..]);
        self.debug_step(pos_start + n as u32 - 1, &per_stage_rms, &logits);
        Ok(logits)
    }

    /// The layer half of a chunk round: `tokens` through every stage in order,
    /// leaving each stage's K/V and GDN state exactly as a token-by-token
    /// replay would, and returning the `[n, d_model]` final-norm block plus
    /// the per-stage residual RMS the debug dump wants.
    ///
    /// Split out from [`Self::stack_prefill_chunk`] because a prompt round and
    /// a speculative VERIFY round want different halves of the same work: a
    /// prompt round projects only its last row (the rest exists to have been
    /// computed), a verify round must score every row. The layer pass is
    /// identical, and having one copy of it is what makes "a verify is
    /// structurally a prefill round" true in the code and not just in a
    /// comment.
    fn stack_chunk_carry(&self, tokens: &[u32], pos_start: u32, slot: u32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let d = self.cfg.d_model as usize;
        let n = tokens.len();
        assert!(n > 0, "{MODEL}: stack_chunk_carry on an empty round");
        let embedded = self.embed_rows(tokens)?;
        let debug = std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_some();
        let mut per_stage_rms = Vec::new();
        let mut taps: Vec<(usize, Vec<f32>)> = Vec::new();
        let adapter_slot = self.prefill_adapter_slot.get();
        self.set_adapter_rows(&vec![adapter_slot; if adapter_slot == LORA_SLOT_NONE { 0 } else { n }]);
        let carry = self.chain_stages(
            &embedded,
            |s, rows| s.qwen35.run_prefill_chunk_stage(tokens, pos_start, &s.caches.view(slot), Some(rows)),
            |g, out| {
                if debug {
                    // The round's LAST row, so the dump is directly comparable to
                    // the per-token one `stack_step` prints.
                    let block = g.read(out, n * d);
                    let last = &block[(n - 1) * d..];
                    per_stage_rms.push((last.iter().map(|v| v * v).sum::<f32>() / last.len() as f32).sqrt());
                }
            },
        );
        for s in &self.shards {
            taps.extend(s.qwen35.take_hidden_taps());
        }
        self.set_adapter_rows(&[]);
        self.record_taps(pos_start, n, &taps)?;
        Ok((carry, per_stage_rms))
    }

    /// **Turn the per-layer residual tap on**, for the ABSOLUTE decoder-layer
    /// indices `layers`, in the order a consumer wants them concatenated.
    ///
    /// Allocates the whole `cap * layers.len() * d_model` slab up front - a
    /// tap that grew as it filled would reallocate inside a decode round, and
    /// this path's whole design is that a round allocates nothing.
    ///
    /// Only the CHUNK tape fills it ([`Self::stack_chunk_carry`], which is
    /// every prefill round and every speculative verify). The one-token decode
    /// tape does not, deliberately: a drafter that needs the target's hidden
    /// states runs inside [`Self::generate_speculative`], which never takes
    /// that tape, and tapping it too would put a readback on the hot path of
    /// plain generation for a consumer that has none.
    pub fn enable_hidden_taps(&self, layers: &[usize]) {
        for &l in layers {
            assert!(l < self.cfg.n_layers as usize, "{MODEL}: hidden tap asked for layer {l} of a {}-layer model", self.cfg.n_layers);
        }
        for s in &self.shards {
            s.qwen35.set_hidden_taps(layers);
        }
        let width = layers.len() * self.cfg.d_model as usize;
        *self.taps.borrow_mut() = Some(HiddenTaps {
            layers: layers.to_vec(),
            rows: RefCell::new(vec![0.0; self.cap as usize * width]),
            filled: Cell::new(0),
            generation: Cell::new(0),
        });
    }

    /// One card this instance is resident on, for a SECOND model that wants to
    /// live beside it. `Gpu::new_like` off this handle keeps the same adapter,
    /// queue and memory accounting while registering a different kernel set -
    /// which is what a co-resident drafter (`crate::dflash2`) needs, since its
    /// architecture dispatches kernels this one never compiled.
    pub fn stage_gpu(&self, stage: usize) -> &Gpu {
        &self.shards[stage].qwen35.gpu
    }

    /// Move every stage's scratch-arena threshold - see
    /// [`Qwen35::set_chunk_arena_min_rows`]. What the cost ladder in
    /// `tests/gguf_resident_verify_cost_real.rs` sweeps to measure the two
    /// regimes against each other on the real checkpoint instead of quoting one
    /// of them from a rebuild.
    pub fn set_chunk_arena_min_rows(&self, rows: u32) {
        for s in &self.shards {
            s.qwen35.set_chunk_arena_min_rows(rows);
        }
    }

    /// Every stage's [`Qwen35::set_decode_fusion`]: the native fused kernels of the decode step and of a prefill round
    /// (on by default; off is the chain of dispatches they replace, bit for bit).
    pub fn set_decode_fusion(&self, on: bool) {
        for s in &self.shards {
            s.qwen35.set_decode_fusion(on);
        }
    }

    /// Every stage's [`Qwen35::set_chunk_arena_drain`].
    pub fn set_chunk_arena_drain(&self, drain: Option<bool>) {
        for s in &self.shards {
            s.qwen35.set_chunk_arena_drain(drain);
        }
    }

    /// Every stage's [`Qwen35::set_exact_rounds`]: whether a short chunk round
    /// (a speculative verify round, a ragged prompt tail) runs as plain decode
    /// steps and so agrees with plain decode bit for bit. On by default; the
    /// off position is how a test measures the chunk tape's own distance from
    /// the decode tape.
    pub fn set_exact_rounds(&self, on: bool) {
        for s in &self.shards {
            s.qwen35.set_exact_rounds(on);
        }
    }

    /// How many cards the plan placed this instance on.
    pub fn stages(&self) -> usize {
        self.shards.len()
    }

    /// The TARGET's own embedding rows for `tokens`, `[tokens.len(), d_model]`.
    ///
    /// A DFlash2-style drafter ships no embedding table - it embeds
    /// `[anchor, MASK, ...]` with the target's, by design - so the table has to
    /// be reachable. It is not a device buffer here and never was (see
    /// [`EmbedTable`]); this reads the rows straight out of the mapping, which
    /// is what every prefill round already does.
    pub fn embed_rows_of(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.embed_rows(tokens)
    }

    /// **Project `rows` PRE-final-norm hidden states to `[rows, vocab]`
    /// logits with this instance's own resident INT8 `lm_head`**, applying
    /// `norm` as the final RMSNorm weight instead of the target's.
    ///
    /// The substitution is the point. A draft model that reuses the target's
    /// head has its OWN final norm in front of it, and the composition
    /// "draft's norm, then target's head" is exactly one
    /// `head_logits_rows_on` call with a different weight vector - so a
    /// co-resident drafter needs no second copy of a 1.42 GB head, which at
    /// this shape is the difference between fitting on two cards and not.
    ///
    /// `hidden` is `[rows, d_model]` and must come from a model whose hidden
    /// width matches this one's; nothing else about it is assumed.
    pub fn logits_with_norm(&self, hidden: &[f32], rows: u32, norm: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.cfg.d_model as usize;
        if norm.len() != d {
            return Err(format!("{MODEL}: logits_with_norm got a {}-wide final norm for a {d}-wide model", norm.len()));
        }
        if hidden.len() != rows as usize * d {
            return Err(format!("{MODEL}: logits_with_norm got {} floats for {rows} rows of {d}", hidden.len()));
        }
        let last = self.shards.last().expect("a plan always has at least one stage");
        let nb = last.qwen35.gpu.storage_init("qwen35.head.foreign_norm", norm);
        Ok(crate::stream::head_logits_rows_on(&last.qwen35.gpu, &self.head.ops, &self.cfg, &nb, &self.head.w, hidden, rows))
    }

    /// Which SEQUENCE the tapped rows belong to - see [`HiddenTaps::generation`].
    ///
    /// A consumer caching anything derived from [`Self::target_hidden`] must
    /// compare this every round and throw its cache away when it changes.
    /// Nothing about the context length can tell it: a new generation with a
    /// longer prompt is indistinguishable from the previous one continuing,
    /// and the failure mode is a drafter conditioned on another request's
    /// context, which produces fluent proposals that are simply never
    /// accepted.
    pub fn hidden_tap_generation(&self) -> u64 {
        self.taps.borrow().as_ref().map(|t| t.generation.get()).unwrap_or(0)
    }

    /// How wide one tapped row is - `layers.len() * d_model`, the width a
    /// consumer's own projection must expect.
    pub fn hidden_tap_width(&self) -> usize {
        self.taps.borrow().as_ref().map(|t| t.layers.len() * self.cfg.d_model as usize).unwrap_or(0)
    }

    /// The tapped rows for absolute positions `from .. from+n`, concatenated
    /// in [`Self::enable_hidden_taps`]' layer order - `[n, hidden_tap_width()]`.
    ///
    /// Errors rather than returning stale rows when asked past what the
    /// current sequence has actually computed: at this scale a plausible-
    /// looking wrong hidden state produces plausible-looking wrong drafts,
    /// which is a silent acceptance-rate loss and not a visible failure.
    pub fn target_hidden(&self, from: u32, n: u32) -> Result<Vec<f32>, String> {
        let taps = self.taps.borrow();
        let taps = taps.as_ref().ok_or_else(|| format!("{MODEL}: target_hidden before enable_hidden_taps"))?;
        if from + n > taps.filled.get() {
            return Err(format!("{MODEL}: target_hidden asked for positions {from}..{} but only {} have been computed", from + n, taps.filled.get()));
        }
        let w = taps.layers.len() * self.cfg.d_model as usize;
        let out = taps.rows.borrow()[from as usize * w..(from + n) as usize * w].to_vec();
        Ok(out)
    }

    /// File one chunk's taps into the position-indexed slab. `collected` is
    /// `(absolute layer, [n, d_model])` from every stage, in whatever order
    /// the pipeline produced them; the slab wants them in the CONCATENATION
    /// order the consumer asked for, which is a different thing and is why
    /// this looks each layer up rather than appending.
    fn record_taps(&self, pos_start: u32, n: usize, collected: &[(usize, Vec<f32>)]) -> Result<(), String> {
        let taps = self.taps.borrow();
        let Some(taps) = taps.as_ref() else { return Ok(()) };
        let d = self.cfg.d_model as usize;
        let w = taps.layers.len() * d;
        let mut rows = taps.rows.borrow_mut();
        for (band, &l) in taps.layers.iter().enumerate() {
            let src = collected
                .iter()
                .find(|(li, _)| *li == l)
                .map(|(_, v)| v)
                .ok_or_else(|| format!("{MODEL}: hidden tap for layer {l} was requested but no stage produced it"))?;
            assert_eq!(src.len(), n * d, "{MODEL}: hidden tap for layer {l} is {} floats, expected {}", src.len(), n * d);
            for i in 0..n {
                let dst = (pos_start as usize + i) * w + band * d;
                rows[dst..dst + d].copy_from_slice(&src[i * d..(i + 1) * d]);
            }
        }
        taps.filled.set(taps.filled.get().max(pos_start + n as u32));
        Ok(())
    }

    /// **The speculative verify pass**: `tokens` at consecutive positions from
    /// `pos_start`, scored to `[vocab]` logits at EVERY row rather than only
    /// the last. One head projection for the whole block, exactly as
    /// [`Self::stack_step_batch`] does it - the head is this model's largest
    /// single weight and projecting it once per row instead of once per block
    /// would hand back most of what speculation buys.
    fn stack_verify_chunk(&self, tokens: &[u32], pos_start: u32, slot: u32) -> Result<Vec<Vec<f32>>, String> {
        let (carry, _) = self.stack_chunk_carry(tokens, pos_start, slot)?;
        let last = self.shards.last().expect("a plan always has at least one stage");
        let v = self.cfg.vocab as usize;
        let flat = crate::stream::head_logits_rows_on(
            &last.qwen35.gpu,
            &self.head.ops,
            &self.cfg,
            &self.head.norm,
            &self.head.w,
            &carry,
            tokens.len() as u32,
        );
        Ok(flat.chunks(v).map(|r| r.to_vec()).collect())
    }

    /// **One decode token for EVERY live sequence**, through every stage in
    /// order, then the head - [`Self::stack_step`]'s cross-sequence sibling,
    /// and the whole of what makes `run_batch` a batch rather than a loop.
    ///
    /// `tokens[i]`/`positions[i]` belong to slot `i` of every stage's
    /// [`ShardCaches`] (slot order IS batch-row order and IS the physical
    /// block id - see that struct's own doc). The residual crosses each card
    /// boundary device to device as the whole `[bsz, d_model]` block, so the
    /// hand-offs per step are `n_stages`, not `n_stages * bsz`, and the head
    /// is projected ONCE for the whole batch rather than once per sequence -
    /// which matters most of all, the head being this model's single largest
    /// weight.
    ///
    /// Returns one `[vocab]` logits row per sequence, in batch order.
    fn stack_step_batch(&self, tokens: &[u32], positions: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        let carry = self.stack_decode_hidden(tokens, &slots, positions)?;
        let flat = self.head_logits_rows(&carry, tokens.len() as u32);
        Ok(flat.chunks(self.cfg.vocab as usize).map(|r| r.to_vec()).collect())
    }

    /// The layer half of a batched decode step: row `i` is `tokens[i]` at
    /// `positions[i]` of slot `slots[i]`, and the result is the `[bsz,
    /// d_model]` pre-norm hidden block the head projects. Slots must be
    /// distinct (two rows of one sequence would race on its state).
    fn stack_decode_hidden(&self, tokens: &[u32], slots: &[u32], positions: &[u32]) -> Result<Vec<f32>, String> {
        let hidden = self.stack_decode_hidden_dev(tokens, slots, positions)?;
        Ok(self.hidden_to_host(&hidden, tokens.len()))
    }

    /// A decode step's `[rows, d_model]` hidden block, read from the last stage's card.
    fn hidden_to_host(&self, hidden: &DeviceBuffer, rows: usize) -> Vec<f32> {
        let (g, ..) = self.head_parts();
        g.read(hidden, rows * self.cfg.d_model as usize)
    }

    /// Hold the last stage's dispatches from here to the end of the scope and issue them together: a decode step's
    /// layers and its head are then one submission (one graph), with no read-back of the residual between them.
    fn step_scope(&self) -> impl Drop + '_ {
        self.head_parts().0.pass_scope()
    }

    /// [`Self::stack_decode_hidden`] leaving the block on the last stage's card, still being computed.
    fn stack_decode_hidden_dev(&self, tokens: &[u32], slots: &[u32], positions: &[u32]) -> Result<DeviceBuffer, String> {
        let bsz = tokens.len();
        assert!(bsz == positions.len() && bsz == slots.len(), "{MODEL}: a decode step needs one slot and position per token ({bsz} tokens, {} slots, {} positions)", slots.len(), positions.len());
        assert!(bsz > 0, "{MODEL}: stack_decode_hidden on an empty batch");
        for s in &self.shards {
            assert!(bsz <= s.caches.slots as usize, "{MODEL}: a decode step got {bsz} sequences but a stage holds only {} slots", s.caches.slots);
            if let Some(&bad) = slots.iter().find(|&&sl| sl >= s.caches.slots) {
                panic!("{MODEL}: slot {bad} is outside this instance's {} slots", s.caches.slots);
            }
        }
        debug_assert!(slots.iter().collect::<std::collections::HashSet<_>>().len() == bsz, "{MODEL}: a decode step names a slot twice");
        let mut host_rows = Vec::new();
        let first = {
            let _embed = gpu_core::markers::range(c"decode.embed");
            self.first_rows(tokens, &mut host_rows)?
        };
        Ok(self.chain_stages_with(
            first,
            |s, rows| {
                let seqs = s.caches.batch(slots, positions);
                let caches = BatchDecodeCaches { gqa_kv: &s.caches.gqa_kv, gqa_cap: s.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: Some(s.caches.gdn_pool()), memory: &[] };
                s.qwen35.run_decode_batch(tokens, &caches, Some(rows))
            },
            |_, _| {},
            |_, out| out.clone(),
        ))
    }

    /// The last stage's card and the head epilogue it runs - the one place a
    /// hidden block becomes logits, greedy picks or top-k candidates.
    fn head_parts(&self) -> (&Gpu, &model::ops::Ops, &DeviceBuffer, &model::ops::Weight) {
        let last = self.shards.last().expect("a plan always has at least one stage");
        (&last.qwen35.gpu, &self.head.ops, &self.head.norm, &self.head.w)
    }

    /// The exact-decode group size for the head, `None` when the regime is off:
    /// the resident head is int8 here, so its row-count gate is the decode
    /// projections' own and it takes the same groups.
    fn exact_head_step(&self) -> Option<u32> {
        if !self.exact_decode {
            return None;
        }
        let last = self.shards.last()?;
        Some(gpu_core::select::exact_decode_group_rows(&last.qwen35.gpu.caps()))
    }

    /// `[rows, vocab]` logits of a `[rows, d_model]` hidden block, on the host.
    fn head_logits_rows(&self, hidden: &[f32], rows: u32) -> Vec<f32> {
        let (g, ops, norm, w) = self.head_parts();
        let d = self.cfg.d_model as usize;
        let Some(step) = self.exact_head_step().filter(|&s| rows > s) else {
            return crate::stream::head_logits_rows_on(g, ops, &self.cfg, norm, w, hidden, rows);
        };
        let mut out = Vec::with_capacity(rows as usize * self.cfg.vocab as usize);
        for (start, end) in head_groups(rows, step) {
            out.extend(crate::stream::head_logits_rows_on(g, ops, &self.cfg, norm, w, &hidden[start * d..end * d], (end - start) as u32));
        }
        out
    }

    /// Each row's greedy token, picked on the device.
    fn head_argmax_rows<'h>(&self, hidden: impl Into<HeadRows<'h>>, rows: u32) -> Vec<u32> {
        let (g, ops, norm, w) = self.head_parts();
        let hidden = hidden.into();
        let Some(step) = self.exact_head_step().filter(|&s| rows > s) else {
            return crate::stream::head_argmax_rows_on(g, ops, &self.cfg, norm, w, hidden, rows);
        };
        // Exact decode runs the head in groups of rows, which are cut on the host.
        let d = self.cfg.d_model as usize;
        let host = hidden.to_host(g, rows as usize * d);
        let mut out = Vec::with_capacity(rows as usize);
        for (start, end) in head_groups(rows, step) {
            out.extend(crate::stream::head_argmax_rows_on(g, ops, &self.cfg, norm, w, &host[start * d..end * d], (end - start) as u32));
        }
        out
    }

    /// The head of `kind` over `hidden`, read back.
    fn head_rows(&self, hidden: &DeviceBuffer, rows: u32, kind: HeadKind) -> HeadOut {
        match kind {
            HeadKind::Argmax => HeadOut::Ids(self.head_argmax_rows(hidden, rows)),
            HeadKind::TopK(cap) => HeadOut::TopK(self.head_topk_rows(hidden, rows, cap)),
        }
    }

    /// Turn recorded decode steps on or off. Off is the reference every recording is gated against: the same dispatches
    /// built and launched the ordinary way.
    pub fn set_decode_plans(&self, on: bool) {
        self.plans_on.set(on);
        if !on {
            self.plans.borrow_mut().clear();
        }
    }

    /// Recorded decode steps held now.
    pub fn decode_plans_held(&self) -> usize {
        self.plans.borrow().len()
    }

    /// One decode step of rows `tokens` (slot `slots[i]`, position `positions[i]`) and its head of `kind`, as a replay of
    /// the recording for this `(rows, context bucket, head)` - made once the key has held for [`RECORD_PLAN_AFTER`]
    /// steps. The step then costs the host a handful of small input writes and one graph launch: its ~2500 dispatches
    /// are neither rebuilt nor compared, and the layers and the head are one submission.
    ///
    /// `None` when the step does not qualify (the caller runs it the ordinary way, bit-for-bit the same arithmetic): more
    /// than one stage (the hand-off between cards is not a dispatch a recording holds), embedding rows decoded on the
    /// host, the exact-decode regime, or a key that has not held yet.
    fn decode_planned(&self, tokens: &[u32], slots: &[u32], positions: &[u32], kind: HeadKind) -> Option<HeadOut> {
        if !self.plans_on.get() || self.shards.len() != 1 || self.exact_decode || !self.embeds_on_device() {
            return None;
        }
        let s = &self.shards[0];
        let rows = tokens.len() as u32;
        if rows == 0 || rows > s.caches.slots || slots.iter().any(|&sl| sl >= s.caches.slots) || tokens.iter().any(|&t| t >= self.cfg.vocab) {
            return None;
        }
        let seqs = s.caches.batch(slots, positions);
        let caches = BatchDecodeCaches { gqa_kv: &s.caches.gqa_kv, gqa_cap: s.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: Some(s.caches.gdn_pool()), memory: &[] };
        let key = PlanKey { rows, cap: s.qwen35.paged_step_shape(&caches).0, head: kind };
        let looked = self.plans.borrow_mut().lookup(key);
        let plan = match looked {
            Ok(plan) => {
                s.qwen35.write_decode_inputs(&plan.inputs, Some(tokens), &caches);
                plan
            }
            Err(false) => return None,
            Err(true) => {
                let (tape, inputs, out) = s.qwen35.record_decode_step(tokens, &caches, |hidden| self.record_head(hidden, rows, kind));
                DecodePlan { key, tape, inputs, out }
            }
        };
        let g = &s.qwen35.gpu;
        g.replay_tape(&plan.tape);
        let out = match &plan.out {
            PlanOut::Argmax(ids) => HeadOut::Ids(model::readout::read_argmax(g, ids, rows)),
            PlanOut::TopK(top) => HeadOut::TopK(top.read(g)),
        };
        self.plans.borrow_mut().put(plan);
        Some(out)
    }

    /// The head of `kind` over `hidden` recorded into the open tape: the final norm, the projection and the reduction,
    /// whose answer stays in the returned buffers.
    fn record_head(&self, hidden: &DeviceBuffer, rows: u32, kind: HeadKind) -> PlanOut {
        let (g, ops, norm, w) = self.head_parts();
        let logits = crate::stream::head_logits_issue(g, ops, &self.cfg, norm, w, hidden, rows);
        match kind {
            HeadKind::Argmax => PlanOut::Argmax(crate::model::argmax_rows_issue_of(g, &logits, rows, self.cfg.vocab)),
            HeadKind::TopK(cap) => PlanOut::TopK(crate::model::topk_rows_issue_of(g, &logits, rows, self.cfg.vocab, cap)),
        }
    }

    /// Each row's top-`cap` candidates, extracted on the device.
    fn head_topk_rows<'h>(&self, hidden: impl Into<HeadRows<'h>>, rows: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
        let (g, ops, norm, w) = self.head_parts();
        let hidden = hidden.into();
        let Some(step) = self.exact_head_step().filter(|&s| rows > s) else {
            return crate::stream::head_topk_rows_on(g, ops, &self.cfg, norm, w, hidden, rows, cap);
        };
        // Exact decode runs the head in groups of rows, which are cut on the host.
        let d = self.cfg.d_model as usize;
        let host = hidden.to_host(g, rows as usize * d);
        let mut out = Vec::with_capacity(rows as usize);
        for (start, end) in head_groups(rows, step) {
            out.extend(crate::stream::head_topk_rows_on(g, ops, &self.cfg, norm, w, &host[start * d..end * d], (end - start) as u32, cap));
        }
        out
    }

    /// Prefill `prompt` into slot 0 through the same path a request takes
    /// (rounds of [`MAX_PREFILL_TOKENS`] when the weight tier makes a round
    /// profitable), wait for the device, and return the wall time in seconds.
    /// The instance is reset first, so this is a cold prefill of exactly
    /// `prompt`, never a continuation.
    pub fn prefill_timed(&self, prompt: &[u32]) -> Result<f64, String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: prefill_timed needs a non-empty prompt"));
        }
        if prompt.len() as u64 > self.cap as u64 {
            return Err(format!("{MODEL}: a prompt of {} tokens exceeds this instance's context capacity {}", prompt.len(), self.cap));
        }
        self.reset();
        let t = std::time::Instant::now();
        self.replay_prompt(prompt, 0)?;
        self.poll_wait();
        Ok(t.elapsed().as_secs_f64())
    }

    /// How this instance's stages store their GQA layers' K/V.
    pub fn kv_tier(&self) -> KvTier {
        self.kv_tier
    }

    /// One batched decode step with an explicit position per row, returning each
    /// row's `[vocab]` logits: the step a server runs every token, without the
    /// prompt that normally precedes it.
    ///
    /// For measuring decode at a context length that would take hours to
    /// prefill. Row `i` attends over positions `0..=positions[i]` of its slot;
    /// whatever those cache rows hold is read (zeros, on a fresh instance), so
    /// the memory traffic and the kernel work are those of a real context of
    /// that length, and the logits are not meaningful. A caller that wants
    /// meaningful logits builds the context with [`Self::generate_batch`].
    pub fn decode_batch_at(&self, tokens: &[u32], positions: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.check_batch_at(tokens, positions)?;
        self.stack_step_batch(tokens, positions)
    }

    /// [`Self::decode_batch_at`] for the tape a served GREEDY request runs:
    /// the same layer stack, then the device head
    /// ([`Self::stack_decode_hidden`] followed by the greedy reduction) - so
    /// only the winning token ids come back, never a `[bsz, vocab]` block.
    /// `forward_batched_greedy` is this call with the scheduler's own
    /// positions.
    pub fn decode_batch_greedy_at(&self, tokens: &[u32], positions: &[u32]) -> Result<Vec<u32>, String> {
        self.check_batch_at(tokens, positions)?;
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        let _step = self.step_scope();
        let hidden = self.stack_decode_hidden_dev(tokens, &slots, positions)?;
        Ok(self.head_argmax_rows(&hidden, tokens.len() as u32))
    }

    /// [`Self::decode_batch_at`]'s input contract, shared with the greedy
    /// sibling so both refuse the same shapes identically.
    fn check_batch_at(&self, tokens: &[u32], positions: &[u32]) -> Result<(), String> {
        if tokens.len() != positions.len() || tokens.is_empty() {
            return Err(format!("{MODEL}: decode_batch_at needs one position per token ({} tokens, {} positions)", tokens.len(), positions.len()));
        }
        if tokens.len() > self.max_batch as usize {
            return Err(format!("{MODEL}: batch of {} exceeds this instance's max_batch {}", tokens.len(), self.max_batch));
        }
        if let Some(&p) = positions.iter().find(|&&p| p >= self.cap) {
            return Err(format!("{MODEL}: position {p} is beyond this instance's context capacity {}", self.cap));
        }
        Ok(())
    }

    /// **Replay a whole prompt**, leaving every stage's GQA cache and GDN
    /// state exactly where a following decode step expects them, and returning
    /// the last prompt token's `[vocab]` logits.
    ///
    /// Rounds of [`Self::prefill_round_rows`] rows via [`Self::stack_prefill_chunk`] when
    /// this build's weight tier makes a round profitable, one token at a time
    /// via [`Self::stack_step`] when it does not
    /// ([`Qwen35::chunked_prefill_is_profitable`] owns that judgement and its
    /// reasons; `MAX_PREFILL_TOKENS`'s doc has the numbers). The two leave
    /// IDENTICAL state either way - that equivalence is the whole contract of
    /// `Qwen35::run_prefill_chunk_stage` and is gated bit-for-bit by
    /// `crate::model`'s `two_shard_chunked_prefill_matches_token_by_token_replay`
    /// - so this is purely a cost choice, never a behavioural one.
    ///
    /// One function rather than a loop at each call site, because
    /// [`Self::generate`] and [`Self::profile_decode`] must replay a prompt
    /// the SAME way: a profiler that warms up through a different tape than
    /// the one production uses is measuring the wrong thing.
    fn replay_prompt(&self, prompt: &[u32], slot: u32) -> Result<(Vec<f32>, u32), String> {
        let mut pos = 0u32;
        let mut logits = Vec::new();
        if self.shards.iter().all(|s| s.qwen35.chunked_prefill_is_profitable()) {
            for round in prompt.chunks(self.prefill_round_rows() as usize) {
                logits = self.stack_prefill_chunk(round, pos, slot)?;
                pos += round.len() as u32;
            }
        } else {
            for &t in prompt {
                logits = self.stack_step(t, pos, slot)?;
                pos += 1;
            }
        }
        Ok((logits, pos))
    }

    /// [`Self::replay_prompt`] for a sequence JOINING a running batch: prefill
    /// `prompt[start..]` into slot `slot` on the working GDN state, write that
    /// state into the slot's pool row, and return the last prompt token's
    /// `[d_model]` pre-norm hidden row for the caller's head - a scheduler
    /// picks the first token from it (greedy or sampled) without a `[vocab]`
    /// readback. No other slot is touched.
    ///
    /// `start == 0` is a cold prefill (the working state is cleared first).
    /// `start > 0` CONTINUES a cached prefix: the caller has already copied KV
    /// rows `0..start` into the slot and restored the working GDN state to the
    /// state after those `start` tokens ([`Self::seed_from_prefix`]); the rest
    /// is prefilled exactly as the chunk loop would have reached it, so the
    /// result is the cold prefill's.
    ///
    /// `collect`, when given, receives every prefilled row's `[d_model]` pre-norm
    /// hidden state (what the MTP head is fed from).
    ///
    /// Only `prompt[start..end]` is prefilled, so a long prompt can go in over
    /// several calls: every call ends with the working state written into the
    /// slot's pool row, and `resume` says the working state is NOT already the
    /// state after `start` tokens (another sequence used it since), so it is
    /// loaded from that pool row first.
    fn prefill_hidden(&self, prompt: &[u32], start: usize, end: usize, slot: u32, resume: bool, mut collect: Option<&mut Vec<f32>>) -> Result<Vec<f32>, String> {
        let d = self.cfg.d_model as usize;
        assert!(start < end && end <= prompt.len(), "{MODEL}: a continued prefill needs at least one new token");
        if start == 0 {
            for s in &self.shards {
                s.caches.reset_work(&s.qwen35.gpu);
            }
        } else if resume {
            for s in &self.shards {
                s.caches.load_work(&s.qwen35.gpu, &self.cfg, slot);
            }
        }
        let rest = &prompt[start..end];
        let mut last = Vec::new();
        if self.shards.iter().all(|s| s.qwen35.chunked_prefill_is_profitable()) {
            let mut pos = start as u32;
            for round in rest.chunks(self.prefill_round_rows() as usize) {
                let (carry, _) = self.stack_chunk_carry(round, pos, slot)?;
                last = carry[(round.len() - 1) * d..].to_vec();
                if let Some(all) = collect.as_deref_mut() {
                    all.extend_from_slice(&carry);
                }
                pos += round.len() as u32;
            }
        } else {
            let adapter_slot = self.prefill_adapter_slot.get();
            for (i, &t) in rest.iter().enumerate() {
                let embedded = self.embed_row(t)?;
                self.set_adapter_rows(&[adapter_slot]);
                let carry = self.chain_stages(&embedded, |s, rows| s.qwen35.run_decode_step(t, (start + i) as u32, &s.caches.view(slot), Some(rows)), |_, _| {});
                self.set_adapter_rows(&[]);
                if let Some(all) = collect.as_deref_mut() {
                    all.extend_from_slice(&carry);
                }
                last = carry;
            }
        }
        for s in &self.shards {
            s.caches.store_work(&s.qwen35.gpu, &self.cfg, slot);
        }
        Ok(last)
    }

    /// Whether [`Self::prefill_hidden_multi`] can serve this instance now: a
    /// multi-row round on every stage, no hidden taps armed (they record one
    /// sequence's rows) and no served-adapter bank in play (its slot is one per
    /// round). The sequential path serves everything else, identically.
    fn multi_prefill_available(&self) -> bool {
        self.taps.borrow().is_none()
            && self.shards.iter().all(|s| s.qwen35.chunked_prefill_is_profitable() && s.qwen35.multi_prefill_supported())
            && self.prefill_adapter_slot.get() == LORA_SLOT_NONE
    }

    /// Whether a forward can carry running sequences' next tokens and prompt rows together
    /// ([`Self::mixed_hidden`]): the multi-sequence round is available, nothing needs the rows of a decode step
    /// to be those of a step of their own (the MTP head records each decode step's hidden states; exact decode
    /// promises a row its bits whatever its neighbours), and no adapter bank, sliding window or anchor memory is
    /// in play.
    pub(crate) fn mixed_available(&self) -> bool {
        self.multi_prefill_available()
            && !self.has_mtp()
            && !self.exact_decode
            && adapters::bytes(self.stage_models()) == 0
            && self.cfg.swa_window.is_none()
            && self.cfg.anchor_memory.is_none()
    }

    /// [`Self::prefill_hidden`] for SEVERAL sequences in one forward: job `i`
    /// prefills `prompt[start..end]` of slot `slot`, and every projection and MLP
    /// runs once over the jobs' rows together (see
    /// [`Qwen35::run_prefill_multi_stage`]).
    ///
    /// The state a sequence continues from is its pool row (not the working copy
    /// the one-sequence path uses), so the caller has written it there first, and
    /// the pool row is where the state is left. Returns each job's last row's
    /// `[d_model]` pre-norm hidden state and, when `collect`, all its rows'.
    fn prefill_hidden_multi(&self, jobs: &[MultiPrefillJob<'_>], collect: bool) -> Result<Vec<MultiPrefillRows>, String> {
        self.forward_rows(None, jobs, collect).map(|(_, rows)| rows)
    }

    /// [`Self::prefill_hidden_multi`] with the next token of each sequence of `decode` (tokens, slots and positions,
    /// as [`Self::stack_decode_hidden`] takes them) carried by the same forward: its first element is the decode
    /// rows' `[bsz, d_model]` pre-norm hidden block, empty without any.
    fn mixed_hidden(&self, decode: (&[u32], &[u32], &[u32]), jobs: &[MultiPrefillJob<'_>], collect: bool) -> Result<(Vec<f32>, Vec<MultiPrefillRows>), String> {
        self.forward_rows(Some(decode), jobs, collect)
    }

    fn forward_rows(&self, decode: Option<(&[u32], &[u32], &[u32])>, jobs: &[MultiPrefillJob<'_>], collect: bool) -> Result<(Vec<f32>, Vec<MultiPrefillRows>), String> {
        let d = self.cfg.d_model as usize;
        assert!(!jobs.is_empty() && jobs.iter().all(|j| j.start < j.end && j.end <= j.prompt.len()), "{MODEL}: a multi-sequence prefill needs at least one new token per job");
        let lead = decode.map_or(0, |(tokens, slots, positions)| {
            assert!(tokens.len() == slots.len() && tokens.len() == positions.len(), "{MODEL}: a decode row needs one slot and position per token");
            debug_assert!(slots.iter().collect::<std::collections::HashSet<_>>().len() == slots.len(), "{MODEL}: a mixed forward names a decode slot twice");
            tokens.len()
        });
        let tokens: Vec<u32> = decode.map_or(&[][..], |(tokens, ..)| tokens).iter().copied().chain(jobs.iter().flat_map(|j| j.prompt[j.start..j.end].iter().copied())).collect();
        let mut host_rows = Vec::new();
        let first = {
            let _embed = gpu_core::markers::range(c"mixed.embed");
            self.first_rows(&tokens, &mut host_rows)?
        };
        // Only the rows the caller uses cross to the host: the decode rows, and per job its last row (every row
        // when `collect`, the drafter's feed).
        let wanted: Vec<(usize, usize)> = {
            let mut row0 = lead;
            let mut spans = vec![(0, lead)];
            for j in jobs {
                let rows = j.end - j.start;
                spans.push(if collect { (row0, rows) } else { (row0 + rows - 1, 1) });
                row0 += rows;
            }
            spans
        };
        let blocks = self.chain_stages_with(
            first,
            |s, rows| {
                let segs: Vec<PrefillSeg<'_>> =
                    jobs.iter().map(|j| PrefillSeg { tokens: &j.prompt[j.start..j.end], pos_start: j.start as u32, gqa_window: s.caches.window(j.slot), slot: j.slot }).collect();
                match decode {
                    None => s.qwen35.run_prefill_multi_stage(&segs, &s.caches.pooled(), Some(rows)),
                    Some((tokens, slots, positions)) => {
                        let seqs = s.caches.batch(slots, positions);
                        let caches = BatchDecodeCaches { gqa_kv: &s.caches.gqa_kv, gqa_cap: s.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: Some(s.caches.gdn_pool()), memory: &[] };
                        s.qwen35.run_mixed_stage(Some((tokens, &caches)), &segs, &s.caches.pooled(), Some(rows))
                    }
                }
            },
            |_, _| {},
            |g, out| {
                wanted
                    .iter()
                    .map(|&(row0, rows)| {
                        let mut block = vec![0.0f32; rows * d];
                        if rows > 0 {
                            g.read_into(out, (row0 * d) as u64, &mut block);
                        }
                        block
                    })
                    .collect::<Vec<_>>()
            },
        );
        let mut blocks = blocks.into_iter();
        let decoded = blocks.next().expect("the decode rows come first");
        let out = blocks
            .map(|block| {
                let last = block[block.len() - d..].to_vec();
                MultiPrefillRows { last, all: if collect { block } else { Vec::new() } }
            })
            .collect();
        Ok((decoded, out))
    }

    /// Recurrent-state rows the stages hold ([`KvLayout::state_rows`]).
    fn state_rows(&self) -> u32 {
        self.kv_layout.state_rows(self.max_batch, self.prefix_slots)
    }

    /// Publish the blocks of the sequence in state row `slot` to every stage and
    /// to the drafter ([`ShardCaches::set_blocks`]).
    fn set_blocks(&mut self, slot: u32, blocks: &[u32]) {
        for s in &mut self.shards {
            s.caches.set_blocks(slot, blocks);
        }
        if let Some(m) = self.mtp.as_mut() {
            m.caches.set_blocks(slot, blocks);
        }
    }

    /// Seed the sequence in `slot` (KV window `to`) with a cached prefix of
    /// `rows` tokens: KV rows `0..rows` copied from the window `from`, and the
    /// working GDN state restored from snapshot row `snap_row` (see
    /// [`Self::prefill_hidden`] with `start > 0`).
    fn seed_from_prefix(&self, from: KvWindow<'_>, to: KvWindow<'_>, rows: u32, snap_row: u32) {
        for s in &self.shards {
            s.caches.copy_kv_rows(&s.qwen35.gpu, &self.cfg, from, to, rows);
            s.caches.restore_work(&s.qwen35.gpu, &self.cfg, snap_row);
        }
    }

    /// Resume from a cached prefix whose KV blocks the sequence already shares: only the working GDN state is
    /// restored from snapshot row `snap_row`.
    fn seed_state_from_snapshot(&self, snap_row: u32) {
        for s in &self.shards {
            s.caches.restore_work(&s.qwen35.gpu, &self.cfg, snap_row);
        }
    }

    /// Copy the first `rows` rows of physical KV block `from` into block `to`, in every stage and the MTP head: the
    /// copy-on-write of a block a cached prefix shares with a sequence that is about to write into it.
    fn copy_block_rows(&self, from: u32, to: u32, rows: u32) {
        for s in &self.shards {
            s.caches.copy_kv_rows(&s.qwen35.gpu, &self.cfg, KvWindow::Table(&[from]), KvWindow::Table(&[to]), rows);
        }
        self.mtp_copy_block_rows(from, to, rows);
    }

    /// Snapshot the working GDN state (the state after the prefill that just
    /// ran) into snapshot row `row`.
    fn snapshot_prefill_state(&self, row: u32) {
        for s in &self.shards {
            s.caches.snapshot_work(&s.qwen35.gpu, &self.cfg, row);
        }
    }

    /// Zero the working GDN state (a sequence about to be prefilled from nothing).
    fn reset_work_state(&self) {
        for s in &self.shards {
            s.caches.reset_work(&s.qwen35.gpu);
        }
    }

    /// Make slot `slot`'s pool row the working GDN state.
    fn load_work_state(&self, slot: u32) {
        for s in &self.shards {
            s.caches.load_work(&s.qwen35.gpu, &self.cfg, slot);
        }
    }

    /// Start slot `slot`'s pool row from zero state, without the working copy.
    fn zero_pool_row(&self, slot: u32) {
        for s in &self.shards {
            s.caches.zero_pool_row(&s.qwen35.gpu, &self.cfg, slot);
        }
    }

    /// Write the working GDN state into slot `slot`'s pool row (a prompt that
    /// is wholly cached needs no prefill, only its state in place).
    fn store_prefilled(&self, slot: u32) {
        for s in &self.shards {
            s.caches.store_work(&s.qwen35.gpu, &self.cfg, slot);
        }
    }

    /// Rows per prefill round on this stack: [`TENSOR_CORE_PREFILL_TOKENS`] when
    /// every stage runs the round on tensor cores, [`MAX_PREFILL_TOKENS`]
    /// otherwise.
    fn prefill_round_rows(&self) -> u32 {
        if self.shards.iter().all(|s| gpu_core::provider::cuda::tensor_core_kernels_enabled(&s.qwen35.gpu)) {
            TENSOR_CORE_PREFILL_TOKENS
        } else {
            MAX_PREFILL_TOKENS
        }
    }

    fn reset(&self) {
        for s in &self.shards {
            s.caches.reset(&s.qwen35.gpu);
        }
        // The tap slab is position-indexed exactly like the KV cache, so a new
        // sequence invalidates it the same way - by rewinding the watermark,
        // not by zeroing 104 MB nothing will read before it is overwritten.
        if let Some(t) = self.taps.borrow().as_ref() {
            t.filled.set(0);
            t.generation.set(t.generation.get() + 1);
        }
    }

    /// Opt-in (`BRAIN_QWEN35_GGUF_DEBUG=1`) per-step dump: the RMS of the
    /// residual leaving each card, and the top-5 `(token, logit, text)` the
    /// head prefers.
    ///
    /// This is the diagnostic that distinguishes the two ways a big sharded
    /// stack goes wrong, which decoded text alone cannot: a residual whose
    /// magnitude explodes or collapses across a card boundary (a plumbing
    /// bug) versus a healthy residual whose top logits are simply the wrong
    /// tokens (a weights bug). Costs nothing when unset.
    fn debug_step(&self, pos: u32, per_stage_rms: &[f32], logits: &[f32]) {
        if std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_none() {
            return;
        }
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let shown: Vec<String> =
            top.iter().take(5).map(|&(i, l)| format!("({i}, {l:.3}, {:?})", self.tok.decode(&[i as u32]))).collect();
        eprintln!("{MODEL}: pos {pos}: stage residual rms {per_stage_rms:?} | top5 {}", shown.join(" "));
    }

    /// Real generation: CHUNKED prefill of `prompt` (bounded rounds of
    /// [`MAX_PREFILL_TOKENS`] through [`Self::stack_prefill_chunk`], each
    /// continuing from the state the previous one left), then greedy/sampled
    /// per-token decode until `max_new` tokens or a stop id. `on_token` sees
    /// every generated id as it is produced (streaming), and returning `true`
    /// from it stops early.
    ///
    /// The two phases are deliberately different primitives: prefill knows all
    /// its tokens up front and so can fill a round, decode does not and cannot.
    /// The state each leaves is the same either way - `Qwen35::
    /// run_prefill_chunk_stage`'s own contract - so the first decode step
    /// continues from a chunked prefill exactly as it would from a
    /// token-by-token one.
    ///
    /// Returns the GENERATED ids only (prompt excluded) - the same contract
    /// `crate::sample::generate_kv` has.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: u32,
        temp: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
        on_token: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: empty prompt"));
        }
        let need = prompt.len() as u64 + max_new as u64;
        if need > self.cap as u64 {
            return Err(format!("{MODEL}: prompt ({}) + max_new ({max_new}) = {need} exceeds this instance's context capacity {}", prompt.len(), self.cap));
        }
        if let Some(bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            return Err(format!("{MODEL}: prompt token id {bad} is outside vocab {}", self.cfg.vocab));
        }
        self.reset();
        let mut rng = Rng::new(seed);
        let (mut logits, mut pos) = self.replay_prompt(prompt, 0)?;
        let mut out = Vec::with_capacity(max_new as usize);
        for _ in 0..max_new {
            let next = crate::sample::sample_logits(&logits, temp, top_k, top_p, &mut rng);
            if self.eos.contains(&next) {
                break;
            }
            out.push(next);
            if out.len() as u32 == max_new {
                break;
            }
            if on_token(&out) {
                break;
            }
            logits = self.stack_step(next, pos, 0)?;
            pos += 1;
        }
        Ok(out)
    }

    /// **Diagnostic**: score the SAME positions of the same continuation with
    /// both of this crate's decode tapes, so the numerical distance between
    /// them can be measured on a real checkpoint instead of argued about.
    ///
    /// Both runs share the same chunked prompt replay and the same token
    /// sequence (greedy on the decode tape), so the only difference between
    /// the two traces is which dispatch shape computed each generated
    /// position: `run_decode_batch` at one row, versus
    /// `run_prefill_chunk_stage` at one row.
    ///
    /// Why it matters: a speculative decoder verifies on the chunk tape, so
    /// wherever the two tapes' argmaxes differ, speculative and plain decoding
    /// pick different tokens for reasons that have nothing to do with the
    /// accept/reject logic. `tests/decode_step.rs` gates that crossing at
    /// `tiny()` dims; this measures it at the real one.
    pub fn tape_comparison_trace(&self, prompt: &[u32], steps: u32) -> Result<TapeTrace, String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: empty prompt"));
        }
        self.reset();
        let (mut logits, mut pos) = self.replay_prompt(prompt, 0)?;
        let (mut dec, mut ids) = (Vec::new(), Vec::new());
        for _ in 0..steps {
            dec.push(logits.clone());
            let next = crate::sample::argmax(&logits) as u32;
            ids.push(next);
            logits = self.stack_step(next, pos, 0)?;
            pos += 1;
        }

        self.reset();
        let (mut logits, mut pos) = self.replay_prompt(prompt, 0)?;
        let mut chunk = Vec::new();
        for &id in &ids {
            chunk.push(logits.clone());
            logits = self.stack_prefill_chunk(&[id], pos, 0)?;
            pos += 1;
        }
        Ok(TapeTrace { decode: dec, chunk, ids })
    }

    /// The `[vocab]` logits the model assigns at each position of `forced`
    /// when its continuation of `prompt` is TEACHER-FORCED: row `i` is what the
    /// model predicts after `prompt` and `forced[..i]`, whatever it would have
    /// sampled itself.
    ///
    /// What a numerical change to the stack (a narrower KV tier, a weight tier)
    /// is measured with: two configurations scored along the SAME token
    /// stream can be compared logit for logit, where two free-running greedy
    /// generations diverge at the first flipped argmax and stop being
    /// comparable from there on. The prompt goes through the chunked prefill
    /// tape and each forced token through the decode tape - the path a served
    /// request takes.
    pub fn teacher_forced_logits(&self, prompt: &[u32], forced: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: empty prompt"));
        }
        self.reset();
        let (mut logits, mut pos) = self.replay_prompt(prompt, 0)?;
        let mut rows = Vec::with_capacity(forced.len());
        for &id in forced {
            rows.push(logits);
            logits = self.stack_step(id, pos, 0)?;
            pos += 1;
        }
        Ok(rows)
    }

    /// Per-position teacher-forced scores of a WHOLE record in one prefill: the
    /// log-probability of every recorded token and the token the model itself
    /// picks, for positions `1..tokens.len()` (position `p` is predicted from
    /// `tokens[..p]`; nothing predicts position 0).
    ///
    /// This is [`Self::teacher_forced_logits`]'s scoring form, and it differs
    /// from it in exactly the two ways a scoring pass wants: the whole stream
    /// goes through ONE chunked prefill (which leaves the same K/V and recurrent
    /// state a token-by-token replay does) instead of a prefill followed by
    /// decode steps, and the head is projected in row batches, so eight logits
    /// rows are alive at a time instead of every position's. The math is the
    /// served path's: the same
    /// `prefill_chunk_stage`, the same final norm and int8 head
    /// ([`Self::head_logits_rows`]), and `adapter_slot` is the bank slot every
    /// row of the prefill runs under ([`LORA_SLOT_NONE`] for the base).
    pub fn teacher_forced_scores(&self, tokens: &[u32], adapter_slot: u32) -> Result<Vec<crate::score::TokenScore>, String> {
        if tokens.len() < 2 {
            return Err(format!("{MODEL}: a scored record needs at least two tokens, got {}", tokens.len()));
        }
        if tokens.len() as u64 > self.cap as u64 {
            return Err(format!("{MODEL}: a record of {} tokens exceeds this instance's context capacity {}", tokens.len(), self.cap));
        }
        let d = self.cfg.d_model as usize;
        self.reset();
        let mut hidden = Vec::with_capacity(tokens.len() * d);
        self.prefill_adapter_slot.set(adapter_slot);
        let prefilled = self.prefill_hidden(tokens, 0, tokens.len(), 0, false, Some(&mut hidden));
        self.prefill_adapter_slot.set(LORA_SLOT_NONE);
        prefilled?;
        assert_eq!(hidden.len(), tokens.len() * d, "{MODEL}: the prefill collected {} hidden values for {} rows", hidden.len() / d, tokens.len());
        let mut out = Vec::with_capacity(tokens.len() - 1);
        for (first, rows) in (0..tokens.len() - 1).step_by(SCORE_HEAD_ROWS).map(|r| (r, SCORE_HEAD_ROWS.min(tokens.len() - 1 - r))) {
            let logits = self.head_logits_rows(&hidden[first * d..(first + rows) * d], rows as u32);
            for (i, row) in logits.chunks(self.cfg.vocab as usize).enumerate() {
                let position = (first + i + 1) as u32;
                out.push(crate::score::readout(position, row, tokens[position as usize]));
            }
        }
        Ok(out)
    }

    /// [`Self::teacher_forced_logits`] for several sequences decoded TOGETHER: sequence `i` is prefilled into slot
    /// `i`, then every forced token of every sequence goes through one batched decode step per position - the
    /// path a server's decode batch takes, and the one whose projections run the batch-sized GEMV kernels (the
    /// single-sequence method only ever runs one row). Row `j` of sequence `i` is the logits after `prompts[i]` and
    /// `forced[i][..j]`; every `forced[i]` has the same length.
    pub fn teacher_forced_logits_batch(&self, prompts: &[Vec<u32>], forced: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let n = prompts.len();
        if n == 0 || n != forced.len() || forced.iter().any(|f| f.len() != forced[0].len() || f.is_empty()) || prompts.iter().any(|p| p.is_empty()) {
            return Err(format!("{MODEL}: a batched teacher-forced run needs equally long non-empty forced streams and prompts, one per sequence"));
        }
        if n > self.max_batch as usize {
            return Err(format!("{MODEL}: batch of {n} exceeds this instance's max_batch {}", self.max_batch));
        }
        self.reset();
        let mut rows: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(forced[0].len()); n];
        let mut positions: Vec<u32> = Vec::with_capacity(n);
        for (slot, prompt) in prompts.iter().enumerate() {
            let hidden = self.prefill_hidden(prompt, 0, prompt.len(), slot as u32, false, None)?;
            rows[slot].push(self.head_logits_rows(&hidden, 1));
            positions.push(prompt.len() as u32);
        }
        for step in 0..forced[0].len() - 1 {
            let tokens: Vec<u32> = forced.iter().map(|f| f[step]).collect();
            let logits = self.stack_step_batch(&tokens, &positions)?;
            for (slot, l) in logits.into_iter().enumerate() {
                rows[slot].push(l);
                positions[slot] += 1;
            }
        }
        Ok(rows)
    }

    /// **Real SPECULATIVE generation** of one sequence, greedy, across every
    /// stage - [`Self::generate`]'s sibling, not its replacement, and
    /// byte-for-byte identical to it at `temp = 0` whatever `draft` proposes.
    ///
    /// Why this is a different lever from [`Self::generate_batch`]: batching
    /// amortizes the weight stream over several INDEPENDENT sequences and does
    /// nothing for one of them, while this amortizes it over several
    /// candidate continuations of the SAME sequence. Both attack the same
    /// bound (a decode step reads every weight in the model to produce one
    /// token) and they do not compose for free - a speculative round inside a
    /// batch slot makes each slot's row count vary per step, which
    /// [`Self::stack_step_batch`]'s one-row-per-slot contract does not admit.
    ///
    /// `draft(ctx, want) -> tokens` proposes up to `want` continuations of
    /// `ctx`. Nothing is assumed about it: a wrong proposal costs a round, not
    /// correctness. Returns the generated ids and what the run cost - read
    /// [`SpecDecodeStats::accepted_per_round`] before believing any speedup,
    /// since that is the number the technique lives or dies on.
    ///
    /// Greedy only, deliberately. Lossless SAMPLED speculation needs the
    /// drafter's own per-token distribution to build the residual-rejection
    /// correction; a drafter that hands back only token ids cannot supply it,
    /// and helping ourselves to the target's distribution instead would change
    /// the sampled distribution while looking like it worked.
    ///
    /// **The per-round floor, and what it is now.** Every round verifies
    /// through the chunk path, including a round whose drafter proposed
    /// nothing, and that is deliberate: routing the `kk == 0` round to
    /// [`Self::stack_step`] would put different rounds of one generation on
    /// different tapes and forfeit the one exact real-checkpoint equivalence
    /// gate this path has (`tests/gguf_resident_spec_real.rs`, which holds the
    /// tape fixed and varies only the speculation).
    ///
    // perf-number: measured by `tests/gguf_resident_verify_cost_real.rs`
    /// That uniformity used to cost ~1.8x: a one-row verify chunk ran at 3.7
    /// tok/s against the one-row decode step's 6.6, so a drafter that never
    // perf-number: measured by `tests/gguf_resident_verify_cost_real.rs`
    /// fired made this ~0.55x plain decode rather than ~1.0x, and no drafter
    /// however good could reach past the ceiling that implied. The cause was
    /// NOT the round's setup (a re-uploaded M-RoPE table and index buffers, a
    // perf-number: measured by `tests/gguf_resident_verify_cost_real.rs`
    /// measured 0.1% of the round) but a blocking `poll_wait` at every layer
    /// boundary, which a 256-row prefill round hides behind its own device
    /// work and a 1-row verify round cannot. It is now paid only where the
    /// scratch pool that requires it is open (`model::CHUNK_ARENA_MIN_ROWS`),
    /// which no verify round reaches. The qwen35 ledger's M29 records the
    /// profile and the measured before/after;
    /// `tests/gguf_resident_verify_cost_real.rs` is the gate that keeps it from
    /// coming back.
    pub fn generate_speculative(
        &self,
        prompt: &[u32],
        max_new: u32,
        k: u32,
        draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>,
        on_token: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<(Vec<u32>, SpecDecodeStats), String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: empty prompt"));
        }
        if k == 0 {
            return Err(format!("{MODEL}: speculative decode needs k >= 1"));
        }
        // The verify chunk is `k+1` rows long and lands past the last
        // generated token, so the capacity check is one row wider than
        // `generate`'s.
        let need = prompt.len() as u64 + max_new as u64 + k as u64;
        if need > self.cap as u64 {
            return Err(format!(
                "{MODEL}: prompt ({}) + max_new ({max_new}) + speculation window ({k}) = {need} exceeds this instance's context capacity {}",
                prompt.len(),
                self.cap
            ));
        }
        if let Some(bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            return Err(format!("{MODEL}: prompt token id {bad} is outside vocab {}", self.cfg.vocab));
        }
        self.reset();
        let snaps: Vec<_> = self.shards.iter().map(|s| s.qwen35.new_gdn_snapshot()).collect();

        // All but the last prompt token; the last is the first verify row,
        // exactly as it is the first `pending` of the textbook loop.
        let mut pos = 0u32;
        if prompt.len() > 1 {
            let (_, p) = self.replay_prompt(&prompt[..prompt.len() - 1], 0)?;
            pos = p;
        }

        let mut pending = *prompt.last().expect("non-empty (checked above)");
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(max_new as usize);
        let mut stats = SpecDecodeStats::default();

        'outer: while (out.len() as u32) < max_new {
            let want = (max_new - out.len() as u32).min(k);
            let mut props = draft(&ctx, want);
            props.truncate(want as usize);
            // A drafter is untrusted input. Cut at the FIRST out-of-vocab
            // proposal rather than filtering them out: the proposals are a
            // SEQUENCE, and dropping one from the middle would silently
            // splice two unrelated continuations together and then verify
            // that, which is a wrong answer dressed as a rejected one.
            if let Some(bad) = props.iter().position(|&t| t >= self.cfg.vocab) {
                props.truncate(bad);
            }
            let kk = props.len();
            stats.rounds += 1;
            stats.proposed += kk;

            let rows: Vec<u32> = std::iter::once(pending).chain(props.iter().copied()).collect();

            // Only a round that can REJECT needs the recurrent state
            // preserved, and the copy is ~150 MB at the real shape, so it is
            // paid only when it could be used.
            if kk > 0 {
                for (s, snap) in self.shards.iter().zip(&snaps) {
                    s.qwen35.gdn_snapshot_xfer(&s.caches.view(0), snap, true);
                }
            }
            let logits = self.stack_verify_chunk(&rows, pos, 0)?;
            stats.target_forwards += 1;

            let mut accepted = 0usize;
            while accepted < kk && crate::sample::argmax(&logits[accepted]) as u32 == props[accepted] {
                accepted += 1;
            }
            let correction = crate::sample::argmax(&logits[accepted]) as u32;

            if accepted == kk {
                // Nothing was rejected, so the round's own end state is the
                // right state for every committed row: no restore, no second
                // forward. This is the path a good drafter spends its time on.
                pos += kk as u32 + 1;
            } else {
                // The recurrent layers absorbed the rejected tail and have to
                // be rewound; the GQA layers do not, their stale rows past
                // `pos` are simply never read again.
                for (s, snap) in self.shards.iter().zip(&snaps) {
                    s.qwen35.gdn_snapshot_xfer(&s.caches.view(0), snap, false);
                }
                self.stack_prefill_chunk(&rows[..=accepted], pos, 0)?;
                stats.target_forwards += 1;
                pos += accepted as u32 + 1;
            }
            stats.accepted += accepted;

            // Emit in order and stop at the first stop id: a round can produce
            // several tokens at once, and everything after an EOS in the same
            // round was never really generated.
            for tok in props.iter().take(accepted).copied().chain(std::iter::once(correction)) {
                if self.eos.contains(&tok) {
                    break 'outer;
                }
                out.push(tok);
                ctx.push(tok);
                if out.len() as u32 == max_new {
                    break 'outer;
                }
                if on_token(&out) {
                    break 'outer;
                }
            }
            pending = correction;
        }

        Ok((out, stats))
    }

    /// Encode text with the checkpoint's OWN embedded tokenizer, raw (no chat
    /// template). What a profiler needs to build a realistic prompt without
    /// going through `qwen3::chat`'s whole request parser.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// [`Self::tokenize`]'s inverse, with the same checkpoint-embedded
    /// tokenizer - what a gate comparing two decode paths needs to claim they
    /// produced the same TEXT and not merely the same ids.
    pub fn detokenize(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    /// Drain every stage's queue, so a wall clock taken around a decode region
    /// measures work that has actually FINISHED rather than work that has been
    /// enqueued.
    ///
    /// [`Self::stack_step`] already ends in a `read` per stage plus the head's
    /// logits read, so in practice the queues are drained when it returns; this
    /// makes that explicit at the region boundary instead of relying on it.
    pub fn poll_wait(&self) {
        for s in &self.shards {
            s.qwen35.gpu.poll_wait();
        }
    }

    /// The `n = 1` decode loop, profiled per kernel kind.
    ///
    /// Two separate measurements, because they answer different questions and
    /// only one of them can be trusted as a cost:
    ///
    /// * `wall_s` is `steps` decode passes on the PRODUCTION flush path,
    ///   `poll_wait`-bracketed. This is the whole-pass number, and the only one
    ///   an optimization is allowed to be judged by.
    /// * `rows` is a second, separately-driven set of passes with the backend's
    ///   timestamp path armed, which runs one compute pass per dispatch and
    ///   reads back per flush. It RANKS the kernels; its total is an upper
    ///   bound on their real cost, not the cost, because each dispatch is
    ///   drained on its own and loses the overlap it would have had in the
    ///   production submit.
    ///
    /// Both regions are preceded by a warm-up that also establishes the
    /// recurrent state: `prompt` is replayed through the real stack first, so
    /// the profiled steps are steady-state decode at a realistic position, not
    /// a cold first token. `rows` merges every stage's table (the head's card
    /// is the last stage's, so it is already included) - a per-card split would
    /// invite tuning one card's share of a pipeline that runs them in series.
    ///
    /// Panics rather than reporting zero if the prompt plus the profiled steps
    /// would run past this instance's capacity: a silently truncated profile is
    /// worse than no profile.
    pub fn profile_decode(&self, prompt: &[u32], steps: u32) -> DecodeProfile {
        assert!(steps > 0, "profile_decode: steps must be > 0");
        let need = prompt.len() as u64 + PROFILE_WARM_STEPS as u64 + 2 * steps as u64;
        assert!(
            need <= self.cap as u64,
            "profile_decode: prompt ({}) + {PROFILE_WARM_STEPS} warm-up + 2*{steps} profiled steps = {need} exceeds capacity {}",
            prompt.len(),
            self.cap
        );
        assert!(!prompt.is_empty(), "profile_decode: needs a non-empty prompt to establish decode state");

        // Warm-up: real prompt replay, through whichever tape `generate` would
        // have used (`replay_prompt`, so a profile can never warm up through a
        // path production does not take). Establishes the GDN recurrent state
        // and the GQA cache, compiles every pipeline, and leaves `pos` where a
        // real decode step would find it. On a chunked replay the DECODE tape's
        // own pipelines are compiled by the first profiled step instead, which
        // is why the production region below is measured over `steps` passes
        // rather than one.
        self.reset();
        let (_, mut pos) = self.replay_prompt(prompt, 0).expect("profile_decode warm-up");
        let last = prompt[prompt.len() - 1];
        // Steady state, not first-token: the backend records a repeating step on
        // its second sight and replays from its third, so the first few decode
        // steps pay one-off costs (an eager step, a capture and instantiation)
        // that a served token never pays again.
        for _ in 0..PROFILE_WARM_STEPS {
            self.stack_step(last, pos, 0).expect("profile_decode steady-state warm-up");
            pos += 1;
        }
        self.poll_wait();

        self.profile_regions(steps, |_| {
            self.stack_step(last, pos, 0).expect("profile_decode step");
            pos += 1;
        })
    }

    /// [`Self::profile_decode`]'s two regions over `step`, called once per
    /// profiled pass with the pass index: the production flush path timed on
    /// the wall, then the same passes with per-dispatch timestamps armed.
    fn profile_regions(&self, steps: u32, mut step: impl FnMut(u32)) -> DecodeProfile {
        // Region 1: the production path. The token fed back is `last` rather
        // than a sampled id - the cost of a decode step is its shapes, which do
        // not depend on WHICH token it is, and reusing one id keeps the profile
        // from wandering into an EOS.
        let totals0 = gpu_core::cuda_call_totals();
        let t0 = std::time::Instant::now();
        let mut step_s = Vec::with_capacity(steps as usize);
        for i in 0..steps {
            let ts = std::time::Instant::now();
            step(i);
            step_s.push(ts.elapsed().as_secs_f64());
        }
        self.poll_wait();
        let wall_s = t0.elapsed().as_secs_f64();
        let totals1 = gpu_core::cuda_call_totals();

        // Region 2: the same work with timestamp queries armed. A backend that
        // cannot time kernels reports no rows rather than a table of zeros.
        let timed = self.shards.iter().all(|s| s.qwen35.gpu.set_kernel_timing(true));
        for s in &self.shards {
            s.qwen35.gpu.reset_kernel_times();
        }
        let t1 = std::time::Instant::now();
        for i in steps..2 * steps {
            step(i);
        }
        self.poll_wait();
        let timed_wall_s = t1.elapsed().as_secs_f64();

        let mut merged: std::collections::BTreeMap<String, (f64, u64)> = Default::default();
        if timed {
            for s in &self.shards {
                for (name, ms, calls) in s.qwen35.gpu.kernel_times().unwrap_or_default() {
                    let e = merged.entry(name).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += calls;
                }
            }
        }
        for s in &self.shards {
            s.qwen35.gpu.set_kernel_timing(false);
        }

        let mut rows: Vec<(String, f64, u64)> = merged.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        DecodeProfile {
            steps,
            wall_s,
            step_s,
            alloc_calls: totals1.device_alloc_calls - totals0.device_alloc_calls,
            host_launches: totals1.host_launches - totals0.host_launches,
            graph_replays: totals1.graph_replays - totals0.graph_replays,
            timed_wall_s,
            rows,
        }
    }

    /// [`Self::profile_decode`] for a BATCH decoding at a SYNTHETIC context:
    /// `batch` rows, every one at absolute position `position + pass`, over
    /// whatever the cache rows hold ([`Self::decode_batch_at`]'s contract).
    ///
    /// The device time per kernel is what a loaded host cannot distort: the
    /// wall clock of a decode pass includes the host recording ~1250
    /// dispatches, which another process's CPU load stretches, while a kernel's
    /// own timestamp pair does not move. This is how a long-context decode step
    /// is priced when prefilling the context for real would take hours.
    pub fn profile_decode_batch_at(&self, batch: u32, position: u32, steps: u32) -> Result<DecodeProfile, String> {
        self.profile_batch_at(batch, position, steps, |tokens, positions| self.decode_batch_at(tokens, positions).map(|_| ()))
    }

    /// [`Self::profile_decode_batch_at`] for the tape a served GREEDY request
    /// runs: the layer stack followed by the device head
    /// ([`Self::decode_batch_greedy_at`]), so no `[bsz, vocab]` block is read
    /// back and the measurement is the production step, not the
    /// logits-returning sibling's.
    pub fn profile_decode_batch_greedy_at(&self, batch: u32, position: u32, steps: u32) -> Result<DecodeProfile, String> {
        self.profile_batch_at(batch, position, steps, |tokens, positions| self.decode_batch_greedy_at(tokens, positions).map(|_| ()))
    }

    /// The shared body of the two batch-at profiles: validate, one unrecorded
    /// pass through `pass` (pipeline compilation is not decode time), then the
    /// two measured regions.
    fn profile_batch_at(&self, batch: u32, position: u32, steps: u32, mut pass: impl FnMut(&[u32], &[u32]) -> Result<(), String>) -> Result<DecodeProfile, String> {
        if batch == 0 || steps == 0 {
            return Err(format!("{MODEL}: a batch profile needs a batch and a step count (got {batch} x {steps})"));
        }
        let last = position as u64 + 2 * steps as u64;
        if last >= self.cap as u64 {
            return Err(format!("{MODEL}: profiling {steps} passes from position {position} runs to {last}, past this instance's capacity {}", self.cap));
        }
        let tokens = vec![1u32; batch as usize];
        // One unrecorded pass first: pipeline compilation is not decode time.
        pass(&tokens, &vec![position; batch as usize])?;
        self.poll_wait();
        let mut failure = None;
        let profile = self.profile_regions(steps, |i| {
            if failure.is_none() {
                if let Err(e) = pass(&tokens, &vec![position + 1 + i; batch as usize]) {
                    failure = Some(e);
                }
            }
        });
        match failure {
            Some(e) => Err(e),
            None => Ok(profile),
        }
    }

    /// **The head alone, priced separately** - the epilogue
    /// [`Self::profile_decode_batch_at`]'s table cannot split out, because the
    /// head projection is the same `matmul_i8_gemv`/`_mma` kernel the layer
    /// linears dispatch and merges into their rows.
    ///
    /// Runs one real decode pass to obtain the batch's `[bsz, d_model]`
    /// pre-norm hidden block, then times the production head for `steps`
    /// passes: the final RMSNorm, the `[bsz, d_model] x [d_model, vocab]`
    /// projection and the greedy reduction (`Self::head_argmax_rows`, exactly
    /// what `forward_batched_greedy` runs). One head pass is one decode step's
    /// head cost, so `wall_s / steps` is already that step's head time.
    ///
    /// The table is the last stage's alone - the head lives on the last card
    /// and is the only work this region runs - so its total is that card's
    /// head device time, not the model's.
    pub fn profile_head_at(&self, batch: u32, position: u32, steps: u32) -> Result<DecodeProfile, String> {
        if batch == 0 || steps == 0 {
            return Err(format!("{MODEL}: profile_head_at needs a batch and a step count (got {batch} x {steps})"));
        }
        if position as u64 >= self.cap as u64 {
            return Err(format!("{MODEL}: profile_head_at: position {position} is beyond this instance's capacity {}", self.cap));
        }
        let slots: Vec<u32> = (0..batch).collect();
        let tokens = vec![1u32; batch as usize];
        // The pass's own hidden block: a real decode step's output, so the head
        // is priced at the shape and magnitudes production gives it.
        let hidden = self.stack_decode_hidden(&tokens, &slots, &vec![position; batch as usize])?;
        let head = self.shards.last().expect("a plan always has at least one stage");
        let g = &head.qwen35.gpu;
        // One unrecorded pass first: pipeline compilation is not head time.
        self.head_argmax_rows(&hidden, batch);
        self.poll_wait();

        let totals0 = gpu_core::cuda_call_totals();
        let t0 = std::time::Instant::now();
        let mut step_s = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            let ts = std::time::Instant::now();
            self.head_argmax_rows(&hidden, batch);
            step_s.push(ts.elapsed().as_secs_f64());
        }
        self.poll_wait();
        let wall_s = t0.elapsed().as_secs_f64();
        let totals1 = gpu_core::cuda_call_totals();

        let timed = g.set_kernel_timing(true);
        g.reset_kernel_times();
        let t1 = std::time::Instant::now();
        for _ in 0..steps {
            self.head_argmax_rows(&hidden, batch);
        }
        self.poll_wait();
        let timed_wall_s = t1.elapsed().as_secs_f64();
        let mut rows: Vec<(String, f64, u64)> = if timed { g.kernel_times().unwrap_or_default() } else { Vec::new() };
        g.set_kernel_timing(false);
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));

        Ok(DecodeProfile {
            steps,
            wall_s,
            step_s,
            alloc_calls: totals1.device_alloc_calls - totals0.device_alloc_calls,
            host_launches: totals1.host_launches - totals0.host_launches,
            graph_replays: totals1.graph_replays - totals0.graph_replays,
            timed_wall_s,
            rows,
        })
    }

    /// **The CHUNK tape at a chosen row count**, profiled the same way
    /// [`Self::profile_decode`] profiles the decode tape - and the measurement
    /// a speculative decoder's per-round floor is read off.
    ///
    /// A verify round is structurally a prefill round
    /// ([`Self::stack_chunk_carry`]), so the cost this reports at `rows = 1..8`
    /// IS the cost `generate_speculative` pays per round before any draft
    /// token is accepted, and the cost at `rows = 256` is the one chunked
    /// prefill was tuned for. Reporting both from one entry point is the point:
    /// a fixed per-round cost is only visible as the RATIO between them.
    ///
    /// The two halves are timed apart because they scale differently and a
    /// merged figure hides which one moved:
    /// * `carry_s` - the layer stack over every stage
    ///   ([`Self::stack_chunk_carry`]), where a per-round fixed cost lives;
    /// * `head_s` - the `[rows, vocab]` INT8 head projection, which is real
    ///   work proportional to `rows` and is not a fixed cost at all.
    ///
    /// `prompt` establishes a realistic recurrent state and KV depth first, so
    /// the profiled rounds are steady-state rather than cold. Panics rather
    /// than truncating if the prompt plus the profiled rounds run past
    /// capacity.
    pub fn profile_chunk_round(&self, prompt: &[u32], rows: u32, rounds: u32) -> ChunkRoundProfile {
        assert!(rows > 0 && rounds > 0, "profile_chunk_round: rows and rounds must be > 0");
        assert!(!prompt.is_empty(), "profile_chunk_round: needs a non-empty prompt to establish decode state");
        let need = prompt.len() as u64 + 2 * rows as u64 * rounds as u64;
        assert!(need <= self.cap as u64, "profile_chunk_round: prompt ({}) + 2*{rounds} rounds of {rows} = {need} exceeds capacity {}", prompt.len(), self.cap);

        self.reset();
        let (_, mut pos) = self.replay_prompt(prompt, 0).expect("profile_chunk_round warm-up");
        let last = prompt[prompt.len() - 1];
        let block: Vec<u32> = vec![last; rows as usize];
        self.poll_wait();

        let d = self.cfg.d_model as usize;
        let head = self.shards.last().expect("a plan always has at least one stage");
        let (mut carry_s, mut head_s) = (0.0f64, 0.0f64);
        let t0 = std::time::Instant::now();
        for _ in 0..rounds {
            let tc = std::time::Instant::now();
            let (hidden, _) = self.stack_chunk_carry(&block, pos, 0).expect("profile_chunk_round carry");
            carry_s += tc.elapsed().as_secs_f64();
            let th = std::time::Instant::now();
            let flat = crate::stream::head_logits_rows_on(&head.qwen35.gpu, &self.head.ops, &self.cfg, &self.head.norm, &self.head.w, &hidden, rows);
            head_s += th.elapsed().as_secs_f64();
            std::hint::black_box(flat.len() + hidden.len() / d);
            pos += rows;
        }
        self.poll_wait();
        let wall_s = t0.elapsed().as_secs_f64();

        // The kernel table, from a second set of rounds with timestamps armed -
        // same split of concerns `profile_decode` documents: this RANKS the
        // kernels, the wall clock above is the cost.
        let timed = self.shards.iter().all(|s| s.qwen35.gpu.set_kernel_timing(true));
        for s in &self.shards {
            s.qwen35.gpu.reset_kernel_times();
        }
        for _ in 0..rounds {
            self.stack_chunk_carry(&block, pos, 0).expect("profile_chunk_round timed region");
            pos += rows;
        }
        self.poll_wait();
        let mut merged: std::collections::BTreeMap<String, (f64, u64)> = Default::default();
        if timed {
            for s in &self.shards {
                for (name, ms, calls) in s.qwen35.gpu.kernel_times().unwrap_or_default() {
                    let e = merged.entry(name).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += calls;
                }
            }
        }
        for s in &self.shards {
            s.qwen35.gpu.set_kernel_timing(false);
        }
        let mut table: Vec<(String, f64, u64)> = merged.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
        table.sort_by(|a, b| b.1.total_cmp(&a.1));
        ChunkRoundProfile { rows, rounds, wall_s, carry_s, head_s, table }
    }
}

impl Qwen35GgufInstance {
    /// **Prefill rounds, timed one at a time** - the measurement a shared
    /// device needs.
    ///
    /// [`Self::profile_chunk_round`] sums its rounds, which is the right shape
    /// for a quiet card and the wrong one for a card other processes are
    /// time-slicing: interference only ever ADDS time, so the honest figure is
    /// the fastest round, not the mean. This runs `rounds` real chunk rounds
    /// (layer stack + head, the production path) after warming up on `prompt`,
    /// timing each, then runs `rounds` more with per-kernel device timing
    /// armed (the layer stack only), one table per round.
    /// [`PrefillRoundsProfile`] reports the best round of each region.
    ///
    /// Panics rather than truncating if the prompt plus the profiled rounds run
    /// past capacity.
    pub fn profile_prefill_rounds(&self, prompt: &[u32], rows: u32, rounds: u32) -> PrefillRoundsProfile {
        assert!(rows > 0 && rounds > 0, "profile_prefill_rounds: rows and rounds must be > 0");
        assert!(!prompt.is_empty(), "profile_prefill_rounds: needs a non-empty prompt to establish decode state");
        let need = prompt.len() as u64 + 2 * rows as u64 * rounds as u64;
        assert!(need <= self.cap as u64, "profile_prefill_rounds: prompt ({}) + 2*{rounds} rounds of {rows} = {need} exceeds capacity {}", prompt.len(), self.cap);

        self.reset();
        let (_, mut pos) = self.replay_prompt(prompt, 0).expect("profile_prefill_rounds warm-up");
        let block: Vec<u32> = vec![prompt[prompt.len() - 1]; rows as usize];
        self.poll_wait();

        // The production round: the layer stack plus the LAST row's logits, exactly
        // what `replay_prompt` runs per round.
        let mut round_s = Vec::with_capacity(rounds as usize);
        for _ in 0..rounds {
            let t = std::time::Instant::now();
            let logits = self.stack_prefill_chunk(&block, pos, 0).expect("profile_prefill_rounds round");
            self.poll_wait();
            std::hint::black_box(logits.len());
            round_s.push(t.elapsed().as_secs_f64());
            pos += rows;
        }

        // `BRAIN_PREFILL_PROFILE_SKIP_TABLE=1` stops after the production rounds -
        // for attaching a sampling profiler to them without the kernel timers.
        if std::env::var_os("BRAIN_PREFILL_PROFILE_SKIP_TABLE").is_some() {
            return PrefillRoundsProfile { rows, round_s, table: Vec::new() };
        }
        let timed = self.shards.iter().all(|s| s.qwen35.gpu.set_kernel_timing(true));
        let mut best: Option<(f64, KernelTable)> = None;
        for _ in 0..rounds {
            for s in &self.shards {
                s.qwen35.gpu.reset_kernel_times();
            }
            self.stack_chunk_carry(&block, pos, 0).expect("profile_prefill_rounds timed round");
            self.poll_wait();
            pos += rows;
            if !timed {
                break;
            }
            let mut merged: std::collections::BTreeMap<String, (f64, u64)> = Default::default();
            for s in &self.shards {
                for (name, ms, calls) in s.qwen35.gpu.kernel_times().unwrap_or_default() {
                    let e = merged.entry(name).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += calls;
                }
            }
            let mut table: KernelTable = merged.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
            table.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = table.iter().map(|r| r.1).sum();
            if best.as_ref().is_none_or(|(t, _)| total < *t) {
                best = Some((total, table));
            }
        }
        for s in &self.shards {
            s.qwen35.gpu.set_kernel_timing(false);
        }
        PrefillRoundsProfile { rows, round_s, table: best.map(|b| b.1).unwrap_or_default() }
    }
}

/// `(kernel, device ms, calls)` rows, descending by time.
type KernelTable = Vec<(String, f64, u64)>;

/// What [`Qwen35GgufInstance::profile_prefill_rounds`] measured.
#[derive(Clone, Debug)]
pub struct PrefillRoundsProfile {
    /// Rows in each round.
    pub rows: u32,
    /// Wall seconds of each production round (layer stack + last-row logits),
    /// in run order.
    pub round_s: Vec<f64>,
    /// `(kernel, device ms, calls)` of the layer stack for the round with the
    /// least total device time, descending by time. Empty when the backend
    /// cannot time kernels.
    pub table: KernelTable,
}

impl PrefillRoundsProfile {
    /// Seconds of the fastest round.
    pub fn best_round_s(&self) -> f64 {
        self.round_s.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Prompt tokens per second of the fastest round.
    pub fn best_tok_per_s(&self) -> f64 {
        self.rows as f64 / self.best_round_s()
    }

    /// Summed device time of the table, in ms.
    pub fn device_ms(&self) -> f64 {
        self.table.iter().map(|(_, ms, _)| ms).sum()
    }
}

impl Qwen35GgufInstance {
    /// **What the GDN state snapshot costs**, in ms per round - the second
    /// structural cost `generate_speculative` pays on every round that proposes
    /// anything, and one that is easy to over-estimate by reading its BYTE
    /// count (a `[state + conv history]` copy per recurrent layer, hundreds of
    /// MB across both cards) rather than measuring it.
    ///
    /// It has to be measured with an explicit drain, not by timing the call:
    /// `Qwen35::gdn_snapshot_xfer` only RECORDS its `region_copy` dispatches
    /// (this backend's `submit` appends to a pending list), so a wall clock
    /// around it reads near zero and says nothing about the device time it
    /// adds to the round it is part of. `reps` snapshots are recorded and
    /// drained together, so the figure is the device cost amortized the way a
    /// real round's is - never a per-call fence latency.
    pub fn profile_gdn_snapshot(&self, reps: u32) -> f64 {
        assert!(reps > 0, "profile_gdn_snapshot: reps must be > 0");
        let snaps: Vec<_> = self.shards.iter().map(|s| s.qwen35.new_gdn_snapshot()).collect();
        // Warm-up: the first copy of each shape compiles its pipeline.
        for (s, snap) in self.shards.iter().zip(&snaps) {
            s.qwen35.gdn_snapshot_xfer(&s.caches.view(0), snap, true);
        }
        self.poll_wait();

        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (s, snap) in self.shards.iter().zip(&snaps) {
                s.qwen35.gdn_snapshot_xfer(&s.caches.view(0), snap, true);
            }
        }
        self.poll_wait();
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    }
}

/// What [`Qwen35GgufInstance::profile_chunk_round`] measured.
#[derive(Clone, Debug)]
pub struct ChunkRoundProfile {
    /// Rows in each profiled round.
    pub rows: u32,
    /// Rounds in each measured region.
    pub rounds: u32,
    /// Wall seconds for `rounds` whole rounds (layer stack + head).
    pub wall_s: f64,
    /// Of `wall_s`, the layer stack's share - where a per-round fixed cost lives.
    pub carry_s: f64,
    /// Of `wall_s`, the `[rows, vocab]` head projection's share.
    pub head_s: f64,
    /// `(kernel, device ms, calls)` over every stage for the LAYER half,
    /// descending by time. Empty when the backend cannot time kernels.
    pub table: Vec<(String, f64, u64)>,
}

impl ChunkRoundProfile {
    /// Milliseconds per whole round.
    pub fn round_ms(&self) -> f64 {
        self.wall_s * 1e3 / self.rounds as f64
    }

    /// Milliseconds per round spent in the layer stack.
    pub fn carry_ms(&self) -> f64 {
        self.carry_s * 1e3 / self.rounds as f64
    }

    /// Milliseconds per round spent in the head projection.
    pub fn head_ms(&self) -> f64 {
        self.head_s * 1e3 / self.rounds as f64
    }

    /// Milliseconds of layer-stack cost per ROW - the number that exposes a
    /// fixed per-round cost, because real work is flat in it and a fixed cost
    /// is not.
    pub fn carry_ms_per_row(&self) -> f64 {
        self.carry_ms() / self.rows as f64
    }

    /// Summed device time across the layer-half table, in ms.
    pub fn device_ms(&self) -> f64 {
        self.table.iter().map(|(_, ms, _)| ms).sum()
    }
}

/// What [`Qwen35GgufInstance::profile_decode`] measured. See that method's doc
/// for why the wall clock and the table are kept apart.
#[derive(Clone, Debug)]
pub struct DecodeProfile {
    /// Decode passes in each measured region.
    pub steps: u32,
    /// Wall seconds for `steps` passes on the production flush path - the
    /// whole-pass number an optimization is judged by.
    pub wall_s: f64,
    /// Wall seconds of each of those passes, in order. The host is shared and
    /// the card often is, so one mean hides the spread; the minimum is the best
    /// the code can do, the median what it usually does.
    pub step_s: Vec<f64>,
    /// `cuMemAlloc` calls, kernel launches made one at a time, and graph
    /// replays across those passes. A steady-state token should show no
    /// allocation, no individual launch and one replay.
    pub alloc_calls: u64,
    pub host_launches: u64,
    pub graph_replays: u64,
    /// Wall seconds for the same passes with per-dispatch timestamps armed.
    /// Always larger; reported so the inflation is visible rather than implied.
    pub timed_wall_s: f64,
    /// `(kernel, device ms, calls)` over every stage, descending by time.
    /// Empty when the backend cannot time individual kernels.
    pub rows: Vec<(String, f64, u64)>,
}

impl DecodeProfile {
    /// Whole-pass decode rate, tokens per second.
    pub fn tok_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.steps as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Wall milliseconds per token at quantile `q` (0 = best, 0.5 = median).
    pub fn step_ms_at(&self, q: f64) -> f64 {
        let mut v = self.step_s.clone();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return 0.0;
        }
        1e3 * v[((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize]
    }

    /// Summed device time across the table, in ms.
    pub fn device_ms(&self) -> f64 {
        self.rows.iter().map(|(_, ms, _)| ms).sum()
    }
}




/// Decode steps run before the measured region of
/// [`Qwen35GgufInstance::profile_decode`] so it measures steady state.
const PROFILE_WARM_STEPS: u32 = 3;

// ------------------------------------------------------------ the resident

/// The placement this resident committed to, computed once from the GGUF
/// header alone (no GPU, no tensor data) and reused by BOTH `estimate_multi`
/// and `activate_multi` - so the bytes the scheduler reserves and the bytes
/// the loader places can never describe different cards.
#[derive(Clone, Debug, Default)]
struct Plan {
    /// Why nothing was placed, when the configuration is the reason
    /// ([`sizing_refusal`]); reported to every request through
    /// [`MultiDeviceResidentModel::unavailable_reason`].
    refusal: Option<String>,
    cfg: Option<Qwen35Config>,
    /// `(device, that stage's shard, that stage's bytes)`.
    stages: Vec<(Device, Shard, u64)>,
}

/// The [`ResidentModel`] / [`MultiDeviceResidentModel`] adapter. See this
/// module's doc for what it composes.
pub struct Qwen35GgufResident {
    gguf_path: String,
    /// Candidate devices with each one's USABLE bytes - a real number the
    /// caller queried, not an assumption. Capacity travels with identity
    /// because the split has to RESPECT it (see
    /// `model::shard::plan_by_capacity`).
    devices: Vec<(Device, u64)>,
    cap: u32,
    /// The per-leaf storage tier the decoder body uploads at (M24) - the
    /// `lm_head`/embedding endpoints are unaffected (INT8 and row-mmap-read
    /// respectively, always, regardless of `tier` - see [`head_bytes`]).
    /// Feeds both [`layer_cost`] (so a placement is honest about what a
    /// narrower tier actually saves) and [`Self::activate_owned`]'s
    /// [`Qwen35::new_shard_dt`] call (so the plan and the load agree on what
    /// was budgeted).
    tier: TierPolicy,
    /// How many sequences one [`Instance::run_batch`] may decode together -
    /// [`Self::with_max_batch`]. Charged into [`layer_cost`], so the planner
    /// budgets the slots it will actually allocate.
    max_batch: u32,
    /// Cached prompts kept for prefix reuse - [`Self::with_prefix_slots`].
    prefix_slots: u32,
    /// How the GQA layers' K/V are stored - [`Self::with_kv_tier`]. Charged
    /// into [`layer_cost`] at the bytes that tier really occupies.
    kv_tier: KvTier,
    /// Rows per KV block when the attention addresses each sequence's KV through a
    /// block table ([`Self::with_kv_block_rows`]); 0 keeps one block per sequence.
    kv_block_rows: u32,
    /// Rows of the shared KV pool of the paged layout ([`Self::with_kv_pool_rows`]);
    /// 0 sizes it as the one-block layout's `(max_batch + prefix_slots) * cap`.
    kv_pool_rows: u32,
    /// The id this resident is SERVED under (`Manifest::model`,
    /// `InstanceKey::model`) - set by the caller that resolved the
    /// checkpoint ([`Self::with_model_id`], what
    /// `catalog::resident_qwen35::gguf_resident` does with the artifact's
    /// own store-relative `<vendor>/<repo>`). Defaults to [`MODEL`], the
    /// well-known release id, for a caller that resolved nothing to derive
    /// from.
    model_id: String,
    /// The adapter bank this resident builds beside its weights, when the
    /// operator asked for adapters ([`Self::with_adapter_bank`]). Charged to
    /// the stages that hold the layers it corrects, and enabled on the engine
    /// [`Self::activate_engine`] returns.
    bank: Option<AdapterBankRequest>,
    /// Speculative decoding - [`Self::with_speculation`].
    spec: SpecServing,
    /// Run every decode row independently of its co-batch -
    /// [`Self::with_exact_decode`].
    exact_decode: bool,
    /// The activation stream's element type - [`Self::with_act_choice`].
    act_choice: gpu_core::ActChoice,
    /// How the engine spends prefill - [`Self::with_prefill_policy`].
    prefill: PrefillPolicy,
    plan: OnceLock<Plan>,
    /// The sampling defaults the GGUF declares, read once (see [`Self::sampling_defaults`]).
    sampling: OnceLock<data::generation::SamplingDefaults>,
}

impl Qwen35GgufResident {
    pub fn new(gguf_path: String, devices: Vec<(Device, u64)>, cap: u32, tier: TierPolicy) -> Qwen35GgufResident {
        Qwen35GgufResident { gguf_path, devices, cap: cap.max(1), tier, max_batch: 1, prefix_slots: 0, kv_tier: KvTier::F32, kv_block_rows: 0, kv_pool_rows: 0, model_id: MODEL.to_string(), bank: None, spec: SpecServing::default(), exact_decode: false, act_choice: gpu_core::ActChoice::Auto, prefill: PrefillPolicy::default(), plan: OnceLock::new(), sampling: OnceLock::new() }
    }

    /// **Run every decode row independently of its co-batch** (`--exact-decode`):
    /// each stage groups the batch into rows whose int8 projections are the
    /// bit-identical kernel, sizes every row's attention from this resident's
    /// per-sequence context, and the head projects in the same groups - so a
    /// request's tokens and per-token logits are the same alone as beside any
    /// batch of neighbours. Speculation is turned off in this regime (its
    /// verify pass is not grouped). Costs one extra pass over the weights per
    /// group: two at `max_batch` 8 on a device with int8 tensor cores. Must be
    /// set before the first `estimate_multi` call.
    pub fn with_exact_decode(mut self, on: bool) -> Qwen35GgufResident {
        self.exact_decode = on;
        self
    }

    /// **Choose the element type of the activation stream between the FP8 products** (`--act-precision`, see
    /// [`gpu_core::ActPrecision`]): `Auto` is bf16 on a device with bf16 tensor cores, f32 elsewhere; a forced `F32` is the
    /// A/B switch. The choice is resolved per card when the model is placed.
    pub fn with_act_choice(mut self, choice: gpu_core::ActChoice) -> Qwen35GgufResident {
        self.act_choice = choice;
        self
    }

    /// How the serving engine spends prefill between decode steps - the slice one
    /// sequence gets and the rows one forward carries across sequences.
    pub fn with_prefill_policy(mut self, policy: PrefillPolicy) -> Qwen35GgufResident {
        self.prefill = policy;
        self
    }

    /// Serve this resident under `id` instead of [`MODEL`] - what the
    /// resolver-derived id needs when the configured artifact is not the
    /// well-known release file (a different quant, a different fork of the
    /// same architecture): two checkpoints on disk are two models, each
    /// addressable under the name its own store position carries.
    /// Must be set before the first `manifest()`/`instance_key()` call.
    pub fn with_model_id(mut self, id: impl Into<String>) -> Qwen35GgufResident {
        self.model_id = id.into();
        self
    }

    /// Build an adapter bank of `request` beside the weights, so sequences can
    /// run under LoRA adapters ([`Qwen35GgufEngine::load_adapter`]). Its device
    /// bytes are part of the placement. Must be set before the first
    /// `estimate_multi` call.
    pub fn with_adapter_bank(mut self, request: AdapterBankRequest) -> Qwen35GgufResident {
        self.bank = Some(request);
        self
    }

    /// The id this resident is served under - [`Self::with_model_id`]'s
    /// value, [`MODEL`] by default.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The per-sequence `prompt + max_new` ceiling - the constructor's
    /// `cap`, floored at 1.
    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// The configured checkpoint path - the artifact this resident loads.
    /// The sampling settings the GGUF declares as its defaults (`general.sampling.*`), read once. A file that
    /// cannot be opened or declares a value out of range yields none, with the reason logged: the file is
    /// reported against the real open at activation, and the action's own constants apply meanwhile.
    pub fn sampling_defaults(&self) -> &data::generation::SamplingDefaults {
        self.sampling.get_or_init(|| {
            let declared = ResidentCheckpoint::open(&self.gguf_path).map_err(|e| format!("cannot open '{}': {e}", self.gguf_path)).and_then(|c| c.sampling_defaults());
            declared.unwrap_or_else(|e| {
                residency::log::warn(&format!("{MODEL}: ignoring the checkpoint's sampling defaults: {e}"));
                data::generation::SamplingDefaults::default()
            })
        })
    }

    pub fn gguf_path(&self) -> &str {
        &self.gguf_path
    }

    /// How many sequences one [`Instance::run_batch`] call may decode
    /// together. Default `1`, which makes an instance byte-identical to an
    /// unbatched one.
    ///
    /// This is charged into the placement, not taken out of thin air: every
    /// stage allocates `n` slots' worth of KV pool and GDN state, so
    /// [`layer_cost`] multiplies the per-layer decode-state bytes by it and
    /// the planner sees the real footprint. Raising it makes the model need
    /// more cards, honestly, rather than silently overrunning one - exactly
    /// the rule `cap` already follows.
    ///
    /// Must be set BEFORE the first `estimate_multi`/`activate_multi` call
    /// (the plan is computed once and cached).
    pub fn with_max_batch(mut self, n: u32) -> Qwen35GgufResident {
        self.max_batch = n.max(1);
        self
    }

    /// How many sequences one batch may carry - [`Self::with_max_batch`].
    pub fn max_batch(&self) -> u32 {
        self.max_batch
    }

    /// Keep up to `n` prompts' KV rows and recurrent-state snapshots so a later
    /// prompt that starts with one prefills only its new tokens (0, the default,
    /// is off). Each cached prompt holds a whole sequence slot, charged into the
    /// placement like `max_batch`'s. Must be set BEFORE the first
    /// `estimate_multi`/`activate_multi` call.
    pub fn with_prefix_slots(mut self, n: u32) -> Qwen35GgufResident {
        self.prefix_slots = n;
        self
    }

    /// Address every sequence's KV through a block table of `rows`-row blocks
    /// instead of one block per sequence (0, the default), allocated from one
    /// shared pool as the sequence grows and shared with the prefix cache: a
    /// sequence holds the blocks its tokens need, not a whole `cap`-row window.
    /// The pool is as large as the one-block layout's memory unless
    /// [`Self::with_kv_pool_rows`] says otherwise. Must be set BEFORE the first
    /// `estimate_multi`/`activate_multi` call.
    pub fn with_kv_block_rows(mut self, rows: u32) -> Qwen35GgufResident {
        self.kv_block_rows = rows;
        self
    }

    /// Rows of the paged layout's shared KV pool (0, the default:
    /// `(max_batch + prefix_slots) * cap`, the memory of the one-block layout).
    /// Smaller pools serve more sequences at their real lengths; the scheduler
    /// admits by free blocks and preempts when they run out. Only meaningful
    /// with [`Self::with_kv_block_rows`].
    pub fn with_kv_pool_rows(mut self, rows: u32) -> Qwen35GgufResident {
        self.kv_pool_rows = rows;
        self
    }

    /// How the stages address a sequence's K/V rows.
    fn kv_layout(&self) -> KvLayout {
        match self.kv_block_rows {
            0 => KvLayout::OneBlock,
            block_rows => {
                // The default pool backs every sequence and cached prompt at full capacity, whole blocks each.
                let default_rows = (self.max_batch + self.prefix_slots).saturating_mul(self.cap.div_ceil(block_rows)).saturating_mul(block_rows);
                let rows = if self.kv_pool_rows > 0 { self.kv_pool_rows } else { default_rows };
                KvLayout::Paged { block_rows, pool_rows: rows.div_ceil(block_rows).saturating_mul(block_rows) }
            }
        }
    }

    /// Rows of the paged pool for the sizing functions; 0 for the one-block layout.
    fn paged_pool_rows(&self) -> u32 {
        match self.kv_layout() {
            KvLayout::OneBlock => 0,
            KvLayout::Paged { pool_rows, .. } => pool_rows,
        }
    }

    /// See [`Self::with_prefix_slots`].
    pub fn prefix_slots(&self) -> u32 {
        self.prefix_slots
    }

    /// Store the GQA layers' K and V in `kv` instead of `f32` (the default).
    /// `bf16` halves the resident cache and `int8` (one scale per (token,
    /// kv-head) row) quarters it, at the cost of rounding each cached element
    /// once as it is written; the attention arithmetic stays `f32`. The
    /// recurrent state of the GDN layers is not part of the cache and stays
    /// `f32`.
    ///
    /// Charged into the placement ([`layer_cost`]), so a compact tier lets more
    /// sequences (or a longer context) fit on the same cards. Must be set
    /// BEFORE the first `estimate_multi`/`activate_multi` call.
    pub fn with_kv_tier(mut self, kv: KvTier) -> Qwen35GgufResident {
        self.kv_tier = kv;
        self
    }

    /// How the GQA layers' K/V are stored - [`Self::with_kv_tier`].
    pub fn kv_tier(&self) -> KvTier {
        self.kv_tier
    }

    /// The placement, computed once. Returns a plan naming ZERO devices -
    /// never a panic - when there are no candidate devices, the GGUF cannot
    /// be opened or understood, or the model does not fit across the devices
    /// given. That is [`MultiDeviceResidentModel::estimate_multi`]'s
    /// documented "unavailable" signal, which `ResidencyManager::claim_multi`
    /// turns into a clean per-job error instead of a dispatcher crash.
    /// [`Self::unservable_reason`] says which of those it was.
    fn plan(&self) -> &Plan {
        // `plan_uncached` is a pure function of `self`, so which racer's value
        // is kept cannot matter.
        self.plan.get_or_init(|| self.plan_uncached())
    }

    /// Why this resident can never be loaded as configured, or `None` when it
    /// can: no card to place on, a GGUF that cannot be read or is not a
    /// Qwen3.8, weights that do not fit the cards, or no embedded vocabulary
    /// to tokenize with. A caller that lists models checks this first, so a
    /// listed model is one whose first request can actually load.
    pub fn unservable_reason(&self) -> Option<String> {
        if let Some(reason) = &self.plan().refusal {
            return Some(reason.clone());
        }
        match ResidentCheckpoint::open(&self.gguf_path) {
            Ok(c) if !c.has_tokenizer() => Some(format!("'{}' carries no tokenizer (tokenizer.ggml.* KV, or a sibling tokenizer.json)", self.gguf_path)),
            Ok(_) => None,
            Err(e) => Some(format!("cannot open '{}': {e}", self.gguf_path)),
        }
    }

    fn plan_uncached(&self) -> Plan {
        let refused = |why: String| Plan { refusal: Some(format!("{MODEL}: {why}")), ..Plan::default() };
        if self.devices.is_empty() {
            return refused("no GPU is budgeted for it (its weights are device buffers; there is no CPU path)".to_string());
        }
        let mg = match ResidentCheckpoint::open(&self.gguf_path) {
            Ok(mg) => mg,
            Err(e) => return refused(format!("cannot open '{}': {e}", self.gguf_path)),
        };
        let cfg = match mg.config(self.cap) {
            Ok(c) => c,
            Err(e) => return refused(format!("'{}' is not a servable Qwen3.8 checkpoint: {e}", self.gguf_path)),
        };
        // `plan_fewest_devices` wants `(index into self.devices, capacity)`;
        // mapping back afterwards is what makes a non-GPU device in the list
        // (which this model cannot use) rejected rather than mis-indexed.
        let caps = gpu_capacities(&self.devices, 0);
        if caps.is_empty() {
            return refused("no GPU is budgeted for it (this model is GPU-only)".to_string());
        }
        let shape = self.serving_shape(&mg, &cfg);
        if let Some(why) = sizing_refusal(&cfg, &shape, &caps.iter().map(|&(_, c)| c).collect::<Vec<_>>()) {
            return refused(why);
        }
        if let Some(Err(why)) = self.bank.as_ref().map(|request| request.config(&cfg)) {
            return refused(format!("adapter bank: {why}"));
        }
        // The shape carries the bank (serving_shape), so the sizing that chose
        // the context and this placement charge the same bytes.
        let cost = serving_cost(&cfg, &shape);
        // The same cards less the prefill round's activations, as `sizing_refusal` held them to.
        let workspace = shape.stage_workspace(&cfg);
        let caps: Vec<(usize, u64)> = caps.iter().map(|&(i, c)| (i, c.saturating_sub(workspace))).collect();
        let Some(placements) = model::shard::plan_fewest_devices(&cost, &caps) else {
            return refused("the weights fit the cards but the adapter bank does not: lower --adapter-slots, --adapter-rank-cap or --adapter-targets".to_string());
        };
        let stages = placements
            .iter()
            .map(|p| {
                let (device, _) = self.devices[p.shard.gpu_index];
                // `plan_*` indexes `caps`, whose `.0` is an index into
                // `self.devices`; the physical card is that entry's own
                // `Device::Gpu(i)`.
                let physical = match device {
                    Device::Gpu(i) => i as usize,
                    _ => unreachable!("only Gpu devices enter `caps`"),
                };
                (device, Shard { gpu_index: physical, ..p.shard.clone() }, p.bytes)
            })
            .collect();
        Plan { refusal: None, cfg: Some(cfg), stages }
    }

    /// The total device bytes this checkpoint needs whatever the split, and
    /// the per-layer profile behind it - for a caller sizing budgets or
    /// reporting "will this fit at all?".
    pub fn total_device_bytes(&self) -> Result<u64, String> {
        let mg = ResidentCheckpoint::open(&self.gguf_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.gguf_path))?;
        let cfg = mg.config(self.cap)?;
        Ok(serving_cost(&cfg, &self.serving_shape(&mg, &cfg)).total())
    }

    /// The longest per-sequence context this configuration places on its
    /// cards ([`auto_cap`]), or why not even one token fits.
    ///
    /// In the paged layout ([`Self::with_kv_block_rows`]) the longest context is
    /// the largest pool one sequence may take whole ([`auto_pool_rows`]).
    pub fn longest_cap(&self) -> Result<u32, String> {
        let mg = ResidentCheckpoint::open(&self.gguf_path).map_err(|e| format!("cannot open '{}': {e}", self.gguf_path))?;
        let cfg = mg.config(1).map_err(|e| format!("'{}' is not a servable Qwen3.8 checkpoint: {e}", self.gguf_path))?;
        let shape = self.serving_shape(&mg, &cfg);
        let longest = if self.kv_block_rows > 0 { auto_pool_rows(&cfg, &self.devices, &shape, None) } else { auto_cap(&cfg, &self.devices, &shape) };
        longest.ok_or_else(|| format!("{} does not fit across the {} budgeted device(s) even with a one-token context", self.gguf_path, self.devices.len()))
    }

    /// The largest shared KV pool, in rows, this configuration leaves room for
    /// when every sequence is held to `cap` tokens ([`auto_pool_rows`]).
    pub fn longest_pool(&self, cap: u32) -> Result<u32, String> {
        let mg = ResidentCheckpoint::open(&self.gguf_path).map_err(|e| format!("cannot open '{}': {e}", self.gguf_path))?;
        let cfg = mg.config(1).map_err(|e| format!("'{}' is not a servable Qwen3.8 checkpoint: {e}", self.gguf_path))?;
        auto_pool_rows(&cfg, &self.devices, &self.serving_shape(&mg, &cfg), Some(cap)).ok_or_else(|| format!("a {cap}-token context does not fit the {} budgeted device(s) beside the weights", self.devices.len()))
    }

    /// The tier the first stage holds the embedding table at, when it is worth the card's memory: the checkpoint
    /// stores a plain 16-bit table ([`ResidentCheckpoint::embed_tier`]) and the table is at most
    /// [`EMBED_RESIDENT_MAX_SHARE`] of the smallest budgeted card.
    fn embed_tier(&self, mg: &ResidentCheckpoint, cfg: &Qwen35Config) -> Option<Dtype> {
        let tier = mg.embed_tier().filter(|_| !cfg.mtp)?;
        let table = model::ops::Weight::bytes_for(tier, cfg.vocab as usize, cfg.d_model as usize);
        let smallest = gpu_capacities(&self.devices, 0).iter().map(|&(_, c)| c).min()?;
        (table <= smallest / EMBED_RESIDENT_MAX_SHARE).then_some(tier)
    }

    /// What this resident will hold on the cards besides the weights, as
    /// configured: speculation is charged only where loading will allocate it
    /// ([`SpecServing::cost`]) - the verify kernels on every budgeted card and,
    /// for the MTP modes, an MTP block in the checkpoint.
    fn serving_shape(&self, mg: &ResidentCheckpoint, cfg: &Qwen35Config) -> ServingShape {
        let cards: Vec<u32> = self.devices.iter().filter_map(|(d, _)| if let Device::Gpu(i) = d { Some(*i) } else { None }).collect();
        let hw = gpu_core::defaults::Hardware::detected();
        let verify = crate::model::verify::verify_shape_supported(cfg) && [gpu_core::Fused::GdnVerifyPool, gpu_core::Fused::GdnCommitPool].iter().all(|&k| hw.offers_fused(k, &cards));
        ServingShape {
            prefix_slots: self.prefix_slots,
            kv_pool_rows: self.paged_pool_rows(),
            // Exact decode runs plain steps, so the verify scratch is not
            // charged: the placement must describe what will be allocated.
            spec: if self.exact_decode { SpecCost::NONE } else { self.spec.cost(mg.has_mtp(cfg), verify) },
            adapter_bank: self.bank_bytes(cfg),
            head: mg.head_tier(),
            embed: self.embed_tier(mg, cfg),
            ..ServingShape::new(self.cap, self.tier.clone(), self.kv_tier, self.max_batch).with_prefill_rows(self.prefill.batch_tokens.try_into().unwrap_or(u32::MAX))
        }
    }

    /// The device bytes of the adapter bank this resident serves, if any: the
    /// bank lives beside the weights of the layers it corrects, on the card
    /// that holds them, and each stage also carries the row scratch, charged
    /// once to the head (a stage is at least a layer). `None` without a bank,
    /// or for a request the model cannot hold (`plan` refuses that by name).
    fn bank_bytes(&self, cfg: &Qwen35Config) -> Option<BankBytes> {
        let bank = self.bank.as_ref()?.config(cfg).ok()?;
        Some(BankBytes {
            per_layer: bank.device_bytes_by_layer(cfg, cfg.layer_types().len()),
            head: bank.row_scratch_bytes(self.max_batch.max(MAX_PREFILL_TOKENS.max(TENSOR_CORE_PREFILL_TOKENS))),
        })
    }

    /// Which layer range and how many bytes each device holds, as planned -
    /// `(device, start, end, bytes)`. Empty when unplaceable.
    pub fn placement(&self) -> Vec<(Device, usize, usize, u64)> {
        self.plan().stages.iter().map(|(d, s, b)| (*d, s.start, s.end, *b)).collect()
    }
}

/// This model renders the 3.8 chat template by default; `parse_request`
/// alone defaults to the Qwen3-era flavor and would mis-render its prompt.
pub fn parse_request(tok: &QwenBpe, inv: &Invocation) -> Result<qwen3::chat::ParsedRequest, String> {
    qwen3::chat::parse_request(tok, &crate::caps::with_template_flavor_default(inv))
}

impl ResidentModel for Qwen35GgufResident {
    fn manifest(&self) -> Manifest {
        let generate = ActionSpec::new(
            "generate",
            "generate text (Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, INT8 weights loaded straight from a released GGUF, layer-sharded and resident across as many GPUs as its real per-layer bytes need; fp32 GDN state and a KV cache in f32, bf16, int8 or fp8 E4M3 (`--kv-cache`); up to --max-batch sequences per batched dispatch)",
        )
        .streaming()
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("messages", ParamType::Str, "JSON array of {role,content,...} chat turns (overrides prompt)"))
        .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
        .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate; omitted, the reply runs to its end of turn or to the end of the context"))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.0)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
        .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{\"type\":\"function\",...}); none withholds tool schemas, required/named are enforced post-generation (finish_reason \"tool_choice_unmet\" when unmet)"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)))
        .param(ParamSpec::new("reasoning_effort", ParamType::Str, "reasoning effort level: xhigh (default, detailed deliberation), medium (no instruction), or low (brief thinking)").default(json!("xhigh")))
        .param(ParamSpec::new("preserve_thinking", ParamType::Bool, "Qwen3.8 chat-template kwarg: keep <think> blocks from prior assistant turns in the rendered history (takes effect on the Qwen3.8-flavor render, this model's default)").default(json!(true)))
        .param(ParamSpec::new("template_flavor", ParamType::Str, "chat template flavor: qwen3.8 (default; XML <function=> tool-call payloads, prefilled open <think>, preserve_thinking kwarg) or qwen3 (JSON <tool_call> payloads, positional think framing)").default(json!("qwen3.8")))
        .output(BlobSpec::new("text", Media::Text, "the generated text"));
        let generate = llmserve::with_token_capture(generate);
        let generate = llmserve::with_sampling_defaults(generate, MODEL, self.sampling_defaults());
        Manifest::new(
            &self.model_id,
            "Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, served INT8 and GPU-resident directly from a released GGUF (no fp32 intermediate on disk), layer-sharded across as many cards as its real per-layer bytes need. Text only; no MTP self-speculative decode; continuous batching of up to max_batch sequences per dispatch.",
            vec![generate],
        )
        .with_max_context_tokens(self.cap as u64)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.model_id, "default")
    }

    /// Deliberately unusable: this model occupies real bytes on SEVERAL cards
    /// at once, so it has no meaningful single-device footprint. It is
    /// registered via `Executor::register_multi` and budgeted through
    /// [`MultiDeviceResidentModel::estimate_multi`]; see
    /// `crates/residency/src/multi.rs`' module doc for why a plain `register`
    /// would let it spend VRAM the scheduler never budgeted.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    /// The decode slots this resident was built with: the dispatcher may hand
    /// it that many queued requests per call.
    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_batch as usize)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: single-device activate is not supported -- this model is multi-device only, claim it via ResidencyManager::claim_multi"))
    }
}

impl MultiDeviceResidentModel for Qwen35GgufResident {
    fn unavailable_reason(&self, _key: &InstanceKey) -> Option<String> {
        self.plan().refusal.clone()
    }

    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        let plan = self.plan();
        // Host RAM: the mapping is header-only plus whatever one streamed
        // tensor costs at a time (`MmapGguf::with_tensor_chunks` decodes a
        // bounded block window, `Weight::upload` materialises one leaf), so
        // the honest steady-state figure is the largest single leaf.
        let ram = plan
            .cfg
            .as_ref()
            .map(|c| c.intermediate_size as u64 * c.d_model as u64 * 4)
            .unwrap_or(0);
        MultiDeviceCost::new(plan.stages.iter().map(|&(d, _, bytes)| (d, bytes)).collect(), ram)
    }

    /// The served instance: the checkpoint as a [`Qwen35GgufEngine`] under
    /// the generic continuous-batching `model::serve::Scheduler`, so up to
    /// `max_batch` requests decode in ONE batched step per iteration and join
    /// and leave between steps - the same loop every brain LLM is served by.
    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let (sched, tok, stop) = self.activate_scheduler(devices)?;
        Ok(Box::new(llmserve::ScheduledInstance::new(llmserve::TextFrontend { tok, stop, label: "qwen35", parse: std::sync::Arc::new(parse_request) }, sched, self.max_batch as usize)))
    }
}

impl Qwen35GgufResident {
    /// The scheduler every consumer of this resident serves through: the
    /// activated engine ([`Self::activate_engine`]) under the generic
    /// continuous-batching `model::serve::Scheduler` with `max_batch` running
    /// sequences and the configured speculation installed. `brain serve` and
    /// the serving benchmark both build it here, so they cannot drift.
    pub fn activate_scheduler(&self, devices: &[Device]) -> Result<(model::serve::Scheduler<Qwen35GgufEngine>, QwenBpe, Vec<u32>), String> {
        let (engine, tok, stop) = self.activate_engine(devices)?;
        let mut sched = model::serve::Scheduler::new(engine, self.max_batch as usize);
        self.configure_scheduler(&mut sched);
        Ok((sched, tok, stop))
    }

    /// The serving surface: the activated checkpoint as a
    /// `model::serve::PagedDecoder` for the generic continuous-batching
    /// scheduler, with the tokenizer and stop ids its requests need.
    /// `max_batch` is the number of sequences the scheduler may run at once.
    pub fn activate_engine(&self, devices: &[Device]) -> Result<(Qwen35GgufEngine, QwenBpe, Vec<u32>), String> {
        let inst = self.activate_owned(devices)?;
        let (tok, stop) = inst.ckpt.tokenizer().map_err(|e| format!("{MODEL}: '{}': {e}", self.gguf_path))?;
        let mut engine = Qwen35GgufEngine::new(inst);
        engine.set_prefill_policy(self.prefill);
        if let Some(request) = &self.bank {
            // Refusing here is deliberate: a server asked for adapters that
            // cannot give them must say so, not serve the base under their names.
            let bank = request.config(engine.instance().config())?;
            engine.enable_adapters(bank).map_err(|e| format!("{MODEL}: adapter bank: {e}"))?;
        }
        Ok((engine, tok, stop))
    }

    /// [`MultiDeviceResidentModel::activate_multi`] returning the CONCRETE
    /// instance rather than a `Box<dyn Instance>`.
    ///
    /// The trait object is the right surface for the executor, which only ever
    /// calls `run`/`metrics`. A profiler is the other kind of caller: it needs
    /// [`Qwen35GgufInstance::profile_decode`], which is not on `Instance` and
    /// should not be (nothing about serving wants a per-kernel table). So the
    /// build lives here and `activate_multi` boxes what this returns - one
    /// loader, two surfaces, rather than a second copy of the load sequence
    /// that could drift from the one being measured.
    pub fn activate_owned(&self, devices: &[Device]) -> Result<Qwen35GgufInstance, String> {
        let plan = self.plan();
        if let Some(reason) = &plan.refusal {
            return Err(format!("{}: no placement: {reason}", self.model_id));
        }
        let cfg = plan.cfg.clone().ok_or_else(|| format!("{}: no placement", self.model_id))?;
        if plan.stages.is_empty() {
            return Err(format!("{}: no placement", self.model_id));
        }
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named, so it hands back the same set. Insisting on that (rather
        // than silently re-planning for whatever arrives) is what makes the
        // reservation and the allocation describe the same bytes.
        if devices.len() != plan.stages.len() || !devices.iter().all(|d| plan.stages.iter().any(|(pd, _, _)| pd == d)) {
            return Err(format!(
                "{MODEL}: activate_multi got devices {devices:?} but the plan placed {:?} -- the reservation and the load would describe different cards",
                plan.stages.iter().map(|&(d, _, _)| d).collect::<Vec<_>>()
            ));
        }

        let mg = ResidentCheckpoint::open(&self.gguf_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.gguf_path))?;
        let (tok, eos) = mg.tokenizer().map_err(|e| format!("{MODEL}: '{}': {e}", self.gguf_path))?;
        let (embed_name, norm_name, head_name) = mg.endpoint_names(&cfg)?;

        let layout = self.kv_layout();
        let state_rows = layout.state_rows(self.max_batch, self.prefix_slots);
        let embed_resident = self.serving_shape(&mg, &cfg).embed.is_some();
        residency::log::info(&format!("{MODEL}: the embedding table is {}", if embed_resident { "held on the first stage's card" } else { "read from the checkpoint on the host" }));
        let mut shards = Vec::with_capacity(plan.stages.len());
        for (_, placed, _) in &plan.stages {
            // The planner's `embed`/`head` flags say which stage is CHARGED
            // the endpoint bytes; the stage itself is built holding neither
            // (see `DeviceShard`'s own doc - this instance owns both
            // endpoints, because an fp32 `[vocab, d_model]` cannot be a
            // device buffer here at all).
            let shard = &Shard { embed: placed.embed && embed_resident, head: false, ..placed.clone() };
            let shard_src = mg.shard_source(&cfg, shard)?;
            let src = shard_src.as_source();
            // `b = t = 1`: this instance's own `res`/`logits`/`tokens` and
            // its per-instance decode state are sized to the smallest legal
            // value (`gdn_chunk_size(1) == 1`, so `t % chunk == 0` holds for
            // every config). The real per-sequence cache is `ShardCaches`
            // below, at `self.cap` - the same split `crate::serve::Engine`
            // makes for the same reason.
            let qwen35 = Qwen35::new_shard_dt(cfg.clone(), 1, 1, src, shard.clone(), &self.tier);
            // Exact decode is a property of the STEP, not of one card: every
            // stage groups its rows and sizes its attention from this
            // instance's per-sequence context.
            qwen35.set_exact_decode(self.exact_decode.then_some(self.cap));
            qwen35.set_act_precision(self.act_choice.resolve(qwen35.gpu.caps().arch.compute_capability));
            let caches = ShardCaches::new(&qwen35.gpu, &cfg, shard, self.cap, state_rows, self.prefix_slots, self.kv_tier, layout);
            shards.push(DeviceShard { qwen35, caches });
        }

        // The head epilogue, on the LAST stage's card - see `Head`'s own doc
        // for why it is the instance's and not that shard's.
        let last = shards.last().ok_or_else(|| format!("{MODEL}: plan has zero stages"))?;
        let gpu = &last.qwen35.gpu;
        let (v, d) = (cfg.vocab as usize, cfg.d_model as usize);
        // model::int8::upload_quantized takes the fastest route `mg` can
        // serve (a Q8_0 GGUF byte repack straight from blocks, no fp32
        // anywhere, when it applies) and otherwise bounds host scratch to
        // its own row-chunked fp32 fallback - never the whole 5.09 GB table.
        let head_ops = model::ops::Ops::new(gpu.share()).map_err(|e| format!("{MODEL}: Ops::new on the head card: {e}"))?;
        let head_w = upload_head(&mg, &head_ops, gpu, &head_name, v, d)?;
        let norm = match mg.final_norm(&norm_name) {
            Some(Ok(w)) if w.len() == d => gpu.storage_init("qwen35.gguf.final_norm", &w),
            Some(Ok(w)) => return Err(format!("{MODEL}: '{norm_name}' has {} elements, expected {d}", w.len())),
            Some(Err(e)) => return Err(format!("{MODEL}: '{norm_name}': {e}")),
            None => return Err(format!("{MODEL}: '{norm_name}' vanished between planning and load")),
        };
        let head = Head {
            ops: head_ops,
            norm,
            w: head_w,
        };
        let embed = EmbedTable { name: embed_name, d };

        let mut inst = Qwen35GgufInstance {
            cfg,
            shards,
            ckpt: mg,
            embed,
            embed_resident,
            head,
            tok,
            eos,
            cap: self.cap,
            exact_decode: self.exact_decode,
            max_batch: self.max_batch,
            prefix_slots: self.prefix_slots,
            kv_tier: self.kv_tier,
            tier: self.tier.clone(),
            kv_layout: layout,
            taps: RefCell::new(None),
            prefill_adapter_slot: Cell::new(LORA_SLOT_NONE),
            spec_rows: 0,
            mtp: None,
            plans: RefCell::default(),
            plans_on: Cell::new(true),
        };
        let last_gpu = plan.stages.last().map(|(_, s, _)| s.gpu_index).unwrap_or(0);
        // A verify pass is a sequence's consecutive tokens, not unrelated
        // sequences, and the exact regime does not group it: turn speculation
        // off rather than promise an exactness its tape does not have.
        let spec = if self.exact_decode { SpecServing { mode: SpeculativeMode::Off, ..self.spec } } else { self.spec };
        inst.enable_speculation(spec, &self.gguf_path, &self.tier, last_gpu)?;
        Ok(inst)
    }
}

#[cfg(test)]
mod tests {
    /// A tool given one card plans on that card only: another job's card is never in its budget.
    #[test]
    fn the_device_budget_names_only_the_cards_it_is_allowed() {
        const GIB: u64 = 1 << 30;
        let cards = [(0u32, 96 * GIB), (1, 96 * GIB)];
        assert_eq!(super::budget_cards(cards, &[1], 2 * GIB), vec![(Device::Gpu(1), 94 * GIB)], "--device gpu1 never reaches card 0's budget");
        assert_eq!(super::budget_cards(cards, &[0, 1], 2 * GIB).len(), 2);
        assert!(super::budget_cards(cards, &[], 2 * GIB).is_empty(), "an empty set allows nothing");
        assert!(super::budget_cards([(1u32, GIB)], &[1], 2 * GIB).is_empty(), "a card smaller than the reserve has no usable bytes");
    }

    use super::*;

    const GB: u64 = 1 << 30;

    /// Diagnostic (not gated - prints, does not assert): drives
    /// [`Qwen35GgufInstance::stack_step`] (pure TOKEN-BY-TOKEN decode, never
    /// [`Qwen35GgufInstance::stack_prefill_chunk`]/`replay_prompt`'s round
    /// path) for ~900 real tokens on the real checkpoint, printing the
    /// per-stage residual RMS the same way `debug_step` does.
    ///
    /// `tests/gguf_resident_real.rs`'s `where_does_the_real_checkpoint_
    /// start_producing_garbage` found the residual RMS exploding
    /// (single-digit -> tens -> hundreds) starting right around the 3rd
    /// `MAX_PREFILL_TOKENS`-sized CHUNKED round (~pos 680) and never
    /// recovering. This test asks the one question that result cannot
    /// answer on its own: does the SAME real checkpoint also diverge at the
    /// same real position when replayed one token at a time, with no
    /// chunked round boundary ever crossed? If yes, the defect is long-
    /// context GDN/GQA state drift independent of chunking; if the residual
    /// stays healthy here, the defect is specific to the chunked-round-carry
    /// path (`Qwen35::run_prefill_chunk_stage`'s GDN/GQA state hand-off)
    /// interacting with the REAL checkpoint's own weight values (every
    /// synthetic random-weight test at this crate's `model.rs` level already
    /// passed clean at this same round count and real dims).
    #[test]
    fn token_by_token_replay_on_the_real_checkpoint_past_where_chunked_replay_diverges() {
        // Test-gating convention only - selects the real-checkpoint gate, it
        // is not a serving knob. The serving path finds its GGUFs in the
        // model store.
        const REAL_GGUF_ENV: &str = "BRAIN_QWEN35_GGUF";
        let Ok(path) = std::env::var(REAL_GGUF_ENV) else {
            brain_testutil::skip(&format!("{REAL_GGUF_ENV} unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)"));
            return;
        };
        const RESERVE: u64 = 2 * GB;
        let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
        if devices.is_empty() {
            brain_testutil::skip_unavailable("no GPU with queryable VRAM - this resident is GPU-only");
            return;
        }
        let r = Qwen35GgufResident::new(path, devices, 2048, TierPolicy::uniform(Dtype::I8));
        let key = r.instance_key("generate", &Invocation::new());
        let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
        let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

        let vocab = inst.cfg.vocab;
        let n = 900u32;
        for pos in 0..n {
            let tok = (pos * 5 + 3) % vocab;
            let logits = inst.stack_step(tok, pos, 0).expect("stack_step");
            if pos % 64 == 0 || pos == n - 1 {
                let rms = (logits.iter().map(|v| v * v).sum::<f32>() / logits.len() as f32).sqrt();
                let maxabs = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                println!("token-by-token pos {pos:4}: logits rms={rms:.4} maxabs={maxabs:.4}");
            }
        }
    }

    /// A table held on the card is charged at the bytes it is stored in; the default charges nothing.
    #[test]
    fn a_resident_embedding_table_is_charged_at_its_stored_tier() {
        let cfg = Qwen35Config::qwen38_27b();
        let base = ServingShape::new(2048, TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
        let held = serving_cost(&cfg, &ServingShape { embed: Some(Dtype::BF16), ..base.clone() });
        assert_eq!(serving_cost(&cfg, &base).embed, 0);
        assert_eq!(held.embed, cfg.vocab as u64 * cfg.d_model as u64 * 2, "a bf16 table is two bytes an element");
    }

    /// The endpoints are charged for what this resident ACTUALLY places, and
    /// what it places is what a 24 GB P40 can hold.
    ///
    /// Both `[vocab, d_model]` tables are 5_085_593_600 bytes as fp32 -
    // perf-number: arithmetic: the fp32 table size above over the 2047 MiB binding limit
    /// simultaneously over that card's `max_buffer_size` (~4.09 GiB) and 2.4x
    /// its 2047 MiB storage-BINDING limit, so "just allocate it" is not an
    /// option this cost model may describe. The embedding is therefore not on
    /// the card at all (row-at-a-time out of the mapping ⇒ 0 bytes) and the
    /// head is INT8 (1.42 GB, inside both limits). A cost model that charged
    /// either at the fp32 rate would plan a split that cannot load.
    #[test]
    fn the_endpoints_are_charged_for_what_is_really_placed() {
        let cfg = Qwen35Config::qwen38_27b();
        let cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
        let fp32_table = cfg.vocab as u64 * cfg.d_model as u64 * 4;
        assert_eq!(fp32_table, 5_085_593_600, "the fp32 table this resident refuses to place");
        assert_eq!(cost.embed, 0, "the embedding is read from the mapping a row at a time, never uploaded");
        assert_eq!(cost.head, head_bytes(&cfg, Dtype::I8) + cfg.d_model as u64 * 4, "int8 lm_head + fp32 norm.weight");
        assert!(cost.head < fp32_table / 3, "the int8 head must be far smaller than the fp32 table: {}", cost.head);
        // Inside the 2047 MiB storage-buffer BINDING limit, which is the
        // constraint that actually decides whether this can run at all.
        assert!(head_bytes(&cfg, Dtype::I8) < 2047 << 20, "the packed head must be bindable: {}", head_bytes(&cfg, Dtype::I8));
    }

    /// A layer's decode state is charged to the stage that owns the layer -
    /// without it a placement would under-budget every card by the whole KV
    /// pool. GQA layers scale with context, GDN layers do not.
    #[test]
    fn per_layer_cost_includes_this_sequences_decode_state() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let small = layer_cost(&cfg, 128, &i8, KvTier::F32, 1);
        let large = layer_cost(&cfg, 8192, &i8, KvTier::F32, 1);
        let types = cfg.layer_types();
        let gqa = types.iter().position(|t| *t == LayerType::Full).unwrap();
        let gdn = types.iter().position(|t| *t == LayerType::Linear).unwrap();
        assert!(small.per_layer[gqa] > cfg.layer_weight_bytes(LayerType::Full, &i8), "a GQA layer must be charged its KV cache");
        assert!(small.per_layer[gdn] > cfg.layer_weight_bytes(LayerType::Linear, &i8), "a GDN layer must be charged its recurrent state");
        assert!(large.per_layer[gqa] > small.per_layer[gqa], "GQA cache scales with context");
        assert_eq!(large.per_layer[gdn], small.per_layer[gdn], "GDN state is O(1) in context, not O(T)");
    }

    /// The KV tier is constructor data (`with_kv_tier`), parsed by the CLI
    /// from `KvTier::parse`'s names - the parse contract the serve flag
    /// inherits: known names map, an unknown name is refused (never a silent
    /// fallback to a different precision than was asked for).
    #[test]
    fn the_kv_tier_is_parsed_data_refusing_unknown_names() {
        assert_eq!(KvTier::parse("int8"), Ok(KvTier::INT8));
        assert_eq!(KvTier::parse("bf16"), Ok(KvTier::BF16));
        assert_eq!(KvTier::parse("f32"), Ok(KvTier::F32));
        assert_eq!(KvTier::parse("fp8"), Ok(KvTier::FP8));
        let e = KvTier::parse("fp4").unwrap_err();
        assert!(e.contains("fp4"), "{e}");
        // ...and the constructor carries it through.
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8)).with_kv_tier(KvTier::parse("int8").unwrap());
        assert_eq!(r.kv_tier(), KvTier::INT8);
    }

    /// The served id is caller-set constructor data
    /// (`with_model_id`): a checkpoint resolved from a non-canonical store
    /// position is served under the id its own position carries, not the
    /// [`MODEL`] constant, and `manifest()`/`instance_key()` agree on it -
    /// the property that makes two quants of one release two separately
    /// addressable models.
    #[test]
    fn the_served_id_is_constructor_data_shared_by_manifest_and_instance_key() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8)).with_model_id("unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.model_id(), "unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.manifest().model, "unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.instance_key("generate", &Invocation::new()).model, "unsloth/Qwen3.8-27B-GGUF");
        // No override: the well-known release id, unchanged.
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8));
        assert_eq!(r.manifest().model, MODEL);
    }

    /// The most sequences of `cap` tokens the planner admits on ONE card of
    /// `usable` bytes, by the same `plan_fewest_devices` the resident uses.
    fn max_batch_on_one_card(cfg: &Qwen35Config, cap: u32, kv: KvTier, usable: u64) -> u32 {
        let i8 = TierPolicy::uniform(Dtype::I8);
        (1..=64u32)
            .take_while(|&b| kv_pool_addressable(cfg, cap, b) && model::shard::plan_fewest_devices(&layer_cost(cfg, cap, &i8, kv, b), &[(0, usable)]).is_some())
            .last()
            .unwrap_or(0)
    }

    /// The headline sizing: Qwen3.8-27B at 128k context on one 97 GiB GH200
    /// (2 GiB reserve). The KV cache is what bounds the batch there, so each
    /// compact tier must buy sequences in proportion to the bytes it saves:
    /// ~16 GiB per sequence at f32, half that at bf16, a bit over a quarter
    /// at int8 (one f32 scale per 256-element row).
    #[test]
    fn a_compact_kv_tier_admits_proportionally_more_sequences_at_128k_on_one_gh200() {
        let cfg = Qwen35Config::qwen38_27b();
        let usable = (97_871u64 << 20) - (2 << 30);
        let cap = 131_072;
        let [f32b, bf16b, int8b, fp8b] = KvTier::ALL.map(|kv| max_batch_on_one_card(&cfg, cap, kv, usable));
        println!("128k on one GH200: planner admits f32 {f32b}, bf16 {bf16b}, int8 {int8b}, fp8 {fp8b} sequences");
        assert_eq!(fp8b, int8b, "fp8 has the int8 footprint");
        assert!(f32b >= 1, "one f32 sequence must fit, or this sizing is not the one under test");
        assert!(bf16b >= 2 * f32b, "bf16 halves the cache, so it must at least double the batch: f32 {f32b}, bf16 {bf16b}");
        assert!(int8b >= 3 * f32b, "int8 is ~3.9x smaller, so it must at least triple the batch: f32 {f32b}, int8 {int8b}");
        assert!(int8b > bf16b);
    }

    /// A batch that cannot fit is refused with the operator's levers named
    /// and the arithmetic shown, and the same batch is accepted once the KV
    /// cache is stored compactly - on one GH200-sized card.
    #[test]
    fn a_batch_that_cannot_fit_is_refused_naming_the_flags_to_change() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let usable = [(97_871u64 << 20) - (2 << 30)];
        assert_eq!(sizing_refusal(&cfg, &ServingShape::new(4096, i8.clone(), KvTier::BF16, 64), &usable), None, "64 x 4096 with a bf16 cache fits one GH200");
        let why = sizing_refusal(&cfg, &ServingShape::new(4096, i8.clone(), KvTier::F32, 256), &usable).expect("256 f32 sequences of 4096 tokens cannot fit");
        for needle in ["--max-batch 256", "--ctx 4096", "GiB weights", "more compact --kv-cache"] {
            assert!(why.contains(needle), "refusal must mention {needle:?}: {why}");
        }
        let why = sizing_refusal(&cfg, &ServingShape::new(131_072, i8.clone(), KvTier::BF16, 40), &usable).expect("past the u32 plane index");
        assert!(why.contains("u32 element index"), "{why}");
    }

    /// The refusal states the planner's own arithmetic per category and names the smallest KV tier
    /// that places on these cards; a batch no tier can hold gets no tier suggestion.
    #[test]
    fn a_refusal_names_the_kv_tier_that_would_fit_and_the_per_category_bytes() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let usable = [(97_871u64 << 20) - (2 << 30)];
        let why = sizing_refusal(&cfg, &ServingShape::new(4096, i8.clone(), KvTier::F32, 200), &usable).expect("200 f32 sequences of 4096 tokens cannot fit");
        assert!(why.contains("a more compact --kv-cache (bf16) would fit") || why.contains("a more compact --kv-cache (int8) would fit"), "{why}");
        assert!(why.contains("GiB weights") && why.contains("GiB KV cache") && why.contains("GiB recurrent state"), "{why}");
        let why = sizing_refusal(&cfg, &ServingShape::new(4096, i8.clone(), KvTier::F32, 1000), &usable).unwrap();
        assert!(!why.contains("would fit") && why.contains("choose a more compact --kv-cache"), "{why}");
        // The categories add up to what the placement cost charges.
        let total: u64 = serving_demands(&cfg, &ServingShape::new(4096, i8.clone(), KvTier::F32, 200)).iter().map(|d| d.variants[0].device_bytes).sum();
        assert_eq!(total, layer_cost(&cfg, 4096, &i8, KvTier::F32, 200).total());
    }

    /// The sizing a batch-serving deployment plans with: at the default
    /// 4096-token context one 97 GiB GH200 holds 64 sequences in any KV tier
    /// (the sweep's largest batch), and the compact tiers buy proportionally
    /// more slots. Pure arithmetic over the planner, printed so a deployment
    /// can read its ceiling off the test log.
    #[test]
    fn one_gh200_holds_64_sequences_of_4096_tokens_in_every_kv_tier() {
        let cfg = Qwen35Config::qwen38_27b();
        let usable = (97_871u64 << 20) - (2 << 30);
        let i8 = TierPolicy::uniform(Dtype::I8);
        let ceiling = |kv: KvTier| (1..=1024u32).take_while(|&b| sizing_refusal(&cfg, &ServingShape::new(4096, i8.clone(), kv, b), &[usable]).is_none()).last().unwrap_or(0);
        let [f32b, bf16b, int8b, _fp8b] = KvTier::ALL.map(ceiling);
        println!("4096 tokens on one GH200: planner admits f32 {f32b}, bf16 {bf16b}, int8 {int8b} sequences");
        assert!(f32b >= 64, "64 f32 sequences of 4096 tokens must fit: {f32b}");
        assert!(bf16b > f32b && int8b > bf16b, "a more compact tier admits more sequences: f32 {f32b}, bf16 {bf16b}, int8 {int8b}");
    }

    /// A plane is addressed by `u32` element offsets, so past 4.29G elements
    /// a pool would alias itself whatever the cards could hold: at 128k tokens
    /// x 1024 elements that is 33 sequences, and the planner must refuse there.
    #[test]
    fn a_kv_pool_past_the_u32_element_index_is_not_addressable() {
        let cfg = Qwen35Config::qwen38_27b();
        assert!(kv_pool_addressable(&cfg, 131_072, 31));
        assert!(!kv_pool_addressable(&cfg, 131_072, 33));
        let huge = 1u64 << 40; // far more than any card holds, so only addressing can stop the count
        assert_eq!(max_batch_on_one_card(&cfg, 131_072, KvTier::INT8, huge), 32);
    }

    /// The real model at the real shape does NOT fit one 24 GB P40 and DOES
    /// fit two - the whole reason this resident exists. Pure arithmetic, no
    /// GPU and no checkpoint: it is a property of the published dims.
    #[test]
    fn the_real_model_needs_two_24gb_cards_and_fits_them() {
        let cfg = Qwen35Config::qwen38_27b();
        let cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
        let p40 = 24 * GB; // 24 GiB usable, i.e. a card with no reserve at all
        assert!(cost.total() > p40, "if it fitted one card this resident would be pointless: {} bytes", cost.total());
        assert!(model::shard::plan_by_capacity(&cost, &[(0, p40)]).is_none(), "one card must be reported infeasible, not planned");
        let two = model::shard::plan_fewest_devices(&cost, &[(0, p40), (1, p40)]).expect("two 24 GiB cards must hold it");
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].shard.start, 0);
        assert!(two[0].shard.embed && !two[0].shard.head);
        assert_eq!(two[1].shard.end, cfg.n_layers as usize);
        assert!(two[1].shard.head && !two[1].shard.embed);
        assert_eq!(two[0].shard.end, two[1].shard.start, "contiguous, no gap");
        for p in &two {
            assert!(p.bytes <= p40, "stage of {} bytes overruns a {p40}-byte card", p.bytes);
        }
    }

    /// M24's headline capacity claim, as a standing gate rather than a
    /// one-off calculation: a per-leaf tier that puts the bulk MLP
    /// projections at Q4 while holding the two GDN state-sensitive gates
    /// (`in_proj_a`/`in_proj_b`, the decay/beta projections - ~94 MB total
    /// across 48 GDN layers, essentially free to keep at full precision)
    /// FITS ONE 24 GB P40, unlike the uniform-INT8 tier above which needs
    /// two. Pure arithmetic, no GPU and no checkpoint - if this regresses,
    /// the single-card cascade the M24 plan is built on (no cross-card
    /// residual handoff, MTP legal again since a whole shard fits one card,
    /// a second card free for a concurrent sequence) silently stops being
    /// true.
    #[test]
    fn a_q4_mlp_tier_with_gdn_gates_held_at_f32_fits_one_24gb_card() {
        let cfg = Qwen35Config::qwen38_27b();
        let policy_c = TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32);
        let cost = layer_cost(&cfg, 2048, &policy_c, KvTier::F32, 1);
        let i8_cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
        assert!(
            cost.total() < i8_cost.total(),
            "a narrower tier must cost fewer bytes than uniform I8: q4 {} >= i8 {}",
            cost.total(),
            i8_cost.total()
        );
        let p40 = 24 * GB;
        let one = model::shard::plan_by_capacity(&cost, &[(0, p40)]);
        assert!(
            one.is_some(),
            "policy C ({} bytes) must fit one {p40}-byte card - it is the whole point of narrowing the tier; \
             uniform I8 needs {} bytes and correctly does not fit",
            cost.total(),
            i8_cost.total()
        );
        let one = one.unwrap();
        assert_eq!(one.len(), 1, "a single-card plan must be exactly one stage");
        assert!(one[0].shard.embed && one[0].shard.head, "the lone stage must own both endpoints");
        assert_eq!(one[0].shard.start, 0);
        assert_eq!(one[0].shard.end, cfg.n_layers as usize);
        assert!(one[0].bytes <= p40);
    }

    /// A [`TierPolicy`] that is uniform Q4 with NO exception also fits one
    /// card, with more headroom than policy C - the reference point that
    /// isolates what holding the GDN gates at F32 costs (a few MB, per
    /// [`Qwen35Config::layer_weight_bytes`]'s own ground-truth test) against
    /// what it buys (see M24's roadmap entry for the position-sweep quality
    /// comparison this arithmetic alone cannot make).
    #[test]
    fn uniform_q4_fits_one_24gb_card_with_more_headroom_than_policy_c() {
        let cfg = Qwen35Config::qwen38_27b();
        let uniform_q4 = TierPolicy::uniform(Dtype::Q4);
        let policy_c = TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32);
        let cost_uniform = layer_cost(&cfg, 2048, &uniform_q4, KvTier::F32, 1);
        let cost_c = layer_cost(&cfg, 2048, &policy_c, KvTier::F32, 1);
        assert!(
            cost_uniform.total() < cost_c.total(),
            "uniform Q4 must cost fewer bytes than policy C's F32 exception: {} >= {}",
            cost_uniform.total(),
            cost_c.total()
        );
        let p40 = 24 * GB;
        assert!(model::shard::plan_by_capacity(&cost_uniform, &[(0, p40)]).is_some(), "uniform Q4 must fit one card");
    }

    /// `estimate_multi` must never panic and must report ZERO devices for an
    /// unreadable checkpoint - the documented "unavailable" signal. It runs
    /// on the `Executor` dispatcher thread, where a panic takes every other
    /// model on the server down with it.
    #[test]
    fn an_unreadable_gguf_reports_zero_devices_rather_than_panicking() {
        let r = Qwen35GgufResident::new("/nonexistent/qwen35.gguf".to_string(), vec![(Device::Gpu(0), 24 * GB)], 2048, TierPolicy::uniform(Dtype::I8));
        let cost = r.estimate_multi(&InstanceKey::new(MODEL, "default"));
        assert_eq!(cost.devices().count(), 0);
        assert!(r.placement().is_empty());
        assert!(r.activate_multi(&InstanceKey::new(MODEL, "default"), &[]).is_err());
        // The plan is memoized, so a second call must agree with the first
        // rather than re-deciding (and must still not panic).
        assert_eq!(r.estimate_multi(&InstanceKey::new(MODEL, "default")).devices().count(), 0);
    }

    /// No budgeted GPU is also "unavailable", not a panic and not a CPU
    /// fallback (there is no CPU path - the weights are int8 device buffers).
    #[test]
    fn no_gpu_reports_zero_devices() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8));
        assert_eq!(r.estimate_multi(&InstanceKey::new(MODEL, "default")).devices().count(), 0);
    }

    /// `reasoning_effort` is declared in `caps.rs`'s manifest but was missing
    /// from this resident's own - `qwen3::chat::parse_request` still defaulted
    /// it to `xhigh` either way, but a caller passing it EXPLICITLY through
    /// this path was rejected by param validation before it ever reached the
    /// parser. `Qwen35GgufResident::new` needs no real file (`manifest()`
    /// touches only `self.cap`, never the checkpoint), so this needs no GPU.
    #[test]
    fn manifest_declares_reasoning_effort() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8));
        let m = r.manifest();
        assert_eq!(m.actions.len(), 1);
        let g = &m.actions[0];
        assert!(
            g.params.iter().any(|p| p.name == "reasoning_effort" && !p.required),
            "reasoning_effort must be declared (and optional, defaulting to xhigh) so a caller passing it is not rejected"
        );
    }

    /// Single-device activation is refused, loudly - registering this model
    /// with the plain `Executor::register` would budget only one of the cards
    /// it actually occupies.
    #[test]
    fn single_device_activation_is_refused() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![(Device::Gpu(0), 24 * GB)], 2048, TierPolicy::uniform(Dtype::I8));
        let key = InstanceKey::new(MODEL, "default");
        assert_eq!(r.estimate(&key).vram, 0);
        let err = match r.activate(&key, Device::Gpu(0)) {
            Ok(_) => panic!("single-device activate must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("multi-device only"), "{err}");
    }

    // ---- context sized from the memory the placement leaves

    fn p40_pair() -> Vec<(Device, u64)> {
        vec![(Device::Gpu(0), 24 * GB), (Device::Gpu(1), 24 * GB)]
    }

    /// The same feasibility test `auto_cap` applies, spelled out from the
    /// planner so the assertions below do not copy its search.
    fn fits_at(cfg: &Qwen35Config, devices: &[(Device, u64)], cap: u32, kv: KvTier) -> bool {
        let caps: Vec<(usize, u64)> = devices.iter().enumerate().map(|(i, &(_, c))| (i, c.saturating_sub(ServingShape::new(cap, TierPolicy::uniform(Dtype::I8), kv, 1).stage_workspace(cfg)))).collect();
        kv_pool_addressable(cfg, cap, 1) && model::shard::plan_fewest_devices(&layer_cost(cfg, cap, &TierPolicy::uniform(Dtype::I8), kv, 1), &caps).is_some()
    }

    /// A D-Bus client validates against the manifest: the token-capture flags the
    /// scheduler honours must be declared there too.
    #[test]
    fn the_generate_spec_declares_the_token_capture_flags() {
        let r = Qwen35GgufResident::new("/nonexistent.gguf".to_string(), vec![], 512, TierPolicy::uniform(Dtype::I8));
        let spec = r.manifest().actions.into_iter().find(|a| a.name == "generate").expect("generate");
        let inv = spec.validate(Invocation::new().set("prompt", serde_json::json!("hi")).set("logprobs", serde_json::json!(true)).set("return_token_ids", serde_json::json!(true)));
        assert!(inv.is_ok(), "{inv:?}");
    }

    /// The context is whatever the cards have room for after the weights, not a
    /// development constant: it is the largest cap the planner still accepts,
    /// and the very next step up is refused.
    #[test]
    fn auto_cap_is_the_largest_context_the_cards_hold() {
        let cfg = Qwen35Config::qwen38_27b();
        let devices = p40_pair();
        let cap = auto_cap(&cfg, &devices, &ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::F32, 1)).expect("27B INT8 fits two 24 GiB cards");
        assert!(cap > 8192, "the leftover memory must buy far more than the old default context: {cap}");
        assert!(fits_at(&cfg, &devices, cap, KvTier::F32), "the chosen context must be placeable");
        assert!(!fits_at(&cfg, &devices, cap + AUTO_CAP_GRANULE, KvTier::F32), "one granule more must not be: the context is the memory bound, not below it");
    }

    /// More room buys more context; a KV tier that stores fewer bytes per
    /// token buys more still.
    #[test]
    fn auto_cap_follows_the_memory_and_the_kv_tier() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let pair = p40_pair();
        let roomier = vec![(Device::Gpu(0), 40 * GB), (Device::Gpu(1), 40 * GB)];
        let f32_cap = auto_cap(&cfg, &pair, &ServingShape::new(1, i8.clone(), KvTier::F32, 1)).unwrap();
        assert!(auto_cap(&cfg, &roomier, &ServingShape::new(1, i8.clone(), KvTier::F32, 1)).unwrap() > f32_cap);
        assert!(auto_cap(&cfg, &pair, &ServingShape::new(1, i8.clone(), KvTier::INT8, 1)).unwrap() > f32_cap);
        assert!(auto_cap(&cfg, &pair, &ServingShape::new(1, i8.clone(), KvTier::F32, 4)).unwrap() < f32_cap, "more concurrent slots share the same leftover");
    }

    /// The context the cards hold leaves room for the prefill round: a served model
    /// that fills its card to the last KV row loads, then dies on its first long prompt.
    #[test]
    fn auto_cap_leaves_the_prefill_workspace_free() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let card = [(Device::Gpu(0), 96 * GB)];
        let shape = ServingShape::new(1, i8, KvTier::INT8, 8);
        let cap = auto_cap(&cfg, &card, &shape).expect("27B INT8 fits a 96 GiB card");
        let (at, more_at) = (shape.at(cap), shape.at(cap + AUTO_CAP_GRANULE));
        assert!(prefill_workspace_bytes(&cfg, at.prefill_rows) > 0, "a prefill round allocates activations");
        let (placed, workspace) = (serving_cost(&cfg, &at).total(), at.stage_workspace(&cfg));
        assert!(placed + workspace <= 96 * GB, "{placed} bytes placed + {workspace} of step workspace overrun the card");
        let (more, more_workspace) = (serving_cost(&cfg, &more_at).total(), more_at.stage_workspace(&cfg));
        assert!(more + more_workspace > 96 * GB, "one granule more still fits beside the workspace: the context stops short of the memory bound");
    }

    /// The paged layout bounds memory by the shared pool, not by a window per slot: the
    /// same card that holds a ~100k context for 8 slots + 4 cached prompts one-block holds
    /// a pool of a million rows, and a batch of 32 costs the recurrent state only.
    #[test]
    fn the_paged_pool_is_bounded_by_memory_not_multiplied_by_the_batch() {
        let cfg = Qwen35Config::qwen38_27b();
        let card = [(Device::Gpu(0), 96 * GB)];
        let shape = ServingShape { prefix_slots: 32, ..ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::INT8, 32) };
        let windowed = auto_cap(&cfg, &card, &shape).expect("fits");
        let pool = auto_pool_rows(&cfg, &card, &shape, None).expect("fits");
        assert!(pool > 8 * windowed, "pool {pool} rows against {windowed} rows per slot");
        assert_eq!(pool % KV_BLOCK_ROWS, 0, "whole blocks");
        // The pool, the placement and the workspace fit together; one granule more does not.
        let at = |rows: u32| ServingShape { kv_pool_rows: rows, cap: rows, ..shape.clone() };
        let total = |rows: u32| serving_cost(&cfg, &at(rows)).total() + at(rows).stage_workspace(&cfg);
        assert!(total(pool) <= 96 * GB && total(pool + AUTO_POOL_GRANULE) > 96 * GB);
        // A fixed context that fits the pool is honoured; one longer than any pool is refused.
        assert!(auto_pool_rows(&cfg, &card, &shape, Some(65_536)).expect("fits") >= 65_536);
        assert_eq!(auto_pool_rows(&cfg, &card, &shape, Some(u32::MAX / 2)), None);
    }

    /// An adapter bank is part of what the automatic pool leaves room for: with one
    /// the pool is smaller by at least its bytes and still places, instead of filling
    /// the card and refusing the bank afterwards.
    #[test]
    fn the_automatic_pool_leaves_room_for_an_adapter_bank() {
        let cfg = Qwen35Config::qwen38_27b();
        let card = [(Device::Gpu(0), 96 * GB)];
        let shape = ServingShape { prefix_slots: 32, ..ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::INT8, 32) };
        let layers = cfg.layer_types().len();
        let bank = BankBytes { per_layer: vec![64 << 20; layers], head: 2 * GB };
        let banked = ServingShape { adapter_bank: Some(bank.clone()), ..shape.clone() };
        let plain_pool = auto_pool_rows(&cfg, &card, &shape, None).expect("fits");
        let banked_pool = auto_pool_rows(&cfg, &card, &banked, None).expect("fits beside the bank");
        assert!(banked_pool < plain_pool, "pool {banked_pool} with the bank against {plain_pool} without");
        let at = ServingShape { kv_pool_rows: banked_pool, cap: banked_pool, ..banked.clone() };
        let placed = serving_cost(&cfg, &at).total();
        let without = serving_cost(&cfg, &ServingShape { adapter_bank: None, ..at.clone() }).total();
        assert_eq!(placed - without, bank.per_layer.iter().sum::<u64>() + bank.head, "the bank is charged once, all of it");
        assert!(placed + at.stage_workspace(&cfg) <= 96 * GB, "the banked pool places");
    }

    /// An operator-fixed context is held to the same budget as the automatic one.
    #[test]
    fn an_explicit_context_that_leaves_no_room_for_the_prefill_round_is_refused() {
        let cfg = Qwen35Config::qwen38_27b();
        let shape = ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::INT8, 8);
        let shape = shape.at(65_536);
        let workspace = shape.stage_workspace(&cfg);
        // Exactly the bytes the weights and state need: it places, with nothing left for a round.
        let tight = serving_cost(&cfg, &shape).total();
        assert_eq!(sizing_refusal(&cfg, &shape, &[tight + workspace]), None, "with the workspace beside it the configuration fits");
        let why = sizing_refusal(&cfg, &shape, &[tight]).expect("no room for the prefill round");
        assert!(why.contains("prefill"), "the refusal names the prefill workspace: {why}");
    }

    /// The workspace follows the rows a round carries, and the batched-prefill setting widens them.
    #[test]
    fn the_prefill_workspace_grows_with_the_rows_of_a_round() {
        let cfg = Qwen35Config::qwen38_27b();
        assert_eq!(prefill_workspace_bytes(&cfg, 2048), 2 * prefill_workspace_bytes(&cfg, 1024));
        let i8 = TierPolicy::uniform(Dtype::I8);
        let base = ServingShape::new(8192, i8, KvTier::INT8, 8);
        assert!(base.with_prefill_rows(4096).stage_workspace(&cfg) > base.stage_workspace(&cfg));
    }

    /// Weights that do not fit the cards at all leave no context to offer.
    #[test]
    fn auto_cap_is_none_when_the_weights_alone_do_not_fit() {
        let cfg = Qwen35Config::qwen38_27b();
        let one_card = [(Device::Gpu(0), 24 * GB)];
        assert_eq!(auto_cap(&cfg, &one_card, &ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::F32, 1)), None);
        assert_eq!(auto_cap(&cfg, &[], &ServingShape::new(1, TierPolicy::uniform(Dtype::I8), KvTier::F32, 1)), None);
    }

    // ---- a model is only worth listing if it can load

    fn write_fixture(tag: &str, servable: bool) -> String {
        let path = std::env::temp_dir().join(format!("brain-qwen35-resident-{tag}-{}.gguf", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        if servable {
            crate::gguf_import::testing::write_synthetic_servable_gguf(&path);
        } else {
            crate::gguf_import::testing::write_synthetic_gguf(&path);
        }
        path
    }

    #[test]
    fn a_complete_gguf_on_a_budgeted_card_has_no_reason_to_be_unlisted() {
        let path = write_fixture("ok", true);
        let r = Qwen35GgufResident::new(path.clone(), vec![(Device::Gpu(1), 24 * GB)], 64, TierPolicy::uniform(Dtype::I8));
        assert_eq!(r.unservable_reason(), None);
        assert_eq!(r.placement().iter().map(|p| p.0).collect::<Vec<_>>(), vec![Device::Gpu(1)], "placed only on the card it was offered");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn every_way_to_never_load_is_named_before_the_model_is_listed() {
        let complete = write_fixture("reasons-ok", true);
        let tokenizerless = write_fixture("reasons-no-tok", false);
        let on_gpu = |path: &str, bytes: u64| Qwen35GgufResident::new(path.to_string(), vec![(Device::Gpu(0), bytes)], 64, TierPolicy::uniform(Dtype::I8));

        let no_card = Qwen35GgufResident::new(complete.clone(), vec![], 64, TierPolicy::uniform(Dtype::I8));
        assert!(no_card.unservable_reason().unwrap().contains("no GPU"), "{:?}", no_card.unservable_reason());
        let missing = on_gpu("/nonexistent/model.gguf", 24 * GB);
        assert!(missing.unservable_reason().unwrap().contains("/nonexistent/model.gguf"));
        let no_room = on_gpu(&complete, 64);
        assert!(no_room.unservable_reason().unwrap().contains("budgeted GPU(s) offer"), "{:?}", no_room.unservable_reason());
        let no_vocab = on_gpu(&tokenizerless, 24 * GB);
        assert!(no_vocab.unservable_reason().unwrap().contains("tokenizer"), "{:?}", no_vocab.unservable_reason());

        std::fs::remove_file(complete).ok();
        std::fs::remove_file(tokenizerless).ok();
    }
}

#[cfg(test)]
mod decode_phase_tests {
    use super::*;

    /// A GGUF that declares `general.sampling.*` defaults is served with them: the manifest a client reads carries
    /// them as the defaults of `temp`, `top_k` and `top_p`, which is what a request that omits them gets. One that
    /// declares nothing gets the chat fallback.
    #[test]
    fn the_checkpoints_declared_sampling_defaults_are_the_manifests_param_defaults() {
        use checkpoint::gguf::GgufValue;
        let dir = std::env::temp_dir().join(format!("brain-gguf-sampling-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, sampling: &[(&str, GgufValue)]| {
            let path = dir.join(name);
            let mut kv = vec![("general.architecture".to_string(), GgufValue::String("qwen35".into()))];
            kv.extend(sampling.iter().map(|(k, v)| (k.to_string(), v.clone())));
            let tensor = checkpoint::gguf_write::TensorOut { name: "w".into(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] };
            checkpoint::gguf_write::write(path.to_str().unwrap(), &kv, &[tensor], 32).unwrap();
            path.to_string_lossy().into_owned()
        };
        let defaults = |path: String| -> Vec<(String, serde_json::Value)> {
            let manifest = Qwen35GgufResident::new(path, Vec::new(), 64, TierPolicy::uniform(gpu_core::select::Dtype::I8)).manifest();
            let generate = manifest.actions.iter().find(|a| a.name == "generate").expect("generate").clone();
            ["temp", "top_k", "top_p"].iter().map(|n| (n.to_string(), generate.params.iter().find(|p| p.name == *n).and_then(|p| p.default.clone()).expect("a default"))).collect()
        };
        let declared = write("declared.gguf", &[("general.sampling.top_k", GgufValue::I32(20)), ("general.sampling.top_p", GgufValue::F32(0.95)), ("general.sampling.temp", GgufValue::F32(1.0))]);
        let got = defaults(declared);
        assert_eq!(got[0].1, serde_json::json!(1.0));
        assert_eq!(got[1].1, serde_json::json!(20));
        assert_eq!(got[2].1.as_f64().map(|p| (p * 100.0).round()), Some(95.0));
        let silent = defaults(write("silent.gguf", &[]));
        assert_eq!((silent[0].1.clone(), silent[1].1.clone(), silent[2].1.as_f64().map(|p| (p * 100.0).round())), (serde_json::json!(0.7), serde_json::json!(20), Some(95.0)), "nothing declared: the chat fallback");
        std::fs::remove_dir_all(&dir).ok();
    }

}
