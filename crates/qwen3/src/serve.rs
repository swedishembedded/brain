// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concurrent serving engine for the Qwen3 decoder: a **paged** KV cache shared by
//! many sequences + **batched** decode that advances every active sequence by one
//! token per iteration. Each sequence's KV grows a block at a time from a shared
//! pool (no per-sequence worst-case reservation), and one batched forward serves
//! the whole running set - so more sequences stay resident and decode together.
//!
//! Self-contained: it owns its `Gpu` (with the batched paged kernels), a
//! `ParamStore` of the decoder weights, per-layer block pools, and the block
//! allocator. The forward math is the shared [`model::block`] Qwen3 block; only
//! the attention is paged + ragged-batched.

use std::collections::HashMap;

use gpu_core::select::{
    AutoTuner, CachedSelector, DefaultSelector, Dtype, KernelSelector, KernelVariant, Op, OpShape,
    DECODE_REGIME_MAX_ROWS,
};
use gpu_core::{DeviceBuffer, DeviceCaps, Gpu, Step};
use model::block::{self, KernelIds};
use model::dispatch::I8Scratch;
use model::ops::{Ops, Weight};
use model::kv_offload::{HostKvPool, KvOffload, KvOffloadError, OffloadStats};
use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use paramstore::ParamStore;

use crate::config::QwenConfig;

// The untiled whole-table gather `Self::embed_tiled`/`EMBED_TILE` replaced in
// `Self::batched_tape` - kept registered (and named) only as the oracle
// `embed_tiled_matches_the_plain_embed_kernel` dispatches directly, at a
// vocab safely under any real binding cap.
#[allow(dead_code)]
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const SILU_MUL: usize = 4;
const ADD2: usize = 5;
const ROPE_PAGED: usize = 6;
const KV_APPEND_B: usize = 7;
const SCORES_B: usize = 8;
const SOFTMAX_B: usize = 9;
const APPLY_B: usize = 10;
// int8 paged KV (dequant on read). The append kernel is the calibrated
// (clipped) one - the ONLY i8 append since the unclipped twin was deleted
// (audit F42): with the f32::MAX-sentinel clip table the kernel is
// bit-identical to that old twin by its own contract, so one kernel serves
// both the calibrated and the uncalibrated path.
const APPEND_I8_CLIPPED: usize = 11;
const SCORES_I8: usize = 12;
const APPLY_I8: usize = 13;
// Device-side greedy head: matmul -> row argmax, so decode never ships a
// [batch, vocab] logit block to the host.
const ARGMAX_ROW: usize = 14;
const ARGMAX_PART: usize = 15;
const ARGMAX_FINAL: usize = 16;
// Decode-regime kernels: selected per dispatch by row count.
const RMSNORM_ROWS: usize = 17;
const MATMUL_GEMV: usize = 18;
// Int8 weight path (A0): per-token activation quant + DP4A GEMMs with
// per-token x per-group dequant scales - the tile GEMM for prefill shapes,
// the packed GEMV for decode row counts.
const MAX_ABS_ROW: usize = 19;
const QUANT_PACK: usize = 20;
const MATMUL_I8_DYN: usize = 21;
const MATMUL_I8_GEMV: usize = 22;
// On-device decode window (A4): feed the argmax back as the next input and
// advance the paged metadata without a host round-trip.
const DECODE_FEED: usize = 23;
const DECODE_ADVANCE: usize = 24;
// Iterative on-device top-K extraction (W3): composes with ARGMAX_PART/
// ARGMAX_FINAL/ARGMAX_ROW above, so real (non-greedy) sampling reads back
// `[bsz, TOPK_CAPACITY]` candidates instead of the whole `[bsz, vocab]` row.
const TOPK_EXTRACT_STEP: usize = 25;
// The 128x128 register-tiled fp32 GEMM. It was MISSING from this engine's
// pipeline table entirely, so `Engine::mm` had only the decode GEMV and the
// naive kernel to choose between and every chunked-prefill chunk above
// `DECODE_REGIME_MAX_ROWS` ran one thread per output element - while the
// batched forward next door dispatched this same kernel at ~80x the rate.
const MATMUL_REG3: usize = 26;
// Coalesced paged scores: one workgroup per score, lanes split the head_dim
// reduction. Same Params and same output as SCORES_B; selected on the queried
// `workgroup_reductions`, since it carries a barrier the CPU JIT gates on.
const SCORES_B_WG: usize = 27;
// Split-K forward GEMM + its fold, for the skinny-M shapes a served step is made
// of. `matmul_reg3`'s tile grid is ceil(m/128)*ceil(n/128) and does not grow
// with k, so at m=128 it launches 16 workgroups on a 30-SM card: barely half
// the SMs get any work, and that one dispatch is a large share of a served
// step. See `matmul_reg3_splitk.wgsl` for the measured occupancy curve and the
// arithmetic that says splitting pays at THIS shape.
const MATMUL_REG3_SPLITK: usize = 28;
const SPLITK_REDUCE: usize = 29;
const PAGED_FLASH_PREFILL: usize = 30;
// M4.1: splits the fused QKV / gate-up GEMM's wide output into the compact,
// densely-packed buffers QK-norm/RoPE/KV-append and SwiGLU still need -
// `qwen35moe`'s own kernel-reuse note names this the correct existing tool
// for exactly this job (a wide strided row gathered into a fresh compact
// buffer), not `region_copy` (which requires src/dst to share one
// row_stride/off - it copies a sub-region, it cannot narrow one).
const CONCAT_SPLIT: usize = 31;
// M4.2: QK-norm + RoPE fused into one dispatch per q/k row (see
// `qknorm_rope_fused.wgsl`'s own header); the K-only sibling additionally
// folds the fp32 paged KV append into the same pass.
const QKNORM_ROPE_FUSED: usize = 32;
const QKNORM_ROPE_APPEND_FUSED: usize = 33;
// M4.3: RMSNorm fused with its own immediately-following int8 activation
// quant (see `rmsnorm_quant_fused.wgsl`'s own header) - one dispatch instead
// of `rms` -> `max_abs_row` -> `quant_pack`, never materialising the fp32
// intermediate at all on an all-int8-weight engine.
const RMSNORM_QUANT_FUSED: usize = 34;
// Host-RAM KV offload (`model::kv_offload`): gather a set of physical KV
// blocks out of the pools into one staging buffer (swap out), and scatter
// staged blocks back into whatever physical slots the allocator handed out
// (swap in). Never dispatched on the decode path - only when the scheduler
// preempts or resumes a whole sequence.
const KV_GATHER: usize = 35;
const KV_SCATTER: usize = 36;
// Vocab-tiled embedding gather (mirrors `qwen3::model::Qwen::embed_tiled`):
// `emb` is bound to a vocab SUB-RANGE per dispatch, so a `tok.weight` table
// larger than one storage binding (`max_storage_buffer_binding_size`, which
// wgpu clamps to `i32::MAX` = 2047 MiB on every backend) is still gathered
// correctly, in several passes each within the limit. See `Engine::
// embed_tiled`.
const EMBED_TILE: usize = 37;
// Column-tiled fp32 GEMM: `out[:, n_off..n_off+n_tile] = x @ Wᵀ`, `W` bound as
// a vocab/output-row SUB-RANGE while `out` stays a whole (unsliced) binding -
// `matmul_tile.wgsl`'s own `n_off`/`n_full` params place each tile's columns
// at the right STRIDED offset of a `[m, n_full]` row-major buffer, which no
// `step_sliced` byte-range alone could express. `Self::mm_into` reaches for
// this only when a `[n, k]` weight cannot be bound whole at all
// (`Self::fits_one_binding`) - the LM head at a real vocab is the one weight
// this engine holds that can cross `max_storage_buffer_binding_size`.
// `qwen3::model::Qwen::forward`'s own lm_head epilogue already uses exactly
// this kernel and tiling rule (`model::block::vocab_tiles_on`) for the same
// reason; this ports it here, generalised to `m > 1` rows (a batched decode
// step's `bsz`, not just a single decode row).
const MATMUL_TILE: usize = 38;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("rope_paged", kernels::ROPE_PAGED),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("paged_kv_append_i8_clipped_batched", kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED),
    ("paged_decode_scores_i8_batched", kernels::PAGED_DECODE_SCORES_I8_BATCHED),
    ("paged_decode_apply_i8_batched", kernels::PAGED_DECODE_APPLY_I8_BATCHED),
    ("argmax_row", kernels::ARGMAX_ROW),
    ("argmax_part", kernels::ARGMAX_PART),
    ("argmax_final", kernels::ARGMAX_FINAL),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
    ("decode_feed", kernels::DECODE_FEED),
    ("decode_advance", kernels::DECODE_ADVANCE),
    ("topk_extract_step", kernels::TOPK_EXTRACT_STEP),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("paged_decode_scores_wg", kernels::PAGED_DECODE_SCORES_WG),
    ("matmul_reg3_splitk", kernels::MATMUL_REG3_SPLITK),
    ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
    ("paged_flash_prefill", kernels::PAGED_FLASH_PREFILL),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("qknorm_rope_fused", kernels::QKNORM_ROPE_FUSED),
    ("qknorm_rope_append_fused", kernels::QKNORM_ROPE_APPEND_FUSED),
    ("rmsnorm_quant_fused", kernels::RMSNORM_QUANT_FUSED),
    ("kv_block_gather", kernels::KV_BLOCK_GATHER),
    ("kv_block_scatter", kernels::KV_BLOCK_SCATTER),
    ("embed_tile", kernels::EMBED_TILE),
    ("matmul_tile", kernels::MATMUL_TILE),
];

/// The `model::ops::Ops` façade's required kernel set (B7), registered on a
/// throwaway side `Gpu` (`Gpu::new_like`) purely so `from_map_with_gpu` can
/// call `Weight::upload` for its capability-aware quantize+upload - this
/// engine's own dispatch (`Engine::linear`/`Engine::mm8`/`Engine::tune_i8`)
/// never routes through `Ops::matmul`, so - unlike `qwen3::model::Qwen`'s
/// `self.ops` (built via `Gpu::share`, see that crate's `pipelines()` doc
/// comment) - this side `Gpu` never needs index-space compatibility with
/// `self.gpu`: `Weight::upload` only ever touches buffers, never builds a
/// `Step` this engine submits.
fn ops_kernel_list() -> &'static [(&'static str, &'static str)] {
    // The canonical facade set, from `model::ops::kernel_list`. This function
    // used to hand-maintain its own copy and famously drifted 15 kernels
    // short of what `Ops::new` requires -- the drift that made one shared
    // source worth having. There is nothing engine-specific to add here: this
    // side `Gpu` exists only so `Weight::upload` can quantize and upload, and
    // never has a `Step` submitted through it.
    model::ops::kernel_list()
}

/// Longest on-device decode window (tokens per host round-trip). The scheduler
/// picks `min(this, tokens remaining)`; the window trades one readback per
/// token for one per window, at the cost of up to `window - 1` wasted decode
/// steps when a sequence hits EOS mid-window.
pub const DECODE_WINDOW: usize = 4;

/// Workgroups a split-K GEMM aims to launch.
///
/// The same number `vae::blocks::DW_SPLITK_TARGET_WGS` was swept to for
/// `matmul_dw_reg_splitk`, on the same card and against the same defect (a tile
/// grid that does not grow with the contraction). That sweep found a RULE, not
/// a constant - every shape's optimum landed on ~288 workgroups - which is why
/// it transfers here rather than needing its own sweep to start from.
const SPLITK_TARGET_WGS: u32 = 288;

/// Ceiling on the split-K partial scratch, in f32 words (64 MiB).
///
/// Partials are `m * n * slices`, so an unbounded rule would let a large
/// prefill chunk allocate hundreds of megabytes to save a few milliseconds.
/// A shape whose partials do not fit simply keeps the plain kernel.
const SPLITK_SCRATCH_WORDS: u64 = 16 << 20;

/// Most k-slices a single GEMM is split into. Bounds both the scratch and the
/// fold's read amplification (the fold reads `slices` partials per output).
const SPLITK_MAX_SLICES: u32 = 48;

fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        silu_mul: SILU_MUL,
        // Every slot below is UNREGISTERED, not a stand-in index. This engine
        // never calls `block::gqa_fwd` (prefill and decode share the PAGED
        // attention kernels, `paged_decode_*`), never runs a backward, and
        // rotates through `rope_paged` rather than `block::rope_fwd`.
        //
        // These used to hold live indices for OTHER kernels - `RMSNORM` in the
        // RMSNorm-backward slots, `ROPE_PAGED` in the RoPE slots, `0` in the
        // GQA ones - which reads as "harmless placeholder" and is not: a
        // builder reaching one dispatches a real kernel against another
        // kernel's bindings and uniform. `UNREGISTERED` is out of range of
        // PIPELINES, so the same mistake is a panic instead.
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: block::UNREGISTERED,
        gqa_apply: block::UNREGISTERED,
        attn_softmax: block::UNREGISTERED,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: block::UNREGISTERED,
    }
}

// The decode-regime boundaries (max rows, argmax split vocab) live in the
// shared selection policy - `gpu_core::select` - not here: which kernel runs
// for a shape on a device is the selector's single job.

/// Chunks per row for the two-stage argmax; 256 threads per row saturates the
/// reduction without a large partial buffer (256*2 f32 per row).
const ARGMAX_CHUNKS: u32 = 256;

/// The largest `k` a real (non-greedy) sampling decode step will request -
/// device scratch for the iterative top-K extraction is sized to this bound.
/// 64 comfortably covers the standard `top_k = 40` default with headroom for
/// a wider top-p nucleus; a request asking for more is clamped to this.
pub const TOPK_CAPACITY: u32 = 64;

/// One batched decode step's paged-KV metadata, as
/// [`Engine::append_meta`] returns it: `(batch size, positions, sequence
/// lengths, block ids, in-block offsets, the flattened block table)`. The last
/// four are one entry per sequence except `bt`, which is
/// `bsz * max_blocks_per_seq` wide.
type BatchMeta = (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn fb(x: f32) -> u32 {
    x.to_bits()
}

/// M4.1: gather the `width`-wide column range starting at `off` out of a
/// `[rows, ctot]` fused GEMM output into a fresh compact `[rows, width]`
/// buffer - `concat_split.wgsl` with `H=W=1` (`Params: [N, Ctot, Csrc, c_off,
/// H, W]`), the existing kernel `qwen35moe::model`'s own kernel-reuse note
/// names for exactly this "narrow a wide strided row" job.
fn concat_split_step(g: &Gpu, src: &DeviceBuffer, dst: &DeviceBuffer, rows: u32, ctot: u32, width: u32, off: u32) -> Step {
    g.step(CONCAT_SPLIT, &[src, dst], &[rows, ctot, width, off, 1, 1], rows * width)
}

/// Fault injection (G): `brain perf faults` arms a fault, the next pass
/// through its check point fires it. Feature-gated - a build without
/// `fault-injection` compiles the sink to nothing, so there is no release
/// cost and nothing to accidentally arm in production.
#[cfg(feature = "fault-injection")]
pub mod fault {
    use std::sync::atomic::{AtomicBool, Ordering};
    static KERNEL_FAILURE: AtomicBool = AtomicBool::new(false);

    /// Arm a one-shot kernel-dispatch failure: the next batched decode
    /// panics at its dispatch point, as a real device fault would surface.
    pub fn arm_kernel_failure() {
        KERNEL_FAILURE.store(true, Ordering::SeqCst);
    }
    pub(crate) fn take_kernel_failure() -> bool {
        KERNEL_FAILURE.swap(false, Ordering::SeqCst)
    }
}

/// A batched-forward input: token ids (embedded via `tok.weight`) or ready-made
/// per-row embeddings written straight into the residual stream (the tts Talker
/// feeds codec/text-conditioned embeddings rather than ids).
///
/// `Copy` (every variant holds only a `Copy` slice reference or nothing) so
/// M6.3's tape cache can pass one value to `write_batch_meta`/
/// `write_batch_input`/`batched_tape` without fighting the borrow checker
/// over a shared reference.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    Tokens(&'a [u32]),
    Embeds(&'a [f32]),
    /// Token ids already resident in the engine's `tok_buf` - the on-device
    /// decode window (A4): `decode_feed` wrote them from the previous step's
    /// argmax, and `decode_advance` already advanced the paged metadata, so
    /// the forward performs NO host writes at all.
    Resident,
}

/// M4.1: the fused-QKV leaf name (`attn.wqkv.weight`, `[hq+2*hkv, d]`
/// row-major - `wq`'s rows, then `wk`'s, then `wv`'s), never written to a
/// checkpoint - it exists only in `Engine::lin_weights`, built at weight-load
/// time from the split leaves `decoder_param_list`/`import.rs` still name.
const WQKV: &str = "attn.wqkv.weight";
/// M4.1: the fused gate/up leaf name (`mlp.gateup.weight`, `[2*ff, d]`
/// row-major - `gate`'s rows, then `up`'s), same non-checkpoint status as
/// [`WQKV`].
const WGATEUP: &str = "mlp.gateup.weight";

/// Is `name` one of the five split projections M4.1 folds into [`WQKV`] /
/// [`WGATEUP`] - the loop over `crate::q8::Q8::LINEARS` skips these (built as
/// fused weights instead), and `ParamStore` never holds them either (see
/// `from_map_with_gpu`'s `roles` filter): unlike `wo`/`down`, nothing reads
/// them by their split name once the engine is built.
///
/// A suffix match, not an exact one - `crate::q8::Q8::is_i8_linear`'s own
/// idiom - so this accepts both a bare leaf (`"attn.wq.weight"`, from the
/// `Q8::LINEARS` loop) and a full per-layer name (`"blocks.5.attn.wq.weight"`,
/// from the `ParamStore` `roles` filter) with one function.
fn is_fused_source_leaf(name: &str) -> bool {
    ["attn.wq.weight", "attn.wk.weight", "attn.wv.weight", "mlp.gate.weight", "mlp.up.weight"]
        .iter()
        .any(|leaf| name.ends_with(leaf))
}

/// One decoder-param leaf name → element count (mirrors the decode weight set).
fn decoder_param_list(cfg: &QwenConfig) -> Vec<(String, usize)> {
    let (d, ff) = (cfg.d_model as usize, cfg.d_ff as usize);
    let (hq, hkv, hd) = (cfg.q_dim() as usize, cfg.kv_dim() as usize, cfg.head_dim as usize);
    let mut out = Vec::new();
    for l in 0..cfg.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        out.push((p("ln1.weight"), d));
        out.push((p("attn.wq.weight"), hq * d));
        out.push((p("attn.wk.weight"), hkv * d));
        out.push((p("attn.wv.weight"), hkv * d));
        out.push((p("attn.q_norm.weight"), hd));
        out.push((p("attn.k_norm.weight"), hd));
        out.push((p("attn.wo.weight"), d * hq));
        out.push((p("ln2.weight"), d));
        out.push((p("mlp.gate.weight"), ff * d));
        out.push((p("mlp.up.weight"), ff * d));
        out.push((p("mlp.down.weight"), d * ff));
    }
    out.push(("norm.weight".to_string(), d));
    out.push(("tok.weight".to_string(), cfg.vocab as usize * d)); // embedding gather
    out
}

/// Per-(K or V)-buffer word counts for the paged KV pool at this sizing - the
/// ONE place the int8/fp32 layout is derived. `from_map_with_gpu`'s
/// allocation loop and [`kv_pool_bytes`] both call this rather than each
/// carrying their own copy of the arithmetic, so a caller reading
/// `Engine::kv_pool_bytes` back is reading what was actually allocated, never
/// a second guess at it.
fn kv_pool_words(cfg: &QwenConfig, block_size: u32, num_blocks: u32, kv_int8: bool) -> (u64, u64) {
    let hkv = cfg.kv_dim() as u64;
    let n_kv = cfg.n_kv_heads as u64;
    let slots = num_blocks as u64 * block_size as u64;
    let pool_words = if kv_int8 { slots * hkv / 4 } else { slots * hkv };
    let scale_words = if kv_int8 { slots * n_kv } else { 0 };
    (pool_words, scale_words)
}

/// Device words ONE physical KV block costs across the whole model - every
/// layer's K and V pool words plus, on an int8 pool, their per-`(slot,
/// kv-head)` dequant scales. This is the width
/// `model::kv_offload::HostKvPool` holds per block and the stride one block's
/// staging record spans, so it is derived from [`kv_pool_words`] (the one
/// place the pool layout lives) at `num_blocks = 1` rather than restated.
pub fn kv_block_words(cfg: &QwenConfig, block_size: u32, kv_int8: bool) -> usize {
    let (pool_words, scale_words) = kv_pool_words(cfg, block_size, 1, kv_int8);
    cfg.n_layers as usize * 2 * (pool_words + scale_words) as usize
}

/// Device bytes the paged KV pool costs at this sizing: K + V pools (packed
/// int8 4/`u32` + a fp32 scale per `(token slot, kv-head)`, or plain fp32) for
/// every layer.
///
/// The exact ratio `fp32 / int8` is `4·head_dim / (head_dim + 4)`: close to
/// the 4-to-1 limit at Qwen3's `head_dim=128`, but a DIFFERENT number at any
/// other `head_dim` (well short of 3-to-1 at `QwenConfig::tiny()`'s
/// `head_dim=8`) - see
/// `kv_pool_bytes_identity_holds_at_the_real_shape`, which pins both.
pub fn kv_pool_bytes(cfg: &QwenConfig, block_size: u32, num_blocks: u32, kv_int8: bool) -> u64 {
    let (pool_words, scale_words) = kv_pool_words(cfg, block_size, num_blocks, kv_int8);
    let n_layers = cfg.n_layers as u64;
    n_layers * 2 * (pool_words + scale_words) * 4 // K + V, every layer, 4 bytes/word
}

/// Device bytes `Scratch::{scores,probs}` costs at this sizing (M2.4) - the
/// single largest serving scratch buffer before this milestone (this
/// campaign's own audit finding). Decode's own worst case (`max_batch *
/// n_heads * cap`) is UNCONDITIONAL: decode never gets `paged_flash_prefill`'s
/// fused kernel (`Op::PagedAttentionFused`'s own doc - M2.1/M2.2's own
/// measured non-win for the decode-shaped fused kernels, inherited at every
/// dtype). Causal-chunk prefill's own worst case (`max_prefill^2 * n_heads`,
/// the `[nh,N,N]` shape this function's own call site originally derived it
/// from) is only shed when `fused_prefill_available` - the SAME condition
/// `run_batched_steps`'s own dispatch gates on
/// (`Op::PagedAttentionFused`'s selector, i.e. F32 KV storage AND
/// `caps.workgroup_reductions`), computed by the ONE call site that has both
/// `kv_int8` and `caps` (`Engine::from_map_with_gpu`), not re-derived here:
/// a device without `workgroup_reductions` (the CPU JIT) falls back to the
/// triad for causal-chunk prefill exactly as it always did, so shrinking this
/// buffer there would be an out-of-bounds write waiting to happen, not a
/// memory saving.
pub fn paged_attn_scratch_bytes(cfg: &QwenConfig, max_batch: u32, max_prefill: u32, cap: u32, fused_prefill_available: bool) -> u64 {
    let nh = cfg.n_heads as u64;
    let words = if fused_prefill_available {
        max_batch as u64 * nh * cap as u64
    } else {
        let b = max_batch.max(max_prefill) as u64;
        (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh)
    };
    words * 2 * 4 // scores + probs, 4 bytes/word
}

/// Whether `cfg` can take int8 KV at all: the append kernels pack 4 int8
/// lanes into one `u32`, so a packed word must stay within one head (else its
/// lanes would span two heads' scales) - `head_dim % 4 == 0`. Every shipped
/// Qwen3 config (`head_dim` 128) and `QwenConfig::tiny()` (`head_dim` 8)
/// satisfy this; an imported HF config with an unusual `head_dim` might not.
///
/// The three DEFAULT-selecting call sites (`QwenResident::activate`,
/// `qwen_cli::serve`, the perf `SynthSpec` builders) call this FIRST and
/// degrade to fp32 with a printed reason when it is `false` - an explicit
/// `kv_int8: true` request still hits `from_map_with_gpu`'s hard assert,
/// because a caller that asked for int8 by name should hear about a mismatch
/// as a failure, not a silent substitution. An implicit default should not
/// turn an unusual imported checkpoint into a serving-process panic nobody
/// asked for.
pub fn kv_int8_supported(cfg: &QwenConfig) -> bool {
    cfg.head_dim.is_multiple_of(4)
}

/// A running sequence: its block table, generated tokens, and completion flag.
struct Seq {
    table: BlockTable,
    generated: Vec<u32>,
    done: bool,
}

/// Batched scratch (sized for `max_batch` rows), reused every iteration.
struct Scratch {
    res: Vec<DeviceBuffer>, // n_layers+1, each [B*d]
    xn1: DeviceBuffer,
    /// M4.1: the fused QKV GEMM's raw `[b, hq+2*hkv]` output, before
    /// `concat_split` narrows it into `q_pre`/`k_pre`/`v` below.
    qkv_pre: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    /// M4.1: the fused gate/up GEMM's raw `[b, 2*ff]` output, before
    /// `concat_split` narrows it into `gate_pre`/`up` below.
    gateup_pre: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    // per-step metadata (uploaded each iteration)
    tok_buf: DeviceBuffer,
    pos_buf: DeviceBuffer,
    seqlen_buf: DeviceBuffer,
    blk_buf: DeviceBuffer,
    off_buf: DeviceBuffer,
    bt_buf: DeviceBuffer,
    /// `[DECODE_WINDOW-1, max_batch, 3]` device (block, offset, bt_index)
    /// schedule for the on-device decode window (A4).
    sched_buf: DeviceBuffer,
    /// `[DECODE_WINDOW, max_batch]` window token history (greedy indices).
    hist_buf: DeviceBuffer,
}

/// Paged, batched Qwen3 serving engine.
pub struct Engine {
    cfg: QwenConfig,
    gpu: Gpu,
    /// The device's capabilities, read once at build - the selector's input.
    caps: DeviceCaps,
    /// The shared static policy (`DefaultSelector`, memoised) every op below
    /// `Op::MatMul`'s int8 arm resolves through, and `Op::MatMul`'s own
    /// fallback for an int8 shape `self.tuned_i8` has no measurement for.
    /// An `Arc<dyn KernelSelector>` (not the bare `CachedSelector<
    /// DefaultSelector>` this used to be) so it is injectable into any
    /// `model::ops::Ops` this engine builds (`Ops::with_selector`) - the
    /// measured `tuned_i8` table stays a plain field (see its own doc): it is
    /// consulted directly by `Self::mm8`, not wrapped into this selector.
    selector: std::sync::Arc<dyn KernelSelector>,
    ps: ParamStore,
    block_size: u32,
    max_batch: u32,
    max_blocks_per_seq: u32,
    max_prefill: u32,
    cap: u32,
    /// Scratch for the split-K GEMM's per-slice partials, and its size in f32
    /// words. `None` on a device that cannot run the kernel, which makes
    /// `splitk_slices` return `None` and every GEMM take the plain path.
    splitk_part: Option<DeviceBuffer>,
    splitk_cap: Option<u64>,
    alloc: BlockAllocator,
    /// Prompt-prefix cache (D): full prompt blocks are indexed after prefill
    /// and adopted (shared, refcounted) by later prompts with the same prefix,
    /// so prefill computes only the unmatched tail. Purely a prefill
    /// optimisation - decode never touches it.
    prefix: PrefixCache,
    /// Prefix-reuse counters: tokens looked up / tokens served from the cache.
    prefix_lookup_tokens: u64,
    prefix_hit_tokens: u64,
    pool_k: Vec<DeviceBuffer>,
    pool_v: Vec<DeviceBuffer>,
    // int8 KV: pools hold packed int8 (4/u32, ~4x smaller) + per-(token,kv-head)
    // dequant scales. Empty when kv_int8 is false (fp32 pools).
    //
    // INVARIANT: `pool_k`/`pool_v` and `scales_k`/`scales_v` MUST both be
    // addressed by the same `slot = physical*block_size + offset` -- this is
    // what makes `PrefixCache` block sharing work for int8 with NO
    // scales-aware code anywhere in `paged.rs` (the allocator only ever
    // reasons about physical block ids). If a shared block's scale were ever
    // read from a different slot than its pool words, `warm_prefill_is_
    // identical_to_cold`'s int8 arm is what would fail.
    kv_int8: bool,
    scales_k: Vec<DeviceBuffer>,
    scales_v: Vec<DeviceBuffer>,
    /// What [`kv_pool_bytes`] computed for this sizing - recorded once at
    /// construction (not re-derived by the accessor) so it can never drift
    /// from what `pool_k`/`pool_v`/`scales_k`/`scales_v` actually allocated.
    kv_pool_bytes: u64,
    /// What [`paged_attn_scratch_bytes`] computed for this sizing (M2.4) -
    /// recorded once at construction, same reason as `kv_pool_bytes` above,
    /// so it can never drift from what `sc.scores`/`sc.probs` actually
    /// allocated.
    scratch_bytes: u64,
    /// `Some` uploads real calibrated ceilings into `clip_k`/`clip_v`; `None`
    /// (the default) keeps the f32::MAX sentinel there, which the append
    /// kernel's contract documents as bit-identical to the deleted unclipped
    /// twin (audit F42) - see [`Engine::set_kv_calib`].
    kv_calib: Option<model::kvcalib::KvCalib>,
    /// Per-layer `[n_kv]` clip-ceiling upload buffers (allocated whenever
    /// `kv_int8`; MAX-sentinel-filled until a real calibration is installed).
    clip_k: Vec<DeviceBuffer>,
    clip_v: Vec<DeviceBuffer>,
    /// Int8 WEIGHT path (A0): every linear this engine dispatches - the 7
    /// per-layer projections (`blocks.<l>.<leaf>`) plus the LM head
    /// (`cfg.head_weight()`) - as a `model::ops::Weight` (B7), packed 4/u32
    /// (a quarter of the weight bytes in the bandwidth-bound decode regime) when
    /// `weights_int8` was requested AND the device's `int8_dot` capability
    /// allows it (`Weight::upload`'s own `want.promote(caps.numeric)` gate -
    /// never the case on a device whose caps report no packed-int8 path),
    /// else `F32`. Replaces the old `w8`/`head8` pair (an `Option` wrapping
    /// `q8`'s own `Q8`, and one wrapping its `Lin8`, respectively):
    /// dispatch (`Self::linear`, below) reads whatever tier a `Weight` value
    /// carries, never a separate on/off flag.
    /// Named `lin_weights` (not `weights`) to stay distinct from this
    /// module's own `weights: &HashMap<String, Vec<f32>>` constructor
    /// parameter (the raw host checkpoint tensors this is built FROM).
    lin_weights: HashMap<String, Weight>,
    /// Int8 activation-quantization scratch (`model::dispatch::I8Scratch` -
    /// the SAME struct `model::ops::Ops::act` wraps, B3's façade) - `Some`
    /// only when at least one `Weight` in `weights` is `I8` (mirrors the old
    /// `w8.is_some()` gate); nothing ever reads it otherwise, so there is
    /// nothing to allocate. Sized once for this engine's widest row count and
    /// re-quantized (overwritten in place) every layer's forward, exactly
    /// like the old `Q8::sx`/`Q8::xq` it replaces.
    i8_scratch: Option<I8Scratch>,
    /// Measured GEMV/tile choices for the int8 linears (S5), keyed by
    /// `(row bucket, n, k)` - tuned once at build on THIS device (persisted
    /// per adapter), so the hot path never measures. Empty on fp32 engines.
    /// Looked up directly by `Self::mm8` (a plain `HashMap`, not folded into
    /// `self.selector`); `self.selector` is `Op::MatMul`'s FALLBACK for a
    /// shape this table has no measurement for, not a wrapper around it.
    tuned_i8: HashMap<(u32, u32, u32), KernelVariant>,
    sc: Scratch,
    /// `[max_batch, vocab]` decode logits, and `[max_batch]` argmax indices.
    logits_dev: DeviceBuffer,
    argmax_dev: DeviceBuffer,
    /// `[max_batch, ARGMAX_CHUNKS, 2]` partial (value, index) pairs for the
    /// two-stage argmax reduction.
    argmax_part_dev: DeviceBuffer,
    /// `[max_batch, TOPK_CAPACITY]` on-device top-K extraction output: the
    /// row's best-to-worst logit values and their vocab indices, filled by
    /// `submit_topk_head` iterating the argmax pair + `topk_extract_step`.
    topk_vals_dev: DeviceBuffer,
    topk_idx_dev: DeviceBuffer,
    /// M6.3: the decode tape (uniform buffers + bind groups `batched_tape`
    /// records), keyed by `bsz` - a pure function of `bsz` alone once
    /// `causal_chunk` is `false` and the input is `Tokens`/`Resident` (every
    /// kernel choice, buffer identity and uniform PARAMETER in that tape
    /// comes from this engine's own fixed config; the only thing that varies
    /// step to step is buffer CONTENTS, written separately by
    /// `write_batch_meta`/`write_batch_input` before a cached tape is
    /// replayed) - so it is recorded once per bucket and reused unchanged,
    /// instead of a fresh uniform + bind group per dispatch every step. Never
    /// populated above `DECODE_REGIME_MAX_ROWS` (prefill's row count is not
    /// bucketed) or for `Input::Embeds` (its tape has no embed step at all,
    /// a structurally different shape at the same `bsz`) - both keep
    /// rebuilding via `Self::run_batched_steps`, unchanged from before this
    /// cache existed.
    tape_cache: HashMap<u32, Vec<Step>>,
    /// Host-RAM KV offload (`model::kv_offload`): the host pool demoted
    /// sequences' KV lives in. Zero-capacity (offload off) until
    /// [`Engine::set_kv_offload_bytes`] is called, so an engine nobody asked
    /// to offload allocates and copies nothing.
    host_kv: HostKvPool,
    /// Words one physical block occupies across all layers - `host_kv`'s block
    /// width and the swap staging stride. See [`kv_block_words`].
    kv_block_words: usize,
    /// Staging buffers the swap moves through, built on first use (an engine
    /// that never swaps never pays for them).
    swap: Option<SwapBufs>,
}

/// Device buffers one swap chunk moves through. A whole sequence is
/// transferred `blocks` at a time so the staging cost is bounded no matter how
/// long the sequence is: one `write`/`read` per chunk, `2 * n_layers` (fp32) or
/// `4 * n_layers` (int8) gather/scatter dispatches inside it.
struct SwapBufs {
    /// `[blocks]` physical block ids for the chunk being moved.
    ids: DeviceBuffer,
    /// `[blocks * kv_block_words]`, block-major.
    staging: DeviceBuffer,
    /// Blocks one chunk carries.
    blocks: u32,
}

/// Staging budget for one KV swap chunk. Large enough that a swap is a handful
/// of PCIe transfers rather than one per block (the measured device->host rate
/// on this class of card is ~1.2 GB/s, so a 64 MiB chunk is ~50 ms of real
/// transfer - far past the per-submit overhead it amortises), small enough to
/// be irrelevant next to the pool itself.
const KV_SWAP_STAGING_BYTES: u64 = 64 << 20;

impl Engine {
    /// Build from an in-memory decoder weight map (tests / embedded weights).
    /// `num_blocks` physical blocks of `block_size` tokens, up to `max_batch`
    /// concurrent sequences of at most `max_blocks_per_seq * block_size` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn from_map(cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        Self::from_map_with_gpu(Gpu::new(PIPELINES), cfg, weights, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    /// [`Engine::from_map`] on an EXISTING device (F1 warm start): the caller's
    /// `Gpu` parents this engine via [`Gpu::new_like`], so building another
    /// engine costs pipeline compilation only - never a second full device
    /// init. This is what a serving process or the residency executor should
    /// use: one device per process, many engines on it (many concurrent
    /// devices on one card is both slow and hostile to the driver).
    #[allow(clippy::too_many_arguments)]
    pub fn from_map_on(parent: &Gpu, cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        Self::from_map_with_gpu(parent.new_like(PIPELINES), cfg, weights, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_map_with_gpu(gpu: Gpu, cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        // Int8 weights are capability-driven, never assumed: the request only
        // takes effect where the packed-dot GEMM executes (the selector's
        // PackedInt8 gate). Elsewhere - the CPU JIT - fp32 weights stay, and
        // the fallback is said out loud rather than silently absorbed.
        let caps = gpu.caps();
        // The gate is the CAPABILITY, not the selector's head: which int8
        // variant is best at some shape is a tuning question, but whether the
        // packed-dot kernels execute at all is numeric.int8_dot.
        let w8_on = weights_int8 && caps.numeric.int8_dot;
        if weights_int8 && !w8_on {
            eprintln!("serve: int8 weights requested but this device has no packed-int8 path; using fp32 weights");
        }
        // The 7 per-layer linears live in the int8 bank when it is on - loading
        // them into the fp32 ParamStore as well would keep both copies resident
        // and forfeit the memory the quantisation buys.
        //
        // M4.1: `attn.{wq,wk,wv}` and `mlp.{gate,up}` are ALSO excluded here,
        // unconditionally (not just when `w8_on`) - `lin_weights` below builds
        // them as two fused weights (`attn.wqkv.weight`, `mlp.gateup.weight`)
        // read straight from the host `weights` map, so a `ps` copy of the
        // split leaves would sit unused (nothing reads them by their split
        // names any more) and cost real resident memory for nothing.
        let roles = decoder_param_list(&cfg)
            .into_iter()
            .filter(|(n, _)| !(is_fused_source_leaf(n) || (w8_on && crate::q8::Q8::is_i8_linear(n))))
            .map(|(n, c)| (n, c, paramstore::Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, weights);
        let head = weights.get(cfg.head_weight()).cloned().unwrap_or_else(|| weights.get("tok.weight").cloned().expect("head weight"));

        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        // Scratch rows serve both decode (max_batch sequences) and prefill (a whole
        // prompt of up to max_prefill tokens processed in one forward).
        let b = max_batch.max(max_prefill) as u64;
        let cap = max_blocks_per_seq * block_size;
        // Split-K scratch, sized for the widest GEMM this engine can present
        // (the widest row count it admits, times the widest output). Allocated
        // once; `splitk_slices` refuses any shape whose partials exceed it, so
        // the buffer is a hard ceiling rather than a hint.
        let widest_n = cfg.d_ff.max(cfg.q_dim()).max(cfg.d_model) as u64;
        let widest_m = max_batch.max(max_prefill) as u64;
        // Sized for the MAX slice count the rule can emit, not a guess. Sizing
        // it for 8 silently refused the wide shapes - only `wk`/`wv` fitted, so
        // 56 of 196 GEMMs split and the rest kept the starved kernel.
        let splitk_cap = (widest_m * widest_n * SPLITK_MAX_SLICES as u64).min(SPLITK_SCRATCH_WORDS);
        let (splitk_part, splitk_cap) = if gpu.caps().workgroup_reductions && splitk_cap > 0 {
            (Some(gpu.storage(splitk_cap)), Some(splitk_cap))
        } else {
            (None, None)
        };
        // scores/probs hold decode [rows,nh,cap] always, and prefill causal
        // [nh,N,N] too unless the fused kernel actually replaces the triad
        // there - see `paged_attn_scratch_bytes`'s own doc for the M2.4
        // shrink. `fused_prefill_available` asks `Op::PagedAttentionFused`'s
        // OWN selector (the exact same call `run_batched_steps`'s dispatch
        // makes, m/n irrelevant to that Op - see its own `candidates()` arm)
        // rather than hand-duplicating its dtype/capability rule here, so
        // the two can never drift apart: a device without `caps.
        // workgroup_reductions` (the CPU JIT) or an int8-KV engine both
        // correctly keep the full, unshrunk size.
        let fused_prefill_available = DefaultSelector.select(
            Op::PagedAttentionFused,
            OpShape { m: 0, n: 0, k: 1, dtype: if kv_int8 { Dtype::I8 } else { Dtype::F32 } },
            &caps,
        ) == KernelVariant::FusedFlash;
        // `/2/4` undoes `paged_attn_scratch_bytes`'s own "scores+probs, 4
        // bytes/word" to recover the per-buffer WORD count `st` (below) wants.
        let scratch_bytes_val = paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, fused_prefill_available);
        let bcap = scratch_bytes_val / 2 / 4;
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(b * d));
        }
        // int8 pools pack 4 values/u32 (÷4 words) + a scale per (token slot, kv-head).
        // The WGSL append kernels require head_dim % 4 == 0 (a packed u32 must
        // stay within one head, else its 4 lanes span two heads' scales) --
        // asserted here, once, rather than left as an unguarded `/ 4` that
        // would silently under-allocate the pool for an odd head_dim.
        if kv_int8 {
            assert!(cfg.head_dim.is_multiple_of(4), "int8 KV requires head_dim % 4 == 0 (got {})", cfg.head_dim);
        }
        let (pool_words, scale_words) = kv_pool_words(&cfg, block_size, num_blocks, kv_int8);
        let mut pool_k = Vec::new();
        let mut pool_v = Vec::new();
        let mut scales_k = Vec::new();
        let mut scales_v = Vec::new();
        let mut clip_k = Vec::new();
        let mut clip_v = Vec::new();
        // The MAX sentinel: the clipped append kernel with this table is
        // bit-identical to the deleted unclipped twin (audit F42), so the
        // uncalibrated path allocates it once and never branches.
        let max_row = vec![f32::MAX; cfg.n_kv_heads as usize];
        for _ in 0..cfg.n_layers {
            pool_k.push(st(pool_words));
            pool_v.push(st(pool_words));
            if kv_int8 {
                scales_k.push(st(scale_words));
                scales_v.push(st(scale_words));
                let ck = st(cfg.n_kv_heads as u64);
                gpu.write(&ck, bytemuck::cast_slice(&max_row));
                clip_k.push(ck);
                let cv = st(cfg.n_kv_heads as u64);
                gpu.write(&cv, bytemuck::cast_slice(&max_row));
                clip_v.push(cv);
            }
        }
        let kv_pool_bytes_val = cfg.n_layers as u64 * 2 * (pool_words + scale_words) * 4;
        let block_words = kv_block_words(&cfg, block_size, kv_int8);
        let sc = Scratch {
            res,
            xn1: st(b * d),
            qkv_pre: st(b * (hq + 2 * hkv)),
            q_pre: st(b * hq),
            q: st(b * hq),
            k_pre: st(b * hkv),
            k: st(b * hkv),
            v: st(b * hkv),
            ctx: st(b * hq),
            xmid: st(b * d),
            xn2: st(b * d),
            gateup_pre: st(b * 2 * ff),
            gate_pre: st(b * ff),
            up: st(b * ff),
            h: st(b * ff),
            proj: st(b * d),
            mlp_out: st(b * d),
            xn_final: st(b * d),
            scores: st(bcap),
            probs: st(bcap),
            tok_buf: st(b),
            pos_buf: st(b),
            seqlen_buf: st(b),
            blk_buf: st(b),
            off_buf: st(b),
            bt_buf: st(b * max_blocks_per_seq as u64),
            sched_buf: st((DECODE_WINDOW as u64 - 1) * max_batch as u64 * 3),
            hist_buf: st(DECODE_WINDOW as u64 * max_batch as u64),
        };
        // Per-linear weights (B7): every layer's 7 projections plus the LM
        // head, as `model::ops::Weight` - `Weight::upload`'s own `want.
        // promote(caps.numeric)` is the ONE capability gate for int8 (agrees
        // with `w8_on` above by construction: both read the SAME `caps.
        // numeric.int8_dot`). Built via a throwaway `Ops` on a side `Gpu`
        // (`Gpu::new_like`) purely for `Weight::upload`'s convenience
        // (capability-aware quantize+upload): `Weight::upload` only touches
        // buffers and never asks a selector anything, so which selector this
        // particular `Ops` holds is irrelevant here - unlike `qwen3::
        // model::Qwen`'s `self.ops` (see that crate's `pipelines()` doc
        // comment), index-space compatibility with `self.gpu` is NOT
        // required either, since `Weight::upload` never builds a `Step` this
        // engine would submit.
        //
        // **This engine still keeps its OWN GEMM dispatch
        // (`Self::linear`/`Self::mm`/`Self::mm8`), rather than calling
        // `Ops::matmul` directly - but no longer because the SELECTOR is
        // unreachable (`Ops::with_selector` fixed that).** Two real, narrower
        // reasons remain: (1) `Self::mm_into`'s split-K fold has no
        // equivalent in `Ops::matmul` at all (`select::KernelVariant` has no
        // split-K member - splitting is a dispatch-COUNT decision orthogonal
        // to which kernel variant runs); (2) this engine's int8 activations
        // live in ONE persistent, reused `I8Scratch` (`self.i8_scratch`,
        // requantized in place every layer), while `Ops::act` allocates a
        // FRESH `I8Scratch` per call - fine for a model forward pass that
        // quantizes once per layer, a real per-decode-step allocation
        // regression for a hot serving loop. What WAS unreachable - the
        // measured `Self::tune_i8` policy - is now `self.selector`, an
        // injectable `Arc<dyn KernelSelector>` (`model::ops::Ops::
        // with_selector`'s whole reason to exist), so `Self::mm8` asks the
        // SAME kind of seam `Ops::matmul` does instead of hand-checking a
        // private `HashMap` before falling back to a second, separate
        // selector field.
        let want = if weights_int8 { Dtype::I8 } else { Dtype::F32 };
        let ops = Ops::new(gpu.new_like(ops_kernel_list())).unwrap_or_else(|e| panic!("serve: Ops::new: {e}"));
        let (dm, ffm) = (cfg.d_model as usize, cfg.d_ff as usize);
        let (hqm, hkvm) = (cfg.q_dim() as usize, cfg.kv_dim() as usize);
        let dims = move |leaf: &str| -> (usize, usize) {
            match leaf {
                "attn.wq.weight" => (hqm, dm),
                "attn.wk.weight" | "attn.wv.weight" => (hkvm, dm),
                "attn.wo.weight" => (dm, hqm),
                "mlp.gate.weight" | "mlp.up.weight" => (ffm, dm),
                "mlp.down.weight" => (dm, ffm),
                other => panic!("serve: unknown linear {other}"),
            }
        };
        let mut lin_weights: HashMap<String, Weight> = HashMap::new();
        for l in 0..cfg.n_layers as usize {
            for leaf in crate::q8::Q8::LINEARS {
                if is_fused_source_leaf(leaf) {
                    continue; // folded into the fused weights built below
                }
                let name = format!("blocks.{l}.{leaf}");
                let (wn, wk) = dims(leaf);
                let w = if w8_on {
                    let raw = weights.get(&name).unwrap_or_else(|| panic!("serve: missing weight {name}"));
                    Weight::upload(&ops, raw, wn, wk, Dtype::I8)
                } else {
                    Weight::F32 { w: ps.w(&name).clone(), n: wn as u32, k: wk as u32 }
                };
                lin_weights.insert(name, w);
            }
            // M4.1: fused Q/K/V and gate/up projections - one GEMM replaces
            // three (Q/K/V) and one replaces two (gate/up). The concat is
            // done HERE, at engine weight-load time, by row-concatenating the
            // split tensors' own host data straight out of `weights` (the
            // same host map the split leaves above are read from, and the
            // same one `import.rs` writes unmodified to the on-disk
            // checkpoint - `W:[out,in]` is row-major, so concatenating along
            // `out` is exactly concatenating the flat row-major arrays end to
            // end). Bit-identical to the split path: `Weight::upload`
            // quantizes group-wise PER OUTPUT ROW (`model::int8::
            // quantize_weight`), so concatenating rows first and quantizing
            // once produces exactly the same packed bytes and scales as
            // quantizing each split matrix and concatenating the packed rows
            // - see `fused_qkv_and_gateup_are_bit_identical_to_split` for
            // both dtypes.
            let fused_weight = |leaves: &[&str], n_total: usize| -> Weight {
                let mut raw = Vec::with_capacity(n_total * dm);
                for leaf in leaves {
                    let name = format!("blocks.{l}.{leaf}");
                    raw.extend_from_slice(weights.get(&name).unwrap_or_else(|| panic!("serve: missing weight {name}")));
                }
                Weight::upload(&ops, &raw, n_total, dm, want)
            };
            lin_weights.insert(format!("blocks.{l}.{WQKV}"), fused_weight(&["attn.wq.weight", "attn.wk.weight", "attn.wv.weight"], hqm + 2 * hkvm));
            lin_weights.insert(format!("blocks.{l}.{WGATEUP}"), fused_weight(&["mlp.gate.weight", "mlp.up.weight"], 2 * ffm));
        }
        let head_name = cfg.head_weight().to_string();
        // The head never lived in `ps` (untied models have no `lm_head.
        // weight` in `decoder_param_list` at all), so - unlike the 7
        // per-layer linears above - there is no existing buffer to `.clone()`
        // for the `F32` tier either; `Weight::upload` always uploads it
        // fresh, exactly the cost the old `head_dev = gpu.storage_init(...)`
        // this replaces already paid.
        lin_weights.insert(head_name, Weight::upload(&ops, &head, cfg.vocab as usize, cfg.d_model as usize, want));
        // Int8 activation-quantization scratch (`model::dispatch::I8Scratch`,
        // the same struct `Ops::act` wraps) - one slot per distinct K width
        // among the 7 linears (`d`/`hq`/`ff`; the head shares `d`), reused
        // (re-quantized in place) every layer's forward, exactly like the
        // old `Q8::sx`/`Q8::xq` this replaces. `None` on an all-fp32 engine:
        // nothing ever reads it, so nothing is allocated.
        let i8_scratch = if w8_on { Some(I8Scratch::new(&gpu, b, b, &[d as u32, hq as u32, ff as u32])) } else { None };
        // S5: measure the GEMV/tile crossover for THIS device's int8 shapes at
        // build time (a few ms; persisted per adapter + kernel sources), so
        // the hot path only ever looks the choice up. Row counts vary freely
        // at runtime, so choices are keyed by power-of-two bucket.
        let tuned_i8 = match &i8_scratch {
            Some(scratch) => Self::tune_i8(&gpu, &caps, &lin_weights, scratch, b as u32),
            None => HashMap::new(),
        };
        // Decode-side head/logits. Sized by max_batch (NOT the prefill row
        // count): only decode rows need logits, and [max_prefill, vocab]
        // would be gigabytes.
        let vocab = cfg.vocab as u64;
        let logits_dev = st(max_batch as u64 * vocab);
        let argmax_dev = st(max_batch as u64);
        let argmax_part_dev = st(max_batch as u64 * ARGMAX_CHUNKS as u64 * 2);
        let topk_vals_dev = st(max_batch as u64 * TOPK_CAPACITY as u64);
        let topk_idx_dev = st(max_batch as u64 * TOPK_CAPACITY as u64);
        Engine {
            cfg,
            caps,
            gpu,
            selector: std::sync::Arc::new(CachedSelector::new(DefaultSelector)),
            ps,
            block_size,
            max_batch,
            max_blocks_per_seq,
            max_prefill,
            cap,
            splitk_part,
            splitk_cap,
            alloc: BlockAllocator::new(num_blocks, block_size),
            prefix: PrefixCache::new(),
            prefix_lookup_tokens: 0,
            prefix_hit_tokens: 0,
            pool_k,
            pool_v,
            kv_int8,
            scales_k,
            scales_v,
            kv_pool_bytes: kv_pool_bytes_val,
            scratch_bytes: scratch_bytes_val,
            kv_calib: None,
            clip_k,
            clip_v,
            lin_weights,
            i8_scratch,
            tuned_i8,
            sc,
            logits_dev,
            argmax_dev,
            argmax_part_dev,
            topk_vals_dev,
            topk_idx_dev,
            tape_cache: HashMap::new(),
            host_kv: HostKvPool::new(block_words, 0),
            kv_block_words: block_words,
            swap: None,
        }
    }

    /// Load a serving engine from a brain Qwen checkpoint (fp32 tensors;
    /// `weights_int8` quantizes the linears + head at load where the device
    /// has a packed-int8 path).
    #[allow(clippy::too_many_arguments)]
    pub fn load(path: &str, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        let c = checkpoint::load(path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let mut map = HashMap::new();
        for (name, _) in decoder_param_list(&cfg) {
            let t = c.find(&name, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing tensor {name}"));
            map.insert(name, t);
        }
        let hw = cfg.head_weight();
        if !map.contains_key(hw) {
            let h = c.find(hw, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing head {hw}"));
            map.insert(hw.to_string(), h);
        }
        Engine::from_map(cfg, &map, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    /// Install a calibrated KV clip table, uploading its per-layer ceilings
    /// once. `None` (or a [`model::kvcalib::KvCalib::disabled`] table) clears
    /// calibration - the append dispatch falls back to the plain online-
    /// absmax kernel. A no-op on a fp32-KV engine (`kv_int8: false`): there
    /// is nothing to clip, since `run_batched_submit`'s int8 branch (the only
    /// place `kv_calib` is read) never runs. Printed loudly rather than
    /// silent, because a caller installing a table it then never sees take
    /// effect is exactly the kind of no-op AGENTS.md calls out (a gate/config
    /// that never runs is worse than none) - [`kv_calibrated`] reflects the
    /// same "did it actually bind" question.
    pub fn set_kv_calib(&mut self, calib: Option<model::kvcalib::KvCalib>) {
        let calib = calib.filter(|c| !c.is_disabled());
        if calib.is_some() && !self.kv_int8 {
            eprintln!("serve: a KV clip table was installed on an fp32-KV engine; it will never be dispatched (calibration only applies to int8 KV)");
        }
        if let Some(c) = &calib {
            assert_eq!(c.n_layers as u32, self.cfg.n_layers, "kv_calib: n_layers mismatch");
            assert_eq!(c.n_kv as u32, self.cfg.n_kv_heads, "kv_calib: n_kv mismatch");
            assert_eq!(c.head_dim as u32, self.cfg.head_dim, "kv_calib: head_dim mismatch");
            let g = &self.gpu;
            self.clip_k = c.k.iter().map(|row| { let b = g.storage(row.len() as u64); g.write(&b, bytemuck::cast_slice(row)); b }).collect();
            self.clip_v = c.v.iter().map(|row| { let b = g.storage(row.len() as u64); g.write(&b, bytemuck::cast_slice(row)); b }).collect();
        } else {
            // No (or disabled) calibration: refill the resident per-layer clip
            // tables with the f32::MAX sentinel - bit-identical to the deleted
            // unclipped twin by the kernel's own contract (audit F42).
            let max_row = vec![f32::MAX; self.cfg.n_kv_heads as usize];
            for b in self.clip_k.iter().chain(self.clip_v.iter()) {
                self.gpu.write(b, bytemuck::cast_slice(&max_row));
            }
        }
        self.kv_calib = calib;
    }

    /// True when the decode path runs on int8 weights (the request survived the
    /// device capability gate). What a caller should report, rather than what
    /// was asked for.
    pub fn weights_int8(&self) -> bool {
        self.i8_scratch.is_some()
    }

    /// True when the KV cache is packed int8 rather than fp32 (unlike
    /// `weights_int8`, this is exactly what the constructor was built with -
    /// int8 KV has no capability gate to fall back from: the packed-int8 KV
    /// kernels are plain scalar WGSL, portable to every backend, unlike the
    /// DP4A-bound `matmul_i8*` GEMM family `weights_int8`/`w8_on` gates).
    pub fn kv_int8(&self) -> bool {
        self.kv_int8
    }

    /// Device bytes the KV pool costs at this engine's sizing - recorded once
    /// at construction from [`kv_pool_bytes`], never re-derived, so this can
    /// never drift from what `pool_k`/`pool_v`/`scales_k`/`scales_v` actually
    /// allocated.
    pub fn kv_pool_bytes(&self) -> u64 {
        self.kv_pool_bytes
    }

    /// Device bytes `Scratch::{scores,probs}` costs at this engine's sizing
    /// (M2.4) - recorded once at construction from
    /// [`paged_attn_scratch_bytes`], never re-derived, so this can never
    /// drift from what `sc.scores`/`sc.probs` actually allocated.
    pub fn paged_attn_scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// The pool's total theoretical cached-token capacity (`num_blocks *
    /// block_size`), independent of dtype - the number that answers "how
    /// many tokens could this pool ever hold at once", as opposed to
    /// [`kv_pool_bytes`] answering "at what memory cost".
    pub fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * self.block_size as u64
    }

    /// Whether the installed KV clip table is a real, binding calibration
    /// that is ACTUALLY DISPATCHED (not `None`, not `KvCalib::disabled`, and
    /// the engine is int8 - the clip binding is only read by the i8 append on
    /// the int8 branch of `run_batched_submit`). A table
    /// installed on an fp32 engine is `Some` in `self.kv_calib` but never
    /// read by anything, so this must say `false` for it or it would claim a
    /// calibration is binding when it provably is not.
    pub fn kv_calibrated(&self) -> bool {
        self.kv_int8 && self.kv_calib.is_some()
    }

    /// The device this engine runs on - the parent handle for building more
    /// engines on the same device ([`Engine::from_map_on`]).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model's vocabulary size (for a caller doing its own sampling).
    pub fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    /// Append one slot per sequence and gather the batched-forward metadata.
    fn append_meta(&mut self, tables: &mut [&mut BlockTable]) -> BatchMeta {
        let mbt = self.max_blocks_per_seq as usize;
        let bsz = tables.len() as u32;
        assert!(bsz <= self.max_batch);
        let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u32; bsz as usize * mbt];
        for (i, table) in tables.iter_mut().enumerate() {
            let pos = table.len();
            let (block, offset) = table.append(&mut self.alloc).expect("KV pool exhausted");
            positions.push(pos);
            seqlens.push(pos + 1);
            blocks.push(block);
            offsets.push(offset);
            for (lb, &phys) in table.blocks().iter().enumerate() {
                bt[i * mbt + lb] = phys;
            }
        }
        (bsz, positions, seqlens, blocks, offsets, bt)
    }

    /// Advance every sequence by one token (decode).
    pub(crate) fn forward_batched(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt, false)
    }

    /// Advance every sequence by one token and return the **greedy next token**
    /// per row, with the LM head applied on the device (see
    /// [`Engine::run_batched_greedy`]).
    pub(crate) fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched_greedy(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt, false)
    }

    /// Advance every sequence by one token from a ready-made embedding per sequence
    /// (`[bsz, d_model]`) - the tts Talker multi-stream path: concurrent voice
    /// streams decode together on the shared paged pool.
    pub fn forward_batched_embed(&mut self, tables: &mut [&mut BlockTable], embeds: &[f32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        assert_eq!(embeds.len(), bsz as usize * self.cfg.d_model as usize);
        self.run_batched(bsz, Input::Embeds(embeds), &positions, &seqlens, &blocks, &offsets, &bt, false)
    }

    /// Run one batched forward over `bsz` rows given fully-computed metadata:
    /// `positions[i]` RoPE position, `seqlens[i]` the cached length row i attends
    /// (row i's query attends `j < seqlens[i]` - set to start+i+1 for causal
    /// prefill), `(blocks[i], offsets[i])` the pool slot to write row i's K/V, and
    /// `bt` the per-row block tables (`bsz * max_blocks_per_seq`). Serves decode
    /// (one new token per sequence) and prefill chunks alike - `causal_chunk`
    /// tells the two apart for `Op::PagedAttentionFused` (M2.4; see
    /// `run_batched_steps`'s own doc).
    #[allow(clippy::too_many_arguments)]
    fn run_batched(&mut self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32], causal_chunk: bool) -> Vec<f32> {
        let b = self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt, causal_chunk);
        self.gpu.read(&self.sc.xn_final, (b * self.cfg.d_model) as usize)
    }

    /// The transformer body only: records and submits every stage, leaving the
    /// final norm in `sc.xn_final` **without reading it back**. Submits are
    /// accumulated lazily and flushed on the next read, so a caller that appends
    /// more device work (the greedy head) still pays one flush per step rather
    /// than two. Returns the row count.
    // qwen3-serve-manual-gemm-dispatch BEGIN (B7, `no_kernel_names.rs`'s own
    // allow-list) - this engine's own tuned fp32/int8 GEMM selection, kept
    // OFF the `model::ops::Ops` façade for the reasons `Self::from_map_with_
    // gpu`'s own comment gives (split-K has no `Ops::matmul` equivalent, and
    // `I8Scratch` reuse vs. per-call allocation) - NOT because the selector
    // is unreachable there any more (`Ops::with_selector` fixed that; see
    // this struct's `selector` field doc). See `no_kernel_names.rs`'s own
    // module doc for exactly what this allow-lists and why.
    /// `out = x @ W^T`, choosing the decode-regime GEMV (one workgroup per
    /// output column, W streamed once across all rows) when the selector says
    /// the shape is in that regime. Same contract, same result.
    fn mm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        let (kind, threads) = block::gemm_variant(self.gemm_tier(), m, n);
        self.gpu.step(kind, &[x, w, out], &[m, k, n], threads)
    }

    /// Whether a `[n, k]` fp32 weight fits ONE storage binding on this device.
    /// `wgpu` clamps `max_storage_buffer_binding_size` to `i32::MAX` (2047
    /// MiB) on every backend regardless of a card's actual VRAM, so this is a
    /// real ceiling any `[n, k]` table can cross at large enough `n`. The
    /// per-layer projections never approach it, bounded by `d_model`/`d_ff`,
    /// but the LM head's `n = vocab` does at a real vocabulary size, e.g.
    /// Qwen3-8B's 151936 x 4096 = ~2.32 GiB. `Self::mm_into` is this check's
    /// only caller.
    fn fits_one_binding(&self, k: u32, n: u32) -> bool {
        (n as u64) * (k as u64) * 4 <= self.gpu.max_storage_binding_bytes()
    }

    /// [`Self::mm`], but for a `[n, k]` weight [`Self::fits_one_binding`]
    /// says cannot be bound whole at all: `w` is bound as vocab/output-row
    /// SUB-RANGES (`step_sliced`), one [`MATMUL_TILE`] dispatch per tile,
    /// each tile's weight rows sized under the device's queried binding
    /// limit ([`block::vocab_tiles_on`] - the SAME rule and the SAME kernel
    /// `qwen3::model::Qwen::forward`'s own lm_head epilogue already uses for
    /// this exact gap, ported here and generalised to `m > 1` rows). `out`
    /// stays a WHOLE (unsliced) binding: a tile's columns are a STRIDED
    /// slice of the `[m, n]` row-major output (`out[row*n + n_off + col]`),
    /// which no `step_sliced` byte-range could express - `matmul_tile.wgsl`'s
    /// own `n_off`/`n_full` params place each tile at the right absolute
    /// column instead.
    fn mm_tiled_into(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        let kw = k as u64;
        for (n_off, n_tile) in block::vocab_tiles_on(&self.gpu, n as u64, kw) {
            s.push(self.gpu.step_sliced(
                MATMUL_TILE,
                &[x, w, out],
                &[(0, 0), (n_off as u64 * kw, n_tile as u64 * kw), (0, 0)],
                &[m, k, n, n_off, n_tile],
                m * n_tile,
            ));
        }
    }

    /// [`Self::mm`], but free to emit MORE than one dispatch: the split-K GEMM
    /// needs a fold after it.
    ///
    /// Split-K only when the tile grid is too small to fill the device - the
    /// same rule and the same 288-workgroup target `vae::blocks` measured for
    /// `matmul_dw_reg_splitk`, which is the identical defect on the backward.
    /// `slices = 1` means the plain kernel, so a shape that already fills the
    /// card is untouched.
    ///
    /// Checked ahead of split-K, not after: a weight too large for one
    /// binding at all is a correctness gap, not a performance choice, and
    /// `Self::mm_tiled_into`'s only real caller (the LM head) dispatches at
    /// decode row counts anyway, where `Self::splitk_slices` already declines
    /// (`m <= DECODE_REGIME_MAX_ROWS`).
    fn mm_into(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        if !self.fits_one_binding(k, n) {
            self.mm_tiled_into(s, x, w, out, m, k, n);
            return;
        }
        match self.splitk_slices(m, k, n) {
            Some(slices) => {
                let tiles = m.div_ceil(128) * n.div_ceil(128);
                let part = self.splitk_part.as_ref().expect("split-K chosen without a scratch buffer");
                s.push(self.gpu.step(
                    MATMUL_REG3_SPLITK,
                    &[x, w, part],
                    &[m, k, n, slices],
                    slices * tiles * 256,
                ));
                // acc = 0: a forward GEMM owns its output and ASSIGNS, so the
                // destination needs no clear (unlike a parameter gradient).
                s.push(self.gpu.step(
                    SPLITK_REDUCE,
                    &[part, out],
                    &[m * n, slices, 0],
                    (m * n).div_ceil(64) * 64,
                ));
            }
            None => s.push(self.mm(x, w, out, m, k, n)),
        }
    }

    /// How many k-slices to split this GEMM into, or `None` for the plain
    /// kernel. `None` whenever the device has no scratch, the tile grid already
    /// fills the card, the shape is in the GEMV regime, or the partials would
    /// not fit the scratch - the last one keeps this from silently allocating.
    fn splitk_slices(&self, m: u32, k: u32, n: u32) -> Option<u32> {
        if !self.caps.workgroup_reductions || m <= DECODE_REGIME_MAX_ROWS {
            return None;
        }
        let cap = self.splitk_cap?;
        let tiles = m.div_ceil(128) * n.div_ceil(128);
        // Enough k to split at all: each slice must still hold whole BK chunks.
        let slices = SPLITK_TARGET_WGS.div_ceil(tiles).min(k / 64).clamp(1, SPLITK_MAX_SLICES);
        let need = (m as u64) * (n as u64) * (slices as u64);
        if slices <= 1 || need > cap {
            return None;
        }
        Some(slices)
    }

    /// The fp32 GEMM tier for this device - the SAME rule `flux1`, `flux2` and
    /// `model::rowemit` use, so the serving engine stops having a private one.
    ///
    /// `mm` picks the actual kernel by calling `block::gemm_variant(self.
    /// gemm_tier(), m, n)`, which is now (B2) a thin adapter over
    /// `backend_api::select::candidates` - including its `RegisterTiled`
    /// member, so every prefill chunk above `DECODE_REGIME_MAX_ROWS` reaches
    /// `MATMUL_REG3` instead of falling through to the naive reference the way
    /// it used to before that member existed. `gemm_variant`'s decode-regime
    /// cutoff is `DECODE_REGIME_MAX_ROWS` itself now (not a private `m <= 32`
    /// copy), so the decode regime this engine already tuned for is unchanged;
    /// only the rows above it move, onto the kernel the batched forward was
    /// already using.
    ///
    /// Gated on the queried `workgroup_reductions`: both fast kernels cooperate
    /// across a workgroup, so a device without it keeps the naive reference
    /// (which `backend-cpu` routes to its AVX2 GEMM anyway).
    fn gemm_tier(&self) -> block::GemmVariants {
        if self.caps.workgroup_reductions {
            block::GemmVariants::Fast { gemv: Some(MATMUL_GEMV), tiled: MATMUL_REG3 }
        } else {
            block::GemmVariants::Reference(MATMUL)
        }
    }

    /// Quantize `x`'s rows `[0, rows)` once (`model::dispatch::I8Scratch`),
    /// shared by every linear reading it this layer (e.g. xn1 -> q/k/v) - a
    /// no-op on an all-fp32 engine (`self.i8_scratch` is `None`), unlike
    /// `qwen3::model::Qwen::ops.act`'s unconditional cost (see that crate's
    /// B7 ledger note): this engine's own `Self::linear` KNOWS which weights
    /// are `I8` before it ever quantizes, since the tier is a per-engine
    /// build-time choice, not inspected per call.
    fn quant_once(&self, s: &mut Vec<Step>, x: &DeviceBuffer, k: u32, rows: u32) {
        if let Some(scratch) = &self.i8_scratch {
            scratch.quant_rows(&self.gpu, [MAX_ABS_ROW, QUANT_PACK], s, x, 0, rows, k);
        }
    }

    /// Vocab tiles for `tok.weight`'s gather - `block::vocab_tiles_on`, sized
    /// to THIS device's queried storage-binding limit (never the portable
    /// floor: see that function's own doc for why a P40's real, larger limit
    /// matters for tile count).
    fn vocab_tiles(&self) -> Vec<(u32, u32)> {
        block::vocab_tiles_on(&self.gpu, self.cfg.vocab as u64, self.cfg.d_model as u64)
    }

    /// The token-embedding gather for `n` token rows, as tiled steps -
    /// mirrors `qwen3::model::Qwen::embed_tiled` exactly (same kernel, same
    /// tiling rule, same reason).
    ///
    /// `tok.weight` stays ONE buffer in `self.ps` (`ParamStore::new_with_roles`
    /// allocates and streams it in bounded chunks regardless of size - see
    /// `paramstore::UPLOAD_CHUNK_WORDS` - so allocation was never the
    /// problem). The problem is BINDING it: a plain `Gpu::step` binds a
    /// buffer's ENTIRE range, and `max_storage_buffer_binding_size` is
    /// clamped to `i32::MAX` (2047 MiB) on every `wgpu` backend regardless of
    /// how much VRAM the card has - Qwen3-8B's real `[151936, 4096]` fp32
    /// table is ~2.32 GiB, over that cap on a card with 24 GB to spare. This
    /// dispatches `EMBED_TILE` once per vocab tile instead, each a
    /// `step_sliced` binding of only that tile's rows
    /// (`[v0*d_model, (v0+cnt)*d_model)`), which the tiling rule sizes to
    /// stay under the limit; every token belongs to exactly one tile, so
    /// across all tiles every output element is written exactly once.
    fn embed_tiled(&self, g: &Gpu, out: &DeviceBuffer, n: u32) -> Vec<Step> {
        let d = self.cfg.d_model;
        let dw = d as u64;
        self.vocab_tiles()
            .into_iter()
            .map(|(v0, cnt)| {
                g.step_sliced(
                    EMBED_TILE,
                    &[&self.sc.tok_buf, self.ps.w("tok.weight"), out],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[d, n, v0, cnt],
                    n * d,
                )
            })
            .collect()
    }

    /// Dispatch one linear `out = x @ Wᵀ` by `w`'s own tag (B7) - `self.
    /// lin_weights`'s tier, never a separate on/off flag inspected here.
    /// `x` must already be quantized (`Self::quant_once`) when `w` is `I8`.
    fn linear(&self, s: &mut Vec<Step>, w: &Weight, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32) {
        match w {
            Weight::F32 { w, n, k } => self.mm_into(s, x, w, out, rows, *k, *n),
            Weight::I8 { w, s: sw, n, k } => {
                let scratch = self.i8_scratch.as_ref().expect("qwen3 serve: I8 weight built without i8_scratch");
                self.mm8(s, scratch, w, sw, *n, *k, out, rows);
            }
            _ => unreachable!("qwen3 serve only ever builds F32/I8 weights (Weight::upload's `want` is always one of the two - see `from_map_with_gpu`)"),
        }
    }

    /// One int8 linear: the MEASURED choice for this device where one exists
    /// (S5, tuned at build, keyed by row bucket), else the static policy. The
    /// packed GEMV owns few rows, the 128x128 tile owns prefill shapes; the
    /// crossover is per-device. Must be preceded by a matching `Self::
    /// quant_once` writing into `scratch`.
    #[allow(clippy::too_many_arguments)]
    fn mm8(&self, s: &mut Vec<Step>, scratch: &I8Scratch, w: &DeviceBuffer, sw: &DeviceBuffer, n: u32, k: u32, out: &DeviceBuffer, rows: u32) {
        let shape = OpShape { m: rows, n, k, dtype: Dtype::I8 };
        let variant = if rows <= DECODE_REGIME_MAX_ROWS {
            let bucket = rows.next_power_of_two().min(DECODE_REGIME_MAX_ROWS);
            self.tuned_i8
                .get(&(bucket, n, k))
                .copied()
                .unwrap_or_else(|| self.selector.select(Op::MatMul, shape, &self.caps))
        } else {
            self.selector.select(Op::MatMul, shape, &self.caps)
        };
        match variant {
            KernelVariant::WorkgroupPerOutput => s.push(self.gpu.step(
                MATMUL_I8_GEMV,
                &[scratch.xq_for(k), w, &scratch.sx, sw, out],
                &[rows, k / 4, n],
                n * 64,
            )),
            _ => s.push(self.gpu.step(
                MATMUL_I8_DYN,
                &[scratch.xq_for(k), w, &scratch.sx, sw, out],
                &[rows, k / 4, n],
                rows.div_ceil(128) * n.div_ceil(128) * 256,
            )),
        }
    }

    /// Measure the GEMV/tile crossover for every distinct int8 linear shape
    /// and row bucket on THIS device (S5). Both candidates are dispatched on
    /// the engine's real buffers - REPS dispatches per timing so submit/poll
    /// overhead amortises - and the winner persists per adapter + kernel
    /// sources. `BRAIN_NO_AUTOTUNE=1` skips every measurement (static policy).
    fn tune_i8(gpu: &Gpu, caps: &DeviceCaps, weights: &HashMap<String, Weight>, scratch: &I8Scratch, max_rows: u32) -> HashMap<(u32, u32, u32), KernelVariant> {
        let fp = gpu_core::tune::source_fingerprint(&[kernels::MATMUL_I8_GEMV, kernels::MATMUL_I8_DYN]);
        let store = gpu_core::tune::FileTuneStore::for_adapter(fp)
            .map(|s| Box::new(s) as Box<dyn gpu_core::select::TuneStore>);
        let tuner = AutoTuner::new(store);
        // Distinct (n, k) shapes across every `I8` weight this engine holds
        // (every layer shares the same 7 shapes; the head is usually a
        // distinct one).
        let mut shapes: Vec<(u32, u32, &DeviceBuffer, &DeviceBuffer)> = Vec::new();
        for w in weights.values() {
            if let Weight::I8 { w: wb, s: sw, n, k } = w {
                if !shapes.iter().any(|&(sn, sk, _, _)| sn == *n && sk == *k) {
                    shapes.push((*n, *k, wb, sw));
                }
            }
        }
        let mut out = HashMap::new();
        let cap_bucket = max_rows.next_power_of_two().min(DECODE_REGIME_MAX_ROWS);
        for &m in &[1u32, 2, 4, 8, 16, 32] {
            if m > cap_bucket {
                break;
            }
            for &(n, k, wb, sw) in &shapes {
                let shape = OpShape { m, n, k, dtype: Dtype::I8 };
                let mut measure = |v: KernelVariant| Self::measure_i8(gpu, scratch, wb, sw, n, k, m, v);
                let choice = tuner.resolve(Op::MatMul, shape, caps, &mut measure);
                out.insert((m, n, k), choice);
            }
        }
        out
    }

    /// Time one int8 GEMM variant on real buffers: REPS dispatches in one
    /// submission, mean milliseconds per dispatch. `None` = not measurable.
    #[allow(clippy::too_many_arguments)]
    fn measure_i8(gpu: &Gpu, scratch: &I8Scratch, w: &DeviceBuffer, sw: &DeviceBuffer, n: u32, k: u32, m: u32, variant: KernelVariant) -> Option<f64> {
        const REPS: usize = 8;
        let out = gpu.storage(m as u64 * n as u64);
        let step = |_: usize| match variant {
            KernelVariant::WorkgroupPerOutput => gpu.step(
                MATMUL_I8_GEMV,
                &[scratch.xq_for(k), w, &scratch.sx, sw, &out],
                &[m, k / 4, n],
                n * 64,
            ),
            KernelVariant::PackedInt8 => gpu.step(
                MATMUL_I8_DYN,
                &[scratch.xq_for(k), w, &scratch.sx, sw, &out],
                &[m, k / 4, n],
                m.div_ceil(128) * n.div_ceil(128) * 256,
            ),
            other => unreachable!("int8 candidates are GEMV or tile, got {other:?}"),
        };
        // Warm-up (pipeline residency, first-touch allocations), then timed.
        gpu.submit(&[], &[step(0)]);
        gpu.poll_wait();
        let steps: Vec<Step> = (0..REPS).map(step).collect();
        let t0 = std::time::Instant::now();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        Some(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64)
    }

    /// RMSNorm, choosing the workgroup-per-row kernel at decode row counts
    /// (the per-element kernel runs `rows` threads - 8 threads on a 3840-core
    /// card at batch 8, and measured as a sizeable share of decode time).
    fn rms(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, d: u32, rows: u32) -> Step {
        let g = &self.gpu;
        let shape = OpShape { m: rows, n: d, k: 0, dtype: Dtype::F32 };
        match self.selector.select(Op::RmsNorm, shape, &self.caps) {
            KernelVariant::WorkgroupPerOutput => g.step(RMSNORM_ROWS, &[x, w, out], &[d, rows, gpu_core::f(1e-6)], rows * 64),
            _ => g.step(RMSNORM, &[x, w, out], &[d, rows], rows),
        }
    }
    // qwen3-serve-manual-gemm-dispatch END

    /// M4.3: `Self::rms` immediately followed by `Self::quant_once` over its
    /// OWN output (`ln1`->`xn1`, `ln2`->`xn2`) collapsed into one dispatch on
    /// an all-int8-weight engine: `Self::linear`'s `Weight::I8` arm never
    /// reads the `x` parameter it is handed (it reads only the pre-quantized
    /// `i8_scratch`), and `w8_on` is a single engine-wide tier (`Engine::
    /// from_map_with_gpu`'s `want`/`w8_on`), so whenever `self.i8_scratch` is
    /// `Some` the fp32 value `rms` would have written to `out` has NO reader
    /// at all - `max_abs_row` then `quant_pack` re-read it twice more purely
    /// to throw it away. `rmsnorm_quant_fused.wgsl` folds the abs-max
    /// reduction into the same cooperative pass `rmsnorm_rows` already runs
    /// and never writes the fp32 row - see its own header for the bit-
    /// identity argument (same expression, same operand order, recomputed
    /// rather than cached in a runtime-sized register array).
    ///
    /// Falls back to the unfused `Self::rms` + `Self::quant_once` pair - `out`
    /// IS written there, matching every existing reader's contract - on an
    /// all-fp32 engine (`self.i8_scratch` is `None`, so `quant_once` is
    /// already a no-op and `out` is the only real result) or a device without
    /// `workgroup_reductions` (the fused kernel carries 3 barriers, gated
    /// exactly like `Self::rms`'s own cooperative arm and `Self::
    /// qk_norm_rope`'s fused pair).
    fn rms_quant(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, d: u32, rows: u32) {
        match &self.i8_scratch {
            Some(scratch) if self.caps.workgroup_reductions => {
                s.push(self.gpu.step(
                    RMSNORM_QUANT_FUSED,
                    &[x, w, &scratch.sx, scratch.xq_for(d)],
                    &[d, rows, gpu_core::f(1e-6)],
                    rows * 64,
                ));
            }
            _ => {
                s.push(self.rms(x, w, out, d, rows));
                self.quant_once(s, out, d, rows);
            }
        }
    }

    /// M4.2: fused QK-norm + RoPE, one dispatch over `x`'s `rows = b * heads`
    /// per-head rows (`heads` is `nh` for Q, `nkv` for K - the SAME
    /// flattening `Self::rms`'s own `b * nh` / `b * nkv` row counts already
    /// assume) instead of `Self::rms` followed by a separate `ROPE_PAGED`.
    /// See `qknorm_rope_fused.wgsl`'s own header for the derivation and the
    /// bit-identity argument.
    ///
    /// Gated on `caps.workgroup_reductions` exactly like `Self::rms`'s own
    /// cooperative arm: the fused kernel carries the same single
    /// `workgroupBarrier()` the split-at-barrier CPU JIT mis-executes for
    /// `rmsnorm_rows` (`backend-cpu`'s own doc), so a device without that
    /// capability keeps the original two-dispatch pair rather than an
    /// unconditional fused dispatch reproducing that defect.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, hd: u32, heads: u32, rows: u32, theta: f32) {
        let g = &self.gpu;
        if self.caps.workgroup_reductions {
            s.push(g.step(
                QKNORM_ROPE_FUSED,
                &[x, w, &self.sc.pos_buf, out],
                &[rows, heads, hd, gpu_core::f(1e-6), fb(theta)],
                rows * 64,
            ));
        } else {
            s.push(self.rms(x, w, out, hd, rows));
            let b = rows / heads;
            s.push(g.step(ROPE_PAGED, &[out, &self.sc.pos_buf], &[b, heads, hd, heads * hd, fb(theta)], rows * (hd / 2)));
        }
    }

    /// M4.2: `Self::qk_norm_rope`'s K-only sibling, additionally folding the
    /// fp32 paged KV append into the same fused dispatch - `out` still
    /// receives the normalized+rotated K (mirroring `Self::rms` + `ROPE_PAGED`'s
    /// old contract on `sc.k`, which `Engine::calibrate_kv` and test fixtures
    /// read directly), and `pool` receives the SAME values at their paged
    /// slot in one write instead of a separate `KV_APPEND_B` re-reading what
    /// RoPE just wrote. Same `workgroup_reductions` gate as `Self::qk_norm_rope`.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_append(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, pool: &DeviceBuffer, hd: u32, heads: u32, rows: u32, theta: f32, block_size: u32) {
        let g = &self.gpu;
        if self.caps.workgroup_reductions {
            s.push(g.step(
                QKNORM_ROPE_APPEND_FUSED,
                &[x, w, &self.sc.pos_buf, &self.sc.blk_buf, &self.sc.off_buf, out, pool],
                &[rows, heads, hd, gpu_core::f(1e-6), fb(theta), block_size],
                rows * 64,
            ));
        } else {
            s.push(self.rms(x, w, out, hd, rows));
            let b = rows / heads;
            s.push(g.step(ROPE_PAGED, &[out, &self.sc.pos_buf], &[b, heads, hd, heads * hd, fb(theta)], rows * (hd / 2)));
            s.push(g.step(KV_APPEND_B, &[out, &self.sc.blk_buf, &self.sc.off_buf, pool], &[b, heads * hd, block_size], rows * hd));
        }
    }

    /// Write the paged batch metadata (`positions`/`seqlens`/`blocks`/
    /// `offsets`/`bt`) a forward's dispatches read out of `self.sc`'s buffers -
    /// factored out of `Self::run_batched_steps` (M6.3) so the tape-cache path
    /// in `Self::run_batched_submit` can perform exactly this write, and
    /// nothing else, ahead of REPLAYING an already-recorded tape.
    ///
    /// Resident mode (A4): every input - token ids AND paged metadata - was
    /// produced on the device by `decode_feed`/`decode_advance`, so writing
    /// host copies here would both be wrong (stale) and force a flush.
    fn write_batch_meta(&self, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) {
        if !matches!(input, Input::Resident) {
            let g = &self.gpu;
            g.write(&self.sc.pos_buf, positions);
            g.write(&self.sc.seqlen_buf, seqlens);
            g.write(&self.sc.blk_buf, blocks);
            g.write(&self.sc.off_buf, offsets);
            g.write(&self.sc.bt_buf, bt);
        }
    }

    /// Write this step's actual input values - `Input::Tokens`' ids into
    /// `sc.tok_buf` (embedded by the tape's own `EMBED` step), `Input::
    /// Embeds`' vectors straight into the residual stream, or nothing for
    /// `Input::Resident` (already on-device). Split out of `Self::
    /// run_batched_steps` for the same reason as `Self::write_batch_meta`.
    fn write_batch_input(&self, input: Input) {
        match input {
            Input::Tokens(t) => self.gpu.write(&self.sc.tok_buf, t),
            Input::Resident => {}
            Input::Embeds(e) => self.gpu.write(&self.sc.res[0], bytemuck::cast_slice(e)),
        }
    }

    /// `causal_chunk` distinguishes the two regimes `Op::PagedAttentionFused`
    /// itself cannot infer from `bsz`/`cap` alone (M2.4): `true` for a
    /// prefill/spec-decode-verify chunk of ONE sequence's causally-increasing
    /// rows sharing one physical block table (`bt` duplicated identically
    /// across every row - `Engine::prefill`'s own construction, checked
    /// against source), `false` for decode's independent-sequences rows. The
    /// two are different physical kernels answering the same shape
    /// signature in different call-site semantics, not points on one shape
    /// gradient - see `Op::PagedAttentionFused`'s own doc.
    ///
    /// The dispatch list itself (kernel, buffers, uniform PARAMETERS, thread
    /// counts): a pure function of `(bsz, matches!(input, Embeds), causal_chunk)`
    /// and this engine's own fixed config - nothing here reads a position,
    /// seqlen, block or token VALUE (those live in buffer contents `Self::
    /// write_batch_meta`/`Self::write_batch_input` wrote separately, above).
    /// That purity is what makes M6.3's tape cache (`Self::run_batched_submit`)
    /// safe: caching this method's OUTPUT and replaying it unchanged for a
    /// later call at the same key reuses the same kernels/buffers/uniform
    /// values a fresh call would have produced, byte for byte.
    fn batched_tape(&self, bsz: u32, input: Input, causal_chunk: bool) -> Vec<Step> {
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let bs = self.block_size;
        let cap = self.cap;
        let mbt = self.max_blocks_per_seq;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let theta = c.rope_theta;
        let g = &self.gpu;
        let kids = ids();
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let b = bsz;
        let mut s: Vec<Step> = Vec::new();
        if !matches!(input, Input::Embeds(_)) {
            // Vocab-tiled (`Self::embed_tiled`), not a single `EMBED` dispatch
            // against the whole `tok.weight` table: that table exceeds one
            // storage binding at real vocab sizes (Qwen3-8B's is ~2.32 GiB,
            // over wgpu's 2047 MiB `max_storage_buffer_binding_size` clamp)
            // regardless of `weights_int8` - the embedding gather is always
            // fp32, so this is the ONE place that limit bites unconditionally.
            s.extend(self.embed_tiled(g, &sc.res[0], b));
        }
        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            // M4.3: RMSNorm fused with its own activation quant on an
            // all-int8-weight engine (`Self::rms_quant`) - `sc.xn1`'s fp32
            // value has no reader once `Self::linear`'s `I8` arm ignores it,
            // shared by every linear reading the quantized result (xn1 ->
            // q/k/v). Falls back to the unfused `rms` + `Self::quant_once`
            // pair (a no-op on an all-fp32 engine) otherwise.
            self.rms_quant(&mut s, &sc.res[l], w(&p("ln1.weight")), &sc.xn1, d, b);
            // M4.1: one fused GEMM (`wq;wk;wv` concatenated at weight-load
            // time - see `from_map_with_gpu`) instead of three, then
            // `concat_split` narrows the wide `[b, hq+2*hkv]` output back
            // into the same compact `q_pre`/`k_pre`/`v` buffers QK-norm/
            // RoPE/KV-append already require.
            self.linear(&mut s, &self.lin_weights[&p(WQKV)], &sc.xn1, &sc.qkv_pre, b);
            let qkv_width = hq + 2 * hkv;
            s.push(concat_split_step(g, &sc.qkv_pre, &sc.q_pre, b, qkv_width, hq, 0));
            s.push(concat_split_step(g, &sc.qkv_pre, &sc.k_pre, b, qkv_width, hkv, hq));
            s.push(concat_split_step(g, &sc.qkv_pre, &sc.v, b, qkv_width, hkv, hq + hkv));
            // M4.2: QK-norm + RoPE fused into one dispatch each for Q and K
            // (`Self::qk_norm_rope`) instead of the four separate `self.rms`/
            // `ROPE_PAGED` dispatches this used to be - see that method's own
            // doc for the derivation and the `workgroup_reductions` gate.
            self.qk_norm_rope(&mut s, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, nh, b * nh, theta);
            if self.kv_int8 {
                // K's fused pass stops at norm+RoPE here - the int8 append
                // below needs the whole per-head row for its own absmax
                // reduction and quantizes into a packed `u32` pool, a
                // different shape than the fp32 append `Self::
                // qk_norm_rope_append` folds in, so it is NOT merged into
                // this milestone.
                self.qk_norm_rope(&mut s, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, nkv, b * nkv, theta);
                // ONE append kernel for both paths (audit F42): the clip
                // buffers hold either the calibrated ceilings or the
                // f32::MAX sentinel, which the kernel's contract documents
                // as bit-identical to the old unclipped twin.
                s.push(g.step(APPEND_I8_CLIPPED, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.clip_k[l], &self.pool_k[l], &self.scales_k[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(APPEND_I8_CLIPPED, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.clip_v[l], &self.pool_v[l], &self.scales_v[l]], &[b, hkv, bs, hd], b * nkv));
                // No `Op::PagedAttentionFused` check here: `paged_flash_
                // prefill` has no int8-KV tier yet (its own `@dtype f32`
                // header), and decode's fused kernels never measured a win at
                // any dtype (M2.1/M2.2) - every candidate this Op could
                // return at `Dtype::I8` is `Reference` (see its own doc), so
                // asking would be a dead call.
                s.push(g.step(SCORES_I8, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_k[l], &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_I8, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_v[l], &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            } else {
                // K's norm+RoPE+append collapse into ONE dispatch here (the
                // fp32-KV branch has no quantization reduction blocking the
                // merge, unlike the int8 branch above) - `sc.k` still comes
                // out normalized+rotated for `Engine::calibrate_kv`/tests,
                // `self.pool_k[l]` gets the same values at their paged slot.
                self.qk_norm_rope_append(&mut s, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, &self.pool_k[l], hd, nkv, b * nkv, theta, bs);
                s.push(g.step(KV_APPEND_B, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l]], &[b, hkv, bs], b * hkv));
                // M2.4: whole-triad-vs-single-fused-dispatch choice, through
                // `Op::PagedAttentionFused` (a SEPARATE Op from
                // `Op::PagedAttention` below - see its own doc for why),
                // factored into `model::block::paged_attention_fused` rather
                // than inlined here - `no_kernel_names.rs`'s own gate bans
                // this function's body from naming the selector's return
                // enum directly, the same reason `paged_scores_variant`
                // below already lives in `model::block` instead of here. KV
                // storage dtype is always F32 in this branch (the `kv_int8`
                // arm above never reaches here).
                if model::block::paged_attention_fused(g, causal_chunk, false) {
                    // `paged_flash_prefill` (M2.3): one dispatch per (head,
                    // query-tile), no `scores`/`probs` at all - BR=64 is the
                    // kernel's own tile size, @workgroup_size(256) its own
                    // launch shape (both pinned in its own WGSL header).
                    let ntiles_q = b.div_ceil(64);
                    s.push(g.step(
                        PAGED_FLASH_PREFILL,
                        &[&sc.q, &self.pool_k[l], &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx],
                        &[b, nh, nkv, hd, group, bs, mbt],
                        nh * ntiles_q * 256,
                    ));
                } else {
                    // One workgroup per score where the device runs workgroup
                    // reductions: the per-element kernel's lanes are `kv_stride`
                    // floats apart (4 KB at 0.6B), so a fetched sector serves one
                    // useful float in eight: measured at a small fraction of the
                    // bandwidth roof while taking about half of a served step.
                    // Gated via `model::block::paged_scores_variant`
                    // (`Op::PagedAttention`), not a hand-rolled
                    // `caps.workgroup_reductions` check - this engine registers
                    // both kernels unconditionally, so `coop` is always `Some`.
                    let (sk, st) = model::block::paged_scores_variant(g, SCORES_B, Some(SCORES_B_WG), b * nh, cap);
                    s.push(g.step(sk, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], st));
                    s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                    s.push(g.step(APPLY_B, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
                }
            }
            self.quant_once(&mut s, &sc.ctx, hq, b);
            self.linear(&mut s, &self.lin_weights[&p("attn.wo.weight")], &sc.ctx, &sc.proj, b);
            s.push(g.step(ADD2, &[&sc.res[l], &sc.proj, &sc.xmid], &[b * d], b * d));
            // M4.3: same fusion as ln1 above, for `sc.xn2` -> gate/up.
            self.rms_quant(&mut s, &sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, b);
            // M4.1: one fused GEMM (`gate;up` concatenated at weight-load
            // time) instead of two, `concat_split` narrowing its `[b, 2*ff]`
            // output back into the compact `gate_pre`/`up` buffers
            // `swiglu_fwd` requires.
            self.linear(&mut s, &self.lin_weights[&p(WGATEUP)], &sc.xn2, &sc.gateup_pre, b);
            s.push(concat_split_step(g, &sc.gateup_pre, &sc.gate_pre, b, 2 * ff, ff, 0));
            s.push(concat_split_step(g, &sc.gateup_pre, &sc.up, b, 2 * ff, ff, ff));
            s.push(block::swiglu_fwd(g, &kids, &sc.gate_pre, &sc.up, &sc.h, b * ff));
            self.quant_once(&mut s, &sc.h, ff, b);
            self.linear(&mut s, &self.lin_weights[&p("mlp.down.weight")], &sc.h, &sc.mlp_out, b);
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &sc.res[l + 1]], &[b * d], b * d));
        }
        let last = c.n_layers as usize;
        s.push(self.rms(&sc.res[last], w("norm.weight"), &sc.xn_final, d, b));
        s
    }

    /// [`Self::write_batch_meta`] + [`Self::write_batch_input`] +
    /// [`Self::batched_tape`], for callers that want the dispatch list
    /// itself rather than a submitted step - `qwen_bench serve`'s profiler
    /// (`Self::steps_for_profile`) and this file's own `causal_chunk`/
    /// decode-regime kernel-selection tests, neither of which goes through
    /// M6.3's tape cache: a profiler needs a step list `gpu_core::profile`
    /// can resubmit standalone, and these tests assert on the FRESHLY BUILT
    /// tape's own kernel choices, at shapes the cache does not even key on
    /// (`causal_chunk = true`, or `bsz` above `DECODE_REGIME_MAX_ROWS`).
    #[allow(clippy::too_many_arguments)]
    fn run_batched_steps(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32], causal_chunk: bool) -> (Vec<Step>, u32) {
        #[cfg(feature = "fault-injection")]
        if fault::take_kernel_failure() {
            panic!("injected fault: kernel dispatch failure");
        }
        self.write_batch_meta(input, positions, seqlens, blocks, offsets, bt);
        self.write_batch_input(input);
        (self.batched_tape(bsz, input, causal_chunk), bsz)
    }

    /// [`Self::run_batched_steps`] plus the submit - decode's hot path
    /// (`causal_chunk = false`, `bsz <= DECODE_REGIME_MAX_ROWS`,
    /// `Input::Tokens`/`Input::Resident`) goes through M6.3's tape cache
    /// instead: `Self::batched_tape` is a pure function of `(bsz, causal_chunk,
    /// matches!(input, Embeds))` (see its own doc), so the FIRST call at a
    /// given `bsz` records the tape into `self.tape_cache` and every later
    /// call at that same `bsz` replays it unchanged - `write_batch_meta`/
    /// `write_batch_input` still run every call (real per-step values, not
    /// cacheable), only the uniform-buffer/bind-group churn `batched_tape`
    /// itself costs is skipped on a hit. Prefill's chunked rows
    /// (`causal_chunk = true`, an unbucketed row count) and `Input::Embeds`
    /// (a structurally different tape at the same `bsz` - no embed step at
    /// all) keep rebuilding every call, exactly as before this cache existed.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_submit(&mut self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32], causal_chunk: bool) -> u32 {
        self.write_batch_meta(input, positions, seqlens, blocks, offsets, bt);
        self.write_batch_input(input);
        if !causal_chunk && bsz <= DECODE_REGIME_MAX_ROWS && !matches!(input, Input::Embeds(_)) {
            if !self.tape_cache.contains_key(&bsz) {
                let s = self.batched_tape(bsz, input, causal_chunk);
                self.tape_cache.insert(bsz, s);
            }
            let cached = self.tape_cache.get(&bsz).expect("just inserted or already present");
            self.gpu.submit(&[], cached);
            return bsz;
        }
        let s = self.batched_tape(bsz, input, causal_chunk);
        self.gpu.submit(&[], &s);
        bsz
    }

    /// One batched decode step that returns the **greedy next token per row**,
    /// with the LM head evaluated on the device.
    ///
    /// The head is the largest single matmul in a small model (`vocab x d_model`
    /// = 16.4M MACs per row at vocab 32k). Applying it on the host, once per
    /// sequence per token, made decode host-bound: cost grew linearly with batch
    /// size while the GPU idled, so continuous batching stopped paying: it
    /// measured as the dominant share of each decode step, and throughput
    /// barely moved from concurrency 1 to 16 before regressing outright.
    ///
    /// Here the hidden state never leaves the device: `matmul` produces
    /// `[bsz, vocab]` logits (parallel over every output element) and
    /// `argmax_row` reduces each row, so only `bsz` indices are read back
    /// instead of a `[bsz, vocab]` block.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_greedy(&mut self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32], causal_chunk: bool) -> Vec<u32> {
        assert!(
            bsz <= self.max_batch,
            "greedy decode is sized for max_batch={} rows, got {bsz}",
            self.max_batch
        );
        self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt, causal_chunk);
        self.greedy_from_hidden(bsz)
    }

    /// `argmax(xn_final @ head^T)` per row, entirely on the device.
    ///
    /// Two-stage reduction: `argmax_part` splits each row into
    /// [`ARGMAX_CHUNKS`] chunks reduced by independent threads, `argmax_final`
    /// folds the partials - `bsz * chunks` threads instead of `bsz`. The
    /// original single-thread-per-row `argmax_row` scanned 32k logits alone
    /// and was 10.3% of decode time; it remains registered as the small-vocab
    /// path and the reference the tests compare against.
    fn greedy_from_hidden(&self, bsz: u32) -> Vec<u32> {
        self.submit_greedy_head(bsz);
        // Indices come back as f32 (exact below 2^24, far above any vocabulary).
        self.gpu.read(&self.argmax_dev, bsz as usize).into_iter().map(|x| x as u32).collect()
    }

    /// Record the head steps that turn `sc.xn_final` into `[bsz, vocab]`
    /// `logits_dev` (int8 or fp32, whichever the engine holds) - shared by
    /// [`Self::submit_greedy_head`] and [`Self::submit_topk_head`], which
    /// otherwise diverge only in what they do with the resulting logits.
    fn head_steps(&self, steps: &mut Vec<Step>, bsz: u32) {
        let d = self.cfg.d_model;
        // The head is just one more `Weight` in `lin_weights` (B7) - int8 or
        // fp32, whichever `weights_int8`/the device's capability landed on -
        // dispatched through the SAME `Self::quant_once`/`Self::linear` pair
        // every per-layer projection uses, no separate int8-head branch.
        self.quant_once(steps, &self.sc.xn_final, d, bsz);
        self.linear(steps, &self.lin_weights[self.cfg.head_weight()], &self.sc.xn_final, &self.logits_dev, bsz);
    }

    /// Record + submit the greedy head (logits + row argmax into
    /// `argmax_dev`) WITHOUT reading back - the on-device decode window feeds
    /// the result straight into the next step.
    fn submit_greedy_head(&self, bsz: u32) {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let mut steps: Vec<Step> = Vec::new();
        self.head_steps(&mut steps, bsz);
        let argmax_shape = OpShape { m: bsz, n: v, k: 0, dtype: Dtype::F32 };
        if self.selector.select(Op::ArgMaxRow, argmax_shape, &self.caps)
            == KernelVariant::SplitReduction
        {
            let chunk = v.div_ceil(ARGMAX_CHUNKS);
            steps.push(g.step(
                ARGMAX_PART,
                &[&self.logits_dev, &self.argmax_part_dev],
                &[bsz, v, ARGMAX_CHUNKS, chunk],
                bsz * ARGMAX_CHUNKS,
            ));
            steps.push(g.step(ARGMAX_FINAL, &[&self.argmax_part_dev, &self.argmax_dev], &[bsz, ARGMAX_CHUNKS], bsz));
        } else {
            steps.push(g.step(ARGMAX_ROW, &[&self.logits_dev, &self.argmax_dev], &[bsz, v], bsz));
        }
        g.submit(&[], &steps);
    }

    /// The row's top-`k` (token id, logit) candidates, best first, entirely
    /// from device work - see [`Self::submit_topk_head`]. `k` is clamped to
    /// [`TOPK_CAPACITY`].
    fn topk_from_hidden(&self, bsz: u32, k: u32) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, TOPK_CAPACITY);
        self.submit_topk_head(bsz, k);
        let vals = self.gpu.read(&self.topk_vals_dev, (bsz * TOPK_CAPACITY) as usize);
        let idx = self.gpu.read(&self.topk_idx_dev, (bsz * TOPK_CAPACITY) as usize);
        (0..bsz as usize)
            .map(|row| {
                let base = row * TOPK_CAPACITY as usize;
                (0..k as usize).map(|c| (idx[base + c] as u32, vals[base + c])).collect()
            })
            .collect()
    }

    /// Record + submit `k` iterations of (argmax pair, `topk_extract_step`)
    /// in ONE submission: each iteration finds the current row maximum over
    /// `logits_dev` (the same two-stage reduction [`Self::submit_greedy_head`]
    /// uses) via the shared `argmax_dev` scratch, records it into column `col`
    /// of `topk_vals_dev`/`topk_idx_dev`, and masks it out of `logits_dev` so
    /// the next iteration finds the row's next-best value. This is what turns
    /// a `[bsz, vocab]` row into a `[bsz, k]` candidate list with only ONE
    /// host round-trip (the final readback in [`Self::topk_from_hidden`]) -
    /// the design point of the whole seam: real (non-greedy) sampling must
    /// never ship the full vocab back to the host.
    fn submit_topk_head(&self, bsz: u32, k: u32) {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let mut steps: Vec<Step> = Vec::new();
        self.head_steps(&mut steps, bsz);
        let argmax_shape = OpShape { m: bsz, n: v, k: 0, dtype: Dtype::F32 };
        let split = self.selector.select(Op::ArgMaxRow, argmax_shape, &self.caps) == KernelVariant::SplitReduction;
        let chunk = v.div_ceil(ARGMAX_CHUNKS);
        for col in 0..k {
            if split {
                steps.push(g.step(
                    ARGMAX_PART,
                    &[&self.logits_dev, &self.argmax_part_dev],
                    &[bsz, v, ARGMAX_CHUNKS, chunk],
                    bsz * ARGMAX_CHUNKS,
                ));
                steps.push(g.step(ARGMAX_FINAL, &[&self.argmax_part_dev, &self.argmax_dev], &[bsz, ARGMAX_CHUNKS], bsz));
            } else {
                steps.push(g.step(ARGMAX_ROW, &[&self.logits_dev, &self.argmax_dev], &[bsz, v], bsz));
            }
            steps.push(g.step(
                TOPK_EXTRACT_STEP,
                &[&self.argmax_dev, &self.logits_dev, &self.topk_vals_dev, &self.topk_idx_dev],
                &[bsz, v, TOPK_CAPACITY, col],
                bsz,
            ));
        }
        g.submit(&[], &steps);
    }

    /// Advance every sequence by one token, returning the row's top-`k`
    /// (token id, logit) candidates instead of a single greedy token - the
    /// entry point a caller doing real (non-greedy) sampling uses in place of
    /// [`Self::forward_batched_greedy`]. `logits_dev` is mutated (masked) by
    /// the extraction, exactly as `argmax_dev` already is by the greedy path -
    /// callers never read either between decode steps.
    pub(crate) fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: u32) -> Vec<Vec<(u32, f32)>> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched_submit(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt, false);
        self.topk_from_hidden(bsz, k)
    }

    /// Advance every sequence by `k` tokens with ONE host round-trip (A4).
    ///
    /// The host allocates the window's K/V slots up front and uploads them as
    /// a device schedule; between sub-steps `decode_feed` turns the argmax
    /// into the next input and `decode_advance` walks the metadata, so the
    /// whole window records into one submission and the only readback is the
    /// `[k, bsz]` token history at the end. Callers own EOS handling: a
    /// sequence that finishes mid-window has its surplus tokens trimmed and
    /// its surplus K/V slots rolled back (`BlockTable::truncate`).
    ///
    /// Requires `k <= DECODE_WINDOW` and enough free blocks for every append;
    /// the scheduler falls back to the single-step path when either fails.
    pub(crate) fn forward_batched_greedy_window(
        &mut self,
        tables: &mut [&mut BlockTable],
        inputs: &[u32],
        k: usize,
    ) -> Vec<Vec<u32>> {
        assert!((1..=DECODE_WINDOW).contains(&k), "window {k} out of range");
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        let mbt = self.max_blocks_per_seq;
        // Pre-allocate sub-steps 1..k and build the device schedule:
        // (block, offset, bt_index | NO_BT) per row per sub-step.
        const NO_BT: u32 = u32::MAX;
        let n = bsz as usize;
        let mut sched = vec![0u32; (k - 1) * n * 3];
        for s in 1..k {
            for (i, table) in tables.iter_mut().enumerate() {
                let before = table.blocks().len();
                let (block, offset) =
                    table.append(&mut self.alloc).expect("KV pool exhausted mid-window");
                let bti = if table.blocks().len() > before {
                    (table.blocks().len() - 1) as u32
                } else {
                    NO_BT
                };
                let base = ((s - 1) * n + i) * 3;
                sched[base] = block;
                sched[base + 1] = offset;
                sched[base + 2] = bti;
            }
        }
        let g = &self.gpu;
        if k > 1 {
            g.write(&self.sc.sched_buf, &sched);
        }
        // Sub-step 0: host-fed, as today - but the argmax stays on the device.
        self.run_batched_submit(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt, false);
        self.submit_greedy_head(bsz);
        for s in 1..k {
            let g = &self.gpu;
            let feed = g.step(
                DECODE_FEED,
                &[&self.argmax_dev, &self.sc.tok_buf, &self.sc.hist_buf],
                &[bsz, (s - 1) as u32],
                bsz,
            );
            let adv = g.step(
                DECODE_ADVANCE,
                &[
                    &self.sc.sched_buf,
                    &self.sc.pos_buf,
                    &self.sc.seqlen_buf,
                    &self.sc.blk_buf,
                    &self.sc.off_buf,
                    &self.sc.bt_buf,
                ],
                &[bsz, (s - 1) as u32, mbt, NO_BT],
                bsz,
            );
            g.submit(&[], &[feed, adv]);
            self.run_batched_submit(bsz, Input::Resident, &[], &[], &[], &[], &[], false);
            self.submit_greedy_head(bsz);
        }
        // Record the final sub-step's tokens, then read the whole window once.
        let g = &self.gpu;
        let last = g.step(
            DECODE_FEED,
            &[&self.argmax_dev, &self.sc.tok_buf, &self.sc.hist_buf],
            &[bsz, (k - 1) as u32],
            bsz,
        );
        g.submit(&[], &[last]);
        let flat = g.read(&self.sc.hist_buf, k * n);
        (0..n).map(|i| (0..k).map(|s| flat[s * n + i] as u32).collect()).collect()
    }

    /// **Chunked prefill**: process the prompt in chunks of up to `max_prefill`
    /// tokens. Each chunk is a batched forward whose C queries attend the paged
    /// prefix + the causal chunk (seqlens[i] = start+i+1), scattering K/V into the
    /// pool for the decode phase. One chunk == whole-prompt prefill; larger prompts
    /// stream through without a giant single forward. Returns the last token's
    /// final-norm hidden `[d_model]`.
    pub(crate) fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "prefill expects a fresh sequence");
        // A prompt longer than the per-sequence capacity would write past its
        // row of the block table (`bt` is sized cc * max_blocks_per_seq), which
        // silently corrupts the next row's mapping. Callers must check
        // `max_seq_len()`; the scheduler rejects such requests at admission.
        assert!(
            prompt.len() <= self.max_seq_len(),
            "prompt of {} tokens exceeds the engine's per-sequence capacity of {} \
             (max_blocks_per_seq {} x block_size {})",
            prompt.len(),
            self.max_seq_len(),
            self.max_blocks_per_seq,
            self.block_size,
        );
        // An out-of-vocab id would make the embedding gather read out of
        // bounds - the kernels are trusted (no per-access clamps on either
        // backend), so the failure is silent garbage, not a clean error. The
        // scheduler rejects such requests at admission; this backstop catches
        // callers that bypass it.
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("prompt token {bad} is outside the model vocabulary ({})", self.cfg.vocab);
        }
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let n = prompt.len() as u32;
        let chunk = self.max_prefill.max(1);
        // Prefix reuse (D): adopt the longest cached chain of full prompt
        // blocks and compute only the tail. Always leave at least one token to
        // compute - the caller needs the LAST token's hidden state, which only
        // a real forward produces.
        let max_reuse = prompt.len().saturating_sub(1);
        let hits = self.prefix.lookup(prompt, bs, max_reuse);
        let matched = hits.len();
        if matched > 0 {
            table.adopt_prefix(&hits, &mut self.alloc);
        }
        self.prefix_lookup_tokens += prompt.len() as u64;
        self.prefix_hit_tokens += (matched as u64) * bs as u64;
        let mut last = Vec::new();
        let mut start = matched as u32 * bs;
        while start < n {
            let cc = (n - start).min(chunk);
            table.reserve(cc, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; cc as usize * mbt];
            for i in 0..cc {
                let pos = start + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1); // causal: query i attends positions 0..=pos
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt, true);
            let cu = cc as usize;
            last = hidden[(cu - 1) * d..cu * d].to_vec();
            start += cc;
        }
        // Index this prompt's freshly-computed full blocks for later prompts.
        self.prefix.insert_chain(prompt, table.blocks(), matched, &mut self.alloc);
        last
    }

    /// Like [`Engine::prefill`], but returns EVERY position's final-norm
    /// hidden state (`[prompt.len() * d_model]`, row-major) instead of only
    /// the last - what teacher-forced held-out scoring needs (`qwen3::eval`),
    /// where every position's loss counts, not just the next-token
    /// prediction after the whole prompt.
    ///
    /// Deliberately bypasses the prefix cache (`self.prefix`) entirely: an
    /// eval pass scores a set of independent held-out samples, not a live
    /// conversation, so there is no shared prefix to exploit and no reason
    /// to let one sample's cache entries affect another's - full recompute
    /// per sample keeps the measurement simple and reproducible. `run_batched`
    /// itself already computes every chunk row's hidden state (`prefill`
    /// just slices out the last one); this keeps all of them.
    pub(crate) fn score_positions(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "score_positions expects a fresh sequence");
        assert!(
            prompt.len() <= self.max_seq_len(),
            "prompt of {} tokens exceeds the engine's per-sequence capacity of {}",
            prompt.len(),
            self.max_seq_len()
        );
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("prompt token {bad} is outside the model vocabulary ({})", self.cfg.vocab);
        }
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let n = prompt.len() as u32;
        let chunk = self.max_prefill.max(1);
        let mut out = vec![0f32; prompt.len() * d];
        let mut start = 0u32;
        while start < n {
            let cc = (n - start).min(chunk);
            table.reserve(cc, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; cc as usize * mbt];
            for i in 0..cc {
                let pos = start + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1);
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt, true);
            out[start as usize * d..(start + cc) as usize * d].copy_from_slice(&hidden[..cc as usize * d]);
            start += cc;
        }
        out
    }

    /// Release up to `want` least-recently-used cache-only prefix blocks back
    /// to the pool - the admission path calls this when the pool is short.
    pub(crate) fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.prefix.evict(want, &mut self.alloc)
    }

    /// Prefix-cache effectiveness: `(tokens served from cache, tokens looked
    /// up, full blocks currently cached)`.
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        (self.prefix_hit_tokens, self.prefix_lookup_tokens, self.prefix.len())
    }

    /// Device-op accounting for this engine's handle (K) - what a benchmark
    /// records so submit/dispatch/readback cost is machine-readable. `None`
    /// where the backend does not count.
    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }

    /// `logits = hidden @ head^T` for ONE row - what a host-side caller doing
    /// its own argmax/sampling over the full vocabulary needs
    /// ([`Self::generate_greedy`], [`Self::spec_decode`], `qwen3::eval`).
    /// Writes `hidden` into `sc.xn_final` and reuses [`Self::head_steps`] -
    /// the SAME device dispatch [`Self::submit_greedy_head`]/
    /// [`Self::submit_topk_head`] build on - so this head matmul takes the
    /// identical path decode's does rather than a separate host GEMV.
    /// [`Self::admit_greedy`]/[`Self::admit_topk`] are what admission itself
    /// calls; they skip this method entirely so admission never reads back a
    /// full `[vocab]` block.
    pub(crate) fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        assert_eq!(hidden.len(), d, "logits: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        self.gpu.write_f32(&self.sc.xn_final, hidden);
        let mut steps: Vec<Step> = Vec::new();
        self.head_steps(&mut steps, 1);
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.logits_dev, v)
    }

    /// Admission's greedy pick, entirely on the device: writes `hidden` (ONE
    /// row) into `sc.xn_final` and reuses [`Self::submit_greedy_head`] - the
    /// same head matmul + argmax reduction the batched decode loop
    /// dispatches - so admission never ships a `[vocab]` block to the host to
    /// pick one token.
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        let d = self.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "admit_greedy: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        self.gpu.write_f32(&self.sc.xn_final, hidden);
        self.greedy_from_hidden(1)[0]
    }

    /// Admission's non-greedy candidates, entirely on the device: writes
    /// `hidden` (ONE row) into `sc.xn_final` and reuses
    /// [`Self::topk_from_hidden`] - the same top-k extraction the non-greedy
    /// decode path uses - so admission never sorts a `[vocab]` vector on the
    /// host either.
    fn admit_topk(&self, hidden: &[f32], k: u32) -> Vec<(u32, f32)> {
        let d = self.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "admit_topk: hidden must be exactly one row of {d} floats, got {}", hidden.len());
        self.gpu.write_f32(&self.sc.xn_final, hidden);
        self.topk_from_hidden(1, k).pop().unwrap_or_default()
    }

    /// Blocks free in the pool - the capacity figure `brain perf kvcache` sizes
    /// its overcommitted session mix against.
    pub fn free_blocks_for_perf(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// Prefill a prompt and return the final hidden state. Public so the
    /// `startup` benchmark can time a first real forward without going through
    /// the scheduler.
    pub fn prefill_for_perf(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        self.prefill(table, prompt)
    }

    /// The dispatches of one served step, in submit order, WITHOUT submitting.
    ///
    /// Profiler-only (`qwen_bench serve`), the same contract and the same
    /// reason as `Qwen::fwd_steps`: `gpu_core::profile` needs a step list, and
    /// this tape is rebuilt per step rather than recorded once. `bsz` rows with
    /// `seqlens[i] = positions[i] + 1` is a decode step; a chunk of `cc` rows
    /// from one sequence is a prefill chunk - the two share this tape, which is
    /// exactly why profiling it is worth doing. `causal_chunk` selects which
    /// one the caller built (see `run_batched_steps`'s own doc, M2.4).
    #[allow(clippy::too_many_arguments)]
    pub fn steps_for_profile(&self, bsz: u32, tokens: &[u32], positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32], causal_chunk: bool) -> Vec<Step> {
        // `Input::Tokens`, NOT `Input::Resident`. Resident mode is the on-device
        // decode window: it deliberately performs no host writes because
        // `decode_feed`/`decode_advance` already produced the token ids AND the
        // paged metadata on the device. Using it from a profiler leaves
        // `seq_lens` at whatever was in the buffer - zero - so every attention
        // thread early-returns and the kernels appear to do almost no work.
        // That is exactly how `paged_decode_scores_batched` came to report a
        // bandwidth well above what the card can physically deliver: the timing
        // was right and the kernel really was that fast, because it was not
        // attending to anything.
        self.run_batched_steps(bsz, Input::Tokens(tokens), positions, seqlens, blocks, offsets, bt, causal_chunk).0
    }

    /// Physical KV blocks currently free in the pool.
    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    /// The prefill chunk size this engine was built with - the unit the
    /// scheduler's per-iteration prefill budget is expressed against.
    pub fn max_prefill_tokens(&self) -> u32 {
        self.max_prefill
    }

    /// The longest sequence (prompt + generated) this engine can hold for one
    /// request: `max_blocks_per_seq * block_size`.
    pub fn max_seq_len(&self) -> usize {
        (self.max_blocks_per_seq * self.block_size) as usize
    }

    /// Blocks a sequence of `tokens` length occupies (for admission checks).
    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }
    pub(crate) fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }

    // ---- host-RAM KV offload (`model::kv_offload`) ----------------------

    /// Give this engine `capacity_bytes` of host RAM to hold preempted
    /// sequences' KV in, enabling [`model::serve::Scheduler`]'s swap-based
    /// preemption. `0` (the default) turns it off.
    ///
    /// What it buys is CONCURRENCY, not a longer single sequence: an admitted
    /// sequence the scheduler is not advancing this round costs the device
    /// nothing while its bytes sit here, so the pool backs the sequences that
    /// are actually decoding. Per-sequence context is unchanged - a sequence
    /// being decoded still needs all of its KV resident, because causal
    /// attention reads all of it every step (see `model::kv_offload`'s header
    /// for the measured bus numbers behind that).
    pub fn set_kv_offload_bytes(&mut self, capacity_bytes: u64) {
        self.host_kv.set_capacity_bytes(capacity_bytes);
    }

    /// Host bytes this engine may hold demoted KV in.
    pub fn kv_offload_bytes(&self) -> u64 {
        self.host_kv.capacity_bytes()
    }

    /// Swap accounting - demotions, promotions, blocks moved, host bytes held.
    pub fn kv_offload_stats(&self) -> OffloadStats {
        self.host_kv.stats()
    }

    /// Host bytes one cached token costs if its sequence is demoted -
    /// identical to what it costs on the device, since a swap is verbatim.
    pub fn kv_offload_bytes_per_token(&self) -> f64 {
        self.kv_block_words as f64 * 4.0 / self.block_size as f64
    }

    /// Build the swap staging buffers on first use. Sized to
    /// [`KV_SWAP_STAGING_BYTES`], clamped to the pool's own block count (never
    /// stage more blocks than can exist) and to what one storage binding on
    /// this device allows.
    fn ensure_swap_bufs(&mut self) {
        if self.swap.is_some() {
            return;
        }
        let per_block = self.kv_block_words as u64 * 4;
        let cap_bytes = KV_SWAP_STAGING_BYTES.min(self.gpu.max_storage_binding_bytes());
        let mut blocks = (cap_bytes / per_block).clamp(1, self.alloc.num_blocks() as u64) as u32;
        // Halve on a memory-ceiling refusal rather than panicking: this
        // allocation happens the first time a sequence is preempted, which is
        // exactly when the device is under pressure, and a smaller chunk costs
        // transfers, not correctness. One block always fits - it is a fraction
        // of a pool that is already resident.
        let (ids, staging) = loop {
            let want = blocks as u64 * self.kv_block_words as u64;
            match (self.gpu.try_storage(blocks as u64), self.gpu.try_storage(want)) {
                (Ok(ids), Ok(staging)) => break (ids, staging),
                _ if blocks > 1 => blocks /= 2,
                (ids, staging) => panic!("qwen3 serve: KV swap staging for a single block was refused: {:?}", ids.err().or(staging.err())),
            }
        };
        self.swap = Some(SwapBufs { ids, staging, blocks });
    }

    /// Every `(pool buffer, words per block)` a swap has to move, in the order
    /// that defines one block's staging record: per layer, K pool, V pool,
    /// then (int8 only) K scales, V scales. Returned as offsets so the gather
    /// and the scatter cannot drift apart - both walk this one list.
    fn kv_swap_plan(&self) -> Vec<(&DeviceBuffer, u32, u32)> {
        let (pool_words, scale_words) = kv_pool_words(&self.cfg, self.block_size, 1, self.kv_int8);
        let (pw, sw) = (pool_words as u32, scale_words as u32);
        let mut plan = Vec::new();
        let mut off = 0u32;
        for l in 0..self.cfg.n_layers as usize {
            for buf in [&self.pool_k[l], &self.pool_v[l]] {
                plan.push((buf, pw, off));
                off += pw;
            }
            if self.kv_int8 {
                for buf in [&self.scales_k[l], &self.scales_v[l]] {
                    plan.push((buf, sw, off));
                    off += sw;
                }
            }
        }
        debug_assert_eq!(off as usize, self.kv_block_words, "the swap plan must cover exactly one block's words");
        plan
    }

    /// Gather one chunk of blocks into the staging buffer and read it back.
    fn swap_out_chunk(&mut self, chunk: &[u32], out: &mut Vec<u32>) {
        self.ensure_swap_bufs();
        let sw = self.swap.as_ref().expect("just built");
        let n = chunk.len() as u32;
        let stride = self.kv_block_words as u32;
        self.gpu.write(&sw.ids, chunk);
        let steps: Vec<Step> = self
            .kv_swap_plan()
            .into_iter()
            .map(|(buf, wpb, off)| self.gpu.step(KV_GATHER, &[&sw.ids, buf, &sw.staging], &[n, wpb, off, stride], n * wpb))
            .collect();
        self.gpu.submit(&[], &steps);
        // `read` hands back `f32`, but the bytes are whatever the pool holds
        // (packed int8, a scale, an fp32 K) - `to_bits` recovers the words
        // exactly, with no float ever computed on them.
        let got = self.gpu.read(&sw.staging, (n * stride) as usize);
        out.extend(got.iter().map(|v| v.to_bits()));
    }

    /// Upload one chunk of staged blocks and scatter it back into the pool.
    fn swap_in_chunk(&mut self, chunk: &[u32], words: &[u32]) {
        self.ensure_swap_bufs();
        let sw = self.swap.as_ref().expect("just built");
        let n = chunk.len() as u32;
        let stride = self.kv_block_words as u32;
        debug_assert_eq!(words.len(), (n * stride) as usize);
        self.gpu.write(&sw.ids, chunk);
        self.gpu.write(&sw.staging, words);
        let steps: Vec<Step> = self
            .kv_swap_plan()
            .into_iter()
            .map(|(buf, wpb, off)| self.gpu.step(KV_SCATTER, &[&sw.ids, &sw.staging, buf], &[n, wpb, off, stride], n * wpb))
            .collect();
        self.gpu.submit(&[], &steps);
    }

    fn argmax(s: &[f32]) -> u32 {
        let mut bi = 0;
        for i in 1..s.len() {
            if s[i] > s[bi] {
                bi = i;
            }
        }
        bi as u32
    }

    /// Prefill every prompt, then read back the K/V rows each one wrote and
    /// accumulate per-`(layer, K|V, kv_head)` activation-magnitude statistics
    /// (`model::actstats`) - the design input for a calibrated INT8 KV scale
    /// (`brain qwen calib`, `crates/qwen3/src/calib.rs`).
    ///
    /// Offline-only, never called from the hot serving path (`run_batched_submit`
    /// stays untouched): this reads the pool directly with plain [`Gpu::read`]
    /// calls between prefills, which is fine for a one-shot calibration pass
    /// over a modest prompt set but is NOT the shape a per-request tap could
    /// use without a real perf cost.
    ///
    /// Needs an fp32-KV engine (`kv_int8: false`) - calibration wants the
    /// pre-quantization distribution, not a value already thrown away by
    /// today's online absmax. K rows are read POST-RoPE (RoPE runs before
    /// the KV append in `run_batched_submit`), matching exactly
    /// what a real INT8 KV scale would be quantizing.
    pub fn calibrate_kv(&mut self, prompts: &[Vec<u32>]) -> model::actstats::Collector {
        assert!(!self.kv_int8, "calibrate_kv needs an fp32-KV engine (build with kv_int8: false)");
        let collector = model::actstats::Collector::new();
        let hd = self.cfg.head_dim as usize;
        let n_kv = self.cfg.n_kv_heads as usize;
        let hkv = n_kv * hd;
        let bs = self.block_size as usize;

        // Prefill every prompt first, keeping every table alive - the
        // allocator must not recycle a prompt's blocks before we've read
        // them back below.
        let mut tables: Vec<(BlockTable, usize)> = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            assert!(!prompt.is_empty(), "calibrate_kv: empty prompt");
            let mut table = BlockTable::new();
            self.prefill(&mut table, prompt);
            tables.push((table, prompt.len()));
        }

        // One full-pool readback per layer (not per prompt): cheap relative
        // to the prefills themselves, and simplest to get right.
        let slots = self.alloc.num_blocks() as u64 * self.block_size as u64;
        let pool_words = (slots * hkv as u64) as usize;
        for l in 0..self.cfg.n_layers as usize {
            let pk = self.gpu.read(&self.pool_k[l], pool_words);
            let pv = self.gpu.read(&self.pool_v[l], pool_words);
            for (table, len) in &tables {
                for tok in 0..*len {
                    let phys = table.blocks()[tok / bs] as usize;
                    let off = tok % bs;
                    let base = (phys * bs + off) * hkv;
                    for h in 0..n_kv {
                        collector.observe(&format!("layer{l:02}.k.head{h}"), &pk[base + h * hd..base + (h + 1) * hd]);
                        collector.observe(&format!("layer{l:02}.v.head{h}"), &pv[base + h * hd..base + (h + 1) * hd]);
                    }
                }
            }
        }

        for (mut table, _) in tables {
            self.release_table(&mut table);
        }
        collector
    }

    /// Greedy generation of `max_new` tokens for each prompt, run with a **paged
    /// KV cache** and **batched decode** across all prompts concurrently. Prompts
    /// are prefilled per-sequence (one token per step), then every still-running
    /// sequence advances together each decode iteration. Returns the generated
    /// tokens per prompt. `eos` (when set) stops a sequence early.
    pub fn generate_greedy(&mut self, prompts: &[Vec<u32>], max_new: usize, eos: Option<u32>) -> Vec<Vec<u32>> {
        let mut seqs: Vec<Seq> = prompts.iter().map(|_| Seq { table: BlockTable::new(), generated: Vec::new(), done: false }).collect();

        // Prefill each sequence and sample its first token.
        for (i, prompt) in prompts.iter().enumerate() {
            assert!(!prompt.is_empty(), "empty prompt");
            let hidden = self.prefill(&mut seqs[i].table, prompt);
            let first = Self::argmax(&self.logits(&hidden));
            seqs[i].generated.push(first);
            if Some(first) == eos {
                seqs[i].done = true;
            }
        }

        // Batched decode: feed each running sequence its last token together.
        for _ in 1..max_new {
            let active: Vec<usize> = (0..seqs.len()).filter(|&i| !seqs[i].done).collect();
            if active.is_empty() {
                break;
            }
            let inputs: Vec<u32> = active.iter().map(|&i| *seqs[i].generated.last().unwrap()).collect();
            // Borrow the active sequences' block tables mutably for the batched step.
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, seq) in seqs.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut seq.table);
                    }
                }
                self.forward_batched(&mut refs, &inputs)
            };
            let d = self.cfg.d_model as usize;
            for (bi, &si) in active.iter().enumerate() {
                let next = Self::argmax(&self.logits(&hidden[bi * d..(bi + 1) * d]));
                seqs[si].generated.push(next);
                if Some(next) == eos {
                    seqs[si].done = true;
                }
            }
        }
        for s in seqs.iter_mut() {
            self.release_table(&mut s.table);
        }
        seqs.into_iter().map(|s| s.generated).collect()
    }

    /// **Speculative decoding** (greedy): a `draft` proposes up to `k` tokens from
    /// the running context; the target verifies them in ONE batched forward,
    /// accepting the longest correct prefix plus a bonus/correction token, and
    /// rolling the paged cache back over any rejected tokens. The output is
    /// identical to plain greedy target decoding - the win is fewer (expensive)
    /// target forwards when the draft guesses well. `draft(ctx, want) -> tokens`.
    /// Returns `(generated_tokens, target_forward_count)`.
    pub fn spec_decode<D: FnMut(&[u32], u32) -> Vec<u32>>(&mut self, prompt: &[u32], max_new: usize, k: u32, mut draft: D) -> (Vec<u32>, usize) {
        assert!(!prompt.is_empty() && k >= 1);
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let mut table = BlockTable::new();
        // Prefill all but the last prompt token; the last is the first `pending`.
        if prompt.len() > 1 {
            self.prefill(&mut table, &prompt[..prompt.len() - 1]);
        }
        let mut pending = *prompt.last().unwrap();
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut forwards = 0usize;

        while generated.len() < max_new {
            let want = ((max_new - generated.len()) as u32).min(k);
            let mut props = draft(&ctx, want);
            props.truncate(want as usize);
            let kk = props.len() as u32;

            // Verify forward over [pending, props...] at positions base..=base+kk.
            let base = table.len();
            let inputs: Vec<u32> = std::iter::once(pending).chain(props.iter().copied()).collect();
            let rows = kk + 1;
            table.reserve(rows, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; rows as usize * mbt];
            for i in 0..rows {
                let pos = base + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1);
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            // `causal_chunk = false`: this verify-forward structurally
            // matches the fused kernel's contract too (one sequence, one
            // shared block table, causally-increasing `seqlens`) but is left
            // on the triad for THIS milestone - `paged_flash_prefill`'s own
            // correctness gate and M2.4's own wiring test only cover
            // `Engine::prefill`/`score_positions`'s own construction; wiring
            // this call site in as well is a follow-on, not re-litigated here.
            let hidden = self.run_batched(rows, Input::Tokens(&inputs), &positions, &seqlens, &blocks, &offsets, &bt, false);
            forwards += 1;

            // hidden[j] gives the target distribution that should have produced
            // props[j]; accept while it matches, else take the target's own token.
            let mut accepted = 0usize;
            let correction;
            loop {
                if accepted < kk as usize {
                    let pred = Self::argmax(&self.logits(&hidden[accepted * d..(accepted + 1) * d]));
                    if pred == props[accepted] {
                        accepted += 1;
                        continue;
                    }
                    correction = pred;
                    break;
                }
                // All drafts accepted → the bonus token from the last position.
                correction = Self::argmax(&self.logits(&hidden[kk as usize * d..(kk as usize + 1) * d]));
                break;
            }
            for prop in props.iter().take(accepted) {
                generated.push(*prop);
                ctx.push(*prop);
            }
            generated.push(correction);
            ctx.push(correction);
            // Commit pending + accepted drafts; the correction is the next pending.
            table.truncate(base + accepted as u32 + 1, &mut self.alloc);
            pending = correction;
        }
        generated.truncate(max_new);
        table.release(&mut self.alloc);
        (generated, forwards)
    }
}

// The continuous-batching scheduler (`Request`, `RejectReason`, `QueueState`,
// `AdmissionPolicy` + `UnboundedQueue`/`MaxQueueDepth`/`DeadlineAware`,
// `StepReport`, `Scheduler`) moved to `model::serve` -- it is architecture-
// agnostic over `model::serve::PagedDecoder`, not Qwen-specific. Re-exported
// here so every existing `qwen3::serve::Scheduler`/`Request`/... caller
// (`crates/cli/src/perf_cli.rs`, `crates/perf/src/targets.rs`,
// `crates/cli/src/perf_engine.rs`) needs zero changes.
pub use model::serve::{AdmissionPolicy, DeadlineAware, MaxQueueDepth, QueueState, RejectReason, Request, SampleParams, StepReport, UnboundedQueue};
pub type Scheduler = model::serve::Scheduler<Engine>;

/// The device half of host-RAM KV offload for this engine: how a set of
/// physical blocks moves between the paged pools and host memory. The swap
/// PROTOCOL itself (budgets, refcounts, rollback) is
/// `model::kv_offload::KvOffload`'s provided methods, shared with every future
/// adopter - only these four are Qwen-specific, and even they are written
/// against `kv_pool_words`, this engine's own single source of pool layout.
impl KvOffload for Engine {
    fn kv_block_words(&self) -> usize {
        self.kv_block_words
    }
    fn kv_alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn host_kv_mut(&mut self) -> &mut HostKvPool {
        &mut self.host_kv
    }
    fn read_kv_blocks(&mut self, blocks: &[u32], out: &mut Vec<u32>) {
        let per_chunk = {
            self.ensure_swap_bufs();
            self.swap.as_ref().expect("just built").blocks as usize
        };
        for chunk in blocks.chunks(per_chunk) {
            self.swap_out_chunk(chunk, out);
        }
    }
    fn write_kv_blocks(&mut self, blocks: &[u32], words: &[u32]) {
        let per_chunk = {
            self.ensure_swap_bufs();
            self.swap.as_ref().expect("just built").blocks as usize
        };
        let stride = self.kv_block_words;
        for (i, chunk) in blocks.chunks(per_chunk).enumerate() {
            let at = i * per_chunk * stride;
            self.swap_in_chunk(chunk, &words[at..at + chunk.len() * stride]);
        }
    }
}

impl model::serve::PagedDecoder for Engine {
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
        self.release_table(t)
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        Engine::prefill(self, table, prompt)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.logits(hidden)
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        Engine::admit_greedy(self, hidden)
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        Engine::admit_topk(self, hidden, k as u32)
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
        DECODE_WINDOW
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        Engine::forward_batched_topk(self, tables, inputs, k as u32)
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY as usize
    }
    fn kv_offload_bytes(&self) -> u64 {
        Engine::kv_offload_bytes(self)
    }
    fn demote_sequence(&mut self, key: u64, table: &mut BlockTable) -> Result<u32, KvOffloadError> {
        self.demote_kv(key, table)
    }
    fn promote_sequence(&mut self, key: u64) -> Result<BlockTable, KvOffloadError> {
        self.promote_kv(key)
    }
    fn offloaded_blocks(&self, key: u64) -> Option<u32> {
        self.host_kv.blocks_of(key)
    }
    fn discard_offloaded(&mut self, key: u64) -> bool {
        self.discard_kv(key)
    }
    fn offload_stats(&self) -> OffloadStats {
        self.host_kv.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Qwen;
    use data::rng::Rng;

    /// `ops_kernel_list()` (the list `Ops::new` is built from at
    /// `Engine::from_map_with_gpu` - see that function's own call site)
    /// against `model::ops::REQUIRED_KERNELS` - a pure name-set comparison,
    /// no `Gpu`/GPU device required, so it runs even where every OTHER test
    /// in this module needs a real device. This is exactly the check that
    /// would have caught `ops_kernel_list` being 15 kernels short of
    /// `REQUIRED_KERNELS` (missing `embed`, `moe_linear_gated`, every
    /// `paged_*_batched` bf16 tier, and `matmul_dx`/`matmul_dw`) at `cargo
    /// test` time - the gap instead only surfaced as `Ops::new`'s own `Err`
    /// on a live server's first real request, because `Engine::from_map_*`
    /// is only ever reached via the residency pool's lazy `activate()` (GPU
    /// activation on-demand so many resident models can share one GPU), not
    /// eagerly at `brain serve` startup.
    #[test]
    fn ops_kernel_list_has_every_kernel_ops_new_requires() {
        model::ops::assert_kernel_list_complete(ops_kernel_list());
    }

    fn tiny_weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(1);
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") { vec![1.0f32; count] } else { (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect() };
            map.insert(name, v);
        }
        map
    }

    /// A small config where every dimension is DISTINCT and non-degenerate --
    /// unlike `QwenConfig::tiny()` (`n_kv_heads=2, head_dim=8`, several numbers
    /// coincide), this exercises a real GQA ratio (group 2) at a `head_dim`
    /// that packs to more than one int8 word/head (16/4=4), with
    /// `n_heads*head_dim=96 != d_model=20` so a transposed/mismatched dimension
    /// cannot accidentally read as correct. Replaces `tiny()` for int8-KV
    /// numeric gates -- degenerate test dims hide bug
    /// classes, and a toy-fitted constant cannot predict the real shape.
    fn kv_probe_cfg() -> QwenConfig {
        QwenConfig {
            vocab: 29,
            block_size: 64,
            n_layers: 3,
            d_model: 20,
            n_heads: 6,
            n_kv_heads: 3,
            head_dim: 16,
            d_ff: 28,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 64,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// G3 (the scale-bug gate - lesson 2: cosine cannot see a dropped scale,
    /// and the int8 bug class IS a scale bug). ONE int8 engine, ONE prefill
    /// (the whole prompt fits in a single `max_prefill`-sized chunk, so it is
    /// exactly one forward pass): the ground truth for "what was quantized"
    /// is read straight out of the engine's OWN scratch (`sc.k`/`sc.v`), which
    /// still hold the last layer's post-RoPE K/V - the literal `src` the
    /// append kernel just packed - because nothing overwrites them after the
    /// final layer's dispatch. This deliberately avoids comparing against a
    /// SEPARATE fp32 engine: two independently-built engines can select
    /// different autotuned kernel variants for the identical (op, shape) and
    /// differ by GPU floating-point noise well under any real scale bug but
    /// well above `assert_eq!` - the oracle must come from the same
    /// computation being checked, not a second one hoped to agree with it.
    ///
    /// Per `(token, kv-head)`, every element: the scale is EXACTLY
    /// `absmax/127` (or `1.0` when `absmax==0`), the stored byte is EXACTLY
    /// `clamp(round(x/scale), -127, 127)`, the dequantized value sits within
    /// half a quantization step of the truth, and the whole-tensor `rel_l2`
    /// stays under a DERIVED bound (not a hand-fitted one) - `rel_l2` because
    /// cosine alone cannot see a dropped or doubled scale factor.
    #[test]
    fn int8_kv_scale_and_bytes_match_a_host_oracle() {
        let cfg = kv_probe_cfg();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 11, 4];
        let (bs, nb) = (5u32, 32u32);
        let mut e8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, nb, 1, 8, 32, true, false);
        let mut t8 = BlockTable::new();
        e8.prefill(&mut t8, &prompt);

        let hd = cfg.head_dim as usize;
        let n_kv = cfg.n_kv_heads as usize;
        let hkv = n_kv * hd;
        let block_size = bs as usize;
        let last_layer = cfg.n_layers as usize - 1;
        let (pool_words8, scale_words) = kv_pool_words(&cfg, bs, nb, true);

        // sc.k/sc.v: [prompt.len(), hkv] row-major, one row per TOKEN position
        // (not physical slot) -- exactly what `run_batched(cc, ...)` wrote for
        // this single unchunked prefill.
        let xk = e8.gpu.read(&e8.sc.k, prompt.len() * hkv);
        let xv = e8.gpu.read(&e8.sc.v, prompt.len() * hkv);
        let pk8: Vec<u32> = e8.gpu.read(&e8.pool_k[last_layer], pool_words8 as usize).iter().map(|f| f.to_bits()).collect();
        let pv8: Vec<u32> = e8.gpu.read(&e8.pool_v[last_layer], pool_words8 as usize).iter().map(|f| f.to_bits()).collect();
        let scales_k = e8.gpu.read(&e8.scales_k[last_layer], scale_words as usize);
        let scales_v = e8.gpu.read(&e8.scales_v[last_layer], scale_words as usize);

        let check = |name: &str, x: &[f32], packed: &[u32], scales: &[f32]| {
            let mut sq_err = 0f64;
            let mut sq_mag = 0f64;
            for tok in 0..prompt.len() {
                let phys = t8.blocks()[tok / block_size] as usize;
                let off = tok % block_size;
                let slot = phys * block_size + off;
                for h in 0..n_kv {
                    let xbase = tok * hkv + h * hd;
                    let xh = &x[xbase..xbase + hd];
                    let absmax = xh.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    let expected_scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
                    let actual_scale = scales[slot * n_kv + h];
                    // A GPU division instruction is not guaranteed IEEE-correctly-
                    // rounded (WGSL allows ~1 ULP slack), so host and device can
                    // legitimately disagree in the last bit of `absmax/127.0` --
                    // tolerance is 8+ orders of magnitude tighter than any real
                    // scale bug (dropped/halved/doubled), which lands orders of
                    // magnitude away, not one ULP away.
                    let tol = (expected_scale.abs() * 2e-6).max(1e-12);
                    assert!((actual_scale - expected_scale).abs() <= tol, "{name} tok{tok} head{h}: scale drifted past a GPU-division ULP: actual={actual_scale} expected={expected_scale}");

                    let base = slot * hkv + h * hd;
                    for (d, &xv) in xh.iter().enumerate() {
                        let e = base + d;
                        let word = packed[e / 4];
                        let byte = (word >> (8 * (e % 4))) & 0xff;
                        let signed = if byte > 127 { byte as i32 - 256 } else { byte as i32 };
                        let expected_qv = (xv / actual_scale).round().clamp(-127.0, 127.0) as i32;
                        assert_eq!(signed, expected_qv, "{name} tok{tok} head{h} d{d}: quantized byte must be exact");
                        let dequant = signed as f32 * actual_scale;
                        let err = (dequant - xv).abs();
                        assert!(err <= 0.5 * actual_scale + 1e-6, "{name} tok{tok} head{h} d{d}: dequant off by more than half a step: {err} vs scale {actual_scale}");
                        sq_err += (err as f64).powi(2);
                        sq_mag += (xv as f64).powi(2);
                    }
                }
            }
            let rel_l2 = (sq_err / sq_mag.max(1e-12)).sqrt();
            assert!(rel_l2 < 0.01, "{name}: rel_l2 {rel_l2} too high");
        };
        check("K", &xk, &pk8, &scales_k);
        check("V", &xv, &pv8, &scales_v);
    }

    /// A config with `head_dim=6` -- even (RoPE-legal) but not a multiple of
    /// 4, so a packed int8 `u32` would span two heads' scales. Used only by
    /// the G5 boundary-policy tests below.
    fn head_dim_not_multiple_of_4_cfg() -> QwenConfig {
        QwenConfig {
            vocab: 11,
            block_size: 16,
            n_layers: 1,
            d_model: 12,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 6,
            d_ff: 8,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 16,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// G5a: [`kv_int8_supported`] must say `false` exactly for `head_dim % 4
    /// != 0` configs, and `true` for every real shape this repo ships.
    #[test]
    fn kv_int8_supported_matches_head_dim_mod_4() {
        assert!(!kv_int8_supported(&head_dim_not_multiple_of_4_cfg()), "head_dim=6 must not be int8-supported");
        assert!(kv_int8_supported(&QwenConfig::tiny()), "tiny()'s head_dim=8 must be int8-supported");
        assert!(kv_int8_supported(&QwenConfig::qwen3_0_6b()), "the real Qwen3-0.6B head_dim=128 must be int8-supported");
    }

    /// G5b: an EXPLICIT `kv_int8: true` request on an unsupported config must
    /// fail loudly (a caller that asked for int8 by name should hear about a
    /// mismatch, not get a silent fp32 substitution) -- see
    /// [`kv_int8_supported`]'s doc comment for why this differs from what a
    /// DEFAULT-selecting caller should do.
    #[test]
    #[should_panic(expected = "head_dim % 4")]
    fn explicit_int8_kv_request_panics_on_an_unsupported_head_dim() {
        let cfg = head_dim_not_multiple_of_4_cfg();
        let map = tiny_weights(&cfg);
        let _ = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 16, 1, 4, 8, true, false);
    }

    /// G1: the KV pool byte count at REAL Qwen3-0.6B serving defaults (`ctx=2048`
    /// -> `block_size=16`, `num_blocks=272`), pinned exactly -- pure arithmetic,
    /// no device, no weights. The int8/fp32 ratio is `4*head_dim/(head_dim+4)`,
    /// which is why `tiny()`'s ratio (`head_dim=8`) is a DIFFERENT number from
    /// the real one (`head_dim=128`) -- a toy config cannot stand in for it.
    #[test]
    fn kv_pool_bytes_identity_holds_at_the_real_shape() {
        let real = QwenConfig::qwen3_0_6b();
        let (bs, nb) = (16u32, 2048u32.div_ceil(16) * 2 + 16); // = 272
        assert_eq!(nb, 272);
        assert_eq!(kv_pool_bytes(&real, bs, nb, false), 998_244_352);
        assert_eq!(kv_pool_bytes(&real, bs, nb, true), 257_359_872);

        let ratio = |c: &QwenConfig| kv_pool_bytes(c, bs, nb, false) as f64 / kv_pool_bytes(c, bs, nb, true) as f64;
        assert!((ratio(&real) - 4.0 * 128.0 / 132.0).abs() < 1e-9, "real head_dim=128 ratio must be 3.8788...");
        assert!((ratio(&QwenConfig::tiny()) - 4.0 * 8.0 / 12.0).abs() < 1e-9, "tiny head_dim=8 ratio must be 2.6667... -- NOT the same number as the real shape");
    }

    /// M2.4's own memory-budget gate: `Scratch::{scores,probs}` (`docs`'s own
    /// audit finding, "the single largest serving scratch buffer") shrinks
    /// once the fused causal-chunk-prefill path (`paged_flash_prefill`)
    /// replaces the triad for an fp32-KV engine - decode's own worst case
    /// (`max_batch * n_heads * cap`) is the ONLY term left, dropping the
    /// `max_prefill^2 * n_heads` `[nh,N,N]` term the old, unconditional
    /// `max(...)` formula always paid. Pinned exact numbers at round values
    /// (not the real Qwen3-0.6B shape - `kv_pool_bytes_identity_holds_at_the_
    /// real_shape` already covers pinning against that; this test's own job
    /// is the REDUCTION shape, which round numbers make easy to verify by
    /// hand): `max_batch=128, max_prefill=512, n_heads=16, cap=2048`.
    ///
    /// An int8-KV engine gets NO reduction - documented, not assumed:
    /// `paged_flash_prefill` has no int8-KV tier yet, so a `kv_int8` engine
    /// still runs causal-chunk prefill through the triad and needs the SAME
    /// scratch this milestone shrinks for fp32.
    #[test]
    fn paged_attn_scratch_shrinks_once_the_fused_prefill_path_replaces_the_triad() {
        let mut cfg = QwenConfig::tiny();
        cfg.n_heads = 16;
        let (max_batch, max_prefill, cap) = (128u32, 512u32, 2048u32);

        // The OLD, unconditional formula every prior milestone shipped
        // (`b = max_batch.max(max_prefill)`, `max(decode, causal-chunk)`) -
        // inlined here as the regression floor, not re-derived from the
        // current (shrunk) implementation.
        let b = max_batch.max(max_prefill) as u64;
        let nh = cfg.n_heads as u64;
        let old_words = (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh);
        let old_bytes = old_words * 2 * 4;
        assert_eq!(old_bytes, 134_217_728, "sanity: the old formula's own value at this shape");

        let new_fp32 = paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, true);
        assert_eq!(new_fp32, 33_554_432, "fused prefill available: decode's own worst case only");
        assert!(new_fp32 < old_bytes, "the shrunk size must be smaller than the old unconditional formula");
        assert_eq!(old_bytes / new_fp32, 4, "exactly 4x at this shape (matches max_prefill^2/(max_batch*cap))");

        let new_unavailable = paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, false);
        assert_eq!(new_unavailable, old_bytes, "fused prefill unavailable (int8 KV, or no workgroup_reductions): no reduction");
    }

    /// [`paged_attn_scratch_bytes`] (pure arithmetic) must match what
    /// [`Engine::paged_attn_scratch_bytes`] recorded at construction, the same
    /// identity `kv_pool_bytes_identity_holds_at_the_real_shape` pins for the
    /// KV pool. `fused_prefill_available` is derived through the SAME
    /// `Op::PagedAttentionFused` selector call `Engine::from_map_with_gpu`
    /// itself makes, not assumed from `kv_int8` alone - on a device without
    /// `caps.workgroup_reductions` this would need to stay `false` even at
    /// fp32 KV, which `paged_attn_scratch_shrinks_only_when_fused_prefill_is_
    /// actually_reachable` below pins directly.
    #[test]
    fn engine_paged_attn_scratch_bytes_matches_the_free_function() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        for kv_int8 in [false, true] {
            let (bs, num_blocks, max_batch, mbt, max_prefill) = (4u32, 64u32, 4u32, 8u32, 16u32);
            let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, num_blocks, max_batch, mbt, max_prefill, kv_int8, false);
            let cap = mbt * bs;
            let dtype = if kv_int8 { Dtype::I8 } else { Dtype::F32 };
            let fused_prefill_available = DefaultSelector.select(Op::PagedAttentionFused, OpShape { m: 0, n: 0, k: 1, dtype }, &eng.gpu().caps()) == KernelVariant::FusedFlash;
            assert_eq!(
                eng.paged_attn_scratch_bytes(),
                paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, fused_prefill_available),
                "kv_int8={kv_int8}"
            );
        }
    }

    /// The capability gate itself (M2.4's own correctness fix, caught before
    /// this was ever wired to a real device): a device WITHOUT `caps.
    /// workgroup_reductions` (the CPU JIT's own caps - `FusedFlash` requires
    /// it, same as `WorkgroupPerOutput`) must NOT shrink the scratch even at
    /// fp32 KV, because `run_batched_steps`'s own dispatch falls back to the
    /// triad there and the triad needs the full, unshrunk size. Pure
    /// arithmetic against a hand-built `DeviceCaps` - no CPU backend needed.
    #[test]
    fn paged_attn_scratch_shrinks_only_when_fused_prefill_is_actually_reachable() {
        let cfg = QwenConfig::tiny();
        let (max_batch, max_prefill, cap) = (4u32, 16u32, 32u32);
        let mut cpu_like = gpu_core::DeviceCaps::portable_baseline(gpu_core::DeviceClass::Cpu);
        cpu_like.workgroup_reductions = false;

        let available = DefaultSelector.select(Op::PagedAttentionFused, OpShape { m: 0, n: 0, k: 1, dtype: Dtype::F32 }, &cpu_like) == KernelVariant::FusedFlash;
        assert!(!available, "sanity: a device without workgroup_reductions never gets FusedFlash");

        let unshrunk = paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, available);
        let old_formula = {
            let nh = cfg.n_heads as u64;
            let b = max_batch.max(max_prefill) as u64;
            (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh) * 2 * 4
        };
        assert_eq!(unshrunk, old_formula, "fp32 KV on a device without workgroup_reductions must keep the full size");
    }

    /// Single-sequence paged/batched serving must match the reference contiguous
    /// KV generation (`Qwen::generate_kv`) token-for-token, and a two-sequence
    /// batch must equal running each prompt on its own - proving batched paged
    /// decode is exact. G4: at BOTH KV dtypes, not just fp32 - the reference is
    /// always fp32 (`Qwen::generate_kv` has no paging or quantization at all),
    /// so the int8 arm is asking whether quantization noise ever flips an
    /// argmax; kept `assert_eq!` deliberately, since a flip here on real
    /// (non-degenerate) logit gaps would itself be worth knowing about.
    #[test]
    fn batched_serving_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let bs = 4;
        let (num_blocks, max_batch, mbt) = (64u32, 4u32, 8u32);

        // Reference: the committed single-sequence KV generation.
        let model = Qwen::new(cfg.clone(), 1, 64, &map);
        let p0 = vec![1u32, 5, 3, 9];
        let p1 = vec![7u32, 2, 4];
        let mut r0 = Rng::new(0);
        let mut r1 = Rng::new(0);
        let ref0 = crate::sample::generate_kv(&model, &p0, 12, 0.0, 0, 1.0, None, &mut r0);
        let ref1 = crate::sample::generate_kv(&model, &p1, 12, 0.0, 0, 1.0, None, &mut r1);

        for kv_int8 in [false, true] {
            // Engine: run both prompts concurrently (batched paged).
            let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, num_blocks, max_batch, mbt, 32, kv_int8, false);
            let out = eng.generate_greedy(&[p0.clone(), p1.clone()], 12, None);

            assert_eq!(out[0], ref0, "kv_int8={kv_int8}: seq0 batched paged != reference");
            assert_eq!(out[1], ref1, "kv_int8={kv_int8}: seq1 batched paged != reference");
        }
    }

    /// M6.3: the served decode tape (uniform buffers + bind groups
    /// `run_batched_steps` records per dispatch) is built once per `bsz`
    /// bucket and REUSED for every later decode step at that same bucket,
    /// instead of a fresh uniform + bind group per dispatch every step.
    /// `crates/backend-{wgpu,vulkan,cpu}`'s `bind_groups` stat is exactly the
    /// per-dispatch churn this targets - RED before the cache existed (a
    /// second decode step at the same `bsz` always added the tape's full
    /// dispatch count in fresh bind groups); GREEN after (it adds zero).
    ///
    /// Byte-identical decode correctness against an INDEPENDENT reference
    /// (`crate::sample::generate_kv`, untouched by this change) is already
    /// pinned by `batched_serving_matches_reference`/`warm_prefill_is_
    /// identical_to_cold` just above/below - both decode several steps at a
    /// stable `bsz`, so both already exercise a cache HIT, not just the
    /// first MISS. This test pins the MECHANISM directly.
    #[test]
    fn decode_step_at_a_stable_bucket_reuses_the_cached_tapes_bind_groups() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 4, 8, 16, false, false);

        let mut t0 = BlockTable::new();
        let mut t1 = BlockTable::new();
        let h0 = eng.prefill(&mut t0, &[1u32, 5, 3]);
        let h1 = eng.prefill(&mut t1, &[7u32, 2, 4]);
        let tok0 = Engine::argmax(&eng.logits(&h0));
        let tok1 = Engine::argmax(&eng.logits(&h1));

        // Step 1 at bsz=2: cache miss - builds and records the tape.
        let bg_before = eng.device_stats().expect("every backend this crate targets reports bind_groups").bind_groups;
        let out1 = eng.forward_batched(&mut [&mut t0, &mut t1], &[tok0, tok1]);
        let bg_after_miss = eng.device_stats().unwrap().bind_groups;
        assert!(bg_after_miss > bg_before, "the first decode step at a new bucket must still build (and record) a tape");

        // `Self::logits` is a SEPARATE, uncached host-argmax dispatch (not
        // this milestone's decode tape) - computing next-step tokens through
        // it between the two decode steps must not count against the tape
        // cache, so the cache assertion below is taken immediately before/
        // after ONLY the second `forward_batched` call.
        let d = cfg.d_model as usize;
        let tok0b = Engine::argmax(&eng.logits(&out1[..d]));
        let tok1b = Engine::argmax(&eng.logits(&out1[d..2 * d]));

        // Step 2 at the SAME bsz=2 with DIFFERENT token/position inputs
        // (positions advanced, new tokens): cache hit.
        let bg_before_hit = eng.device_stats().unwrap().bind_groups;
        let _out2 = eng.forward_batched(&mut [&mut t0, &mut t1], &[tok0b, tok1b]);
        let bg_after_hit = eng.device_stats().unwrap().bind_groups;
        assert_eq!(
            bg_after_hit, bg_before_hit,
            "a decode step at an already-cached bucket must create ZERO new bind groups"
        );
    }

    /// THE prefix-cache invariant: a warm prefill (served from cached blocks)
    /// must produce output IDENTICAL to the cold one - a cache hit that
    /// changes a single token is corruption, not a cache. Also pins that the
    /// cache actually engaged (a test that silently measured two cold runs
    /// would prove nothing). G4: at BOTH KV dtypes - this is the load-bearing
    /// proof that `PrefixCache` block sharing works for int8 KV, not just by
    /// accident of the pool and scales sharing the same `slot` indexing (see
    /// the comment on [`Engine`]'s `scales_k`/`scales_v` fields): if a shared
    /// block's scales were ever addressed differently from its pool words,
    /// THIS is where it would show up, at bit-exact `assert_eq!`.
    #[test]
    fn warm_prefill_is_identical_to_cold() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        for kv_int8 in [false, true] {
            let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 2, 12, 16, kv_int8, false);
            let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
            let cold = eng.generate_greedy(std::slice::from_ref(&prompt), 10, None);
            let (hit0, _, cached) = eng.prefix_stats();
            assert_eq!(hit0, 0, "kv_int8={kv_int8}: first prefill must be cold");
            assert!(cached > 0, "kv_int8={kv_int8}: full prompt blocks must be indexed after prefill");
            let warm = eng.generate_greedy(std::slice::from_ref(&prompt), 10, None);
            let (hit1, _, _) = eng.prefix_stats();
            assert!(hit1 > 0, "kv_int8={kv_int8}: the second prefill must actually reuse the prefix");
            assert_eq!(warm, cold, "kv_int8={kv_int8}: a cache hit must be byte-identical to computing the prefix");
        }
    }

    /// `calibrate_kv` must report one stream per (layer, K|V, kv_head) with a
    /// sane, non-degenerate absmax/p99.99, and must refuse an int8-KV engine
    /// (calibration wants the pre-quantization distribution).
    #[test]
    fn calibrate_kv_reports_one_stream_per_layer_kv_head() {
        let cfg = QwenConfig::tiny(); // n_layers=2, n_kv_heads=2, head_dim=8
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 4, 8, 16, false, false);
        let prompts = vec![vec![1u32, 5, 3, 9, 2], vec![7u32, 2, 4]];
        let report = eng.calibrate_kv(&prompts).report();

        let expected = cfg.n_layers as usize * cfg.n_kv_heads as usize * 2; // K + V
        assert_eq!(report.len(), expected, "one stream per (layer, K|V, kv_head)");
        for r in &report {
            assert!(r.absmax > 0.0, "{}: absmax must be nonzero for real (non-degenerate) weights", r.name);
            assert!(r.outlier_ratio.is_finite() && r.outlier_ratio >= 1.0 - 1e-6, "{}: ratio {}", r.name, r.outlier_ratio);
        }
        // Spot-check the naming convention a report/CLI consumer depends on.
        assert!(report.iter().any(|r| r.name == "layer00.k.head0"));
        assert!(report.iter().any(|r| r.name == "layer01.v.head1"));
    }

    #[test]
    #[should_panic(expected = "fp32-KV engine")]
    fn calibrate_kv_refuses_an_int8_kv_engine() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 4, 8, 16, true, false);
        eng.calibrate_kv(&[vec![1u32, 2, 3]]);
    }

    /// Random shared prefixes: a prompt sharing a random-length prefix with
    /// earlier traffic must prefill (through adopted cached blocks) to the
    /// same final hidden state a fresh engine computes - within rounding.
    ///
    /// Deliberately NOT a token-equality test: reused KV is bit-identical to
    /// its original computation, but the CPU backend's blocked GEMMs are not
    /// row-count-invariant in final-bit rounding, so a tail-only prefill can
    /// differ from a full one by an ulp - which flips argmax on a degenerate
    /// random model while meaning nothing. Structural corruption (a wrongly
    /// adopted block) produces O(1) relative error; rounding produces ~1e-6.
    /// The 1e-3 gate separates them cleanly. Token-level identity is pinned by
    /// `warm_prefill_is_identical_to_cold` where chunking is identical.
    /// G4: at BOTH KV dtypes - the `rel < 1e-3` tolerance here is deliberately
    /// NOT bit-exact (see the doc comment above), for reasons unrelated to
    /// `kv_int8`, so this test keeps its existing tolerance under int8 too
    /// rather than tightening it to `assert_eq!`.
    #[test]
    fn random_shared_prefixes_stay_exact() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        for kv_int8 in [false, true] {
            let mut cached_eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 128, 2, 12, 16, kv_int8, false);
            let mut rng = Rng::new(42);
            let vocab = cfg.vocab as u64;
            let base: Vec<u32> = (0..14).map(|_| (rng.next_u64() % vocab) as u32).collect();
            for trial in 0..6 {
                let keep = (rng.next_u64() as usize) % base.len();
                let mut prompt = base[..keep].to_vec();
                let extra = 3 + (rng.next_u64() as usize) % 6;
                prompt.extend((0..extra).map(|_| (rng.next_u64() % vocab) as u32));
                // Reference: a fresh engine has an empty cache by construction.
                let mut fresh = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 128, 2, 12, 16, kv_int8, false);
                let mut tf = BlockTable::new();
                let cold = fresh.prefill(&mut tf, &prompt);
                let mut tc = BlockTable::new();
                let warm = cached_eng.prefill(&mut tc, &prompt);
                let err: f32 = warm.iter().zip(&cold).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                let norm: f32 = cold.iter().map(|v| v * v).sum::<f32>().sqrt();
                let rel = err / norm.max(1e-12);
                assert!(
                    rel < 1e-3,
                    "kv_int8={kv_int8} trial {trial}: warm prefill diverged structurally (rel L2 {rel:.6}) on prompt {prompt:?}"
                );
                cached_eng.release_table(&mut tc);
            }
            let (hit, looked, _) = cached_eng.prefix_stats();
            assert!(hit > 0, "kv_int8={kv_int8}: at least one trial must have actually reused a prefix ({hit}/{looked})");
        }
    }

    /// Int8 weights (A0) must stay numerically faithful to the fp32 engine:
    /// same weights, same prompt, and the final-norm hidden state after a
    /// quantized prefill must sit within a few percent of the fp32 one. A
    /// scale-handling bug (the realistic failure mode) produces error of the
    /// same order as the signal, so the gate below separates cleanly while
    /// tolerating honest quant noise.
    /// The stream-level greedy-agreement threshold lives in `brain perf`'s
    /// fidelity gate, which measures it on real checkpoints.
    #[test]
    fn int8_weights_track_fp32() {
        // `tiny_i8`, not `tiny`: every quantized `k` must be a whole
        // `model::int8::GROUP` (see `QwenConfig::tiny_i8`'s own doc).
        let cfg = QwenConfig::tiny_i8();
        let map = tiny_weights(&cfg);
        let mut eng8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 2, 8, 32, false, true);
        if !eng8.weights_int8() {
            // Capability-gated fallback (CPU JIT): the engine must run fp32
            // and say so. A device whose caps DO report the packed-int8 path
            // must never land here - a silent fallback on capable hardware is
            // exactly the regression this branch once masked.
            assert!(
                !eng8.gpu().caps().numeric.int8_dot,
                "device reports int8_dot but the engine fell back to fp32"
            );
            brain_testutil::skip_unavailable("int8 comparison: device has no packed-int8 path");
            return;
        }
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 8, 32, false, false);
        let prompt = vec![1u32, 5, 3, 9, 2, 7];
        let mut t8 = BlockTable::new();
        let h8 = eng8.prefill(&mut t8, &prompt);
        let mut tf = BlockTable::new();
        let hf = eng.prefill(&mut tf, &prompt);
        let dot_err: f32 = h8.iter().zip(&hf).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let norm: f32 = hf.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = dot_err / norm.max(1e-12);
        assert!(rel < 0.10, "int8 hidden state diverged from fp32: relative L2 {rel:.4}");
        // And the engine still decodes end-to-end on the int8 path.
        let out = eng8.generate_greedy(&[prompt], 8, None);
        assert_eq!(out[0].len(), 8, "int8 engine must produce the requested tokens");
    }

    /// A request too long for the engine must be REJECTED, not crash the process
    /// and not sit in the queue forever. Before the capacity check, a prompt
    /// longer than `max_blocks_per_seq * block_size` wrote past its row of the
    /// block table.
    #[test]
    fn oversized_requests_are_rejected_not_fatal() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        // capacity = max_blocks_per_seq(4) * block_size(4) = 16 tokens.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 4, 8, false, false);
        assert_eq!(eng.max_seq_len(), 16);
        let mut sched = Scheduler::new(eng, 2);

        let huge = sched.submit(Request { prompt: vec![1u32; 64], max_new: 4, eos: None });
        let ok = sched.submit(Request { prompt: vec![2u32, 3, 4], max_new: 4, eos: None });

        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1, "the oversized request must be refused");
        assert_eq!(rep.rejected[0].0, huge);
        assert!(matches!(rep.rejected[0].1, RejectReason::ExceedsCapacity { .. }));

        // The queue keeps moving and the viable request completes. With the
        // decode window a short request can finish inside the SAME iteration
        // that admitted it, so its tokens arrive in that report's `completed`.
        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&ok));
        assert!(!out.contains_key(&huge));
    }

    /// An out-of-vocab token must be REJECTED at admission with a typed
    /// reason. Admitting it would make the embedding gather read out of
    /// bounds - the kernels are trusted, so the failure would be silent
    /// garbage in the hidden states (found the hard way: NaN on CPU, wrong
    /// finite values on GPU), not an error anyone can see.
    #[test]
    fn out_of_vocab_tokens_are_rejected_not_gathered() {
        let cfg = QwenConfig::tiny();
        let vocab = cfg.vocab;
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 8, 8, false, false);
        let mut sched = Scheduler::new(eng, 2);
        let bad = sched.submit(Request { prompt: vec![1, vocab + 7, 2], max_new: 4, eos: None });
        let ok = sched.submit(Request { prompt: vec![1, 2, 3], max_new: 4, eos: None });
        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1);
        assert_eq!(rep.rejected[0].0, bad);
        assert!(matches!(rep.rejected[0].1, RejectReason::InvalidToken { token, .. } if token == vocab + 7));
        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&ok), "valid work behind the bad request still completes");
        assert!(!out.contains_key(&bad));
    }

    /// Admission policies must refuse at submit time, report the refusal, and
    /// leave already-queued work untouched.
    #[test]
    fn admission_policy_rejects_and_reports() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 2, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 2);
        sched.set_admission(Box::new(MaxQueueDepth(1)));

        let a = sched.submit(Request { prompt: vec![1, 5, 3], max_new: 4, eos: None }); // queued (0 ahead)
        let b = sched.submit(Request { prompt: vec![2, 6, 4], max_new: 4, eos: None }); // queued (1 ahead? depth=1 => 1 not < 1 => REJECTED)
        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1, "the over-depth submit must be refused");
        assert_eq!(rep.rejected[0].0, b);
        assert!(matches!(rep.rejected[0].1, RejectReason::PolicyRejected { policy: "max_queue_depth" }));

        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&a), "admitted work completes normally");
        assert!(!out.contains_key(&b));
    }

    /// DeadlineAware admits until a service time is measured, then refuses work
    /// that provably cannot start in time.
    #[test]
    fn deadline_aware_uses_measured_service_time() {
        let p = DeadlineAware { deadline_ms: 100.0 };
        let mk = |queued, svc| QueueState {
            queued_ahead: queued,
            running: 0,
            free_blocks: 99,
            mean_service_ms: svc,
        };
        let r = Request { prompt: vec![1], max_new: 1, eos: None };
        assert!(p.admit(&r, &mk(50, None)), "no measurement -> cannot prove lateness");
        assert!(p.admit(&r, &mk(4, Some(20.0))), "4 queued at the mocked service time fits the deadline");
        assert!(!p.admit(&r, &mk(6, Some(20.0))), "6 queued at the mocked service time provably misses it");
    }

    /// Cancelling must free the sequence's KV blocks immediately and stop its
    /// decoding, without disturbing the requests running alongside it.
    #[test]
    fn cancel_reclaims_blocks_and_spares_neighbours() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 3);

        // Everything long enough to still be decoding after two (windowed)
        // iterations, and inside the per-sequence capacity (12 blocks x 4 = 48).
        let keep_a = sched.submit(Request { prompt: vec![1u32, 5, 3], max_new: 20, eos: None });
        let doomed = sched.submit(Request { prompt: vec![7u32, 2, 9], max_new: 30, eos: None });
        let keep_b = sched.submit(Request { prompt: vec![4u32, 4, 1], max_new: 20, eos: None });

        // Admit everything and decode a couple of steps.
        sched.step();
        sched.step();
        assert_eq!(sched.running_len(), 3);
        let free_before = sched.free_blocks();

        let produced = sched.cancel(doomed).expect("cancelling a running request must succeed");
        assert!(!produced.is_empty(), "it had already produced tokens");
        assert_eq!(sched.running_len(), 2, "only the cancelled request is removed");
        assert!(sched.free_blocks() > free_before, "its KV blocks must return to the pool at once");

        // The survivors still finish normally.
        let out = sched.run();
        assert_eq!(out.len(), 2);
        assert_eq!(out[&keep_a].len(), 20);
        assert_eq!(out[&keep_b].len(), 20);
        assert!(!out.contains_key(&doomed), "a cancelled request must not complete");

        // Cancelling twice, or an unknown id, is a no-op rather than a panic.
        assert!(sched.cancel(doomed).is_none());
        assert!(sched.cancel(9999).is_none());
    }

    /// Cancelling a request that was never admitted just drops it from the queue.
    #[test]
    fn cancel_before_admission_produces_nothing() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        // One slot, so the second request cannot be admitted.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 32, 1, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 1);
        let _a = sched.submit(Request { prompt: vec![1u32, 5, 3], max_new: 4, eos: None });
        let queued = sched.submit(Request { prompt: vec![2u32, 6, 4], max_new: 4, eos: None });
        sched.step();
        assert_eq!(sched.waiting_len(), 1);
        assert_eq!(sched.cancel(queued), Some(Vec::new()));
        assert_eq!(sched.waiting_len(), 0);
    }

    /// The batched split/two-stage argmax reduction (`greedy_from_hidden`,
    /// reading the whole batch's rows from `sc.xn_final` at once) must select
    /// exactly the token a plain host linear-scan argmax over the SAME
    /// per-row device head matmul (`logits`, one row at a time) would. This
    /// is the invariant that lets decode skip shipping a `[batch, vocab]`
    /// logit block back to the host - if it ever drifted, the engine would
    /// silently generate different text at speed.
    ///
    /// `dev` is read BEFORE the per-row `logits` calls: both share the
    /// `sc.xn_final` scratch buffer, and `logits` overwrites its row 0 with
    /// one hidden row at a time - reading `dev` first (over the batch
    /// `forward_batched` just left resident) avoids that aliasing entirely.
    #[test]
    fn device_head_argmax_matches_the_host_head() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 4, 12, 8, false, false);

        // Drive a few real decode steps so the hidden states are genuine.
        let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
        let prompts = [vec![1u32, 5, 3], vec![7u32, 2, 9], vec![4u32, 4, 1]];
        let mut inputs = Vec::new();
        for (t, p) in tables.iter_mut().zip(prompts.iter()) {
            let h = eng.prefill(t, p);
            inputs.push(Engine::argmax(&eng.logits(&h)));
        }
        for _ in 0..4 {
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                eng.forward_batched(&mut refs, &inputs)
            };
            // Device: same hidden still resident in sc.xn_final, head + split
            // argmax applied on device, read BEFORE the per-row calls below.
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            // Host reference: per row, the SAME device head matmul (`logits`),
            // reduced by a plain host linear-scan argmax instead.
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            assert_eq!(dev, host, "device head picked a different token than the host head");
            inputs = host;
        }
    }

    /// M3.2: `admit_greedy`/`admit_topk` (what `PagedDecoder::admit_greedy`/
    /// `admit_topk` call for the SCHEDULER's admission path) must agree with
    /// a TRUE, independent host matvec against the same head weight - not
    /// merely with another device computation, since a real defect (a
    /// transposed head, a swapped `d_model`/`vocab`) would agree with itself
    /// perfectly. The device path is a tiled GEMM and the reference here is
    /// a scalar host dot product, so their reduction orders genuinely
    /// differ; comparing the value AT each returned candidate's own index
    /// (rather than requiring the two to rank a near-tie the same way) is
    /// what "within reduction-order tolerance" means for this gate.
    #[test]
    fn admission_head_matches_a_true_host_matvec_within_tolerance() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 4, 12, 8, false, false);

        let head = map.get(cfg.head_weight()).cloned().unwrap_or_else(|| map.get("tok.weight").cloned().unwrap());
        let (d, v) = (cfg.d_model as usize, cfg.vocab as usize);

        let mut table = BlockTable::new();
        let hidden = eng.prefill(&mut table, &[1u32, 5, 3]);
        assert_eq!(hidden.len(), d);

        // Never routed through `Engine::logits` (device-backed since M3.2) -
        // a plain scalar dot product against the raw weight this engine
        // separately uploaded to the device at construction.
        let host_logits = model::hostmath::matvec(&head, &hidden, v, d);
        let tol = |x: f32| 1e-3 * x.abs().max(1.0) + 1e-4;

        let k = 16u32;
        let candidates = eng.admit_topk(&hidden, k);
        assert_eq!(candidates.len(), k as usize);
        for &(idx, dev_val) in &candidates {
            let host_val = host_logits[idx as usize];
            assert!(
                (dev_val - host_val).abs() <= tol(host_val),
                "admit_topk candidate {idx}: device {dev_val} vs true host matvec {host_val}"
            );
        }

        let greedy = eng.admit_greedy(&hidden);
        let greedy_dev_val = candidates[0].1;
        let greedy_host_val = host_logits[greedy as usize];
        assert_eq!(greedy, candidates[0].0, "admit_greedy must pick admit_topk's own best candidate");
        assert!(
            (greedy_dev_val - greedy_host_val).abs() <= tol(greedy_host_val),
            "admit_greedy token {greedy}: device {greedy_dev_val} vs true host matvec {greedy_host_val}"
        );
    }

    /// M4.1: the fused `attn.wqkv.weight`/`mlp.gateup.weight` GEMMs
    /// (`run_batched_steps`) must be BIT-IDENTICAL to the split path they
    /// replace - not merely close. Comparing against a HOST matmul would
    /// prove nothing here (`admission_head_matches_a_true_host_matvec_within_
    /// tolerance`'s own doc: a tiled device GEMM and a scalar host loop
    /// genuinely reduce in different orders), so the reference is the split
    /// path itself, run through the SAME device kernel (`Engine::mm`) this
    /// engine would have dispatched before M4.1 - three/two independent
    /// GEMMs against the three/two ORIGINAL (unconcatenated) weight
    /// matrices, over the exact `xn1`/`xn2` activations the fused dispatch
    /// itself produced for a real prefill. Per-output-column GEMV is
    /// independent of how many other columns share the dispatch, so this
    /// must hold exactly, no tolerance.
    #[test]
    fn fused_qkv_and_gateup_are_bit_identical_to_split() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 32, 1, 12, 8, false, false);

        let mut table = BlockTable::new();
        let prompt = [1u32, 5, 3];
        let _ = eng.prefill(&mut table, &prompt);
        let rows = prompt.len() as u32;

        // `sc.xn1`/`sc.xn2` etc. are reused every layer, so after `prefill`
        // they hold the LAST layer's activations - reference the fused
        // weights of that same layer.
        let l = cfg.n_layers as usize - 1;
        let (d, hq, hkv, ff) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim(), cfg.d_ff);
        let g = &eng.gpu;

        let split_matmul = |x: &DeviceBuffer, weight_name: &str, n: u32| -> Vec<f32> {
            let w = g.storage_init("t_split_w", &map[&format!("blocks.{l}.{weight_name}")]);
            let out = g.storage((rows * n) as u64);
            let step = eng.mm(x, &w, &out, rows, d, n);
            g.submit(&[], &[step]);
            g.poll_wait();
            g.read(&out, (rows * n) as usize)
        };

        let fused_q = g.read(&eng.sc.q_pre, (rows * hq) as usize);
        let fused_k = g.read(&eng.sc.k_pre, (rows * hkv) as usize);
        let fused_v = g.read(&eng.sc.v, (rows * hkv) as usize);
        assert_eq!(fused_q, split_matmul(&eng.sc.xn1, "attn.wq.weight", hq), "fused Q must be bit-identical to a split wq GEMM");
        assert_eq!(fused_k, split_matmul(&eng.sc.xn1, "attn.wk.weight", hkv), "fused K must be bit-identical to a split wk GEMM");
        assert_eq!(fused_v, split_matmul(&eng.sc.xn1, "attn.wv.weight", hkv), "fused V must be bit-identical to a split wv GEMM");

        let fused_gate = g.read(&eng.sc.gate_pre, (rows * ff) as usize);
        let fused_up = g.read(&eng.sc.up, (rows * ff) as usize);
        assert_eq!(fused_gate, split_matmul(&eng.sc.xn2, "mlp.gate.weight", ff), "fused gate must be bit-identical to a split mlp.gate GEMM");
        assert_eq!(fused_up, split_matmul(&eng.sc.xn2, "mlp.up.weight", ff), "fused up must be bit-identical to a split mlp.up GEMM");
    }

    /// RED before `Self::fits_one_binding`/`Self::mm_tiled_into` existed:
    /// `Self::mm_into` dispatched EVERY fp32 GEMM (including the LM head,
    /// `Self::head_steps`'s `n = vocab`) through `Self::mm`, which binds `w`
    /// WHOLE - a plain `gpu.step`, no `step_sliced`. wgpu clamps
    /// `max_storage_buffer_binding_size` to `i32::MAX` (2047 MiB) on every
    /// backend regardless of a card's actual VRAM, so a `[n, k]` weight past
    /// that panics in `Device::create_bind_group` before a single logit is
    /// computed - independent of `--weights-int8` (which quantizes the head
    /// down to ~1/4 size and usually sidesteps this, but the fp32 tier this
    /// device falls back to without that flag, or on a device with no
    /// packed-int8 path, does not).
    ///
    /// This is deliberately NOT routed through `Engine::prefill`/a real
    /// vocab: this tree's `Self::batched_tape` still dispatches the
    /// token-embedding gather (`tok.weight`) as one untiled binding too (a
    /// separate, already-landed fix elsewhere), so a vocab past the cap would
    /// panic THERE first and never reach the head at all. Calling
    /// `Self::mm_into` directly - exactly [`fused_qkv_and_gateup_are_bit_
    /// identical_to_split`]'s own pattern of dispatching `Self::mm`/`Self::
    /// mm_into` against a hand-built weight buffer - isolates the head/logits
    /// binding-size gap from that unrelated one.
    #[test]
    fn head_matmul_over_binding_cap_does_not_panic() {
        // `head_matmul_tiled_matches_untiled_within_tolerance` holds this same
        // lock while it mutates the process-global `BRAIN_TILE_BUDGET_WORDS`
        // env var. Cargo runs `#[test]`s on parallel threads in one process,
        // and without this guard the two can race: if this test's dispatch
        // happens to run while the other has forced the tile budget down to
        // 128 words, `block::vocab_tiles_on` would slice THIS test's ~8.4M-row
        // synthetic weight into millions of separate `MATMUL_TILE` dispatches
        // instead of the handful the real (unmodified) budget produces -
        // measured: minutes of 99% CPU and tens of GB RSS building `Step`s,
        // not a correctness failure, but indistinguishable from a hang.
        let _guard = brain_testutil::env_lock();
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 32, 1, 12, 8, false, false);
        let g = &eng.gpu;

        // A synthetic [n, k] fp32 weight sized to exceed THIS device's real
        // queried binding cap - the same class of shape Qwen3-8B's real
        // untied lm_head (151936 x 4096, ~2.32 GiB) presents on a P40 (2047
        // MiB). Zero-initialized (`gpu.storage`, no host upload): the values
        // are irrelevant to whether the dispatch panics.
        let k = 64u32;
        let budget = g.max_storage_binding_bytes();
        let n = (budget / (k as u64 * 4)) as u32 + 4096;
        let bytes = n as u64 * k as u64 * 4;
        assert!(bytes > budget, "test shape must itself exceed the binding cap ({bytes} vs {budget})");

        let x = g.storage_init("t_head_x", &vec![0.01f32; k as usize]);
        let w = g.storage(n as u64 * k as u64);
        let out = g.storage(n as u64);

        let mut steps = Vec::new();
        eng.mm_into(&mut steps, &x, &w, &out, 1, k, n);
        g.submit(&[], &steps);
        g.poll_wait();
    }

    /// Correctness gate: `Self::mm_tiled_into` (several `MATMUL_TILE`
    /// dispatches, each binding a weight-row SUB-RANGE, writing the OUTPUT'S
    /// strided column slice via `n_off`/`n_full`) must agree with `Self::mm`
    /// (the single-dispatch production kernel `Self::mm_into` used for every
    /// GEMM before this weight's `[n, k]` size could exceed the binding cap)
    /// within GPU reduction-order tolerance - not bitwise, since the two
    /// dispatch different physical kernels (`Self::mm` may pick the
    /// register-tiled/GEMV fast kernel; `Self::mm_tiled_into` always uses the
    /// naive per-element `matmul_tile`), and floating-point addition is not
    /// associative across different summation orders. Same tolerance
    /// [`admission_head_matches_a_true_host_matvec_within_tolerance`] already
    /// uses for exactly this reason.
    ///
    /// `n`/`k` stay tiny (well under the real binding cap) so the fast path
    /// would ordinarily never tile at all; `BRAIN_TILE_BUDGET_WORDS` forces
    /// [`Self::mm_tiled_into`]'s own tile split (via `block::vocab_tiles_on`)
    /// down to a few rows per tile, so this exercises MULTIPLE dispatches and
    /// checks output columns at tile boundaries, not just the trivial
    /// one-tile case.
    #[test]
    fn head_matmul_tiled_matches_untiled_within_tolerance() {
        let _guard = brain_testutil::env_lock();
        let prev = std::env::var("BRAIN_TILE_BUDGET_WORDS").ok();
        // k=8 words/row -> a 128-word budget is 16 rows/tile; n=40 columns
        // splits into tiles [0,16) [16,32) [32,40) - two full boundaries and
        // a ragged last tile, all inside one small dispatch.
        std::env::set_var("BRAIN_TILE_BUDGET_WORDS", "128");

        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 32, 1, 12, 8, false, false);
        let g = &eng.gpu;

        let (m, k, n) = (3u32, 8u32, 40u32);
        let mut rng = Rng::new(7);
        let x: Vec<f32> = (0..(m * k) as usize).map(|_| rng.next_gaussian() as f32 * 0.3).collect();
        let w: Vec<f32> = (0..(n * k) as usize).map(|_| rng.next_gaussian() as f32 * 0.3).collect();
        let xb = g.storage_init("t_mmtile_x", &x);
        let wb = g.storage_init("t_mmtile_w", &w);

        let out_ref = g.storage((m * n) as u64);
        let step = eng.mm(&xb, &wb, &out_ref, m, k, n);
        g.submit(&[], &[step]);
        g.poll_wait();
        let reference = g.read(&out_ref, (m * n) as usize);

        let out_tiled = g.storage((m * n) as u64);
        let mut steps = Vec::new();
        eng.mm_tiled_into(&mut steps, &xb, &wb, &out_tiled, m, k, n);
        assert!(steps.len() > 1, "BRAIN_TILE_BUDGET_WORDS=128 at k=8 must force more than one tile, got {}", steps.len());
        g.submit(&[], &steps);
        g.poll_wait();
        let tiled = g.read(&out_tiled, (m * n) as usize);

        let tol = |x: f32| 1e-3 * x.abs().max(1.0) + 1e-4;
        for row in 0..m as usize {
            for col in 0..n as usize {
                let i = row * n as usize + col;
                assert!(
                    (tiled[i] - reference[i]).abs() <= tol(reference[i]),
                    "row {row} col {col} (tile boundaries at 16/32): tiled={} untiled={}",
                    tiled[i],
                    reference[i]
                );
            }
        }

        match prev {
            Some(v) => std::env::set_var("BRAIN_TILE_BUDGET_WORDS", v),
            None => std::env::remove_var("BRAIN_TILE_BUDGET_WORDS"),
        }
    }

    /// M4.2, fp32-KV branch: `Self::qk_norm_rope`/`Self::qk_norm_rope_append`
    /// must be bit-identical to the unfused `rms` -> `ROPE_PAGED` (-> `KV_APPEND_B`
    /// for K) triple this milestone collapsed - not merely close, since normalizing
    /// then rotating the same values in the same order is not a reassociation.
    /// Runs against the SAME `q_pre`/`k_pre` inputs `prefill`'s last layer already
    /// fused, so this is a real A/B on real activations, not a synthetic input.
    #[test]
    fn qk_norm_rope_fused_is_bit_identical_to_the_unfused_pair() {
        let block_size = 4u64;
        let num_blocks = 32u64;
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, block_size as u32, num_blocks as u32, 1, 12, 8, false, false);
        let mut table = BlockTable::new();
        let prompt = [1u32, 5, 3];
        let _ = eng.prefill(&mut table, &prompt);
        let rows = prompt.len() as u32;

        // `sc.q_pre`/`sc.k_pre`/`sc.pos_buf`/`sc.blk_buf`/`sc.off_buf` are reused
        // every layer (the last two are per-CHUNK, not per-layer, and this prompt
        // fits one chunk), so after `prefill` they still hold exactly the inputs
        // the LAST layer's fused dispatch consumed.
        let l = cfg.n_layers as usize - 1;
        let (hd, nh, nkv, hkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.kv_dim());
        let g = &eng.gpu;

        let unfused = |x: &DeviceBuffer, weight_name: &str, heads: u32| -> Vec<f32> {
            let w = g.storage_init("t_unfused_w", &map[&format!("blocks.{l}.{weight_name}")]);
            let out = g.storage((rows * heads * hd) as u64);
            let rms_step = eng.rms(x, &w, &out, hd, rows * heads);
            g.submit(&[], &[rms_step]);
            g.poll_wait();
            let rope_step = g.step(ROPE_PAGED, &[&out, &eng.sc.pos_buf], &[rows, heads, hd, heads * hd, fb(cfg.rope_theta)], rows * heads * (hd / 2));
            g.submit(&[], &[rope_step]);
            g.poll_wait();
            g.read(&out, (rows * heads * hd) as usize)
        };

        let fused_q = g.read(&eng.sc.q, (rows * cfg.q_dim()) as usize);
        let ref_q = unfused(&eng.sc.q_pre, "attn.q_norm.weight", nh);
        assert_eq!(fused_q, ref_q, "M4.2: fused Q (norm+RoPE) must be bit-identical to the unfused pair");

        let ref_k = unfused(&eng.sc.k_pre, "attn.k_norm.weight", nkv);
        let fused_k = g.read(&eng.sc.k, (rows * hkv) as usize);
        assert_eq!(fused_k, ref_k, "M4.2: fused K (norm+RoPE) must be bit-identical to the unfused pair");

        // The K-only append fusion must ALSO land the same values in the
        // paged pool, at the SAME (block, offset) slot `KV_APPEND_B` would
        // have written to - the address arithmetic is new code this milestone
        // added, not shared with `qk_norm_rope`'s Q path.
        let blocks: Vec<u32> = g.read(&eng.sc.blk_buf, rows as usize).iter().map(|f| f.to_bits()).collect();
        let offsets: Vec<u32> = g.read(&eng.sc.off_buf, rows as usize).iter().map(|f| f.to_bits()).collect();
        let pool = g.read(&eng.pool_k[l], (num_blocks * block_size * hkv as u64) as usize);
        for r in 0..rows as usize {
            let slot = blocks[r] as usize * block_size as usize + offsets[r] as usize;
            let want = &ref_k[r * hkv as usize..(r + 1) * hkv as usize];
            let got = &pool[slot * hkv as usize..(slot + 1) * hkv as usize];
            assert_eq!(got, want, "M4.2: pool_k row {r} (slot {slot}) must match the unfused reference");
        }
    }

    /// M4.2, int8-KV branch: the fused Q/K norm+RoPE pass must stay
    /// bit-identical when `kv_int8` is set too - this branch does NOT fuse
    /// the append (see `Self::qk_norm_rope`'s call site doc for why), so only
    /// the norm+RoPE stage is checked here; `int8_kv_close_to_fp32` already
    /// covers the append+scores+apply chain end to end.
    #[test]
    fn qk_norm_rope_fused_is_bit_identical_to_the_unfused_pair_kv_int8() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 32, 1, 12, 8, true, false);
        let mut table = BlockTable::new();
        let prompt = [1u32, 5, 3];
        let _ = eng.prefill(&mut table, &prompt);
        let rows = prompt.len() as u32;
        let l = cfg.n_layers as usize - 1;
        let (hd, nh, nkv, hkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.kv_dim());
        let g = &eng.gpu;

        let unfused = |x: &DeviceBuffer, weight_name: &str, heads: u32| -> Vec<f32> {
            let w = g.storage_init("t_unfused_w", &map[&format!("blocks.{l}.{weight_name}")]);
            let out = g.storage((rows * heads * hd) as u64);
            let rms_step = eng.rms(x, &w, &out, hd, rows * heads);
            g.submit(&[], &[rms_step]);
            g.poll_wait();
            let rope_step = g.step(ROPE_PAGED, &[&out, &eng.sc.pos_buf], &[rows, heads, hd, heads * hd, fb(cfg.rope_theta)], rows * heads * (hd / 2));
            g.submit(&[], &[rope_step]);
            g.poll_wait();
            g.read(&out, (rows * heads * hd) as usize)
        };

        let fused_q = g.read(&eng.sc.q, (rows * cfg.q_dim()) as usize);
        assert_eq!(fused_q, unfused(&eng.sc.q_pre, "attn.q_norm.weight", nh), "M4.2 (kv_int8): fused Q must be bit-identical to the unfused pair");
        let fused_k = g.read(&eng.sc.k, (rows * hkv) as usize);
        assert_eq!(fused_k, unfused(&eng.sc.k_pre, "attn.k_norm.weight", nkv), "M4.2 (kv_int8): fused K must be bit-identical to the unfused pair");
    }

    /// M4.3: on an all-int8-weight engine, `Self::rms_quant`'s fused dispatch
    /// (`RMSNORM_QUANT_FUSED`) must produce EXACTLY the `(sx, xq)` pair the
    /// unfused `rms` -> `max_abs_row` -> `quant_pack` triad would have - not
    /// a tolerance check, since `v = x*inv*w[c]` is recomputed with the
    /// identical expression and operand order every time, which IEEE754
    /// guarantees reproduces the same bits.
    ///
    /// Dispatches `Self::rms_quant` directly (rather than reading `i8_scratch`
    /// back after a full `prefill`) because `I8Scratch::sx` is ONE buffer
    /// SHARED across every distinct K-width a layer quantizes (`xn1`'s `d`,
    /// `ctx`'s `hq`, `xn2`'s `d` again, `h`'s `ff` - `Self::quant_once`'s own
    /// call sites) - a real forward pass overwrites it several times per
    /// layer, so its state after `prefill` returns reflects whichever call
    /// happened LAST in program order (`h`'s `ff`-width quant), not `xn2`'s.
    /// Reading it back immediately after one isolated dispatch sidesteps that
    /// entirely and tests the kernel itself, on synthetic but non-degenerate
    /// (non-uniform, sign-mixed) input.
    #[test]
    fn rms_quant_fused_is_bit_identical_to_the_unfused_triad() {
        let cfg = QwenConfig::tiny_i8();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 8, 8, 32, false, true);
        if !eng.weights_int8() {
            // Capability-gated fallback (CPU JIT/no packed-int8 device): the
            // fused kernel is never dispatched there (`Self::rms_quant`'s own
            // gate), so there is nothing this test can compare.
            assert!(!eng.gpu().caps().numeric.int8_dot, "device reports int8_dot but the engine fell back to fp32");
            brain_testutil::skip_unavailable("rms+quant fusion: device has no packed-int8 path");
            return;
        }
        let d = cfg.d_model;
        let rows = 5u32;
        let g = &eng.gpu;

        let xs: Vec<f32> = (0..rows * d).map(|i| (i as f32 * 0.037).sin() * 3.0 - 0.5).collect();
        let ws: Vec<f32> = (0..d).map(|i| 0.5 + (i as f32 * 0.011).cos() * 0.3).collect();
        let x = g.storage_init("t_rq_x", &xs);
        let w = g.storage_init("t_rq_w", &ws);
        let out = g.storage((rows * d) as u64); // unread by the fused branch; kept to match `rms_quant`'s signature.

        let mut fused_steps: Vec<Step> = Vec::new();
        eng.rms_quant(&mut fused_steps, &x, &w, &out, d, rows);
        g.submit(&[], &fused_steps);
        g.poll_wait();
        let scratch = eng.i8_scratch.as_ref().expect("weights_int8() true implies i8_scratch is Some");
        let got_sx = g.read(&scratch.sx, rows as usize);
        let got_xq = g.read(scratch.xq_for(d), (rows * d / 4) as usize);

        // Reference: the ORIGINAL three-dispatch sequence over the SAME `x`/`w`,
        // into fresh buffers untouched by the fused dispatch above.
        let xn_ref = g.storage((rows * d) as u64);
        let rms_step = eng.rms(&x, &w, &xn_ref, d, rows);
        g.submit(&[], &[rms_step]);
        g.poll_wait();
        let sx_ref = g.storage(rows as u64);
        let xq_ref = g.storage((rows * d / 4) as u64);
        let quant_steps = model::int8::quant_rows_steps(
            g,
            model::int8::QuantRows { kernels: [MAX_ABS_ROW, QUANT_PACK], x: &xn_ref, sx: &sx_ref, xq: &xq_ref, xgs: None },
            0,
            rows,
            d,
        );
        g.submit(&[], &quant_steps);
        g.poll_wait();
        let want_sx = g.read(&sx_ref, rows as usize);
        let want_xq = g.read(&xq_ref, (rows * d / 4) as usize);

        assert_eq!(got_sx, want_sx, "M4.3: fused per-row scale must be bit-identical to the unfused triad");
        assert_eq!(got_xq, want_xq, "M4.3: fused packed int8 activation must be bit-identical to the unfused triad");
    }

    /// The on-device iterative top-K extraction (`topk_extract_step` composed
    /// with the existing `argmax_part`/`argmax_final`) must return EXACTLY the
    /// row's true top-K logits+indices, sorted descending - an exact,
    /// deterministic operation with no tolerance to gate on
    /// (dims chosen so vocab != any other dimension, so
    /// a transposed/wrong-stride bug can't hide behind a coincidence).
    #[test]
    fn topk_extraction_matches_host_reference() {
        let cfg = medium_cfg();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 4, 12, 8, false, false);
        let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
        let prompts = [vec![1u32, 5, 3], vec![7u32, 2, 9], vec![4u32, 4, 1]];
        let mut inputs = Vec::new();
        for (t, p) in tables.iter_mut().zip(prompts.iter()) {
            let h = eng.prefill(t, p);
            inputs.push(Engine::argmax(&eng.logits(&h)));
        }
        let k = 40usize;
        let bsz = inputs.len() as u32;
        let vocab = eng.cfg.vocab as usize;
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        eng.forward_batched(&mut refs, &inputs); // leaves the hidden in sc.xn_final

        // Ground truth: the SAME device head computation (`submit_greedy_head`'s
        // matmul into `logits_dev`), read back in full BEFORE the extraction
        // loop masks it. Comparing against a HOST-computed matmul instead would
        // fail on ordinary float non-associativity (device tiled GEMM vs a
        // serial host dot product reduce differently) -- this test's job is to
        // verify the EXTRACTION, not re-litigate the head matmul itself.
        eng.submit_greedy_head(bsz);
        let full_logits = eng.gpu.read(&eng.logits_dev, (bsz as usize) * vocab);

        // Same hidden, same head matmul, now via the top-k extraction path.
        let candidates = eng.topk_from_hidden(bsz, k as u32);
        for (row, cands) in candidates.iter().enumerate() {
            let full = &full_logits[row * vocab..(row + 1) * vocab];
            let mut host: Vec<(u32, f32)> = full.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
            host.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let host_top_k = &host[..k];
            assert_eq!(cands.len(), k);
            for (c, h) in cands.iter().zip(host_top_k.iter()) {
                assert_eq!(c.0, h.0, "row {row}: device top-k index diverges from the host reference");
                assert_eq!(c.1, h.1, "row {row}: device top-k value diverges from the host reference");
            }
        }
    }

    /// The prefill budget must spread a burst of admissions across iterations
    /// (decode runs between them) without changing ANY output, and must never
    /// starve: at least one waiting request is admitted per iteration.
    #[test]
    fn prefill_budget_spreads_admissions_without_changing_outputs() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let reqs = || {
            (0..4u32)
                .map(|i| Request { prompt: vec![1 + i, 5, 3, 7, 2], max_new: 5, eos: None })
                .collect::<Vec<_>>()
        };

        // Reference: unlimited budget (the old behaviour).
        let mut a = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 4, 12, 8, false, false), 4);
        a.set_prefill_budget(u32::MAX);
        for r in reqs() {
            a.submit(r);
        }
        let want = a.run();

        // Tight budget: one 5-token prompt exhausts it, so the 4 arrivals must
        // be admitted over MULTIPLE iterations - with decode in between - and
        // still produce token-identical outputs.
        let mut b = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 4, 12, 8, false, false), 4);
        b.set_prefill_budget(5);
        for r in reqs() {
            b.submit(r);
        }
        let mut admit_iters = 0;
        let mut got = std::collections::HashMap::new();
        while b.pending() {
            let rep = b.step_report();
            if !rep.admitted.is_empty() {
                admit_iters += 1;
                assert!(
                    rep.admitted.len() <= 2,
                    "a 5-token budget must not admit a whole burst at once, got {}",
                    rep.admitted.len()
                );
            }
            for (id, toks) in rep.completed {
                got.insert(id, toks);
            }
        }
        assert!(admit_iters >= 2, "admissions must be spread across iterations");
        assert_eq!(got, want, "the budget changes WHEN tokens appear, never WHICH");
    }

    /// The two-stage argmax (vocab >= ARGMAX_SPLIT_MIN_VOCAB) must pick exactly
    /// the token the host head picks - including the lowest-index tie-break.
    #[test]
    fn split_argmax_matches_the_host_head_at_large_vocab() {
        let mut cfg = QwenConfig::tiny();
        cfg.vocab = 8192; // forces the argmax_part/argmax_final path
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);

        let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
        let prompts = [vec![11u32, 55, 33], vec![77u32, 22, 99], vec![44u32, 45, 46]];
        let mut inputs = Vec::new();
        for (t, p) in tables.iter_mut().zip(prompts.iter()) {
            let h = eng.prefill(t, p);
            inputs.push(Engine::argmax(&eng.logits(&h)));
        }
        for _ in 0..3 {
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                eng.forward_batched(&mut refs, &inputs)
            };
            // `dev` first: it reads the batch still resident in `sc.xn_final`;
            // the per-row `logits` calls below overwrite that same scratch
            // buffer's row 0 one row at a time (see
            // `device_head_argmax_matches_the_host_head`'s own doc).
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            assert_eq!(dev, host, "split argmax diverged from the host head");
            inputs = host;
        }
    }

    /// Every KV word `blocks` occupies, read straight off the engine's pool
    /// buffers - an independent oracle for the swap tests below.
    ///
    /// Deliberately NOT `Engine::read_kv_blocks` (the swap's own gather): a
    /// swap that forgot a tensor - the int8 dequant scales are the obvious
    /// candidate, being a second buffer addressed by the same slot - would
    /// forget it identically on both sides of a self-comparison and pass. This
    /// restates the pool layout from the config, so the two have to agree.
    fn pool_bytes_of_blocks(eng: &Engine, blocks: &[u32]) -> Vec<u32> {
        let words = |buf: &DeviceBuffer, n: usize| -> Vec<u32> { eng.gpu.read(buf, n).iter().map(|v| v.to_bits()).collect() };
        let nb = eng.alloc.num_blocks() as usize;
        let (pw, sw) = kv_pool_words(&eng.cfg, eng.block_size, 1, eng.kv_int8);
        let (pw, sw) = (pw as usize, sw as usize);
        let mut out = Vec::new();
        let pools: Vec<(Vec<u32>, usize)> = (0..eng.cfg.n_layers as usize)
            .flat_map(|l| {
                let mut v = vec![(words(&eng.pool_k[l], nb * pw), pw), (words(&eng.pool_v[l], nb * pw), pw)];
                if eng.kv_int8 {
                    v.push((words(&eng.scales_k[l], nb * sw), sw));
                    v.push((words(&eng.scales_v[l], nb * sw), sw));
                }
                v
            })
            .collect();
        for &b in blocks {
            for (buf, width) in &pools {
                let at = b as usize * width;
                out.extend_from_slice(&buf[at..at + width]);
            }
        }
        out
    }

    /// **The host-RAM KV offload gate, at the model level.** A sequence whose
    /// whole KV is demoted to host RAM mid-generation and promoted back must
    /// then decode BIT-IDENTICALLY to one that was never demoted: same engine,
    /// same weights, same prompt, same tokens out.
    ///
    /// Run on ONE engine (never two, per this module's own precedent: two
    /// independently-built engines can select different autotuned kernel
    /// variants for the same shape and differ by float noise well under a real
    /// bug), and deliberately with a DECOY sequence occupying the pool while
    /// the demoted one is away, so the promote lands on different physical
    /// blocks than it left on and any assumption of slot identity fails here.
    ///
    /// Both KV dtypes: the fp32 pool moves one word per element, the int8 pool
    /// moves packed bytes PLUS a separate per-`(slot, kv-head)` scale array,
    /// and a swap that forgot the scales would still round-trip plausible
    /// numbers - just wrong ones.
    #[test]
    fn a_demoted_and_restored_sequence_decodes_identically() {
        for kv_int8 in [false, true] {
            let cfg = kv_probe_cfg();
            let map = tiny_weights(&cfg);
            let prompt = vec![1u32, 5, 3, 9, 2, 7, 11, 4, 6];
            let (bs, num_blocks, max_batch, mbt, max_prefill) = (4u32, 48u32, 2u32, 12u32, 16u32);
            let mut eng =
                Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, bs, num_blocks, max_batch, mbt, max_prefill, kv_int8, false);
            eng.set_kv_offload_bytes(4 << 20);
            let steps = 12usize;
            let swap_at = 5usize;

            // Reference: prefill, then `steps` greedy decodes, straight through.
            let mut t = BlockTable::new();
            let hidden = eng.prefill(&mut t, &prompt);
            let mut next = eng.admit_greedy(&hidden);
            let mut want = vec![next];
            for _ in 0..steps {
                next = eng.forward_batched_greedy(&mut [&mut t], &[next])[0];
                want.push(next);
            }
            eng.release_table(&mut t);

            // The same generation, interrupted by a full demote/promote cycle.
            let mut t = BlockTable::new();
            let hidden = eng.prefill(&mut t, &prompt);
            let mut next = eng.admit_greedy(&hidden);
            let mut got = vec![next];
            for i in 0..steps {
                if i == swap_at {
                    let free_before = eng.free_blocks();
                    let held = t.blocks().len() as u32;
                    let t_blocks = t.blocks().to_vec();
                    // The device-level byte gate. Read through an ORACLE
                    // that walks the pool buffers itself (see
                    // `pool_bytes_of_blocks`), never through the engine's own
                    // gather: a swap that silently skipped a tensor would
                    // skip it on both sides of a self-comparison and pass.
                    let want_words = pool_bytes_of_blocks(&eng, &t_blocks);
                    let reclaimed = eng.demote_kv(77, &mut t).expect("demote");
                    // Not necessarily every block: the prefix cache holds its
                    // own reference on this prompt's full blocks, so those stay
                    // live (and are copied all the same - that is exactly the
                    // sharing case the round-trip has to survive).
                    assert!(reclaimed > 0 && reclaimed <= held, "demote freed {reclaimed} of {held} blocks");
                    assert_eq!(eng.free_blocks(), free_before + reclaimed);
                    assert!(eng.kv_offload_stats().bytes_resident > 0, "the KV must actually be in host RAM");

                    // A decoy takes the freed blocks and scribbles its own
                    // K/V (and, on the int8 pool, its own dequant scales) over
                    // them, and HOLDS them across the promote so the restore
                    // must land somewhere else entirely. Its prompt differs
                    // from ours deliberately: an identical one would hit the
                    // prefix cache, adopt the very blocks it is supposed to be
                    // trampling, and quietly make this decoy a no-op.
                    let mut decoy = BlockTable::new();
                    let other: Vec<u32> = prompt.iter().map(|t| (t + 13) % 29).collect();
                    let dh = eng.prefill(&mut decoy, &other);
                    let _ = eng.admit_greedy(&dh);

                    let before = t_blocks.clone();
                    t = eng.promote_kv(77).expect("promote");
                    assert_ne!(t.blocks(), before.as_slice(), "the decoy must have forced a restore onto different physical blocks");
                    let got_words = pool_bytes_of_blocks(&eng, t.blocks());
                    // Reported as the first differing word rather than two
                    // 1000-element vectors: the failure is a layout/plan bug,
                    // and WHERE it first diverges names the tensor.
                    let diff = got_words.iter().zip(&want_words).position(|(a, b)| a != b);
                    assert_eq!(got_words.len(), want_words.len());
                    assert_eq!(diff, None, "kv_int8={kv_int8}: restored blocks must hold byte-identical KV (first differing word of {})", want_words.len());
                    eng.release_table(&mut decoy);
                    assert_eq!(eng.kv_offload_stats().bytes_resident, 0);
                }
                next = eng.forward_batched_greedy(&mut [&mut t], &[next])[0];
                got.push(next);
            }
            assert_eq!(got, want, "kv_int8={kv_int8}: a demoted-and-restored sequence must decode identically");
        }
    }

    /// The same gate one level up, through the real scheduler: a KV pool far
    /// too small for the admitted set must produce EXACTLY the outputs a pool
    /// with room to spare does, by preempting sequences to host RAM and
    /// resuming them - not by truncating, reordering or dropping anything.
    ///
    /// The small pool is sized so preemption is unavoidable, and the test
    /// asserts swaps actually happened: without that it would pass trivially
    /// on an engine that never swapped at all.
    #[test]
    fn a_pool_too_small_for_the_batch_still_produces_identical_outputs() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let reqs = || {
            (0..4u32)
                .map(|i| Request { prompt: vec![1 + i, 5, 3, 7, 2], max_new: 8, eos: None })
                .collect::<Vec<_>>()
        };
        // Everything but `num_blocks` is identical between the two engines, so
        // no kernel choice (all of which key off `cap`/`mbt`/`max_batch`, never
        // the pool's block count) can differ between them.
        let (bs, max_batch, mbt, max_prefill) = (4u32, 4u32, 12u32, 8u32);

        let roomy = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, 96, max_batch, mbt, max_prefill, false, false);
        let mut a = Scheduler::new(roomy, 4);
        for r in reqs() {
            a.submit(r);
        }
        let want = a.run();
        assert_eq!(a.offload_stats().demotions, 0, "the roomy baseline must never have needed to swap");

        // 8 blocks of 4 tokens = 32 cached tokens for four sequences that need
        // 13 each: the batch cannot be resident all at once.
        let mut tight = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, bs, 8, max_batch, mbt, max_prefill, false, false);
        tight.set_kv_offload_bytes(8 << 20);
        let mut b = Scheduler::new(tight, 4);
        for r in reqs() {
            b.submit(r);
        }
        let mut got = HashMap::new();
        let mut demoted = 0usize;
        let mut promoted = 0usize;
        let mut iters = 0usize;
        while b.pending() {
            let rep = b.step_report();
            demoted += rep.demoted.len();
            promoted += rep.promoted.len();
            for (id, toks) in rep.completed {
                got.insert(id, toks);
            }
            iters += 1;
            assert!(iters < 10_000, "the scheduler must make progress, not thrash: {demoted} demotions so far");
        }
        assert!(demoted > 0, "the tight pool must have forced at least one preemption");
        assert_eq!(promoted, demoted, "every preempted sequence must come back exactly once");
        assert_eq!(got, want, "swapping through host RAM must not change a single token");

        let s = b.offload_stats();
        assert_eq!(s.bytes_resident, 0, "nothing may be left in host RAM once every sequence has finished");
        assert_eq!(s.blocks_in, s.blocks_out, "every block copied out must be copied back");
    }

    /// `step_report` must account for **every** token exactly once and admit each
    /// request exactly once. `brain perf` derives time-to-first-token and
    /// inter-token latency purely from these counts, so a double-count or a
    /// dropped token silently corrupts every latency number computed from it.
    #[test]
    fn step_report_accounts_for_every_token_exactly_once() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 3);

        let wants = [6usize, 4, 9];
        let ids: Vec<u64> = wants
            .iter()
            .enumerate()
            .map(|(i, &n)| sched.submit(Request { prompt: vec![1u32 + i as u32, 5, 3], max_new: n, eos: None }))
            .collect();

        let mut produced: HashMap<u64, usize> = HashMap::new();
        let mut admitted: Vec<u64> = Vec::new();
        let mut finished: Vec<u64> = Vec::new();
        let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
        while sched.pending() {
            let rep = sched.step_report();
            admitted.extend(rep.admitted.iter().copied());
            for (id, n) in &rep.produced {
                *produced.entry(*id).or_default() += n;
            }
            finished.extend(rep.finished.iter().copied());
            for (id, toks) in rep.completed {
                outputs.insert(id, toks);
            }
        }

        admitted.sort_unstable();
        finished.sort_unstable();
        let mut expect = ids.clone();
        expect.sort_unstable();
        assert_eq!(admitted, expect, "every request must be admitted exactly once");
        assert_eq!(finished, expect, "every request must finish exactly once");

        for (i, id) in ids.iter().enumerate() {
            let out = outputs.get(id).expect("request must complete");
            assert_eq!(out.len(), wants[i], "request {id} produced the wrong length");
            assert_eq!(
                produced.get(id).copied().unwrap_or(0),
                out.len(),
                "incremental token report for {id} must sum to the tokens returned"
            );
        }
    }

    /// The reporting path must not change what the scheduler computes.
    #[test]
    fn step_report_does_not_perturb_outputs() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let reqs = || {
            vec![
                Request { prompt: vec![1u32, 5, 3], max_new: 7, eos: None },
                Request { prompt: vec![9u32, 2], max_new: 5, eos: None },
            ]
        };

        let mut a = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 2, 12, 8, false, false), 2);
        for r in reqs() {
            a.submit(r);
        }
        let via_step = a.run();

        let mut b = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 2, 12, 8, false, false), 2);
        for r in reqs() {
            b.submit(r);
        }
        let mut via_report: HashMap<u64, Vec<u32>> = HashMap::new();
        while b.pending() {
            for (id, toks) in b.step_report().completed {
                via_report.insert(id, toks);
            }
        }
        assert_eq!(via_step, via_report, "the reporting path must be observationally identical");
    }

    /// Continuous batching: requests submitted at DIFFERENT times (one mid-flight)
    /// must each produce the same tokens as run alone - the scheduler admits,
    /// batches, completes, and frees dynamically without changing any output.
    #[test]
    fn scheduler_dynamic_admission_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let model = Qwen::new(cfg.clone(), 1, 64, &map);

        let prompts = [vec![1u32, 5, 3, 9], vec![7u32, 2, 4], vec![3u32, 3, 8, 1, 6]];
        let maxn = [10usize, 6, 8];
        let refs: Vec<Vec<u32>> = prompts
            .iter()
            .zip(maxn)
            .map(|(p, n)| {
                let mut r = Rng::new(0);
                crate::sample::generate_kv(&model, p, n, 0.0, 0, 1.0, None, &mut r)
            })
            .collect();

        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 4, 8, 32, false, false);
        let mut sched = Scheduler::new(eng, 4);
        let mut out: HashMap<u64, Vec<u32>> = HashMap::new();

        let id0 = sched.submit(Request { prompt: prompts[0].clone(), max_new: maxn[0], eos: None });
        let id1 = sched.submit(Request { prompt: prompts[1].clone(), max_new: maxn[1], eos: None });
        // Run two iterations with only the first two requests active...
        for _ in 0..2 {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }
        // ...then submit a third mid-flight; it must batch in and still be correct.
        let id2 = sched.submit(Request { prompt: prompts[2].clone(), max_new: maxn[2], eos: None });
        while sched.pending() {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }

        assert_eq!(out[&id0], refs[0], "req0 under continuous batching != reference");
        assert_eq!(out[&id1], refs[1], "req1 under continuous batching != reference");
        assert_eq!(out[&id2], refs[2], "mid-flight req2 != reference");
    }

    /// REGRESSION (attention-scratch dispatch width): the
    /// on-device decode-WINDOW path (`Engine::forward_batched_greedy_window`,
    /// `Input::Resident` sub-steps 1..k) had ZERO test coverage before this -
    /// every other test here keeps the scheduler in single-step (`k=1`)
    /// territory by always having a waiting/mixed-sampling request in flight,
    /// which forces `k=1` (`model::serve::Scheduler::step`'s `all_greedy &&
    /// self.waiting.is_empty()` gate). `Input::Resident` is also the one
    /// `Input` variant `run_batched_submit` gets NO host seqlens for (`&[]` -
    /// see `serve.rs::forward_batched_greedy_window`'s sub-step 1..k calls);
    /// its per-row KV length lives only on-device (`sc.seqlen_buf`, walked by
    /// `decode_advance`). A single request, nothing else submitted, comfortably
    /// exceeding `DECODE_WINDOW` in `max_new`, is exactly the shape that makes
    /// the scheduler choose `k = DECODE_WINDOW` for most of the run - if the
    /// window path's on-device bookkeeping (positions/seqlens/block-table
    /// scheduling for those resident sub-steps) were wrong, the argmax'd
    /// tokens would diverge from the independent single-step reference below.
    /// G4: at BOTH KV dtypes - the on-device window bookkeeping
    /// (positions/seqlens/block-table scheduling for the resident sub-steps)
    /// is dtype-independent code, but it feeds `run_batched_submit`'s int8
    /// branch just the same as the single-step path, so this is worth proving
    /// separately rather than assuming.
    #[test]
    fn decode_window_path_matches_the_single_step_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let model = Qwen::new(cfg.clone(), 1, 64, &map);
        let prompt = vec![1u32, 5, 3, 9, 2];
        let max_new = 13usize; // > DECODE_WINDOW (4), so k=4 fires for several rounds
        let mut r = Rng::new(0);
        let reference = crate::sample::generate_kv(&model, &prompt, max_new, 0.0, 0, 1.0, None, &mut r);

        // Plenty of batch/block headroom: a single request must never fall
        // back to k=1 for lack of free blocks (serve.rs's own guard: `k > 1
        // && free_blocks < active.len() * k` => k=1).
        const { assert!(DECODE_WINDOW > 1, "test is meaningless if the window is disabled") };
        for kv_int8 in [false, true] {
            let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 128, 4, 8, 32, kv_int8, false);
            let mut sched = Scheduler::new(eng, 4);
            let id = sched.submit(Request { prompt: prompt.clone(), max_new, eos: None });

            let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut saw_a_window_step = false;
            while sched.pending() {
                let report = sched.step_report();
                // `produced` is `(id, tokens produced THIS iteration)`; k=1 always
                // produces exactly one token per running row, so a row producing
                // more than one is direct evidence a window step (k>1) actually
                // ran. Nothing else was ever submitted, so `waiting` is empty from
                // the first iteration on - the scheduler has no reason to prefer
                // k=1 beyond block pressure, which the oversized pool above rules out.
                if report.produced.iter().any(|&(_, n)| n > 1) {
                    saw_a_window_step = true;
                }
                for (id, t) in report.completed {
                    out.insert(id, t);
                }
            }
            assert!(saw_a_window_step, "kv_int8={kv_int8}: this test is meaningless if the scheduler never actually chose k>1");
            assert_eq!(out[&id], reference, "kv_int8={kv_int8}: the decode-window path must match single-step decoding exactly");
        }
    }

    /// Real (non-greedy) sampling through `Scheduler::submit_sampled`, driven
    /// by the real `Engine` (both the admission-time host sampling and the
    /// per-token on-device top-K path exercised together, mixed with an
    /// ordinary greedy sequence in the SAME batch - the `all_greedy` fallback
    /// this plan's Scheduler change hinges on). A fixed seed must reproduce
    /// bit-for-bit; a high temperature must, with overwhelming probability,
    /// diverge from the greedy continuation of the same prompt.
    #[test]
    fn scheduler_sampled_requests_are_reproducible_and_differ_from_greedy() {
        let cfg = medium_cfg();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9];
        let max_new = 12usize;
        let params = SampleParams { temp: 2.0, top_k: 20, top_p: 1.0 };

        let run_sampled = |seed: u64| {
            let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 4, 12, 32, false, false);
            let mut sched = Scheduler::new(eng, 4);
            // A plain greedy sequence rides in the SAME batch, proving the
            // mixed-batch fallback doesn't perturb it.
            let greedy_id = sched.submit(Request { prompt: prompt.clone(), max_new, eos: None });
            let sampled_id = sched.submit_sampled(Request { prompt: prompt.clone(), max_new, eos: None }, params, seed);
            let out = sched.run();
            (out[&greedy_id].clone(), out[&sampled_id].clone())
        };

        let (greedy_a, sampled_a) = run_sampled(1234);
        let (greedy_b, sampled_b) = run_sampled(1234);
        assert_eq!(sampled_a, sampled_b, "same seed must reproduce the sampled continuation exactly");
        assert_eq!(greedy_a, greedy_b, "the greedy sequence in the same batch must stay deterministic regardless");

        let (_, sampled_other_seed) = run_sampled(5678);
        assert_ne!(sampled_a, sampled_other_seed, "different seeds should not collide over 12 tokens at temp=2.0");
        assert_ne!(sampled_a, greedy_a, "temp=2.0 sampling should diverge from the greedy continuation of the same prompt");

        // Reference: the admission-time-and-decode-loop sampling never
        // strayed outside the model's vocabulary (a debugging class this
        // engine's own lessons flag as "silent garbage, not a crash").
        for &t in &sampled_a {
            assert!(t < cfg.vocab, "sampled token {t} outside vocab {}", cfg.vocab);
        }
    }

    /// G2: `Engine::kv_pool_bytes()` must equal the independently-recomputed
    /// [`kv_pool_bytes`] free function (two derivations that must agree - the
    /// engine's is recorded at construction, this test's is a fresh call), and
    /// an int8 engine's pool must be strictly smaller than an fp32 one at the
    /// SAME `num_blocks`.
    #[test]
    fn engine_kv_pool_bytes_matches_what_it_allocated() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let (bs, nb) = (4u32, 64u32);
        let e32 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, nb, 1, 8, 32, false, false);
        let e8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, bs, nb, 1, 8, 32, true, false);
        assert_eq!(e32.kv_pool_bytes(), kv_pool_bytes(&cfg, bs, nb, false));
        assert_eq!(e8.kv_pool_bytes(), kv_pool_bytes(&cfg, bs, nb, true));
        assert!(e8.kv_pool_bytes() < e32.kv_pool_bytes(), "int8 pool must be smaller than fp32 at the same num_blocks");
    }

    /// int8 paged KV stays close to fp32 through prefill + decode (both read
    /// the quantised cache) - a structural sanity check that CUMULATIVE
    /// divergence over several autoregressive steps stays small, not a
    /// precision claim: G3 already derives the exact per-element quantization
    /// bound (0.5 of a step) for a single append, and
    /// the REAL accuracy measurement (loss
    /// delta +0.0154 on Qwen3-0.6B) - this test cannot substitute for either
    /// (lesson 18: a toy config's error magnitude cannot predict the real
    /// one). What it CAN catch is a wiring break that makes int8 decode wildly
    /// diverge from fp32 (a dropped scale propagating through several steps,
    /// a slot/head index swap) - hence a bound with real headroom above the
    /// measured baseline, not the old hand-fit constant. Runs on [`kv_probe_cfg`],
    /// not `tiny()` (lesson 4: `tiny()`'s degenerate dims don't exercise real
    /// GQA). Two independently-built engines (fp32, int8) also carry their
    /// own small autotuner-driven kernel-variant noise, independent of
    /// quantization - see `int8_kv_scale_and_bytes_match_a_host_oracle`'s doc
    /// comment - which is exactly why this bound is loose and G3's is tight.
    #[test]
    fn int8_kv_close_to_fp32() {
        let cfg = kv_probe_cfg();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 11, 4];
        let run = |int8: bool| -> Vec<f32> {
            let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 5, 32, 1, 8, 32, int8, false);
            let mut t = BlockTable::new();
            let mut hidden = e.prefill(&mut t, &prompt);
            for _ in 0..6 {
                let next = Engine::argmax(&e.logits(&hidden));
                let mut one = [&mut t];
                hidden = e.forward_batched(&mut one, &[next]);
            }
            hidden
        };
        let h32 = run(false);
        let h8 = run(true);
        let err = h32.iter().zip(&h8).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let mag = h32.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!("int8 KV vs fp32 (prefill + 6 decode) maxabs={err:e} (mag {mag:e})");
        // Measured on this probe config, the relative error sits orders of
        // magnitude below the bound asserted here. That headroom is deliberate
        // (run-to-run autotuner noise, a different random weight draw), and
        // the bound is still far tighter than the hand-fit one it replaced.
        assert!(err < 0.01 * mag + 1e-3, "int8 diverges too far: {err} vs mag {mag}");
    }

    /// A [`model::kvcalib::KvCalib::disabled`] table must be bit-identical
    /// to no calibration at all (`set_kv_calib(None)`): both leave the
    /// f32::MAX sentinel in the clip buffers, and the (single) clipped
    /// append kernel degrades to the old unclipped behaviour under it.
    #[test]
    fn a_disabled_calib_table_is_bit_identical_to_no_calibration() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2];
        let run = |calib: Option<model::kvcalib::KvCalib>| -> Vec<f32> {
            let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, true, false);
            e.set_kv_calib(calib);
            let mut t = BlockTable::new();
            e.prefill(&mut t, &prompt)
        };
        let uncalibrated = run(None);
        let disabled = run(Some(model::kvcalib::KvCalib::disabled(cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize)));
        assert_eq!(uncalibrated, disabled, "a disabled clip table must be bit-identical to no calibration at all");
    }

    /// A REAL (binding) calibrated clip must change the KV pool's contents
    /// relative to the uncalibrated kernel - proving the clipped kernel path
    /// actually dispatches and its clip ceiling actually takes effect, not
    /// just that the selector compiles.
    #[test]
    fn a_binding_calib_clip_changes_output_vs_uncalibrated() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 6, 8];
        let mut uncal = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, true, false);
        let mut t1 = BlockTable::new();
        let h_uncal = uncal.prefill(&mut t1, &prompt);

        // A deliberately tiny clip (far below any real activation magnitude)
        // on every stream -- guaranteed to bind on every token, so this is a
        // maximally aggressive, unambiguous "does the clip do anything" probe.
        let mut calib = model::kvcalib::KvCalib::disabled(cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
        for row in calib.k.iter_mut().chain(calib.v.iter_mut()) {
            row.fill(1e-6);
        }
        let mut cal = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, true, false);
        cal.set_kv_calib(Some(calib));
        let mut t2 = BlockTable::new();
        let h_cal = cal.prefill(&mut t2, &prompt);

        assert_ne!(h_uncal, h_cal, "an aggressively binding clip must change the decoded hidden state");
    }

    /// `kv_calibrated()` must say `false` for a table installed on an fp32
    /// engine, even though `self.kv_calib` is internally `Some(_)` - the
    /// table is provably never dispatched there (the int8 branch of
    /// `run_batched_submit` is the only reader). An accessor that reported
    /// `true` here would claim a calibration is binding when it is not.
    #[test]
    fn kv_calibrated_is_false_on_an_fp32_engine_even_with_a_table_installed() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut e32 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, false, false);
        assert!(!e32.kv_calibrated(), "no table installed yet");
        let calib = model::kvcalib::KvCalib::disabled(cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
        e32.set_kv_calib(Some(calib));
        assert!(!e32.kv_calibrated(), "a table on an fp32 engine must never report as calibrated -- it is never dispatched");

        let mut e8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, true, false);
        assert!(!e8.kv_calibrated(), "no table installed yet");
        let mut binding = model::kvcalib::KvCalib::disabled(cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
        for row in binding.k.iter_mut().chain(binding.v.iter_mut()) {
            row.fill(1e-6);
        }
        e8.set_kv_calib(Some(binding));
        assert!(e8.kv_calibrated(), "a real table on an int8 engine must report as calibrated");
        e8.set_kv_calib(None);
        assert!(!e8.kv_calibrated(), "clearing the table must un-report calibration");
    }

    /// Chunked prefill (small chunk) must produce the same hidden as whole-prompt
    /// prefill - the prompt streams through in pieces attending the paged prefix.
    /// G4: at BOTH KV dtypes - chunk boundaries must not change which slot a
    /// token's K/V lands in, at either dtype. Kept at the ORIGINAL `1e-4`
    /// tolerance (not tightened to `assert_eq!`): `prefill_last` builds a
    /// fresh `Engine` per call, so "whole" and "chunked" are two independent
    /// builds subject to the same small autotuner-driven kernel-variant noise
    /// documented on `int8_kv_scale_and_bytes_match_a_host_oracle`.
    #[test]
    fn chunked_prefill_matches_whole() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 4, 8];
        for kv_int8 in [false, true] {
            let prefill_last = |max_prefill: u32| -> Vec<f32> {
                let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, max_prefill, kv_int8, false);
                let mut t = BlockTable::new();
                e.prefill(&mut t, &prompt)
            };
            let whole = prefill_last(16); // one chunk
            let chunked = prefill_last(2); // 4 chunks of 2
            let err = whole.iter().zip(&chunked).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            println!("kv_int8={kv_int8}: chunked (2) vs whole prefill: maxabs={err:e}");
            assert!(err < 1e-4, "kv_int8={kv_int8}: chunked prefill != whole prefill: {err}");
        }
    }

    /// REGRESSION GATE for a class of defect the serving-performance audit
    /// named directly: a "batched prefill" that batches the READBACK but still
    /// dispatches once per TOKEN (the old `Qwen::prefill`'s per-position
    /// `decode_submit` loop at `m=1`). The paged `Engine::prefill` must cost device
    /// submits proportional to the number of CHUNKS (`ceil(len / max_prefill)`),
    /// never to the raw token count within one chunk - model-agnostic in spirit
    /// (any future `PagedDecoder` gets this same shape), asserted here on the one
    /// concrete implementation that exists.
    /// The asserted shape is `submits == chunks * per_chunk` exactly - strictly
    /// PROPORTIONAL, with no fixed term. Anything one-off (a device the engine's
    /// construction left with staged-but-unsubmitted uploads, say) is baselined
    /// out of the measurement below rather than folded into `per_chunk`, because
    /// a constant that only the first chunk pays is not a per-chunk cost and
    /// multiplying it by the chunk count is simply wrong arithmetic.
    /// G4: at BOTH KV dtypes - submit counts are integers, unaffected by any
    /// floating-point noise, so this stays `assert_eq!` at both dtypes with no
    /// caveat.
    #[test]
    fn prefill_submits_scale_with_chunks_not_with_token_count() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        // A device of this test's OWN, never `testgpu::dev`'s pooled one. Submit
        // counting is a property of a wgpu device and its handles, not of one
        // engine: the pool hands every test in this binary a handle on ONE
        // device, whose submit counter and whose "a host write is staged but
        // unsubmitted" flag (`backend-wgpu`'s `writes_pending`, claimed by
        // whichever handle flushes next) are then SHARED with whatever else the
        // harness is running in parallel. Measured against the pool, this gate's
        // counts drifted by +1 at random, in a different measurement each run.
        // The device is dropped with the test, which is the orderly teardown
        // `testgpu`'s own doc calls for.
        let own = Gpu::new(PIPELINES);
        // `device_stats()` is `None` on backends that don't count (only
        // backend-wgpu does; `backend_api::Backend::stats`'s own doc comment:
        // "a consumer must report null, never zero") -- this claim is
        // structurally unverifiable there, so skip loudly rather than let
        // `unwrap_or(0)` silently turn "not counted" into a false failure.
        let probe = Engine::from_map_with_gpu(own.share_or_new(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, false, false);
        if probe.device_stats().is_none() {
            brain_testutil::skip_unavailable("this backend does not count device submits");
            return;
        }
        drop(probe);
        for kv_int8 in [false, true] {
            let submits_for = |prompt: &[u32], max_prefill: u32| -> u64 {
                let mut e = Engine::from_map_with_gpu(own.share_or_new(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, max_prefill, kv_int8, false);
                let mut t = BlockTable::new();
                // Baseline the counter on a QUIESCED device, so what is measured
                // is prefill's own work and nothing else. Engine construction can
                // leave host uploads recorded but unsubmitted (`kv_int8` writes
                // the per-layer clip-ceiling buffers - `from_map_with_gpu`'s
                // `max_row` - and `backend-wgpu`'s `write` only stages them);
                // without this flush the FIRST host write inside the first chunk
                // is what submits them, so the engine's construction cost lands
                // on chunk 1 and on no other chunk. That is a fixed per-RUN
                // cost, not a per-chunk one - it made the two-chunk run measure
                // `2 * per_chunk + 1` against a `2 * (per_chunk + 1)` expectation
                // and fail by exactly the one-off, at `kv_int8` only.
                e.gpu.flush();
                let before = e.device_stats().map(|s| s.submits).unwrap_or(0);
                e.prefill(&mut t, prompt);
                let after = e.device_stats().map(|s| s.submits).unwrap_or(0);
                after - before
            };
            // One chunk large enough to hold either prompt whole: a 4-token prompt and a
            // 16-token prompt must cost the SAME number of submits - proof the dispatch
            // is per-CALL, not per-TOKEN (a per-token dispatcher would cost four times more here).
            let short = vec![1u32, 5, 3, 9];
            let long: Vec<u32> = (0..16).map(|i| (i % 20) as u32 + 1).collect();
            let submits_short = submits_for(&short, 16);
            let submits_long = submits_for(&long, 16);
            assert_eq!(submits_short, submits_long, "kv_int8={kv_int8}: prefill submits must not scale with in-chunk token count: {submits_short} (4 tok) vs {submits_long} (16 tok)");
            assert!(submits_short > 0, "kv_int8={kv_int8}: prefill must dispatch SOMETHING");

            // The SAME 16-token prompt split into 2 then 4 chunks must cost exactly
            // twice and four times the one-chunk submit count - proportional to
            // CHUNKS, not tokens (scaling with tokens would be sixteenfold, and
            // identical at every split here). The 4-chunk point is what separates
            // "proportional to chunks" from "affine in chunks": a per-run one-off
            // would show up as a CONSTANT gap that the doubling check alone could
            // be mistaken for a per-chunk cost.
            let submits_2chunks = submits_for(&long, 8);
            let submits_4chunks = submits_for(&long, 4);
            assert_eq!(submits_2chunks, 2 * submits_long, "kv_int8={kv_int8}: 2 chunks must cost exactly 2 x 1 chunk's submits, not scale with the (unchanged) token count: {submits_2chunks} vs 2 x {submits_long}. A CONSTANT excess (2 x n + c) is a fixed per-run cost baselined into the measurement, not a prefill regression - see the flush above.");
            assert_eq!(submits_4chunks, 4 * submits_long, "kv_int8={kv_int8}: 4 chunks must cost exactly 4 x 1 chunk's submits - prefill cost must be proportional to chunks, with no fixed term: {submits_4chunks} vs 4 x {submits_long}");
        }
    }

    /// Speculative decoding output equals plain greedy - with a good (oracle)
    /// draft it takes far fewer target forwards; with a bad draft it falls back to
    /// ~one token per forward. Either way the tokens are identical. G4: at BOTH
    /// KV dtypes - the accept/reject mechanism, and `BlockTable::truncate` on a
    /// rejection, must agree with plain greedy at int8 too, not just fp32; the
    /// three engines below are built at the SAME dtype within an iteration (a
    /// within-dtype question, not a cross-dtype one), so this is
    /// `assert_eq!` on tokens either way, unaffected by the cross-engine
    /// floating-point noise the other gates account for.
    #[test]
    fn spec_decode_matches_greedy() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9];
        let max_new = 20usize;

        for kv_int8 in [false, true] {
            let mut e_ref = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, kv_int8, false);
            let greedy = e_ref.generate_greedy(std::slice::from_ref(&prompt), max_new, None)[0].clone();
            let full: Vec<u32> = prompt.iter().copied().chain(greedy.iter().copied()).collect();

            // Oracle draft: proposes the true continuation → all accepted.
            let mut e1 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, kv_int8, false);
            let (out_oracle, fwd_oracle) = e1.spec_decode(&prompt, max_new, 4, |ctx, want| {
                (0..want as usize).map(|i| full.get(ctx.len() + i).copied().unwrap_or(0)).collect()
            });
            // Bad draft: always proposes token 0 → mostly rejected.
            let mut e2 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, kv_int8, false);
            let (out_bad, fwd_bad) = e2.spec_decode(&prompt, max_new, 4, |_ctx, want| vec![0u32; want as usize]);

            println!("kv_int8={kv_int8}: spec decode: greedy={max_new} tokens | oracle-draft {fwd_oracle} target-forwards | bad-draft {fwd_bad} forwards");
            assert_eq!(out_oracle, greedy, "kv_int8={kv_int8}: spec (oracle draft) != greedy");
            assert_eq!(out_bad, greedy, "kv_int8={kv_int8}: spec (bad draft) != greedy");
            assert!(fwd_oracle < max_new, "kv_int8={kv_int8}: oracle draft should cut target forwards ({fwd_oracle} vs {max_new})");
            assert!(fwd_bad >= fwd_oracle, "kv_int8={kv_int8}: bad draft should need more forwards");
        }
    }

    /// tts multi-stream: N Talker streams (embedding inputs) decoded together on
    /// the shared paged pool must match each stream decoded alone - bit-for-bit.
    /// (The Talker is the same Qwen3 block, so the tiny config stands in for it.)
    /// G4: at BOTH KV dtypes, same `1e-6` threshold - quantization is a
    /// deterministic per-activation function with no cross-stream state, so
    /// batching must not perturb it any more than it already doesn't at fp32.
    #[test]
    fn tts_multistream_embed_matches_per_stream() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let d = cfg.d_model as usize;
        let (n_streams, steps) = (3usize, 5usize);
        let mut rng = Rng::new(42);
        let embs: Vec<Vec<Vec<f32>>> = (0..n_streams)
            .map(|_| (0..steps).map(|_| (0..d).map(|_| rng.next_gaussian() as f32).collect()).collect())
            .collect();

        for kv_int8 in [false, true] {
            // Batched: all streams advance together each step.
            let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, n_streams as u32, 8, 4, kv_int8, false);
            let mut tables: Vec<BlockTable> = (0..n_streams).map(|_| BlockTable::new()).collect();
            let mut batched: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
            // `s` is the step index into each stream's inner `Vec` (`embs[i][s]`), not an
            // index into `embs` itself - `embs` has `n_streams` entries, not `steps`.
            #[allow(clippy::needless_range_loop)]
            for s in 0..steps {
                let flat: Vec<f32> = (0..n_streams).flat_map(|i| embs[i][s].clone()).collect();
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                let h = e.forward_batched_embed(&mut refs, &flat);
                for (i, b) in batched.iter_mut().enumerate() {
                    b.push(h[i * d..(i + 1) * d].to_vec());
                }
            }

            // Per-stream reference.
            let mut worst = 0f32;
            for (i, se) in embs.iter().enumerate() {
                let mut e1 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 4, kv_int8, false);
                let mut t = BlockTable::new();
                for (s, emb) in se.iter().enumerate() {
                    let mut refs = [&mut t];
                    let h = e1.forward_batched_embed(&mut refs, emb);
                    worst = worst.max(h.iter().zip(&batched[i][s]).fold(0f32, |m, (a, b)| m.max((a - b).abs())));
                }
            }
            println!("kv_int8={kv_int8}: tts multi-stream (embed) vs per-stream: worst maxabs = {worst:e}");
            assert!(worst < 1e-6, "kv_int8={kv_int8}: batched embed decode != per-stream: {worst}");
        }
    }

    fn medium_cfg() -> QwenConfig {
        let mut c = QwenConfig::tiny();
        c.n_layers = 8;
        c.d_model = 256;
        c.head_dim = 64;
        c.n_heads = 8;
        c.n_kv_heads = 4;
        c.d_ff = 1024;
        c.vocab = 256;
        c
    }

    /// Throughput: N concurrent requests served with continuous batching vs run one
    /// at a time. Batched decode should give higher aggregate tokens/sec.
    ///   cargo test -p brain-qwen --lib serve_throughput -- --ignored --nocapture
    #[test]
    #[ignore]
    fn serve_throughput() {
        let cfg = medium_cfg();
        let (dm, nl) = (cfg.d_model, cfg.n_layers);
        let map = tiny_weights(&cfg);
        let n_req = 8usize;
        let max_new = 48usize;
        let prompts: Vec<Vec<u32>> = (0..n_req).map(|i| vec![(i as u32 % 200) + 1, 5, 3, 9, 2]).collect();

        // Sequential: one request at a time (fresh reuse of one engine's pool).
        let mut eng_seq = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 16, 512, n_req as u32, 16, 32, false, false);
        let t0 = std::time::Instant::now();
        for p in &prompts {
            eng_seq.generate_greedy(std::slice::from_ref(p), max_new, None);
        }
        let seq_s = t0.elapsed().as_secs_f64();

        // Continuous batching: all requests admitted + decoded together.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 16, 512, n_req as u32, 16, 32, false, false);
        let mut sched = Scheduler::new(eng, n_req);
        for p in &prompts {
            sched.submit(Request { prompt: p.clone(), max_new, eos: None });
        }
        let t1 = std::time::Instant::now();
        let out = sched.run();
        let batch_s = t1.elapsed().as_secs_f64();

        let total_tokens = (n_req * max_new) as f64;
        assert_eq!(out.len(), n_req);
        println!(
            "serve throughput ({n_req} reqs x {max_new} tok, d{dm} L{nl}): sequential {:.1} tok/s ({seq_s:.2}s) | continuous-batched {:.1} tok/s ({batch_s:.2}s) | {:.1}x",
            total_tokens / seq_s,
            total_tokens / batch_s,
            seq_s / batch_s,
        );
    }

    /// M2.4: `run_batched_steps`'s own attention dispatch, inspected via
    /// `Step::meta().kernel` rather than any output value - the fused
    /// kernel's own numerical agreement with the triad is already gated at
    /// the kernel level (`model::paged::flash_tests::
    /// paged_flash_prefill_matches_batched_triad`); this test's whole job is
    /// proving `run_batched_steps` actually PICKS one path or the other,
    /// matching `Op::PagedAttentionFused`'s own candidates (see
    /// `gpu_core::select`'s own `paged_attention_fused_only_offers_the_
    /// fused_kernel_at_causal_chunk_f32`): causal-chunk (prefill) at fp32 KV
    /// gets the ONE fused dispatch and none of the triad's three; every
    /// other regime (decode, or causal-chunk under int8 KV, which
    /// `paged_flash_prefill` has no tier for yet) keeps the triad and never
    /// dispatches the fused kernel at all.
    #[allow(clippy::type_complexity)]
    fn causal_chunk_metadata(cc: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let bs = 4u32;
        let mut alloc = BlockAllocator::new(64, bs);
        let mut table = BlockTable::new();
        table.reserve(cc, &mut alloc).expect("KV pool exhausted");
        let mbt = 8usize;
        let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u32; cc as usize * mbt];
        for i in 0..cc {
            let (bl, off) = table.locate(i, bs);
            positions.push(i);
            seqlens.push(i + 1); // causal: query i attends 0..=i
            blocks.push(bl);
            offsets.push(off);
            for (lb, &phys) in table.blocks().iter().enumerate() {
                bt[i as usize * mbt + lb] = phys;
            }
        }
        (positions, seqlens, blocks, offsets, bt)
    }

    #[test]
    fn causal_chunk_fp32_kv_dispatches_the_fused_kernel_not_the_triad() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 4, 8, 32, false, false);
        let cc = 6u32;
        let (positions, seqlens, blocks, offsets, bt) = causal_chunk_metadata(cc);
        let tokens: Vec<u32> = (0..cc).map(|i| i % cfg.vocab).collect();
        let (steps, _) = eng.run_batched_steps(cc, Input::Tokens(&tokens), &positions, &seqlens, &blocks, &offsets, &bt, true);
        let kinds: Vec<usize> = steps.iter().filter_map(|s| s.meta().map(|m| m.kernel)).collect();
        let fused = kinds.iter().filter(|&&k| k == PAGED_FLASH_PREFILL).count();
        let triad = kinds.iter().filter(|&&k| k == SCORES_B || k == SCORES_B_WG || k == SOFTMAX_B || k == APPLY_B).count();
        assert_eq!(fused, cfg.n_layers as usize, "one fused dispatch per layer, causal-chunk fp32 KV");
        assert_eq!(triad, 0, "the triad must not run when the fused kernel does");
    }

    #[test]
    fn decode_regime_never_dispatches_the_fused_kernel() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 4, 8, 32, false, false);
        // Decode shape: independent single-token rows, NOT a causal chunk -
        // `causal_chunk_metadata`'s own construction happens to build valid
        // decode metadata too (one row per position, its own block), which is
        // exactly the point: only the CALL SITE's own `causal_chunk` flag
        // decides the regime, not anything inferable from the shape alone
        // (M2.1's own finding: the fused decode kernel never wins here).
        let cc = 3u32;
        let (positions, seqlens, blocks, offsets, bt) = causal_chunk_metadata(cc);
        let tokens: Vec<u32> = (0..cc).map(|i| i % cfg.vocab).collect();
        let (steps, _) = eng.run_batched_steps(cc, Input::Tokens(&tokens), &positions, &seqlens, &blocks, &offsets, &bt, false);
        let kinds: Vec<usize> = steps.iter().filter_map(|s| s.meta().map(|m| m.kernel)).collect();
        assert_eq!(kinds.iter().filter(|&&k| k == PAGED_FLASH_PREFILL).count(), 0, "decode must never pick the fused kernel");
        let triad = kinds.iter().filter(|&&k| k == SCORES_B || k == SCORES_B_WG || k == SOFTMAX_B || k == APPLY_B).count();
        assert!(triad > 0, "decode must still run the triad");
    }

    #[test]
    fn causal_chunk_int8_kv_still_uses_the_triad() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 4, 8, 32, true, false);
        let cc = 6u32;
        let (positions, seqlens, blocks, offsets, bt) = causal_chunk_metadata(cc);
        let tokens: Vec<u32> = (0..cc).map(|i| i % cfg.vocab).collect();
        let (steps, _) = eng.run_batched_steps(cc, Input::Tokens(&tokens), &positions, &seqlens, &blocks, &offsets, &bt, true);
        let kinds: Vec<usize> = steps.iter().filter_map(|s| s.meta().map(|m| m.kernel)).collect();
        assert_eq!(kinds.iter().filter(|&&k| k == PAGED_FLASH_PREFILL).count(), 0, "paged_flash_prefill has no int8-KV tier yet");
        assert!(kinds.contains(&SCORES_I8), "causal-chunk int8 KV must still run the int8 triad");
    }

    /// A config whose `tok.weight` is deliberately sized past `wgpu`'s
    /// `max_storage_buffer_binding_size` clamp (`i32::MAX` = 2047 MiB, on
    /// every `wgpu` backend regardless of the card's actual VRAM) - the exact
    /// shape of the real bug: Qwen3-8B's real `[151936, 4096]` embedding
    /// table is ~2.32 GiB. `140000 * 4096 * 4 B` = 2,293,760,000 B (~2.14
    /// GiB) safely exceeds the ~2,147,483,644 B cap with margin, at the
    /// smallest vocab/d_model split that still does (the byte count is fixed
    /// by the product, not by how it is split - shrinking either dimension
    /// only grows the other). Per-layer dims stay tiny so everything else
    /// this config allocates is cheap.
    fn oversized_vocab_cfg() -> QwenConfig {
        QwenConfig {
            vocab: 140_000,
            block_size: 64,
            n_layers: 1,
            d_model: 4096,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            d_ff: 256,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 64,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// [`tiny_weights`], but with a cheap deterministic fill instead of
    /// per-element `Rng::next_gaussian` draws for every param - at
    /// `oversized_vocab_cfg`'s real scale (`tok.weight` alone is ~573M
    /// elements), the RNG cost would dwarf the dispatch this test actually
    /// exercises. The fill is still non-constant (so a tiling bug that reads
    /// the wrong rows would show up as a mismatch), just cheap to compute.
    fn oversized_vocab_weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|i| (i % 997) as f32 * 1e-3 - 0.5).collect()
            };
            map.insert(name, v);
        }
        map
    }

    /// RED before the fix (`Engine::embed_tiled`, wired into `Self::
    /// batched_tape` in place of a plain `EMBED` dispatch against the whole
    /// `tok.weight` buffer): `wgpu::Device::create_bind_group` panics with
    /// "Buffer binding ... exceeds `max_*_buffer_binding_size` limit" the
    /// moment a forward tries to bind a table this large as one storage
    /// binding - exactly the crash `brain qwen3 serve` hit on Qwen3-8B's real
    /// vocab. GREEN after: the SAME dispatch, now several `step_sliced`
    /// bindings each within the limit.
    ///
    /// Deliberately calls `Self::batched_tape` directly and submits/reads
    /// only `sc.res[0]` (the post-embed residual) - never `Self::head_steps`
    /// - to isolate the embedding gather under test here from the SEPARATE,
    /// pre-existing fp32 head/logits binding-size limit `Self::head_steps`
    /// has never tiled (out of scope for this fix - see this crate's own
    /// `serve.rs` module doc / the task that introduced this test for the
    /// scope line).
    ///
    /// wgpu-only: the 2047 MiB clamp is a `wgpu` backend fact (present on
    /// every `wgpu` target, including this one on Vulkan), not a property of
    /// the CPU JIT backend, which has no such binding cap to reproduce.
    #[test]
    fn embed_step_survives_a_vocab_table_that_exceeds_one_storage_binding() {
        // `Gpu::new`, not the shared `testgpu::dev` pool: this test's buffers
        // run into the GiB range and its dispatches take real wall time, and
        // `cargo test` runs the suite in parallel threads - sharing the same
        // pooled device's command-encoder state with a concurrently-running
        // test raced (measured: intermittent, unrelated `min_storage_buffer_
        // offset_alignment` failures in OTHER tests appeared only when this
        // one ran alongside them, never alone and never on baseline code). An
        // independent device sidesteps that without touching the pool.
        let gpu = Gpu::new(PIPELINES);
        if gpu.kind() != "wgpu" {
            brain_testutil::skip_unavailable(&format!(
                "embed binding-size tiling: needs a real wgpu 2047 MiB storage-binding clamp, current backend is {}",
                gpu.kind()
            ));
            return;
        }
        let cfg = oversized_vocab_cfg();
        let table_bytes = cfg.vocab as u64 * cfg.d_model as u64 * 4;
        assert!(table_bytes > 2_147_483_644, "test setup: tok.weight ({table_bytes} B) must exceed wgpu's binding cap");
        let map = oversized_vocab_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu, cfg.clone(), &map, 4, 8, 1, 4, 8, false, false);

        let tokens = [0u32, 55_555, cfg.vocab - 1];
        eng.gpu.write(&eng.sc.tok_buf, &tokens);
        let steps = eng.batched_tape(tokens.len() as u32, Input::Tokens(&tokens), false);
        eng.gpu.submit(&[], &steps);

        let d = cfg.d_model as usize;
        let got = eng.gpu.read(&eng.sc.res[0], tokens.len() * d);
        for (row, &tok) in tokens.iter().enumerate() {
            let want: Vec<f32> = (0..d as u32).map(|c| ((tok * cfg.d_model + c) % 997) as f32 * 1e-3 - 0.5).collect();
            assert_eq!(&got[row * d..(row + 1) * d], want.as_slice(), "embedding row for token {tok}");
        }
    }

    /// The tiled embedding gather (the same `EMBED_TILE` kernel and
    /// `step_sliced` offset convention `Engine::embed_tiled` dispatches) is
    /// bit-identical to the plain, untiled `EMBED` kernel it replaces, at a
    /// vocab safely under any real binding cap - so the untiled dispatch is a
    /// valid oracle here (unlike the test above, whose whole point is that
    /// the untiled dispatch panics at real scale).
    ///
    /// Deliberately does NOT force tiling via `BRAIN_TILE_BUDGET_WORDS`
    /// (`block::tile_budget_words_for`'s override, already exercised by
    /// `block::tests::a_small_budget_still_forces_the_tiled_path` and
    /// `crates/t5/tests/smoke.rs`): that env var is process-global, and after
    /// this fix `Engine::vocab_tiles` reads it on EVERY forward pass, not
    /// just tiling-focused tests - `cargo test`'s parallel runner mutating it
    /// here would race every OTHER concurrently-running test in this module
    /// that builds an `Engine` and embeds a token, not just ones that opted
    /// into the convention via `brain_testutil::env_lock`. Three tile
    /// boundaries are chosen directly instead (this is a kernel/dispatch
    /// correctness check, not a re-test of the budget MATH - `block::
    /// vocab_tiles_on`'s own tests already cover that, including the exact
    /// real Qwen3-8B shape).
    #[test]
    fn embed_tiled_matches_the_plain_embed_kernel() {
        // Vocab safely under any real binding cap (40 * 8 * 4 B = 1280 B) -
        // the untiled `EMBED` dispatch below is a valid oracle only because
        // of that.
        let cfg = QwenConfig { vocab: 40, d_model: 8, ..QwenConfig::tiny() };
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 8, 8, 4, 8, false, false);

        let d = cfg.d_model;
        let dw = d as u64;
        // Three uneven tiles over the 40-row vocab, boundaries at 8 and 24 -
        // multiples of 8 rows, so `v0*d_model` (words) is a multiple of 64
        // (`min_storage_buffer_offset_alignment` is 256 B = 64 f32 words on
        // every adapter this repo has met, same constant `HEAD_TILE_ALIGN`
        // hardcodes). Real callers get this for free (`Self::embed_tiled`'s
        // own doc: `v0*d_model` is already a large multiple of 64 for any
        // real `d_model`); a toy `d_model=8` test has to pick boundaries
        // that keep it true rather than inherit it.
        let manual_tiles: [(u32, u32); 3] = [(0, 8), (8, 16), (24, 16)];

        // token 0, tile0's last id, tile1's first/last id, tile2's first id,
        // vocab-1.
        let ids = [0u32, 7, 8, 23, 24, 39];
        let n = ids.len() as u32;
        eng.gpu.write(&eng.sc.tok_buf, &ids);

        let out_tiled = eng.gpu.storage(n as u64 * dw);
        let tiled_steps: Vec<Step> = manual_tiles
            .iter()
            .map(|&(v0, cnt)| {
                eng.gpu.step_sliced(
                    EMBED_TILE,
                    &[&eng.sc.tok_buf, eng.ps.w("tok.weight"), &out_tiled],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[d, n, v0, cnt],
                    n * d,
                )
            })
            .collect();
        eng.gpu.submit(&[], &tiled_steps);
        let tiled = eng.gpu.read(&out_tiled, (n * d) as usize);

        let out_plain = eng.gpu.storage(n as u64 * dw);
        let plain_step = eng.gpu.step(EMBED, &[&eng.sc.tok_buf, eng.ps.w("tok.weight"), &out_plain], &[d, n], d * n);
        eng.gpu.submit(&[], &[plain_step]);
        let plain = eng.gpu.read(&out_plain, (n * d) as usize);

        assert_eq!(tiled, plain, "tiled embed must be bit-identical to the untiled EMBED kernel");
    }
}
