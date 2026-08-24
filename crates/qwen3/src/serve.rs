// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concurrent serving engine for the Qwen3 decoder: a **paged** KV cache shared by
//! many sequences + **batched** decode that advances every active sequence by one
//! token per iteration. Each sequence's KV grows a block at a time from a shared
//! pool (no per-sequence worst-case reservation), and one batched forward serves
//! the whole running set — so more sequences stay resident and decode together.
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
use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use paramstore::ParamStore;

use crate::config::QwenConfig;

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
// (clipped) one — the ONLY i8 append since the unclipped twin was deleted
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
// per-token x per-channel dequant scales — the tile GEMM for prefill shapes,
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
// `DECODE_REGIME_MAX_ROWS` ran one thread per output element — while the
// batched forward next door dispatched this same kernel at ~80x the rate.
const MATMUL_REG3: usize = 26;
// Coalesced paged scores: one workgroup per score, lanes split the head_dim
// reduction. Same Params and same output as SCORES_B; selected on the queried
// `workgroup_reductions`, since it carries a barrier the CPU JIT gates on.
const SCORES_B_WG: usize = 27;
/// Scores one `paged_decode_scores_wg` workgroup owns — `64 / LPS` in the
/// kernel. Must match, or the dispatch covers the wrong number of scores.
const SCORES_WG_PER_GROUP: u32 = 16;
// Split-K forward GEMM + its fold, for the skinny-M shapes a served step is made
// of. `matmul_reg3`'s tile grid is ceil(m/128)*ceil(n/128) and does not grow
// with k, so at m=128 it launches 16 workgroups on a 30-SM card (11% of peak,
// and 47% of a served step). See `matmul_reg3_splitk.wgsl` for the measured
// occupancy curve and the arithmetic that says splitting pays at THIS shape.
const MATMUL_REG3_SPLITK: usize = 28;
const SPLITK_REDUCE: usize = 29;

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
    use gpu_core::select::Dtype as OpsDtype;
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = vec![
            ("matmul", kernels::MATMUL),
            ("matmul_gemv", kernels::MATMUL_GEMV),
            ("matmul_reg2", kernels::MATMUL_REG2),
            ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
            ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
            ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
            ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
            ("max_abs_row", kernels::MAX_ABS_ROW),
            ("quant_pack", kernels::QUANT_PACK),
            ("embed", kernels::EMBED),
            ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
            ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
            ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
            ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
            ("matmul_dx", kernels::MATMUL_DX),
            ("matmul_dw", kernels::MATMUL_DW),
        ];
        // `Ops::REQUIRED_KERNELS` also demands the bf16/f16 storage-tier
        // variants (B4/B5/B8/B9/B10) even though this crate never builds a
        // `Weight::BF16`/`Weight::F16` and has its own KV-cache mechanism
        // (never dispatches the generic `paged_*_batched`/`embed`/
        // `moe_linear_gated`/`matmul_dx`/`matmul_dw` family through this
        // façade at all) - see `Ops::new`'s own doc comment ("every model
        // that builds an `Ops` must register the full façade kernel set, not
        // just the tiers it plans to use"). Compiled, never dispatched.
        // Mirrors `model::ops`'s own test-only `kernel_list()` and
        // `qwen3::model::pipelines()` exactly (see those - the same list,
        // kept in sync by hand since there is no single shared source all
        // three can pull from).
        for dt in [OpsDtype::BF16, OpsDtype::F16] {
            v.push(kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("embed", kernels::EMBED, "emb", dt).unwrap());
            v.push(kernels::template::dtype_variant("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", dt).unwrap());
        }
        v.push(
            kernels::template::dtype_variant_store(
                "paged_kv_append_batched_word",
                kernels::PAGED_KV_APPEND_BATCHED_WORD,
                "pool",
                OpsDtype::BF16,
            )
            .unwrap(),
        );
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_scores_batched",
                kernels::PAGED_DECODE_SCORES_BATCHED,
                "pool_k",
                OpsDtype::BF16,
            )
            .unwrap(),
        );
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_apply_batched",
                kernels::PAGED_DECODE_APPLY_BATCHED,
                "pool_v",
                OpsDtype::BF16,
            )
            .unwrap(),
        );
        v.push(kernels::template::dtype_variant("matmul_dx", kernels::MATMUL_DX, "w", OpsDtype::BF16).unwrap());
        v
    })
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
/// a constant — every shape's optimum landed on ~288 workgroups — which is why
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
        // unused on the forward decode path:
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: ROPE_PAGED,
        rope_bwd: ROPE_PAGED,
        // This engine never calls `block::gqa_fwd`: prefill and decode share
        // the PAGED attention kernels (`paged_decode_*`), so the batched causal
        // trio was registered — three pipelines compiled at every Engine
        // build — and never dispatched. Placeholders, same convention as the
        // backward ids below, so nothing reads a live index for a path that
        // does not exist.
        gqa_scores: 0,
        gqa_apply: 0,
        attn_softmax: 0,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_da: SILU_MUL,
        silu_db: SILU_MUL,
    }
}

// The decode-regime boundaries (max rows, argmax split vocab) live in the
// shared selection policy — `gpu_core::select` — not here: which kernel runs
// for a shape on a device is the selector's single job.

/// Chunks per row for the two-stage argmax; 256 threads per row saturates the
/// reduction without a large partial buffer (256*2 f32 per row).
const ARGMAX_CHUNKS: u32 = 256;

/// The largest `k` a real (non-greedy) sampling decode step will request —
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

/// Fault injection (G): `brain perf faults` arms a fault, the next pass
/// through its check point fires it. Feature-gated — a build without
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
pub enum Input<'a> {
    Tokens(&'a [u32]),
    Embeds(&'a [f32]),
    /// Token ids already resident in the engine's `tok_buf` — the on-device
    /// decode window (A4): `decode_feed` wrote them from the previous step's
    /// argmax, and `decode_advance` already advanced the paged metadata, so
    /// the forward performs NO host writes at all.
    Resident,
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

/// Per-(K or V)-buffer word counts for the paged KV pool at this sizing — the
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

/// Device bytes the paged KV pool costs at this sizing: K + V pools (packed
/// int8 4/`u32` + a fp32 scale per `(token slot, kv-head)`, or plain fp32) for
/// every layer.
///
/// The exact ratio `fp32 / int8` is `4·head_dim / (head_dim + 4)` — 3.8788x
/// at Qwen3's `head_dim=128`, but a DIFFERENT number at any other `head_dim`
/// (2.6667x at `QwenConfig::tiny()`'s `head_dim=8`) — see
/// `kv_pool_bytes_identity_holds_at_the_real_shape`, which pins both.
pub fn kv_pool_bytes(cfg: &QwenConfig, block_size: u32, num_blocks: u32, kv_int8: bool) -> u64 {
    let (pool_words, scale_words) = kv_pool_words(cfg, block_size, num_blocks, kv_int8);
    let n_layers = cfg.n_layers as u64;
    n_layers * 2 * (pool_words + scale_words) * 4 // K + V, every layer, 4 bytes/word
}

/// Whether `cfg` can take int8 KV at all: the append kernels pack 4 int8
/// lanes into one `u32`, so a packed word must stay within one head (else its
/// lanes would span two heads' scales) — `head_dim % 4 == 0`. Every shipped
/// Qwen3 config (`head_dim` 128) and `QwenConfig::tiny()` (`head_dim` 8)
/// satisfy this; an imported HF config with an unusual `head_dim` might not.
///
/// The three DEFAULT-selecting call sites (`QwenResident::activate`,
/// `qwen_cli::serve`, the perf `SynthSpec` builders) call this FIRST and
/// degrade to fp32 with a printed reason when it is `false` — an explicit
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
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
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
    /// The device's capabilities, read once at build — the selector's input.
    caps: DeviceCaps,
    /// Which kernel variant runs for each (op, shape) — the shared decode-regime
    /// policy, memoised per distinct shape.
    selector: CachedSelector<DefaultSelector>,
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
    /// optimisation — decode never touches it.
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
    /// What [`kv_pool_bytes`] computed for this sizing — recorded once at
    /// construction (not re-derived by the accessor) so it can never drift
    /// from what `pool_k`/`pool_v`/`scales_k`/`scales_v` actually allocated.
    kv_pool_bytes: u64,
    /// `Some` uploads real calibrated ceilings into `clip_k`/`clip_v`; `None`
    /// (the default) keeps the f32::MAX sentinel there, which the append
    /// kernel's contract documents as bit-identical to the deleted unclipped
    /// twin (audit F42) — see [`Engine::set_kv_calib`].
    kv_calib: Option<model::kvcalib::KvCalib>,
    /// Per-layer `[n_kv]` clip-ceiling upload buffers (allocated whenever
    /// `kv_int8`; MAX-sentinel-filled until a real calibration is installed).
    clip_k: Vec<DeviceBuffer>,
    clip_v: Vec<DeviceBuffer>,
    /// Int8 WEIGHT path (A0): every linear this engine dispatches - the 7
    /// per-layer projections (`blocks.<l>.<leaf>`) plus the LM head
    /// (`cfg.head_weight()`) - as a `model::ops::Weight` (B7), packed 4/u32
    /// (~4x fewer weight bytes in the bandwidth-bound decode regime) when
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
    /// `(row bucket, n, k)` — tuned once at build on THIS device (persisted
    /// per adapter), so the hot path never measures. Empty on fp32 engines.
    tuned_i8: HashMap<(u32, u32, u32), KernelVariant>,
    sc: Scratch,
    /// `[vocab, d]` tied/untied head, kept on the host for the prefill path
    /// (applied once per request) and for callers that need full logits.
    head: Vec<f32>,
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
}

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
    /// engine costs pipeline compilation only — never a second full device
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
        // PackedInt8 gate). Elsewhere — the CPU JIT — fp32 weights stay, and
        // the fallback is said out loud rather than silently absorbed.
        let caps = gpu.caps();
        // The gate is the CAPABILITY, not the selector's head: which int8
        // variant is best at some shape is a tuning question, but whether the
        // packed-dot kernels execute at all is numeric.int8_dot.
        let w8_on = weights_int8 && caps.numeric.int8_dot;
        if weights_int8 && !w8_on {
            eprintln!("serve: int8 weights requested but this device has no packed-int8 path; using fp32 weights");
        }
        // The 7 per-layer linears live in the int8 bank when it is on — loading
        // them into the fp32 ParamStore as well would keep both copies resident
        // and forfeit the memory the quantisation buys.
        let roles = decoder_param_list(&cfg)
            .into_iter()
            .filter(|(n, _)| !(w8_on && crate::q8::Q8::is_i8_linear(n)))
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
        // it for 8 silently refused the wide shapes — only `wk`/`wv` fitted, so
        // 56 of 196 GEMMs split and the rest kept the starved kernel.
        let splitk_cap = (widest_m * widest_n * SPLITK_MAX_SLICES as u64).min(SPLITK_SCRATCH_WORDS);
        let (splitk_part, splitk_cap) = if gpu.caps().workgroup_reductions && splitk_cap > 0 {
            (Some(gpu.storage(splitk_cap)), Some(splitk_cap))
        } else {
            (None, None)
        };
        let nh = cfg.n_heads as u64;
        // scores/probs hold decode [rows,nh,cap] OR prefill causal [nh,N,N].
        let bcap = (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh);
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
        let sc = Scratch {
            res,
            xn1: st(b * d),
            q_pre: st(b * hq),
            q: st(b * hq),
            k_pre: st(b * hkv),
            k: st(b * hkv),
            v: st(b * hkv),
            ctx: st(b * hq),
            xmid: st(b * d),
            xn2: st(b * d),
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
        // (capability-aware quantize+upload) - this engine keeps its OWN
        // tuned dispatch (`Self::linear`/`Self::mm8`/`Self::tune_i8`, below)
        // rather than routing through `Ops::matmul`, so - unlike `qwen3::
        // model::Qwen`'s `self.ops` (see that crate's `pipelines()` doc
        // comment) - index-space compatibility with `self.gpu` is NOT
        // required: `Weight::upload` only touches buffers, it never builds a
        // `Step` this engine would submit.
        //
        // **Why this engine does NOT route dispatch through `Ops::matmul`,
        // unlike `qwen3::model::Qwen` (B7).** `Ops::matmul` always resolves
        // its kernel via a FIXED internal `CachedSelector<DefaultSelector>` -
        // there is no way for a caller to inject a different one. This
        // engine's `tuned_i8` (below) is a REAL, per-device MEASURED
        // selector (`Self::tune_i8`/`AutoTuner`/`FileTuneStore`) that the
        // serving-performance regression gate this phase must not regress
        // directly depends on - routing through `Ops::matmul` would silently
        // discard it in favour of the static policy.
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
            selector: CachedSelector::new(DefaultSelector),
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
            kv_calib: None,
            clip_k,
            clip_v,
            lin_weights,
            i8_scratch,
            tuned_i8,
            sc,
            head,
            logits_dev,
            argmax_dev,
            argmax_part_dev,
            topk_vals_dev,
            topk_idx_dev,
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
    /// calibration — the append dispatch falls back to the plain online-
    /// absmax kernel. A no-op on a fp32-KV engine (`kv_int8: false`): there
    /// is nothing to clip, since `run_batched_submit`'s int8 branch (the only
    /// place `kv_calib` is read) never runs. Printed loudly rather than
    /// silent, because a caller installing a table it then never sees take
    /// effect is exactly the kind of no-op AGENTS.md calls out (a gate/config
    /// that never runs is worse than none) — [`kv_calibrated`] reflects the
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
            // tables with the f32::MAX sentinel — bit-identical to the deleted
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
    /// `weights_int8`, this is exactly what the constructor was built with —
    /// int8 KV has no capability gate to fall back from).
    pub fn kv_int8(&self) -> bool {
        self.kv_int8
    }

    /// Device bytes the KV pool costs at this engine's sizing — recorded once
    /// at construction from [`kv_pool_bytes`], never re-derived, so this can
    /// never drift from what `pool_k`/`pool_v`/`scales_k`/`scales_v` actually
    /// allocated.
    pub fn kv_pool_bytes(&self) -> u64 {
        self.kv_pool_bytes
    }

    /// The pool's total theoretical cached-token capacity (`num_blocks *
    /// block_size`), independent of dtype — the number that answers "how
    /// many tokens could this pool ever hold at once", as opposed to
    /// [`kv_pool_bytes`] answering "at what memory cost".
    pub fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * self.block_size as u64
    }

    /// Whether the installed KV clip table is a real, binding calibration
    /// that is ACTUALLY DISPATCHED (not `None`, not `KvCalib::disabled`, and
    /// the engine is int8 — the clip binding is only read by the i8 append on
    /// the int8 branch of `run_batched_submit`). A table
    /// installed on an fp32 engine is `Some` in `self.kv_calib` but never
    /// read by anything, so this must say `false` for it or it would claim a
    /// calibration is binding when it provably is not.
    pub fn kv_calibrated(&self) -> bool {
        self.kv_int8 && self.kv_calib.is_some()
    }

    /// The device this engine runs on — the parent handle for building more
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
        self.run_batched(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Advance every sequence by one token and return the **greedy next token**
    /// per row, with the LM head applied on the device (see
    /// [`Engine::run_batched_greedy`]).
    pub(crate) fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched_greedy(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Advance every sequence by one token from a ready-made embedding per sequence
    /// (`[bsz, d_model]`) — the tts Talker multi-stream path: concurrent voice
    /// streams decode together on the shared paged pool.
    pub fn forward_batched_embed(&mut self, tables: &mut [&mut BlockTable], embeds: &[f32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        assert_eq!(embeds.len(), bsz as usize * self.cfg.d_model as usize);
        self.run_batched(bsz, Input::Embeds(embeds), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Run one batched forward over `bsz` rows given fully-computed metadata:
    /// `positions[i]` RoPE position, `seqlens[i]` the cached length row i attends
    /// (row i's query attends `j < seqlens[i]` — set to start+i+1 for causal
    /// prefill), `(blocks[i], offsets[i])` the pool slot to write row i's K/V, and
    /// `bt` the per-row block tables (`bsz * max_blocks_per_seq`). Serves decode
    /// (one new token per sequence) and prefill chunks alike.
    #[allow(clippy::too_many_arguments)]
    fn run_batched(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<f32> {
        let b = self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt);
        self.gpu.read(&self.sc.xn_final, (b * self.cfg.d_model) as usize)
    }

    /// The transformer body only: records and submits every stage, leaving the
    /// final norm in `sc.xn_final` **without reading it back**. Submits are
    /// accumulated lazily and flushed on the next read, so a caller that appends
    /// more device work (the greedy head) still pays one flush per step rather
    /// than two. Returns the row count.
    // qwen3-serve-manual-gemm-dispatch BEGIN (B7, `no_kernel_names.rs`'s own
    // allow-list) - this engine's own tuned fp32/int8 GEMM selection, kept
    // OFF the `model::ops::Ops` façade deliberately: `Ops::matmul` always
    // resolves through a FIXED internal `CachedSelector<DefaultSelector>`
    // with no way to inject a different one, but `tuned_i8` (below) is a
    // REAL, per-device MEASURED selector (`Self::tune_i8`/`AutoTuner`/
    // `FileTuneStore`) the qwen-serving-perf-gate directly depends on -
    // routing through `Ops::matmul` would silently discard it. See
    // `no_kernel_names.rs`'s own module doc for exactly what this allow-lists
    // and why.
    /// `out = x @ W^T`, choosing the decode-regime GEMV (one workgroup per
    /// output column, W streamed once across all rows) when the selector says
    /// the shape is in that regime. Same contract, same result.
    fn mm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        let (kind, threads) = block::gemm_variant(self.gemm_tier(), m, n);
        self.gpu.step(kind, &[x, w, out], &[m, k, n], threads)
    }

    /// [`Self::mm`], but free to emit MORE than one dispatch: the split-K GEMM
    /// needs a fold after it.
    ///
    /// Split-K only when the tile grid is too small to fill the device — the
    /// same rule and the same 288-workgroup target `vae::blocks` measured for
    /// `matmul_dw_reg_splitk`, which is the identical defect on the backward.
    /// `slices = 1` means the plain kernel, so a shape that already fills the
    /// card is untouched.
    fn mm_into(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
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
    /// not fit the scratch — the last one keeps this from silently allocating.
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

    /// The fp32 GEMM tier for this device — the SAME rule `flux1`, `flux2` and
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
    /// the engine's real buffers — REPS dispatches per timing so submit/poll
    /// overhead amortises — and the winner persists per adapter + kernel
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
    /// (the per-element kernel runs `rows` threads — 8 threads on a 3840-core
    /// card at batch 8, measured at 16.6% of decode time).
    fn rms(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, d: u32, rows: u32) -> Step {
        let g = &self.gpu;
        let shape = OpShape { m: rows, n: d, k: 0, dtype: Dtype::F32 };
        match self.selector.select(Op::RmsNorm, shape, &self.caps) {
            KernelVariant::WorkgroupPerOutput => g.step(RMSNORM_ROWS, &[x, w, out], &[d, rows, gpu_core::f(1e-6)], rows * 64),
            _ => g.step(RMSNORM, &[x, w, out], &[d, rows], rows),
        }
    }
    // qwen3-serve-manual-gemm-dispatch END

    #[allow(clippy::too_many_arguments)]
    fn run_batched_steps(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> (Vec<Step>, u32) {
        #[cfg(feature = "fault-injection")]
        if fault::take_kernel_failure() {
            panic!("injected fault: kernel dispatch failure");
        }
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let half = hd / 2;
        let bs = self.block_size;
        let cap = self.cap;
        let mbt = self.max_blocks_per_seq;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let theta = c.rope_theta;
        let g = &self.gpu;
        // Resident mode (A4): every input — token ids AND paged metadata — was
        // produced on the device by `decode_feed`/`decode_advance`, so writing
        // host copies here would both be wrong (stale) and force a flush.
        if !matches!(input, Input::Resident) {
            g.write(&self.sc.pos_buf, positions);
            g.write(&self.sc.seqlen_buf, seqlens);
            g.write(&self.sc.blk_buf, blocks);
            g.write(&self.sc.off_buf, offsets);
            g.write(&self.sc.bt_buf, bt);
        }
        let kids = ids();
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let b = bsz;
        let mut s: Vec<Step> = Vec::new();
        match input {
            Input::Tokens(t) => {
                g.write(&sc.tok_buf, t);
                s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
            }
            Input::Resident => {
                s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
            }
            Input::Embeds(e) => {
                g.write(&sc.res[0], bytemuck::cast_slice(e));
            }
        }
        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(self.rms(&sc.res[l], w(&p("ln1.weight")), &sc.xn1, d, b));
            // One activation quant per distinct input (B7: `Self::
            // quant_once`, a no-op on an all-fp32 engine), shared by every
            // linear reading it (xn1 -> q/k/v).
            self.quant_once(&mut s, &sc.xn1, d, b);
            self.linear(&mut s, &self.lin_weights[&p("attn.wq.weight")], &sc.xn1, &sc.q_pre, b);
            self.linear(&mut s, &self.lin_weights[&p("attn.wk.weight")], &sc.xn1, &sc.k_pre, b);
            self.linear(&mut s, &self.lin_weights[&p("attn.wv.weight")], &sc.xn1, &sc.v, b);
            // QK-norm goes through `self.rms` like every other norm in this
            // tape. It used to call `block::rmsnorm_fwd` directly — the
            // per-element kernel, one thread per row — which is the coalescing
            // bug measured at 19.4x for exactly
            // this op, and it was the only norm here bypassing the selector.
            // QK-norm goes through `self.rms` like every other norm in this
            // tape. It used to call `block::rmsnorm_fwd` directly — the
            // per-element kernel, one thread per row — which is the coalescing
            // bug measured at 19.4x for exactly
            // this op, and it was the only norm here bypassing the selector.
            s.push(self.rms(&sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, b * nh));
            s.push(self.rms(&sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, b * nkv));
            s.push(g.step(ROPE_PAGED, &[&sc.q, &sc.pos_buf], &[b, nh, hd, hq, fb(theta)], b * nh * half));
            s.push(g.step(ROPE_PAGED, &[&sc.k, &sc.pos_buf], &[b, nkv, hd, hkv, fb(theta)], b * nkv * half));
            if self.kv_int8 {
                // ONE append kernel for both paths (audit F42): the clip
                // buffers hold either the calibrated ceilings or the
                // f32::MAX sentinel, which the kernel's contract documents
                // as bit-identical to the old unclipped twin.
                s.push(g.step(APPEND_I8_CLIPPED, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.clip_k[l], &self.pool_k[l], &self.scales_k[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(APPEND_I8_CLIPPED, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.clip_v[l], &self.pool_v[l], &self.scales_v[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(SCORES_I8, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_k[l], &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_I8, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_v[l], &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            } else {
                s.push(g.step(KV_APPEND_B, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l]], &[b, hkv, bs], b * hkv));
                s.push(g.step(KV_APPEND_B, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l]], &[b, hkv, bs], b * hkv));
                // One workgroup per score where the device runs workgroup
                // reductions: the per-element kernel's lanes are `kv_stride`
                // floats apart (4 KB at 0.6B), 8x read amplification, measured
                // at 12.2% of the bandwidth roof while taking 52.2% of a
                // served step.
                let (sk, st) = if self.caps.workgroup_reductions {
                    (SCORES_B_WG, b.saturating_mul(nh).saturating_mul(cap).div_ceil(SCORES_WG_PER_GROUP) * 64)
                } else {
                    (SCORES_B, b * nh * cap)
                };
                s.push(g.step(sk, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], st));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_B, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            }
            self.quant_once(&mut s, &sc.ctx, hq, b);
            self.linear(&mut s, &self.lin_weights[&p("attn.wo.weight")], &sc.ctx, &sc.proj, b);
            s.push(g.step(ADD2, &[&sc.res[l], &sc.proj, &sc.xmid], &[b * d], b * d));
            s.push(self.rms(&sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, b));
            self.quant_once(&mut s, &sc.xn2, d, b);
            self.linear(&mut s, &self.lin_weights[&p("mlp.gate.weight")], &sc.xn2, &sc.gate_pre, b);
            self.linear(&mut s, &self.lin_weights[&p("mlp.up.weight")], &sc.xn2, &sc.up, b);
            s.push(block::swiglu_fwd(g, &kids, &sc.gate_pre, &sc.up, &sc.h, b * ff));
            self.quant_once(&mut s, &sc.h, ff, b);
            self.linear(&mut s, &self.lin_weights[&p("mlp.down.weight")], &sc.h, &sc.mlp_out, b);
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &sc.res[l + 1]], &[b * d], b * d));
        }
        let last = c.n_layers as usize;
        s.push(self.rms(&sc.res[last], w("norm.weight"), &sc.xn_final, d, b));
        (s, b)
    }

    /// [`Self::run_batched_steps`] plus the submit. Split so a profiler can
    /// time the served tape per kernel kind without driving a whole request —
    /// the tape is rebuilt per step rather than recorded once, so there was
    /// nothing to hand `gpu_core::profile`.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_submit(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> u32 {
        let (s, b) = self.run_batched_steps(bsz, input, positions, seqlens, blocks, offsets, bt);
        self.gpu.submit(&[], &s);
        b
    }

    /// One batched decode step that returns the **greedy next token per row**,
    /// with the LM head evaluated on the device.
    ///
    /// The head is the largest single matmul in a small model (`vocab x d_model`
    /// = 16.4M MACs per row at vocab 32k). Applying it on the host, once per
    /// sequence per token, made decode host-bound: cost grew linearly with batch
    /// size while the GPU idled, so continuous batching stopped paying — measured
    /// at ~85% of each decode step, and throughput that gained only 1.4x from
    /// concurrency 1->16 before regressing.
    ///
    /// Here the hidden state never leaves the device: `matmul` produces
    /// `[bsz, vocab]` logits (parallel over every output element) and
    /// `argmax_row` reduces each row, so only `bsz` indices are read back
    /// instead of a `[bsz, vocab]` block.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_greedy(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<u32> {
        assert!(
            bsz <= self.max_batch,
            "greedy decode is sized for max_batch={} rows, got {bsz}",
            self.max_batch
        );
        self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt);
        self.greedy_from_hidden(bsz)
    }

    /// `argmax(xn_final @ head^T)` per row, entirely on the device.
    ///
    /// Two-stage reduction: `argmax_part` splits each row into
    /// [`ARGMAX_CHUNKS`] chunks reduced by independent threads, `argmax_final`
    /// folds the partials — `bsz * chunks` threads instead of `bsz`. The
    /// original single-thread-per-row `argmax_row` scanned 32k logits alone
    /// and was 10.3% of decode time; it remains registered as the small-vocab
    /// path and the reference the tests compare against.
    fn greedy_from_hidden(&self, bsz: u32) -> Vec<u32> {
        self.submit_greedy_head(bsz);
        // Indices come back as f32 (exact below 2^24, far above any vocabulary).
        self.gpu.read(&self.argmax_dev, bsz as usize).into_iter().map(|x| x as u32).collect()
    }

    /// Record the head steps that turn `sc.xn_final` into `[bsz, vocab]`
    /// `logits_dev` (int8 or fp32, whichever the engine holds) — shared by
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
    /// `argmax_dev`) WITHOUT reading back — the on-device decode window feeds
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
    /// from device work — see [`Self::submit_topk_head`]. `k` is clamped to
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
    /// host round-trip (the final readback in [`Self::topk_from_hidden`]) —
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
    /// (token id, logit) candidates instead of a single greedy token — the
    /// entry point a caller doing real (non-greedy) sampling uses in place of
    /// [`Self::forward_batched_greedy`]. `logits_dev` is mutated (masked) by
    /// the extraction, exactly as `argmax_dev` already is by the greedy path —
    /// callers never read either between decode steps.
    pub(crate) fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: u32) -> Vec<Vec<(u32, f32)>> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched_submit(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt);
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
        // Sub-step 0: host-fed, as today — but the argmax stays on the device.
        self.run_batched_submit(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt);
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
            self.run_batched_submit(bsz, Input::Resident, &[], &[], &[], &[], &[]);
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
        // bounds — the kernels are trusted (no per-access clamps on either
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
        // compute — the caller needs the LAST token's hidden state, which only
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
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt);
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
    /// the last — what teacher-forced held-out scoring needs (`qwen3::eval`),
    /// where every position's loss counts, not just the next-token
    /// prediction after the whole prompt.
    ///
    /// Deliberately bypasses the prefix cache (`self.prefix`) entirely: an
    /// eval pass scores a set of independent held-out samples, not a live
    /// conversation, so there is no shared prefix to exploit and no reason
    /// to let one sample's cache entries affect another's — full recompute
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
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt);
            out[start as usize * d..(start + cc) as usize * d].copy_from_slice(&hidden[..cc as usize * d]);
            start += cc;
        }
        out
    }

    /// Release up to `want` least-recently-used cache-only prefix blocks back
    /// to the pool — the admission path calls this when the pool is short.
    pub(crate) fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.prefix.evict(want, &mut self.alloc)
    }

    /// Prefix-cache effectiveness: `(tokens served from cache, tokens looked
    /// up, full blocks currently cached)`.
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        (self.prefix_hit_tokens, self.prefix_lookup_tokens, self.prefix.len())
    }

    /// Device-op accounting for this engine's handle (K) — what a benchmark
    /// records so submit/dispatch/readback cost is machine-readable. `None`
    /// where the backend does not count.
    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }

    /// Admission's one-time first-token logits (before the batched decode
    /// loop, which never reaches this — see `submit_greedy_head`/
    /// `forward_batched_topk`). `matvec_par` (rayon over `vocab` rows +
    /// AVX2/FMA per row) replaces a single-threaded scalar loop that measured
    /// hundreds of ms at the real 151936×1024 LM-head shape (see
    /// `hostmath::matvec`'s doc comment) — a real per-request stall this
    /// device-side-everything-else engine had left unfixed at exactly the one
    /// remaining host head.
    pub(crate) fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        model::hostmath::matvec_par(&self.head, hidden, v, d)
    }

    /// Blocks free in the pool — the capacity figure `brain perf kvcache` sizes
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
    /// from one sequence is a prefill chunk — the two share this tape, which is
    /// exactly why profiling it is worth doing.
    #[allow(clippy::too_many_arguments)]
    pub fn steps_for_profile(&self, bsz: u32, tokens: &[u32], positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<Step> {
        // `Input::Tokens`, NOT `Input::Resident`. Resident mode is the on-device
        // decode window: it deliberately performs no host writes because
        // `decode_feed`/`decode_advance` already produced the token ids AND the
        // paged metadata on the device. Using it from a profiler leaves
        // `seq_lens` at whatever was in the buffer — zero — so every attention
        // thread early-returns and the kernels appear to do almost no work.
        // That is exactly how `paged_decode_scores_batched` came to report
        // 7060 GB/s: the timing was right and the kernel really was that fast,
        // because it was not attending to anything.
        self.run_batched_steps(bsz, Input::Tokens(tokens), positions, seqlens, blocks, offsets, bt).0
    }

    /// Physical KV blocks currently free in the pool.
    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    /// The prefill chunk size this engine was built with — the unit the
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
    /// (`model::actstats`) — the design input for a calibrated INT8 KV scale
    /// (`brain qwen calib`, `crates/qwen3/src/calib.rs`).
    ///
    /// Offline-only, never called from the hot serving path (`run_batched_submit`
    /// stays untouched): this reads the pool directly with plain [`Gpu::read`]
    /// calls between prefills, which is fine for a one-shot calibration pass
    /// over a modest prompt set but is NOT the shape a per-request tap could
    /// use without a real perf cost.
    ///
    /// Needs an fp32-KV engine (`kv_int8: false`) — calibration wants the
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

        // Prefill every prompt first, keeping every table alive — the
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
    /// identical to plain greedy target decoding — the win is fewer (expensive)
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
            let hidden = self.run_batched(rows, Input::Tokens(&inputs), &positions, &seqlens, &blocks, &offsets, &bt);
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

    /// G3 (the scale-bug gate — lesson 2: cosine cannot see a dropped scale,
    /// and the int8 bug class IS a scale bug). ONE int8 engine, ONE prefill
    /// (the whole prompt fits in a single `max_prefill`-sized chunk, so it is
    /// exactly one forward pass): the ground truth for "what was quantized"
    /// is read straight out of the engine's OWN scratch (`sc.k`/`sc.v`), which
    /// still hold the last layer's post-RoPE K/V — the literal `src` the
    /// append kernel just packed — because nothing overwrites them after the
    /// final layer's dispatch. This deliberately avoids comparing against a
    /// SEPARATE fp32 engine: two independently-built engines can select
    /// different autotuned kernel variants for the identical (op, shape) and
    /// differ by GPU floating-point noise well under any real scale bug but
    /// well above `assert_eq!` — the oracle must come from the same
    /// computation being checked, not a second one hoped to agree with it.
    ///
    /// Per `(token, kv-head)`, every element: the scale is EXACTLY
    /// `absmax/127` (or `1.0` when `absmax==0`), the stored byte is EXACTLY
    /// `clamp(round(x/scale), -127, 127)`, the dequantized value sits within
    /// half a quantization step of the truth, and the whole-tensor `rel_l2`
    /// stays under a DERIVED bound (not a hand-fitted one) — `rel_l2` because
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

    /// Single-sequence paged/batched serving must match the reference contiguous
    /// KV generation (`Qwen::generate_kv`) token-for-token, and a two-sequence
    /// batch must equal running each prompt on its own — proving batched paged
    /// decode is exact. G4: at BOTH KV dtypes, not just fp32 — the reference is
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

    /// THE prefix-cache invariant: a warm prefill (served from cached blocks)
    /// must produce output IDENTICAL to the cold one — a cache hit that
    /// changes a single token is corruption, not a cache. Also pins that the
    /// cache actually engaged (a test that silently measured two cold runs
    /// would prove nothing). G4: at BOTH KV dtypes — this is the load-bearing
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
    /// same final hidden state a fresh engine computes — within rounding.
    ///
    /// Deliberately NOT a token-equality test: reused KV is bit-identical to
    /// its original computation, but the CPU backend's blocked GEMMs are not
    /// row-count-invariant in final-bit rounding, so a tail-only prefill can
    /// differ from a full one by an ulp — which flips argmax on a degenerate
    /// random model while meaning nothing. Structural corruption (a wrongly
    /// adopted block) produces O(1) relative error; rounding produces ~1e-6.
    /// The 1e-3 gate separates them cleanly. Token-level identity is pinned by
    /// `warm_prefill_is_identical_to_cold` where chunking is identical.
    /// G4: at BOTH KV dtypes — the `rel < 1e-3` tolerance here is deliberately
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
    /// scale-handling bug (the realistic failure mode) produces ~100% error,
    /// so the 10% gate separates cleanly while tolerating honest quant noise.
    /// The stream-level greedy-agreement threshold lives in `brain perf`'s
    /// fidelity gate, which measures it on real checkpoints.
    #[test]
    fn int8_weights_track_fp32() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 2, 8, 32, false, true);
        if !eng8.weights_int8() {
            // Capability-gated fallback (CPU JIT): the engine must run fp32
            // and say so. A device whose caps DO report the packed-int8 path
            // must never land here — a silent fallback on capable hardware is
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
    /// bounds — the kernels are trusted, so the failure would be silent
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
        assert!(p.admit(&r, &mk(4, Some(20.0))), "4 x 20ms fits a 100ms deadline");
        assert!(!p.admit(&r, &mk(6, Some(20.0))), "6 x 20ms provably misses it");
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

    /// The device-side greedy head must select exactly the token the host head
    /// would. This is the invariant that lets decode skip shipping a
    /// `[batch, vocab]` logit block back to the host — if it ever drifted, the
    /// engine would silently generate different text at speed.
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
            // Host reference: hidden -> host matmul -> argmax, per row.
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                eng.forward_batched(&mut refs, &inputs)
            };
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            // Device: same hidden already in sc.xn_final, head applied on device.
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            assert_eq!(dev, host, "device head picked a different token than the host head");
            inputs = host;
        }
    }

    /// The on-device iterative top-K extraction (`topk_extract_step` composed
    /// with the existing `argmax_part`/`argmax_final`) must return EXACTLY the
    /// row's true top-K logits+indices, sorted descending — an exact,
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
        // be admitted over MULTIPLE iterations — with decode in between — and
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
    /// the token the host head picks — including the lowest-index tie-break.
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
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            assert_eq!(dev, host, "split argmax diverged from the host head");
            inputs = host;
        }
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
    /// must each produce the same tokens as run alone — the scheduler admits,
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
    /// `Input::Resident` sub-steps 1..k) had ZERO test coverage before this —
    /// every other test here keeps the scheduler in single-step (`k=1`)
    /// territory by always having a waiting/mixed-sampling request in flight,
    /// which forces `k=1` (`model::serve::Scheduler::step`'s `all_greedy &&
    /// self.waiting.is_empty()` gate). `Input::Resident` is also the one
    /// `Input` variant `run_batched_submit` gets NO host seqlens for (`&[]` —
    /// see `serve.rs::forward_batched_greedy_window`'s sub-step 1..k calls);
    /// its per-row KV length lives only on-device (`sc.seqlen_buf`, walked by
    /// `decode_advance`). A single request, nothing else submitted, comfortably
    /// exceeding `DECODE_WINDOW` in `max_new`, is exactly the shape that makes
    /// the scheduler choose `k = DECODE_WINDOW` for most of the run — if the
    /// window path's on-device bookkeeping (positions/seqlens/block-table
    /// scheduling for those resident sub-steps) were wrong, the argmax'd
    /// tokens would diverge from the independent single-step reference below.
    /// G4: at BOTH KV dtypes — the on-device window bookkeeping
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
                // the first iteration on — the scheduler has no reason to prefer
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
    /// ordinary greedy sequence in the SAME batch — the `all_greedy` fallback
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
    /// [`kv_pool_bytes`] free function (two derivations that must agree — the
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
    /// the quantised cache) — a structural sanity check that CUMULATIVE
    /// divergence over several autoregressive steps stays small, not a
    /// precision claim: G3 already derives the exact per-element quantization
    /// bound (0.5 of a step) for a single append, and
    /// the REAL accuracy measurement (loss
    /// delta +0.0154 on Qwen3-0.6B) — this test cannot substitute for either
    /// (lesson 18: a toy config's error magnitude cannot predict the real
    /// one). What it CAN catch is a wiring break that makes int8 decode wildly
    /// diverge from fp32 (a dropped scale propagating through several steps,
    /// a slot/head index swap) — hence a bound with real headroom above the
    /// measured baseline, not the old hand-fit 20%. Runs on [`kv_probe_cfg`],
    /// not `tiny()` (lesson 4: `tiny()`'s degenerate dims don't exercise real
    /// GQA). Two independently-built engines (fp32, int8) also carry their
    /// own small autotuner-driven kernel-variant noise, independent of
    /// quantization — see `int8_kv_scale_and_bytes_match_a_host_oracle`'s doc
    /// comment — which is exactly why this bound is loose and G3's is tight.
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
        // Measured on this probe config: maxabs ~8.9e-4 at mag ~2.28 (~0.039%
        // relative) -- 1% is ~25x headroom for run-to-run autotuner noise and
        // a different random weight draw, and still 20x tighter than the old
        // hand-fit 20%.
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
    /// relative to the uncalibrated kernel — proving the clipped kernel path
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
    /// engine, even though `self.kv_calib` is internally `Some(_)` — the
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
    /// prefill — the prompt streams through in pieces attending the paged prefix.
    /// G4: at BOTH KV dtypes — chunk boundaries must not change which slot a
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
    /// never to the raw token count within one chunk — model-agnostic in spirit
    /// (any future `PagedDecoder` gets this same shape), asserted here on the one
    /// concrete implementation that exists.
    /// The asserted shape is `submits == chunks * per_chunk` exactly - strictly
    /// PROPORTIONAL, with no fixed term. Anything one-off (a device the engine's
    /// construction left with staged-but-unsubmitted uploads, say) is baselined
    /// out of the measurement below rather than folded into `per_chunk`, because
    /// a constant that only the first chunk pays is not a per-chunk cost and
    /// multiplying it by the chunk count is simply wrong arithmetic.
    /// G4: at BOTH KV dtypes — submit counts are integers, unaffected by any
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
            // 16-token prompt must cost the SAME number of submits — proof the dispatch
            // is per-CALL, not per-TOKEN (a per-token dispatcher would cost 4x more here).
            let short = vec![1u32, 5, 3, 9];
            let long: Vec<u32> = (0..16).map(|i| (i % 20) as u32 + 1).collect();
            let submits_short = submits_for(&short, 16);
            let submits_long = submits_for(&long, 16);
            assert_eq!(submits_short, submits_long, "kv_int8={kv_int8}: prefill submits must not scale with in-chunk token count: {submits_short} (4 tok) vs {submits_long} (16 tok)");
            assert!(submits_short > 0, "kv_int8={kv_int8}: prefill must dispatch SOMETHING");

            // The SAME 16-token prompt split into 2 then 4 chunks must cost exactly
            // 2x and 4x the one-chunk submit count - proportional to CHUNKS, not
            // tokens (which would be 16x, and identical at every split here). The
            // 4-chunk point is what separates "proportional to chunks" from "affine
            // in chunks": a per-run one-off would show up as a CONSTANT gap that the
            // 2x check alone could be mistaken for a per-chunk cost.
            let submits_2chunks = submits_for(&long, 8);
            let submits_4chunks = submits_for(&long, 4);
            assert_eq!(submits_2chunks, 2 * submits_long, "kv_int8={kv_int8}: 2 chunks must cost exactly 2x 1 chunk's submits, not scale with the (unchanged) token count: {submits_2chunks} vs 2x{submits_long}. A CONSTANT excess (2x+c) is a fixed per-run cost baselined into the measurement, not a prefill regression - see the flush above.");
            assert_eq!(submits_4chunks, 4 * submits_long, "kv_int8={kv_int8}: 4 chunks must cost exactly 4x 1 chunk's submits - prefill cost must be proportional to chunks, with no fixed term: {submits_4chunks} vs 4x{submits_long}");
        }
    }

    /// Speculative decoding output equals plain greedy — with a good (oracle)
    /// draft it takes far fewer target forwards; with a bad draft it falls back to
    /// ~one token per forward. Either way the tokens are identical. G4: at BOTH
    /// KV dtypes — the accept/reject mechanism, and `BlockTable::truncate` on a
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
    /// the shared paged pool must match each stream decoded alone — bit-for-bit.
    /// (The Talker is the same Qwen3 block, so the tiny config stands in for it.)
    /// G4: at BOTH KV dtypes, same `1e-6` threshold — quantization is a
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
}
