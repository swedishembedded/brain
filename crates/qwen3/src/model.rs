// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 dense decoder Transformer - forward + backprop as WGSL compute
//! dispatches, sharing the engine with the GPT/MoE/PID models (`gpu_core`,
//! `paramstore`, `optim`, `kernels`).
//!
//! Per pre-norm block (no biases anywhere):
//!   h  = RMSNorm(x)·ln1
//!   q,k,v = h·Wq, h·Wk, h·Wv         (separate, GQA: Wk/Wv narrower)
//!   q  = RoPE(QKNorm(q)·q_norm) ;  k = RoPE(QKNorm(k)·k_norm)
//!   x += Wo · GQA-attention(q,k,v)
//!   h  = RMSNorm(x)·ln2
//!   x += Wdown · ( SiLU(Wgate·h) ⊙ (Wup·h) )
//!   logits = tok.weightᵀ · RMSNorm(x)·norm    (tied head)
//!   loss   = masked cross-entropy (ignore_index = IGNORE)
//!
//! RoPE uses Qwen's half-split convention + base 1e6 (`rope_base.wgsl`), QK-norm
//! reuses `rmsnorm` over `head_dim`, and the tied head accumulates both the
//! lm_head and embedding gradients into `tok.weight` (matmul_dw then emb_bwd).

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::select::Dtype;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use model::ops::{Act, Ops, Weight};
pub use model::Shard;
use optim::Optim;
use paramstore::ParamStore;

use crate::config::QwenConfig;

/// Cross-entropy ignore index (masked target positions); the loader's `-1 i32`
/// reinterpreted as `u32`.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches STATIC_PIPELINES) ----
// B7: the per-layer linears (attn q/k/v/o, mlp gate/up/down) no longer
// dispatch through a hand-numbered index here at all - they go through
// `model::ops::Ops` (see `self.ops`/`Weight`/`self.weights` below), which
// resolves its own kernel set BY NAME on a `Gpu` handle sharing this table's
// FULL compiled pipeline set (`pipelines()`, below `STATIC_PIPELINES`). That
// retired six entries this table used to carry positionally (`matmul_reg2`,
// `quant_pack`, `matmul_i8_dyn`, `max_abs_row`, `matmul_i8_gemv`,
// `matmul_gemv`) - deleted here, not just unreferenced,
// since nothing on `self.gpu` needs them any more. Everything else
// (attention/RoPE/norms/embed/LM-head/training backward) is legitimately
// still manual - `Ops` doesn't cover those kernel categories.
/// Plain (untiled) embedding gather - kept in PIPELINES at index 0 for stable
/// indexing; the forward uses the vocab-tiled `EMBED_TILE` instead.
#[allow(dead_code)]
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const RMSNORM_DX: usize = 4;
const RMSNORM_DW: usize = 5;
const ROPE: usize = 6;
const ROPE_BWD: usize = 7;
const GQA_SCORES: usize = 8;
const ATTN_SOFTMAX: usize = 9;
const GQA_APPLY: usize = 10;
const GQA_DSCORES: usize = 11;
const GQA_DV: usize = 12;
const GQA_DQ: usize = 13;
const GQA_DK: usize = 14;
const SILU_MUL: usize = 15;
const SILU_DA: usize = 16;
const SILU_DB: usize = 17;
const ADD2: usize = 18;
const CE_VALUE: usize = 19;
#[allow(dead_code)]
const CE_GRAD: usize = 20;
const MATMUL_DX: usize = 21;
const MATMUL_DW: usize = 22;
const EMB_BWD: usize = 23;
const ADAMW: usize = 24;
const GRADNORM_SQ: usize = 25;
const GRAD_SCALE: usize = 26;
const CLIP_COEF: usize = 27;
const GRAD_SCALE_BUF: usize = 28;
const AXPY: usize = 29;
const EMBED_TILE: usize = 30;
const MATMUL_TILE: usize = 31;
const MATMUL_DX_REG: usize = 32;
const MATMUL_DW_REG: usize = 33;
const CE_STATS: usize = 34;
const CE_GRAD_STATS: usize = 35;
// Incremental KV-cache decode kernels (single new token vs the growing cache).
const ATTN_DECODE_SCORES: usize = 36;
const DECODE_SOFTMAX: usize = 37;
const ATTN_DECODE_APPLY: usize = 38;
const KV_APPEND: usize = 39;
const ROPE_AT: usize = 40;
// Vision-language residual splice (image-embedding injection). Off unless
// `enable_mm_splice` was called; see `model::vlm`.
const SPLICE: usize = 41;
const SPLICE_BWD: usize = 42;
// Table-driven RoPE for the interleaved M-RoPE path (Qwen3-VL). Off unless
// `enable_mrope` was called; replaces the analytic rope_base on q/k.
const ROPE2D: usize = 43;
// DeepStack residual add (Qwen3-VL): adds a level's merged vision features into
// the residual at the image rows after a layer. Off unless `enable_deepstack`.
const SPLICE_ADD: usize = 44;
// Qwen2 q/k/v projection bias (add fwd, row-sum grad bwd). Used only when
// `cfg.attn_bias` (FastVLM's Qwen2 decoder); Qwen3 is bias-free.
const BIAS_ADD: usize = 45;
const BIAS_GRAD: usize = 46;
// Decode-regime fp32 kernel (A1/A2): workgroup-per-row rmsnorm - the m=1
// shapes KV decode is made of (the matching GEMV now lives behind `Ops`).
const RMSNORM_ROWS: usize = 47;
// Encoder right-padding key mask (FLUX.2 text-encoder parity).
const GQA_SCORES_KMASK: usize = 48;
// Workgroup-per-row softmax over the [B*H*T, T] score slab - the coalesced twin
// of `attn_softmax` (see the kmask attention below).
const SOFTMAX_ROWS: usize = 49;
// `matmul_reg2` with its shared-memory bank conflicts removed; bit-identical
// output, so it is a pure speed swap (see `linear_kernel`, now LM-head-only).
const MATMUL_REG3: usize = 50;
// DeepStack decode's own add: `splice_add`'s `base` lands on `dst` only, but
// decode needs to read `deepstack_bufs[level]` at an offset (`local_row * d`)
// while writing this step's own zero-offset residual row -- see
// `decode_steps`'s own call site.
const SPLICE_ADD_OFFSET_SRC: usize = 53;
const SCALE_ROW: usize = 54;

const STATIC_PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("axpy", kernels::AXPY),
    ("embed_tile", kernels::EMBED_TILE),
    ("matmul_tile", kernels::MATMUL_TILE),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("kv_append", kernels::KV_APPEND),
    ("rope_at", kernels::ROPE_AT),
    ("splice", kernels::SPLICE),
    ("splice_bwd", kernels::SPLICE_BWD),
    ("rope2d", kernels::ROPE2D),
    ("splice_add", kernels::SPLICE_ADD),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("gqa_scores_kmask", kernels::GQA_SCORES_KMASK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
    ("splice_add_offset_src", kernels::SPLICE_ADD_OFFSET_SRC),
    // Per-position CE-gradient weighting (model::Batch::LmWeighted /
    // `enable_weighted_loss`) - appended here (and only here) is the whole
    // opt-in, same convention as `gradnorm_part`/`clip_coef_wg` above.
    ("scale_row", kernels::SCALE_ROW),
];

/// This model's FULL kernel set: `STATIC_PIPELINES` (every hand-numbered
/// const above indexes into this - unchanged positions 0..53) followed by
/// the `model::ops::Ops` façade's own required kernels (B7), appended with
/// NO named consts of their own - exactly like `gradnorm_part`/`clip_coef_wg`
/// above, resolved by `Ops::new` purely BY NAME (`Gpu::kernel_index`), never
/// by position.
///
/// **Why these live on `self.gpu`'s own list instead of a second `Gpu` (e.g.
/// `Gpu::new_like`).** A `Step`'s `kind` is an index into the SPECIFIC `Gpu`
/// handle's own compiled pipeline vector - it is NOT portable across two
/// different handles, even two `new_like`d handles on the same physical
/// device, because each independently-built kernel list gets its own
/// pipeline vector with its own index assignment. `forward_steps`/
/// `decode_steps` build ONE combined `Vec<Step>` (mixing `self.gpu.step(...)`
/// calls with `self.ops.act`/`self.ops.matmul`'s own pushes) and submit it
/// through ONE `self.gpu.submit(...)` call - so `self.ops`'s internal `Gpu`
/// MUST resolve every kernel name to the SAME index `self.gpu` would, or a
/// mixed submission dispatches the wrong pipeline at that slot (confirmed the
/// hard way: `Gpu::new_like` initially used here produced exactly that -
/// `self.ops`'s `max_abs_row` index collided with `self.gpu`'s own `silu_mul`
/// slot, a wgpu bind-group-layout validation failure at test time, not a
/// silent corruption, but only because the two pipelines' binding COUNTS
/// happened to differ - a same-shape collision would NOT have been caught by
/// the backend at all). `Ops` is therefore built from `gpu.share()` - "same
/// adapter, queue and compiled pipelines" (`Gpu::share`'s own doc) - which
/// only works if the ORIGINAL handle already compiled every name `Ops::new`
/// looks up; hence this function, not a second kernel list.
fn pipelines() -> &'static [(&'static str, &'static str)] {
    use gpu_core::select::Dtype;
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = STATIC_PIPELINES.to_vec();
        v.push(("matmul_gemv", kernels::MATMUL_GEMV));
        // `Ops::bind` resolves its `(RegisterTiled, F32)` kernel by the NAME
        // "matmul_reg2" - but `matmul_reg3` is the bit-identical, measurably
        // faster twin (identical `Params`, identical tiling, only the
        // shared-memory bank conflicts removed - swept across a wide shape
        // range with zero max|Δ| and a consistent speedup, no shape where
        // preferring `matmul_reg2` is correct) this crate's own
        // `linear_kernel` already prefers. Registering the NAME "matmul_reg2"
        // against the `matmul_reg3` SOURCE is exactly the sanctioned escape
        // hatch `Ops::bind`'s own doc comment describes: "a model with a
        // differently-named but bit-identical physical kernel simply
        // registers it under that canonical name when it builds its `Gpu`."
        // Using the real (slower) `matmul_reg2` source here would silently
        // undo that speed-up for every RegisterTiled fp32 dispatch this
        // façade makes.
        v.push(("matmul_reg2", kernels::MATMUL_REG3));
        v.push(("matmul_i8_dyn", kernels::MATMUL_I8_DYN));
        v.push(("matmul_i8_gemv", kernels::MATMUL_I8_GEMV));
        v.push(("matmul_q4_dyn", kernels::MATMUL_Q4_DYN));
        v.push(("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV));
        v.push(("max_abs_row", kernels::MAX_ABS_ROW));
        v.push(("quant_pack", kernels::QUANT_PACK));
        // `Ops::REQUIRED_KERNELS` also demands the bf16/f16 storage-tier
        // variants (B4/B5/B8/B9/B10) even though this crate never builds a
        // `Weight::BF16`/`Weight::F16` and has its own KV-cache mechanism
        // (never dispatches the generic `paged_*_batched` family at all) -
        // see `Ops::new`'s own doc comment ("every model that builds an
        // `Ops` must register the full façade kernel set, not just the
        // tiers it plans to use"). Compiled, never dispatched.
        //
        // This list deliberately does NOT delegate to the canonical
        // `model::ops::kernel_list()` (which every other `Ops`-building call
        // site now does), because it is not the facade set: it extends
        // `STATIC_PIPELINES` and, crucially, registers the NAME
        // "matmul_reg2" against the `matmul_reg3` SOURCE above - the
        // sanctioned name/source override `Ops::bind` documents. Pulling in
        // the canonical list would re-register "matmul_reg2" with the real
        // (slower) `matmul_reg2` source and silently undo that speed-up. The
        // `pipelines_has_every_kernel_ops_new_requires` test below is what
        // keeps this superset honest against `REQUIRED_KERNELS`.
        for dt in [Dtype::BF16, Dtype::F16] {
            v.push(kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("embed", kernels::EMBED, "emb", dt).unwrap());
            v.push(kernels::template::dtype_variant("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", dt).unwrap());
        }
        v.push(("moe_linear_gated", kernels::MOE_LINEAR_GATED));
        v.push(("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED));
        v.push(
            kernels::template::dtype_variant_store(
                "paged_kv_append_batched_word",
                kernels::PAGED_KV_APPEND_BATCHED_WORD,
                "pool",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED));
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_scores_batched",
                kernels::PAGED_DECODE_SCORES_BATCHED,
                "pool_k",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED));
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_apply_batched",
                kernels::PAGED_DECODE_APPLY_BATCHED,
                "pool_v",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(kernels::template::dtype_variant("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16).unwrap());
        v
    })
}

/// Pick the GEMM kernel + dispatch thread count for a forward linear
/// `[m,k]·[n,k]ᵀ`. Delegates to `model::block::pick_gemm` (B2: a thin adapter
/// over `backend_api::select::candidates`, whose `GEMM_TILE_MIN_ROWS`/
/// `GEMM_TILE_MIN_COLS` carry the measured P40 crossover this used to restate
/// here) - same math either way (parity gated by `tests/backend_parity` +
/// gradcheck), so this only changes speed. `BRAIN_QWEN_NAIVE_MM=1` forces the
/// naive kernel.
///
/// B7: the seven per-layer linears (attention q/k/v/o, SwiGLU gate/up/down)
/// no longer call this - they go through `Ops::matmul` (`self.ops`/
/// `self.weights`), which owns its own kernel selection. This function's ONE
/// remaining caller is the LM head (`forward_steps`'s single-tile case):
/// deliberately NOT migrated onto `Ops` (see the B7 ledger entry) - `Ops::act`
/// always quantizes its input eagerly and unconditionally, and the LM head's
/// activation (`xn_final`) is never paired with anything but an `F32` weight
/// (there is no int8 LM head in this crate), so routing it through `Ops`
/// would pay a real `max_abs_row`/`quant_pack` dispatch on every forward for
/// a quantized form nothing ever reads - a measurable prefill-throughput
/// regression for zero benefit. Embedding/LM-head dispatch is explicitly
/// listed as still-legitimately-manual in B7's own scope.
fn linear_kernel(m: usize, n: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    // `matmul_reg3` = `matmul_reg2` with the shared-memory bank conflicts
    // removed: identical tiling and identical K accumulation order, therefore
    // BIT-IDENTICAL output (measured max_abs 0.0), and measurably faster on the
    // FLUX.2 text encoder's prefill shapes (196 GEMMs at 512 tokens).
    // Same dispatch geometry, and the CPU backend routes both to one AVX2 GEMM.
    block::pick_gemm(m, n, MATMUL, MATMUL_REG3, naive)
}

/// Backward GEMM pickers: tiled `matmul_{dx,dw}_reg` (bit-identical to naive,
/// and a large fraction of the card's fp32 peak) once both output dims fill a
/// 128-tile, else naive. Small
/// LoRA-rank matmuls fall back automatically. `BRAIN_QWEN_NAIVE_MM=1` forces naive.
/// Full fine-tuning with AdamW moments offloaded to system RAM (`BRAIN_OFFLOAD_ADAM=1`).
fn offload_adam() -> bool {
    std::env::var("BRAIN_OFFLOAD_ADAM").map(|v| v != "0").unwrap_or(false)
}

/// Storage-binding offset alignment, in f32 words. WebGPU's
/// `min_storage_buffer_offset_alignment` is 256 B (the downlevel default every
/// adapter brain has met reports), and 256 B / 4 B = 64 f32.
const HEAD_TILE_ALIGN: u32 = 64;

/// `block::vocab_tiles*`'s tiling, re-split so every tile BOUNDARY is a
/// multiple of [`HEAD_TILE_ALIGN`] rows.
///
/// The embedding gather does not need this: `embed_tile` slices only the
/// WEIGHT (offset `v0 * d_model`, already a large multiple of 64 for any real
/// `d_model`) and writes its output through a whole binding. The decode head
/// slices the `[vocab]` OUTPUT too, at word offset `v0`, so `v0` itself has to
/// clear the alignment - and the budget-derived row count generally does not
/// (a P40's 2047 MiB binding limit over `d_model = 4096` yields 65503 rows,
/// which is odd).
///
/// Rounding the stride DOWN keeps every tile inside the budget it was sized
/// for. A single tile needs no alignment at all (its offset is 0), so a small
/// vocab is returned untouched; a stride under one alignment unit (only
/// reachable via an artificially tiny `BRAIN_TILE_BUDGET_WORDS`) is raised to
/// one, since a correct binding beats a budget that no real device imposes.
fn align_head_tiles(base: &[(u32, u32)], vocab: u32) -> Vec<(u32, u32)> {
    if base.len() <= 1 {
        return base.to_vec();
    }
    let stride = (base[0].1 / HEAD_TILE_ALIGN * HEAD_TILE_ALIGN).max(HEAD_TILE_ALIGN);
    let mut out = Vec::new();
    let mut v0 = 0u32;
    while v0 < vocab {
        let cnt = stride.min(vocab - v0);
        out.push((v0, cnt));
        v0 += cnt;
    }
    out
}

fn dx_kernel_bw(m: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    block::pick_gemm(m as usize, k as usize, MATMUL_DX, MATMUL_DX_REG, naive)
}
fn dw_kernel_bw(nrows: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    block::pick_gemm(nrows as usize, k as usize, MATMUL_DW, MATMUL_DW_REG, naive)
}



/// The parameter subset a shard holds. A whole shard returns `cfg.param_list()`
/// verbatim (so the single-device store is byte-identical). A partial shard keeps
/// only its layers' weights, plus `tok.weight` when it embeds and/or carries the
/// tied head, and `norm.weight`+head when it is the head stage.
///
/// Public because it is also the *required-tensor set* for a shard-aware
/// import: a loader that will build this shard needs exactly these names and
/// no others, so `crate::import::hf_shard_source` derives its coverage check
/// from here rather than from the full `cfg.param_list()`. Keeping one
/// definition means the set a checkpoint is validated against cannot drift
/// from the set the build actually reads.
pub fn shard_param_list(cfg: &QwenConfig, shard: &Shard) -> Vec<(String, usize)> {
    let full = cfg.param_list();
    if shard.is_whole(cfg.n_layers as usize) {
        return full;
    }
    let head = cfg.head_weight(); // "tok.weight" (tied) or "lm_head.weight"
    let tied = head == "tok.weight";
    full.into_iter()
        .filter(|(name, _)| {
            if let Some(rest) = name.strip_prefix("blocks.") {
                let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
                return shard.owns(l);
            }
            match name.as_str() {
                "tok.weight" => shard.embed || (shard.head && tied),
                "norm.weight" => shard.head,
                _ if name == head => shard.head, // untied lm_head
                _ => false,
            }
        })
        .collect()
}

struct Layer {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

pub struct Qwen {
    pub gpu: Gpu,
    pub cfg: QwenConfig,
    pub ps: ParamStore,
    /// Pipeline shard this instance owns (whole model on GPU 0 by default).
    pub shard: Shard,
    opt: Optim,
    /// Host AdamW for offloaded params (RAM-resident moments); None otherwise.
    offload_opt: std::cell::RefCell<Option<optim::OffloadAdam>>,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    res: Vec<DeviceBuffer>,
    layers: Vec<Layer>,

    // Vision-language embedding splice (off = None). When set to `(row0, n_rows)`
    // the forward overwrites residual rows `[row0, row0+n_rows)` with `img_embeds`
    // (written by the vision front-end via `write_img_embeds`) after the text
    // token-embedding gather, and the backward routes those rows' gradient into
    // `d_img_embeds` (read via `read_d_img_embeds`) instead of `tok.weight`.
    mm_splice: Cell<Option<(u32, u32)>>,
    img_embeds: DeviceBuffer,
    d_img_embeds: DeviceBuffer,

    // Interleaved M-RoPE (Qwen3-VL): when set, q/k use the table-driven `rope2d`
    // with these host-precomputed per-token cos/sin tables `[b·t, head_dim/2]`
    // (write via `write_mrope_tables`) instead of the analytic rope_base.
    mrope: Cell<bool>,
    mrope_cos: DeviceBuffer,
    mrope_sin: DeviceBuffer,

    // Reward/advantage-weighted CE gradient (off = None, ordinary supervised
    // training, zero extra dispatch). When enabled via `enable_weighted_loss`,
    // the backward scales `d_logits` per ROW (token position) by
    // `loss_weights` (`scale_row.wgsl`) into `d_logits_weighted`, and every
    // downstream consumer (`head` dw/dx) reads that instead of the raw
    // `d_logits`. Named `loss_weights`, not `weights` - that name is already
    // the Ops/Weight façade's per-tensor map (`self.weights: HashMap<String,
    // Weight>`, B7) a few fields below. See `model::Batch::LmWeighted`'s doc
    // comment for the contract this realizes.
    weighted: Cell<bool>,
    loss_weights: DeviceBuffer,
    d_logits_weighted: DeviceBuffer,

    // Decode-path M-RoPE: a single-row `[1, head_dim/2]` cos/sin table, reused
    // (overwritten) every `step_mrope`/`step_embed_mrope` call rather than
    // sliced from `mrope_cos`/`mrope_sin` above -- those are sized for the
    // batched forward's whole KNOWN sequence, but decode generates tokens
    // beyond it one at a time, so each new token needs its OWN freshly
    // written table (mirrors `qwen3omnimoe::thinker::layer_decode_step`'s pattern:
    // a 1-row table needs no separate "decode" kernel, just `rope2d`'s
    // existing `tmod`-driven table indexing at `rows = tmod = 1`). Always
    // allocated (cheap: `head_dim/2` floats) so `step_mrope` needs no
    // separate `enable_*` call, unlike the batched path's `mrope_cos`/`sin`.
    decode_mrope_cos: DeviceBuffer,
    decode_mrope_sin: DeviceBuffer,

    // DeepStack (Qwen3-VL): `(row0, n_rows, n_levels)` and one `[n_rows·d]` buffer
    // per level. When set, level `l`'s features are added into the residual at the
    // image rows right after decoder layer `l` (for `l < n_levels`). Write each
    // level via `write_deepstack`.
    deepstack: Cell<Option<(u32, u32, u32)>>,
    deepstack_bufs: Vec<DeviceBuffer>,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    scores: DeviceBuffer,
    // Additive per-key attention mask ([t] f32; 0 live / -3.4e38 excluded) and
    // its arming flag - the padded-encoder path (`encode_hiddens_padded`).
    kmask: DeviceBuffer,
    kmask_on: Cell<bool>,
    /// The device runs workgroup-cooperative reductions (barriers). Selects the
    /// coalesced workgroup-per-row RMSNorm / softmax; false on the CPU JIT,
    /// which keeps the per-element reference kernels (whose native AVX2 fast
    /// paths are the fast CPU route anyway).
    coop: bool,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    /// `[vocab]` output for the DEVICE decode head ([`Qwen::decode_logits`]),
    /// allocated on first use. Deliberately NOT the batched `logits` slab
    /// above: that one is `n · vocab` and is a size-1 dummy on a decode-only
    /// build, which is exactly the allocation a decode build must not
    /// resurrect (~311 MB at vocab 152k, block 512). One decode row is 800 kB
    /// at vocab 200k.
    dec_logits: std::cell::RefCell<Option<DeviceBuffer>>,
    ce_buf: DeviceBuffer,

    // backward temporaries
    dres: Vec<DeviceBuffer>,
    d_logits: DeviceBuffer,
    ce_stats: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    dq_pre: DeviceBuffer,
    dk_pre: DeviceBuffer,
    d_v: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    inv: DeviceBuffer,

    // LoRA scratch (sized for rank `r`; trivially small when LoRA is off).
    lora_a: DeviceBuffer,   // [n*r] : a = x @ A^T
    lora_da: DeviceBuffer,  // [n*r] : grad wrt a
    lora_out: DeviceBuffer, // [n*max_out] : delta = a @ B^T

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: DeviceBuffer,

    // Persistent per-layer KV cache for incremental decode ([max_t, kv_dim] each),
    // and the next absolute position `step` will decode (cache fill level). Sized
    // for the whole model; the decode path requires a single-device (whole) shard.
    kcache: Vec<DeviceBuffer>,
    vcache: Vec<DeviceBuffer>,
    dec_pos: Cell<u32>,
    /// The `Ops` façade (B3/B7) this model dispatches its per-layer linears
    /// through - a second handle onto the SAME device AND compiled pipeline
    /// set as `gpu` (`Gpu::share` - see `pipelines()`'s own doc comment for
    /// why sharing, not a second independent kernel list, is required),
    /// resolving kernels by name rather than a hand-numbered index.
    ops: Ops,
    /// Every one of the 7 per-layer linears (`attn.{wq,wk,wv,wo}`/
    /// `mlp.{gate,up,down}`, keyed `blocks.<l>.<leaf>`) this shard owns, as a
    /// `model::ops::Weight` - uniformly at whichever tier the constructor
    /// asked for AND the device's caps allow (`Weight::upload`'s own
    /// `want.promote(caps.numeric)` gate): `F32`, `I8` (needs
    /// `numeric.int8_dot`) or `F16` (needs `numeric.f16 || f16_storage`;
    /// the packed-f16 decode is plain integer/bitcast WGSL, so no device
    /// FEATURE is involved). Replaces the
    /// old per-model `q8` field (an `Option` wrapping `crate::q8::Q8`): the
    /// forward dispatches whatever tier a `Weight` value itself carries,
    /// never a separate on/off flag inspected at dispatch time. The `F32` tier costs no extra
    /// VRAM over the pre-B7 path - it wraps a `.clone()` of the SAME
    /// `DeviceBuffer` `ps` already holds (a cheap `Arc` bump, not a second
    /// upload).
    weights: HashMap<String, Weight>,
    /// True for a [`Self::from_reader_decode`] build: activations are sized for
    /// a single token and `scores`/`probs` for `n_heads·ctx` (KV-cache decode
    /// only - the KV cache is the only ctx-scaled allocation). The batched
    /// forward/backward entry points assert against being called on such an
    /// instance instead of silently reading/writing past the smaller buffers.
    decode_only: bool,
}

impl Qwen {
    /// Load a trainable model (weights + grad + AdamW moments) from a checkpoint.
    /// Streams the weights one tensor at a time off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader) - peak host ≈ one
    /// tensor of f32, never the whole-model `checkpoint::load` + `by_role("")`
    /// host copy. AdamW moments are device zero-init (not read from disk), so
    /// this is byte-identical to the former eager path.
    pub fn load(path: &str, b: u32, t: u32) -> Qwen {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, &reader, true, shard, Dtype::F32, false)
    }

    /// Load an **inference-only** model: parameters are frozen (weights only, no
    /// grad/AdamW buffers), cutting device memory ~4× - essential for loading a
    /// real 0.6B checkpoint for generation. Builds only the forward graph.
    pub fn load_inference(path: &str, b: u32, t: u32) -> Qwen {
        Self::load_inference_with(path, b, t, Dtype::F32)
    }

    /// Streaming inference load: build from a mmap-backed [`WeightReader`],
    /// uploading one tensor at a time (peak host ≈ one tensor of f32) - never the
    /// `checkpoint::load` + `by_role("")` whole-model host copy. Numerically
    /// identical to [`Qwen::load_inference`]; used by the resident serve path.
    pub fn from_reader_inference(reader: &checkpoint::weightio::WeightReader, b: u32, t: u32) -> Qwen {
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, reader, false, shard, Dtype::F32, false)
    }

    /// Streaming **decode-only** load: like [`Self::from_reader_inference`]
    /// (mmap-backed, one tensor at a time), but shaped for incremental
    /// KV-cache decode only ([`Self::step`]/[`Self::prefill`]/[`Self::step_embed`])
    /// rather than the batched forward. Activations are sized for a single
    /// token (`n = 1`) instead of `b·t`, and `scores`/`probs` for `n_heads·ctx`
    /// instead of `n_heads·ctx²` - the KV cache (`[ctx, kv_dim]` per layer) is
    /// the only allocation that scales with `ctx`. No backward buffers and no
    /// `logits`/`d_logits` buffer (the LM head is applied host-side; see
    /// `sample::generate_kv_stream`). Calling a batched forward/backward entry
    /// point on the result panics loudly rather than reading/writing past the
    /// smaller buffers - use the KV-cache decode API instead.
    pub fn from_reader_decode(reader: &checkpoint::weightio::WeightReader, ctx: u32) -> Qwen {
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, 1, ctx, reader, false, shard, Dtype::F32, true)
    }

    /// [`Self::from_reader_decode`], but from an in-memory tensor map instead of
    /// a mmap'd checkpoint file -- for serving a named LoRA adapter, whose delta
    /// must be folded into the base tensors (`qwen3::lora::fold_adapter_into`)
    /// before a decode-only KV-cache model can be built from the result. Pays
    /// the whole-model host copy `from_reader_decode` avoids, but only for the
    /// (rare, one-off-per-activation) adapter-serving path.
    pub fn from_tensors_decode(cfg: QwenConfig, tensors: &HashMap<String, Vec<f32>>, ctx: u32) -> Qwen {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, 1, ctx, tensors, false, shard, Dtype::F32, true)
    }

    /// [`Self::load_inference`] with the int8 numeric tier: per-channel weight
    /// quantisation + dynamic activation quant, for both batched forwards and
    /// KV-cache decode (the m=1 packed GEMV).
    pub fn load_inference_i8(path: &str, b: u32, t: u32) -> Qwen {
        Self::load_inference_with(path, b, t, Dtype::I8)
    }

    /// Streaming inference load shared by [`Self::load_inference`] and
    /// [`Self::load_inference_i8`]: drive the builder straight off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader), uploading one tensor
    /// at a time - peak host ≈ one tensor of f32, never the whole-model
    /// `checkpoint::load` + `by_role("")` host copy on top of the device copy.
    /// Every non-fp32 tier reads + packs one linear at a time
    /// (`src.with_tensor`, the reader as a
    /// [`TensorSource`](checkpoint::TensorSource), feeding `Weight::upload`
    /// per leaf - see `new_impl`'s `weights` construction).
    fn load_inference_with(path: &str, b: u32, t: u32, dt: Dtype) -> Qwen {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, &reader, false, shard, dt, false)
    }

    pub fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, init, true, shard, Dtype::F32, false)
    }

    /// Build a single pipeline **stage**: only the layers (and endpoint weights)
    /// in `shard` are allocated on this device. `train` selects the parameter
    /// roles (offload/LoRA/frozen) exactly as the whole-model path does.
    /// `shard.gpu_index` names the canonical physical card (device registry);
    /// `Shard::ANY_GPU` keeps the ambient selection.
    /// Takes any `checkpoint::TensorSource` - the eager `&HashMap<String,
    /// Vec<f32>>` every existing caller passes (coerces, unchanged), or a
    /// streaming mmap'd `WeightReader`/`RemapSource` pair, which never
    /// materializes the whole checkpoint on the host.
    pub fn new_shard(cfg: QwenConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, train: bool, shard: Shard) -> Qwen {
        Qwen::new_impl(cfg, b, t, init, train, shard, Dtype::F32, false)
    }

    /// Inference-only shard with the 7 per-layer linears quantized to int8 (DP4A).
    /// Weights are ~4× smaller than fp32, so the whole Qwen3-4B encoder (~4.8 GB of
    /// weights → ~9.5 GB resident) fits a single 24 GB card - where the fp32
    /// encoder (~30 GB resident on non-ReBAR Pascal) does not. Frozen, no LoRA.
    /// See [`Self::new_shard`]'s doc: `init` may be any `TensorSource`.
    ///
    /// A thin alias for [`Self::new_shard_dt`] at [`Dtype::I8`] - there is ONE
    /// implementation of the reduced-precision build, not one per tier.
    pub fn new_shard_i8(cfg: QwenConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, shard: Shard) -> Qwen {
        Qwen::new_shard_dt(cfg, b, t, init, shard, Dtype::I8)
    }

    /// Inference-only shard at an explicit **weight storage tier** for the 7
    /// per-layer linears (`crate::q8::Q8::LINEARS`) - the dtype-parameterised
    /// constructor [`Self::new_shard_i8`] is an alias of.
    ///
    /// * [`Dtype::F32`] - identical to `new_shard(.., train = false, ..)`: the
    ///   `Weight`s alias the `ParamStore`'s own buffers, no second upload.
    /// * [`Dtype::I8`] - per-channel symmetric DP4A weights + dynamic
    ///   per-token activation quant (~4× smaller). Needs
    ///   `caps.numeric.int8_dot`.
    /// * [`Dtype::F16`] / [`Dtype::BF16`] - the **storage** tiers: the weight
    ///   is packed two-per-`u32` (`model::half::pack_f16`/`pack_bf16`, exactly
    ///   2× smaller) and decoded back to f32 *inside* the `#w=f16`/`#w=bf16`
    ///   kernel variant with plain integer/bitcast WGSL. The arithmetic stays
    ///   fp32, so **no device feature** (`wgpu::Features::SHADER_F16` and
    ///   friends) is required and the tier is available identically on the CPU
    ///   JIT, wgpu and in the browser.
    /// * [`Dtype::Q4`] - W4A8, same shape as `I8`.
    ///
    /// The tier is a REQUEST, not a guarantee: `Weight::upload` runs it
    /// through `want.promote(caps.numeric)`, so a device that cannot execute
    /// it lands back on fp32 rather than dispatching a kernel it has no path
    /// for. Ask [`Self::linear_dtype`] what actually happened.
    ///
    /// **Only the 7 per-layer linears change tier.** The token embedding
    /// (`tok.weight`) and the LM head stay fp32 for the same reason they do on
    /// the int8 path, and the reason is structural, not a policy choice: this
    /// crate's embedding gather is `embed_tile.wgsl` and its head GEMM is
    /// `linear_kernel`'s plain `matmul`/`matmul_reg3`, neither of which is
    /// registered here in a packed-storage variant - they read the
    /// `ParamStore` buffer as raw f32, so handing them packed words would
    /// reinterpret bit patterns as garbage floats. The per-layer RMSNorm/
    /// QK-norm gains stay fp32 too: they are `[d]`/`[head_dim]` vectors (a
    /// rounding error away from free) consumed by norm kernels, not GEMMs.
    pub fn new_shard_dt(cfg: QwenConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, shard: Shard, dt: Dtype) -> Qwen {
        Qwen::new_impl(cfg, b, t, init, false, shard, dt, false)
    }

    /// [`Self::new_shard_dt`] shaped for **KV-cache decode**, the way
    /// [`Self::from_reader_decode`] is to [`Self::new_shard`] - the two axes
    /// (weight tier, activation shape) were previously only available one at
    /// a time.
    ///
    /// `new_shard_dt` builds the BATCHED forward shape: per-layer activations
    /// at `n = b·t`, `scores`/`probs` at `n_heads·ctx²`, a `logits` buffer at
    /// `n·vocab`, and - because the `bwd` scratch is dummied out by
    /// `decode_only` rather than by `train` - backward buffers, even on an
    /// inference build. A caller that only ever drives `prefill`/`step`/
    /// `step_embed` pays for every one of those and touches none of them.
    ///
    /// That is not a rounding error at real scale. An 8B model with
    /// `vocab = 200000` reaching for int8 to fit a 24 GB card gets its
    /// linears down to ~7 GB and then spends the saving back on batched
    /// scratch it never reads. This constructor is what makes "int8 so it
    /// fits" actually fit.
    ///
    /// Same tier semantics as [`Self::new_shard_dt`]: the request goes
    /// through `Weight::upload`'s own `promote`, so ask [`Self::linear_dtype`]
    /// what landed. Calling a batched forward/backward entry point on the
    /// result panics loudly rather than running past the smaller buffers.
    pub fn new_shard_dt_decode(cfg: QwenConfig, ctx: u32, init: &dyn checkpoint::TensorSource, shard: Shard, dt: Dtype) -> Qwen {
        Qwen::new_impl(cfg, 1, ctx, init, false, shard, dt, true)
    }

    /// The shared builder behind every constructor. `decode_only` (set only by
    /// [`Self::from_reader_decode`]) shapes the model for single-token KV-cache
    /// decode instead of the batched forward: activations at `n = 1`,
    /// `scores`/`probs` at `n_heads·ctx` (not `n_heads·ctx²`), no backward
    /// scratch, no `logits`/`d_logits`/CE buffers.
    fn new_impl(cfg: QwenConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool, shard: Shard, dt: Dtype, decode_only: bool) -> Qwen {
        // ONE reduced-precision build, parameterised by `dt` - the int8 tier
        // is not a second code path, it is this one at `Dtype::I8`. Every
        // non-fp32 tier is inference-only for the same reason int8 always
        // was: the trainable master copy lives in the fp32 `ParamStore`, and
        // a packed/quantized `Weight` is built ONCE at construction from the
        // source tensor, never re-derived after an optimiser step.
        assert!(!(dt != Dtype::F32 && train), "the {dt:?} weight tier is inference-only");
        assert!(!(decode_only && train), "decode-only build is inference-only");
        // An explicitly-placed shard binds its canonical card through the
        // device registry; `Shard::ANY_GPU` (the `Shard::whole` default) keeps
        // the ambient selection (`--device` / scoped `with_gpu`).
        let gpu = if shard.gpu_index == Shard::ANY_GPU {
            Gpu::new(pipelines())
        } else {
            Gpu::new_on_index(shard.gpu_index as u32, pipelines())
                .unwrap_or_else(|e| panic!("qwen shard placement: {e}"))
        };
        // A second handle onto the SAME device AND the SAME compiled
        // pipeline set (`Gpu::share`, not `Gpu::new_like` - see `pipelines`'s
        // own doc comment for why index-space compatibility, not just
        // physical-device sharing, is required here) for the `Ops` façade
        // (B7).
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("qwen: Ops::new: {e}"));
        // The parameter set this stage actually holds: the whole list for a whole
        // shard (byte-identical to before), or just this stage's slice otherwise.
        // At any non-fp32 tier the 7 per-layer linears live in `weights`
        // (packed int8/f16/bf16/q4 `Weight`s), NOT the fp32 store - filter
        // them out so no fp32 copy is ever uploaded alongside the packed one.
        // Without this filter an f16 build would be BIGGER than fp32 (1.0x
        // master copy + 0.5x packed), not half the size.
        let quantized = dt != Dtype::F32;
        let plist: Vec<(String, usize)> = shard_param_list(&cfg, &shard)
            .into_iter()
            .filter(|(name, _)| !(quantized && crate::q8::Q8::is_i8_linear(name)))
            .collect();
        // Role assignment:
        //  - inference (`!train`): every parameter Frozen (weights only).
        //  - LoRA training: only `*.lora_a`/`*.lora_b` trainable; base Frozen.
        //  - full training: every parameter Trainable (or Offload).
        let ps = if !train {
            let roles = plist
                .into_iter()
                .map(|(n, c)| (n, c, paramstore::Role::Frozen))
                .collect();
            ParamStore::new_with_roles_src(&gpu, roles, src)
        } else if cfg.lora.is_some() {
            let roles = plist
                .into_iter()
                .map(|(n, c)| {
                    let role = if n.ends_with(".lora_a") || n.ends_with(".lora_b") {
                        paramstore::Role::Trainable
                    } else {
                        paramstore::Role::Frozen
                    };
                    (n, c, role)
                })
                .collect();
            ParamStore::new_with_roles_src(&gpu, roles, src)
        } else if offload_adam() {
            // Full fine-tuning with the AdamW moments in system RAM (Role::Offload):
            // GPU holds only weight+grad (2×model) instead of 4×model.
            let roles = plist
                .into_iter()
                .map(|(n, c)| (n, c, paramstore::Role::Offload))
                .collect();
            ParamStore::new_with_roles_src(&gpu, roles, src)
        } else {
            ParamStore::new_src(&gpu, plist, src)
        };
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);
        // Lazily-built host optimiser for the offloaded params (None unless any
        // parameter took Role::Offload).
        let offload_opt: std::cell::RefCell<Option<optim::OffloadAdam>> = std::cell::RefCell::new(None);

        // Decode-only: activations at n=1 (one token) instead of b·t, and the
        // score/prob extent at n_heads·ctx instead of n_heads·ctx² - the KV
        // cache below is the only allocation left that scales with ctx.
        let n = if decode_only { 1u64 } else { (b * t) as u64 };
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let bht2 = if decode_only { cfg.n_heads as u64 * t as u64 } else { (b * cfg.n_heads * t * t) as u64 };
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer(
            "tokens",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let targets = gpu.buffer(
            "targets",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let ce_grad_uni = gpu.uniform_dynamic(4); // [n, vocab, IGNORE, count]

        // Residual stream: `res[i]` is live only at this shard's boundaries
        // (`start..=end`); non-boundary indices are size-1 dummies so the model's
        // absolute `res[l]`/`res[l+1]` indexing is preserved unchanged. For a
        // whole shard every index is live - identical to the single-device path.
        let mut res = Vec::new();
        let mut dres = Vec::new();
        for i in 0..=cfg.n_layers as usize {
            let live = i >= shard.start && i <= shard.end;
            res.push(if live { st(n * d) } else { st(1) });
            dres.push(if live && !decode_only { st(n * d) } else { st(1) });
        }
        let dummy_layer = || Layer {
            xn1: st(1), q_pre: st(1), q: st(1), k_pre: st(1), k: st(1), v: st(1),
            probs: st(1), ctx: st(1), xmid: st(1), xn2: st(1), gate_pre: st(1), up: st(1), h: st(1),
        };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers as usize {
            layers.push(if shard.owns(l) {
                Layer {
                    xn1: st(n * d),
                    q_pre: st(n * hq),
                    q: st(n * hq),
                    k_pre: st(n * hkv),
                    k: st(n * hkv),
                    v: st(n * hkv),
                    probs: st(bht2),
                    ctx: st(n * hq),
                    xmid: st(n * d),
                    xn2: st(n * d),
                    gate_pre: st(n * ff),
                    up: st(n * ff),
                    h: st(n * ff),
                }
            } else {
                dummy_layer()
            });
        }
        // `inv` must hold the per-row RMS for the largest norm: QK-norm-q has
        // n*n_heads rows (>= n*n_kv and >= n).
        let inv_rows = n * cfg.n_heads as u64;
        // LoRA scratch (rank r; max projection output across all sites).
        let r = cfg.lora.as_ref().map(|l| l.rank as u64).unwrap_or(0).max(1);
        let max_out = hq.max(ff).max(d).max(hkv);

        // Head-only buffers (final norm + lm_head + cross-entropy). Only the last
        // pipeline stage carries them; on other stages they are size-1 dummies -
        // this is where sharding saves the most (`logits`/`d_logits` are
        // `n·vocab`, ~311 MB each at vocab 152k, block 512).
        let head = shard.head;
        let hd_v = |x: u64| if head { st(x) } else { st(1) };
        // Decode-only builds skip the CE-head buffers (the LM head is applied
        // host-side, see `sample::generate_kv_stream`) and all backward scratch
        // (backward never runs - `train` is forced false), regardless of `head`.
        let hd_or_dummy = |x: u64| if decode_only { st(1) } else { hd_v(x) };
        let bwd = |x: u64| if decode_only { st(1) } else { st(x) };

        // Per-layer linear weights (B7): every layer this shard owns gets its 7
        // projections as a `model::ops::Weight`, built ONCE here. A non-fp32
        // `dt` is handed straight to `Weight::upload` - its own
        // `want.promote(ops.caps().numeric)` is the ONE capability gate
        // (never blindly sending int8/f16 dispatch work to a device that
        // can't execute it, unlike the old `q8.rs` path, which quantized
        // unconditionally regardless of
        // backend). The `else` (fp32) arm does NOT go through `Weight::
        // upload` - it wraps a `.clone()` of the buffer `ps` already holds
        // (a cheap `Arc` bump, `backend_api::DeviceBuffer`'s own doc comment),
        // so the common non-i8 case costs no extra VRAM or re-upload.
        let (dd, hqd, hkvd, ffd) = (d as usize, hq as usize, hkv as usize, ff as usize);
        let dims = |leaf: &str| -> (usize, usize) {
            match leaf {
                "attn.wq.weight" => (hqd, dd),
                "attn.wk.weight" => (hkvd, dd),
                "attn.wv.weight" => (hkvd, dd),
                "attn.wo.weight" => (dd, hqd),
                "mlp.gate.weight" => (ffd, dd),
                "mlp.up.weight" => (ffd, dd),
                "mlp.down.weight" => (dd, ffd),
                other => panic!("qwen: unexpected linear leaf {other}"),
            }
        };
        let mut weights: HashMap<String, Weight> = HashMap::new();
        for l in shard.start..shard.end {
            for leaf in crate::q8::Q8::LINEARS {
                let name = format!("blocks.{l}.{leaf}");
                let (wn, wk) = dims(leaf);
                let w = if quantized {
                    let mut built: Option<Weight> = None;
                    let found = src.with_tensor(&name, &mut |raw| {
                        built = Some(Weight::upload(&ops, raw, wn, wk, dt));
                    });
                    if !found {
                        panic!("qwen: missing init weight {name}");
                    }
                    built.unwrap()
                } else {
                    Weight::F32 { w: ps.w(&name).clone(), n: wn as u32, k: wk as u32 }
                };
                weights.insert(name, w);
            }
        }

        // Incremental-decode KV cache: one [t, kv_dim] key/value buffer per layer.
        // Only meaningful for a whole (single-device) model - `step` asserts that -
        // so allocate for every layer regardless of `shard`.
        let mut kcache = Vec::with_capacity(cfg.n_layers as usize);
        let mut vcache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            kcache.push(st(t as u64 * hkv));
            vcache.push(st(t as u64 * hkv));
        }

        let decode_mrope_cos = st((cfg.head_dim / 2) as u64);
        let decode_mrope_sin = st((cfg.head_dim / 2) as u64);
        let mut m = Qwen {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            shard,
            offload_opt,
            opt,
            tokens,
            targets,
            res,
            layers,
            mm_splice: Cell::new(None),
            img_embeds: st(1),
            d_img_embeds: st(1),
            mrope: Cell::new(false),
            mrope_cos: st(1),
            mrope_sin: st(1),
            weighted: Cell::new(false),
            loss_weights: st(1),
            d_logits_weighted: st(1),
            decode_mrope_cos,
            decode_mrope_sin,
            deepstack: Cell::new(None),
            deepstack_bufs: Vec::new(),
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(bht2),
            // Not ctx-scaled in the decode-only build: unused there (no padded
            // encoder mask during single-token decode), so the only allocation
            // left that scales with ctx is the KV cache below.
            kmask: if decode_only { st(1) } else { st(t as u64) },
            kmask_on: Cell::new(false),
            coop: gpu.caps().workgroup_reductions,
            xn_final: hd_v(n * d),
            logits: hd_or_dummy(n * v),
            dec_logits: std::cell::RefCell::new(None),
            ce_buf: hd_or_dummy(n),
            dres,
            d_logits: hd_or_dummy(n * v),
            ce_stats: hd_or_dummy(n * 2),
            d_xn: bwd(n * d),
            d_tmp: bwd(n * d),
            dxmid: bwd(n * d),
            d_ctx: bwd(n * hq),
            d_scores: bwd(bht2),
            d_q: bwd(n * hq),
            d_k: bwd(n * hkv),
            dq_pre: bwd(n * hq),
            dk_pre: bwd(n * hkv),
            d_v: bwd(n * hkv),
            d_h: bwd(n * ff),
            d_gate_pre: bwd(n * ff),
            d_up: bwd(n * ff),
            inv: bwd(inv_rows),
            lora_a: st(n * r),
            lora_da: st(n * r),
            lora_out: st(n * max_out),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            kcache,
            vcache,
            dec_pos: Cell::new(0),
            ops,
            weights,
            decode_only,
            gpu,
        };
        // Decode-only builds never call the batched forward_steps (its dispatch
        // sizes assume the b·t-sized buffers this build deliberately doesn't
        // have) - the KV-cache decode path builds its own tape per call
        // (`decode_submit`). `forward()`/`run_forward()` assert against being
        // called on a decode-only instance rather than relying on this being empty.
        m.fwd_steps = if decode_only { Vec::new() } else { m.forward_steps(m.b, m.t) };
        m.bwd_steps = if train { m.build_backward_steps() } else { Vec::new() };
        m
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        // `embed_tile.wgsl` (the batched forward's embedding gather) tile-gates
        // its own reads, so an out-of-vocab token here can't OOB-read the way
        // decode_submit's single-token EMBED could - but it silently leaves
        // that position's embedding row at whatever the buffer previously held
        // (no tile ever claims it), which is a correctness bug, not a crash.
        // Same root cause as the decode-path segfault (a checkpoint/tokenizer
        // vocab mismatch); fail loudly here too rather than serving garbage.
        if let Some(&bad) = x.iter().find(|&&t| t as usize >= self.cfg.vocab as usize) {
            panic!("batched-forward token id {bad} exceeds vocab {} (checkpoint/tokenizer mismatch?)", self.cfg.vocab);
        }
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// True if `name` has a gradient buffer (i.e. is optimised). Frozen
    /// parameters (LoRA base, inference) have none, so their weight-gradient
    /// dispatches must be skipped - only the input-gradient (dX) path runs to
    /// keep backprop flowing to lower-layer adapters.
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// Kernel-index map for the shared `model::block` Step-builders.
    fn ids() -> KernelIds {
        KernelIds {
            rmsnorm: RMSNORM,
            rms_inv: RMS_INV,
            rmsnorm_dx: RMSNORM_DX,
            rmsnorm_dw: RMSNORM_DW,
            rope: ROPE,
            rope_bwd: ROPE_BWD,
            gqa_scores: GQA_SCORES,
            gqa_apply: GQA_APPLY,
            attn_softmax: ATTN_SOFTMAX,
            gqa_dscores: GQA_DSCORES,
            gqa_dv: GQA_DV,
            gqa_dq: GQA_DQ,
            gqa_dk: GQA_DK,
            silu_mul: SILU_MUL,
            silu_da: SILU_DA,
            silu_db: SILU_DB,
        }
    }

    /// Kernel-index map for [`block::gqa_decode_step`] - the hoisted twin of
    /// this struct's own original inline KV-cache decode dispatch (`decode_steps`
    /// below), migrated onto `model::block` so `qwen3omnimoe::thinker` (the primitive's
    /// second user) and this, its original owner, share one implementation
    /// instead of two copies that can drift apart.
    fn decode_ids() -> block::GqaDecodeIds {
        block::GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: ATTN_DECODE_SCORES, decode_softmax: DECODE_SOFTMAX, attn_decode_apply: ATTN_DECODE_APPLY }
    }

    /// RMSNorm, choosing the workgroup-per-row kernel wherever the device runs
    /// workgroup reductions. `rmsnorm.wgsl` gives thread *t* row *t*, so a
    /// warp's 32 loads are `dim` floats apart and each 32-byte sector fetched
    /// serves ONE useful float; `rmsnorm_rows` walks a row with 64 threads and
    /// is coalesced by construction. That penalty is per-access, not per-thread,
    /// so it does not go away at prefill row counts - measured on the FLUX.2
    /// text encoder (512 tokens, 28 layers, 112 dispatches) as **an order of
    /// magnitude**.
    /// The reference kernel's epsilon is a hard-coded 1e-6, which is what the
    /// runtime-eps twin is handed here.
    fn rms_step(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
        if self.coop {
            self.gpu.step(RMSNORM_ROWS, &[x, w, out], &[dim, rows, f(1e-6)], rows * 64)
        } else {
            block::rmsnorm_fwd(&self.gpu, &Self::ids(), x, w, out, dim, rows)
        }
    }

    /// Masked-pad GQA attention (the FLUX.2 text-encoder path): scores with an
    /// additive per-key mask, row softmax, context. `block::gqa_fwd_kmask`'s
    /// shape, with the softmax choosing the workgroup-per-row kernel where the
    /// device runs workgroup reductions.
    ///
    /// `softmax_rows` normalises the WHOLE row while `attn_softmax` normalises
    /// `j <= i`, and here the two are identical: `gqa_scores_kmask` already
    /// writes `-3.4e38` into every `j > i` slot, so those terms exponentiate to
    /// 0 and cannot move the max or the sum. (No row is ever fully masked: a
    /// query at position `i` always sees the content keys at `j <= i`, pad
    /// queries included.) One thread per row vs 64 cooperating on one row, at
    /// [B*H*T = 16384, T = 512]: **several times faster** over the encoder's 28
    /// layers.
    fn gqa_kmask_steps(
        &self,
        a: &block::Gqa,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        probs: &DeviceBuffer,
        ctx: &DeviceBuffer,
    ) -> Vec<Step> {
        let rows = a.b * a.n_heads * a.t;
        let p = [a.b, a.n_heads, a.n_kv_heads, a.t, a.head_dim, a.group()];
        let softmax = if self.coop {
            self.gpu.step(SOFTMAX_ROWS, &[&self.scores, probs], &[rows, a.t], rows * 64)
        } else {
            self.gpu.step(ATTN_SOFTMAX, &[&self.scores, probs], &[a.b, a.n_heads, a.t], rows)
        };
        vec![
            self.gpu.step(GQA_SCORES_KMASK, &[q, k, &self.kmask, &self.scores], &p, rows * a.t),
            softmax,
            self.gpu.step(GQA_APPLY, &[probs, v, ctx], &p, rows * a.head_dim),
        ]
    }

    /// GQA shape for `b`×`t` (the buffers are sized for the max `b`/`t`).
    fn gqa(&self, b: u32, t: u32) -> Gqa {
        Gqa { b, t, n_heads: self.cfg.n_heads, n_kv_heads: self.cfg.n_kv_heads, head_dim: self.cfg.head_dim }
    }

    /// One table-driven M-RoPE rotation (`rope2d`) over the q or k buffer (region
    /// offset 0, table rows = token rows). `sign` = 1 forward, -1 = the exact
    /// inverse rotation (backward). Uses the host-written `mrope_cos/sin` tables.
    fn rope2d_step(&self, buf: &DeviceBuffer, rows: u32, heads: u32, head_dim: u32, row_stride: u32, sign: f32) -> Step {
        let half = head_dim / 2;
        self.gpu.step(
            ROPE2D,
            &[buf, &self.mrope_cos, &self.mrope_sin],
            &[rows, heads, half, row_stride, 0, rows, f(sign)],
            rows * heads * half,
        )
    }

    /// RMSNorm backward via the shared builder: input grad always, gain grad only
    /// when the gain is trainable (frozen LoRA base / inference skip it).
    fn rmsnorm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let gw = self.trainable(wname).then(|| self.g(wname));
        s.extend(block::rmsnorm_bwd(&self.gpu, &Self::ids(), x, self.w(wname), dy, dx, &self.inv, gw, dim, rows));
    }

    /// True if a LoRA adapter is configured for the given projection leaf.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg
            .lora
            .as_ref()
            .filter(|lc| lc.targets_leaf(leaf))
            .map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Dispatch one per-layer linear `out = act @ Wᵀ` through the `Ops`
    /// façade (B7): `self.weights[wname]` carries whichever tier `Weight::
    /// upload` picked for this model at construction (uniformly `F32` unless
    /// this model was built int8 AND the device's capability allowed it, in
    /// which case every one of the 7 per-layer linears is `I8`) - the
    /// forward never branches on a separate int8-on/off flag itself, only on
    /// what `self.weights` actually holds. Returns whether the dispatch was
    /// `F32` (LoRA only ever targets an unquantized base weight - `q8.rs`'s
    /// former module doc: "Inference-only (frozen, no LoRA, no backward)" -
    /// so a caller only runs `lora_fwd` when this is `true`, matching this
    /// function's pre-B7 shape exactly: LoRA used to live only in the fp32
    /// arm of the fp32-vs-int8 fork this replaces).
    /// The activation every per-layer linear on this shard reads, packed for
    /// int8 only where an int8 weight will actually read it.
    ///
    /// `Ops::act`'s packing is two dispatches and one `I8Scratch` allocation
    /// per activation, and a decode tape builds four of them per layer per
    /// token - all of it dead on an fp32 model, which is what every
    /// `from_tensors_decode`/`load_inference` build is. The tier is read off
    /// the resident weights rather than off a remembered request, the same
    /// rule `Self::linear_dtype` states, so a tier that silently fell back to
    /// fp32 gets the cheap path it deserves.
    fn ops_act(&self, s: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, k: u32) -> Act {
        if self.weights.values().all(|w| matches!(w, Weight::F32 { .. })) {
            return self.ops.act_f32(x, 0, rows, k);
        }
        self.ops.act(s, x, 0, rows, k)
    }

    fn ops_linear(&self, s: &mut Vec<Step>, act: &Act, wname: &str, out: &DeviceBuffer) -> bool {
        let w = self.weights.get(wname).unwrap_or_else(|| panic!("qwen: no Ops weight for {wname}"));
        self.ops.matmul(s, w, act, out, 0);
        matches!(w, Weight::F32 { .. })
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)·(x·Aᵀ)·Bᵀ`.
    /// No-op for an untargeted leaf. `m`×`k` is the input, `nout` the output.
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(self.gpu.step(MATMUL, &[&self.lora_a, self.w(&bnm), &self.lora_out], &[m, r, nout], m * nout));
        s.push(self.gpu.step(AXPY, &[y, &self.lora_out], &[m * nout, f(scale)], m * nout));
    }

    /// Backward for a (possibly-LoRA) linear `y = x·Wᵀ`. Accumulates the input
    /// gradient into `dx` (flag `acc`). For a full weight: base dW + dX. For a
    /// LoRA-targeted leaf: the base weight is frozen (dX only, no dW) and the
    /// adapter grads gA/gB are produced (scale folded in by scaling `d_out`).
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, s: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        match self.lora_for(leaf) {
            Some((r, scale)) => {
                // base: dx += d_out·W (frozen weight - no dW). d_out is NOT mutated
                // here: for `wo` it is `dxmid`, reused downstream as the residual
                // grad, so the adapter scale is folded into the private scratch.
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[d_out, self.w(wname), dx], &[m, k, nout, acc], bt));
                let a = format!("{wname}.lora_a");
                let bnm = format!("{wname}.lora_b");
                // a = (alpha/r)·(x·Aᵀ)  -> gB += d_outᵀ·a
                s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_a], &[m * r, f(scale)], m * r));
                let (bk, bt) = dw_kernel_bw(nout, r);
                s.push(self.gpu.step(bk, &[d_out, &self.lora_a, self.g(&bnm)], &[m, r, nout], bt));
                // da = (alpha/r)·(d_out·B) -> gA += daᵀ·x ; dx += da·A
                let (bk, bt) = dx_kernel_bw(m, r);
                s.push(self.gpu.step(bk, &[d_out, self.w(&bnm), &self.lora_da], &[m, r, nout, 0], bt));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_da], &[m * r, f(scale)], m * r));
                let (bk, bt) = dw_kernel_bw(r, k);
                s.push(self.gpu.step(bk, &[&self.lora_da, x, self.g(&a)], &[m, k, r], bt));
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[&self.lora_da, self.w(&a), dx], &[m, k, r, 1], bt));
            }
            None => {
                if self.trainable(wname) {
                    let (bk, bt) = dw_kernel_bw(nout, k);
                    s.push(self.gpu.step(bk, &[d_out, x, self.g(wname)], &[m, k, nout], bt));
                }
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[d_out, self.w(wname), dx], &[m, k, nout, acc], bt));
            }
        }
    }

    /// Vocab tiles for the embedding / lm_head (shared `block::vocab_tiles`).
    fn vocab_tiles(&self) -> Vec<(u32, u32)> {
        block::vocab_tiles_on(&self.gpu, self.cfg.vocab as u64, self.cfg.d_model as u64)
    }

    /// The token-embedding gather for `n` token rows, as tiled steps.
    ///
    /// `tok.weight` is bound as SUB-RANGES (`step_sliced`), one per vocab
    /// tile, because a single storage binding is capped at
    /// `max_storage_buffer_binding_size` - which wgpu clamps to `i32::MAX`
    /// (2047 MiB) on EVERY backend, not just small ones. A `[200000, 4096]`
    /// fp32 table is 3.28 GB, so it cannot be bound whole on any card,
    /// including a 24 GB P40 whose `max_buffer_size` is twice the binding
    /// limit. `EMBED_TILE` writes an output element only when its token
    /// falls in the bound tile, and every token belongs to exactly one, so
    /// across the tiles each element is written exactly once.
    ///
    /// This is the ONE embedding gather. The batched forward already
    /// tiled; the decode path did not, and dispatched the untiled `EMBED`
    /// against the whole table - which is fine for a small vocab and a hard
    /// validation error for a large one. That split is why an 8B model with
    /// `vocab = 200000` failed in `create_bind_group` on real hardware
    /// while its own batched forward was perfectly happy.
    fn embed_tiled(&self, g: &Gpu, out: &DeviceBuffer, n: u32) -> Vec<Step> {
        let d = self.cfg.d_model;
        let dw = d as u64;
        self.vocab_tiles()
            .into_iter()
            .map(|(v0, cnt)| {
                g.step_sliced(
                    EMBED_TILE,
                    &[&self.tokens, self.w("tok.weight"), out],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[d, n, v0, cnt],
                    n * d,
                )
            })
            .collect()
    }

    /// The vocab tiles the DECODE lm_head uses: [`Self::vocab_tiles`], re-split
    /// so every tile boundary is [`HEAD_TILE_ALIGN`]-row aligned. See
    /// [`align_head_tiles`] for why the head needs that and the embedding does
    /// not.
    fn head_tiles(&self) -> Vec<(u32, u32)> {
        align_head_tiles(&self.vocab_tiles(), self.cfg.vocab)
    }

    /// The lm_head dispatches for the ONE decode row sitting in `xn_final`:
    /// `out[v] = head[v] · xn_final[0]` over the whole vocab, one dispatch per
    /// entry of `tiles`.
    ///
    /// All three bindings are sub-ranges (`step_sliced`):
    ///
    /// * the head weight, because `max_storage_buffer_binding_size` is clamped
    ///   to `i32::MAX` (2047 MiB) on every backend and a `[200000, 4096]` fp32
    ///   table is 3.28 GB - the same reason [`Self::embed_tiled`] slices;
    /// * the OUTPUT, because the kernels selected here (`matmul_gemv` /
    ///   `matmul_reg3`) write `out[row * n + col]` with `col` local to the
    ///   dispatch, so a tile lands at the right absolute vocab id only if the
    ///   binding itself starts at `v0`. That is what forces [`head_tiles`]'s
    ///   alignment, and it is why this does NOT reuse `matmul_tile` (whose
    ///   `n_off`/`n_full` params exist precisely to avoid an offset output
    ///   binding): `matmul_tile` is one thread per output element with a
    ///   `k`-long serial reduction, so adjacent threads read weight rows
    ///   `d_model` floats apart - uncoalesced, on a dispatch that is pure
    ///   memory traffic over multiple GB.
    ///
    /// The kernel comes from the shared [`block::gemm_variant`] selector, not a
    /// hand-rolled branch. `gemv` is offered only on a device that runs
    /// workgroup reductions: on the CPU JIT the register-tiled kernel is what
    /// `backend-cpu` recognises and routes to its native AVX2 GEMM
    /// (`matmul_gemv` has no such fast path), so offering it there would be a
    /// GPU win paid for on CPU.
    ///
    /// The head stays **fp32** here. It is not dispatched through
    /// `Ops::matmul`, so an f16/int8-packed weight would be read as garbage by
    /// these kernels; `self.weights` (the packed tier) only ever holds the 7
    /// per-layer linears, and `tok.weight`/`lm_head.weight` live in `ps` as
    /// plain f32 - see `linear_kernel`'s own note on why the head was
    /// deliberately left off `Ops`.
    fn head_steps(&self, out: &DeviceBuffer, tiles: &[(u32, u32)]) -> Vec<Step> {
        let d = self.cfg.d_model;
        let dw = d as u64;
        let head = self.cfg.head_weight();
        let gemv = if self.coop { self.gpu.kernel_index("matmul_gemv") } else { None };
        tiles
            .iter()
            .map(|&(v0, cnt)| {
                let (mk, mt) = block::gemm_variant(block::GemmVariants::Fast { gemv, tiled: MATMUL_REG3 }, 1, cnt);
                self.gpu.step_sliced(
                    mk,
                    &[&self.xn_final, self.w(head), out],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (v0 as u64, cnt as u64)],
                    &[1, d, cnt],
                    mt,
                )
            })
            .collect()
    }

    /// The `[vocab]` decode-logits buffer, allocated on first use.
    ///
    /// A decode-only build deliberately sizes `logits` as a size-1 dummy (the
    /// batched `[n, vocab]` slab is what pipeline sharding saves the most on),
    /// so the device head needs an output of its own. One row is 800 kB at
    /// `vocab = 200000` - trivial - but it is still only paid by a caller that
    /// asks for logits on device.
    fn decode_logits_buf(&self) -> DeviceBuffer {
        let mut slot = self.dec_logits.borrow_mut();
        slot.get_or_insert_with(|| self.gpu.storage(self.cfg.vocab as u64)).clone()
    }

    /// [`Self::decode_logits`] against a caller-supplied tiling - the seam the
    /// tests drive to exercise the multi-tile sliced-binding path at a size
    /// that does not need a multi-GB weight.
    fn decode_logits_tiled(&self, tiles: &[(u32, u32)]) -> Vec<f32> {
        assert!(self.shard.head, "Qwen::decode_logits: this shard does not own the lm_head");
        let buf = self.decode_logits_buf();
        self.gpu.submit(&[], &self.head_steps(&buf, tiles));
        self.gpu.read(&buf, self.cfg.vocab as usize)
    }

    /// **The lm_head, on device**, applied to the final-norm hidden state the
    /// last [`Self::step`] / [`Self::step_embed`] / [`Self::prefill`] left in
    /// `xn_final`. Returns `[vocab]` logits.
    ///
    /// The decode entry points return the hidden state and leave the head to
    /// the caller, which is right for a caller that wants the hidden state -
    /// but every caller that wants LOGITS then applied a `[vocab, d_model]`
    /// table on the HOST (`model::hostmath::matvec_par` over a `read_weight`
    /// copy). At `minimaxmusic3`'s dims that table is 3.28 GB, applied twice
    /// per 25 Hz frame: ~6.6 GB of host memory streamed per frame, plus a
    /// full duplicate of the table held in RAM. This is the same GEMV where it
    /// belongs - on the device the weights are already resident on, vocab-tiled
    /// so the binding limit is respected (see [`Self::head_steps`]).
    ///
    /// **Stateful by design**: it reads whatever the last decode submission
    /// wrote, so call it directly after the step whose logits you want. It does
    /// NOT advance the KV-cache position and does not disturb it, so
    /// `let h = lm.step(id); let l = lm.decode_logits();` is exactly
    /// `h`'s logits. [`Self::embed_row`] is likewise safe to interleave (it
    /// never touches `xn_final`).
    ///
    /// The host path is unchanged and still supported - this is an addition,
    /// not a replacement.
    pub fn decode_logits(&self) -> Vec<f32> {
        self.decode_logits_tiled(&self.head_tiles())
    }

    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        assert!(
            !self.decode_only,
            "forward_steps: batched forward called on a decode-only-built Qwen \
             (activations sized for n=1, no logits buffer) - use step/prefill/step_embed instead"
        );
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let ids = Self::ids();
        let ga = self.gqa(b_use, t_use);
        let theta = c.rope_theta;
        let mut s: Vec<Step> = Vec::new();
        let dw = d as u64;
        let tiles = self.vocab_tiles();

        // Token embedding, tiled over vocab so each `tok.weight` binding stays
        // under the backend's max-binding size (GL: 128MB). Only the embed stage
        // runs it; other stages receive `res[start]` from the previous stage.
        if self.shard.embed {
            s.extend(self.embed_tiled(&self.gpu, &self.res[0], n));
            // Vision-language splice: overwrite the image-placeholder rows of the
            // freshly-gathered residual stream with the projected image tokens.
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                s.push(model::vlm::splice_fwd(&self.gpu, SPLICE, &self.img_embeds, &self.res[0], row0 * d, n_rows * d));
            }
        }

        for l in self.shard.start..self.shard.end {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // --- attention --- (projections stay here: they carry LoRA/bias;
            // norms/RoPE/attention-core come from the shared block builders)
            s.push(self.rms_step(&self.res[l], self.w(&p("ln1.weight")), &lb.xn1, d, n));
            // xn1 quantized once (B7: `Ops::act`), shared by q/k/v.
            // NOTE - `Ops::act` quantizes UNCONDITIONALLY, unlike the pre-B7
            // `q8.quant` call this replaces (which only ever ran inside the
            // int8 fork): on an all-`F32` model, this dispatches two small
            // kernels (`max_abs_row`/`quant_pack`) whose output nothing
            // reads. This is `Ops`'s own documented, deliberate limitation
            // (`model::ops`'s module doc: "a call site that never pairs an
            // activation with a quantized weight pays for a quantization it
            // does not use... a reasonable follow-up... deliberately left"),
            // not something introduced here - see the B7 ledger entry for
            // the sizing/measurement discussion.
            let act1 = self.ops_act(&mut s, &lb.xn1, n, d);
            if self.ops_linear(&mut s, &act1, &p("attn.wq.weight"), &lb.q_pre) {
                self.lora_fwd(&mut s, "wq", &lb.xn1, &p("attn.wq.weight"), &lb.q_pre, n, d, hq);
            }
            if self.ops_linear(&mut s, &act1, &p("attn.wk.weight"), &lb.k_pre) {
                self.lora_fwd(&mut s, "wk", &lb.xn1, &p("attn.wk.weight"), &lb.k_pre, n, d, hkv);
            }
            if self.ops_linear(&mut s, &act1, &p("attn.wv.weight"), &lb.v) {
                self.lora_fwd(&mut s, "wv", &lb.xn1, &p("attn.wv.weight"), &lb.v, n, d, hkv);
            }
            // Qwen2 q/k/v projection bias (Qwen3 is bias-free).
            if self.cfg.attn_bias {
                s.push(self.gpu.step(BIAS_ADD, &[&lb.q_pre, self.w(&p("attn.wq.bias"))], &[n, hq], n * hq));
                s.push(self.gpu.step(BIAS_ADD, &[&lb.k_pre, self.w(&p("attn.wk.bias"))], &[n, hkv], n * hkv));
                s.push(self.gpu.step(BIAS_ADD, &[&lb.v, self.w(&p("attn.wv.bias"))], &[n, hkv], n * hkv));
            }
            // Optional per-head QK-RMSNorm (Qwen3); Qwen2 uses q_pre/k_pre directly.
            let (q_buf, k_buf): (&DeviceBuffer, &DeviceBuffer) = if self.cfg.qk_norm {
                s.push(self.rms_step(&lb.q_pre, self.w(&p("attn.q_norm.weight")), &lb.q, hd, n * nh));
                s.push(self.rms_step(&lb.k_pre, self.w(&p("attn.k_norm.weight")), &lb.k, hd, n * nkv));
                (&lb.q, &lb.k)
            } else {
                (&lb.q_pre, &lb.k_pre)
            };
            // Half-split RoPE on q/k (in place on the routed buffers).
            if self.mrope.get() {
                s.push(self.rope2d_step(q_buf, n, nh, hd, hq, 1.0));
                s.push(self.rope2d_step(k_buf, n, nkv, hd, hkv, 1.0));
            } else {
                s.push(block::rope_fwd(&self.gpu, &ids, q_buf, n, nh, hd, hq, t_use, theta));
                s.push(block::rope_fwd(&self.gpu, &ids, k_buf, n, nkv, hd, hkv, t_use, theta));
            }
            if self.kmask_on.get() {
                s.extend(self.gqa_kmask_steps(&ga, q_buf, k_buf, &lb.v, &lb.probs, &lb.ctx));
            } else {
                s.extend(block::gqa_fwd(&self.gpu, &ids, &ga, q_buf, k_buf, &lb.v, &self.scores, &lb.probs, &lb.ctx));
            }
            let act_o = self.ops_act(&mut s, &lb.ctx, n, hq);
            if self.ops_linear(&mut s, &act_o, &p("attn.wo.weight"), &self.proj) {
                self.lora_fwd(&mut s, "wo", &lb.ctx, &p("attn.wo.weight"), &self.proj, n, hq, d);
            }
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // --- SwiGLU MLP ---
            s.push(self.rms_step(&lb.xmid, self.w(&p("ln2.weight")), &lb.xn2, d, n));
            // xn2 quantized once, shared by gate/up.
            let act2 = self.ops_act(&mut s, &lb.xn2, n, d);
            if self.ops_linear(&mut s, &act2, &p("mlp.gate.weight"), &lb.gate_pre) {
                self.lora_fwd(&mut s, "gate", &lb.xn2, &p("mlp.gate.weight"), &lb.gate_pre, n, d, ff);
            }
            if self.ops_linear(&mut s, &act2, &p("mlp.up.weight"), &lb.up) {
                self.lora_fwd(&mut s, "up", &lb.xn2, &p("mlp.up.weight"), &lb.up, n, d, ff);
            }
            s.push(block::swiglu_fwd(&self.gpu, &ids, &lb.gate_pre, &lb.up, &lb.h, n * ff));
            let act_h = self.ops_act(&mut s, &lb.h, n, ff);
            if self.ops_linear(&mut s, &act_h, &p("mlp.down.weight"), &self.mlp_out) {
                self.lora_fwd(&mut s, "down", &lb.h, &p("mlp.down.weight"), &self.mlp_out, n, ff, d);
            }
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
            // DeepStack: add level `l`'s merged vision features into the image rows
            // of this layer's output (level i -> layer i), for l < n_levels.
            if let Some((row0, n_rows, nl)) = self.deepstack.get() {
                if self.shard.embed && (l as u32) < nl {
                    s.push(self.gpu.step(SPLICE_ADD, &[&self.deepstack_bufs[l], &self.res[l + 1]], &[n_rows * d, row0 * d], n_rows * d));
                }
            }
        }

        // Head epilogue (final norm + lm_head + CE): only the head stage.
        if !self.shard.head {
            return s;
        }
        let last = c.n_layers as usize;
        s.push(self.rms_step(&self.res[last], self.w("norm.weight"), &self.xn_final, d, n));
        // lm_head. When the whole vocab fits one tile (v0=0, cnt=v - the common
        // case for a small vocab like the TTS Talker's 3072), it is a plain
        // `[n,d]·[v,d]ᵀ` matmul, so dispatch the size-adaptive fast kernel
        // (`matmul_reg3`) instead of the naive column-tiled `matmul_tile` - the
        // Talker lm_head was ~50 ms (naive) vs ~2 ms (reg2). Only when the weight
        // genuinely exceeds a binding budget do we fall back to the tiled path.
        let head = c.head_weight();
        if tiles.len() == 1 && tiles[0] == (0, v) {
            let (mk, mt) = linear_kernel(n as usize, v as usize);
            s.push(self.gpu.step(mk, &[&self.xn_final, self.w(head), &self.logits], &[n, d, v], mt));
        } else {
            for &(v0, cnt) in &tiles {
                s.push(self.gpu.step_sliced(
                    MATMUL_TILE,
                    &[&self.xn_final, self.w(head), &self.logits],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[n, d, v, v0, cnt],
                    n * cnt,
                ));
            }
        }
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
        s
    }

    pub fn forward(&self) -> f32 {
        assert!(!self.decode_only, "Qwen::forward: batched forward called on a decode-only-built model; use step/prefill/step_embed instead");
        self.gpu.submit(&[], &self.fwd_steps);
        let n = (self.b * self.t) as usize;
        let losses = self.gpu.read(&self.ce_buf, n);
        if self.weighted.get() {
            // Must return the SAME scalar `backward`'s (weighted) `d_logits`
            // differentiates (the `Model::forward` contract), not the plain
            // mean CE - `d_logits_weighted[row] = loss_weights[row] *
            // d_logits[row]` is exactly the gradient of
            // `Σ loss_weights[i]·ce_loss[i] / count` (per-row terms don't
            // cross, so a per-row scalar factor commutes with `d/d(logits)`),
            // never of `Σ ce_loss[i] / count`. Read back `loss_weights`
            // itself rather than caching the host `Vec<f32>` from
            // `write_weights` - one extra small host read per forward, kept
            // simple and impossible to let drift from what `backward` reads.
            let w = self.gpu.read(&self.loss_weights, n);
            losses.iter().zip(&w).map(|(l, wi)| l * wi).sum::<f32>() / self.count.get()
        } else {
            losses.iter().sum::<f32>() / self.count.get()
        }
    }

    pub fn backward(&self) {
        assert!(!self.decode_only, "Qwen::backward: batched backward called on a decode-only-built model (no backward buffers were allocated)");
        let n = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        assert!(!self.decode_only, "build_backward_steps: no backward buffers on a decode-only-built Qwen");
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let theta = c.rope_theta;
        let head = c.head_weight();
        let b = self.b;
        let t = self.t;
        let ids = Self::ids();
        let ga = self.gqa(b, t);
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final norm ---- (head stage only; other stages receive
        // dres[end] from the next stage and start straight at the layer loop)
        if self.shard.head {
            // Two-pass CE gradient: compute per-row softmax stats ONCE (ce_stats),
            // then the per-element gradient reads them - O(rows*vocab) instead of
            // the naive per-element softmax recompute's O(rows*vocab^2). At vocab
            // 151936 this is the difference between ~10 ms and ~56 s per backward.
            s.push(self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &self.ce_stats], &[n, v, IGNORE], n));
            s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.ce_stats, &self.d_logits], n * v));
            // `enable_weighted_loss()`-opt-in only: scale the freshly-computed
            // per-position CE gradient by `self.loss_weights` (`scale_row.wgsl`,
            // NOT in-place - see that kernel's own doc comment) into
            // `d_logits_weighted`, then read THAT everywhere downstream reads
            // what would otherwise be the raw `d_logits`. An instance that
            // never called `enable_weighted_loss` never pushes this step and
            // pays no extra dispatch (matches `model::Batch::LmWeighted`'s doc
            // comment: ordinary training pays zero extra kernel dispatches).
            let d_logits_bw: &DeviceBuffer = if self.weighted.get() {
                s.push(self.gpu.step(SCALE_ROW, &[&self.d_logits, &self.loss_weights, &self.d_logits_weighted], &[n * v, v], n * v));
                &self.d_logits_weighted
            } else {
                &self.d_logits
            };
            if self.trainable(head) {
                let (bk, bt) = dw_kernel_bw(v, d);
                s.push(self.gpu.step(bk, &[d_logits_bw, &self.xn_final, self.g(head)], &[n, d, v], bt));
            }
            let (bk, bt) = dx_kernel_bw(n, d);
            s.push(self.gpu.step(bk, &[d_logits_bw, self.w(head), &self.d_xn], &[n, d, v, 0], bt));
            let last = c.n_layers as usize;
            self.rmsnorm_bwd(&mut s, &self.res[last], "norm.weight", &self.d_xn, &self.dres[last], d, n);
        }

        for l in (self.shard.start..self.shard.end).rev() {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- SwiGLU MLP backward (input grad = dres[l+1]) ----
            self.proj_bwd(&mut s, "down", &self.dres[l + 1], &lb.h, &p("mlp.down.weight"), &self.d_h, n, ff, d, 0);
            s.extend(block::swiglu_bwd(&self.gpu, &ids, &lb.gate_pre, &lb.up, &self.d_h, &self.d_gate_pre, &self.d_up, n * ff));
            self.proj_bwd(&mut s, "up", &self.d_up, &lb.xn2, &p("mlp.up.weight"), &self.d_xn, n, d, ff, 0);
            self.proj_bwd(&mut s, "gate", &self.d_gate_pre, &lb.xn2, &p("mlp.gate.weight"), &self.d_xn, n, d, ff, 1);
            self.rmsnorm_bwd(&mut s, &lb.xmid, &p("ln2.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // ---- attention backward (input grad = dxmid) ----
            self.proj_bwd(&mut s, "wo", &self.dxmid, &lb.ctx, &p("attn.wo.weight"), &self.d_ctx, n, hq, d, 0);
            // The roped q/k live in q/k (QK-norm) or q_pre/k_pre (Qwen2).
            let (q_buf, k_buf): (&DeviceBuffer, &DeviceBuffer) =
                if self.cfg.qk_norm { (&lb.q, &lb.k) } else { (&lb.q_pre, &lb.k_pre) };
            s.extend(block::gqa_bwd(
                &self.gpu, &ids, &ga, q_buf, k_buf, &lb.v, &lb.probs, &self.d_ctx, &self.d_scores, &self.d_q, &self.d_k, &self.d_v,
            ));
            // RoPE backward (in place on d_q/d_k)
            if self.mrope.get() {
                s.push(self.rope2d_step(&self.d_q, n, nh, hd, hq, -1.0));
                s.push(self.rope2d_step(&self.d_k, n, nkv, hd, hkv, -1.0));
            } else {
                s.push(block::rope_bwd(&self.gpu, &ids, &self.d_q, n, nh, hd, hq, t, theta));
                s.push(block::rope_bwd(&self.gpu, &ids, &self.d_k, n, nkv, hd, hkv, t, theta));
            }
            // Optional QK-norm backward -> dq_pre/dk_pre; else d_q/d_k is the
            // projection-output grad directly.
            let (dq_buf, dk_buf): (&DeviceBuffer, &DeviceBuffer) = if self.cfg.qk_norm {
                self.rmsnorm_bwd(&mut s, &lb.q_pre, &p("attn.q_norm.weight"), &self.d_q, &self.dq_pre, hd, n * nh);
                self.rmsnorm_bwd(&mut s, &lb.k_pre, &p("attn.k_norm.weight"), &self.d_k, &self.dk_pre, hd, n * nkv);
                (&self.dq_pre, &self.dk_pre)
            } else {
                (&self.d_q, &self.d_k)
            };
            // Qwen2 q/k/v bias grad = row-sum of each projection-output grad.
            if self.cfg.attn_bias {
                if self.trainable(&p("attn.wq.bias")) {
                    s.push(self.gpu.step(BIAS_GRAD, &[dq_buf, self.g(&p("attn.wq.bias"))], &[n, hq], hq));
                }
                if self.trainable(&p("attn.wk.bias")) {
                    s.push(self.gpu.step(BIAS_GRAD, &[dk_buf, self.g(&p("attn.wk.bias"))], &[n, hkv], hkv));
                }
                if self.trainable(&p("attn.wv.bias")) {
                    s.push(self.gpu.step(BIAS_GRAD, &[&self.d_v, self.g(&p("attn.wv.bias"))], &[n, hkv], hkv));
                }
            }
            // q/k/v projection backward -> accumulate into d_xn (= grad wrt xn1)
            self.proj_bwd(&mut s, "wv", &self.d_v, &lb.xn1, &p("attn.wv.weight"), &self.d_xn, n, d, hkv, 0);
            self.proj_bwd(&mut s, "wk", dk_buf, &lb.xn1, &p("attn.wk.weight"), &self.d_xn, n, d, hkv, 1);
            self.proj_bwd(&mut s, "wq", dq_buf, &lb.xn1, &p("attn.wq.weight"), &self.d_xn, n, d, hq, 1);
            // ln1 backward -> d_tmp ; dres[l] = dxmid + d_tmp
            self.rmsnorm_bwd(&mut s, &self.res[l], &p("ln1.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // Vision-language splice backward: move the image rows' residual grad into
        // `d_img_embeds` and ZERO them in dres[0] BEFORE emb_bwd, so the scatter
        // below never trains the placeholder token's embedding row.
        if self.shard.embed {
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                s.push(model::vlm::splice_bwd(&self.gpu, SPLICE_BWD, &self.dres[0], &self.d_img_embeds, row0 * d, n_rows * d));
            }
        }

        // embedding backward (tied: accumulates onto the head grad in tok.weight);
        // only the embed stage, which owns the embedding rows and dres[0].
        if self.shard.embed && self.trainable("tok.weight") {
            s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("tok.weight")], &[n, d, v], v * d));
        }
        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // GPU optimiser for `Trainable` params.
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
        // Host (RAM-resident) optimiser for `Offload` params - built lazily on the
        // first step from the store's current weights.
        if !self.ps.offload.is_empty() {
            let mut slot = self.offload_opt.borrow_mut();
            if slot.is_none() {
                *slot = Some(optim::OffloadAdam::new(&self.gpu, &self.ps));
            }
            slot.as_mut().unwrap().step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
        }
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }

    // ---- pipeline-parallel cross-stage seam ----

    /// Residual-stream element count at a stage boundary (`b·t·d_model`).
    fn res_numel(&self) -> usize {
        (self.b * self.t) as usize * self.cfg.d_model as usize
    }
    /// Does this stage hold parameter `name` (weight buffer present)?
    pub fn has_param(&self, name: &str) -> bool {
        self.ps.weight.contains_key(name)
    }
    /// Run the forward graph without reading the loss (non-head stages).
    pub fn run_forward(&self) {
        assert!(!self.decode_only, "Qwen::run_forward: batched forward called on a decode-only-built model");
        self.gpu.submit(&[], &self.fwd_steps);
    }
    /// Read this stage's OUTPUT residual `res[end]` (for the next stage's input).
    pub fn read_out_res(&self) -> Vec<f32> {
        self.gpu.read(&self.res[self.shard.end], self.res_numel())
    }
    /// Write this stage's INPUT residual `res[start]` (from the previous stage).
    pub fn write_in_res(&self, data: &[f32]) {
        self.gpu.write(&self.res[self.shard.start], bytemuck::cast_slice(data));
    }

    // ---- vision-language embedding splice seam ----

    /// Enable the VLM embedding splice at residual rows `[row0, row0+n_rows)`:
    /// after the text token-embedding gather the forward overwrites those rows
    /// with the image tokens written via [`Self::write_img_embeds`], and the
    /// backward routes their gradient to [`Self::read_d_img_embeds`] (zeroing them
    /// in dres[0] so `emb_bwd` never trains the placeholder token). Reallocates the
    /// image buffers and rebuilds the fwd/bwd graphs - call once after construction
    /// (before the first forward). No effect on `tok.weight`/other params.
    pub fn enable_mm_splice(&mut self, row0: u32, n_rows: u32) {
        let sz = (n_rows * self.cfg.d_model) as u64;
        self.img_embeds = self.gpu.storage(sz);
        self.d_img_embeds = self.gpu.storage(sz);
        self.mm_splice.set(Some((row0, n_rows)));
        // A decode-only build (`from_reader_decode`/`from_tensors_decode`) never
        // runs `forward()`/`backward()` (both assert `!self.decode_only`) - only
        // the incremental KV-cache decode path, which reads the state just set
        // above, not `fwd_steps`. Rebuilding the batched step graph here would
        // bind the quadratic (non-decode-shaped) `scores`/`logits` buffers this
        // build never allocated at their real batched-forward size (they're the
        // `st(1)` dummies `new_impl` gives a decode-only build) - dead work at
        // best, an oversized/undersized-buffer bind-group mismatch at worst.
        if !self.decode_only {
            self.fwd_steps = self.forward_steps(self.b, self.t);
            if !self.bwd_steps.is_empty() {
                self.bwd_steps = self.build_backward_steps();
            }
        }
    }

    /// Number of spliced image embedding elements (`n_rows·d_model`); 0 if off.
    fn img_numel(&self) -> usize {
        self.mm_splice.get().map_or(0, |(_, n)| (n * self.cfg.d_model) as usize)
    }

    /// Write the projected image tokens `[n_rows, d_model]` (row-major) to splice
    /// into the residual stream on the next forward.
    pub fn write_img_embeds(&self, data: &[f32]) {
        self.gpu.write(&self.img_embeds, bytemuck::cast_slice(data));
    }

    /// Read the gradient of the spliced image embeddings after `backward` - feeds
    /// the vision connector/encoder backward.
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    /// The splice INPUT buffer itself, for a vision tower sharing THIS decoder's
    /// [`Gpu`] - write into it with a `Step` and the embedding never leaves the
    /// device. [`Self::write_img_embeds`] is the cross-device path (`qwenvl`
    /// runs its tower on a second, possibly CPU-backed device and must round
    /// trip through the host); this accessor is purely additive and changes
    /// nothing about that. Valid only after [`Self::enable_mm_splice`] - before
    /// that this is the 1-float placeholder the constructor allocates.
    pub fn img_embeds_buf(&self) -> &DeviceBuffer {
        &self.img_embeds
    }

    /// The splice GRADIENT buffer itself - the device-side counterpart of
    /// [`Self::read_d_img_embeds`], so a same-device vision tower's backward can
    /// consume it as an input buffer instead of re-uploading a host `Vec`.
    /// Same validity rule as [`Self::img_embeds_buf`].
    pub fn d_img_embeds_buf(&self) -> &DeviceBuffer {
        &self.d_img_embeds
    }

    // ---- interleaved M-RoPE seam (Qwen3-VL) ----

    /// Switch q/k to the table-driven `rope2d` M-RoPE path (from the analytic
    /// rope_base). Allocates the `[b·t, head_dim/2]` cos/sin tables and rebuilds
    /// the fwd/bwd graphs - call once after construction, then supply the tables
    /// each batch via [`Self::write_mrope_tables`] (computed by
    /// `qwen3vl::mrope::{get_rope_index, mrope_tables}`).
    pub fn enable_mrope(&mut self) {
        let sz = (self.b * self.t * self.cfg.head_dim / 2) as u64;
        self.mrope_cos = self.gpu.storage(sz);
        self.mrope_sin = self.gpu.storage(sz);
        self.mrope.set(true);
        // See `enable_mm_splice`'s comment: a decode-only build never executes
        // `fwd_steps`, only the KV-cache decode path (which reads `self.mrope`
        // directly), so rebuilding the batched graph here is skipped for it.
        if !self.decode_only {
            self.fwd_steps = self.forward_steps(self.b, self.t);
            if !self.bwd_steps.is_empty() {
                self.bwd_steps = self.build_backward_steps();
            }
        }
    }

    // ---- reward/advantage-weighted CE gradient (model::Batch::LmWeighted) ----

    /// Opt into per-position weighted CE gradient - call once after
    /// construction (before the first `backward`); every ordinary
    /// (unweighted) `Qwen` pays zero extra dispatch, same opt-in shape as
    /// [`Self::enable_mrope`]/[`Self::enable_mm_splice`]. Once enabled,
    /// `backward` always routes `d_logits` through `scale_row.wgsl` - a
    /// `model::Batch::Lm` batch on an enabled instance implicitly weights
    /// every position `1.0` (via [`Self::set_batch`]'s `Batch::Lm` arm),
    /// reproducing the unweighted gradient exactly; `model::Batch::LmWeighted`
    /// supplies real per-position weights.
    pub fn enable_weighted_loss(&mut self) {
        let n = (self.b * self.t) as u64;
        let v = self.cfg.vocab as u64;
        self.loss_weights = self.gpu.storage(n);
        self.d_logits_weighted = self.gpu.storage(n * v);
        self.weighted.set(true);
        if !self.bwd_steps.is_empty() {
            self.bwd_steps = self.build_backward_steps();
        }
    }

    /// Write the per-position CE gradient weights (`[b·t]`) for the next
    /// `backward`. Panics if [`Self::enable_weighted_loss`] was never called - the
    /// same "opt in before use" contract as the M-RoPE/VLM splice setters above.
    pub fn write_weights(&self, weights: &[f32]) {
        assert!(self.weighted.get(), "Qwen::write_weights: call enable_weighted_loss() first");
        assert_eq!(weights.len(), (self.b * self.t) as usize, "Qwen::write_weights: expected {} weights, got {}", self.b * self.t, weights.len());
        self.gpu.write(&self.loss_weights, bytemuck::cast_slice(weights));
    }

    /// Write the per-token M-RoPE cos/sin tables (`[b·t, head_dim/2]` row-major)
    /// for the next forward.
    pub fn write_mrope_tables(&self, cos: &[f32], sin: &[f32]) {
        self.gpu.write(&self.mrope_cos, bytemuck::cast_slice(cos));
        self.gpu.write(&self.mrope_sin, bytemuck::cast_slice(sin));
    }

    // ---- DeepStack seam (Qwen3-VL) ----

    /// Enable DeepStack: `n_levels` merged vision-feature buffers (each `[n_rows,
    /// d_model]`) added into the image rows `[row0, row0+n_rows)` right after
    /// decoder layers `0..n_levels`. Allocates the level buffers and rebuilds the
    /// forward graph. The add is linear, so the decoder's parameter backward is
    /// unchanged (no backward step is emitted); a full-tower finetune would gather
    /// the residual grad at the image rows for the DeepStack merger grads.
    pub fn enable_deepstack(&mut self, row0: u32, n_rows: u32, n_levels: u32) {
        let sz = (n_rows * self.cfg.d_model) as u64;
        self.deepstack_bufs = (0..n_levels).map(|_| self.gpu.storage(sz)).collect();
        self.deepstack.set(Some((row0, n_rows, n_levels)));
        // See `enable_mm_splice`'s comment: skip the batched-graph rebuild for
        // a decode-only build - `decode_steps` reads `self.deepstack` directly.
        if !self.decode_only {
            self.fwd_steps = self.forward_steps(self.b, self.t);
        }
    }

    /// Write DeepStack level `level`'s merged features `[n_rows, d_model]` for the
    /// next forward.
    pub fn write_deepstack(&self, level: usize, data: &[f32]) {
        self.gpu.write(&self.deepstack_bufs[level], bytemuck::cast_slice(data));
    }
    /// Run the backward graph. The head stage refreshes the CE-grad uniform first
    /// (it drives `ce_grad_stats`); other stages consume `dres[end]` written by
    /// [`Self::write_out_dres`].
    pub fn run_backward(&self) {
        assert!(!self.decode_only, "Qwen::run_backward: batched backward called on a decode-only-built model (no backward buffers were allocated)");
        if self.shard.head {
            let n = self.b * self.t;
            self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, f(self.count.get())]);
        }
        self.gpu.submit(&[], &self.bwd_steps);
    }
    /// Read this stage's INPUT-side residual grad `dres[start]` (for the previous stage).
    pub fn read_in_dres(&self) -> Vec<f32> {
        self.gpu.read(&self.dres[self.shard.start], self.res_numel())
    }
    /// Write this stage's OUTPUT-side residual grad `dres[end]` (from the next stage).
    pub fn write_out_dres(&self, data: &[f32]) {
        self.gpu.write(&self.dres[self.shard.end], bytemuck::cast_slice(data));
    }
    /// Overwrite gradient buffer `name` (used to write back a summed tied grad).
    pub fn write_grad(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.g(name), bytemuck::cast_slice(data));
    }

    /// Build the host offload optimiser on first use (full-offload training).
    fn ensure_offload(&self) {
        if self.ps.offload.is_empty() {
            return;
        }
        let mut slot = self.offload_opt.borrow_mut();
        if slot.is_none() {
            *slot = Some(optim::OffloadAdam::new(&self.gpu, &self.ps));
        }
    }
    /// Sum-of-squares of this stage's offloaded grads, excluding `exclude`
    /// (the pipeline excludes a replicated tied weight on all but one stage so it
    /// is counted exactly once in the global grad-norm).
    pub fn grad_sq(&self, exclude: &[&str]) -> f64 {
        self.ensure_offload();
        self.offload_opt
            .borrow()
            .as_ref()
            .map(|o| o.grad_sq(&self.gpu, &self.ps, exclude))
            .unwrap_or(0.0)
    }
    /// AdamW step over this stage's offloaded params, scaling grads by a
    /// caller-supplied (globally-reduced) `scale`. Keeps tied replicas identical.
    pub fn opt_step_scaled(&self, t: u32, lr: f32, wd: f32, scale: f32) {
        self.ensure_offload();
        if let Some(o) = self.offload_opt.borrow_mut().as_mut() {
            o.step_with_scale(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, scale);
        }
    }

    /// The maximum sequence length this instance was sized for (the `t` it was
    /// built/loaded with) - generation must keep its context within this.
    pub fn ctx_len(&self) -> usize {
        self.t as usize
    }

    /// Logits for every position of a single sequence (B must be 1, t>=len).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "qwen decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab) as usize)
    }

    /// Hidden state (residual stream) at a given depth for a single sequence,
    /// row-major `[len·d_model]`. `layer` indexes the residual buffer: `res[0]`
    /// is the token embedding, `res[l]` (1..=n_layers) is the output of block
    /// `l-1` (pre-final-norm). This is what diffusion text encoders consume -
    /// Z-Image/FLUX.2 use the **penultimate** hidden state (transformers
    /// `hidden_states[-2]`), which is `res[n_layers-1]` (see [`Self::encode`]).
    ///
    /// Runs the full forward with `IGNORE` targets (as [`Self::logits_all`]
    /// does, so the masked CE is a safe no-op); the lm_head is computed but its
    /// result is unused. B must be 1 and `t >= len`.
    pub fn encode_hidden(&self, tokens: &[u32], layer: usize) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "qwen decoder sized too small");
        assert!(layer <= self.cfg.n_layers as usize, "layer {layer} > n_layers");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.res[layer], (t_use * self.cfg.d_model) as usize)
    }

    /// The **penultimate** hidden state (`res[n_layers-1]`, un-normed) - the
    /// caption features Z-Image and FLUX.2 feed to the DiT (diffusers
    /// `text_encoder(...).hidden_states[-2]`). Returns row-major `[len·d_model]`.
    pub fn encode(&self, tokens: &[u32]) -> Vec<f32> {
        self.encode_hidden(tokens, self.cfg.n_layers as usize - 1)
    }

    /// [`Self::encode_hiddens`] over a **right-padded** sequence, reproducing
    /// the HF `attention_mask` semantics: tokens `content_len..` are pad
    /// queries with real outputs (they rope at their own positions and attend
    /// the content), but are **excluded as keys** for every query. Without
    /// this, pad-row features diverge wildly from the reference (the FLUX.2
    /// text encoder feeds all 512 rows - pads included - to the DiT unmasked,
    /// so parity requires the masked values).
    pub fn encode_hiddens_padded(
        &self,
        tokens: &[u32],
        content_len: usize,
        layers: &[usize],
    ) -> Vec<Vec<f32>> {
        self.arm_pad_kmask(tokens, content_len);
        let out = self.encode_hiddens(tokens, layers);
        self.disarm_kmask();
        out
    }

    /// [`Self::encode`] over a **right-padded** sequence - the penultimate
    /// hidden with the same HF-`attention_mask` semantics as
    /// [`Self::encode_hiddens_padded`] (pads excluded as keys). What a
    /// fixed-`cap_len` caption encoder (Z-Image's resident pipeline) needs
    /// for a short prompt to be sound.
    pub fn encode_padded(&self, tokens: &[u32], content_len: usize) -> Vec<f32> {
        self.arm_pad_kmask(tokens, content_len);
        let out = self.encode(tokens);
        self.disarm_kmask();
        out
    }

    /// Arm the per-key pad mask (`tokens[content_len..]` excluded as keys)
    /// for the next forward(s) - public so a SPLIT encoder (two `Qwen`
    /// shards run back to back, e.g. `s3dit::pipeline::Encoder::Split`) can
    /// arm both halves around its manual `run_forward` sequence. Pair with
    /// [`Self::disarm_kmask`].
    pub fn arm_pad_kmask(&self, tokens: &[u32], content_len: usize) {
        assert!(content_len <= tokens.len());
        let mut mask = vec![0.0f32; self.t as usize];
        for m in mask[content_len..tokens.len()].iter_mut() {
            *m = -3.4e38;
        }
        self.gpu.write(&self.kmask, bytemuck::cast_slice(&mask));
        self.kmask_on.set(true);
    }

    /// Disarm the pad mask - see [`Self::arm_pad_kmask`].
    pub fn disarm_kmask(&self) {
        self.kmask_on.set(false);
    }

    /// Several hidden-state taps from **one** forward, each row-major
    /// `[len·d_model]` in the order requested. FLUX.2 Klein concatenates
    /// `hidden_states[9|18|27]` per token - with [`Self::encode_hidden`] that
    /// would be three full forwards; every `res[l]` buffer is live after a
    /// single pass, so this reads them all.
    pub fn encode_hiddens(&self, tokens: &[u32], layers: &[usize]) -> Vec<Vec<f32>> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "qwen decoder sized too small");
        for &l in layers {
            assert!(l <= self.cfg.n_layers as usize, "layer {l} > n_layers");
        }
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        layers
            .iter()
            .map(|&l| self.gpu.read(&self.res[l], (t_use * self.cfg.d_model) as usize))
            .collect()
    }

    // ---- incremental KV-cache decode (the O(T)/token twin of the O(T²) forward) ----

    /// Reset the incremental KV cache to an empty sequence (the next [`Self::step`]
    /// decodes absolute position 0).
    pub fn reset_cache(&self) {
        self.dec_pos.set(0);
    }

    /// The absolute position the next [`Self::step`] will decode (the cache fill level).
    pub fn cache_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// **Incremental KV-cache decode** of a single new token id at the current cache
    /// position, returning the final-norm hidden state (`[d_model]`) for that token.
    /// This is the `O(T)`-per-token twin of the `O(T²)` full recompute
    /// ([`Self::logits_all`] / [`Self::encode_hidden`]): the same Qwen3 block math,
    /// but the new token's K/V are projected, QK-normed, RoPE'd at the absolute
    /// position, appended to the persistent per-layer cache, and attended by a
    /// single query over positions `0..=pos`. Expressed entirely in the WGSL op set,
    /// so it runs on whatever backend `Gpu` selected (GPU or the wgsl-cpu JIT).
    ///
    /// The token is embedded through the tied `tok.weight` table; apply the (tied)
    /// head to the returned hidden to get logits - `logits[v] = tok.weight[v]·hidden`.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        let pos = self.dec_pos.get();
        let hidden = self.decode_at(Some(token_id), pos, None, None);
        self.dec_pos.set(pos + 1);
        hidden
    }

    /// [`Self::step`], with the q/k rotation from a caller-supplied M-RoPE
    /// table instead of the analytic `rope_at` position -- the decode-path
    /// twin of [`Self::enable_mrope`]/[`Self::write_mrope_tables`], for a
    /// multimodal front-end generating past a spliced prompt (`qwenvl`'s own
    /// `mrope::mrope_tables`, called with a single-element `positions` slice
    /// for this token's real 3-axis position). `cos`/`sin` are `[head_dim/2]`
    /// -- one row, this token only; `qwen3::Qwen` stays agnostic to how the
    /// 3-axis position was derived (that is `qwenvl`'s job, avoiding a
    /// `qwen -> qwenvl` dependency cycle, since `qwenvl` already depends on
    /// `qwen`).
    pub fn step_mrope(&self, token_id: u32, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        self.prefill_mrope(PrefillInput::Token(token_id), cos, sin, None);
        self.gpu.read(&self.xn_final, self.cfg.d_model as usize)
    }

    /// One M-RoPE decode step that leaves its result ON THE DEVICE: the KV
    /// cache is filled and `xn_final` holds the step's final-norm hidden
    /// state, but nothing is read back and the cache position is advanced.
    ///
    /// [`Self::step_mrope`] and [`Self::step_embed_mrope`] are this plus the
    /// `[d_model]` readback they promise, so there is one implementation of
    /// the step and the readback is what differs. A caller that only wants the
    /// cache filled (a multimodal PREFILL, where the prompt is mostly image
    /// rows) or that applies the head on the device ([`Self::decode_logits`])
    /// never looks at that vector, and paying a submit+fence+map round trip
    /// per prompt token to produce it is the same pure waste
    /// [`Self::prefill`]'s own doc describes - this is that entry point with
    /// the M-RoPE table and DeepStack row `prefill` cannot carry.
    ///
    /// `deepstack_row`: see [`Self::decode_steps`].
    pub fn prefill_mrope(&self, input: PrefillInput<'_>, cos: &[f32], sin: &[f32], deepstack_row: Option<u32>) {
        let pos = self.dec_pos.get();
        let token = match input {
            PrefillInput::Token(t) => Some(t),
            PrefillInput::Embed(e) => {
                assert_eq!(e.len(), self.cfg.d_model as usize, "prefill_mrope wants one d_model row");
                // The embedding lands where EMBED would have written it.
                self.gpu.write(&self.res[0], bytemuck::cast_slice(e));
                None
            }
        };
        self.write_decode_mrope_table(cos, sin);
        self.decode_submit(token, pos, Some((&self.decode_mrope_cos, &self.decode_mrope_sin)), deepstack_row);
        self.dec_pos.set(pos + 1);
    }

    /// Write this step's 1-row M-RoPE table -- see [`Self::step_mrope`]'s doc.
    fn write_decode_mrope_table(&self, cos: &[f32], sin: &[f32]) {
        let half = (self.cfg.head_dim / 2) as usize;
        assert_eq!(cos.len(), half, "decode M-RoPE cos table must be [head_dim/2]");
        assert_eq!(sin.len(), half, "decode M-RoPE sin table must be [head_dim/2]");
        self.gpu.write(&self.decode_mrope_cos, bytemuck::cast_slice(cos));
        self.gpu.write(&self.decode_mrope_sin, bytemuck::cast_slice(sin));
    }

    /// Prefill many positions with ONE readback: tokens and raw-embedding rows
    /// interleave freely, every position's K/V lands in the cache, and only
    /// the LAST hidden state is read back. During prefill the intermediate
    /// hiddens are thrown away, so the per-step submit+fence+map round trip -
    /// measured at the top of the caption profile - is pure waste.
    pub fn prefill(&self, inputs: &[PrefillInput<'_>]) -> Vec<f32> {
        assert!(!inputs.is_empty(), "prefill of nothing");
        for input in inputs {
            let pos = self.dec_pos.get();
            match input {
                PrefillInput::Token(t) => {
                    self.decode_submit(Some(*t), pos, None, None);
                }
                PrefillInput::Embed(e) => {
                    assert_eq!(e.len(), self.cfg.d_model as usize, "prefill wants d_model rows");
                    self.gpu.write(&self.res[0], bytemuck::cast_slice(e));
                    self.decode_submit(None, pos, None, None);
                }
            }
            self.dec_pos.set(pos + 1);
        }
        self.gpu.read(&self.xn_final, self.cfg.d_model as usize)
    }

    /// [`Self::step`] from a RAW embedding instead of a token id - the seam a
    /// vision-language front-end feeds image embeddings through: prefill walks
    /// text tokens via `step` and image rows via `step_embed`, and the KV cache
    /// never knows the difference. No residual splice needed on this path.
    pub fn step_embed(&self, embed: &[f32]) -> Vec<f32> {
        assert_eq!(embed.len(), self.cfg.d_model as usize, "step_embed wants one d_model row");
        let pos = self.dec_pos.get();
        // The embedding lands where EMBED would have written it (res[0] row 0).
        self.gpu.write(&self.res[0], bytemuck::cast_slice(embed));
        let hidden = self.decode_at(None, pos, None, None);
        self.dec_pos.set(pos + 1);
        hidden
    }

    /// The raw `tok.weight` embedding row for `token_id` - the SAME gather
    /// [`Self::step`]'s own `EMBED` dispatch uses internally, exposed
    /// standalone (no transformer layers run, the KV-cache position is NOT
    /// advanced) for a caller that needs a token's embedding for something
    /// OTHER than this model's own decode state (e.g. a downstream head
    /// that embeds one of THIS model's vocab ids without asking this
    /// instance to "see" that token itself). Writes into a fresh buffer,
    /// never `self.res[0]`/`self.tokens`' own decode-path role, so this has
    /// no effect on a subsequent [`Self::step`]/[`Self::step_embed`] call.
    pub fn embed_row(&self, token_id: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        self.gpu.write(&self.tokens, &[token_id]);
        let out = self.gpu.storage(d as u64);
        self.gpu.submit(&[], &self.embed_tiled(&self.gpu, &out, 1));
        self.gpu.read(&out, d as usize)
    }

    /// [`Self::step_embed`] with M-RoPE -- see [`Self::step_mrope`]'s doc for
    /// the `cos`/`sin` convention. `deepstack_row`: see [`Self::decode_steps`]'s
    /// doc -- `Some(local_row)` when this embedding is image row `local_row`
    /// on a DeepStack-enabled checkpoint, `None` otherwise (every caller
    /// before this parameter existed, unchanged).
    pub fn step_embed_mrope(&self, embed: &[f32], cos: &[f32], sin: &[f32], deepstack_row: Option<u32>) -> Vec<f32> {
        self.prefill_mrope(PrefillInput::Embed(embed), cos, sin, deepstack_row);
        self.gpu.read(&self.xn_final, self.cfg.d_model as usize)
    }

    /// Record + run the incremental decode tape for one token at absolute `pos`.
    /// Mirrors [`Self::forward_steps`] at `n = 1` (row 0 of the sized scratch),
    /// swapping the batched GQA core for the decode kernels + persistent KV cache.
    /// `mrope`: `Some((cos, sin))` (a 1-row `[1, head_dim/2]` table, e.g. from
    /// [`Self::write_decode_mrope_table`]) switches q/k to the table-driven
    /// `rope2d` path for this step only; `None` keeps the analytic `rope_at`
    /// path every existing caller already uses.
    fn decode_at(&self, token_id: Option<u32>, pos: u32, mrope: Option<(&DeviceBuffer, &DeviceBuffer)>, deepstack_row: Option<u32>) -> Vec<f32> {
        self.decode_submit(token_id, pos, mrope, deepstack_row);
        self.gpu.read(&self.xn_final, self.cfg.d_model as usize)
    }

    /// Record + submit one incremental decode step WITHOUT reading back.
    fn decode_submit(&self, token_id: Option<u32>, pos: u32, mrope: Option<(&DeviceBuffer, &DeviceBuffer)>, deepstack_row: Option<u32>) {
        let s = self.decode_steps(token_id, pos, mrope, deepstack_row);
        self.gpu.submit(&[], &s);
    }

    /// The dispatches of one incremental decode step, in submit order, WITHOUT
    /// submitting them.
    ///
    /// Split out of [`Self::decode_submit`] purely so the profiler can time the
    /// decode tape per kernel kind - `gpu_core::profile` needs a step list, and
    /// the decode tape is rebuilt per token rather than recorded once like
    /// `fwd_steps`, so there was nothing to hand it. Behaviour is unchanged:
    /// `decode_submit` records exactly this and submits it.
    ///
    /// `mrope`: see [`Self::decode_at`]'s doc. `None` reproduces this
    /// function's behaviour before M-RoPE decode support existed, bit-for-bit
    /// (the analytic `rope_at` dispatch is untouched in that branch).
    /// `deepstack_row`: `Some(local_row)` when this step decodes image row
    /// `local_row` (0-based within the spliced image) on a checkpoint with
    /// `enable_deepstack` on - applies that row's per-level residual add.
    /// `None` (every existing caller before this parameter existed, and
    /// every non-image-row step) is a no-op, bit-for-bit unchanged.
    pub fn decode_steps(&self, token_id: Option<u32>, pos: u32, mrope: Option<(&DeviceBuffer, &DeviceBuffer)>, deepstack_row: Option<u32>) -> Vec<Step> {
        assert!(
            self.shard.is_whole(self.cfg.n_layers as usize),
            "KV-cache decode requires a whole (single-device) model"
        );
        assert!(pos < self.t, "decode pos {pos} exceeds ctx_len {}", self.t);
        // A token id the tokenizer produced but this checkpoint's embedding
        // table doesn't cover (a checkpoint/tokenizer vocab mismatch) used to
        // reach EMBED's `emb[tokens[t] * d_model + c]` gather unchecked and
        // read arbitrarily far out of bounds of the embedding buffer - on the
        // CPU JIT (raw pointer arithmetic, no bounds checks: see wgsl-cpu's
        // "we synthesise our own bounds via the kernel's early-return mask"
        // doc comment, which only covers `idx`, never a value READ from a
        // buffer) this reliably segfaults; on GPU the same OOB read is just
        // silently wrong. Fail loudly here instead, at the one point every
        // decode caller (`step`, `step_embed`'s sibling, `prefill`) funnels
        // through.
        if let Some(tid) = token_id {
            assert!(
                (tid as usize) < self.cfg.vocab as usize,
                "decode token id {tid} exceeds vocab {} (checkpoint/tokenizer mismatch?)",
                self.cfg.vocab
            );
        }

        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let half = hd / 2;
        let cap = self.t; // scores/probs row stride (== max cached length)
        let theta = c.rope_theta;
        let ids = Self::ids();
        let decode_ids = Self::decode_ids();
        let g = &self.gpu;
        let w = |name: &str| self.ps.w(name);
        // KV decode is m=1 by construction - the decode regime. Use the
        // workgroup-cooperative kernels (A1/A2: rmsnorm_rows, matmul_gemv)
        // wherever the device executes workgroup reductions; the per-element
        // reference kernels run ONE thread per row here (measured: rmsnorm was
        // a large share of prefill GPU time across 13k single-thread calls). Same policy
        // the serving engine's selector applies, at the always-m=1 call site.
        let fast = g.caps().workgroup_reductions;
        let rms = |s: &mut Vec<Step>, x: &DeviceBuffer, wt: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32| {
            if fast {
                s.push(g.step(RMSNORM_ROWS, &[x, wt, out], &[dim, rows, gpu_core::f(1e-6)], rows * 64));
            } else {
                s.push(block::rmsnorm_fwd(g, &ids, x, wt, out, dim, rows));
            }
        };
        // B7: the fp32-vs-int8 GEMV pick used to be a SECOND, independent
        // copy of the selection a (now-deleted) local `mm` closure made by
        // hand (a bare `fast ? <gemv kernel> : <naive kernel>` branch), with
        // a matching (now-deleted) local `mm8` closure hardcoding the
        // decode-regime int8 GEMV kernel unconditionally and NEVER going
        // through any selector at all - a real, structural divergence from
        // both the batched forward's own int8 path (which - pre-B7 - always
        // dispatched the TILED int8 kernel, even at small `n`) and from
        // `serve.rs`'s tuned selector. `self.ops_linear` (below) now drives
        // EVERY decode-path linear through the same `Ops::matmul` call the
        // batched forward uses, which resolves `select::candidates` for the
        // real `m=1` shape - `WorkgroupPerOutput`/GEMV whenever the device's
        // `workgroup_reductions` capability allows it (true on every real
        // GPU backend, false on the CPU JIT, exactly the `fast` split the
        // hand-written `mm` closure this replaces used to encode), never the
        // tiled kernel at this row count.

        // Embed the token id into res[0] row 0 via the tied table (non-tiled
        // gather); `None` = the caller already wrote a raw embedding there
        // (`step_embed`).
        let mut s: Vec<Step> = Vec::new();
        if let Some(token_id) = token_id {
            g.write(&self.tokens, &[token_id]);
            s.extend(self.embed_tiled(g, &self.res[0], 1));
        }

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // --- attention: project, QK-norm, RoPE-at-pos, append, decode-attend ---
            rms(&mut s, &self.res[l], w(&p("ln1.weight")), &lb.xn1, d, 1);
            // Int8 (m=1): quantize the input row once per distinct input
            // (`Ops::act`), then every linear reading it - decode never
            // applies LoRA (a decode-only build's adapter, if any, was
            // already folded into the base weights before construction -
            // `Self::from_tensors_decode`'s own doc), so `ops_linear`'s
            // returned "was this F32" bool is intentionally unused here.
            let act1 = self.ops_act(&mut s, &lb.xn1, 1, d);
            self.ops_linear(&mut s, &act1, &p("attn.wq.weight"), &lb.q_pre);
            self.ops_linear(&mut s, &act1, &p("attn.wk.weight"), &lb.k_pre);
            self.ops_linear(&mut s, &act1, &p("attn.wv.weight"), &lb.v);
            // Qwen2-style biased projections and Qwen3-style QK-norm, gated
            // exactly as the batched forward gates them.
            if c.attn_bias {
                s.push(g.step(BIAS_ADD, &[&lb.q_pre, w(&p("attn.wq.bias"))], &[1, hq], hq));
                s.push(g.step(BIAS_ADD, &[&lb.k_pre, w(&p("attn.wk.bias"))], &[1, hkv], hkv));
                s.push(g.step(BIAS_ADD, &[&lb.v, w(&p("attn.wv.bias"))], &[1, hkv], hkv));
            }
            // Optional per-head QK-RMSNorm (Qwen3); Qwen2 routes q_pre/k_pre.
            let (q_buf, k_buf): (&DeviceBuffer, &DeviceBuffer) = if c.qk_norm {
                rms(&mut s, &lb.q_pre, w(&p("attn.q_norm.weight")), &lb.q, hd, nh);
                rms(&mut s, &lb.k_pre, w(&p("attn.k_norm.weight")), &lb.k, hd, nkv);
                (&lb.q, &lb.k)
            } else {
                (&lb.q_pre, &lb.k_pre)
            };
            // M-RoPE decode: table-driven rope2d over a 1-row table this
            // step's caller already wrote (write_decode_mrope_table),
            // mirroring qwen3omnimoe::thinker::layer_decode_step's pattern -- a
            // 1-row table needs no separate "decode" kernel. `None` (every
            // existing caller) is bit-for-bit the prior ROPE_AT dispatch.
            match mrope {
                Some((cos, sin)) => {
                    s.push(block::rope2d_fwd(g, ROPE2D, q_buf, cos, sin, 1, nh, hd, hq));
                    s.push(block::rope2d_fwd(g, ROPE2D, k_buf, cos, sin, 1, nkv, hd, hkv));
                }
                None => {
                    s.push(g.step(ROPE_AT, &[q_buf], &[1, nh, hd, hq, 0, pos, f(theta)], nh * half));
                    s.push(g.step(ROPE_AT, &[k_buf], &[1, nkv, hd, hkv, 0, pos, f(theta)], nkv * half));
                }
            }
            // Hoisted to model::block (see Self::decode_ids's doc) -- same
            // append+decode-attend dispatch this function always did, now
            // shared with qwen3omnimoe::thinker instead of duplicated.
            s.extend(block::gqa_decode_step(g, &decode_ids, nh, nkv, hd, pos, cap, q_buf, k_buf, &lb.v, &self.kcache[l], &self.vcache[l], &self.scores, &lb.probs, &lb.ctx));
            let act_o = self.ops_act(&mut s, &lb.ctx, 1, hq);
            self.ops_linear(&mut s, &act_o, &p("attn.wo.weight"), &self.proj);
            s.push(g.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[d], d));
            // --- SwiGLU MLP ---
            rms(&mut s, &lb.xmid, w(&p("ln2.weight")), &lb.xn2, d, 1);
            let act2 = self.ops_act(&mut s, &lb.xn2, 1, d);
            self.ops_linear(&mut s, &act2, &p("mlp.gate.weight"), &lb.gate_pre);
            self.ops_linear(&mut s, &act2, &p("mlp.up.weight"), &lb.up);
            s.push(block::swiglu_fwd(g, &ids, &lb.gate_pre, &lb.up, &lb.h, ff));
            let act_h = self.ops_act(&mut s, &lb.h, 1, ff);
            self.ops_linear(&mut s, &act_h, &p("mlp.down.weight"), &self.mlp_out);
            s.push(g.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[d], d));
            // DeepStack decode: this step's row IS one of the image rows
            // (`deepstack_row = Some(local_row)`, the row's 0-based offset
            // within the image) -- add level `l`'s merged vision feature for
            // THAT row into `res[l+1]`, for `l < n_levels`. Unlike
            // `forward_steps`' whole-range `SPLICE_ADD` (line ~1084, which
            // reads the WHOLE compact `deepstack_bufs[l]` from its own index 0
            // and writes it into the big sequence buffer at `row0*d`), decode
            // needs the OPPOSITE offset direction: `res[l+1]` here is already
            // a single `[d]`-sized row needing no offset, while
            // `deepstack_bufs[l]` (still the full `[n_rows,d]` compact block)
            // must be READ starting at `local_row * d` to pick out this step's
            // own row. `splice_add.wgsl`'s `base` param lands on the
            // DESTINATION only, so it can't express this; `splice_add_offset_src`
            // (a source-offset sibling, added for exactly this) can. This used
            // to be a `step_sliced` bind-group offset on the source buffer,
            // which is semantically right but requires the offset to be a
            // multiple of `min_storage_buffer_offset_alignment` (256B) --
            // `local_row * d` has no such guarantee, and that produced a real
            // wgpu validation failure on hardware enforcing the full 256B
            // limit. A uniform-parameter source offset has
            // no such constraint. `None` (every caller except
            // `step_embed_mrope` on an image row) is a pure no-op, bit-for-bit
            // unchanged from before this parameter existed.
            if let Some(local_row) = deepstack_row {
                if let Some((_row0, n_rows, nl)) = self.deepstack.get() {
                    if (l as u32) < nl {
                        assert!(
                            local_row < n_rows,
                            "decode_steps: deepstack_row {local_row} out of range for n_rows {n_rows}"
                        );
                        s.push(g.step(SPLICE_ADD_OFFSET_SRC, &[&self.deepstack_bufs[l], &self.res[l + 1]], &[d, local_row * d, 0], d));
                    }
                }
            }
        }
        let last = c.n_layers as usize;
        rms(&mut s, &self.res[last], w("norm.weight"), &self.xn_final, d, 1);
        s
    }

    /// The device this model runs on (profiling/observability).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The storage tier the 7 per-layer linears ACTUALLY landed on - read off
    /// the resident `Weight` values themselves, never off a remembered
    /// request. `new_shard_dt(.., Dtype::F16)` on a device whose
    /// `caps.numeric` cannot serve f16 reports `F32` here, because
    /// `Weight::upload`'s `want.promote(caps.numeric)` demoted it; that is the
    /// distinction a caller (or a test) needs to tell "the tier ran" from "the
    /// tier silently fell back". `None` when this shard owns no layer at all
    /// (an embed-only or head-only pipeline stage), since there is then no
    /// linear whose tier could be reported.
    pub fn linear_dtype(&self) -> Option<Dtype> {
        self.weights.values().next().map(|w| w.dtype())
    }

    /// Bytes the resident per-layer linears occupy on the device, at whatever
    /// tier [`Self::linear_dtype`] reports - the packed size `Weight::upload`
    /// actually allocated (`Dtype::per_word()` values to a `u32` word, plus
    /// the `[n]` f32 per-channel scale the `I8`/`Q4` tiers carry), not a
    /// driver VRAM reading and not the fp32 size of the source tensor.
    pub fn linear_weight_bytes(&self) -> u64 {
        self.weights
            .values()
            .map(|w| {
                let elems = w.n() as u64 * w.k() as u64;
                let per_word = w.dtype().per_word() as u64;
                let packed = elems.div_ceil(per_word) * 4;
                let scale = match w {
                    Weight::I8 { .. } | Weight::Q4 { .. } => w.n() as u64 * 4,
                    _ => 0,
                };
                packed + scale
            })
            .sum()
    }

    /// OFFLINE FLOP/OPS cost of the recorded batch forward - walks the step
    /// list, executes nothing. Per this device/stage: a sharded instance
    /// reports only its own layers. The int8 path shows up as `int_ops`
    /// (`matmul_i8_*`), fp32 as `flops`; see `gpu_core::cost`.
    pub fn cost_fwd(&self) -> gpu_core::cost::CostReport {
        self.gpu.cost_of(&self.fwd_steps)
    }

    /// OFFLINE cost of the recorded backward (empty when built for inference).
    pub fn cost_bwd(&self) -> gpu_core::cost::CostReport {
        self.gpu.cost_of(&self.bwd_steps)
    }

    /// The forward dispatches of one batched pass, in submit order.
    ///
    /// Exposed for the PROFILER (`qwen_bench`), not for driving - `forward()`
    /// owns the submit and the readback. `gpu_core::profile` needs the step
    /// list to build the per-kernel-kind table needed
    /// before anyone optimises, and until this existed there was no
    /// way to get one for a decoder LM at all: every recorded qwen number
    /// came from `BRAIN_PROFILE`'s timestamp table on a
    /// synthetic shape. Same contract and same reason as
    /// `vqgan::train::VqganTrainer::fwd_steps`.
    pub fn fwd_steps(&self) -> &[Step] {
        &self.fwd_steps
    }

    /// The backward dispatches of one training step, in submit order (empty
    /// when the model was built for inference). Profiler-only, as above.
    pub fn bwd_steps(&self) -> &[Step] {
        &self.bwd_steps
    }

    pub fn save(&self, path: &str) {
        self.save_with_itos(path, None);
    }

    pub fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        let mut config = self.cfg.to_json();
        if let Some(itos) = itos {
            let arr: Vec<Value> = itos.iter().map(|c| Value::from(c.to_string())).collect();
            config["itos"] = Value::Array(arr);
        }
        checkpoint::save(path, config, &tensors);
    }
}

/// One prefill position: a token id or a raw d_model embedding row.
pub enum PrefillInput<'a> {
    Token(u32),
    Embed(&'a [f32]),
}

// ---- architecture-agnostic Model seam ----

impl model::ModelConfig for QwenConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        QwenConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        QwenConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        QwenConfig::from_json(v)
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.vocab = vocab;
        self.block_size = block_size;
        self.with_defaults()
    }
}

impl model::Model for Qwen {
    type Config = QwenConfig;

    fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Qwen::new(cfg, b, t, init)
    }
    fn init_weights(cfg: &QwenConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &QwenConfig {
        &self.cfg
    }
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => {
                Qwen::set_batch(self, tokens, targets);
                // An `enable_weighted_loss()`-enabled instance always routes
                // `backward` through `scale_row` (see that method's doc
                // comment) - an ordinary `Batch::Lm` on such an instance must
                // reproduce the unweighted gradient exactly, so weight every
                // position 1.0 rather than leaving `self.loss_weights` stale
                // from a previous `Batch::LmWeighted` call.
                if self.weighted.get() {
                    let ones = vec![1.0f32; (self.b * self.t) as usize];
                    self.write_weights(&ones);
                }
            }
            model::Batch::LmWeighted { tokens, targets, weights } => {
                assert!(self.weighted.get(), "qwen3::Qwen: Batch::LmWeighted requires enable_weighted_loss() to have been called first");
                Qwen::set_batch(self, tokens, targets);
                self.write_weights(weights);
            }
            _ => panic!("qwen3::Qwen only supports Batch::Lm / Batch::LmWeighted"),
        }
    }
    fn enable_weighted_loss(&mut self) {
        Qwen::enable_weighted_loss(self)
    }
    fn forward(&self) -> f32 {
        Qwen::forward(self)
    }
    fn backward(&self) {
        Qwen::backward(self)
    }
    fn zero_grads(&self) {
        Qwen::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Qwen::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Qwen::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        // The optimised set: full-training `trainable` plus any `offload` params
        // (both carry a gradient; frozen params do not). LoRA -> adapters only.
        self.ps
            .trainable
            .iter()
            .chain(self.ps.offload.iter())
            .map(|(n, _)| n.clone())
            .collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Qwen::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Qwen::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Qwen::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Qwen::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        Qwen::save(self, path)
    }
    fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        Qwen::save_with_itos(self, path, itos)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// The streaming mmap load (`from_reader_inference`) uploads byte-identical
    /// device weights to the eager whole-model-host-map load (`Qwen::new` over
    /// `by_role("")`) - proving equivalence, not merely that it compiles. Also
    /// pins both to the source init exactly. GPU-gated (testgpu / MOE_SKIP_GPU).
    #[test]
    fn streaming_load_matches_eager() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let cfg_json = cfg.to_json();
        let init = crate::init::init_weights(&cfg, 5);
        // Persist as safetensors - flat 1-D tensors (only the values matter here).
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            init.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64], v.clone())).collect();
        let path = std::env::temp_dir().join(format!("qwen-stream-parity-{}.st", std::process::id()));
        let p = path.to_str().unwrap();
        checkpoint::st::save_safetensors(p, &tensors, &cfg_json, None).unwrap();

        // Eager: whole-model host map -> Qwen::new. Streaming: the `::load_inference`
        // entrypoint (opens a mmap WeightReader, uploads one tensor at a time).
        let eager = Qwen::new(cfg, 1, 8, &checkpoint::load(p).by_role(""));
        let streamed = Qwen::load_inference(p, 1, 8);

        for (name, _) in &eager.ps.params {
            assert_eq!(eager.read_weight(name), streamed.read_weight(name), "weight {name}");
            assert_eq!(&streamed.read_weight(name), &init[name], "streamed {name} vs source");
        }
        std::fs::remove_file(&path).ok();
    }

    /// [`Qwen::embed_row`] returns exactly `tok.weight`'s row `token_id` -
    /// checked directly against the whole embedding table (cheap at
    /// `QwenConfig::tiny()`'s vocab) - and does not disturb a subsequent
    /// [`Qwen::step`]'s own decode state (same hidden state whether or not
    /// `embed_row` was called first).
    #[test]
    fn embed_row_matches_the_embedding_table_and_does_not_disturb_decode_state() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 9);
        let d = cfg.d_model as usize;
        let table = init.get("tok.weight").expect("tiny config has an embedding table");

        let m = Qwen::new(cfg.clone(), 1, 4, &init);
        for token_id in [0u32, 1, cfg.vocab - 1] {
            let got = m.embed_row(token_id);
            let want = &table[token_id as usize * d..(token_id as usize + 1) * d];
            assert_eq!(got, want, "embed_row({token_id}) != tok.weight row {token_id}");
        }

        let a = Qwen::new(cfg.clone(), 1, 4, &init);
        let b = Qwen::new(cfg, 1, 4, &init);
        let _ = a.embed_row(0); // read-only probe; must not perturb decode state
        let hidden_a = a.step(2);
        let hidden_b = b.step(2);
        assert_eq!(hidden_a, hidden_b, "embed_row must not affect a subsequent step()'s result");
    }

    /// Writes `cfg`'s `init` to a temp `.st` file and opens a [`checkpoint::weightio::WeightReader`]
    /// on it - the fixture [`streaming_load_matches_eager`] already established
    /// for exercising the streaming (`from_reader_*`) constructors.
    fn write_reader_fixture(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>, tag: &str) -> std::path::PathBuf {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            init.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64], v.clone())).collect();
        let path = std::env::temp_dir().join(format!("qwen-decode-only-{tag}-{}.st", std::process::id()));
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
        path
    }

    /// Read `n` elements from `buf`, or `None` if that panics (out of bounds) -
    /// lets a test prove a buffer's TRUE extent (not just that it's big enough)
    /// by bracketing a read that must succeed against one that must not.
    fn try_read(gpu: &Gpu, buf: &DeviceBuffer, n: usize) -> Option<Vec<f32>> {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // silence the expected panic's backtrace
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.read(buf, n)));
        std::panic::set_hook(prev);
        r.ok()
    }

    /// [`Qwen::from_reader_decode`] must build activations at `n=1` and
    /// `scores`/`probs` at `n_heads·ctx` (NOT `n_heads·ctx²`) - the KV cache is
    /// the only ctx-scaled allocation. For each buffer, reading exactly the
    /// decode-shaped extent succeeds while reading the old training-shaped
    /// (`b·t` / `ctx²`) extent is out of bounds - proving the buffer genuinely
    /// IS the smaller size, not merely that it's big enough to under-read.
    #[test]
    fn from_reader_decode_sizes_activations_and_scores_for_decode_not_prefill() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny(); // n_heads=4, d_model=16
        let init = crate::init::init_weights(&cfg, 5);
        let path = write_reader_fixture(&cfg, &init, "sizes");
        let ctx = 12u32;
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let dec = Qwen::from_reader_decode(&reader, ctx);
        assert!(dec.decode_only);
        assert_eq!(dec.ctx_len(), ctx as usize);

        let d = cfg.d_model as usize;
        let nh = cfg.n_heads as usize;

        // Activation (n=1): `xn1` is `[1, d_model]`, not `[ctx, d_model]`.
        assert!(try_read(&dec.gpu, &dec.layers[0].xn1, d).is_some(), "xn1 must hold at least one row");
        assert!(try_read(&dec.gpu, &dec.layers[0].xn1, ctx as usize * d).is_none(), "xn1 must NOT be ctx-sized (n=1, not n=b*t)");

        // scores/probs: `n_heads·ctx`, not `n_heads·ctx²`.
        let decode_shaped = nh * ctx as usize;
        let prefill_shaped = nh * ctx as usize * ctx as usize;
        assert!(try_read(&dec.gpu, &dec.scores, decode_shaped).is_some(), "scores must hold n_heads*ctx elements");
        assert!(try_read(&dec.gpu, &dec.scores, prefill_shaped).is_none(), "scores must NOT be n_heads*ctx^2 (the batched-forward shape)");
        assert!(try_read(&dec.gpu, &dec.layers[0].probs, decode_shaped).is_some(), "probs must hold n_heads*ctx elements");
        assert!(try_read(&dec.gpu, &dec.layers[0].probs, prefill_shaped).is_none(), "probs must NOT be n_heads*ctx^2");

        // No `logits`/`d_logits` buffers: dummy (size-1) even though this is a
        // head-carrying (whole) shard, because the LM head is applied host-side.
        assert!(try_read(&dec.gpu, &dec.logits, 1).is_some());
        assert!(try_read(&dec.gpu, &dec.logits, cfg.vocab as usize).is_none(), "logits must be a size-1 dummy on a decode-only build");
        assert!(try_read(&dec.gpu, &dec.d_logits, 1).is_some());
        assert!(try_read(&dec.gpu, &dec.d_logits, cfg.vocab as usize).is_none(), "d_logits must be a size-1 dummy on a decode-only build");

        // The KV cache is the one allocation that DOES scale with ctx.
        let hkv = cfg.kv_dim() as usize;
        assert!(try_read(&dec.gpu, &dec.kcache[0], ctx as usize * hkv).is_some(), "the KV cache must still be ctx-sized");

        std::fs::remove_file(&path).ok();
    }

    /// Calling a batched forward/backward entry point on a decode-only build
    /// must panic loudly (a mis-wire) rather than read/write past the smaller
    /// buffers `from_reader_decode` allocated.
    #[test]
    fn from_reader_decode_batched_entry_points_panic() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 6);
        let path = write_reader_fixture(&cfg, &init, "panics");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let dec = Qwen::from_reader_decode(&reader, 12);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.forward())).is_err(), "forward() must panic on a decode-only build");
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.run_forward())).is_err(), "run_forward() must panic on a decode-only build");
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.logits_all(&[1, 2, 3]))).is_err(), "logits_all() must panic on a decode-only build");
        std::panic::set_hook(prev);

        // The decode API itself still works (this is what the build is FOR).
        let hidden = dec.step(1);
        assert_eq!(hidden.len(), cfg.d_model as usize);

        std::fs::remove_file(&path).ok();
    }

    // ---- device LM head on the decode path ----

    /// A device GEMV reassociates the `d_model`-long reduction, so the gate is
    /// numeric, never bit-identical - and never cosine alone. Cosine is blind
    /// to a uniform scale (an RMSNorm epsilon mutation once scored a clean
    /// 1.000000 and was caught only by the magnitude term), so both must hold.
    const HEAD_COS_FLOOR: f64 = 0.999999;
    const HEAD_REL_L2_CEIL: f64 = 1e-5;

    fn gate_head(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        let (cos, max_abs) = brain_testutil::parity::compare(got, want);
        let rel = brain_testutil::parity::rel_l2(got, want);
        println!("{label}: cosine={cos:.9} rel_l2={rel:.3e} max_abs={max_abs:.3e}");
        assert!(cos >= HEAD_COS_FLOOR, "{label}: cosine {cos:.9} below floor {HEAD_COS_FLOOR}");
        assert!(rel <= HEAD_REL_L2_CEIL, "{label}: rel_l2 {rel:.3e} above ceiling {HEAD_REL_L2_CEIL:.0e}");
    }

    /// The host reference this whole change replaces: `logits[v] =
    /// head[v] · hidden`, exactly what every caller does today.
    fn host_head(dec: &Qwen, head: &[f32], hidden: &[f32]) -> Vec<f32> {
        model::hostmath::matvec_par(head, hidden, dec.cfg.vocab as usize, dec.cfg.d_model as usize)
    }

    /// [`Qwen::decode_logits`] must reproduce the host head every decode caller
    /// applies today, at every decode entry point (`prefill`, `step`,
    /// `step_embed`) and after an interleaved [`Qwen::embed_row`] - which must
    /// not disturb `xn_final`.
    #[test]
    fn decode_logits_matches_the_host_lm_head() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 11);
        let path = write_reader_fixture(&cfg, &init, "decode-logits");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let dec = Qwen::from_reader_decode(&reader, 12);
        let head = dec.read_weight(cfg.head_weight());

        let hidden = dec.prefill(&[PrefillInput::Token(1), PrefillInput::Token(4), PrefillInput::Token(9)]);
        gate_head(&dec.decode_logits(), &host_head(&dec, &head, &hidden), "prefill");

        for tok in [2u32, 7, 3] {
            let hidden = dec.step(tok);
            // An embedding lookup between the step and the head must be a
            // no-op for `xn_final` (the depth-decoder feedback loop in
            // `minimaxmusic3` interleaves exactly this).
            let _ = dec.embed_row(5);
            gate_head(&dec.decode_logits(), &host_head(&dec, &head, &hidden), &format!("step({tok})"));
        }

        let embed: Vec<f32> = (0..cfg.d_model).map(|i| ((i as f32) * 0.037).sin()).collect();
        let hidden = dec.step_embed(&embed);
        gate_head(&dec.decode_logits(), &host_head(&dec, &head, &hidden), "step_embed");

        std::fs::remove_file(&path).ok();
    }

    /// The tiled path is the one that matters (a real head cannot be bound
    /// whole), so it must be exercised, must be PROVEN to have dispatched, and
    /// must agree with the host to the same gate as the single-tile path.
    ///
    /// A vocab of 200 over four 64-aligned tiles puts three non-zero sliced
    /// OUTPUT bindings (word offsets 64/128/192 -> 256/512/768 bytes) through
    /// `create_bind_group`, which is exactly the validation an unaligned tiling
    /// would fail.
    #[test]
    fn the_tiled_decode_head_dispatches_per_tile_and_still_matches_the_host() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig { vocab: 200, ..QwenConfig::tiny() };
        let init = crate::init::init_weights(&cfg, 12);
        let path = write_reader_fixture(&cfg, &init, "decode-logits-tiled");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let dec = Qwen::from_reader_decode(&reader, 12);
        let head = dec.read_weight(cfg.head_weight());
        let tiles = [(0u32, 64u32), (64, 64), (128, 64), (192, 8)];

        let hidden = dec.step(17);
        let want = host_head(&dec, &head, &hidden);

        // The dispatch itself: one step per tile, each a real GEMM/GEMV kernel
        // - never a silent host fallback, and never the uncoalesced
        // `matmul_tile` this path deliberately does not use.
        let out = dec.decode_logits_buf();
        let steps = dec.head_steps(&out, &tiles);
        assert_eq!(steps.len(), tiles.len(), "one dispatch per vocab tile");
        let names: Vec<&str> = steps.iter().map(|s| dec.gpu.kernel_name(s.meta().expect("step meta").kernel).expect("kernel name")).collect();
        let expect = if dec.coop { "matmul_gemv" } else { "matmul_reg3" };
        assert!(names.iter().all(|n| *n == expect), "decode head dispatched {names:?}, expected all {expect:?} (coop={})", dec.coop);

        gate_head(&dec.decode_logits_tiled(&tiles), &want, "tiled head");
        // ...and the tiling the model picks for itself agrees with it.
        gate_head(&dec.decode_logits(), &want, "auto-tiled head");

        std::fs::remove_file(&path).ok();
    }

    /// Every tile boundary a sliced OUTPUT binding can land on must clear the
    /// 256-byte `min_storage_buffer_offset_alignment`, at the real
    /// `minimaxmusic3` head shape as well as at a toy budget - and the tiles
    /// must still cover the vocab exactly and stay inside the budget they were
    /// sized from.
    /// The unaligned tiling `model::block::tiles_with_budget` produces, built
    /// here rather than reached for: that function is private, and widening
    /// `model::block`'s public API to let one test see it would be the tail
    /// wagging the dog. `align_head_tiles` is a pure function of a base
    /// tiling, so feeding it one built to the same rule tests exactly what
    /// production feeds it.
    fn base_tiling(vocab: u64, d_model: u64, budget: u64) -> Vec<(u32, u32)> {
        let rows = (budget / d_model.max(1)).max(1);
        let mut out = Vec::new();
        let mut v0 = 0u64;
        while v0 < vocab {
            let cnt = rows.min(vocab - v0);
            out.push((v0 as u32, cnt as u32));
            v0 += cnt;
        }
        out
    }

    #[test]
    fn head_tiles_are_offset_aligned_and_cover_the_vocab() {
        for (vocab, d_model, budget) in [(200_000u64, 4096u64, 268_304_384u64), (151_936, 1024, 24 * 1024 * 1024), (200, 16, 1024), (23, 16, 24 * 1024 * 1024)] {
            let base = base_tiling(vocab, d_model, budget);
            let tiles = align_head_tiles(&base, vocab as u32);
            let mut next = 0u32;
            for &(v0, cnt) in &tiles {
                assert_eq!(v0, next, "vocab={vocab}: tiles must be contiguous");
                assert!(v0 % HEAD_TILE_ALIGN == 0, "vocab={vocab}: tile offset {v0} is not {HEAD_TILE_ALIGN}-row aligned");
                assert!(cnt as u64 * d_model <= budget.max(HEAD_TILE_ALIGN as u64 * d_model), "vocab={vocab}: tile of {cnt} rows exceeds the budget");
                next += cnt;
            }
            assert_eq!(next as u64, vocab, "vocab={vocab}: tiles must cover the vocab exactly");
        }
    }

    /// REGRESSION (CPU-backend JIT dispatch segfault): a
    /// decode token id the checkpoint's embedding table doesn't cover (a
    /// checkpoint/tokenizer vocab mismatch - e.g. a real BPE tokenizer's
    /// `<|im_start|>`-class special token fed to a tiny synthetic checkpoint)
    /// used to reach `EMBED`'s unchecked `emb[tokens[t]*d_model+c]` gather and
    /// read arbitrarily far out of bounds - 100% reproducible SIGSEGV on the
    /// CPU JIT backend (no bounds checks on a value read FROM a buffer, only
    /// on the invocation index), silently wrong on GPU. Must now be a clean,
    /// catchable panic naming the offending id, on every backend.
    #[test]
    fn step_rejects_a_token_id_outside_the_vocab() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny(); // vocab 23
        let init = crate::init::init_weights(&cfg, 9);
        let path = write_reader_fixture(&cfg, &init, "vocab-guard");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let dec = Qwen::from_reader_decode(&reader, 12);

        // The real Qwen3 tokenizer's `<|im_start|>` id - exactly what a
        // checkpoint/tokenizer mismatch fed into the crash.
        let out_of_vocab = 151644u32;
        assert!(out_of_vocab as usize >= cfg.vocab as usize);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.step(out_of_vocab)));
        std::panic::set_hook(prev);
        let err = res.expect_err("step() must panic on an out-of-vocab token, not read out of bounds");
        let msg = err.downcast_ref::<String>().map(String::as_str).or_else(|| err.downcast_ref::<&str>().copied()).unwrap_or_default();
        assert!(msg.contains("151644") && msg.contains("vocab"), "panic message should name the id and 'vocab': {msg:?}");

        // A valid token id right at the vocab boundary still works normally.
        let hidden = dec.step(cfg.vocab - 1);
        assert_eq!(hidden.len(), cfg.d_model as usize);

        std::fs::remove_file(&path).ok();
    }

    /// Sibling of [`step_rejects_a_token_id_outside_the_vocab`] for the
    /// batched-forward path (`set_batch` → `embed_tile.wgsl`): that kernel
    /// tile-gates its own reads so an out-of-vocab id can't OOB-read the way
    /// the single-token `EMBED` kernel could, but it silently leaves the
    /// position's embedding row un-written (stale/garbage) - a correctness
    /// bug, not a crash. Must also fail loudly instead.
    #[test]
    fn logits_all_rejects_a_token_id_outside_the_vocab() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 10);
        let model = Qwen::new(cfg.clone(), 1, 16, &init);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.logits_all(&[1, 2, 151644])));
        std::panic::set_hook(prev);
        let err = res.expect_err("logits_all() must panic on an out-of-vocab token");
        let msg = err.downcast_ref::<String>().map(String::as_str).or_else(|| err.downcast_ref::<&str>().copied()).unwrap_or_default();
        assert!(msg.contains("151644") && msg.contains("vocab"), "panic message should name the id and 'vocab': {msg:?}");
    }

    /// Every kernel this model can dispatch has a cost formula - pins the
    /// FLOP/OPS accounting against silent drift when `pipelines()` grows.
    #[test]
    fn pipelines_fully_costed() {
        for (name, _) in pipelines() {
            assert!(
                gpu_core::cost::covers(name),
                "kernel '{name}' has no formula in gpu_core::cost::kernel_cost; \
                 add one (its dispatches would otherwise be reported UNCOVERED)"
            );
        }
    }

    /// `pipelines()` (the list `self.ops` is built from, via `Gpu::share` -
    /// see that function's own doc comment) against `model::ops::
    /// REQUIRED_KERNELS` - a pure name-set comparison, no `Gpu`/GPU device
    /// required. Catches drift at `cargo test` time; the same class of bug
    /// `qwen3::serve::ops_kernel_list` had (15 kernels short of
    /// `REQUIRED_KERNELS`), which only surfaced on a live server's first
    /// real request rather than here.
    #[test]
    fn pipelines_has_every_kernel_ops_new_requires() {
        model::ops::assert_kernel_list_complete(pipelines());
    }

    /// `step_embed` must be indistinguishable from `step`: feeding token t's
    /// own embedding row produces the identical hidden state. This is the seam
    /// a VLM front-end trusts when it interleaves image rows with text tokens.
    #[test]
    fn step_embed_matches_step() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let w = crate::init::init_weights(&cfg, 7);
        let m = Qwen::new(cfg.clone(), 1, 16, &w);
        let prompt = [1u32, 5, 3, 9];
        m.reset_cache();
        let mut via_step = Vec::new();
        for &t in &prompt {
            via_step = m.step(t);
        }
        let emb = m.read_weight("tok.weight");
        let d = cfg.d_model as usize;
        let m2 = Qwen::new(cfg, 1, 16, &w);
        m2.reset_cache();
        let mut via_embed = Vec::new();
        for &t in &prompt {
            via_embed = m2.step_embed(&emb[t as usize * d..(t as usize + 1) * d]);
        }
        assert_eq!(via_step, via_embed, "an embedding row must be a perfect stand-in for its token");
    }

    /// Batched-submission prefill must be BIT-IDENTICAL to step-by-step: it is
    /// the same tape minus the per-step readbacks, so any difference means the
    /// submit path depends on the read it no longer does.
    #[test]
    fn prefill_matches_step_by_step() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let w = crate::init::init_weights(&cfg, 7);
        let d = cfg.d_model as usize;
        let m = Qwen::new(cfg.clone(), 1, 16, &w);
        m.reset_cache();
        let emb = m.read_weight("tok.weight");
        // The three token steps only advance the cache; the comparison is on the
        // logits of the final (embedding) step.
        for &t in &[1u32, 5, 3] {
            m.step(t);
        }
        let via_steps = m.step_embed(&emb[9 * d..10 * d]);
        let m2 = Qwen::new(cfg, 1, 16, &w);
        m2.reset_cache();
        let via_prefill = m2.prefill(&[
            PrefillInput::Token(1),
            PrefillInput::Token(5),
            PrefillInput::Token(3),
            PrefillInput::Embed(&emb[9 * d..10 * d]),
        ]);
        assert_eq!(via_steps, via_prefill);
        // And decode continues correctly from a prefilled cache.
        assert_eq!(m.step(2), m2.step(2));
    }

    /// Int8 KV decode (the packed GEMV at m=1) must track the fp32 KV decode
    /// within honest quantisation noise: a scale-handling bug is O(1) relative
    /// error, per-channel int8 noise is well under the 10% gate.
    #[test]
    fn int8_kv_decode_tracks_fp32() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let w = crate::init::init_weights(&cfg, 7);
        let f = Qwen::new(cfg.clone(), 1, 16, &w);
        let q = Qwen::new_shard_i8(cfg.clone(), 1, 16, &w, model::shard::Shard::whole(cfg.n_layers as usize));
        f.reset_cache();
        q.reset_cache();
        let (mut hf, mut hq) = (Vec::new(), Vec::new());
        for &t in &[1u32, 5, 3, 9, 2, 7] {
            hf = f.step(t);
            hq = q.step(t);
        }
        let err: f32 = hf.iter().zip(&hq).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let norm: f32 = hf.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = err / norm.max(1e-12);
        assert!(rel < 0.10, "int8 KV decode diverged from fp32: rel L2 {rel:.4}");
        // `linear_weight_bytes` must account for the tier it actually got: a
        // quarter of fp32 for the packed weights, plus the `[n]` f32
        // per-channel scale int8 (unlike f16) also has to keep resident.
        if q.linear_dtype() == Some(Dtype::I8) {
            let rows: u64 = crate::q8::Q8::LINEARS.iter().map(|leaf| dims_of(&cfg, leaf).0 as u64).sum::<u64>() * cfg.n_layers as u64;
            assert_eq!(q.linear_weight_bytes(), f.linear_weight_bytes() / 4 + rows * 4, "int8 resident linear bytes");
        }
    }

    /// `(n_out, k_in)` of one per-layer linear leaf - the test-side mirror of
    /// `new_impl`'s own `dims` closure, for sizing assertions.
    fn dims_of(cfg: &QwenConfig, leaf: &str) -> (u32, u32) {
        let (d, hq, hkv, ff) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim(), cfg.d_ff);
        match leaf {
            "attn.wq.weight" => (hq, d),
            "attn.wk.weight" | "attn.wv.weight" => (hkv, d),
            "attn.wo.weight" => (d, hq),
            "mlp.gate.weight" | "mlp.up.weight" => (ff, d),
            "mlp.down.weight" => (d, ff),
            other => panic!("unexpected linear leaf {other}"),
        }
    }

    /// Cosine similarity and relative L2 of `a` against the reference `b`.
    /// Both, never cosine alone: cosine is scale-invariant, so a uniformly
    /// mis-scaled output still scores 1.0 - the house rule this repo's other
    /// parity gates (`sdxlunet`, `controlnet`, ...) follow.
    fn cos_rel(a: &[f32], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len(), "cos_rel: length mismatch");
        let (mut dot, mut na, mut nb, mut de) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            de += (x - y) * (x - y);
        }
        (dot / (na.sqrt() * nb.sqrt()).max(1e-30), de.sqrt() / nb.sqrt().max(1e-30))
    }

    /// Every kernel NAME the recorded step list `steps` dispatches on `m`'s
    /// device, in order.
    fn kernel_names(m: &Qwen, steps: &[Step]) -> Vec<String> {
        steps
            .iter()
            .filter_map(|s| s.meta())
            .filter_map(|meta| m.gpu().kernel_name(meta.kernel))
            .map(|n| n.to_string())
            .collect()
    }

    /// A **half-precision STORAGE tier** (`new_shard_dt(.., Dtype::F16 |
    /// Dtype::BF16)`) against the fp32 build, at one model shape. Returns the
    /// distinct `#w=<tag>` kernel names the batched forward dispatched, or
    /// `None` when the device has no such storage path at all (a skip,
    /// already reported), paired with this
    /// device's `workgroup_reductions` capability (which is what decides
    /// WHICH f16 kernel the selector can reach - see the caller).
    ///
    /// Three things are asserted, and the FIRST is what makes the other two
    /// mean anything:
    ///
    /// 1. **The tier really dispatched.** `linear_dtype()` must report `F16`
    ///    AND the recorded batched forward must contain real `#w=f16` kernel
    ///    dispatches - one per per-layer linear - while the fp32 build
    ///    contains none. A silent demotion to fp32 would make a closeness
    ///    check pass while proving nothing; it is accepted ONLY when the
    ///    device genuinely reports no f16 storage path (the same "a fallback
    ///    must imply a missing capability" shape
    ///    `serve::tests::int8_weights_track_fp32` uses for int8).
    /// 2. **Numerics.** f16 storage is LOSSY: binary16 has a 10-bit explicit
    ///    mantissa, so round-to-nearest costs each weight a relative error of
    ///    up to 2^-11 = 4.9e-4. Sign-random accumulation over K keeps a dot
    ///    product at that same relative order rather than growing it, and the
    ///    RMSNorm in front of every projection re-normalizes the stream each
    ///    layer, so the end-to-end error stays within a small multiple of the
    ///    per-weight bound. The gate is **cosine >= 0.9999 and rel_l2 <=
    ///    5e-3** - ~10x headroom over that 4.9e-4 bound (enough to absorb the
    ///    depth factor and a different GEMM reduction order between the two
    ///    kernels), while any REAL defect in this tier - a mis-decoded
    ///    exponent bias, a swapped hi/lo half, packed words read as raw f32 -
    ///    is an O(1) error, orders of magnitude clear of it. bf16 keeps only
    ///    7 mantissa bits, so its bound scales to 4e-2 by the same rule (see
    ///    `rel_max`). Checked on both
    ///    the batched forward and the KV-cache decode (m = 1), which dispatch
    ///    DIFFERENT f16 kernel variants.
    /// 3. **The tier pays for itself**: the resident linear bytes must halve.
    fn half_tier_gate(cfg: QwenConfig, t: u32, toks: &[u32], dt: Dtype) -> Option<(Vec<String>, bool)> {
        type Supported = fn(&gpu_core::NumericSupport) -> bool;
        // `rel_max` is 10x the tier's own worst-case per-weight relative
        // rounding error (2^-(mantissa_bits+1)): f16 keeps 10 explicit
        // mantissa bits -> 4.9e-4 -> 5e-3, bf16 only 7 -> 3.9e-3 -> 4e-2.
        // See this function's doc comment for why 10x is the right headroom.
        let (tag, rel_max, supported): (&str, f64, Supported) = match dt {
            Dtype::F16 => ("#w=f16", 5e-3, |n| n.f16 || n.f16_storage),
            Dtype::BF16 => ("#w=bf16", 4e-2, |n| n.bf16 || n.bf16_storage),
            other => panic!("half_tier_gate: {other:?} is not a half-precision storage tier"),
        };
        let w = crate::init::init_weights(&cfg, 7);
        let whole = model::shard::Shard::whole(cfg.n_layers as usize);
        let f32m = Qwen::new_shard(cfg.clone(), 1, t, &w, false, whole.clone());
        let tierm = Qwen::new_shard_dt(cfg.clone(), 1, t, &w, whole, dt);

        // (1) The requested tier must be what is actually resident.
        let got = tierm.linear_dtype().expect("a whole-model build owns every layer's linears");
        if got != dt {
            let n = &tierm.gpu().caps().numeric;
            assert!(
                !supported(n),
                "device reports {dt:?} storage support ({n:?}) but the build demoted the \
                 weights to {got:?} - a silent fp32 fallback on capable hardware is exactly what this \
                 test exists to catch"
            );
            brain_testutil::skip_unavailable(&format!("{dt:?} storage comparison: device has no {dt:?} weight path"));
            return None;
        }

        // ... and the recorded forward must really run the packed-f16 kernels.
        let want = 7 * cfg.n_layers as usize;
        let fwd_names = kernel_names(&tierm, tierm.fwd_steps());
        let dispatched: Vec<String> = fwd_names.iter().filter(|n| n.ends_with(tag)).cloned().collect();
        assert_eq!(
            dispatched.len(),
            want,
            "expected one `{tag}` dispatch per per-layer linear ({want}), got {} in {fwd_names:?}",
            dispatched.len()
        );
        let f32_names = kernel_names(&f32m, f32m.fwd_steps());
        assert!(
            !f32_names.iter().any(|n| n.contains("#w=")),
            "the fp32 build must dispatch no packed-storage kernel at all: {f32_names:?}"
        );
        // The m=1 KV-cache decode is a SEPARATE step list and a different
        // kernel variant - cover it too, or half the tier stays unproven.
        let dec_names = kernel_names(&tierm, &tierm.decode_steps(Some(1), 0, None, None));
        assert_eq!(
            dec_names.iter().filter(|n| n.ends_with(tag)).count(),
            want,
            "the KV-cache decode path must dispatch {dt:?} too: {dec_names:?}"
        );
        tierm.reset_cache();

        // (2) Numerics: batched forward (logits) and KV-cache decode (hidden).
        let (cos_l, rel_l) = cos_rel(&tierm.logits_all(toks), &f32m.logits_all(toks));
        f32m.reset_cache();
        tierm.reset_cache();
        let (mut hf, mut h16) = (Vec::new(), Vec::new());
        for &tok in &toks[..toks.len() / 2] {
            hf = f32m.step(tok);
            h16 = tierm.step(tok);
        }
        let (cos_h, rel_h) = cos_rel(&h16, &hf);
        let d = cfg.d_model;
        println!(
            "{dt:?} vs f32 (d_model {d}, m {t}): logits cos {cos_l:.10} rel_l2 {rel_l:.3e} | decode hidden cos {cos_h:.10} rel_l2 {rel_h:.3e}"
        );
        for (what, cos, rel) in [("logits", cos_l, rel_l), ("decode hidden", cos_h, rel_h)] {
            assert!(cos >= 0.9999, "{dt:?} {what} cosine {cos:.10} below the 0.9999 floor");
            assert!(rel <= rel_max, "{dt:?} {what} rel_l2 {rel:.3e} above the {rel_max:.0e} bound");
        }

        // (3) Half the bytes - the whole point of the tier.
        let (b32, b16) = (f32m.linear_weight_bytes(), tierm.linear_weight_bytes());
        println!("linear weight bytes: fp32 {b32} -> {dt:?} {b16}");
        assert_eq!(b16 * 2, b32, "{dt:?} linears must be exactly half of fp32 ({b16} vs {b32})");

        let mut distinct: Vec<String> = dispatched;
        distinct.sort();
        distinct.dedup();
        Some((distinct, tierm.gpu.caps().workgroup_reductions))
    }

    /// A deliberately WIDE tiny config: `n >= select::GEMM_TILE_MIN_COLS`
    /// (128) on 5 of the 7 projections and a row count above
    /// `DECODE_REGIME_MAX_ROWS` (32), which is what makes the selector reach
    /// for `KernelVariant::RegisterTiled` - a physically DIFFERENT f16 kernel
    /// (`matmul_reg3#w=f16`) from the one `QwenConfig::tiny()`'s narrow
    /// projections take. `wk`/`wv` stay at n = 64 on purpose, so one run
    /// covers both the tiled and the narrow kernel.
    fn tile_shaped_cfg() -> QwenConfig {
        QwenConfig {
            vocab: 32,
            block_size: 64,
            n_layers: 2,
            d_model: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 32,
            d_ff: 256,
            max_position_embeddings: 64,
            ..QwenConfig::tiny()
        }
    }

    /// See [`f16_tier_gate`] for what is asserted. Run at TWO shapes, because
    /// the tier is not one kernel: a narrow/decode-shaped GEMM and a
    /// tile-shaped one bind to different physical `#w=f16` kernels, and the
    /// packed-word decode has to be right in every one of them.
    #[test]
    fn f16_storage_tier_tracks_fp32_and_really_dispatches_f16_kernels() {
        if gpu_disabled() {
            return;
        }
        let narrow: Vec<u32> = vec![1, 5, 3, 9, 2, 7, 11, 4, 0, 13, 6, 8];
        let Some((narrow_kernels, coop)) = half_tier_gate(QwenConfig::tiny(), 16, &narrow, Dtype::F16) else {
            return; // no f16 storage path on this device (already reported)
        };
        let cfg = tile_shaped_cfg();
        let wide: Vec<u32> = (0..cfg.block_size).map(|i| (i * 7 % cfg.vocab) as u32).collect();
        let (wide_kernels, _) = half_tier_gate(cfg, 64, &wide, Dtype::F16).expect("f16 availability cannot change mid-test");
        println!("f16 kernels dispatched: narrow {narrow_kernels:?}, wide {wide_kernels:?}");

        // Which variant each shape lands on is the SELECTOR's business, but
        // the split is a device-capability fact, and asserting it is what
        // keeps this from silently degenerating into "one kernel, tested
        // twice". `matmul_gemv`/`matmul_reg3` both stage their tile behind a
        // `workgroupBarrier()`, so a backend without `workgroup_reductions`
        // (the CPU JIT) legitimately runs everything on the reference kernel.
        let all: Vec<&String> = narrow_kernels.iter().chain(&wide_kernels).collect();
        assert!(
            all.iter().all(|n| ["matmul#w=f16", "matmul_gemv#w=f16", "matmul_reg3#w=f16"].contains(&n.as_str())),
            "unexpected f16 kernel name among {all:?}"
        );
        if coop {
            assert!(
                wide_kernels.iter().any(|n| n == "matmul_reg3#w=f16"),
                "a device with workgroup reductions must reach the register-tiled f16 kernel at these \
                 shapes, got {wide_kernels:?}"
            );
        } else {
            assert!(
                all.iter().all(|n| n.as_str() == "matmul#w=f16"),
                "without workgroup reductions every f16 dispatch must be the reference kernel: {all:?}"
            );
        }
    }

    /// The SAME gate at [`Dtype::BF16`] - not because bf16 is this phase's
    /// deliverable, but because `new_shard_dt` claims to be genuinely
    /// dtype-parameterised rather than an f16 special case, and a claim that
    /// nothing exercises rots. One shape is enough: what differs between the
    /// two tiers is only which packed decode expression
    /// `kernels::template::dtype_variant` substituted, not the model-side
    /// plumbing this test covers.
    #[test]
    fn bf16_storage_tier_is_the_same_one_implementation() {
        if gpu_disabled() {
            return;
        }
        let toks: Vec<u32> = vec![1, 5, 3, 9, 2, 7, 11, 4, 0, 13, 6, 8];
        let Some((kernels, _)) = half_tier_gate(QwenConfig::tiny(), 16, &toks, Dtype::BF16) else {
            return; // no bf16 storage path on this device (already reported)
        };
        assert!(
            kernels.iter().all(|n| n.ends_with("#w=bf16")),
            "every dispatched tier kernel must be a bf16 variant: {kernels:?}"
        );
    }

    #[test]
    fn param_list_shapes_and_tied_head() {
        let cfg = QwenConfig::tiny(); // v23 d16 L2 nh4 nkv2 hd8 ff32 tied
        let m: HashMap<_, _> = cfg.param_list().into_iter().collect();
        assert_eq!(m["tok.weight"], 23 * 16);
        assert_eq!(m["blocks.0.attn.wq.weight"], (4 * 8) * 16); // Hq x d
        assert_eq!(m["blocks.0.attn.wk.weight"], (2 * 8) * 16); // Hkv x d
        assert_eq!(m["blocks.0.attn.q_norm.weight"], 8); // head_dim
        assert_eq!(m["blocks.1.attn.wo.weight"], 16 * (4 * 8)); // d x Hq
        assert_eq!(m["blocks.0.mlp.gate.weight"], 32 * 16);
        assert_eq!(m["blocks.0.mlp.down.weight"], 16 * 32);
        // tied head: no separate lm_head tensor.
        assert!(!m.contains_key("lm_head.weight"));
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = QwenConfig::qwen3_0_6b();
        let back = QwenConfig::from_json(&cfg.to_json());
        assert_eq!(back.vocab, 151936);
        assert_eq!(back.n_kv_heads, 8);
        assert_eq!(back.head_dim, 128);
        assert_eq!(back.group(), 2);
        assert!((back.rope_theta - 1.0e6).abs() < 1.0);
        assert!(back.tie_embeddings);
    }

    #[test]
    fn forward_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let model = Qwen::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 3 % 23) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 3 + 1) % 23) as u32).collect();
        model.set_batch(&x, &y);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss {l1}");
        assert!((l1 - l2).abs() < 1e-6, "not deterministic");
        assert!(l1 < 2.0 * (23f32).ln(), "loss implausibly large: {l1}");
    }

    #[test]
    fn one_overfit_run_reduces_loss() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 11);
        let model = Qwen::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
        model.set_batch(&x, &y);
        let before = model.forward();
        for step in 1..=50 {
            model.zero_grads();
            model.forward();
            model.backward();
            model.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
            model.poll_wait();
        }
        let after = model.forward();
        assert!(after < before, "overfit did not reduce loss: {before} -> {after}");
    }
}

#[cfg(test)]
mod kv_tests {
    use super::*;
    use data::rng::Rng;
    use std::collections::HashMap;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The incremental KV-cache `step` must reproduce the `O(T²)` full-recompute
    /// (`logits_all`) for every prefix - the cache is algebraically exact, same
    /// engine, same weights, so any difference is only attention reduction order.
    /// Runs on GPU and (with `BRAIN_DEVICE=cpu`) the wgsl-cpu JIT; both must pass.
    #[test]
    fn kv_step_matches_full_recompute() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny(); // v23 d16 L2 GQA 4/2 hd8 ff32 tied
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let max_t = 8u32;
        let seq = 6usize;
        let mut rng = Rng::new(1234);

        // Random decoder weights; RMSNorm/QK-norm gains at 1 (as init_weights does),
        // other linears + the tied embedding table Normal(0, 0.08).
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in cfg.param_list() {
            let val = if name == "norm.weight"
                || name.ends_with("ln1.weight")
                || name.ends_with("ln2.weight")
                || name.ends_with("q_norm.weight")
                || name.ends_with("k_norm.weight")
            {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            map.insert(name, val);
        }
        let model = Qwen::new(cfg, 1, max_t, &map);

        // A random token sequence within the tiny vocab.
        let tokens: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % v as u64) as u32).collect();

        // Incremental: feed one token at a time through the KV cache, apply the tied
        // head on the host to get each new token's logits.
        model.reset_cache();
        let tok_w = model.read_weight("tok.weight"); // [v, d]
        let inc_logits: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&tid| {
                let hidden = model.step(tid); // [d], final-norm hidden
                (0..v)
                    .map(|row| {
                        let wr = &tok_w[row * d..(row + 1) * d];
                        wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>()
                    })
                    .collect()
            })
            .collect();

        // Reference: full recompute of each prefix; compare the last row's logits.
        let mut worst = 0.0f32;
        for i in 0..seq {
            let full = model.logits_all(&tokens[..i + 1]); // [(i+1)*v]
            let ref_last = &full[i * v..(i + 1) * v];
            let err = maxabs(&inc_logits[i], ref_last);
            worst = worst.max(err);
            assert!(err < 2e-3, "prefix {i}: KV step vs full recompute maxabs={err}");
        }
        println!("kv_step_matches_full_recompute: worst maxabs over {seq} prefixes = {worst:e}");
    }

    /// [`kv_step_matches_full_recompute`], but M-RoPE: `step_mrope`'s
    /// per-step 1-row table must reproduce `enable_mrope`'s whole-sequence
    /// batched forward for every prefix, exactly as the plain (analytic)
    /// case above does. Positions are a synthetic multimodal-splice-shaped
    /// sequence (T=H=W plain text, THEN a block with varying H/W simulating
    /// an image, THEN more plain text) -- NOT computed via
    /// `qwen3vl::mrope::get_rope_index_multi`/`mrope_tables` (this crate does
    /// not depend on `qwenvl` -- `qwenvl` depends on `qwen`, the other
    /// direction), but via the identical formula, inlined: `qwen3::Qwen`'s
    /// M-RoPE API (`enable_mrope`/`write_mrope_tables`/`step_mrope`) only
    /// ever consumes cos/sin tables, never derives positions itself, so this
    /// test only needs to prove that consumption is correct, not re-test
    /// `mrope_tables`' own formula (covered by `qwenvl`'s own tests).
    #[test]
    fn mrope_step_matches_full_recompute() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let half = (cfg.head_dim / 2) as usize;
        let theta = cfg.rope_theta;
        let max_t = 8u32;
        let seq = 6usize;
        let mut rng = Rng::new(5678);

        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in cfg.param_list() {
            let val = if name == "norm.weight" || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            map.insert(name, val);
        }
        let mut model = Qwen::new(cfg, 1, max_t, &map);
        model.enable_mrope();

        let tokens: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % v as u64) as u32).collect();

        // Synthetic 3-axis positions: rows 0-1 plain text (T=H=W), rows 2-3
        // an "image" (T pinned, H/W vary -- the shape a real vision splice
        // produces), rows 4-5 plain text again, continuing the T axis past
        // the image block -- mirrors get_rope_index_multi's own documented
        // shape without importing it.
        let positions: Vec<[u32; 3]> = vec![[0, 0, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1], [3, 3, 3], [4, 4, 4]];
        assert_eq!(positions.len(), seq);

        // mrope_section = [half/2, half/4, half/4]-ish split (T/H/W channel
        // counts summing to `half`) -- any real split works; this is the
        // same axis_map qwen3vl::mrope::axis_map builds, inlined.
        let section = [half / 2, half / 4, half - half / 2 - half / 4];
        let axis_of = |d: usize| -> usize {
            if d < section[0] {
                0
            } else if d < section[0] + section[1] {
                1
            } else {
                2
            }
        };
        let head_dim = model.cfg.head_dim;
        let table_row = |pos: [u32; 3]| -> (Vec<f32>, Vec<f32>) {
            let mut cos = vec![0f32; half];
            let mut sin = vec![0f32; half];
            for dd in 0..half {
                let inv_freq = theta.powf(-2.0 * dd as f32 / head_dim as f32);
                let angle = pos[axis_of(dd)] as f32 * inv_freq;
                cos[dd] = angle.cos();
                sin[dd] = angle.sin();
            }
            (cos, sin)
        };

        // Reference: whole-sequence table, one batched forward.
        let (mut whole_cos, mut whole_sin) = (Vec::with_capacity(seq * half), Vec::with_capacity(seq * half));
        for &p in &positions {
            let (c, s) = table_row(p);
            whole_cos.extend(c);
            whole_sin.extend(s);
        }
        model.write_mrope_tables(&whole_cos, &whole_sin);
        let full = model.logits_all(&tokens);

        // Incremental: one step_mrope call per token, per-step 1-row table.
        model.reset_cache();
        let tok_w = model.read_weight("tok.weight");
        let mut worst = 0.0f32;
        for (i, &tid) in tokens.iter().enumerate() {
            let (cos, sin) = table_row(positions[i]);
            let hidden = model.step_mrope(tid, &cos, &sin);
            let inc_logits: Vec<f32> = (0..v)
                .map(|row| {
                    let wr = &tok_w[row * d..(row + 1) * d];
                    wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            let ref_row = &full[i * v..(i + 1) * v];
            let err = maxabs(&inc_logits, ref_row);
            worst = worst.max(err);
            assert!(err < 2e-3, "position {i} ({:?}): M-RoPE step vs full recompute maxabs={err}", positions[i]);
        }
        println!("mrope_step_matches_full_recompute: worst maxabs over {seq} positions = {worst:e}");
    }

    /// [`mrope_step_matches_full_recompute`], but for DeepStack too:
    /// `decode_steps`'s `deepstack_row` parameter (this session's addition -
    /// before it existed, `qwen3vl::Qwen3Vl::generate()` called
    /// `write_deepstack` into buffers the incremental path never read, a
    /// real silent bug, fixed by adding `deepstack_row`)
    /// must reproduce `enable_deepstack`'s whole-sequence `SPLICE_ADD` in
    /// `forward_steps` exactly, position by position, the same way M-RoPE's
    /// decode table already does.
    #[test]
    fn deepstack_step_matches_full_recompute() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny(); // n_layers=2
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let half = (cfg.head_dim / 2) as usize;
        let theta = cfg.rope_theta;
        let max_t = 8u32;
        let seq = 6usize;
        let mut rng = Rng::new(9012);

        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in cfg.param_list() {
            let val = if name == "norm.weight" || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            map.insert(name, val);
        }
        let mut model = Qwen::new(cfg.clone(), 1, max_t, &map);
        model.enable_mrope();
        // Rows 2-3 are the "image" (row0=2, n_rows=2), same shape
        // `mrope_step_matches_full_recompute`'s positions already use.
        let (row0, n_rows) = (2u32, 2u32);
        model.enable_mm_splice(row0, n_rows);
        let n_levels = cfg.n_layers.min(2); // tiny() has 2 layers
        model.enable_deepstack(row0, n_rows, n_levels);

        let tokens: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % v as u64) as u32).collect();
        let positions: Vec<[u32; 3]> = vec![[0, 0, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1], [3, 3, 3], [4, 4, 4]];
        assert_eq!(positions.len(), seq);
        let section = [half / 2, half / 4, half - half / 2 - half / 4];
        let axis_of = |dd: usize| -> usize {
            if dd < section[0] { 0 } else if dd < section[0] + section[1] { 1 } else { 2 }
        };
        let head_dim = model.cfg.head_dim;
        let table_row = |pos: [u32; 3]| -> (Vec<f32>, Vec<f32>) {
            let mut cos = vec![0f32; half];
            let mut sin = vec![0f32; half];
            for dd in 0..half {
                let inv_freq = theta.powf(-2.0 * dd as f32 / head_dim as f32);
                let angle = pos[axis_of(dd)] as f32 * inv_freq;
                cos[dd] = angle.cos();
                sin[dd] = angle.sin();
            }
            (cos, sin)
        };

        // Random "visual" embeds for the 2 image rows, and one DeepStack
        // level buffer (also [n_rows, d]) per level.
        let visual: Vec<f32> = (0..(n_rows as usize * d)).map(|_| rng.next_gaussian() as f32 * 0.1).collect();
        let ds_levels: Vec<Vec<f32>> = (0..n_levels as usize)
            .map(|_| (0..(n_rows as usize * d)).map(|_| rng.next_gaussian() as f32 * 0.1).collect())
            .collect();

        // Reference: whole-sequence table + spliced embeds + DeepStack, one
        // batched forward. Token ids at the image rows are irrelevant (the
        // splice overwrites res[0] there).
        let (mut whole_cos, mut whole_sin) = (Vec::with_capacity(seq * half), Vec::with_capacity(seq * half));
        for &p in &positions {
            let (c, s) = table_row(p);
            whole_cos.extend(c);
            whole_sin.extend(s);
        }
        model.write_mrope_tables(&whole_cos, &whole_sin);
        model.write_img_embeds(&visual);
        for (level, data) in ds_levels.iter().enumerate() {
            model.write_deepstack(level, data);
        }
        let full = model.logits_all(&tokens);

        // Incremental: step_mrope for text rows, step_embed_mrope(...,
        // Some(local_row)) for the image rows.
        model.reset_cache();
        let tok_w = model.read_weight("tok.weight");
        let mut worst = 0.0f32;
        let mut local_row = 0usize;
        for (i, &tid) in tokens.iter().enumerate() {
            let (cos, sin) = table_row(positions[i]);
            let hidden = if i as u32 >= row0 && (i as u32) < row0 + n_rows {
                let row = &visual[local_row * d..(local_row + 1) * d];
                let h = model.step_embed_mrope(row, &cos, &sin, Some(local_row as u32));
                local_row += 1;
                h
            } else {
                model.step_mrope(tid, &cos, &sin)
            };
            let inc_logits: Vec<f32> = (0..v)
                .map(|row| {
                    let wr = &tok_w[row * d..(row + 1) * d];
                    wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            let ref_row = &full[i * v..(i + 1) * v];
            let err = maxabs(&inc_logits, ref_row);
            worst = worst.max(err);
            assert!(err < 2e-3, "position {i} ({:?}): DeepStack step vs full recompute maxabs={err}", positions[i]);
        }
        println!("deepstack_step_matches_full_recompute: worst maxabs over {seq} positions = {worst:e}");
    }

    /// Mutation-verify: a WRONG `deepstack_row` (off by one) must actually
    /// change the output vs the correct one - proving
    /// `deepstack_step_matches_full_recompute`'s tolerance isn't so loose it
    /// would pass even with the feature silently disabled/misindexed.
    #[test]
    fn deepstack_row_index_is_load_bearing() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let d = cfg.d_model as usize;
        let max_t = 8u32;
        let mut rng = Rng::new(3344);
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in cfg.param_list() {
            let val = if name == "norm.weight" || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            map.insert(name, val);
        }
        let mut model = Qwen::new(cfg.clone(), 1, max_t, &map);
        model.enable_mrope();
        model.enable_mm_splice(0, 2);
        let n_levels = cfg.n_layers.min(2);
        model.enable_deepstack(0, 2, n_levels);
        // DeepStack buffers start zero-initialised: row 0 and row 1 would be
        // indistinguishable (both add zero) unless real, DISTINCT data is
        // written first -- the actual point under test.
        for level in 0..n_levels as usize {
            let data: Vec<f32> = (0..(2 * d)).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
            model.write_deepstack(level, &data);
        }
        let half = (cfg.head_dim / 2) as usize;
        let cos = vec![1.0f32; half];
        let sin = vec![0.0f32; half];
        let row: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32 * 0.1).collect();

        model.reset_cache();
        let correct = model.step_embed_mrope(&row, &cos, &sin, Some(0));
        model.reset_cache();
        let wrong = model.step_embed_mrope(&row, &cos, &sin, Some(1));
        let diff = maxabs(&correct, &wrong);
        assert!(diff > 1e-6, "deepstack_row=0 vs deepstack_row=1 produced identical output (diff={diff:e}) -- the index is not load-bearing");
    }
}
