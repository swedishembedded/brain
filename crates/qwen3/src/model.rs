// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 dense decoder Transformer — forward + backprop as WGSL compute
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

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
pub use model::Shard;
use optim::Optim;
use paramstore::ParamStore;

use crate::config::QwenConfig;

/// Cross-entropy ignore index (masked target positions); the loader's `-1 i32`
/// reinterpreted as `u32`.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
/// Plain (untiled) embedding gather — kept in PIPELINES at index 0 for stable
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
// Kept registered (and A/B-able) though `linear_kernel` now picks its
// bit-identical, conflict-free twin `matmul_reg3`.
#[allow(dead_code)]
const MATMUL_REG2: usize = 32;
const MATMUL_DX_REG: usize = 33;
const MATMUL_DW_REG: usize = 34;
const CE_STATS: usize = 35;
const CE_GRAD_STATS: usize = 36;
// int8 (DP4A) inference path for the encoder linears — GPU only.
const QUANT_PACK: usize = 37;
const MATMUL_I8: usize = 38;
const MAX_ABS_ROW: usize = 39;
// Incremental KV-cache decode kernels (single new token vs the growing cache).
const ATTN_DECODE_SCORES: usize = 40;
const DECODE_SOFTMAX: usize = 41;
const ATTN_DECODE_APPLY: usize = 42;
const KV_APPEND: usize = 43;
const ROPE_AT: usize = 44;
// Vision-language residual splice (image-embedding injection). Off unless
// `enable_mm_splice` was called; see `model::vlm`.
const SPLICE: usize = 45;
const SPLICE_BWD: usize = 46;
// Table-driven RoPE for the interleaved M-RoPE path (Qwen3-VL). Off unless
// `enable_mrope` was called; replaces the analytic rope_base on q/k.
const ROPE2D: usize = 47;
// DeepStack residual add (Qwen3-VL): adds a level's merged vision features into
// the residual at the image rows after a layer. Off unless `enable_deepstack`.
const SPLICE_ADD: usize = 48;
// Qwen2 q/k/v projection bias (add fwd, row-sum grad bwd). Used only when
// `cfg.attn_bias` (FastVLM's Qwen2 decoder); Qwen3 is bias-free.
const BIAS_ADD: usize = 49;
const BIAS_GRAD: usize = 50;
// Decode-regime int8 GEMV (single-barrier; the m=1 shape KV decode dispatches).
const MATMUL_I8_GEMV: usize = 51;
// Decode-regime fp32 kernels (A1/A2): workgroup-per-row rmsnorm and the
// workgroup-per-column GEMV — the m=1 shapes KV decode is made of.
const RMSNORM_ROWS: usize = 52;
const MATMUL_GEMV: usize = 53;
// Encoder right-padding key mask (FLUX.2 text-encoder parity).
const GQA_SCORES_KMASK: usize = 54;
// Workgroup-per-row softmax over the [B*H*T, T] score slab — the coalesced twin
// of `attn_softmax` (see the kmask attention below).
const SOFTMAX_ROWS: usize = 55;
// `matmul_reg2` with its shared-memory bank conflicts removed; bit-identical
// output, so it is a pure speed swap (see `linear_kernel`).
const MATMUL_REG3: usize = 56;
// DeepStack decode's own add: `splice_add`'s `base` lands on `dst` only, but
// decode needs to read `deepstack_bufs[level]` at an offset (`local_row * d`)
// while writing this step's own zero-offset residual row -- see
// `decode_steps`'s own call site.
const SPLICE_ADD_OFFSET_SRC: usize = 59;

const PIPELINES: &[(&str, &str)] = &[
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
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("max_abs_row", kernels::MAX_ABS_ROW),
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
    ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("gqa_scores_kmask", kernels::GQA_SCORES_KMASK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
    ("splice_add_offset_src", kernels::SPLICE_ADD_OFFSET_SRC),
];

/// Pick the GEMM kernel + dispatch thread count for a forward linear
/// `[m,k]·[n,k]ᵀ`. The software-pipelined `matmul_reg3` (128×128 tile, 256
/// threads, ~4 TFLOP/s on a P40) wins from `m = 8` up — it bounds-guards its
/// tile, so a short M costs only the idle rows, while the naive
/// one-thread-per-output `matmul` collapses on a wide N (`docs/lessons.md` #15). Same math either way
/// (parity gated by `tests/backend_parity` + gradcheck), so this only changes
/// speed. `BRAIN_QWEN_NAIVE_MM=1` forces the naive kernel.
fn linear_kernel(m: usize, n: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    // `matmul_reg3` = `matmul_reg2` with the shared-memory bank conflicts
    // removed: identical tiling and identical K accumulation order, therefore
    // BIT-IDENTICAL output (measured max_abs 0.0), and 1.11x on the FLUX.2 text
    // encoder's prefill shapes (772 -> 695 ms for 196 GEMMs at 512 tokens).
    // Same dispatch geometry, and the CPU backend routes both to one AVX2 GEMM.
    block::pick_gemm(m, n, MATMUL, MATMUL_REG3, naive)
}

/// Backward GEMM pickers: tiled `matmul_{dx,dw}_reg` (bit-identical to naive,
/// ~34% of P40 peak) once both output dims fill a 128-tile, else naive. Small
/// LoRA-rank matmuls fall back automatically. `BRAIN_QWEN_NAIVE_MM=1` forces naive.
/// Full fine-tuning with AdamW moments offloaded to system RAM (`BRAIN_OFFLOAD_ADAM=1`).
fn offload_adam() -> bool {
    std::env::var("BRAIN_OFFLOAD_ADAM").map(|v| v != "0").unwrap_or(false)
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
fn shard_param_list(cfg: &QwenConfig, shard: &Shard) -> Vec<(String, usize)> {
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

    // Decode-path M-RoPE: a single-row `[1, head_dim/2]` cos/sin table, reused
    // (overwritten) every `step_mrope`/`step_embed_mrope` call rather than
    // sliced from `mrope_cos`/`mrope_sin` above -- those are sized for the
    // batched forward's whole KNOWN sequence, but decode generates tokens
    // beyond it one at a time, so each new token needs its OWN freshly
    // written table (mirrors `omni::thinker::layer_decode_step`'s pattern:
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
    // its arming flag — the padded-encoder path (`encode_hiddens_padded`).
    kmask: DeviceBuffer,
    kmask_on: Cell<bool>,
    /// The device runs workgroup-cooperative reductions (barriers). Selects the
    /// coalesced workgroup-per-row RMSNorm / softmax; false on the CPU JIT,
    /// which keeps the per-element reference kernels (whose native AVX2 fast
    /// paths are the fast CPU route anyway).
    coop: bool,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
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
    /// Int8 (DP4A) linears for inference. When present, the 7 per-layer linears
    /// run in int8 (weights ~4× smaller — fits the fp32-encoder-too-big case on
    /// one card) and are absent from the fp32 `ps`. None ⇒ all-fp32 path.
    q8: Option<crate::q8::Q8>,
    /// True for a [`Self::from_reader_decode`] build: activations are sized for
    /// a single token and `scores`/`probs` for `n_heads·ctx` (KV-cache decode
    /// only — the KV cache is the only ctx-scaled allocation). The batched
    /// forward/backward entry points assert against being called on such an
    /// instance instead of silently reading/writing past the smaller buffers.
    decode_only: bool,
}

impl Qwen {
    /// Load a trainable model (weights + grad + AdamW moments) from a checkpoint.
    /// Streams the weights one tensor at a time off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader) — peak host ≈ one
    /// tensor of f32, never the whole-model `checkpoint::load` + `by_role("")`
    /// host copy. AdamW moments are device zero-init (not read from disk), so
    /// this is byte-identical to the former eager path.
    pub fn load(path: &str, b: u32, t: u32) -> Qwen {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, &reader, true, shard, false, false)
    }

    /// Load an **inference-only** model: parameters are frozen (weights only, no
    /// grad/AdamW buffers), cutting device memory ~4× — essential for loading a
    /// real 0.6B checkpoint for generation. Builds only the forward graph.
    pub fn load_inference(path: &str, b: u32, t: u32) -> Qwen {
        Self::load_inference_with(path, b, t, false)
    }

    /// Streaming inference load: build from a mmap-backed [`WeightReader`],
    /// uploading one tensor at a time (peak host ≈ one tensor of f32) — never the
    /// `checkpoint::load` + `by_role("")` whole-model host copy. Numerically
    /// identical to [`Qwen::load_inference`]; used by the resident serve path.
    pub fn from_reader_inference(reader: &checkpoint::weightio::WeightReader, b: u32, t: u32) -> Qwen {
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, reader, false, shard, false, false)
    }

    /// Streaming **decode-only** load: like [`Self::from_reader_inference`]
    /// (mmap-backed, one tensor at a time), but shaped for incremental
    /// KV-cache decode only ([`Self::step`]/[`Self::prefill`]/[`Self::step_embed`])
    /// rather than the batched forward. Activations are sized for a single
    /// token (`n = 1`) instead of `b·t`, and `scores`/`probs` for `n_heads·ctx`
    /// instead of `n_heads·ctx²` — the KV cache (`[ctx, kv_dim]` per layer) is
    /// the only allocation that scales with `ctx`. No backward buffers and no
    /// `logits`/`d_logits` buffer (the LM head is applied host-side; see
    /// `sample::generate_kv_stream`). Calling a batched forward/backward entry
    /// point on the result panics loudly rather than reading/writing past the
    /// smaller buffers — use the KV-cache decode API instead.
    pub fn from_reader_decode(reader: &checkpoint::weightio::WeightReader, ctx: u32) -> Qwen {
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, 1, ctx, reader, false, shard, false, true)
    }

    /// [`Self::from_reader_decode`], but from an in-memory tensor map instead of
    /// a mmap'd checkpoint file -- for serving a named LoRA adapter, whose delta
    /// must be folded into the base tensors (`qwen3::lora::fold_adapter_into`)
    /// before a decode-only KV-cache model can be built from the result. Pays
    /// the whole-model host copy `from_reader_decode` avoids, but only for the
    /// (rare, one-off-per-activation) adapter-serving path.
    pub fn from_tensors_decode(cfg: QwenConfig, tensors: &HashMap<String, Vec<f32>>, ctx: u32) -> Qwen {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, 1, ctx, tensors, false, shard, false, true)
    }

    /// [`Self::load_inference`] with the int8 numeric tier: per-channel weight
    /// quantisation + dynamic activation quant, for both batched forwards and
    /// KV-cache decode (the m=1 packed GEMV).
    pub fn load_inference_i8(path: &str, b: u32, t: u32) -> Qwen {
        Self::load_inference_with(path, b, t, true)
    }

    /// Streaming inference load shared by [`Self::load_inference`] and
    /// [`Self::load_inference_i8`]: drive the builder straight off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader), uploading one tensor
    /// at a time — peak host ≈ one tensor of f32, never the whole-model
    /// `checkpoint::load` + `by_role("")` host copy on top of the device copy.
    /// The int8 tier reads + quantizes one linear at a time (the reader is passed
    /// to `Q8::build` as the [`TensorSource`](checkpoint::TensorSource)).
    fn load_inference_with(path: &str, b: u32, t: u32, i8: bool) -> Qwen {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let cfg = QwenConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, &reader, false, shard, i8, false)
    }

    pub fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen::new_impl(cfg, b, t, init, true, shard, false, false)
    }

    /// Build a single pipeline **stage**: only the layers (and endpoint weights)
    /// in `shard` are allocated on this device. `train` selects the parameter
    /// roles (offload/LoRA/frozen) exactly as the whole-model path does.
    /// `shard.gpu_index` names the canonical physical card (device registry);
    /// `Shard::ANY_GPU` keeps the ambient selection.
    /// Takes any `checkpoint::TensorSource` — the eager `&HashMap<String,
    /// Vec<f32>>` every existing caller passes (coerces, unchanged), or a
    /// streaming mmap'd `WeightReader`/`RemapSource` pair, which never
    /// materializes the whole checkpoint on the host.
    pub fn new_shard(cfg: QwenConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, train: bool, shard: Shard) -> Qwen {
        Qwen::new_impl(cfg, b, t, init, train, shard, false, false)
    }

    /// Inference-only shard with the 7 per-layer linears quantized to int8 (DP4A).
    /// Weights are ~4× smaller than fp32, so the whole Qwen3-4B encoder (~4.8 GB of
    /// weights → ~9.5 GB resident) fits a single 24 GB card — where the fp32
    /// encoder (~30 GB resident on non-ReBAR Pascal) does not. Frozen, no LoRA.
    /// See [`Self::new_shard`]'s doc: `init` may be any `TensorSource`.
    pub fn new_shard_i8(cfg: QwenConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, shard: Shard) -> Qwen {
        Qwen::new_impl(cfg, b, t, init, false, shard, true, false)
    }

    /// The shared builder behind every constructor. `decode_only` (set only by
    /// [`Self::from_reader_decode`]) shapes the model for single-token KV-cache
    /// decode instead of the batched forward: activations at `n = 1`,
    /// `scores`/`probs` at `n_heads·ctx` (not `n_heads·ctx²`), no backward
    /// scratch, no `logits`/`d_logits`/CE buffers.
    fn new_impl(cfg: QwenConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool, shard: Shard, i8: bool, decode_only: bool) -> Qwen {
        assert!(!(i8 && train), "int8 path is inference-only");
        assert!(!(decode_only && train), "decode-only build is inference-only");
        // An explicitly-placed shard binds its canonical card through the
        // device registry; `Shard::ANY_GPU` (the `Shard::whole` default) keeps
        // the ambient selection (`--device` / scoped `with_gpu`).
        let gpu = if shard.gpu_index == Shard::ANY_GPU {
            Gpu::new(PIPELINES)
        } else {
            Gpu::new_on_index(shard.gpu_index as u32, PIPELINES)
                .unwrap_or_else(|e| panic!("qwen shard placement: {e}"))
        };
        // The parameter set this stage actually holds: the whole list for a whole
        // shard (byte-identical to before), or just this stage's slice otherwise.
        // In int8 mode the 7 per-layer linears live in `q8` (packed int8), NOT the
        // fp32 store — filter them out so no fp32 copy is ever uploaded.
        let plist: Vec<(String, usize)> = shard_param_list(&cfg, &shard)
            .into_iter()
            .filter(|(name, _)| !(i8 && crate::q8::Q8::is_i8_linear(name)))
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
        // score/prob extent at n_heads·ctx instead of n_heads·ctx² — the KV
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
        // whole shard every index is live — identical to the single-device path.
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
        // pipeline stage carries them; on other stages they are size-1 dummies —
        // this is where sharding saves the most (`logits`/`d_logits` are
        // `n·vocab`, ~311 MB each at vocab 152k, block 512).
        let head = shard.head;
        let hd_v = |x: u64| if head { st(x) } else { st(1) };
        // Decode-only builds skip the CE-head buffers (the LM head is applied
        // host-side, see `sample::generate_kv_stream`) and all backward scratch
        // (backward never runs — `train` is forced false), regardless of `head`.
        let hd_or_dummy = |x: u64| if decode_only { st(1) } else { hd_v(x) };
        let bwd = |x: u64| if decode_only { st(1) } else { st(x) };

        // Int8 linears (inference): quantize+upload the owned layers' 7 matmul
        // weights from `init`; the fp32 store already excludes them (plist filter).
        let q8 = if i8 {
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
                    other => panic!("q8: unexpected linear leaf {other}"),
                }
            };
            Some(crate::q8::Q8::build(
                &gpu,
                src,
                shard.start..shard.end,
                dims,
                n as u32,
                ff as u32,
                MAX_ABS_ROW,
                QUANT_PACK,
                MATMUL_I8,
            ))
        } else {
            None
        };

        // Incremental-decode KV cache: one [t, kv_dim] key/value buffer per layer.
        // Only meaningful for a whole (single-device) model — `step` asserts that —
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
            q8,
            decode_only,
            gpu,
        };
        // Decode-only builds never call the batched forward_steps (its dispatch
        // sizes assume the b·t-sized buffers this build deliberately doesn't
        // have) — the KV-cache decode path builds its own tape per call
        // (`decode_submit`). `forward()`/`run_forward()` assert against being
        // called on a decode-only instance rather than relying on this being empty.
        m.fwd_steps = if decode_only { Vec::new() } else { m.forward_steps(m.b, m.t) };
        m.bwd_steps = if train { m.build_backward_steps() } else { Vec::new() };
        m
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        // `embed_tile.wgsl` (the batched forward's embedding gather) tile-gates
        // its own reads, so an out-of-vocab token here can't OOB-read the way
        // decode_submit's single-token EMBED could — but it silently leaves
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
    /// dispatches must be skipped — only the input-gradient (dX) path runs to
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

    /// Kernel-index map for [`block::gqa_decode_step`] — the hoisted twin of
    /// this struct's own original inline KV-cache decode dispatch (`decode_steps`
    /// below), migrated onto `model::block` so `omni::thinker` (the primitive's
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
    /// so it does not go away at prefill row counts — measured on the FLUX.2
    /// text encoder (512 tokens, 28 layers, 112 dispatches): **72.0 -> 6.3 ms**.
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
    /// [B*H*T = 16384, T = 512]: **33.6 -> 8.6 ms** over the encoder's 28
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
                // base: dx += d_out·W (frozen weight — no dW). d_out is NOT mutated
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

    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        assert!(
            !self.decode_only,
            "forward_steps: batched forward called on a decode-only-built Qwen \
             (activations sized for n=1, no logits buffer) — use step/prefill/step_embed instead"
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
            for &(v0, cnt) in &tiles {
                s.push(self.gpu.step_sliced(
                    EMBED_TILE,
                    &[&self.tokens, self.w("tok.weight"), &self.res[0]],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[d, n, v0, cnt],
                    n * d,
                ));
            }
            // Vision-language splice: overwrite the image-placeholder rows of the
            // freshly-gathered residual stream with the projected image tokens.
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                s.push(model::vlm::splice_fwd(&self.gpu, SPLICE, &self.img_embeds, &self.res[0], row0 * d, n_rows * d));
            }
        }

        for l in self.shard.start..self.shard.end {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // Int8 linears for this layer, if any (inference path: no LoRA/bias).
            let q8l = self.q8.as_ref().map(|q| (q, q.layers.get(&l).expect("q8 layer present")));
            // --- attention --- (projections stay here: they carry LoRA/bias;
            // norms/RoPE/attention-core come from the shared block builders)
            s.push(self.rms_step(&self.res[l], self.w(&p("ln1.weight")), &lb.xn1, d, n));
            if let Some((q8, ql)) = q8l {
                // xn1 quantized once, shared by q/k/v (DP4A GEMM per projection).
                q8.quant(&self.gpu, &mut s, &lb.xn1, d, n);
                q8.mm8(&self.gpu, &mut s, &ql.wq, &lb.q_pre, n);
                q8.mm8(&self.gpu, &mut s, &ql.wk, &lb.k_pre, n);
                q8.mm8(&self.gpu, &mut s, &ql.wv, &lb.v, n);
            } else {
                let (mk, mt) = linear_kernel(n as usize, hq as usize);
                s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wq.weight")), &lb.q_pre], &[n, d, hq], mt));
                self.lora_fwd(&mut s, "wq", &lb.xn1, &p("attn.wq.weight"), &lb.q_pre, n, d, hq);
                let (mk, mt) = linear_kernel(n as usize, hkv as usize);
                s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wk.weight")), &lb.k_pre], &[n, d, hkv], mt));
                self.lora_fwd(&mut s, "wk", &lb.xn1, &p("attn.wk.weight"), &lb.k_pre, n, d, hkv);
                let (mk, mt) = linear_kernel(n as usize, hkv as usize);
                s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wv.weight")), &lb.v], &[n, d, hkv], mt));
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
            if let Some((q8, ql)) = q8l {
                q8.quant(&self.gpu, &mut s, &lb.ctx, hq, n);
                q8.mm8(&self.gpu, &mut s, &ql.wo, &self.proj, n);
            } else {
                let (mk, mt) = linear_kernel(n as usize, d as usize);
                s.push(self.gpu.step(mk, &[&lb.ctx, self.w(&p("attn.wo.weight")), &self.proj], &[n, hq, d], mt));
                self.lora_fwd(&mut s, "wo", &lb.ctx, &p("attn.wo.weight"), &self.proj, n, hq, d);
            }
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // --- SwiGLU MLP ---
            s.push(self.rms_step(&lb.xmid, self.w(&p("ln2.weight")), &lb.xn2, d, n));
            if let Some((q8, ql)) = q8l {
                // xn2 quantized once, shared by gate/up.
                q8.quant(&self.gpu, &mut s, &lb.xn2, d, n);
                q8.mm8(&self.gpu, &mut s, &ql.gate, &lb.gate_pre, n);
                q8.mm8(&self.gpu, &mut s, &ql.up, &lb.up, n);
            } else {
                let (mk, mt) = linear_kernel(n as usize, ff as usize);
                s.push(self.gpu.step(mk, &[&lb.xn2, self.w(&p("mlp.gate.weight")), &lb.gate_pre], &[n, d, ff], mt));
                self.lora_fwd(&mut s, "gate", &lb.xn2, &p("mlp.gate.weight"), &lb.gate_pre, n, d, ff);
                let (mk, mt) = linear_kernel(n as usize, ff as usize);
                s.push(self.gpu.step(mk, &[&lb.xn2, self.w(&p("mlp.up.weight")), &lb.up], &[n, d, ff], mt));
                self.lora_fwd(&mut s, "up", &lb.xn2, &p("mlp.up.weight"), &lb.up, n, d, ff);
            }
            s.push(block::swiglu_fwd(&self.gpu, &ids, &lb.gate_pre, &lb.up, &lb.h, n * ff));
            if let Some((q8, ql)) = q8l {
                q8.quant(&self.gpu, &mut s, &lb.h, ff, n);
                q8.mm8(&self.gpu, &mut s, &ql.down, &self.mlp_out, n);
            } else {
                let (mk, mt) = linear_kernel(n as usize, d as usize);
                s.push(self.gpu.step(mk, &[&lb.h, self.w(&p("mlp.down.weight")), &self.mlp_out], &[n, ff, d], mt));
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
        // lm_head. When the whole vocab fits one tile (v0=0, cnt=v — the common
        // case for a small vocab like the TTS Talker's 3072), it is a plain
        // `[n,d]·[v,d]ᵀ` matmul, so dispatch the size-adaptive fast kernel
        // (`matmul_reg3`) instead of the naive column-tiled `matmul_tile` — the
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
        losses.iter().sum::<f32>() / self.count.get()
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
            // then the per-element gradient reads them — O(rows*vocab) instead of
            // the naive per-element softmax recompute's O(rows*vocab^2). At vocab
            // 151936 this is the difference between ~10 ms and ~56 s per backward.
            s.push(self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &self.ce_stats], &[n, v, IGNORE], n));
            s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.ce_stats, &self.d_logits], n * v));
            if self.trainable(head) {
                let (bk, bt) = dw_kernel_bw(v, d);
                s.push(self.gpu.step(bk, &[&self.d_logits, &self.xn_final, self.g(head)], &[n, d, v], bt));
            }
            let (bk, bt) = dx_kernel_bw(n, d);
            s.push(self.gpu.step(bk, &[&self.d_logits, self.w(head), &self.d_xn], &[n, d, v, 0], bt));
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
        // Host (RAM-resident) optimiser for `Offload` params — built lazily on the
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
    /// image buffers and rebuilds the fwd/bwd graphs — call once after construction
    /// (before the first forward). No effect on `tok.weight`/other params.
    pub fn enable_mm_splice(&mut self, row0: u32, n_rows: u32) {
        let sz = (n_rows * self.cfg.d_model) as u64;
        self.img_embeds = self.gpu.storage(sz);
        self.d_img_embeds = self.gpu.storage(sz);
        self.mm_splice.set(Some((row0, n_rows)));
        self.fwd_steps = self.forward_steps(self.b, self.t);
        if !self.bwd_steps.is_empty() {
            self.bwd_steps = self.build_backward_steps();
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

    /// Read the gradient of the spliced image embeddings after `backward` — feeds
    /// the vision connector/encoder backward.
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    // ---- interleaved M-RoPE seam (Qwen3-VL) ----

    /// Switch q/k to the table-driven `rope2d` M-RoPE path (from the analytic
    /// rope_base). Allocates the `[b·t, head_dim/2]` cos/sin tables and rebuilds
    /// the fwd/bwd graphs — call once after construction, then supply the tables
    /// each batch via [`Self::write_mrope_tables`] (computed by
    /// `qwenvl::mrope::{get_rope_index, mrope_tables}`).
    pub fn enable_mrope(&mut self) {
        let sz = (self.b * self.t * self.cfg.head_dim / 2) as u64;
        self.mrope_cos = self.gpu.storage(sz);
        self.mrope_sin = self.gpu.storage(sz);
        self.mrope.set(true);
        self.fwd_steps = self.forward_steps(self.b, self.t);
        if !self.bwd_steps.is_empty() {
            self.bwd_steps = self.build_backward_steps();
        }
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
        self.fwd_steps = self.forward_steps(self.b, self.t);
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
    /// built/loaded with) — generation must keep its context within this.
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
    /// `l-1` (pre-final-norm). This is what diffusion text encoders consume —
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

    /// The **penultimate** hidden state (`res[n_layers-1]`, un-normed) — the
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
    /// text encoder feeds all 512 rows — pads included — to the DiT unmasked,
    /// so parity requires the masked values).
    pub fn encode_hiddens_padded(
        &self,
        tokens: &[u32],
        content_len: usize,
        layers: &[usize],
    ) -> Vec<Vec<f32>> {
        assert!(content_len <= tokens.len());
        let mut mask = vec![0.0f32; self.t as usize];
        for m in mask[content_len..tokens.len()].iter_mut() {
            *m = -3.4e38;
        }
        self.gpu.write(&self.kmask, bytemuck::cast_slice(&mask));
        self.kmask_on.set(true);
        let out = self.encode_hiddens(tokens, layers);
        self.kmask_on.set(false);
        out
    }

    /// Several hidden-state taps from **one** forward, each row-major
    /// `[len·d_model]` in the order requested. FLUX.2 Klein concatenates
    /// `hidden_states[9|18|27]` per token — with [`Self::encode_hidden`] that
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
    /// head to the returned hidden to get logits — `logits[v] = tok.weight[v]·hidden`.
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
        let pos = self.dec_pos.get();
        self.write_decode_mrope_table(cos, sin);
        let hidden = self.decode_at(Some(token_id), pos, Some((&self.decode_mrope_cos, &self.decode_mrope_sin)), None);
        self.dec_pos.set(pos + 1);
        hidden
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
    /// hiddens are thrown away, so the per-step submit+fence+map round trip —
    /// measured at the top of the caption profile — is pure waste.
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

    /// [`Self::step`] from a RAW embedding instead of a token id — the seam a
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

    /// [`Self::step_embed`] with M-RoPE -- see [`Self::step_mrope`]'s doc for
    /// the `cos`/`sin` convention. `deepstack_row`: see [`Self::decode_steps`]'s
    /// doc -- `Some(local_row)` when this embedding is image row `local_row`
    /// on a DeepStack-enabled checkpoint, `None` otherwise (every caller
    /// before this parameter existed, unchanged).
    pub fn step_embed_mrope(&self, embed: &[f32], cos: &[f32], sin: &[f32], deepstack_row: Option<u32>) -> Vec<f32> {
        assert_eq!(embed.len(), self.cfg.d_model as usize, "step_embed_mrope wants one d_model row");
        let pos = self.dec_pos.get();
        self.gpu.write(&self.res[0], bytemuck::cast_slice(embed));
        self.write_decode_mrope_table(cos, sin);
        let hidden = self.decode_at(None, pos, Some((&self.decode_mrope_cos, &self.decode_mrope_sin)), deepstack_row);
        self.dec_pos.set(pos + 1);
        hidden
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
    /// decode tape per kernel kind — `gpu_core::profile` needs a step list, and
    /// the decode tape is rebuilt per token rather than recorded once like
    /// `fwd_steps`, so there was nothing to hand it. Behaviour is unchanged:
    /// `decode_submit` records exactly this and submits it.
    ///
    /// `mrope`: see [`Self::decode_at`]'s doc. `None` reproduces this
    /// function's behaviour before M-RoPE decode support existed, bit-for-bit
    /// (the analytic `rope_at` dispatch is untouched in that branch).
    /// `deepstack_row`: `Some(local_row)` when this step decodes image row
    /// `local_row` (0-based within the spliced image) on a checkpoint with
    /// `enable_deepstack` on — applies that row's per-level residual add.
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
        // read arbitrarily far out of bounds of the embedding buffer — on the
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
        // KV decode is m=1 by construction — the decode regime. Use the
        // workgroup-cooperative kernels (A1/A2: rmsnorm_rows, matmul_gemv)
        // wherever the device executes workgroup reductions; the per-element
        // reference kernels run ONE thread per row here (measured: rmsnorm was
        // 19% of prefill GPU time across 13k single-thread calls). Same policy
        // the serving engine's selector applies, at the always-m=1 call site.
        let fast = g.caps().workgroup_reductions;
        let rms = |s: &mut Vec<Step>, x: &DeviceBuffer, wt: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32| {
            if fast {
                s.push(g.step(RMSNORM_ROWS, &[x, wt, out], &[dim, rows, gpu_core::f(1e-6)], rows * 64));
            } else {
                s.push(block::rmsnorm_fwd(g, &ids, x, wt, out, dim, rows));
            }
        };
        let mm = |s: &mut Vec<Step>, x: &DeviceBuffer, wt: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32| {
            if fast {
                s.push(g.step(MATMUL_GEMV, &[x, wt, out], &[1, k, n], n * 64));
            } else {
                s.push(g.step(MATMUL, &[x, wt, out], &[1, k, n], n));
            }
        };

        // Embed the token id into res[0] row 0 via the tied table (non-tiled
        // gather); `None` = the caller already wrote a raw embedding there
        // (`step_embed`).
        let mut s: Vec<Step> = Vec::new();
        if let Some(token_id) = token_id {
            g.write(&self.tokens, &[token_id]);
            s.push(g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, 1], d));
        }
        // Int8 (m=1): quantize the input row once per distinct input, then the
        // single-barrier packed GEMV — the exact decode-regime shape the
        // serving engine measured (runs on the CPU JIT too, unlike the tiled
        // GEMM, so int8 KV decode is not GPU-only).
        let mm8 = |s: &mut Vec<Step>, q8: &crate::q8::Q8, lin: &crate::q8::Lin8, x: &DeviceBuffer, out: &DeviceBuffer, k: u32| {
            q8.quant(g, s, x, k, 1);
            s.push(g.step(
                MATMUL_I8_GEMV,
                &[&q8.xq, &lin.packed, &q8.sx, &lin.scale, out],
                &[1, k / 4, lin.n],
                lin.n * 64,
            ));
        };

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // --- attention: project, QK-norm, RoPE-at-pos, append, decode-attend ---
            rms(&mut s, &self.res[l], w(&p("ln1.weight")), &lb.xn1, d, 1);
            let q8l = self.q8.as_ref().map(|q| (q, q.layers.get(&l).expect("q8 layer present")));
            if let Some((q8, lay)) = q8l {
                mm8(&mut s, q8, &lay.wq, &lb.xn1, &lb.q_pre, d);
                mm8(&mut s, q8, &lay.wk, &lb.xn1, &lb.k_pre, d);
                mm8(&mut s, q8, &lay.wv, &lb.xn1, &lb.v, d);
            } else {
                mm(&mut s, &lb.xn1, w(&p("attn.wq.weight")), &lb.q_pre, d, hq);
                mm(&mut s, &lb.xn1, w(&p("attn.wk.weight")), &lb.k_pre, d, hkv);
                mm(&mut s, &lb.xn1, w(&p("attn.wv.weight")), &lb.v, d, hkv);
            }
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
            // mirroring omni::thinker::layer_decode_step's pattern -- a
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
            // shared with omni::thinker instead of duplicated.
            s.extend(block::gqa_decode_step(g, &decode_ids, nh, nkv, hd, pos, cap, q_buf, k_buf, &lb.v, &self.kcache[l], &self.vcache[l], &self.scores, &lb.probs, &lb.ctx));
            if let Some((q8, lay)) = q8l {
                mm8(&mut s, q8, &lay.wo, &lb.ctx, &self.proj, hq);
            } else {
                mm(&mut s, &lb.ctx, w(&p("attn.wo.weight")), &self.proj, hq, d);
            }
            s.push(g.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[d], d));
            // --- SwiGLU MLP ---
            rms(&mut s, &lb.xmid, w(&p("ln2.weight")), &lb.xn2, d, 1);
            if let Some((q8, lay)) = q8l {
                mm8(&mut s, q8, &lay.gate, &lb.xn2, &lb.gate_pre, d);
                mm8(&mut s, q8, &lay.up, &lb.xn2, &lb.up, d);
                s.push(block::swiglu_fwd(g, &ids, &lb.gate_pre, &lb.up, &lb.h, ff));
                mm8(&mut s, q8, &lay.down, &lb.h, &self.mlp_out, ff);
            } else {
                mm(&mut s, &lb.xn2, w(&p("mlp.gate.weight")), &lb.gate_pre, d, ff);
                mm(&mut s, &lb.xn2, w(&p("mlp.up.weight")), &lb.up, d, ff);
                s.push(block::swiglu_fwd(g, &ids, &lb.gate_pre, &lb.up, &lb.h, ff));
                mm(&mut s, &lb.h, w(&p("mlp.down.weight")), &self.mlp_out, ff, d);
            }
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
            // limit (`docs/lessons.md`). A uniform-parameter source offset has
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

    /// OFFLINE FLOP/OPS cost of the recorded batch forward — walks the step
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
    /// Exposed for the PROFILER (`qwen_bench`), not for driving — `forward()`
    /// owns the submit and the readback. `gpu_core::profile` needs the step
    /// list to build the per-kernel-kind table `docs/kernel-checklist.md` §F.1
    /// asks for before anyone optimises, and until this existed there was no
    /// way to get one for a decoder LM at all: every recorded qwen number in
    /// `docs/performance/` came from `BRAIN_PROFILE`'s timestamp table on a
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
            model::Batch::Lm { tokens, targets } => Qwen::set_batch(self, tokens, targets),
            _ => panic!("qwen3::Qwen only supports Batch::Lm"),
        }
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
    /// `by_role("")`) — proving equivalence, not merely that it compiles. Also
    /// pins both to the source init exactly. GPU-gated (testgpu / MOE_SKIP_GPU).
    #[test]
    fn streaming_load_matches_eager() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let cfg_json = cfg.to_json();
        let init = crate::init::init_weights(&cfg, 5);
        // Persist as safetensors — flat 1-D tensors (only the values matter here).
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

    /// Writes `cfg`'s `init` to a temp `.st` file and opens a [`checkpoint::weightio::WeightReader`]
    /// on it — the fixture [`streaming_load_matches_eager`] already established
    /// for exercising the streaming (`from_reader_*`) constructors.
    fn write_reader_fixture(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>, tag: &str) -> std::path::PathBuf {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            init.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64], v.clone())).collect();
        let path = std::env::temp_dir().join(format!("qwen-decode-only-{tag}-{}.st", std::process::id()));
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
        path
    }

    /// Read `n` elements from `buf`, or `None` if that panics (out of bounds) —
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
    /// `scores`/`probs` at `n_heads·ctx` (NOT `n_heads·ctx²`) — the KV cache is
    /// the only ctx-scaled allocation. For each buffer, reading exactly the
    /// decode-shaped extent succeeds while reading the old training-shaped
    /// (`b·t` / `ctx²`) extent is out of bounds — proving the buffer genuinely
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

    /// REGRESSION (CPU-backend JIT dispatch segfault): a
    /// decode token id the checkpoint's embedding table doesn't cover (a
    /// checkpoint/tokenizer vocab mismatch — e.g. a real BPE tokenizer's
    /// `<|im_start|>`-class special token fed to a tiny synthetic checkpoint)
    /// used to reach `EMBED`'s unchecked `emb[tokens[t]*d_model+c]` gather and
    /// read arbitrarily far out of bounds — 100% reproducible SIGSEGV on the
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

        // The real Qwen3 tokenizer's `<|im_start|>` id — exactly what a
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
    /// position's embedding row un-written (stale/garbage) — a correctness
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

    /// Every kernel this model can dispatch has a cost formula — pins the
    /// FLOP/OPS accounting against silent drift when PIPELINES grows.
    #[test]
    fn pipelines_fully_costed() {
        for (name, _) in PIPELINES {
            assert!(
                gpu_core::cost::covers(name),
                "kernel '{name}' has no formula in gpu_core::cost::kernel_cost; \
                 add one (its dispatches would otherwise be reported UNCOVERED)"
            );
        }
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
    /// (`logits_all`) for every prefix — the cache is algebraically exact, same
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
    /// `qwenvl::mrope::get_rope_index_multi`/`mrope_tables` (this crate does
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
        // same axis_map qwenvl::mrope::axis_map builds, inlined.
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
    /// `decode_steps`'s `deepstack_row` parameter (this session's addition —
    /// before it existed, `qwenvl::Qwen3Vl::generate()` called
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
    /// change the output vs the correct one — proving
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
