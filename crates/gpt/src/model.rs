// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense GPT decoder Transformer (nanogpt parity), forward + backprop as WGSL
//! compute dispatches. Shares the engine with the MoE/PID models (`gpu_core`,
//! `paramstore`, `optim`, `kernels`).
//!
//! Architecture (pre-norm, matches nanogpt `GPT`; dropout disabled):
//!   x = tok_emb[idx] + pos_emb[pos]
//!   per block: h = LN1(x); x += Wo·MHA(h) ;  h = LN2(x); x += proj·GELU(fc·h)
//!   logits = lm_head( LN_f(x) )            // over vocab; lm_head has no bias
//!   loss   = cross-entropy (ignore_index = IGNORE), so masked datasets work.
//!
//! Differences vs nanogpt, intentional and documented:
//!   * `lm_head` is **untied** from `tok.weight` (nanogpt ties them). Untied
//!     keeps each gradient written exactly once, matching the rest of the
//!     engine and the finite-difference gradient check. Tying (which needs grad
//!     accumulation into `tok.weight`) is a follow-up.
//!   * GELU uses the tanh approximation (see `kernels/wgsl/gelu.wgsl`).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{f, Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;
pub use model::Shard;

/// The parameter subset a shard holds. A whole shard returns `cfg.param_list()`
/// verbatim; a partial shard keeps only its layers' weights, plus tok/pos
/// embeddings when it embeds and the final norm + (untied) lm_head when it heads.
fn shard_param_list(cfg: &GptConfig, shard: &Shard) -> Vec<(String, usize)> {
    let full = cfg.param_list();
    if shard.is_whole(cfg.n_layers as usize) {
        return full;
    }
    full.into_iter()
        .filter(|(name, _)| {
            if let Some(rest) = name.strip_prefix("blocks.") {
                let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
                return shard.owns(l);
            }
            match name.as_str() {
                "tok.weight" | "pos.weight" => shard.embed,
                "ln.weight" | "ln.bias" | "lm_head.weight" => shard.head,
                _ => false,
            }
        })
        .collect()
}

/// Cross-entropy ignore index (masked target positions). The data loader emits
/// `-1` as `i32`; reinterpreted as `u32` that is exactly this value.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const POS_ADD: usize = 1;
const LAYERNORM: usize = 2;
const LN_STATS: usize = 3;
const LN_DX: usize = 4;
const LN_DGAMMA: usize = 5;
const LN_DBETA: usize = 6;
const MATMUL: usize = 7;
const BIAS_ADD: usize = 8;
const BIAS_GRAD: usize = 9;
const ATTN_SCORES: usize = 10;
const ATTN_SOFTMAX: usize = 11;
const ATTN_APPLY: usize = 12;
const GELU: usize = 13;
const GELU_BWD: usize = 14;
const CE_VALUE: usize = 15;
#[allow(dead_code)]
const CE_GRAD: usize = 16;
const MATMUL_DX: usize = 17;
const MATMUL_DW: usize = 18;
const ATTN_DSCORES: usize = 19;
const ATTN_DV: usize = 20;
const ATTN_DQ: usize = 21;
const ATTN_DK: usize = 22;
const POS_BWD: usize = 23;
const EMB_BWD: usize = 24;
const ADD2: usize = 25;
const GRADNORM_SQ: usize = 26;
const GRAD_SCALE: usize = 27;
const ADAMW: usize = 28;
const CLIP_COEF: usize = 29;
const GRAD_SCALE_BUF: usize = 30;
const MATMUL_REG: usize = 31;
const MATMUL_REG3: usize = 32;
const MATMUL_DX_REG: usize = 33;
const MATMUL_DW_REG: usize = 34;
const CE_STATS: usize = 35;
const CE_GRAD_STATS: usize = 36;
// Incremental KV-cache decode kernels (single new token vs the growing cache).
const ATTN_DECODE_SCORES: usize = 37;
const DECODE_SOFTMAX: usize = 38;
const ATTN_DECODE_APPLY: usize = 39;
const KV_APPEND: usize = 40;
// Workgroup-per-row LayerNorm (2.3-9.1x the per-element kernels on a P40 — see
// `model::block::LayerNormIds`). Appended, so every index above is unchanged.
const LAYERNORM_ROWS: usize = 41;
const LN_STATS_ROWS: usize = 42;
const LN_DX_ROWS: usize = 43;

/// The LayerNorm family this model dispatches through `model::block`, which
/// picks the coalesced variant per device (`backend_api::select`).
const LN_IDS: model::block::LayerNormIds = model::block::LayerNormIds {
    layernorm: LAYERNORM,
    layernorm_rows: Some(LAYERNORM_ROWS),
    ln_stats: LN_STATS,
    ln_stats_rows: Some(LN_STATS_ROWS),
    layernorm_dx: LN_DX,
    layernorm_dx_rows: Some(LN_DX_ROWS),
};

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("pos_add", kernels::POS_ADD),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("attn_scores", kernels::ATTN_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    ("ce_value_masked", kernels::CE_VALUE_MASKED),
    ("ce_grad_masked", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("pos_bwd", kernels::POS_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("add2", kernels::ADD2),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("matmul_reg", kernels::MATMUL_REG),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("kv_append", kernels::KV_APPEND),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// Pick the forward-linear GEMM kernel + its dispatch thread count for an
/// `[M,K]·[N,K]ᵀ` product. The software-pipelined `matmul_reg3` (128×128 output tile,
/// 256 threads) wins by ~10× once both output dims fill at least one tile; below
/// that the naive one-thread-per-output `matmul` is better (a whole tile for a
/// handful of outputs is mostly masked lanes). Same math either way — parity is
/// gated in `brain-gpu-core`'s `bench_matmul` and by `gradcheck` — so this only
/// ever changes speed, never results. `BRAIN_GPT_NAIVE_MM=1` forces the naive
/// kernel (A/B comparison + a fallback if a driver ever mishandles the tile).
fn linear_kernel(m: usize, n: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_GPT_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    // `matmul_reg3` (software-pipelined) is the default; `BRAIN_GPT_REG1=1`
    // selects the non-pipelined `matmul_reg` for A/B comparison.
    let reg = if std::env::var("BRAIN_GPT_REG1").map(|v| v != "0").unwrap_or(false) {
        MATMUL_REG
    } else {
        MATMUL_REG3
    };
    // The threshold is `block::pick_gemm`'s MEASURED one (`m < 8`), not the
    // `m < 128` this used to carry. That guard is the one `docs/lessons.md` §15
    // records as costing 22x on an SDXL UNet, and it is worth more here than
    // there: A/B'd on a P40 at `k=768, n=3072`, naive vs tiled is
    //
    //     m       8      16      32      64      96     127
    //     x     1.5x    4.0x    8.2x   19.7x   19.8x   34.1x
    //
    // bit-identical at every point (max|delta| 0.0). Every GPT shape with
    // 8 <= m < 128 — short prompts, small eval batches, the m = T generate
    // path — was paying that. `pick_gemm` owns the rule so the next model
    // inherits it instead of copying the constant again.
    model::block::pick_gemm(m, n, MATMUL, reg, naive)
}


/// Backward GEMM pickers — the tiled `matmul_{dx,dw}_reg` (matmul_reg3 structure,
/// ~34% of P40 peak, bit-identical to the naive kernels) once both output dims
/// fill a 128-tile, else the naive per-output kernel. `BRAIN_GPT_NAIVE_MM=1`
/// forces naive (shares the forward's flag). Same math — gradcheck-gated.
fn dx_kernel(m: usize, k: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_GPT_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    if naive || m < 128 || k < 128 { (MATMUL_DX, (m * k) as u32) }
    else { (MATMUL_DX_REG, (m.div_ceil(128) * k.div_ceil(128) * 256) as u32) }
}
fn dw_kernel(nrows: usize, k: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_GPT_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    if naive || nrows < 128 || k < 128 { (MATMUL_DW, (nrows * k) as u32) }
    else { (MATMUL_DW_REG, (nrows.div_ceil(128) * k.div_ceil(128) * 256) as u32) }
}

#[derive(Clone, Debug)]
pub struct GptConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_ff: u32,
}

impl GptConfig {
    /// A tiny config for tests / gradient checks.
    pub fn tiny() -> GptConfig {
        GptConfig {
            vocab: 65,
            block_size: 64,
            n_layers: 2,
            d_model: 32,
            n_heads: 4,
            d_ff: 128,
        }
    }

    /// nanogpt's `4 * d_model` feed-forward width.
    pub fn with_ff_default(mut self) -> Self {
        if self.d_ff == 0 {
            self.d_ff = 4 * self.d_model;
        }
        self
    }

    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "gpt",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff
        })
    }

    pub fn from_json(c: &Value) -> GptConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        GptConfig {
            vocab: g("vocab_size", 65),
            block_size: g("block_size", 64),
            n_layers: g("n_layers", 2),
            d_model: g("d_model", 32),
            n_heads: g("n_heads", 4),
            d_ff: g("d_ff", 128),
        }
    }

    /// Parameter list: `(name, numel)`. Order is irrelevant to correctness.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let mut out = vec![
            ("tok.weight".to_string(), v * d),
            ("pos.weight".to_string(), self.block_size as usize * d),
        ];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("ln1.bias"), d));
            out.push((p("attn.qkv.weight"), 3 * d * d));
            out.push((p("attn.qkv.bias"), 3 * d));
            out.push((p("attn.out.weight"), d * d));
            out.push((p("attn.out.bias"), d));
            out.push((p("ln2.weight"), d));
            out.push((p("ln2.bias"), d));
            out.push((p("mlp.fc.weight"), ff * d));
            out.push((p("mlp.fc.bias"), ff));
            out.push((p("mlp.proj.weight"), d * ff));
            out.push((p("mlp.proj.bias"), d));
        }
        out.push(("ln.weight".to_string(), d));
        out.push(("ln.bias".to_string(), d));
        out.push(("lm_head.weight".to_string(), v * d)); // untied, no bias
        out
    }
}

struct Layer {
    ln1_out: gpu_core::DeviceBuffer,
    qkv: gpu_core::DeviceBuffer,
    scores: gpu_core::DeviceBuffer,
    probs: gpu_core::DeviceBuffer,
    attn_ctx: gpu_core::DeviceBuffer,
    xmid: gpu_core::DeviceBuffer,
    ln2_out: gpu_core::DeviceBuffer,
    fc: gpu_core::DeviceBuffer,   // c_fc pre-activation
    gelu: gpu_core::DeviceBuffer, // GELU(fc)
}

pub struct Gpt {
    pub gpu: Gpu,
    pub cfg: GptConfig,
    pub ps: ParamStore,
    /// Pipeline shard this instance owns (whole model on GPU 0 by default).
    pub shard: Shard,
    opt: Optim,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: gpu_core::DeviceBuffer,
    targets: gpu_core::DeviceBuffer,
    res: Vec<gpu_core::DeviceBuffer>,
    layers: Vec<Layer>,
    proj: gpu_core::DeviceBuffer,
    ffn_out: gpu_core::DeviceBuffer,
    xn_final: gpu_core::DeviceBuffer,
    logits: gpu_core::DeviceBuffer,
    ce_buf: gpu_core::DeviceBuffer,

    // backward temporaries
    dres: Vec<gpu_core::DeviceBuffer>,
    d_logits: gpu_core::DeviceBuffer,
    ce_stats: gpu_core::DeviceBuffer,
    d_xn: gpu_core::DeviceBuffer,
    d_branch: gpu_core::DeviceBuffer,
    d_tmp: gpu_core::DeviceBuffer,
    dxmid: gpu_core::DeviceBuffer,
    d_attn_ctx: gpu_core::DeviceBuffer,
    d_scores: gpu_core::DeviceBuffer,
    d_qkv: gpu_core::DeviceBuffer,
    d_gelu: gpu_core::DeviceBuffer,
    d_fc: gpu_core::DeviceBuffer,
    ln_mean: gpu_core::DeviceBuffer,
    ln_inv: gpu_core::DeviceBuffer,

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: gpu_core::DeviceBuffer,

    // Incremental KV-cache decode state (lazily built on first `step`).
    dec: RefCell<Option<Decode>>,
    // Absolute position the next `step` will decode (cache fill level).
    dec_pos: Cell<u32>,
}

/// Per-layer / shared GPU scratch for the incremental single-token decode path,
/// plus the persistent K/V cache. Built lazily the first time [`Gpt::step`] runs
/// (inference-only; sized for `n=1` rows and a `block_size` cache), so the
/// training buffers above are never disturbed. The fused `attn.qkv.bias` is split
/// once into contiguous per-region `[d]` buffers so the decode path can add each
/// projection's bias without an unaligned buffer slice.
struct Decode {
    cap: u32, // K/V cache capacity == block_size (max context)
    tok_id: gpu_core::DeviceBuffer,
    pos_id: gpu_core::DeviceBuffer,
    tok_e: gpu_core::DeviceBuffer,
    pos_e: gpu_core::DeviceBuffer,
    res: Vec<gpu_core::DeviceBuffer>, // [n_layers+1] residual-stream snapshots, [d]
    xn1: gpu_core::DeviceBuffer,
    q: gpu_core::DeviceBuffer,
    k: gpu_core::DeviceBuffer,
    v: gpu_core::DeviceBuffer,
    scores: gpu_core::DeviceBuffer,
    probs: gpu_core::DeviceBuffer,
    ctx: gpu_core::DeviceBuffer,
    proj: gpu_core::DeviceBuffer,
    xmid: gpu_core::DeviceBuffer,
    ln2_out: gpu_core::DeviceBuffer,
    fc: gpu_core::DeviceBuffer,
    gelu: gpu_core::DeviceBuffer,
    ffn_out: gpu_core::DeviceBuffer,
    xn_final: gpu_core::DeviceBuffer,
    kcache: Vec<gpu_core::DeviceBuffer>, // per layer [cap*d]
    vcache: Vec<gpu_core::DeviceBuffer>,
    qbias: Vec<gpu_core::DeviceBuffer>, // per layer [d], split from attn.qkv.bias
    kbias: Vec<gpu_core::DeviceBuffer>,
    vbias: Vec<gpu_core::DeviceBuffer>,
}

impl Gpt {
    /// Load a model from a `.safetensors` checkpoint, sized for batch `b` × seq `t`.
    /// Streams the weights one tensor at a time off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader) — peak host ≈ one
    /// tensor of f32, never the whole-model `checkpoint::load` + `by_role("")`
    /// host copy on top of the device copy. See [`Gpt::from_reader`].
    pub fn load(path: &str, b: u32, t: u32) -> Gpt {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        Gpt::from_reader(&reader, b, t)
    }

    pub fn new(cfg: GptConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Gpt {
        let shard = Shard::whole(cfg.n_layers as usize);
        Gpt::new_shard(cfg, b, t, init, shard)
    }

    /// Streaming load: build directly from a mmap-backed [`WeightReader`],
    /// uploading one tensor at a time (peak host ≈ one tensor of f32) — the
    /// `checkpoint::load` + `by_role("")` whole-model host copy is never built.
    /// Numerically identical to [`Gpt::load`]; used by the resident serve path.
    pub fn from_reader(reader: &checkpoint::weightio::WeightReader, b: u32, t: u32) -> Gpt {
        let cfg = GptConfig::from_json(&reader.config());
        let shard = Shard::whole(cfg.n_layers as usize);
        Gpt::new_shard(cfg, b, t, reader, shard)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads.
    pub fn new_on(gpu: Gpu, cfg: GptConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Gpt {
        let shard = Shard::whole(cfg.n_layers as usize);
        Gpt::new_shard_on(gpu, cfg, b, t, init, shard)
    }

    /// Build a single pipeline **stage**: only `shard`'s layers (and endpoint
    /// weights) are allocated on this device. `Shard::whole` is the single-device
    /// path, byte-for-byte unchanged. `shard.gpu_index` names the canonical
    /// physical card (device registry); `Shard::ANY_GPU` keeps the ambient
    /// selection.
    pub fn new_shard(cfg: GptConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, shard: Shard) -> Gpt {
        let gpu = if shard.gpu_index == Shard::ANY_GPU {
            Gpu::new(PIPELINES)
        } else {
            Gpu::new_on_index(shard.gpu_index as u32, PIPELINES)
                .unwrap_or_else(|e| panic!("gpt shard placement: {e}"))
        };
        Gpt::new_shard_on(gpu, cfg, b, t, src, shard)
    }

    pub(crate) fn new_shard_on(gpu: Gpu, cfg: GptConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, shard: Shard) -> Gpt {
        let ps = ParamStore::new_src(&gpu, shard_param_list(&cfg, &shard), src);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let bht2 = (b * cfg.n_heads * t * t) as u64;
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

        // Residual stream: live only at this shard's boundaries (`start..=end`);
        // non-boundary indices are size-1 dummies (absolute `res[l]` indexing
        // preserved). A whole shard has every index live — identical to before.
        let mut res = Vec::new();
        let mut dres = Vec::new();
        for i in 0..=cfg.n_layers as usize {
            let live = i >= shard.start && i <= shard.end;
            res.push(if live { st(n * d) } else { st(1) });
            dres.push(if live { st(n * d) } else { st(1) });
        }
        let dummy_layer = || Layer {
            ln1_out: st(1), qkv: st(1), scores: st(1), probs: st(1), attn_ctx: st(1),
            xmid: st(1), ln2_out: st(1), fc: st(1), gelu: st(1),
        };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers as usize {
            layers.push(if shard.owns(l) {
                Layer {
                    ln1_out: st(n * d),
                    qkv: st(n * 3 * d),
                    scores: st(bht2),
                    probs: st(bht2),
                    attn_ctx: st(n * d),
                    xmid: st(n * d),
                    ln2_out: st(n * d),
                    fc: st(n * ff),
                    gelu: st(n * ff),
                }
            } else {
                dummy_layer()
            });
        }
        // Head-only buffers (final norm + lm_head + CE): only the last stage; the
        // `n*vocab` logits buffers dominate the saving on other stages.
        let head = shard.head;
        let hd_v = |x: u64| if head { st(x) } else { st(1) };
        let mut m = Gpt {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            shard,
            opt,
            tokens,
            targets,
            res,
            layers,
            proj: st(n * d),
            ffn_out: st(n * d),
            xn_final: hd_v(n * d),
            logits: hd_v(n * v),
            ce_buf: hd_v(n),
            dres,
            d_logits: hd_v(n * v),
            ce_stats: hd_v(n * 2),
            d_xn: st(n * d),
            d_branch: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_attn_ctx: st(n * d),
            d_scores: st(bht2),
            d_qkv: st(n * 3 * d),
            d_gelu: st(n * ff),
            d_fc: st(n * ff),
            ln_mean: st(n),
            ln_inv: st(n),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            dec: RefCell::new(None),
            dec_pos: Cell::new(0),
            gpu,
        };
        m.fwd_steps = m.forward_steps(m.b, m.t);
        m.bwd_steps = m.build_backward_steps();
        m
    }

    /// Upload a batch. `y` uses [`IGNORE`] for masked positions.
    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.ps.w(name)
    }

    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let mut s: Vec<Step> = Vec::new();

        // Token+positional embedding: only the embed stage; other stages receive
        // res[start] from the previous stage.
        if self.shard.embed {
            s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d));
            s.push(self.gpu.step(POS_ADD, &[&self.res[0], self.w("pos.weight")], &[n * d, d, t_use], n * d));
        }

        for l in self.shard.start..self.shard.end {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // attention
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &self.res[l], self.w(&p("ln1.weight")), self.w(&p("ln1.bias")), &lb.ln1_out, d, n, 1e-5));
            let (mk, mt) = linear_kernel(n as usize, (3 * d) as usize);
            s.push(self.gpu.step(mk, &[&lb.ln1_out, self.w(&p("attn.qkv.weight")), &lb.qkv], &[n, d, 3 * d], mt));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.qkv, self.w(&p("attn.qkv.bias"))], &[n, 3 * d], n * 3 * d));
            s.push(self.gpu.step(ATTN_SCORES, &[&lb.qkv, &lb.scores], &[b_use, c.n_heads, t_use, hd, 3 * d, 0, d], b_use * c.n_heads * t_use * t_use));
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&lb.scores, &lb.probs], &[b_use, c.n_heads, t_use], b_use * c.n_heads * t_use));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_ctx], &[b_use, c.n_heads, t_use, hd, 3 * d, 2 * d, d], b_use * c.n_heads * t_use * hd));
            let (mk, mt) = linear_kernel(n as usize, d as usize);
            s.push(self.gpu.step(mk, &[&lb.attn_ctx, self.w(&p("attn.out.weight")), &self.proj], &[n, d, d], mt));
            s.push(self.gpu.step(BIAS_ADD, &[&self.proj, self.w(&p("attn.out.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // MLP: fc -> GELU -> proj
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &lb.xmid, self.w(&p("ln2.weight")), self.w(&p("ln2.bias")), &lb.ln2_out, d, n, 1e-5));
            let (mk, mt) = linear_kernel(n as usize, ff as usize);
            s.push(self.gpu.step(mk, &[&lb.ln2_out, self.w(&p("mlp.fc.weight")), &lb.fc], &[n, d, ff], mt));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.fc, self.w(&p("mlp.fc.bias"))], &[n, ff], n * ff));
            s.push(self.gpu.step(GELU, &[&lb.fc, &lb.gelu], &[n * ff], n * ff));
            let (mk, mt) = linear_kernel(n as usize, d as usize);
            s.push(self.gpu.step(mk, &[&lb.gelu, self.w(&p("mlp.proj.weight")), &self.ffn_out], &[n, ff, d], mt));
            s.push(self.gpu.step(BIAS_ADD, &[&self.ffn_out, self.w(&p("mlp.proj.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.ffn_out, &self.res[l + 1]], &[n * d], n * d));
        }

        // Head epilogue (final norm + lm_head + CE): only the head stage.
        if !self.shard.head {
            return s;
        }
        let last = c.n_layers as usize;
        s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &self.res[last], self.w("ln.weight"), self.w("ln.bias"), &self.xn_final, d, n, 1e-5));
        let (mk, mt) = linear_kernel(n as usize, v as usize);
        s.push(self.gpu.step(mk, &[&self.xn_final, self.w("lm_head.weight"), &self.logits], &[n, d, v], mt));
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
        s
    }

    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd_steps);
    }

    /// The full `[B*T, vocab]` logits from the last `forward_submit` (blocking
    /// read-back). Exposed for cross-backend/kernel equivalence benchmarks.
    pub fn logits_host(&self) -> Vec<f32> {
        let n = (self.b * self.t * self.cfg.vocab) as usize;
        self.gpu.read(&self.logits, n)
    }

    pub fn loss(&self) -> f32 {
        let n = (self.b * self.t) as usize;
        let losses = self.gpu.read(&self.ce_buf, n);
        losses.iter().sum::<f32>() / self.count.get()
    }

    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    pub fn backward(&self) {
        let n = self.b * self.t;
        let v = self.cfg.vocab;
        self.gpu.write(&self.ce_grad_uni, &[n, v, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    // ---- pipeline-parallel cross-stage seam (see `crate::shard`) ----
    fn res_numel(&self) -> usize {
        (self.b * self.t) as usize * self.cfg.d_model as usize
    }
    /// Read this stage's OUTPUT residual `res[end]` (input to the next stage).
    pub fn read_out_res(&self) -> Vec<f32> {
        self.gpu.read(&self.res[self.shard.end], self.res_numel())
    }
    /// Write this stage's INPUT residual `res[start]` (from the previous stage).
    pub fn write_in_res(&self, data: &[f32]) {
        self.gpu.write(&self.res[self.shard.start], bytemuck::cast_slice(data));
    }
    /// Read this stage's INPUT-side residual grad `dres[start]` (to the previous stage).
    pub fn read_in_dres(&self) -> Vec<f32> {
        self.gpu.read(&self.dres[self.shard.start], self.res_numel())
    }
    /// Write this stage's OUTPUT-side residual grad `dres[end]` (from the next stage).
    pub fn write_out_dres(&self, data: &[f32]) {
        self.gpu.write(&self.dres[self.shard.end], bytemuck::cast_slice(data));
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let g = |name: &str| self.ps.g(name);
        let p = |l: usize, name: &str| format!("blocks.{l}.{name}");
        let mut s: Vec<Step> = Vec::new();

        // head (no bias) + final layernorm — head stage only; other stages receive
        // dres[end] from the next stage and start straight at the layer loop.
        if self.shard.head {
            // Two-pass CE gradient (see qwen): O(rows*vocab) not O(rows*vocab^2).
            s.push(self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &self.ce_stats], &[n, v, IGNORE], n));
            s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.ce_stats, &self.d_logits], n * v));
            let (bk, bt) = dw_kernel(v as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_logits, &self.xn_final, g("lm_head.weight")], &[n, d, v], bt));
            let (bk, bt) = dx_kernel(n as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_logits, self.w("lm_head.weight"), &self.d_xn], &[n, d, v, 0], bt));
            let last = c.n_layers as usize;
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &self.res[last], &self.ln_mean, &self.ln_inv, d, n, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_xn, &self.res[last], &self.ln_mean, &self.ln_inv, g("ln.weight")], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_xn, g("ln.bias")], &[d, n], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &self.res[last], self.w("ln.weight"), &self.d_xn, &self.dres[last], d, n, 1e-5));
        }

        for l in (self.shard.start..self.shard.end).rev() {
            let lb = &self.layers[l];
            // MLP backward (input grad = dres[l+1])
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dres[l + 1], g(&p(l, "mlp.proj.bias"))], &[n, d], d));
            let (bk, bt) = dw_kernel(d as usize, ff as usize);
            s.push(self.gpu.step(bk, &[&self.dres[l + 1], &lb.gelu, g(&p(l, "mlp.proj.weight"))], &[n, ff, d], bt));
            let (bk, bt) = dx_kernel(n as usize, ff as usize);
            s.push(self.gpu.step(bk, &[&self.dres[l + 1], self.w(&p(l, "mlp.proj.weight")), &self.d_gelu], &[n, ff, d, 0], bt));
            s.push(self.gpu.step(GELU_BWD, &[&lb.fc, &self.d_gelu, &self.d_fc], &[n * ff], n * ff));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_fc, g(&p(l, "mlp.fc.bias"))], &[n, ff], ff));
            let (bk, bt) = dw_kernel(ff as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_fc, &lb.ln2_out, g(&p(l, "mlp.fc.weight"))], &[n, d, ff], bt));
            let (bk, bt) = dx_kernel(n as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_fc, self.w(&p(l, "mlp.fc.weight")), &self.d_branch], &[n, d, ff, 0], bt));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &lb.xmid, &self.ln_mean, &self.ln_inv, d, n, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &lb.xmid, &self.ln_mean, &self.ln_inv, g(&p(l, "ln2.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln2.bias"))], &[d, n], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &lb.xmid, self.w(&p(l, "ln2.weight")), &self.d_branch, &self.d_tmp, d, n, 1e-5));
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // attention backward (input grad = dxmid)
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dxmid, g(&p(l, "attn.out.bias"))], &[n, d], d));
            let (bk, bt) = dw_kernel(d as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.dxmid, &lb.attn_ctx, g(&p(l, "attn.out.weight"))], &[n, d, d], bt));
            let (bk, bt) = dx_kernel(n as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.dxmid, self.w(&p(l, "attn.out.weight")), &self.d_attn_ctx], &[n, d, d, 0], bt));
            s.push(self.gpu.step(ATTN_DSCORES, &[&self.d_attn_ctx, &lb.qkv, &lb.probs, &self.d_scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t));
            s.push(self.gpu.step(ATTN_DV, &[&lb.probs, &self.d_attn_ctx, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_DQ, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_DK, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_qkv, g(&p(l, "attn.qkv.bias"))], &[n, 3 * d], 3 * d));
            let (bk, bt) = dw_kernel((3 * d) as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_qkv, &lb.ln1_out, g(&p(l, "attn.qkv.weight"))], &[n, d, 3 * d], bt));
            let (bk, bt) = dx_kernel(n as usize, d as usize);
            s.push(self.gpu.step(bk, &[&self.d_qkv, self.w(&p(l, "attn.qkv.weight")), &self.d_branch], &[n, d, 3 * d, 0], bt));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &self.res[l], &self.ln_mean, &self.ln_inv, d, n, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &self.res[l], &self.ln_mean, &self.ln_inv, g(&p(l, "ln1.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln1.bias"))], &[d, n], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &self.res[l], self.w(&p(l, "ln1.weight")), &self.d_branch, &self.d_tmp, d, n, 1e-5));
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embeddings backward — only the embed stage (owns tok/pos and dres[0]).
        if self.shard.embed {
            s.push(self.gpu.step(POS_BWD, &[&self.dres[0], g("pos.weight")], &[self.b, self.t, d], self.t * d));
            s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], g("tok.weight")], &[n, d, c.vocab], c.vocab * d));
        }
        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// OFFLINE FLOP/OPS cost of the recorded batch forward — walks the step
    /// list, executes nothing. Per this device/stage (a sharded instance
    /// reports only its own layers); see `gpu_core::cost`.
    pub fn cost_fwd(&self) -> gpu_core::cost::CostReport {
        self.gpu.cost_of(&self.fwd_steps)
    }

    /// OFFLINE cost of the recorded backward pass.
    pub fn cost_bwd(&self) -> gpu_core::cost::CostReport {
        self.gpu.cost_of(&self.bwd_steps)
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
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

    /// Logits for every position of a single sequence (B must be 1, t>=len).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab) as usize)
    }

    // ---- incremental KV-cache decode (nanoGPT single-token twin of forward) ----

    /// Reset the incremental KV cache to an empty sequence (next [`Self::step`] is
    /// absolute position 0).
    pub fn reset_cache(&self) {
        self.dec_pos.set(0);
    }

    /// The absolute position the next [`Self::step`] will decode (cache fill level).
    pub fn cache_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// Build the lazy decode state (buffers + K/V cache) the first time it is
    /// needed. The fused `attn.qkv.bias` is read back and split into contiguous
    /// per-region `[d]` bias buffers here.
    fn ensure_decode(&self) {
        if self.dec.borrow().is_some() {
            return;
        }
        assert!(self.shard.is_whole(self.cfg.n_layers as usize), "step() requires a whole-model shard");
        let c = &self.cfg;
        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let nh = c.n_heads as u64;
        let cap = c.block_size;
        let g = &self.gpu;
        let st = |x: u64| g.storage(x);
        let idbuf = || g.buffer("dec_id", 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let mut res = Vec::new();
        for _ in 0..=c.n_layers as usize {
            res.push(st(d));
        }
        let (mut kcache, mut vcache) = (Vec::new(), Vec::new());
        let (mut qbias, mut kbias, mut vbias) = (Vec::new(), Vec::new(), Vec::new());
        let dd = c.d_model as usize;
        for l in 0..c.n_layers as usize {
            kcache.push(st(cap as u64 * d));
            vcache.push(st(cap as u64 * d));
            let bias = self.ps.read_weight(g, &format!("blocks.{l}.attn.qkv.bias")); // [3d]
            qbias.push(g.storage_init("dec_qbias", &bias[0..dd]));
            kbias.push(g.storage_init("dec_kbias", &bias[dd..2 * dd]));
            vbias.push(g.storage_init("dec_vbias", &bias[2 * dd..3 * dd]));
        }
        let dec = Decode {
            cap,
            tok_id: idbuf(),
            pos_id: idbuf(),
            tok_e: st(d),
            pos_e: st(d),
            res,
            xn1: st(d),
            q: st(d),
            k: st(d),
            v: st(d),
            scores: st(nh * cap as u64),
            probs: st(nh * cap as u64),
            ctx: st(d),
            proj: st(d),
            xmid: st(d),
            ln2_out: st(d),
            fc: st(ff),
            gelu: st(ff),
            ffn_out: st(d),
            xn_final: st(d),
            kcache,
            vcache,
            qbias,
            kbias,
            vbias,
        };
        *self.dec.borrow_mut() = Some(dec);
    }

    /// **Incremental KV-cache decode** of a single token id at the current cache
    /// position, returning the final-LayerNorm hidden state (`[d_model]`) for that
    /// new token. This is the `O(T)`-per-token twin of the `O(T²)` full recompute
    /// ([`Self::logits_all`]): the same dense nanoGPT block math, but only the new
    /// token's Q/K/V are projected; its K/V are appended to the persistent per-layer
    /// cache and a single query attends over positions `0..=pos`. Expressed entirely
    /// in the existing WGSL op set, so it runs on whatever backend `Gpu` selected
    /// (GPU or the wgsl-cpu JIT). Apply the (untied) `lm_head` to the returned hidden
    /// row for logits.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        self.ensure_decode();
        let pos = self.dec_pos.get();
        let hidden = self.decode_at(token_id, pos);
        self.dec_pos.set(pos + 1);
        hidden
    }

    /// Record + run the incremental decode tape for one token at absolute `pos`.
    fn decode_at(&self, token_id: u32, pos: u32) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.d_ff;
        let nh = c.n_heads;
        let hd = c.head_dim();
        let dd = d as usize;
        let t = pos + 1; // cached length after appending this token
        let scale = 1.0f32 / (hd as f32).sqrt();
        let g = &self.gpu;
        let w = |name: &str| self.ps.w(name);
        let dec_ref = self.dec.borrow();
        let dec = dec_ref.as_ref().unwrap();
        let cap_p = dec.cap;
        assert!(pos < dec.cap, "decode pos {pos} exceeds block_size {}", dec.cap);

        // --- token+position embedding: res[0] = tok[id] + pos[pos] ---
        g.write(&dec.tok_id, &[token_id]);
        g.write(&dec.pos_id, &[pos]);
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(EMBED, &[&dec.tok_id, w("tok.weight"), &dec.tok_e], &[d, 1], d));
        s.push(g.step(EMBED, &[&dec.pos_id, w("pos.weight"), &dec.pos_e], &[d, 1], d));
        s.push(g.step(ADD2, &[&dec.tok_e, &dec.pos_e, &dec.res[0]], &[d], d));

        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            // --- attention: LN -> fused-QKV (as three contiguous slices) -> attend ---
            s.push(model::block::layernorm_fwd(g, &LN_IDS, &dec.res[l], w(&p("ln1.weight")), w(&p("ln1.bias")), &dec.xn1, d, 1, 1e-5));
            // q/k/v via the fused `attn.qkv.weight [3d,d]` sliced by output-row block
            // (offsets d*d, 2*d*d are large + 256B-aligned) -> contiguous [d] buffers.
            let dw = d * d;
            s.push(g.step_sliced(MATMUL, &[&dec.xn1, w(&p("attn.qkv.weight")), &dec.q], &[(0, 0), (0, dw as u64), (0, 0)], &[1, d, d], d));
            s.push(g.step_sliced(MATMUL, &[&dec.xn1, w(&p("attn.qkv.weight")), &dec.k], &[(0, 0), (dw as u64, dw as u64), (0, 0)], &[1, d, d], d));
            s.push(g.step_sliced(MATMUL, &[&dec.xn1, w(&p("attn.qkv.weight")), &dec.v], &[(0, 0), (2 * dw as u64, dw as u64), (0, 0)], &[1, d, d], d));
            s.push(g.step(BIAS_ADD, &[&dec.q, &dec.qbias[l]], &[1, d], d));
            s.push(g.step(BIAS_ADD, &[&dec.k, &dec.kbias[l]], &[1, d], d));
            s.push(g.step(BIAS_ADD, &[&dec.v, &dec.vbias[l]], &[1, d], d));
            // append this token's K/V to the persistent cache (row = pos)
            s.push(g.step(KV_APPEND, &[&dec.k, &dec.kcache[l]], &[d, pos], d));
            s.push(g.step(KV_APPEND, &[&dec.v, &dec.vcache[l]], &[d, pos], d));
            // single-query attention over positions 0..t (MHA: group=1, kv_stride=d)
            s.push(g.step(ATTN_DECODE_SCORES, &[&dec.q, &dec.kcache[l], &dec.scores], &[nh, 1, hd, t, cap_p, d, scale.to_bits()], nh * t));
            s.push(g.step(DECODE_SOFTMAX, &[&dec.scores, &dec.probs], &[nh, t, cap_p], nh));
            s.push(g.step(ATTN_DECODE_APPLY, &[&dec.probs, &dec.vcache[l], &dec.ctx], &[nh, 1, hd, t, cap_p, d], nh * hd));
            s.push(g.step(MATMUL, &[&dec.ctx, w(&p("attn.out.weight")), &dec.proj], &[1, d, d], d));
            s.push(g.step(BIAS_ADD, &[&dec.proj, w(&p("attn.out.bias"))], &[1, d], d));
            s.push(g.step(ADD2, &[&dec.res[l], &dec.proj, &dec.xmid], &[d], d));
            // --- MLP: LN -> fc -> GELU -> proj ---
            s.push(model::block::layernorm_fwd(g, &LN_IDS, &dec.xmid, w(&p("ln2.weight")), w(&p("ln2.bias")), &dec.ln2_out, d, 1, 1e-5));
            s.push(g.step(MATMUL, &[&dec.ln2_out, w(&p("mlp.fc.weight")), &dec.fc], &[1, d, ff], ff));
            s.push(g.step(BIAS_ADD, &[&dec.fc, w(&p("mlp.fc.bias"))], &[1, ff], ff));
            s.push(g.step(GELU, &[&dec.fc, &dec.gelu], &[ff], ff));
            s.push(g.step(MATMUL, &[&dec.gelu, w(&p("mlp.proj.weight")), &dec.ffn_out], &[1, ff, d], d));
            s.push(g.step(BIAS_ADD, &[&dec.ffn_out, w(&p("mlp.proj.bias"))], &[1, d], d));
            s.push(g.step(ADD2, &[&dec.xmid, &dec.ffn_out, &dec.res[l + 1]], &[d], d));
        }
        let last = c.n_layers as usize;
        s.push(model::block::layernorm_fwd(g, &LN_IDS, &dec.res[last], w("ln.weight"), w("ln.bias"), &dec.xn_final, d, 1, 1e-5));
        g.submit(&[], &s);
        g.read(&dec.xn_final, dd)
    }

    pub fn save(&self, path: &str) {
        self.save_with_itos(path, None);
    }

    /// Save the checkpoint, optionally embedding the char-tokenizer vocab (`itos`)
    /// in the manifest so inference can reconstruct the tokenizer without the
    /// original dataset. `gpt gen` reads it back via [`Gpt::load_itos`].
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
        // "brain/gpt" matches docs/models/naming.md's reserved-vendor fallback
        // -- the same id crates/cli/src/resident_llm.rs::GptResident::from_env
        // synthesizes for an env-loaded checkpoint -- so a checkpoint saved
        // here is auto-discoverable by crates/cli/src/model_dir.rs without
        // requiring BRAIN_GPT_WEIGHTS to be set.
        checkpoint::save_carded(path, config, &tensors, &checkpoint::st::ModelCard::new("brain/gpt", "gpt"));
    }

    /// The embedded char-tokenizer vocab from a config object, if it was trained
    /// on a char-level dataset (else `None`, e.g. BPE checkpoints).
    pub fn itos_from_config(cfg: &Value) -> Option<Vec<char>> {
        let arr = cfg.get("itos")?.as_array()?;
        Some(
            arr.iter()
                .filter_map(|v| v.as_str().and_then(|s| s.chars().next()))
                .collect(),
        )
    }

    /// The embedded char-tokenizer vocab from a checkpoint. Reads only the
    /// mmap'd header/config (no tensor data is faulted in).
    pub fn load_itos(path: &str) -> Option<Vec<char>> {
        Self::itos_from_config(&checkpoint::weightio::WeightReader::open(path).ok()?.config())
    }
}

// ---- the architecture-agnostic Model seam (ADR 0001 §2.2/§2.3) ----
//
// GPT is the reference implementation: it already exposes nearly the whole
// surface as inherent methods, so these impls are thin adapters. `set_batch`
// maps `Batch::Lm` onto the inherent two-slice upload; `logits_all` wraps the
// always-present token head in `Some`.

impl model::ModelConfig for GptConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        GptConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        GptConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        GptConfig::from_json(v)
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
        self.with_ff_default()
    }
}

impl model::Model for Gpt {
    type Config = GptConfig;

    fn new(cfg: GptConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Gpt::new(cfg, b, t, init)
    }

    fn init_weights(cfg: &GptConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &GptConfig {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Gpt::set_batch(self, tokens, targets),
            _ => panic!("gpt::Gpt only supports Batch::Lm"),
        }
    }

    fn forward(&self) -> f32 {
        Gpt::forward(self)
    }
    fn backward(&self) {
        Gpt::backward(self)
    }
    fn zero_grads(&self) {
        Gpt::zero_grads(self)
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Gpt::adamw_step(self, t, lr, wd, clip, extra_scale)
    }

    fn poll_wait(&self) {
        Gpt::poll_wait(self)
    }

    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Gpt::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Gpt::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Gpt::read_grad(self, name)
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Gpt::logits_all(self, tokens))
    }

    fn save(&self, path: &str) {
        Gpt::save(self, path)
    }
    fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        Gpt::save_with_itos(self, path, itos)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The streaming `Gpt::load` (mmap `WeightReader`, one tensor uploaded at a
    /// time) yields byte-identical device weights to the eager whole-model-host-
    /// map path (`Gpt::new` over `by_role("")`), and both match the source init
    /// exactly. GPU-gated (testgpu / MOE_SKIP_GPU).
    #[test]
    fn streaming_load_matches_eager() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig::tiny();
        let cfg_json = cfg.to_json();
        let init = crate::init::init_weights(&cfg, 5);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            init.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64], v.clone())).collect();
        let path = std::env::temp_dir().join(format!("gpt-stream-parity-{}.st", std::process::id()));
        let p = path.to_str().unwrap();
        checkpoint::st::save_safetensors(p, &tensors, &cfg_json, None).unwrap();

        let eager = Gpt::new(cfg, 1, 8, &checkpoint::load(p).by_role(""));
        let streamed = Gpt::load(p, 1, 8);

        for (name, _) in &eager.ps.params {
            assert_eq!(eager.read_weight(name), streamed.read_weight(name), "weight {name}");
            assert_eq!(&streamed.read_weight(name), &init[name], "streamed {name} vs source");
        }
        std::fs::remove_file(&path).ok();
    }

    /// The incremental KV-cache `step` must reproduce the `O(T²)` full-recompute
    /// (`logits_all`) for every prefix: the cache is algebraically exact, same
    /// engine + weights, so the only difference is float reduction order. Runs on
    /// whatever backend `Gpu` selected (GPU, or `BRAIN_DEVICE=cpu`).
    #[test]
    fn kv_step_matches_full_recompute() {
        if gpu_disabled() {
            return;
        }
        // d_model=32 (=> the fused-qkv weight-slice offsets d² are 256B-aligned),
        // biases random so the fused-QKV bias split is genuinely exercised.
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let seq = 10usize;
        let mut rng = data::rng::Rng::new(4321);

        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, numel) in cfg.param_list() {
            let val = if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "ln.weight" {
                vec![1.0f32; numel] // LayerNorm gain = 1
            } else {
                (0..numel).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            init.insert(name, val);
        }
        let m = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, cfg.block_size, &init);
        let lm_head = m.read_weight("lm_head.weight"); // [v, d], untied, no bias

        let tokens: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % v as u64) as u32).collect();

        // Incremental: feed one token at a time; head-project each hidden row.
        m.reset_cache();
        let inc_logits: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                assert_eq!(m.cache_pos(), i as u32);
                let h = m.step(tok); // [d] final-LayerNorm hidden
                (0..v)
                    .map(|o| {
                        let row = &lm_head[o * d..(o + 1) * d];
                        (0..d).map(|k| row[k] * h[k]).sum::<f32>()
                    })
                    .collect()
            })
            .collect();

        // Reference: full recompute of each prefix; compare the last row's logits.
        let mut worst = 0.0f32;
        for i in 0..seq {
            let full = m.logits_all(&tokens[..=i]); // [(i+1)*v]
            let last = &full[i * v..(i + 1) * v];
            let err = maxabs(&inc_logits[i], last);
            worst = worst.max(err);
            assert!(err < 2e-3, "prefix {i}: KV step vs full recompute maxabs={err}");
        }
        println!("kv_step_matches_full_recompute: seq={seq} worst maxabs={worst:e}");
    }

    #[test]
    fn param_list_shapes() {
        let cfg = GptConfig::tiny(); // v=65 d=32 layers=2 ff=128
        let m: HashMap<_, _> = cfg.param_list().into_iter().collect();
        assert_eq!(m["tok.weight"], 65 * 32);
        assert_eq!(m["pos.weight"], 64 * 32);
        assert_eq!(m["blocks.0.mlp.fc.weight"], 128 * 32);
        assert_eq!(m["blocks.1.mlp.proj.weight"], 32 * 128);
        assert_eq!(m["lm_head.weight"], 65 * 32);
        assert!(!m.contains_key("lm_head.bias"));
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = GptConfig { vocab: 100, block_size: 128, n_layers: 3, d_model: 48, n_heads: 6, d_ff: 192 };
        let back = GptConfig::from_json(&cfg.to_json());
        assert_eq!(back.vocab, 100);
        assert_eq!(back.n_layers, 3);
        assert_eq!(back.head_dim(), 8);
    }

    #[test]
    fn forward_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 7);
        let model = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 3 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 3 + 1) % 65) as u32).collect();
        model.set_batch(&x, &y);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss {l1}");
        assert!((l1 - l2).abs() < 1e-6, "not deterministic");
        // untrained CE should be near ln(vocab).
        assert!(l1 < 2.0 * (65f32).ln(), "loss implausibly large: {l1}");
    }

    #[test]
    fn backward_grads_finite() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 9);
        let model = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 5 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 5 + 1) % 65) as u32).collect();
        model.set_batch(&x, &y);
        model.zero_grads();
        model.forward();
        model.backward();
        for (name, _) in model.ps.params.iter() {
            assert!(model.read_grad(name).iter().all(|v| v.is_finite()), "nan grad in {name}");
        }
    }

    #[test]
    fn one_adamw_step_reduces_loss_on_fixed_batch() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 11);
        let model = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 7 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 65) as u32).collect();
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
        assert!(after < before, "overfit step did not reduce loss: {before} -> {after}");
    }
}
