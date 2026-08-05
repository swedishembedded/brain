// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2 (`glm_moe_dsa`) decoder — forward + backprop as WGSL compute
//! dispatches on the shared `gpu_core` engine. Phase 1: the dense MLA-MoE core
//! (the DSA indexer is a no-op while `index_topk >= block_size`, so attention is
//! exact dense MLA — the regime tiny models / tests run in).
//!
//! Per pre-norm block (no biases; RMSNorm everywhere):
//!   h   = RMSNorm(x)·input_ln
//!   --- MLA attention ---
//!   qc  = RMSNorm(h·Wq_a)·q_a_norm
//!   q_pass = qc·Wq_b_nope ;  q_rot = RoPE(qc·Wq_b_rope)
//!   kvc = h·Wkv_a_c ;  k_rot = RoPE(h·Wkv_a_rope)          (shared MQA key)
//!   kvcn = RMSNorm(kvc)·kv_a_norm
//!   k_pass = kvcn·Wkv_b_nope ;  v = kvcn·Wkv_b_v
//!   ctx = softmax(MLA-scores(q_pass,q_rot,k_pass,k_rot)) · v
//!   x  += Wo·ctx
//!   --- MLP: dense SwiGLU (first_k_dense layers) or MoE ---
//!   h   = RMSNorm(x)·post_ln
//!   MoE: x += Σ_topk gate_e·SwiGLU_e(h) + SwiGLU_shared(h)    (sigmoid noaux_tc router)
//!   logits = lm_head·RMSNorm(x)·norm  (untied) ;  loss = masked cross-entropy

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use optim::Optim;
use paramstore::{ParamStore, Role};

use crate::config::{GlmConfig, IdxMode};

/// Cross-entropy ignore index (masked target positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const MATMUL_REG3: usize = 42;
const MATMUL_DX: usize = 2;
const MATMUL_DW: usize = 3;
const RMSNORM: usize = 4;
const RMS_INV: usize = 5;
const RMSNORM_DX: usize = 6;
const RMSNORM_DW: usize = 7;
const ROPE: usize = 8;
const ROPE_BWD: usize = 9;
const MLA_SCORES: usize = 10;
const ATTN_SOFTMAX: usize = 11;
const ATTN_APPLY: usize = 12;
const ATTN_BWD_DSCORES: usize = 13;
const ATTN_BWD_DV: usize = 14;
const MLA_BWD_DQ_PASS: usize = 15;
const MLA_BWD_DK_PASS: usize = 16;
const MLA_BWD_DQ_ROPE: usize = 17;
const MLA_BWD_DK_ROPE: usize = 18;
const SILU_MUL: usize = 19;
const SILU_DA: usize = 20;
const SILU_DB: usize = 21;
const ROUTER_SIG: usize = 22;
const ROUTER_SIG_BWD: usize = 23;
const SCALE_ADD: usize = 24;
const SCALE_ADD_DEXP: usize = 25;
const SCALE_ADD_DGATE: usize = 26;
const ADD2: usize = 27;
const CE_VALUE: usize = 28;
const CE_GRAD: usize = 29;
const EMB_BWD: usize = 30;
const ADAMW: usize = 31;
const GRADNORM_SQ: usize = 32;
const GRAD_SCALE: usize = 33;
const CLIP_COEF: usize = 34;
const GRAD_SCALE_BUF: usize = 35;
// DSA indexer (Phase 2; forward-only — the indexer is detached from the LM loss)
const LAYERNORM: usize = 36;
const MLA_INDEX_SCORES: usize = 37;
const ROPE_SUB: usize = 38;
const TOPK_MASK: usize = 39;
const ADD_INDEX_MASK: usize = 40;
/// `out += a` with a single read_write binding. ADD2 cannot express an
/// accumulate-into-self: binding the same buffer read-only AND read-write in
/// one dispatch is a wgpu usage-scope violation (it panics on the GPU
/// backend), which is exactly what the MTP path used to do.
const ADD_INPLACE: usize = 41;
// ---- incremental KV-cache decode kernels (indices continue after MATMUL_REG3=42) ----
const KV_APPEND: usize = 43;
const DECODE_SOFTMAX: usize = 44;
const ATTN_DECODE_APPLY: usize = 45;
const MLA_DECODE_SCORES: usize = 46;
const ROPE_TRAIN_AT: usize = 47;

/// MLA decode scores: a SINGLE query (the new token at `pos`) against all `t`
/// cached keys, MLA-style (two-part score = per-head `nope` + shared MQA `rope`).
/// Mirrors `kernels::MLA_SCORES` exactly (nope loop then rope loop, then
/// `*inverseSqrt(np+rp)`) so the score is bit-identical to the recompute path
/// for the query row. `q_pass`=[H*np], `q_rot`=[H*rp], `kpass`=[cap, H*np],
/// `krot`=[cap, rp] (shared), `scores`=[H, cap]. One invocation per (h, j<t).
const MLA_DECODE_SCORES_WGSL: &str = r#"
struct Params { n_heads: u32, np: u32, rp: u32, t: u32, cap: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q_pass: array<f32>;
@group(0) @binding(2) var<storage, read>       q_rot:  array<f32>;
@group(0) @binding(3) var<storage, read>       kpass:  array<f32>;
@group(0) @binding(4) var<storage, read>       krot:   array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_heads * p.t;
    if (idx >= total) { return; }
    let h = idx / p.t;
    let j = idx % p.t;
    let np = p.np;
    let rp = p.rp;
    let qp_base = h * np;
    let kp_base = j * (p.n_heads * np) + h * np;
    let qr_base = h * rp;
    let kr_base = j * rp;
    var s = 0.0;
    for (var d: u32 = 0u; d < np; d = d + 1u) { s = s + q_pass[qp_base + d] * kpass[kp_base + d]; }
    for (var d: u32 = 0u; d < rp; d = d + 1u) { s = s + q_rot[qr_base + d] * krot[kr_base + d]; }
    scores[h * p.cap + j] = s * inverseSqrt(f32(np + rp));
}
"#;

/// RoPE at an EXPLICIT absolute position, INTERLEAVED (adjacent pairs `2j,2j+1`)
/// convention — the decode-step twin of `kernels::ROPE_TRAIN` (which GLM's
/// forward uses). Identical math (base 10000, pairs `2j,2j+1`) but the rotary
/// position is `pos_base + row` instead of `row % tcols`. NOT interchangeable
/// with `kernels::ROPE_AT`, which is the half-split (GPT-NeoX) convention.
const ROPE_TRAIN_AT_WGSL: &str = r#"
struct Params { n_rows: u32, n_heads: u32, head_dim: u32, row_stride: u32, base_off: u32, pos_base: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.n_rows * p.n_heads * half;
    if (gidx >= total) { return; }
    let j = gidx % half;
    let tmp = gidx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = p.pos_base + row;
    let base = row * p.row_stride + p.base_off + h * p.head_dim + 2u * j;
    let angle = f32(pos) * pow(10000.0, -f32(2u * j) / f32(p.head_dim));
    let c = cos(angle);
    let sn = sin(angle);
    let e = buf[base];
    let o = buf[base + 1u];
    buf[base]      = e * c - o * sn;
    buf[base + 1u] = e * sn + o * c;
}
"#;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_train", kernels::ROPE_TRAIN),
    ("rope_train_bwd", kernels::ROPE_TRAIN_BWD),
    ("mla_scores", kernels::MLA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("mla_bwd_dq_pass", kernels::MLA_BWD_DQ_PASS),
    ("mla_bwd_dk_pass", kernels::MLA_BWD_DK_PASS),
    ("mla_bwd_dq_rope", kernels::MLA_BWD_DQ_ROPE),
    ("mla_bwd_dk_rope", kernels::MLA_BWD_DK_ROPE),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
    ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
    ("scale_add", kernels::SCALE_ADD),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("layernorm", kernels::LAYERNORM),
    ("mla_index_scores", kernels::MLA_INDEX_SCORES),
    ("rope_sub", kernels::ROPE_SUB),
    ("topk_mask", kernels::TOPK_MASK),
    ("add_index_mask", kernels::ADD_INDEX_MASK),
    ("add_inplace", kernels::ADD_INPLACE),
    ("matmul_reg3", kernels::MATMUL_REG3),
    // incremental KV-cache decode kernels (indices 43..=47)
    ("kv_append", kernels::KV_APPEND),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("mla_decode_scores", MLA_DECODE_SCORES_WGSL),
    ("rope_train_at", ROPE_TRAIN_AT_WGSL),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// MLP variant per layer (cached activations for backprop).
enum Mlp {
    Dense {
        gate_pre: DeviceBuffer,
        up: DeviceBuffer,
        h: DeviceBuffer,
    },
    Moe {
        router_logits: DeviceBuffer,
        gate: DeviceBuffer,   // combine weights [n, E] (0 for non-selected)
        probs: DeviceBuffer,  // sigmoid scores [n, E]
        gate_pre: Vec<DeviceBuffer>,
        up: Vec<DeviceBuffer>,
        h: Vec<DeviceBuffer>,
        expert_out: Vec<DeviceBuffer>,
        sh_gate: DeviceBuffer,
        sh_up: DeviceBuffer,
        sh_h: DeviceBuffer,
    },
}

struct LayerBufs {
    xn1: DeviceBuffer,
    q_c: DeviceBuffer,
    q_c_n: DeviceBuffer,
    q_pass: DeviceBuffer,
    q_rot: DeviceBuffer,
    kv_c: DeviceBuffer,
    kv_c_n: DeviceBuffer,
    k_rot: DeviceBuffer,
    k_pass: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    mlp: Mlp,
}

pub struct Glm {
    pub gpu: Gpu,
    pub cfg: GlmConfig,
    ps: ParamStore,
    opt: Optim,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    ce_grad_uni: DeviceBuffer,

    res: Vec<DeviceBuffer>,
    dres: Vec<DeviceBuffer>,
    layers: Vec<LayerBufs>,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    // shared forward temporaries
    scores: DeviceBuffer,
    proj: DeviceBuffer,
    moe_acc: DeviceBuffer,
    sh_out: DeviceBuffer,
    mlp_out: DeviceBuffer,

    // DSA indexer forward temporaries (used only when `cfg.has_indexer()`).
    // `idx_mask` persists across IndexShare `Shared` layers; the rest are
    // recomputed at each `Full` layer.
    q_idx: DeviceBuffer,
    k_idx_pre: DeviceBuffer,
    k_idx: DeviceBuffer,
    idx_weights: DeviceBuffer,
    index_scores: DeviceBuffer,
    idx_mask: DeviceBuffer,

    // Multi-Token Prediction head (used only when `cfg.mtp`). Predicts token t+2.
    mtp_input: DeviceBuffer,
    mtp_target: DeviceBuffer,
    mtp_e: DeviceBuffer,
    mtp_en: DeviceBuffer,
    mtp_hn: DeviceBuffer,
    mtp_ehp: DeviceBuffer,
    mtp_ehp2: DeviceBuffer,
    mtp_xn: DeviceBuffer,
    mtp_gate_pre: DeviceBuffer,
    mtp_up: DeviceBuffer,
    mtp_h: DeviceBuffer,
    mtp_mlp_out: DeviceBuffer,
    mtp_block_out: DeviceBuffer,
    mtp_final: DeviceBuffer,
    mtp_logits: DeviceBuffer,
    mtp_ce_buf: DeviceBuffer,
    // MTP backward temporaries
    d_mtp_logits: DeviceBuffer,
    d_mtp_final: DeviceBuffer,
    d_mtp_block: DeviceBuffer,
    d_mtp_ehp: DeviceBuffer,
    d_mtp_en: DeviceBuffer,
    d_mtp_hn: DeviceBuffer,
    d_mtp_e: DeviceBuffer,
    d_mtp_res: DeviceBuffer,
    mtp_head_tmp: DeviceBuffer,

    // shared backward temporaries
    d_logits: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_v: DeviceBuffer,
    d_q_pass: DeviceBuffer,
    d_k_pass: DeviceBuffer,
    d_q_rot: DeviceBuffer,
    d_k_rot: DeviceBuffer,
    d_xn1: DeviceBuffer,
    d_qc: DeviceBuffer,
    d_qcn: DeviceBuffer,
    d_kvc: DeviceBuffer,
    d_kvcn: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_router_logits: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_expert_out: DeviceBuffer,
    inv: DeviceBuffer,

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,

    // ---- incremental KV-cache decode (single new token vs the growing cache) ----
    // MLA "materialised per-head" cache: per layer store the up-projected per-head
    // key `k_pass` [cap, H*qk_nope], the shared MQA rope key `k_rot` [cap, qk_rope],
    // and the per-head value `v` [cap, H*v_head]. `cap = t` (the configured ctx).
    kpass_cache: Vec<DeviceBuffer>,
    krot_cache: Vec<DeviceBuffer>,
    v_cache: Vec<DeviceBuffer>,
    // Next absolute position the incremental `step` will decode (cache fill level).
    dec_pos: Cell<u32>,
}

impl Glm {
    /// Trainable load, streaming: build directly off a mmap-backed
    /// [`WeightReader`](checkpoint::weightio::WeightReader), uploading one tensor
    /// at a time — peak host ≈ one tensor of f32, never the whole-model
    /// `checkpoint::load` + `by_role("")` host copy. AdamW moments are device
    /// zero-init (not read from the checkpoint), so streaming is byte-identical
    /// to the former eager path.
    pub fn load(path: &str, b: u32, t: u32) -> Glm {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let cfg = GlmConfig::from_json(&reader.config());
        Glm::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, &reader, true)
    }

    /// Inference load, streaming — see [`Glm::from_reader_inference`].
    pub fn load_inference(path: &str, b: u32, t: u32) -> Glm {
        let reader = checkpoint::weightio::WeightReader::open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        Glm::from_reader_inference(&reader, b, t)
    }

    /// Streaming inference load: build from a mmap-backed [`WeightReader`],
    /// uploading one tensor at a time (peak host ≈ one tensor of f32) — never the
    /// `checkpoint::load` + `by_role("")` whole-model host copy. Numerically
    /// identical to [`Glm::load_inference`]; used by the resident serve path.
    pub fn from_reader_inference(reader: &checkpoint::weightio::WeightReader, b: u32, t: u32) -> Glm {
        let cfg = GlmConfig::from_json(&reader.config());
        Glm::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, reader, false)
    }

    pub fn new(cfg: GlmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Glm {
        Glm::new_impl(cfg, b, t, init, true)
    }

    /// Build in training mode on an existing device handle — see `Gpt::new_on`.
    pub fn new_on(gpu: Gpu, cfg: GlmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Glm {
        Glm::new_impl_on(gpu, cfg, b, t, init, true)
    }

    fn new_impl(cfg: GlmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, train: bool) -> Glm {
        Glm::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, train)
    }

    fn new_impl_on(gpu: Gpu, cfg: GlmConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> Glm {
        // Roles: inference => all Frozen; training => all Trainable EXCEPT the
        // router selection bias (`e_score_correction_bias`), which is never
        // updated by backprop (matches the reference — a load-balance heuristic
        // would drive it), so it stays Frozen and out of the optimiser.
        let roles: Vec<_> = cfg
            .param_list()
            .into_iter()
            .map(|(n, c)| {
                // The router selection bias and the whole DSA indexer are detached
                // from the LM loss (never backprop'd) — keep them Frozen so they
                // stay out of the optimiser and the gradient-checked set.
                let role = if !train || n.ends_with("moe.router.bias") || n.contains(".idx.") {
                    Role::Frozen
                } else {
                    Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let v = cfg.vocab as u64;
        let e = cfg.n_routed_experts as u64;
        let ql = cfg.q_lora_rank as u64;
        let kvl = cfg.kv_lora_rank as u64;
        let nope = cfg.nope_dim() as u64;
        let qrope = cfg.q_rope_dim() as u64;
        let rope1 = cfg.qk_rope_head_dim as u64;
        let vd = cfg.v_dim() as u64;
        let moe_ff = cfg.moe_intermediate_size as u64;
        let dense_ff = cfg.intermediate_size as u64;
        let shared_ff = cfg.shared_ff() as u64;
        let bht2 = (b * cfg.n_heads * t * t) as u64;
        let idx_dim = cfg.idx_dim() as u64;
        let idh = cfg.index_head_dim as u64;
        let nih = cfg.index_n_heads as u64;
        let btt = (b * t * t) as u64;
        let mtp = cfg.mtp;
        let st = |x: u64| gpu.storage(x);
        // MTP buffers are allocated at full size only when MTP is enabled.
        let msz = |x: u64| gpu.storage(if mtp { x.max(1) } else { 1 });

        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4);

        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers {
            let mlp = if cfg.is_dense_layer(l) {
                Mlp::Dense { gate_pre: st(n * dense_ff), up: st(n * dense_ff), h: st(n * dense_ff) }
            } else {
                Mlp::Moe {
                    router_logits: st(n * e),
                    gate: st(n * e),
                    probs: st(n * e),
                    gate_pre: (0..e).map(|_| st(n * moe_ff)).collect(),
                    up: (0..e).map(|_| st(n * moe_ff)).collect(),
                    h: (0..e).map(|_| st(n * moe_ff)).collect(),
                    expert_out: (0..e).map(|_| st(n * d)).collect(),
                    sh_gate: st(n * shared_ff),
                    sh_up: st(n * shared_ff),
                    sh_h: st(n * shared_ff),
                }
            };
            layers.push(LayerBufs {
                xn1: st(n * d),
                q_c: st(n * ql),
                q_c_n: st(n * ql),
                q_pass: st(n * nope),
                q_rot: st(n * qrope),
                kv_c: st(n * kvl),
                kv_c_n: st(n * kvl),
                k_rot: st(n * rope1),
                k_pass: st(n * nope),
                v: st(n * vd),
                probs: st(bht2),
                ctx: st(n * vd),
                xmid: st(n * d),
                xn2: st(n * d),
                mlp,
            });
        }
        // widest per-expert feed-forward width (dense vs moe) for the shared d_h.
        let ff_max = dense_ff.max(moe_ff).max(shared_ff);

        // Incremental-decode KV cache: one set of buffers per layer, sized to the
        // configured context `t` (= cap). Stores the materialised per-head k_pass,
        // the shared MQA k_rot, and per-head v (see field docs).
        let mut kpass_cache = Vec::new();
        let mut krot_cache = Vec::new();
        let mut v_cache = Vec::new();
        for _ in 0..cfg.n_layers {
            kpass_cache.push(st(t as u64 * nope));
            krot_cache.push(st(t as u64 * rope1));
            v_cache.push(st(t as u64 * vd));
        }

        let mut m = Glm {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            opt,
            tokens,
            targets,
            ce_grad_uni,
            res,
            dres,
            layers,
            xn_final: st(n * d),
            logits: st(n * v),
            ce_buf: st(n),
            scores: st(bht2),
            proj: st(n * d),
            moe_acc: st(n * d),
            sh_out: st(n * d),
            mlp_out: st(n * d),
            q_idx: st(n * idx_dim),
            k_idx_pre: st(n * idh),
            k_idx: st(n * idh),
            idx_weights: st(n * nih),
            index_scores: st(btt),
            idx_mask: st(btt),
            // MTP buffers: full size when enabled, 1-element placeholders otherwise.
            mtp_input: gpu.storage(if mtp { n } else { 1 }),
            mtp_target: gpu.storage(if mtp { n } else { 1 }),
            mtp_e: msz(n * d),
            mtp_en: msz(n * d),
            mtp_hn: msz(n * d),
            mtp_ehp: msz(n * d),
            mtp_ehp2: msz(n * d),
            mtp_xn: msz(n * d),
            mtp_gate_pre: msz(n * dense_ff),
            mtp_up: msz(n * dense_ff),
            mtp_h: msz(n * dense_ff),
            mtp_mlp_out: msz(n * d),
            mtp_block_out: msz(n * d),
            mtp_final: msz(n * d),
            mtp_logits: msz(n * v),
            mtp_ce_buf: msz(n),
            d_mtp_logits: msz(n * v),
            d_mtp_final: msz(n * d),
            d_mtp_block: msz(n * d),
            d_mtp_ehp: msz(n * d),
            d_mtp_en: msz(n * d),
            d_mtp_hn: msz(n * d),
            d_mtp_e: msz(n * d),
            d_mtp_res: msz(n * d),
            mtp_head_tmp: msz(v * d),
            d_logits: st(n * v),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_ctx: st(n * vd),
            d_scores: st(bht2),
            d_v: st(n * vd),
            d_q_pass: st(n * nope),
            d_k_pass: st(n * nope),
            d_q_rot: st(n * qrope),
            d_k_rot: st(n * rope1),
            d_xn1: st(n * d),
            d_qc: st(n * ql),
            d_qcn: st(n * ql),
            d_kvc: st(n * kvl),
            d_kvcn: st(n * kvl),
            d_h: st(n * ff_max),
            d_gate_pre: st(n * ff_max),
            d_up: st(n * ff_max),
            d_router_logits: st(n * e),
            d_gate: st(n * e),
            d_expert_out: st(n * d),
            inv: st(n),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            kpass_cache,
            krot_cache,
            v_cache,
            dec_pos: Cell::new(0),
            gpu,
        };
        m.fwd_steps = m.build_forward(m.b, m.t);
        m.bwd_steps = if train { m.build_backward() } else { Vec::new() };
        m
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, bytemuck::cast_slice(x));
        self.gpu.write(&self.targets, bytemuck::cast_slice(y));
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
        if self.cfg.mtp {
            // MTP predicts token t+2 from hidden_t + embed(x[t+1]). Per sequence
            // (b seqs of `seqlen`): input = x shifted +1, target = x shifted +2.
            let bb = self.b as usize;
            let seqlen = x.len() / bb.max(1);
            let mut inp = vec![0u32; x.len()];
            let mut tgt = vec![IGNORE; x.len()];
            for s in 0..bb {
                for t in 0..seqlen {
                    let i = s * seqlen + t;
                    inp[i] = if t + 1 < seqlen { x[s * seqlen + t + 1] } else { 0 };
                    tgt[i] = if t + 2 < seqlen { x[s * seqlen + t + 2] } else { IGNORE };
                }
            }
            self.gpu.write(&self.mtp_input, bytemuck::cast_slice(&inp));
            self.gpu.write(&self.mtp_target, bytemuck::cast_slice(&tgt));
        }
    }

    // ---- dispatch helpers (mirror the moe/qwen style) ----

    fn mm(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        // Size-adaptive GEMM: software-pipelined `matmul_reg3` (128x128 tile,
        // ~4 TFLOP/s on a P40) once both output dims fill a tile, else the naive
        // per-output `matmul`. Same math (parity gated by gradcheck::check_glm),
        // so this only changes speed. `BRAIN_GLM_NAIVE_MM=1` forces naive.
        let naive = std::env::var("BRAIN_GLM_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
        let (mk, mt) = if naive || m < 128 || nout < 128 {
            (MATMUL, m * nout)
        } else {
            (MATMUL_REG3, (m as usize).div_ceil(128) as u32 * (nout as usize).div_ceil(128) as u32 * 256)
        };
        s.push(self.gpu.step(mk, &[x, self.w(wname), out], &[m, k, nout], mt));
    }

    /// Backward for `y = x·Wᵀ`: weight grad (if trainable) + input grad into `dx`
    /// (`acc`=0 initialise, 1 accumulate).
    #[allow(clippy::too_many_arguments)]
    fn mm_bwd(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        if self.trainable(wname) {
            s.push(self.gpu.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
        }
        s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
    }

    fn norm_fwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, dim: u32, rows: u32) {
        s.push(self.gpu.step(RMSNORM, &[x, self.w(wname), out], &[dim, rows], rows));
    }

    /// RMSNorm backward: gain grad (if trainable) via `rms_inv`+`rmsnorm_dw`, then
    /// input grad via `rmsnorm_dx` into `dx`.
    fn norm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        if self.trainable(wname) {
            s.push(self.gpu.step(RMS_INV, &[x, &self.inv], &[dim, rows], rows));
            s.push(self.gpu.step(RMSNORM_DW, &[dy, x, &self.inv, self.g(wname)], &[dim, rows], dim));
        }
        s.push(self.gpu.step(RMSNORM_DX, &[x, self.w(wname), dy, dx], &[dim, rows], rows));
    }

    /// DSA indexer forward for one `Full` layer: project q (from the q residual)
    /// and the shared single-head key, LayerNorm + sub-slice RoPE them, score with
    /// `relu(q·k)` weighted per head, and select the top-`index_topk` causal keys
    /// into `idx_mask`. Detached from the LM loss (Frozen params, no backward).
    fn indexer_fwd(&self, s: &mut Vec<Step>, lb: &LayerBufs, l: usize, b_use: u32, t_use: u32) {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ql = c.q_lora_rank;
        let idx = c.idx_dim();
        let idh = c.index_head_dim;
        let nih = c.index_n_heads;
        let rope = c.index_rope_dim();
        let p = |name: &str| format!("blocks.{l}.{name}");
        // q_idx = q_resid·Wq_bᵀ ; k_idx = LayerNorm(x·Wkᵀ) ; weights = x·Wprojᵀ
        self.mm(s, &lb.q_c_n, &p("idx.wq_b.weight"), &self.q_idx, n, ql, idx);
        self.mm(s, &lb.xn1, &p("idx.wk.weight"), &self.k_idx_pre, n, d, idh);
        s.push(self.gpu.step(LAYERNORM, &[&self.k_idx_pre, self.w(&p("idx.k_norm.weight")), self.w(&p("idx.k_norm.bias")), &self.k_idx], &[idh, n, f(1e-5)], n));
        s.push(self.gpu.step(ROPE_SUB, &[&self.q_idx], &[n, nih, idh, rope, idx, t_use], n * nih * (rope / 2)));
        s.push(self.gpu.step(ROPE_SUB, &[&self.k_idx], &[n, 1, idh, rope, idh, t_use], n * (rope / 2)));
        self.mm(s, &lb.xn1, &p("idx.weights_proj.weight"), &self.idx_weights, n, d, nih);
        s.push(self.gpu.step(MLA_INDEX_SCORES, &[&self.q_idx, &self.k_idx, &self.idx_weights, &self.index_scores], &[b_use, nih, t_use, idh], b_use * t_use * t_use));
        s.push(self.gpu.step(TOPK_MASK, &[&self.index_scores, &self.idx_mask], &[b_use, t_use, c.index_topk], b_use * t_use));
    }

    /// Add the current DSA sparse mask into the MLA scores (shared across heads).
    fn add_index_mask(&self, s: &mut Vec<Step>, b_use: u32, nh: u32, t_use: u32) {
        s.push(self.gpu.step(ADD_INDEX_MASK, &[&self.idx_mask, &self.scores], &[b_use, nh, t_use], b_use * nh * t_use * t_use));
    }

    fn build_forward(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let v = c.vocab;
        let e = c.n_routed_experts;
        let ql = c.q_lora_rank;
        let kvl = c.kv_lora_rank;
        let nope = c.nope_dim();
        let qrope = c.q_rope_dim();
        let rope1 = c.qk_rope_head_dim;
        let vd = c.v_dim();
        let nh = c.n_heads;
        let vhd = c.v_head_dim;
        let moe_ff = c.moe_intermediate_size;
        let dense_ff = c.intermediate_size;
        let shared_ff = c.shared_ff();
        let half_rope = rope1 / 2;
        let mut s: Vec<Step> = Vec::new();

        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- MLA attention ----
            self.norm_fwd(&mut s, &self.res[l], &p("input_ln.weight"), &lb.xn1, d, n);
            // Q: low-rank down -> norm -> split up (nope / rope), rope the rope slice
            self.mm(&mut s, &lb.xn1, &p("attn.q_a.weight"), &lb.q_c, n, d, ql);
            self.norm_fwd(&mut s, &lb.q_c, &p("attn.q_a_norm.weight"), &lb.q_c_n, ql, n);
            self.mm(&mut s, &lb.q_c_n, &p("attn.q_b_nope.weight"), &lb.q_pass, n, ql, nope);
            self.mm(&mut s, &lb.q_c_n, &p("attn.q_b_rope.weight"), &lb.q_rot, n, ql, qrope);
            s.push(self.gpu.step(ROPE, &[&lb.q_rot], &[n, nh, rope1, qrope, 0, t_use], n * nh * half_rope));
            // KV: compressed latent (+ shared rope key), norm, up-project to k_pass / v
            self.mm(&mut s, &lb.xn1, &p("attn.kv_a_c.weight"), &lb.kv_c, n, d, kvl);
            self.mm(&mut s, &lb.xn1, &p("attn.kv_a_rope.weight"), &lb.k_rot, n, d, rope1);
            s.push(self.gpu.step(ROPE, &[&lb.k_rot], &[n, 1, rope1, rope1, 0, t_use], n * half_rope));
            self.norm_fwd(&mut s, &lb.kv_c, &p("attn.kv_a_norm.weight"), &lb.kv_c_n, kvl, n);
            self.mm(&mut s, &lb.kv_c_n, &p("attn.kv_b_nope.weight"), &lb.k_pass, n, kvl, nope);
            self.mm(&mut s, &lb.kv_c_n, &p("attn.kv_b_v.weight"), &lb.v, n, kvl, vd);
            // scores -> softmax -> ctx (v-side reuses the standard MHA apply)
            s.push(self.gpu.step(MLA_SCORES, &[&lb.q_pass, &lb.q_rot, &lb.k_pass, &lb.k_rot, &self.scores], &[b_use, nh, t_use, c.qk_nope_head_dim, rope1], b_use * nh * t_use * t_use));
            // DSA sparse indexer (IndexShare): `Full` layers compute a fresh top-k
            // mask; `Full`+`Shared` layers add it into the scores before softmax.
            // The indexer is detached (Frozen params) — no backward path.
            match c.idx_mode(l as u32) {
                IdxMode::Full => {
                    self.indexer_fwd(&mut s, lb, l, b_use, t_use);
                    self.add_index_mask(&mut s, b_use, nh, t_use);
                }
                IdxMode::Shared => self.add_index_mask(&mut s, b_use, nh, t_use),
                IdxMode::None => {}
            }
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&self.scores, &lb.probs], &[b_use, nh, t_use], b_use * nh * t_use));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.v, &lb.ctx], &[b_use, nh, t_use, vhd, vd, 0, vd], b_use * nh * t_use * vhd));
            self.mm(&mut s, &lb.ctx, &p("attn.o.weight"), &self.proj, n, vd, d);
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));

            // ---- MLP ----
            self.norm_fwd(&mut s, &lb.xmid, &p("post_ln.weight"), &lb.xn2, d, n);
            match &lb.mlp {
                Mlp::Dense { gate_pre, up, h } => {
                    self.mm(&mut s, &lb.xn2, &p("mlp.gate.weight"), gate_pre, n, d, dense_ff);
                    self.mm(&mut s, &lb.xn2, &p("mlp.up.weight"), up, n, d, dense_ff);
                    s.push(self.gpu.step(SILU_MUL, &[gate_pre, up, h], &[n * dense_ff], n * dense_ff));
                    self.mm(&mut s, h, &p("mlp.down.weight"), &self.mlp_out, n, dense_ff, d);
                    s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
                }
                Mlp::Moe { router_logits, gate, probs, gate_pre, up, h, expert_out, sh_gate, sh_up, sh_h } => {
                    // router (sigmoid noaux_tc): logits -> gate weights [n,E]
                    self.mm(&mut s, &lb.xn2, &p("moe.router.weight"), router_logits, n, d, e);
                    s.push(self.gpu.step(
                        ROUTER_SIG,
                        &[router_logits, self.w(&p("moe.router.bias")), gate, probs],
                        &[n, e, c.num_experts_per_tok, c.n_group, c.topk_group, c.norm_topk_prob as u32, f(c.routed_scaling_factor)],
                        n,
                    ));
                    // routed experts (dense eval, gate-weighted accumulate)
                    for ei in 0..e as usize {
                        let ep = |nm: &str| format!("blocks.{l}.moe.experts.{ei}.{nm}");
                        self.mm(&mut s, &lb.xn2, &ep("gate.weight"), &gate_pre[ei], n, d, moe_ff);
                        self.mm(&mut s, &lb.xn2, &ep("up.weight"), &up[ei], n, d, moe_ff);
                        s.push(self.gpu.step(SILU_MUL, &[&gate_pre[ei], &up[ei], &h[ei]], &[n * moe_ff], n * moe_ff));
                        self.mm(&mut s, &h[ei], &ep("down.weight"), &expert_out[ei], n, moe_ff, d);
                        let acc = if ei == 0 { 0 } else { 1 };
                        s.push(self.gpu.step(SCALE_ADD, &[gate, &expert_out[ei], &self.moe_acc], &[n, d, e, ei as u32, acc], n * d));
                    }
                    // shared expert on the same input
                    self.mm(&mut s, &lb.xn2, &p("moe.shared.gate.weight"), sh_gate, n, d, shared_ff);
                    self.mm(&mut s, &lb.xn2, &p("moe.shared.up.weight"), sh_up, n, d, shared_ff);
                    s.push(self.gpu.step(SILU_MUL, &[sh_gate, sh_up, sh_h], &[n * shared_ff], n * shared_ff));
                    self.mm(&mut s, sh_h, &p("moe.shared.down.weight"), &self.sh_out, n, shared_ff, d);
                    // moe_out = routed + shared ; x += moe_out
                    s.push(self.gpu.step(ADD2, &[&self.moe_acc, &self.sh_out, &self.mlp_out], &[n * d], n * d));
                    s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
                }
            }
        }

        // final norm + untied lm_head + masked CE
        let last = c.n_layers as usize;
        self.norm_fwd(&mut s, &self.res[last], "norm.weight", &self.xn_final, d, n);
        self.mm(&mut s, &self.xn_final, c.head_weight(), &self.logits, n, d, v);
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));

        // ---- MTP head (predict t+2) ----
        if c.mtp {
            let head = c.head_weight();
            let ff = c.intermediate_size;
            s.push(self.gpu.step(EMBED, &[&self.mtp_input, self.w("tok.weight"), &self.mtp_e], &[d, n], n * d));
            self.norm_fwd(&mut s, &self.mtp_e, "mtp.enorm.weight", &self.mtp_en, d, n);
            self.norm_fwd(&mut s, &self.res[last], "mtp.hnorm.weight", &self.mtp_hn, d, n);
            // eh_proj: hidden = We·enorm(e) + Wh·hnorm(h)
            self.mm(&mut s, &self.mtp_en, "mtp.eh_proj_e.weight", &self.mtp_ehp, n, d, d);
            self.mm(&mut s, &self.mtp_hn, "mtp.eh_proj_h.weight", &self.mtp_ehp2, n, d, d);
            s.push(self.gpu.step(ADD_INPLACE, &[&self.mtp_ehp, &self.mtp_ehp2], &[n * d], n * d));
            // position-wise SwiGLU block with a residual (no self-attention)
            self.norm_fwd(&mut s, &self.mtp_ehp, "mtp.block_ln.weight", &self.mtp_xn, d, n);
            self.mm(&mut s, &self.mtp_xn, "mtp.mlp.gate.weight", &self.mtp_gate_pre, n, d, ff);
            self.mm(&mut s, &self.mtp_xn, "mtp.mlp.up.weight", &self.mtp_up, n, d, ff);
            s.push(self.gpu.step(SILU_MUL, &[&self.mtp_gate_pre, &self.mtp_up, &self.mtp_h], &[n * ff], n * ff));
            self.mm(&mut s, &self.mtp_h, "mtp.mlp.down.weight", &self.mtp_mlp_out, n, ff, d);
            s.push(self.gpu.step(ADD2, &[&self.mtp_ehp, &self.mtp_mlp_out, &self.mtp_block_out], &[n * d], n * d));
            self.norm_fwd(&mut s, &self.mtp_block_out, "mtp.norm.weight", &self.mtp_final, d, n);
            self.mm(&mut s, &self.mtp_final, head, &self.mtp_logits, n, d, v); // shared lm_head
            s.push(self.gpu.step(CE_VALUE, &[&self.mtp_logits, &self.mtp_target, &self.mtp_ce_buf], &[n, v, IGNORE], n));
        }
        s
    }

    fn build_backward(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let b_use = self.b;
        let t_use = self.t;
        let d = c.d_model;
        let v = c.vocab;
        let e = c.n_routed_experts;
        let ql = c.q_lora_rank;
        let kvl = c.kv_lora_rank;
        let nope = c.nope_dim();
        let qrope = c.q_rope_dim();
        let rope1 = c.qk_rope_head_dim;
        let vd = c.v_dim();
        let nh = c.n_heads;
        let vhd = c.v_head_dim;
        let nope_hd = c.qk_nope_head_dim;
        let moe_ff = c.moe_intermediate_size;
        let dense_ff = c.intermediate_size;
        let shared_ff = c.shared_ff();
        let half_rope = rope1 / 2;
        let head = c.head_weight();
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final norm ----
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * v));
        self.mm_bwd(&mut s, &self.d_logits, &self.xn_final, head, &self.d_xn, n, d, v, 0);
        let last = c.n_layers as usize;
        self.norm_bwd(&mut s, &self.res[last], "norm.weight", &self.d_xn, &self.dres[last], d, n);

        // ---- MTP head backward (adds into dres[last], the shared head grad, and
        // the embedding grad). Runs before the layer loop reads dres[last]. ----
        if c.mtp {
            let head = c.head_weight();
            let ff = c.intermediate_size;
            s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.mtp_logits, &self.mtp_target, &self.d_mtp_logits], n * v));
            // shared lm_head: accumulate its weight grad, then input grad -> d_mtp_final
            if self.trainable(head) {
                s.push(self.gpu.step(MATMUL_DW, &[&self.d_mtp_logits, &self.mtp_final, &self.mtp_head_tmp], &[n, d, v], v * d));
                s.push(self.gpu.step(ADD_INPLACE, &[self.g(head), &self.mtp_head_tmp], &[v * d], v * d));
            }
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_mtp_logits, self.w(head), &self.d_mtp_final], &[n, d, v, 0], n * d));
            self.norm_bwd(&mut s, &self.mtp_block_out, "mtp.norm.weight", &self.d_mtp_final, &self.d_mtp_block, d, n);
            // SwiGLU MLP backward (d_xn reused as the grad wrt mtp_xn)
            self.mm_bwd(&mut s, &self.d_mtp_block, &self.mtp_h, "mtp.mlp.down.weight", &self.d_h, n, ff, d, 0);
            s.push(self.gpu.step(SILU_DA, &[&self.mtp_gate_pre, &self.mtp_up, &self.d_h, &self.d_gate_pre], &[n * ff], n * ff));
            s.push(self.gpu.step(SILU_DB, &[&self.mtp_gate_pre, &self.d_h, &self.d_up], &[n * ff], n * ff));
            self.mm_bwd(&mut s, &self.d_up, &self.mtp_xn, "mtp.mlp.up.weight", &self.d_xn, n, d, ff, 0);
            self.mm_bwd(&mut s, &self.d_gate_pre, &self.mtp_xn, "mtp.mlp.gate.weight", &self.d_xn, n, d, ff, 1);
            self.norm_bwd(&mut s, &self.mtp_ehp, "mtp.block_ln.weight", &self.d_xn, &self.d_mtp_ehp, d, n);
            s.push(self.gpu.step(ADD_INPLACE, &[&self.d_mtp_ehp, &self.d_mtp_block], &[n * d], n * d)); // + residual
            // eh_proj backward -> grads wrt enorm/hnorm outputs
            self.mm_bwd(&mut s, &self.d_mtp_ehp, &self.mtp_en, "mtp.eh_proj_e.weight", &self.d_mtp_en, n, d, d, 0);
            self.mm_bwd(&mut s, &self.d_mtp_ehp, &self.mtp_hn, "mtp.eh_proj_h.weight", &self.d_mtp_hn, n, d, d, 0);
            self.norm_bwd(&mut s, &self.mtp_e, "mtp.enorm.weight", &self.d_mtp_en, &self.d_mtp_e, d, n);
            self.norm_bwd(&mut s, &self.res[last], "mtp.hnorm.weight", &self.d_mtp_hn, &self.d_mtp_res, d, n);
            s.push(self.gpu.step(ADD_INPLACE, &[&self.dres[last], &self.d_mtp_res], &[n * d], n * d));
            if self.trainable("tok.weight") {
                s.push(self.gpu.step(EMB_BWD, &[&self.mtp_input, &self.d_mtp_e, self.g("tok.weight")], &[n, d, v], v * d));
            }
        }

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ===== MLP backward (output grad = dres[l+1]) -> dxmid =====
            match &lb.mlp {
                Mlp::Dense { gate_pre, up, h } => {
                    self.mm_bwd(&mut s, &self.dres[l + 1], h, &p("mlp.down.weight"), &self.d_h, n, dense_ff, d, 0);
                    s.push(self.gpu.step(SILU_DA, &[gate_pre, up, &self.d_h, &self.d_gate_pre], &[n * dense_ff], n * dense_ff));
                    s.push(self.gpu.step(SILU_DB, &[gate_pre, &self.d_h, &self.d_up], &[n * dense_ff], n * dense_ff));
                    self.mm_bwd(&mut s, &self.d_up, &lb.xn2, &p("mlp.up.weight"), &self.d_xn, n, d, dense_ff, 0);
                    self.mm_bwd(&mut s, &self.d_gate_pre, &lb.xn2, &p("mlp.gate.weight"), &self.d_xn, n, d, dense_ff, 1);
                }
                Mlp::Moe { router_logits, gate, probs: _, gate_pre, up, h, expert_out, sh_gate, sh_up, sh_h } => {
                    // shared expert backward (writes d_xn first, acc=0)
                    self.mm_bwd(&mut s, &self.dres[l + 1], sh_h, &p("moe.shared.down.weight"), &self.d_h, n, shared_ff, d, 0);
                    s.push(self.gpu.step(SILU_DA, &[sh_gate, sh_up, &self.d_h, &self.d_gate_pre], &[n * shared_ff], n * shared_ff));
                    s.push(self.gpu.step(SILU_DB, &[sh_gate, &self.d_h, &self.d_up], &[n * shared_ff], n * shared_ff));
                    self.mm_bwd(&mut s, &self.d_up, &lb.xn2, &p("moe.shared.up.weight"), &self.d_xn, n, d, shared_ff, 0);
                    self.mm_bwd(&mut s, &self.d_gate_pre, &lb.xn2, &p("moe.shared.gate.weight"), &self.d_xn, n, d, shared_ff, 1);
                    // router backward: d_gate[n,E] from each expert, then through sigmoid
                    for ei in 0..e as usize {
                        s.push(self.gpu.step(SCALE_ADD_DGATE, &[&expert_out[ei], &self.dres[l + 1], &self.d_gate], &[n, d, e, ei as u32], n));
                    }
                    s.push(self.gpu.step(
                        ROUTER_SIG_BWD,
                        &[router_logits, gate, &self.d_gate, &self.d_router_logits],
                        &[n, e, c.num_experts_per_tok, c.norm_topk_prob as u32, f(c.routed_scaling_factor)],
                        n,
                    ));
                    self.mm_bwd(&mut s, &self.d_router_logits, &lb.xn2, &p("moe.router.weight"), &self.d_xn, n, d, e, 1);
                    // per-expert SwiGLU backward, accumulate into d_xn
                    for ei in 0..e as usize {
                        let ep = |nm: &str| format!("blocks.{l}.moe.experts.{ei}.{nm}");
                        s.push(self.gpu.step(SCALE_ADD_DEXP, &[gate, &self.dres[l + 1], &self.d_expert_out], &[n, d, e, ei as u32], n * d));
                        self.mm_bwd(&mut s, &self.d_expert_out, &h[ei], &ep("down.weight"), &self.d_h, n, moe_ff, d, 0);
                        s.push(self.gpu.step(SILU_DA, &[&gate_pre[ei], &up[ei], &self.d_h, &self.d_gate_pre], &[n * moe_ff], n * moe_ff));
                        s.push(self.gpu.step(SILU_DB, &[&gate_pre[ei], &self.d_h, &self.d_up], &[n * moe_ff], n * moe_ff));
                        self.mm_bwd(&mut s, &self.d_up, &lb.xn2, &ep("up.weight"), &self.d_xn, n, d, moe_ff, 1);
                        self.mm_bwd(&mut s, &self.d_gate_pre, &lb.xn2, &ep("gate.weight"), &self.d_xn, n, d, moe_ff, 1);
                    }
                }
            }
            // post_ln backward -> d_tmp ; dxmid = dres[l+1] + d_tmp
            self.norm_bwd(&mut s, &lb.xmid, &p("post_ln.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // ===== MLA attention backward (output grad = dxmid) =====
            self.mm_bwd(&mut s, &self.dxmid, &lb.ctx, &p("attn.o.weight"), &self.d_ctx, n, vd, d, 0);
            // v-side: softmax+apply backward (reuse standard MHA kernels, v geometry)
            s.push(self.gpu.step(ATTN_BWD_DSCORES, &[&self.d_ctx, &lb.v, &lb.probs, &self.d_scores], &[b_use, nh, t_use, vhd, vd, 0, vd], b_use * nh * t_use));
            s.push(self.gpu.step(ATTN_BWD_DV, &[&lb.probs, &self.d_ctx, &self.d_v], &[b_use, nh, t_use, vhd, vd, 0, vd], b_use * nh * t_use * vhd));
            // q/k grads from d_scores
            s.push(self.gpu.step(MLA_BWD_DQ_PASS, &[&self.d_scores, &lb.k_pass, &self.d_q_pass], &[b_use, nh, t_use, nope_hd, rope1], b_use * nh * t_use * nope_hd));
            s.push(self.gpu.step(MLA_BWD_DK_PASS, &[&self.d_scores, &lb.q_pass, &self.d_k_pass], &[b_use, nh, t_use, nope_hd, rope1], b_use * nh * t_use * nope_hd));
            s.push(self.gpu.step(MLA_BWD_DQ_ROPE, &[&self.d_scores, &lb.k_rot, &self.d_q_rot], &[b_use, nh, t_use, nope_hd, rope1], b_use * nh * t_use * rope1));
            s.push(self.gpu.step(MLA_BWD_DK_ROPE, &[&self.d_scores, &lb.q_rot, &self.d_k_rot], &[b_use, nh, t_use, nope_hd, rope1], b_use * t_use * rope1));
            // RoPE backward on the rope grads (in place)
            s.push(self.gpu.step(ROPE_BWD, &[&self.d_q_rot], &[n, nh, rope1, qrope, 0, t_use], n * nh * half_rope));
            s.push(self.gpu.step(ROPE_BWD, &[&self.d_k_rot], &[n, 1, rope1, rope1, 0, t_use], n * half_rope));
            // KV projections backward -> d_xn1 (grad wrt xn1)
            self.mm_bwd(&mut s, &self.d_v, &lb.kv_c_n, &p("attn.kv_b_v.weight"), &self.d_kvcn, n, kvl, vd, 0);
            self.mm_bwd(&mut s, &self.d_k_pass, &lb.kv_c_n, &p("attn.kv_b_nope.weight"), &self.d_kvcn, n, kvl, nope, 1);
            self.norm_bwd(&mut s, &lb.kv_c, &p("attn.kv_a_norm.weight"), &self.d_kvcn, &self.d_kvc, kvl, n);
            self.mm_bwd(&mut s, &self.d_k_rot, &lb.xn1, &p("attn.kv_a_rope.weight"), &self.d_xn1, n, d, rope1, 0);
            self.mm_bwd(&mut s, &self.d_kvc, &lb.xn1, &p("attn.kv_a_c.weight"), &self.d_xn1, n, d, kvl, 1);
            // Q projections backward -> d_xn1
            self.mm_bwd(&mut s, &self.d_q_pass, &lb.q_c_n, &p("attn.q_b_nope.weight"), &self.d_qcn, n, ql, nope, 0);
            self.mm_bwd(&mut s, &self.d_q_rot, &lb.q_c_n, &p("attn.q_b_rope.weight"), &self.d_qcn, n, ql, qrope, 1);
            self.norm_bwd(&mut s, &lb.q_c, &p("attn.q_a_norm.weight"), &self.d_qcn, &self.d_qc, ql, n);
            self.mm_bwd(&mut s, &self.d_qc, &lb.xn1, &p("attn.q_a.weight"), &self.d_xn1, n, d, ql, 1);
            // input_ln backward -> d_tmp ; dres[l] = dxmid + d_tmp
            self.norm_bwd(&mut s, &self.res[l], &p("input_ln.weight"), &self.d_xn1, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embedding backward (untied: only the embedding path writes tok.weight)
        if self.trainable("tok.weight") {
            s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("tok.weight")], &[n, d, v], v * d));
        }
        s
    }

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        let n = (self.b * self.t) as usize;
        let mut total: f32 = self.gpu.read(&self.ce_buf, n).iter().sum();
        // MTP auxiliary loss, added with the same count divisor (so `backward`,
        // which reuses the same CE-grad uniform, differentiates exactly this).
        if self.cfg.mtp {
            total += self.gpu.read(&self.mtp_ce_buf, n).iter().sum::<f32>();
        }
        total / self.count.get()
    }

    pub fn backward(&self) {
        let n = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.95, 1e-8, clip, extra_scale);
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
    pub fn param_names(&self) -> Vec<String> {
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn ctx_len(&self) -> usize {
        self.t as usize
    }

    /// Per-position logits for one sequence (B must be 1, len <= t).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "glm decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.build_forward(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab) as usize)
    }

    /// Per-position **final-norm hidden** states for one sequence (`[len, d_model]`),
    /// the recompute twin of [`Self::step`]'s output — used to validate the KV
    /// decode. (B must be 1, len <= t.)
    pub fn hidden_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "glm decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.build_forward(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.xn_final, (t_use * self.cfg.d_model) as usize)
    }

    // ================= incremental KV-cache decode =================

    /// Reset the incremental KV cache to an empty sequence (next `step` is pos 0).
    pub fn reset_cache(&self) {
        self.dec_pos.set(0);
    }

    /// The absolute position the next [`Self::step`] will decode.
    pub fn cache_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// **Incremental KV-cache decode** of one new token at the current cache
    /// position, returning the final-norm hidden state (`[d_model]`). The `O(T)`
    /// twin of [`Self::logits_all`]/[`Self::hidden_all`]'s `O(T²)` recompute: the
    /// same GLM MLA + MoE block math, but the new token's k_pass / shared k_rot / v
    /// are projected once, appended to the persistent per-layer cache, and attended
    /// by a single query over cached positions `0..=pos`. Requires `b == 1`.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        let pos = self.dec_pos.get();
        let hidden = self.decode_at(token_id, pos);
        self.dec_pos.set(pos + 1);
        hidden
    }

    /// Record + run the incremental decode tape for one token at absolute `pos`.
    /// Runs entirely in the WGSL op set (GPU or wgsl-cpu), reusing every GLM
    /// forward kernel at a single-query (n=1) shape plus the MLA-decode score
    /// kernel + interleaved rope-at-pos (both inline in this crate) and the
    /// committed `kv_append` / `decode_softmax` / `attn_decode_apply` kernels.
    fn decode_at(&self, token_id: u32, pos: u32) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let e = c.n_routed_experts;
        let ql = c.q_lora_rank;
        let kvl = c.kv_lora_rank;
        let nope = c.nope_dim();
        let qrope = c.q_rope_dim();
        let rope1 = c.qk_rope_head_dim;
        let nope_hd = c.qk_nope_head_dim;
        let vd = c.v_dim();
        let nh = c.n_heads;
        let vhd = c.v_head_dim;
        let moe_ff = c.moe_intermediate_size;
        let dense_ff = c.intermediate_size;
        let shared_ff = c.shared_ff();
        let half_rope = rope1 / 2;
        let cap = self.t;
        let t = pos + 1; // cached length after appending this token
        assert!(self.b == 1, "KV decode requires b == 1");
        assert!(pos < self.t, "decode pos {pos} exceeds ctx {}", self.t);

        // embed the single token into res[0] (row 0).
        self.gpu.write(&self.tokens, &[token_id]);
        let mut s: Vec<Step> = Vec::new();
        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, 1], d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- MLA attention (single query) ----
            self.norm_fwd(&mut s, &self.res[l], &p("input_ln.weight"), &lb.xn1, d, 1);
            // Q: low-rank down -> norm -> up split (nope / rope), rope the rope slice at pos
            self.mm(&mut s, &lb.xn1, &p("attn.q_a.weight"), &lb.q_c, 1, d, ql);
            self.norm_fwd(&mut s, &lb.q_c, &p("attn.q_a_norm.weight"), &lb.q_c_n, ql, 1);
            self.mm(&mut s, &lb.q_c_n, &p("attn.q_b_nope.weight"), &lb.q_pass, 1, ql, nope);
            self.mm(&mut s, &lb.q_c_n, &p("attn.q_b_rope.weight"), &lb.q_rot, 1, ql, qrope);
            s.push(self.gpu.step(ROPE_TRAIN_AT, &[&lb.q_rot], &[1, nh, rope1, qrope, 0, pos], nh * half_rope));
            // KV: compressed latent (+ shared rope key), norm, up-project to k_pass / v
            self.mm(&mut s, &lb.xn1, &p("attn.kv_a_c.weight"), &lb.kv_c, 1, d, kvl);
            self.mm(&mut s, &lb.xn1, &p("attn.kv_a_rope.weight"), &lb.k_rot, 1, d, rope1);
            s.push(self.gpu.step(ROPE_TRAIN_AT, &[&lb.k_rot], &[1, 1, rope1, rope1, 0, pos], half_rope));
            self.norm_fwd(&mut s, &lb.kv_c, &p("attn.kv_a_norm.weight"), &lb.kv_c_n, kvl, 1);
            self.mm(&mut s, &lb.kv_c_n, &p("attn.kv_b_nope.weight"), &lb.k_pass, 1, kvl, nope);
            self.mm(&mut s, &lb.kv_c_n, &p("attn.kv_b_v.weight"), &lb.v, 1, kvl, vd);
            // append this token's materialised k_pass / shared k_rot / v to the cache
            s.push(self.gpu.step(KV_APPEND, &[&lb.k_pass, &self.kpass_cache[l]], &[nope, pos], nope));
            s.push(self.gpu.step(KV_APPEND, &[&lb.k_rot, &self.krot_cache[l]], &[rope1, pos], rope1));
            s.push(self.gpu.step(KV_APPEND, &[&lb.v, &self.v_cache[l]], &[vd, pos], vd));
            // single-query MLA scores over cached keys 0..t, softmax, apply (v side)
            s.push(self.gpu.step(
                MLA_DECODE_SCORES,
                &[&lb.q_pass, &lb.q_rot, &self.kpass_cache[l], &self.krot_cache[l], &self.scores],
                &[nh, nope_hd, rope1, t, cap],
                nh * t,
            ));
            s.push(self.gpu.step(DECODE_SOFTMAX, &[&self.scores, &lb.probs], &[nh, t, cap], nh));
            s.push(self.gpu.step(ATTN_DECODE_APPLY, &[&lb.probs, &self.v_cache[l], &lb.ctx], &[nh, 1, vhd, t, cap, vd], nh * vhd));
            self.mm(&mut s, &lb.ctx, &p("attn.o.weight"), &self.proj, 1, vd, d);
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[d], d));

            // ---- MLP: dense SwiGLU or MoE (single row) ----
            self.norm_fwd(&mut s, &lb.xmid, &p("post_ln.weight"), &lb.xn2, d, 1);
            match &lb.mlp {
                Mlp::Dense { gate_pre, up, h } => {
                    self.mm(&mut s, &lb.xn2, &p("mlp.gate.weight"), gate_pre, 1, d, dense_ff);
                    self.mm(&mut s, &lb.xn2, &p("mlp.up.weight"), up, 1, d, dense_ff);
                    s.push(self.gpu.step(SILU_MUL, &[gate_pre, up, h], &[dense_ff], dense_ff));
                    self.mm(&mut s, h, &p("mlp.down.weight"), &self.mlp_out, 1, dense_ff, d);
                    s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[d], d));
                }
                Mlp::Moe { router_logits, gate, probs, gate_pre, up, h, expert_out, sh_gate, sh_up, sh_h } => {
                    self.mm(&mut s, &lb.xn2, &p("moe.router.weight"), router_logits, 1, d, e);
                    s.push(self.gpu.step(
                        ROUTER_SIG,
                        &[router_logits, self.w(&p("moe.router.bias")), gate, probs],
                        &[1, e, c.num_experts_per_tok, c.n_group, c.topk_group, c.norm_topk_prob as u32, f(c.routed_scaling_factor)],
                        1,
                    ));
                    for ei in 0..e as usize {
                        let ep = |nm: &str| format!("blocks.{l}.moe.experts.{ei}.{nm}");
                        self.mm(&mut s, &lb.xn2, &ep("gate.weight"), &gate_pre[ei], 1, d, moe_ff);
                        self.mm(&mut s, &lb.xn2, &ep("up.weight"), &up[ei], 1, d, moe_ff);
                        s.push(self.gpu.step(SILU_MUL, &[&gate_pre[ei], &up[ei], &h[ei]], &[moe_ff], moe_ff));
                        self.mm(&mut s, &h[ei], &ep("down.weight"), &expert_out[ei], 1, moe_ff, d);
                        let acc = if ei == 0 { 0 } else { 1 };
                        s.push(self.gpu.step(SCALE_ADD, &[gate, &expert_out[ei], &self.moe_acc], &[1, d, e, ei as u32, acc], d));
                    }
                    self.mm(&mut s, &lb.xn2, &p("moe.shared.gate.weight"), sh_gate, 1, d, shared_ff);
                    self.mm(&mut s, &lb.xn2, &p("moe.shared.up.weight"), sh_up, 1, d, shared_ff);
                    s.push(self.gpu.step(SILU_MUL, &[sh_gate, sh_up, sh_h], &[shared_ff], shared_ff));
                    self.mm(&mut s, sh_h, &p("moe.shared.down.weight"), &self.sh_out, 1, shared_ff, d);
                    s.push(self.gpu.step(ADD2, &[&self.moe_acc, &self.sh_out, &self.mlp_out], &[d], d));
                    s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[d], d));
                }
            }
        }
        let last = c.n_layers as usize;
        self.norm_fwd(&mut s, &self.res[last], "norm.weight", &self.xn_final, d, 1);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.xn_final, d as usize)
    }

    /// Read the char tokenizer vocab (`itos`) embedded in a checkpoint at train
    /// time (char datasets), so inference needs no dataset. `None` for token-id
    /// (BPE) checkpoints that carry no char vocab.
    /// Char vocab (`itos`) from a config object, if present (else `None`).
    pub fn itos_from_config(cfg: &serde_json::Value) -> Option<Vec<char>> {
        let arr = cfg.get("itos")?.as_array()?;
        Some(arr.iter().filter_map(|x| x.as_str().and_then(|s| s.chars().next())).collect())
    }

    pub fn load_itos(path: &str) -> Option<Vec<char>> {
        // Header/config only — no tensor data is faulted in.
        Self::itos_from_config(&checkpoint::weightio::WeightReader::open(path).ok()?.config())
    }

    /// One DSA-indexer **distillation** step (host-side): train the `idx.*` params
    /// of every `Full` indexer layer to match the dense MLA attention distribution
    /// over keys (the DeepSeek-V3.2 recipe — the indexer is detached from the LM
    /// loss). Requires a dense forward (`index_topk >= block_size`, so the cached
    /// `probs` are the *unmasked* attention). Each idx tensor is updated with a
    /// per-tensor **RMS-normalized** step of size `lr` (RMSprop-style, so training
    /// is robust to the indexer's tiny near-zero init where raw gradients are
    /// minuscule). Pass `lr = 0` to just measure the loss. Returns the mean
    /// distillation cross-entropy.
    pub fn distill_step(&self, lr: f32) -> f32 {
        use crate::distill::{layer_distill, IdxDims, IdxWeights};
        let c = &self.cfg;
        let n = (self.b * self.t) as usize;
        let d = c.d_model as usize;
        let ql = c.q_lora_rank as usize;
        let bht2 = (self.b * c.n_heads * self.t * self.t) as usize;
        self.forward(); // populate probs / xn1 / q_c_n (dense when index_topk >= block)
        let dims = IdxDims {
            b: self.b as usize,
            t: self.t as usize,
            h: c.index_n_heads as usize,
            d: c.index_head_dim as usize,
            rope: c.index_rope_dim() as usize,
            ql,
            dm: d,
            mla_heads: c.n_heads as usize,
        };
        let mut total = 0.0f32;
        let mut count = 0usize;
        for l in 0..c.n_layers as usize {
            if c.idx_mode(l as u32) != IdxMode::Full {
                continue;
            }
            let lb = &self.layers[l];
            let xn1 = self.gpu.read(&lb.xn1, n * d);
            let q_c_n = self.gpu.read(&lb.q_c_n, n * ql);
            let probs = self.gpu.read(&lb.probs, bht2);
            let p = |s: &str| format!("blocks.{l}.idx.{s}");
            let w = IdxWeights {
                wq_b: self.read_weight(&p("wq_b.weight")),
                wk: self.read_weight(&p("wk.weight")),
                k_norm_w: self.read_weight(&p("k_norm.weight")),
                k_norm_b: self.read_weight(&p("k_norm.bias")),
                weights_proj: self.read_weight(&p("weights_proj.weight")),
            };
            let (loss, g) = layer_distill(&dims, &xn1, &q_c_n, &probs, &w);
            total += loss;
            count += 1;
            let upd = |name: String, cur: &[f32], gr: &[f32]| {
                // Per-tensor RMS-normalized step: effective size ~lr regardless of
                // the (tiny) gradient magnitude at the near-zero indexer init.
                let ms = gr.iter().map(|g| g * g).sum::<f32>() / gr.len().max(1) as f32;
                let scale = if lr == 0.0 { 0.0 } else { lr / (ms.sqrt() + 1e-8) };
                let nw: Vec<f32> = cur.iter().zip(gr).map(|(&x, &gg)| x - scale * gg).collect();
                self.write_weight(&name, &nw);
            };
            upd(p("wq_b.weight"), &w.wq_b, &g.wq_b);
            upd(p("wk.weight"), &w.wk, &g.wk);
            upd(p("k_norm.weight"), &w.k_norm_w, &g.k_norm_w);
            upd(p("k_norm.bias"), &w.k_norm_b, &g.k_norm_b);
            upd(p("weights_proj.weight"), &w.weights_proj, &g.weights_proj);
        }
        if count == 0 {
            return 0.0;
        }
        total / (count as f32 * n as f32)
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
            let arr: Vec<serde_json::Value> = itos.iter().map(|ch| serde_json::Value::from(ch.to_string())).collect();
            config["itos"] = serde_json::Value::Array(arr);
        }
        checkpoint::save(path, config, &tensors);
    }
}

// ---- architecture-agnostic Model seam ----

impl model::ModelConfig for GlmConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        GlmConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        GlmConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        GlmConfig::from_json(v)
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
        self
    }
}

impl model::Model for Glm {
    type Config = GlmConfig;

    fn new(cfg: GlmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Glm::new(cfg, b, t, init)
    }
    fn init_weights(cfg: &GlmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &GlmConfig {
        &self.cfg
    }
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Glm::set_batch(self, tokens, targets),
            _ => panic!("glm::Glm only supports Batch::Lm"),
        }
    }
    fn forward(&self) -> f32 {
        Glm::forward(self)
    }
    fn backward(&self) {
        Glm::backward(self)
    }
    fn zero_grads(&self) {
        Glm::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Glm::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Glm::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        Glm::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Glm::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Glm::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Glm::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Glm::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        Glm::save(self, path)
    }
    fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        Glm::save_with_itos(self, path, itos)
    }
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod kv_step_tests {
    use super::*;
    use crate::config::GlmConfig;

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The streaming `Glm::load_inference` (mmap `WeightReader`, one tensor
    /// uploaded at a time) yields byte-identical device weights to the eager
    /// whole-model-host-map path (`Glm::new` over `by_role("")`), and both match
    /// the source init exactly. GPU-gated (testgpu / MOE_SKIP_GPU).
    #[test]
    fn streaming_load_matches_eager() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = GlmConfig::tiny();
        let cfg_json = cfg.to_json();
        let init = crate::init::init_weights(&cfg, 5);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            init.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64], v.clone())).collect();
        let path = std::env::temp_dir().join(format!("glm-stream-parity-{}.st", std::process::id()));
        let p = path.to_str().unwrap();
        checkpoint::st::save_safetensors(p, &tensors, &cfg_json, None).unwrap();

        let eager = Glm::new(cfg, 1, 8, &checkpoint::load(p).by_role(""));
        let streamed = Glm::load_inference(p, 1, 8);

        for name in eager.param_names() {
            assert_eq!(eager.read_weight(&name), streamed.read_weight(&name), "weight {name}");
            assert_eq!(&streamed.read_weight(&name), &init[&name], "streamed {name} vs source");
        }
        std::fs::remove_file(&path).ok();
    }

    /// lm_head applied on the host to a single final-norm hidden row, so the KV
    /// decode's hidden can be turned into logits and checked against `logits_all`.
    fn head_logits(m: &Glm, hidden: &[f32]) -> Vec<f32> {
        let d = m.cfg.d_model as usize;
        let v = m.cfg.vocab as usize;
        let w = m.read_weight(m.cfg.head_weight()); // [v, d]
        (0..v)
            .map(|o| {
                let row = &w[o * d..(o + 1) * d];
                hidden.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect()
    }

    /// The incremental KV-cache `step` must reproduce GLM's own `O(T²)` recompute
    /// (`hidden_all` / `logits_all`) for every prefix — same engine, same weights,
    /// so any difference is only float reassociation in the MLA attention reduction
    /// and the (naive n=1 vs batched) matmuls. Exercises MLA + one dense + one MoE
    /// layer (`tiny()`: first_k_dense_replace=1, n_layers=2).
    #[test]
    fn kv_step_matches_full_recompute() {
        let cfg = GlmConfig::tiny();
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let t = 8u32;
        let seq = 6usize;
        let init = crate::init::init_weights(&cfg, 7);
        let m = Glm::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, t, &init);

        let tokens: Vec<u32> = (0..seq).map(|i| ((i * 5 + 3) as u32) % cfg.vocab).collect();

        // Incremental: feed one token at a time through the KV cache.
        m.reset_cache();
        let inc: Vec<Vec<f32>> = tokens.iter().map(|&tk| m.step(tk)).collect();
        assert_eq!(m.cache_pos(), seq as u32);

        // Reference: full recompute of each prefix; compare the last row.
        let mut worst_h = 0.0f32;
        let mut worst_l = 0.0f32;
        for i in 0..seq {
            let pref = &tokens[..=i];
            let hid = m.hidden_all(pref);
            worst_h = worst_h.max(maxabs(&inc[i], &hid[i * d..(i + 1) * d]));
            let logits = m.logits_all(pref);
            let dec_l = head_logits(&m, &inc[i]);
            worst_l = worst_l.max(maxabs(&dec_l, &logits[i * v..(i + 1) * v]));
        }
        println!("kv_step_matches_full_recompute: hidden maxabs={worst_h:.3e}  logits maxabs={worst_l:.3e}");
        assert!(worst_h < 3e-3, "KV decode hidden diverges from recompute: maxabs={worst_h}");
        assert!(worst_l < 3e-3, "KV decode logits diverge from recompute: maxabs={worst_l}");
    }
}
