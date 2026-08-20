// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DiT training: device-resident weight/gradient buffers and full
//! activation caching for the 36-layer transformer stack (LayerNorm/QKV/
//! partial-RoPE/bidirectional attention/SwiGLU, over `model::block`'s
//! builders), backward wired to `crates/gradcheck`'s `CheckModel` contract
//! (`param_names`/`read_weight`/`write_weight`/`read_grad`/`loss`/
//! `zero_grads`/`backward`).
//!
//! The top-level glue (the two `k=1` convs, `proj_in`/`proj_out`, the
//! Fourier timestep embedding) stays host math here, matching
//! `dit::forward`'s own choice for those same ops - a `k=1` conv IS a
//! per-position matvec, and a timestep embedding runs once per forward on a
//! `fourier_embedding_dim`-wide vector, so hand-deriving their backward on
//! the host (via `model::hostmath::linear_rows_bwd`/[`crate::dit_train::dsilu_bwd`])
//! is simpler and just as correct as a device round trip at gradcheck's
//! tiny dims. The block stack's own attention/RoPE/LayerNorm/SwiGLU math is
//! device-dispatched through the SAME builders `dit::forward` uses -
//! reusing already-gradchecked primitives instead of hand-deriving softmax
//! backward again is what makes this tractable at all.
//!
//! Separate from `dit::forward` (the served path) for the same reason
//! `train.rs` is separate from `vocoder::forward`: training needs
//! persistent device weight/gradient buffers reused across steps and every
//! activation cached for backward, neither of which the one-shot served
//! forward should pay for.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, bidir_bwd, bidir_fwd, layernorm_dx_bwd, layernorm_fwd, ln_stats_fwd, rope2d_partial_bwd, rope2d_partial_fwd, swiglu_bwd, swiglu_fwd, Bidir, BidirIds, KernelIds, LayerNormIds};
use model::hostmath::{dsilu, linear_rows_bwd, matvec};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::config::DitConfig;
use crate::dit::{AttnW, BlockW, DitWeights};

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("kv_expand", kernels::KV_EXPAND),
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
];
const MATMUL: usize = 0;
const MATMUL_DX: usize = 1;
const MATMUL_DW: usize = 2;
const LAYERNORM: usize = 3;
const LN_STATS: usize = 4;
const LAYERNORM_DX: usize = 5;
const LAYERNORM_DGAMMA: usize = 6;
const LAYERNORM_DBETA: usize = 7;
const ADD2: usize = 8;
const BIAS_ADD: usize = 9;
const BIAS_GRAD: usize = 10;
const KV_EXPAND: usize = 11;
const KV_EXPAND_BWD: usize = 12;
const ROPE2D_PARTIAL: usize = 13;
const ATTN_SCORES_BIDIR: usize = 14;
const ATTN_SOFTMAX_BIDIR: usize = 15;
const ATTN_APPLY_BIDIR: usize = 16;
const ATTN_BWD_DSCORES_BIDIR: usize = 17;
const ATTN_BWD_DV_BIDIR: usize = 18;
const ATTN_BWD_DQ_BIDIR: usize = 19;
const ATTN_BWD_DK_BIDIR: usize = 20;
const SILU_MUL: usize = 21;
const SILU_BWD_DA: usize = 22;
const SILU_BWD_DB: usize = 23;

const LN_EPS: f32 = 1e-5;

fn bidir_ids() -> BidirIds {
    BidirIds { scores: ATTN_SCORES_BIDIR, softmax: ATTN_SOFTMAX_BIDIR, apply: ATTN_APPLY_BIDIR, dscores: ATTN_BWD_DSCORES_BIDIR, dv: ATTN_BWD_DV_BIDIR, dq: ATTN_BWD_DQ_BIDIR, dk: ATTN_BWD_DK_BIDIR }
}
/// Every field but `silu_mul`/`silu_da`/`silu_db` is unread here.
fn kernel_ids() -> KernelIds {
    KernelIds { rmsnorm: 0, rms_inv: 0, rmsnorm_dx: 0, rmsnorm_dw: 0, rope: 0, rope_bwd: 0, gqa_scores: 0, gqa_apply: 0, attn_softmax: 0, gqa_dscores: 0, gqa_dv: 0, gqa_dq: 0, gqa_dk: 0, silu_mul: SILU_MUL, silu_da: SILU_BWD_DA, silu_db: SILU_BWD_DB }
}
fn ln_ids() -> LayerNormIds {
    LayerNormIds { layernorm: LAYERNORM, layernorm_rows: None, ln_stats: LN_STATS, ln_stats_rows: None, layernorm_dx: LAYERNORM_DX, layernorm_dx_rows: None }
}

/// `{prefix}.{weight,bias}` device buffer pair, with matching gradient
/// buffers - one per named linear/norm tensor in a [`BlockW`].
struct Pair {
    w: DeviceBuffer,
    dw: DeviceBuffer,
    name: String,
}
impl Pair {
    fn upload(gpu: &Gpu, name: &str, v: &[f32]) -> Pair {
        Pair { w: gpu.storage_init("w", v), dw: gpu.storage(v.len() as u64), name: name.to_string() }
    }
}

struct BlockD {
    norm1_w: Pair,
    norm1_b: Pair,
    wq: Pair,
    wk: Pair,
    wv: Pair,
    wo: Pair,
    norm2_w: Pair,
    norm2_b: Pair,
    ff_in_w: Pair,
    ff_in_b: Pair,
    ff_out_w: Pair,
    ff_out_b: Pair,
}

impl BlockD {
    fn upload(gpu: &Gpu, i: usize, b: &BlockW) -> BlockD {
        let p = format!("blocks.{i}");
        BlockD {
            norm1_w: Pair::upload(gpu, &format!("{p}.norm1.weight"), &b.norm1_w),
            norm1_b: Pair::upload(gpu, &format!("{p}.norm1.bias"), &b.norm1_b),
            wq: Pair::upload(gpu, &format!("{p}.attn.to_q"), &b.attn.wq),
            wk: Pair::upload(gpu, &format!("{p}.attn.to_k"), &b.attn.wk),
            wv: Pair::upload(gpu, &format!("{p}.attn.to_v"), &b.attn.wv),
            wo: Pair::upload(gpu, &format!("{p}.attn.to_out"), &b.attn.wo),
            norm2_w: Pair::upload(gpu, &format!("{p}.norm2.weight"), &b.norm2_w),
            norm2_b: Pair::upload(gpu, &format!("{p}.norm2.bias"), &b.norm2_b),
            ff_in_w: Pair::upload(gpu, &format!("{p}.ff_in.weight"), &b.ff_in_w),
            ff_in_b: Pair::upload(gpu, &format!("{p}.ff_in.bias"), &b.ff_in_b),
            ff_out_w: Pair::upload(gpu, &format!("{p}.ff_out.weight"), &b.ff_out_w),
            ff_out_b: Pair::upload(gpu, &format!("{p}.ff_out.bias"), &b.ff_out_b),
        }
    }
    fn pairs(&self) -> [&Pair; 12] {
        [&self.norm1_w, &self.norm1_b, &self.wq, &self.wk, &self.wv, &self.wo, &self.norm2_w, &self.norm2_b, &self.ff_in_w, &self.ff_in_b, &self.ff_out_w, &self.ff_out_b]
    }
}

/// Host-resident weights/grads for the top-level glue - see the module doc
/// for why these stay off the device.
#[derive(Clone)]
struct HostGlue {
    preprocess_conv_w: Vec<f32>,
    proj_in_w: Vec<f32>,
    time_proj_weight: Vec<f32>,
    time_embed_l1_w: Vec<f32>,
    time_embed_l1_b: Vec<f32>,
    time_embed_l2_w: Vec<f32>,
    time_embed_l2_b: Vec<f32>,
    proj_out_w: Vec<f32>,
    postprocess_conv_w: Vec<f32>,
}

impl HostGlue {
    fn from_weights(w: &DitWeights) -> HostGlue {
        HostGlue {
            preprocess_conv_w: w.preprocess_conv_w.clone(),
            proj_in_w: w.proj_in_w.clone(),
            time_proj_weight: w.time_proj_weight.clone(),
            time_embed_l1_w: w.time_embed_l1_w.clone(),
            time_embed_l1_b: w.time_embed_l1_b.clone(),
            time_embed_l2_w: w.time_embed_l2_w.clone(),
            time_embed_l2_b: w.time_embed_l2_b.clone(),
            proj_out_w: w.proj_out_w.clone(),
            postprocess_conv_w: w.postprocess_conv_w.clone(),
        }
    }
    fn zero_like(&self) -> HostGlue {
        HostGlue {
            preprocess_conv_w: vec![0.0; self.preprocess_conv_w.len()],
            proj_in_w: vec![0.0; self.proj_in_w.len()],
            time_proj_weight: vec![0.0; self.time_proj_weight.len()],
            time_embed_l1_w: vec![0.0; self.time_embed_l1_w.len()],
            time_embed_l1_b: vec![0.0; self.time_embed_l1_b.len()],
            time_embed_l2_w: vec![0.0; self.time_embed_l2_w.len()],
            time_embed_l2_b: vec![0.0; self.time_embed_l2_b.len()],
            proj_out_w: vec![0.0; self.proj_out_w.len()],
            postprocess_conv_w: vec![0.0; self.postprocess_conv_w.len()],
        }
    }
    /// `(name, field accessor)` pairs, in a fixed order shared by weights
    /// and grads so the two never drift apart.
    fn field_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        match name {
            "preprocess_conv.weight" => Some(&mut self.preprocess_conv_w),
            "proj_in.weight" => Some(&mut self.proj_in_w),
            "time_proj.weight" => Some(&mut self.time_proj_weight),
            "time_embed.linear_1.weight" => Some(&mut self.time_embed_l1_w),
            "time_embed.linear_1.bias" => Some(&mut self.time_embed_l1_b),
            "time_embed.linear_2.weight" => Some(&mut self.time_embed_l2_w),
            "time_embed.linear_2.bias" => Some(&mut self.time_embed_l2_b),
            "proj_out.weight" => Some(&mut self.proj_out_w),
            "postprocess_conv.weight" => Some(&mut self.postprocess_conv_w),
            _ => None,
        }
    }
    fn names() -> [&'static str; 9] {
        [
            "preprocess_conv.weight",
            "proj_in.weight",
            "time_proj.weight",
            "time_embed.linear_1.weight",
            "time_embed.linear_1.bias",
            "time_embed.linear_2.weight",
            "time_embed.linear_2.bias",
            "proj_out.weight",
            "postprocess_conv.weight",
        ]
    }
}

/// Cached activations one block's backward needs.
struct BlockCache {
    x: DeviceBuffer,
    mean1: DeviceBuffer,
    inv1: DeviceBuffer,
    xn1: DeviceBuffer,
    qkv: DeviceBuffer, // post-rope, the exact state bidir_fwd read
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    mean2: DeviceBuffer,
    inv2: DeviceBuffer,
    xn2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    act: DeviceBuffer,
}

/// One block forward, device-dispatched, caching every activation its own
/// backward needs. Mirrors `dit::block_fwd` exactly except for the caching.
fn block_fwd(gpu: &Gpu, cfg: &DitConfig, bd: &BlockD, cos_b: &DeviceBuffer, sin_b: &DeviceBuffer, x: &DeviceBuffer, rows: usize) -> (BlockCache, DeviceBuffer) {
    let inner = cfg.inner_dim() as usize;
    let heads = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let half_rot = cfg.rotary_dim / 2;
    let ff_inner = cfg.ff_inner_dim as usize;
    let stride3 = (3 * inner) as u32;

    let mean1 = gpu.storage(rows as u64);
    let inv1 = gpu.storage(rows as u64);
    let xn1 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[ln_stats_fwd(gpu, &ln_ids(), x, &mean1, &inv1, inner as u32, rows as u32, LN_EPS), layernorm_fwd(gpu, &ln_ids(), x, &bd.norm1_w.w, &bd.norm1_b.w, &xn1, inner as u32, rows as u32, LN_EPS)],
    );

    let q_tmp = gpu.storage((rows * inner) as u64);
    let k_tmp = gpu.storage((rows * inner) as u64);
    let v_tmp = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&xn1, &bd.wq.w, &q_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(MATMUL, &[&xn1, &bd.wk.w, &k_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(MATMUL, &[&xn1, &bd.wv.w, &v_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
        ],
    );

    let qkv = gpu.storage((rows * 3 * inner) as u64);
    gpu.submit(
        &[],
        &[
            block::kv_expand_fwd(gpu, KV_EXPAND, &q_tmp, &qkv, rows as u32, heads, 1, hd, stride3, 0),
            block::kv_expand_fwd(gpu, KV_EXPAND, &k_tmp, &qkv, rows as u32, heads, 1, hd, stride3, inner as u32),
            block::kv_expand_fwd(gpu, KV_EXPAND, &v_tmp, &qkv, rows as u32, heads, 1, hd, stride3, 2 * inner as u32),
        ],
    );
    gpu.submit(
        &[],
        &[
            rope2d_partial_fwd(gpu, ROPE2D_PARTIAL, &qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, 0, hd),
            rope2d_partial_fwd(gpu, ROPE2D_PARTIAL, &qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, inner as u32, hd),
        ],
    );

    let bidir_cfg = Bidir { b: 1, t: rows as u32, n_heads: heads, head_dim: hd, stride: stride3, q_off: 0, k_off: inner as u32, v_off: 2 * inner as u32 };
    let scores = gpu.storage((heads * rows as u32 * rows as u32) as u64);
    let probs = gpu.storage((heads * rows as u32 * rows as u32) as u64);
    let ctx = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &bidir_fwd(gpu, &bidir_ids(), &bidir_cfg, &qkv, &scores, &probs, &ctx));

    let proj = gpu.storage((rows * inner) as u64);
    let xmid = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(MATMUL, &[&ctx, &bd.wo.w, &proj], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32)]);
    gpu.submit(&[], &[gpu.step(ADD2, &[x, &proj, &xmid], &[(rows * inner) as u32], (rows * inner) as u32)]);

    let mean2 = gpu.storage(rows as u64);
    let inv2 = gpu.storage(rows as u64);
    let xn2 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            ln_stats_fwd(gpu, &ln_ids(), &xmid, &mean2, &inv2, inner as u32, rows as u32, LN_EPS),
            layernorm_fwd(gpu, &ln_ids(), &xmid, &bd.norm2_w.w, &bd.norm2_b.w, &xn2, inner as u32, rows as u32, LN_EPS),
        ],
    );

    let ff_raw = gpu.storage((rows * 2 * ff_inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&xn2, &bd.ff_in_w.w, &ff_raw], &[rows as u32, inner as u32, (2 * ff_inner) as u32], (rows * 2 * ff_inner) as u32),
            gpu.step(BIAS_ADD, &[&ff_raw, &bd.ff_in_b.w], &[rows as u32, (2 * ff_inner) as u32], (rows * 2 * ff_inner) as u32),
        ],
    );

    let ff_host = gpu.read(&ff_raw, rows * 2 * ff_inner);
    let mut up = vec![0.0f32; rows * ff_inner];
    let mut gate = vec![0.0f32; rows * ff_inner];
    for r in 0..rows {
        up[r * ff_inner..(r + 1) * ff_inner].copy_from_slice(&ff_host[r * 2 * ff_inner..r * 2 * ff_inner + ff_inner]);
        gate[r * ff_inner..(r + 1) * ff_inner].copy_from_slice(&ff_host[r * 2 * ff_inner + ff_inner..(r + 1) * 2 * ff_inner]);
    }
    let gate_b = gpu.storage_init("ff_gate", &gate);
    let up_b = gpu.storage_init("ff_up", &up);
    let act = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(&[], &[swiglu_fwd(gpu, &kernel_ids(), &gate_b, &up_b, &act, (rows * ff_inner) as u32)]);

    let ff_out = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&act, &bd.ff_out_w.w, &ff_out], &[rows as u32, ff_inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(BIAS_ADD, &[&ff_out, &bd.ff_out_b.w], &[rows as u32, inner as u32], (rows * inner) as u32),
        ],
    );

    let x_next = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(ADD2, &[&xmid, &ff_out, &x_next], &[(rows * inner) as u32], (rows * inner) as u32)]);

    (BlockCache { x: x.clone(), mean1, inv1, xn1, qkv, probs, ctx, xmid, mean2, inv2, xn2, gate: gate_b, up: up_b, act }, x_next)
}

/// One block backward: accumulates every weight gradient into `bd`'s
/// (pre-zeroed) `dw` buffers and returns `d_x` (this block's input grad).
#[allow(clippy::too_many_arguments)]
fn block_bwd(gpu: &Gpu, cfg: &DitConfig, bd: &BlockD, cache: &BlockCache, cos_b: &DeviceBuffer, sin_b: &DeviceBuffer, d_x_next: &DeviceBuffer, rows: usize) -> DeviceBuffer {
    let inner = cfg.inner_dim() as usize;
    let heads = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let half_rot = cfg.rotary_dim / 2;
    let ff_inner = cfg.ff_inner_dim as usize;
    let stride3 = (3 * inner) as u32;

    // x_next = xmid + ff_out (ADD2 backward: both branches get d_x_next).
    let d_ff_out = d_x_next.clone();

    let d_act = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL_DX, &[&d_ff_out, &bd.ff_out_w.w, &d_act], &[rows as u32, ff_inner as u32, inner as u32, 0], (rows * ff_inner) as u32),
            gpu.step(MATMUL_DW, &[&d_ff_out, &cache.act, &bd.ff_out_w.dw], &[rows as u32, ff_inner as u32, inner as u32], (inner * ff_inner) as u32),
            gpu.step(BIAS_GRAD, &[&d_ff_out, &bd.ff_out_b.dw], &[rows as u32, inner as u32], inner as u32),
        ],
    );

    let d_gate = gpu.storage((rows * ff_inner) as u64);
    let d_up = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(&[], &swiglu_bwd(gpu, &kernel_ids(), &cache.gate, &cache.up, &d_act, &d_gate, &d_up, (rows * ff_inner) as u32));

    // Recombine into the ORIGINAL ff_in output layout [up(=gate_states) |
    // gate] (see block_fwd's own comment) - matmul_dw needs one dY buffer
    // shaped like the forward matmul's own output.
    let d_gate_h = gpu.read(&d_gate, rows * ff_inner);
    let d_up_h = gpu.read(&d_up, rows * ff_inner);
    let mut d_ff_raw = vec![0.0f32; rows * 2 * ff_inner];
    for r in 0..rows {
        d_ff_raw[r * 2 * ff_inner..r * 2 * ff_inner + ff_inner].copy_from_slice(&d_up_h[r * ff_inner..(r + 1) * ff_inner]);
        d_ff_raw[r * 2 * ff_inner + ff_inner..(r + 1) * 2 * ff_inner].copy_from_slice(&d_gate_h[r * ff_inner..(r + 1) * ff_inner]);
    }
    let d_ff_raw_b = gpu.storage_init("d_ff_raw", &d_ff_raw);

    let d_xn2 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL_DX, &[&d_ff_raw_b, &bd.ff_in_w.w, &d_xn2], &[rows as u32, inner as u32, (2 * ff_inner) as u32, 0], (rows * inner) as u32),
            gpu.step(MATMUL_DW, &[&d_ff_raw_b, &cache.xn2, &bd.ff_in_w.dw], &[rows as u32, inner as u32, (2 * ff_inner) as u32], (inner * 2 * ff_inner) as u32),
            gpu.step(BIAS_GRAD, &[&d_ff_raw_b, &bd.ff_in_b.dw], &[rows as u32, (2 * ff_inner) as u32], (2 * ff_inner) as u32),
        ],
    );

    let d_xmid_from_norm2 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            layernorm_dx_bwd(gpu, &ln_ids(), &cache.xmid, &bd.norm2_w.w, &d_xn2, &d_xmid_from_norm2, inner as u32, rows as u32, LN_EPS),
            gpu.step(LAYERNORM_DGAMMA, &[&d_xn2, &cache.xmid, &cache.mean2, &cache.inv2, &bd.norm2_w.dw], &[inner as u32, rows as u32], inner as u32),
            gpu.step(LAYERNORM_DBETA, &[&d_xn2, &bd.norm2_b.dw], &[inner as u32, rows as u32], inner as u32),
        ],
    );

    // xmid = x + proj (ADD2 backward): both branches get the SAME sum of
    // grads flowing into xmid (the residual passthrough `d_x_next` and the
    // norm2 branch above).
    let d_xmid = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(ADD2, &[d_x_next, &d_xmid_from_norm2, &d_xmid], &[(rows * inner) as u32], (rows * inner) as u32)]);
    let d_proj = d_xmid.clone();

    let d_ctx = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL_DX, &[&d_proj, &bd.wo.w, &d_ctx], &[rows as u32, inner as u32, inner as u32, 0], (rows * inner) as u32),
            gpu.step(MATMUL_DW, &[&d_proj, &cache.ctx, &bd.wo.dw], &[rows as u32, inner as u32, inner as u32], (inner * inner) as u32),
        ],
    );

    let bidir_cfg = Bidir { b: 1, t: rows as u32, n_heads: heads, head_dim: hd, stride: stride3, q_off: 0, k_off: inner as u32, v_off: 2 * inner as u32 };
    let d_scores = gpu.storage((heads * rows as u32 * rows as u32) as u64);
    let d_qkv = gpu.storage((rows * 3 * inner) as u64);
    gpu.submit(&[], &bidir_bwd(gpu, &bidir_ids(), &bidir_cfg, &cache.qkv, &cache.probs, &d_ctx, &d_scores, &d_qkv));

    // Un-rotate the q/k regions of d_qkv in place (rope2d_partial_bwd is
    // rope2d_partial_fwd's exact inverse - see model::block's doc).
    gpu.submit(
        &[],
        &[
            rope2d_partial_bwd(gpu, ROPE2D_PARTIAL, &d_qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, 0, hd),
            rope2d_partial_bwd(gpu, ROPE2D_PARTIAL, &d_qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, inner as u32, hd),
        ],
    );

    let d_q_tmp = gpu.storage((rows * inner) as u64);
    let d_k_tmp = gpu.storage((rows * inner) as u64);
    let d_v_tmp = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            block::kv_expand_bwd(gpu, KV_EXPAND_BWD, &d_qkv, &d_q_tmp, rows as u32, heads, 1, hd, stride3, 0),
            block::kv_expand_bwd(gpu, KV_EXPAND_BWD, &d_qkv, &d_k_tmp, rows as u32, heads, 1, hd, stride3, inner as u32),
            block::kv_expand_bwd(gpu, KV_EXPAND_BWD, &d_qkv, &d_v_tmp, rows as u32, heads, 1, hd, stride3, 2 * inner as u32),
        ],
    );

    // xn1 feeds THREE matmuls (wq/wk/wv); their dx contributions accumulate
    // into one d_xn1 (each weight's own dw is independent, no accumulation
    // needed across weights). `gpu.storage` zero-inits, so `accumulate=1`
    // from the start is correct for every one of the three.
    let d_xn1 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL_DX, &[&d_q_tmp, &bd.wq.w, &d_xn1], &[rows as u32, inner as u32, inner as u32, 1], (rows * inner) as u32),
            gpu.step(MATMUL_DW, &[&d_q_tmp, &cache.xn1, &bd.wq.dw], &[rows as u32, inner as u32, inner as u32], (inner * inner) as u32),
            gpu.step(MATMUL_DX, &[&d_k_tmp, &bd.wk.w, &d_xn1], &[rows as u32, inner as u32, inner as u32, 1], (rows * inner) as u32),
            gpu.step(MATMUL_DW, &[&d_k_tmp, &cache.xn1, &bd.wk.dw], &[rows as u32, inner as u32, inner as u32], (inner * inner) as u32),
            gpu.step(MATMUL_DX, &[&d_v_tmp, &bd.wv.w, &d_xn1], &[rows as u32, inner as u32, inner as u32, 1], (rows * inner) as u32),
            gpu.step(MATMUL_DW, &[&d_v_tmp, &cache.xn1, &bd.wv.dw], &[rows as u32, inner as u32, inner as u32], (inner * inner) as u32),
        ],
    );

    let d_x_from_norm1 = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            layernorm_dx_bwd(gpu, &ln_ids(), &cache.x, &bd.norm1_w.w, &d_xn1, &d_x_from_norm1, inner as u32, rows as u32, LN_EPS),
            gpu.step(LAYERNORM_DGAMMA, &[&d_xn1, &cache.x, &cache.mean1, &cache.inv1, &bd.norm1_w.dw], &[inner as u32, rows as u32], inner as u32),
            gpu.step(LAYERNORM_DBETA, &[&d_xn1, &bd.norm1_b.dw], &[inner as u32, rows as u32], inner as u32),
        ],
    );

    // x -> norm1 -> xn1 AND x -> (residual) -> xmid: both branches of x's
    // own fan-out sum.
    let d_x = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(ADD2, &[&d_xmid, &d_x_from_norm1, &d_x], &[(rows * inner) as u32], (rows * inner) as u32)]);
    d_x
}

/// Cached top-level activations `run_backward` needs, alongside every
/// block's own [`BlockCache`].
struct ForwardCache {
    concat_lc: Vec<f32>, // [length, concat]
    hidden_lc: Vec<f32>, // [length, concat] (post preprocess residual)
    embed: Vec<f32>,     // [fourier_embedding_dim] cos/sin concat
    h1_lin: Vec<f32>,    // pre-SiLU time_embed.linear_1 output, [inner]
    h1: Vec<f32>,        // post-SiLU, [inner]
    rows: usize,
    blocks: Vec<BlockCache>,
    x_final_host: Vec<f32>, // [rows, inner], the last block's output
    y_lc: Vec<f32>,         // [length, cin] (post proj_out, pre postprocess residual)
    output: Vec<f32>,       // [cin, length] NCL, the trainer's own forward output
}

/// DiT trainer: persistent device weight/gradient buffers for the 36-layer
/// block stack, host weight/gradient storage for the top-level glue (see
/// the module doc), a fixed (latents, condition, timestep, target) tuple,
/// and an MSE reconstruction loss - enough to gradient-check every
/// backward pass here, not a real training loss.
pub struct Trainer {
    gpu: Gpu,
    cfg: DitConfig,
    blocks: Vec<BlockD>,
    host: RefCell<HostGlue>,
    host_grad: RefCell<HostGlue>,
    sizes: HashMap<String, usize>,
    latents: Vec<f32>,   // [cin, length] NCL
    condition: Vec<f32>, // [length, condition_dim]
    timestep: f32,
    length: usize,
    target: Vec<f32>, // [cin, length] NCL
    cache: RefCell<Option<ForwardCache>>,
}

impl Trainer {
    pub fn new(cfg: DitConfig, w: &DitWeights, latents: Vec<f32>, condition: Vec<f32>, timestep: f32, length: usize, target: Vec<f32>) -> Trainer {
        let gpu = Gpu::new_cpu(PIPELINES);
        let cin = cfg.in_channels as usize;
        assert_eq!(latents.len(), cin * length, "Trainer::new: latents length mismatch");
        assert_eq!(condition.len(), length * cfg.condition_dim as usize, "Trainer::new: condition length mismatch");
        assert_eq!(target.len(), cin * length, "Trainer::new: target length mismatch");

        let blocks: Vec<BlockD> = w.blocks.iter().enumerate().map(|(i, b)| BlockD::upload(&gpu, i, b)).collect();
        let host = HostGlue::from_weights(w);
        let host_grad = host.zero_like();

        let mut sizes = HashMap::new();
        for name in HostGlue::names() {
            sizes.insert(name.to_string(), host.clone().field_mut(name).unwrap().len());
        }
        // Keyed by the SAME names `BlockD::upload` gave each `Pair`.
        for (i, b) in w.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            sizes.insert(format!("{p}.norm1.weight"), b.norm1_w.len());
            sizes.insert(format!("{p}.norm1.bias"), b.norm1_b.len());
            sizes.insert(format!("{p}.attn.to_q"), b.attn.wq.len());
            sizes.insert(format!("{p}.attn.to_k"), b.attn.wk.len());
            sizes.insert(format!("{p}.attn.to_v"), b.attn.wv.len());
            sizes.insert(format!("{p}.attn.to_out"), b.attn.wo.len());
            sizes.insert(format!("{p}.norm2.weight"), b.norm2_w.len());
            sizes.insert(format!("{p}.norm2.bias"), b.norm2_b.len());
            sizes.insert(format!("{p}.ff_in.weight"), b.ff_in_w.len());
            sizes.insert(format!("{p}.ff_in.bias"), b.ff_in_b.len());
            sizes.insert(format!("{p}.ff_out.weight"), b.ff_out_w.len());
            sizes.insert(format!("{p}.ff_out.bias"), b.ff_out_b.len());
        }

        Trainer { gpu, cfg, blocks, host: RefCell::new(host), host_grad: RefCell::new(host_grad), sizes, latents, condition, timestep, length, target, cache: RefCell::new(None) }
    }

    fn run_forward(&self) -> f32 {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let host = self.host.borrow();
        let (cin, cdim, concat) = (cfg.in_channels as usize, cfg.condition_dim as usize, cfg.concat_channels() as usize);
        let inner = cfg.inner_dim() as usize;
        let length = self.length;

        let mut concat_lc = vec![0.0f32; length * concat];
        for t in 0..length {
            for c in 0..cin {
                concat_lc[t * concat + c] = self.latents[c * length + t];
            }
            for c in 0..cdim {
                concat_lc[t * concat + 2 * cin + c] = self.condition[t * cdim + c];
            }
        }
        let mut hidden_lc = vec![0.0f32; length * concat];
        for t in 0..length {
            let row = matvec(&host.preprocess_conv_w, &concat_lc[t * concat..(t + 1) * concat], concat, concat);
            for c in 0..concat {
                hidden_lc[t * concat + c] = concat_lc[t * concat + c] + row[c];
            }
        }

        let half = cfg.fourier_embedding_dim as usize / 2;
        let mut embed = vec![0.0f32; 2 * half];
        for i in 0..half {
            let angle = 2.0 * std::f32::consts::PI * self.timestep * host.time_proj_weight[i];
            embed[i] = angle.cos();
            embed[half + i] = angle.sin();
        }
        let h1_lin: Vec<f32> = matvec(&host.time_embed_l1_w, &embed, inner, cfg.fourier_embedding_dim as usize)
            .iter()
            .zip(&host.time_embed_l1_b)
            .map(|(&v, &b)| v + b)
            .collect();
        let h1: Vec<f32> = h1_lin.iter().map(|&v| model::hostmath::silu(v)).collect();
        let temb: Vec<f32> = matvec(&host.time_embed_l2_w, &h1, inner, inner).iter().zip(&host.time_embed_l2_b).map(|(&v, &b)| v + b).collect();

        let rows = length + 1;
        let mut x_host = vec![0.0f32; rows * inner];
        x_host[..inner].copy_from_slice(&temb);
        for t in 0..length {
            let row = matvec(&host.proj_in_w, &hidden_lc[t * concat..(t + 1) * concat], inner, concat);
            x_host[(t + 1) * inner..(t + 2) * inner].copy_from_slice(&row);
        }

        let (cos_t, sin_t) = crate::dit::rope_tables(rows, cfg.rotary_dim as usize, 10000.0);
        let cos_b = gpu.storage_init("rope.cos", &cos_t);
        let sin_b = gpu.storage_init("rope.sin", &sin_t);

        let mut x = gpu.storage_init("x0", &x_host);
        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for bd in &self.blocks {
            let (bc, x_next) = block_fwd(gpu, cfg, bd, &cos_b, &sin_b, &x, rows);
            block_caches.push(bc);
            x = x_next;
        }
        let x_final_host = gpu.read(&x, rows * inner);

        let mut y_lc = vec![0.0f32; length * cin];
        for t in 0..length {
            let row = matvec(&host.proj_out_w, &x_final_host[(t + 1) * inner..(t + 2) * inner], cin, inner);
            y_lc[t * cin..(t + 1) * cin].copy_from_slice(&row);
        }
        let mut out_lc = vec![0.0f32; length * cin];
        for t in 0..length {
            let row = matvec(&host.postprocess_conv_w, &y_lc[t * cin..(t + 1) * cin], cin, cin);
            for c in 0..cin {
                out_lc[t * cin + c] = y_lc[t * cin + c] + row[c];
            }
        }
        let mut output = vec![0.0f32; cin * length];
        for t in 0..length {
            for c in 0..cin {
                output[c * length + t] = out_lc[t * cin + c];
            }
        }

        let n = output.len() as f32;
        let loss: f32 = output.iter().zip(&self.target).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (2.0 * n);

        *self.cache.borrow_mut() = Some(ForwardCache { concat_lc, hidden_lc, embed, h1_lin, h1, rows, blocks: block_caches, x_final_host, y_lc, output });
        loss
    }

    fn run_backward(&self) {
        let cache_ref = self.cache.borrow();
        let cache = cache_ref.as_ref().expect("Trainer::backward called before a forward (loss()) ran");
        let host = self.host.borrow();
        let mut hg = self.host_grad.borrow_mut();
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let (cin, concat) = (cfg.in_channels as usize, cfg.concat_channels() as usize);
        let inner = cfg.inner_dim() as usize;
        let length = self.length;
        let rows = cache.rows;

        let n = cache.output.len() as f32;
        let d_out_ncl: Vec<f32> = cache.output.iter().zip(&self.target).map(|(a, b)| (a - b) / n).collect();
        let mut d_out_lc = vec![0.0f32; length * cin];
        for t in 0..length {
            for c in 0..cin {
                d_out_lc[t * cin + c] = d_out_ncl[c * length + t];
            }
        }

        // postprocess: out_lc[t] = y_lc[t] + Wpost @ y_lc[t] (residual).
        let (d_y_lc_conv, d_post_w) = linear_rows_bwd(&cache.y_lc, &host.postprocess_conv_w, &d_out_lc, length, cin, cin);
        let mut d_y_lc = vec![0.0f32; length * cin];
        for i in 0..d_y_lc.len() {
            d_y_lc[i] = d_out_lc[i] + d_y_lc_conv[i];
        }
        for (g, d) in hg.postprocess_conv_w.iter_mut().zip(&d_post_w) {
            *g += d;
        }

        // proj_out: y_lc[t] = Wpout @ x_final_row(t+1) (plain linear, no residual).
        let x_final_rows = &cache.x_final_host[inner..]; // rows 1..rows, [length, inner]
        let (d_x_final_rows, d_pout_w) = linear_rows_bwd(x_final_rows, &host.proj_out_w, &d_y_lc, length, inner, cin);
        for (g, d) in hg.proj_out_w.iter_mut().zip(&d_pout_w) {
            *g += d;
        }

        let mut d_x_final = vec![0.0f32; rows * inner];
        d_x_final[inner..].copy_from_slice(&d_x_final_rows);
        let mut d_x_next = gpu.storage_init("d_x_final", &d_x_final);

        let (cos_t, sin_t) = crate::dit::rope_tables(rows, cfg.rotary_dim as usize, 10000.0);
        let cos_b = gpu.storage_init("rope.cos", &cos_t);
        let sin_b = gpu.storage_init("rope.sin", &sin_t);

        for (bd, bc) in self.blocks.iter().zip(&cache.blocks).rev() {
            d_x_next = block_bwd(gpu, cfg, bd, bc, &cos_b, &sin_b, &d_x_next, rows);
        }
        let d_x0 = gpu.read(&d_x_next, rows * inner);
        let d_temb = &d_x0[..inner];
        let d_x0_rows = &d_x0[inner..]; // [length, inner]

        // proj_in: x0_row(t+1) = Win @ hidden_lc[t].
        let (d_hidden_lc, d_win) = linear_rows_bwd(&cache.hidden_lc, &host.proj_in_w, d_x0_rows, length, concat, inner);
        for (g, d) in hg.proj_in_w.iter_mut().zip(&d_win) {
            *g += d;
        }

        // timestep_embed: temb = l2 @ silu(l1 @ embed + b1) + b2.
        let (d_h1, d_l2_w) = linear_rows_bwd(&cache.h1, &host.time_embed_l2_w, d_temb, 1, inner, inner);
        for (g, d) in hg.time_embed_l2_w.iter_mut().zip(&d_l2_w) {
            *g += d;
        }
        for (g, d) in hg.time_embed_l2_b.iter_mut().zip(d_temb) {
            *g += d;
        }
        let d_h1_lin: Vec<f32> = d_h1.iter().zip(&cache.h1_lin).map(|(&dh, &pre)| dh * dsilu(pre)).collect();
        let (d_embed, d_l1_w) = linear_rows_bwd(&cache.embed, &host.time_embed_l1_w, &d_h1_lin, 1, cfg.fourier_embedding_dim as usize, inner);
        for (g, d) in hg.time_embed_l1_w.iter_mut().zip(&d_l1_w) {
            *g += d;
        }
        for (g, d) in hg.time_embed_l1_b.iter_mut().zip(&d_h1_lin) {
            *g += d;
        }
        let half = cfg.fourier_embedding_dim as usize / 2;
        let two_pi_t = 2.0 * std::f32::consts::PI * self.timestep;
        for i in 0..half {
            let (cos_i, sin_i) = (cache.embed[i], cache.embed[half + i]);
            hg.time_proj_weight[i] += (d_embed[i] * -sin_i + d_embed[half + i] * cos_i) * two_pi_t;
        }

        // preprocess: hidden_lc[t] = concat_lc[t] + Wpre @ concat_lc[t] (residual).
        let (_d_concat_lc_conv, d_pre_w) = linear_rows_bwd(&cache.concat_lc, &host.preprocess_conv_w, &d_hidden_lc, length, concat, concat);
        for (g, d) in hg.preprocess_conv_w.iter_mut().zip(&d_pre_w) {
            *g += d;
        }
    }

    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = HostGlue::names().iter().map(|s| s.to_string()).collect();
        for bd in &self.blocks {
            for p in bd.pairs() {
                names.push(p.name.clone());
            }
        }
        names
    }

    fn size_of(&self, name: &str) -> usize {
        *self.sizes.get(name).unwrap_or_else(|| panic!("no such parameter {name:?}"))
    }

    fn block_pair(&self, name: &str) -> Option<&Pair> {
        self.blocks.iter().flat_map(|bd| bd.pairs()).find(|p| p.name == name)
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        if let Some(p) = self.block_pair(name) {
            return self.gpu.read(&p.w, self.size_of(name));
        }
        self.host.borrow_mut().field_mut(name).unwrap_or_else(|| panic!("no such weight {name:?}")).clone()
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        if let Some(p) = self.block_pair(name) {
            self.gpu.write_f32(&p.w, data);
            return;
        }
        *self.host.borrow_mut().field_mut(name).unwrap_or_else(|| panic!("no such weight {name:?}")) = data.to_vec();
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        if let Some(p) = self.block_pair(name) {
            return self.gpu.read(&p.dw, self.size_of(name));
        }
        self.host_grad.borrow_mut().field_mut(name).unwrap_or_else(|| panic!("no such grad {name:?}")).clone()
    }
    pub fn zero_grads(&self) {
        for bd in &self.blocks {
            for p in bd.pairs() {
                let n = self.size_of(&p.name);
                self.gpu.write(&p.dw, &vec![0u32; n]);
            }
        }
        *self.host_grad.borrow_mut() = self.host.borrow().zero_like();
    }
    pub fn loss(&self) -> f32 {
        self.run_forward()
    }
    pub fn backward(&self) {
        self.run_backward();
    }

    /// The last forward's output - test-only, to check this module's own
    /// forward against `dit::forward`'s served path.
    pub fn output(&self) -> Vec<f32> {
        let cache_ref = self.cache.borrow();
        cache_ref.as_ref().expect("Trainer::output called before a forward (loss()) ran").output.clone()
    }
}

/// Random weights at `cfg`'s dims, deterministic from `seed` - shared by
/// this crate's own tests and `crates/gradcheck::minimaxmusic3::check_dit`.
pub fn random_weights(cfg: &DitConfig, seed: u64) -> DitWeights {
    let mut r = Lcg::new(seed);
    let inner = cfg.inner_dim() as usize;
    let concat = cfg.concat_channels() as usize;
    let cin = cfg.in_channels as usize;
    let ff_inner = cfg.ff_inner_dim as usize;
    let lin = |out: usize, inn: usize, r: &mut Lcg| r.vec_scaled(out * inn, 0.2);
    let ones = |n: usize| vec![1.0f32; n];
    let zeros = |n: usize| vec![0.0f32; n];

    let blocks = (0..cfg.num_layers as usize)
        .map(|_| BlockW {
            norm1_w: ones(inner),
            norm1_b: zeros(inner),
            attn: AttnW { wq: lin(inner, inner, &mut r), wk: lin(inner, inner, &mut r), wv: lin(inner, inner, &mut r), wo: lin(inner, inner, &mut r) },
            norm2_w: ones(inner),
            norm2_b: zeros(inner),
            ff_in_w: lin(2 * ff_inner, inner, &mut r),
            ff_in_b: r.vec_scaled(2 * ff_inner, 0.1),
            ff_out_w: lin(inner, ff_inner, &mut r),
            ff_out_b: r.vec_scaled(inner, 0.1),
        })
        .collect();

    DitWeights {
        time_proj_weight: r.vec_scaled(cfg.fourier_embedding_dim as usize / 2, 0.5),
        time_embed_l1_w: lin(inner, cfg.fourier_embedding_dim as usize, &mut r),
        time_embed_l1_b: r.vec_scaled(inner, 0.1),
        time_embed_l2_w: lin(inner, inner, &mut r),
        time_embed_l2_b: r.vec_scaled(inner, 0.1),
        preprocess_conv_w: lin(concat, concat, &mut r),
        proj_in_w: lin(inner, concat, &mut r),
        blocks,
        proj_out_w: lin(cin, inner, &mut r),
        postprocess_conv_w: lin(cin, cin, &mut r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit;

    fn fixture(cfg: &DitConfig, seed: u64) -> (DitWeights, Vec<f32>, Vec<f32>, f32, usize, Vec<f32>) {
        let w = random_weights(cfg, seed);
        let length = 3usize;
        let mut r = Lcg::new(seed + 1);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
        let timestep = 0.4f32;
        let target = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        (w, latents, condition, timestep, length, target)
    }

    #[test]
    fn forward_matches_serving_forward() {
        let cfg = DitConfig::tiny();
        let (w, latents, condition, timestep, length, target) = fixture(&cfg, 41);

        let gpu = Gpu::new_cpu(dit::PIPELINES);
        let served = dit::forward(&gpu, &cfg, &w, &latents, &condition, timestep, length);

        let trainer = Trainer::new(cfg, &w, latents, condition, timestep, length, target);
        let _ = trainer.loss();
        let trained_fwd = trainer.output();

        assert_eq!(served.len(), trained_fwd.len());
        let max_abs = served.iter().zip(&trained_fwd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e-3, "dit_train's own forward drifted from dit::forward: max_abs={max_abs}");
    }

    #[test]
    fn backward_matches_finite_differences() {
        let cfg = DitConfig::tiny();
        let (w, latents, condition, timestep, length, target) = fixture(&cfg, 51);

        let trainer = Trainer::new(cfg, &w, latents, condition, timestep, length, target);
        trainer.zero_grads();
        let _ = trainer.loss();
        trainer.backward();

        let eps = 5e-3f32;
        let mut checked = 0;
        for name in trainer.param_names() {
            let base = trainer.read_weight(&name);
            let ana = trainer.read_grad(&name);
            let i = 0usize;
            let mut p = base.clone();
            p[i] = base[i] + eps;
            trainer.write_weight(&name, &p);
            let lp = trainer.loss();
            p[i] = base[i] - eps;
            trainer.write_weight(&name, &p);
            let lm = trainer.loss();
            trainer.write_weight(&name, &base);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - ana[i]).abs() < 3e-2 + 3e-2 * num.abs().max(ana[i].abs()),
                "{name}[{i}]: numeric={num} analytic={} (loss+={lp} loss-={lm})",
                ana[i]
            );
            checked += 1;
        }
        assert!(checked > 20, "expected every named DiT parameter to be checked, got {checked}");
    }

    #[test]
    fn overfits_a_single_batch() {
        let cfg = DitConfig::tiny();
        let (w, latents, condition, timestep, length, target) = fixture(&cfg, 61);

        let trainer = Trainer::new(cfg, &w, latents, condition, timestep, length, target);
        let names = trainer.param_names();
        let lr = 0.08f32;
        let steps = 1500;

        let loss0 = trainer.loss();
        let mut loss = loss0;
        for _ in 0..steps {
            trainer.zero_grads();
            loss = trainer.loss();
            trainer.backward();
            for name in &names {
                let mut wv = trainer.read_weight(name);
                let g = trainer.read_grad(name);
                for (wi, gi) in wv.iter_mut().zip(&g) {
                    *wi -= lr * gi;
                }
                trainer.write_weight(name, &wv);
            }
        }
        assert!(loss < loss0 * 0.05, "loss did not collapse: start={loss0} end={loss} ({steps} steps, lr={lr})");
    }
}
