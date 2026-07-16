// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GenieRedux-G (CoinRun) world model — the shared ST-transformer building
//! blocks, reimplemented from scratch over brain's WGSL kernels and verified
//! against the reference forward (`GenieRedux/models/components/attention.py`).
//!
//! This module provides the two learnable sub-modules every STBlock is built
//! from (the third, PEG, is [`kernels::DWCONV3D`]):
//!
//! - [`attn_forward`] — the GenieRedux `Attention` (num_null_kv = 0): pre-norm,
//!   fused `to_kv`, QK-normalization (L2 over head_dim × learnable per-dim
//!   scale) and a constant score scale of 8, plus an additive per-head bias
//!   (spatial ContinuousPositionBias or temporal ALiBi, supplied by the caller)
//!   and optional causal masking.
//! - [`geglu_forward`] — the GenieRedux `FeedForward`: pre-norm → GEGLU
//!   (`gelu(gate) * x`) → out-projection.
//!
//! Tensor layout matches the rest of brain: rows are the folded batch×sequence
//! (`R = B*n`), features are the last axis. The attention's "batch" `B` is the
//! einops fold (b·t for spatial, b·h·w for temporal) and its "sequence" `n` is
//! the attended axis (h·w spatial, t temporal).

use gpu_core::{DeviceBuffer, Gpu};
use wm_core::attn::BiasedAttn;

/// Kernel-table indices used by this crate. Load [`kernel_sources`] at offset 0
/// and these line up; the biased-attention slice (indices 5..17) also backs
/// [`biased_attn`].
pub mod k {
    pub const LAYERNORM: usize = 0;
    pub const MATMUL: usize = 1;
    pub const CONCAT2: usize = 2;
    pub const GELU: usize = 3;
    pub const MUL: usize = 4;
    // 5..17 == BiasedAttn (see `biased_attn`)
    pub const DWCONV3D: usize = 17;
    pub const ADD2: usize = 18;
}

/// `(name, source)` for `Gpu::new*`, in the index order `k` / [`biased_attn`].
pub fn kernel_sources() -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&str, &str)> = vec![
        ("layernorm", kernels::LAYERNORM),
        ("matmul", kernels::MATMUL),
        ("concat2", kernels::CONCAT2),
        ("gelu", kernels::GELU),
        ("mul", kernels::MUL),
    ];
    v.extend_from_slice(&BiasedAttn::kernel_sources());
    v.push(("dwconv3d", kernels::DWCONV3D));
    v.push(("add2", kernels::ADD2));
    v
}

/// The biased-attention helper over this crate's kernel table (offset 5).
pub fn biased_attn() -> BiasedAttn {
    BiasedAttn {
        l2norm: 5, l2norm_dx: 6, l2norm_dg: 7, scores_bidir: 8, scores_causal: 9,
        softmax: 10, apply: 11, dscores: 12, dv: 13, dq: 14, dk: 15, dbias: 16,
    }
}

/// Weights of one GenieRedux `Attention` module (num_null_kv = 0, no biases on
/// the linears — matching the checkpoint). Shapes in row-major torch order.
pub struct AttnWeights {
    pub norm_gamma: Vec<f32>, // [dim]
    pub to_q: Vec<f32>,       // [inner, dim]   inner = heads*head_dim
    pub to_k: Vec<f32>,       // [inner, dim]   (the first half of to_kv)
    pub to_v: Vec<f32>,       // [inner, dim]   (the second half of to_kv)
    pub to_out: Vec<f32>,     // [dim, inner]
    pub q_scale: Vec<f32>,    // [head_dim]
    pub k_scale: Vec<f32>,    // [head_dim]
}

fn layernorm(gpu: &Gpu, x: &DeviceBuffer, gamma: &DeviceBuffer, beta: &DeviceBuffer, rows: u32, dim: u32) -> DeviceBuffer {
    let out = gpu.storage((rows * dim) as u64);
    gpu.submit(&[], &[gpu.step(k::LAYERNORM, &[x, gamma, beta, &out], &[dim, rows], rows)]);
    out
}

fn matmul(gpu: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, m: u32, kk: u32, n: u32) -> DeviceBuffer {
    let out = gpu.storage((m * n) as u64);
    gpu.submit(&[], &[gpu.step(k::MATMUL, &[x, w, &out], &[m, kk, n], m * n)]);
    out
}

/// Row-wise feature concat `[rows, ca] ++ [rows, cb] -> [rows, ca+cb]` via the
/// [N,C,1,1] channel-concat kernel.
fn concat_cols(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, rows: u32, ca: u32, cb: u32) -> DeviceBuffer {
    let out = gpu.storage((rows * (ca + cb)) as u64);
    gpu.submit(&[], &[gpu.step(k::CONCAT2, &[a, b, &out], &[rows, ca, cb, 1, 1], rows * (ca + cb))]);
    out
}

/// GenieRedux `Attention.forward` for num_null_kv = 0.
///
/// * `x`     — input, `R*dim` row-major (`R = b*n`).
/// * `bias`  — additive per-head score bias `heads*n*n` (ContinuousPositionBias
///   for spatial / ALiBi for temporal), shared across `b`.
/// * `causal` — temporal attention masks `j>i`; spatial passes `false`.
///
/// Returns the output, `R*dim`.
#[allow(clippy::too_many_arguments)]
pub fn attn_forward(
    gpu: &Gpu, x: &[f32], b: u32, n: u32, dim: u32, heads: u32, head_dim: u32,
    w: &AttnWeights, bias: &[f32], causal: bool,
) -> Vec<f32> {
    let att = biased_attn();
    let inner = heads * head_dim;
    let rows = b * n;
    const SCALE: f32 = 8.0;
    const EPS: f32 = 1e-6;

    let up = |d: &[f32]| gpu.storage_init("w", d);
    let xb = up(x);
    let gamma = up(&w.norm_gamma);
    let beta = gpu.storage(dim as u64); // LayerNorm bias is a fixed zero buffer

    // pre-norm, then project
    let xn = layernorm(gpu, &xb, &gamma, &beta, rows, dim);
    let q = matmul(gpu, &xn, &up(&w.to_q), rows, dim, inner);
    let kk = matmul(gpu, &xn, &up(&w.to_k), rows, dim, inner);
    let v = matmul(gpu, &xn, &up(&w.to_v), rows, dim, inner);

    // QK-norm: L2 over head_dim (rows*heads slices) times per-dim scale.
    let qn = gpu.storage((rows * inner) as u64);
    let kn = gpu.storage((rows * inner) as u64);
    let qs = up(&w.q_scale);
    let ks = up(&w.k_scale);
    gpu.submit(&[], &[
        att.step_l2norm(gpu, rows * heads, head_dim, EPS, &q, &qs, &qn),
        att.step_l2norm(gpu, rows * heads, head_dim, EPS, &kk, &ks, &kn),
    ]);

    // pack fused [rows, 3*inner] = [qn | kn | v]
    let qk = concat_cols(gpu, &qn, &kn, rows, inner, inner);
    let packed = concat_cols(gpu, &qk, &v, rows, 2 * inner, inner);

    let bias_b = up(bias);
    let scores = gpu.storage((b * heads * n * n) as u64);
    let probs = gpu.storage((b * heads * n * n) as u64);
    let ao = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[
        att.step_scores(gpu, b, heads, n, head_dim, SCALE, causal, &packed, &bias_b, &scores),
        att.step_softmax(gpu, b, heads, n, &scores, &probs),
        att.step_apply(gpu, b, heads, n, head_dim, &probs, &packed, &ao),
    ]);

    // out-projection
    let y = matmul(gpu, &ao, &up(&w.to_out), rows, inner, dim);
    gpu.read(&y, (rows * dim) as usize)
}

// ---- PEG (position-encoding generator) ----

/// Weights of one PEG module: a depthwise Conv3d(dim,dim,3,groups=dim).
pub struct PegWeights {
    pub dsconv: Vec<f32>, // [dim, 1, 3, 3, 3] == [dim, 27] per-channel kernels
    pub bias: Vec<f32>,   // [dim]
}

/// PEG forward on a channels-LAST video `x` of shape `[b,t,h,w,dim]` (flat).
/// Depthwise 3×3×3 conv with spatial pad 1 and temporal pad `(2,0)` when
/// `causal` (matching `PEG(causal=True)`) else `(1,1)`; output shape unchanged.
/// Returns the convolved video (NOT the residual — the caller adds `+ x`).
pub fn peg_forward_w(gpu: &Gpu, x: &[f32], w: &PegWeights, b: u32, t: u32, h: u32, wd: u32, dim: u32, causal: bool) -> Vec<f32> {
    let (bu, tu, hu, wu, du) = (b as usize, t as usize, h as usize, wd as usize, dim as usize);
    let mut cf = vec![0.0f32; x.len()];
    for bb in 0..bu { for tt in 0..tu { for hh in 0..hu { for ww in 0..wu { for d in 0..du {
        let src = ((((bb*tu+tt)*hu+hh)*wu+ww)*du)+d;
        let dst = ((((bb*du+d)*tu+tt)*hu+hh)*wu)+ww;
        cf[dst] = x[src];
    }}}}}
    let xb = gpu.storage_init("x", &cf);
    let wtb = gpu.storage_init("wt", &w.dsconv);
    let bb_ = gpu.storage_init("b", &w.bias);
    let yb = gpu.storage((x.len()) as u64);
    let pt = if causal { 2u32 } else { 1u32 };
    let params = [b, dim, t, h, wd, 3, 1, pt];
    gpu.submit(&[], &[gpu.step(k::DWCONV3D, &[&xb, &wtb, &bb_, &yb], &params, (bu*du*tu*hu*wu) as u32)]);
    let cf_out = gpu.read(&yb, x.len());
    // channels-first [b,d,t,h,w] -> channels-last [b,t,h,w,d].
    let mut out = vec![0.0f32; x.len()];
    for bb in 0..bu { for d in 0..du { for tt in 0..tu { for hh in 0..hu { for ww in 0..wu {
        let src = ((((bb*du+d)*tu+tt)*hu+hh)*wu)+ww;
        let dst = ((((bb*tu+tt)*hu+hh)*wu+ww)*du)+d;
        out[dst] = cf_out[src];
    }}}}}
    out
}

/// Weights of one GenieRedux `FeedForward` (GEGLU) module.
pub struct FfWeights {
    pub norm_gamma: Vec<f32>, // [dim]
    pub w_x: Vec<f32>,        // [inner, dim]   first chunk of the in-proj
    pub w_gate: Vec<f32>,     // [inner, dim]   second chunk
    pub w_out: Vec<f32>,      // [dim, inner]
}

/// GenieRedux `FeedForward.forward`: pre-norm → GEGLU(`gelu(gate)*x`) → out.
/// `x` is `R*dim`; returns `R*dim`. `inner = round(dim * 4 * 2/3)`.
pub fn geglu_forward(gpu: &Gpu, x: &[f32], rows: u32, dim: u32, inner: u32, w: &FfWeights) -> Vec<f32> {
    let up = |d: &[f32]| gpu.storage_init("w", d);
    let xb = up(x);
    let gamma = up(&w.norm_gamma);
    let beta = gpu.storage(dim as u64);

    let xn = layernorm(gpu, &xb, &gamma, &beta, rows, dim);
    let xp = matmul(gpu, &xn, &up(&w.w_x), rows, dim, inner);
    let gate = matmul(gpu, &xn, &up(&w.w_gate), rows, dim, inner);

    let g = gpu.storage((rows * inner) as u64);
    let act = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[
        gpu.step(k::GELU, &[&gate, &g], &[rows * inner], rows * inner),
        gpu.step(k::MUL, &[&g, &xp, &act], &[rows * inner], rows * inner),
    ]);

    let y = matmul(gpu, &act, &up(&w.w_out), rows, inner, dim);
    gpu.read(&y, (rows * dim) as usize)
}

// ---- full STBlock ----

/// Weights of one STBlock: spatial then temporal, each `PEG → Attention → FF`.
pub struct StBlockWeights {
    pub spatial_peg: PegWeights,
    pub spatial_attn: AttnWeights,
    pub spatial_ff: FfWeights,
    pub temporal_peg: PegWeights,
    pub temporal_attn: AttnWeights,
    pub temporal_ff: FfWeights,
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Permute a channels-last video between the spatial fold `[b,t,h,w,d]` (rows
/// grouped as `(b t)(h w)`) and the temporal fold `[b,h,w,t,d]` (`(b h w) t`).
fn to_temporal(x: &[f32], b: usize, t: usize, h: usize, w: usize, d: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; x.len()];
    for bb in 0..b { for tt in 0..t { for hh in 0..h { for ww in 0..w {
        let src = (((bb*t+tt)*h+hh)*w+ww)*d;
        let dst = (((bb*h+hh)*w+ww)*t+tt)*d;
        o[dst..dst+d].copy_from_slice(&x[src..src+d]);
    }}}}
    o
}
fn from_temporal(x: &[f32], b: usize, t: usize, h: usize, w: usize, d: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; x.len()];
    for bb in 0..b { for hh in 0..h { for ww in 0..w { for tt in 0..t {
        let src = (((bb*h+hh)*w+ww)*t+tt)*d;
        let dst = (((bb*t+tt)*h+hh)*w+ww)*d;
        o[dst..dst+d].copy_from_slice(&x[src..src+d]);
    }}}}
    o
}

/// One STBlock forward over a channels-last video `x` `[b,t,h,w,dim]`.
///
/// Spatial sub-block attends over the `h·w` tokens of each frame
/// (bidirectional; `spatial_bias` = ContinuousPositionBias `[heads,hw,hw]`);
/// temporal attends over the `t` frames at each location (causal; `temporal_bias`
/// = ALiBi `[heads,t,t]`). Each of the six stages is residual, matching
/// `STBlock.spatial_temporal_forward`. PEGs are causal (temporal pad `(2,0)`).
#[allow(clippy::too_many_arguments)]
pub fn stblock_forward(
    gpu: &Gpu, x: &[f32], b: u32, t: u32, h: u32, w: u32, dim: u32, heads: u32, head_dim: u32,
    wts: &StBlockWeights, spatial_bias: &[f32], temporal_bias: &[f32], peg_causal: bool,
) -> Vec<f32> {
    let (bu, tu, hu, wu, du) = (b as usize, t as usize, h as usize, w as usize, dim as usize);
    let hw = h * w;

    // --- spatial: rows folded as (b t)(h w), directly contiguous ---
    let mut xs = x.to_vec();
    let p = peg_forward_w(gpu, &xs, &wts.spatial_peg, b, t, h, w, dim, peg_causal);
    xs = add(&xs, &p);
    let a = attn_forward(gpu, &xs, b * t, hw, dim, heads, head_dim, &wts.spatial_attn, spatial_bias, false);
    xs = add(&xs, &a);
    let f = geglu_forward(gpu, &xs, b * t * hw, dim, ff_inner(dim), &wts.spatial_ff);
    xs = add(&xs, &f);

    // --- temporal: PEG on the same video, then attention over t ---
    let p = peg_forward_w(gpu, &xs, &wts.temporal_peg, b, t, h, w, dim, peg_causal);
    xs = add(&xs, &p);
    let mut xt = to_temporal(&xs, bu, tu, hu, wu, du);
    let a = attn_forward(gpu, &xt, b * h * w, t, dim, heads, head_dim, &wts.temporal_attn, temporal_bias, true);
    xt = add(&xt, &a);
    let f = geglu_forward(gpu, &xt, b * h * w * t, dim, ff_inner(dim), &wts.temporal_ff);
    xt = add(&xt, &f);
    from_temporal(&xt, bu, tu, hu, wu, du)
}

/// GEGLU inner dim used by GenieRedux: `round(dim * 4 * 2/3)`.
pub fn ff_inner(dim: u32) -> u32 {
    ((dim as f64) * 4.0 * 2.0 / 3.0) as u32
}

// ---- STTransformer (a stack of STBlocks + final LayerNorm) ----

/// An STTransformer: `layers` STBlocks then `norm_out` — the body of the
/// tokenizer encoder/decoder (8 blocks each) and the dynamics MaskGIT
/// transformer (12 blocks). The spatial/temporal position biases are shared
/// across layers (a fixed function of the grid), so they are supplied once.
pub struct StTransformerWeights {
    pub layers: Vec<StBlockWeights>,
    pub norm_out_gamma: Vec<f32>, // [dim]
}

/// Run the block stack over a channels-last `[b,t,h,w,dim]` video and apply the
/// final LayerNorm (no bias). Returns `[b,t,h,w,dim]`.
#[allow(clippy::too_many_arguments)]
pub fn sttransformer_forward(
    gpu: &Gpu, x: &[f32], b: u32, t: u32, h: u32, w: u32, dim: u32, heads: u32, head_dim: u32,
    wts: &StTransformerWeights, spatial_bias: &[f32], temporal_bias: &[f32], peg_causal: bool,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for blk in &wts.layers {
        cur = stblock_forward(gpu, &cur, b, t, h, w, dim, heads, head_dim, blk, spatial_bias, temporal_bias, peg_causal);
    }
    let rows = b * t * h * w;
    let xb = gpu.storage_init("x", &cur);
    let g = gpu.storage_init("g", &wts.norm_out_gamma);
    let beta = gpu.storage(dim as u64);
    let out = layernorm(gpu, &xb, &g, &beta, rows, dim);
    gpu.read(&out, (rows * dim) as usize)
}
