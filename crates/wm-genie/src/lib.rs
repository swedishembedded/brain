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

pub mod bias;
pub mod import;

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
    pub const BIAS_ADD: usize = 19;
    pub const EMBED: usize = 20;
    pub const VQ_ARGMAX_DOT: usize = 21;
    pub const GELU_ERF: usize = 22;
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
    v.push(("bias_add", kernels::BIAS_ADD));
    v.push(("embed", kernels::EMBED));
    v.push(("vq_argmax_dot", kernels::VQ_ARGMAX_DOT));
    v.push(("gelu_erf", kernels::GELU_ERF));
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

    // q from the NORMED x; k,v from the RAW x. GenieRedux captures
    // `kv_input = x` before `x = norm(x)`, so for self-attention (context=None)
    // the key/value projections see the un-normalized input — a real parity
    // detail, not an oversight.
    let xn = layernorm(gpu, &xb, &gamma, &beta, rows, dim);
    let q = matmul(gpu, &xn, &up(&w.to_q), rows, dim, inner);
    let kk = matmul(gpu, &xb, &up(&w.to_k), rows, dim, inner);
    let v = matmul(gpu, &xb, &up(&w.to_v), rows, dim, inner);

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

/// Weights of one GenieRedux `FeedForward` (GEGLU) module. The pre-norm is a
/// standard `nn.LayerNorm` (WITH bias), unlike the attention's custom no-bias
/// norm.
pub struct FfWeights {
    pub norm_gamma: Vec<f32>, // [dim]
    pub norm_beta: Vec<f32>,  // [dim]
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
    let beta = up(&w.norm_beta);

    let xn = layernorm(gpu, &xb, &gamma, &beta, rows, dim);
    let xp = matmul(gpu, &xn, &up(&w.w_x), rows, dim, inner);
    let gate = matmul(gpu, &xn, &up(&w.w_gate), rows, dim, inner);

    // GenieRedux uses torch's exact-erf F.gelu (not the tanh approximation).
    let g = gpu.storage((rows * inner) as u64);
    let act = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[
        gpu.step(k::GELU_ERF, &[&gate, &g], &[rows * inner], rows * inner),
        gpu.step(k::MUL, &[&g, &xp, &act], &[rows * inner], rows * inner),
    ]);

    let y = matmul(gpu, &act, &up(&w.w_out), rows, inner, dim);
    gpu.read(&y, (rows * dim) as usize)
}

// ---- cosine vector quantization (tokenizer bottleneck) ----

/// Weights of the tokenizer VQ (`use_cosine_sim`, codebook_dim 32, 1024 codes):
/// `project_in` (dim→cd, bias), the codebook `[K, cd]`, `project_out` (cd→dim,
/// bias).
pub struct VqWeights {
    pub project_in_w: Vec<f32>, pub project_in_b: Vec<f32>, // [cd,dim], [cd]
    pub codebook: Vec<f32>,                                  // [K, cd]
    pub project_out_w: Vec<f32>, pub project_out_b: Vec<f32>, // [dim,cd], [dim]
}

/// Cosine-similarity VQ forward: `project_in` → L2-normalize → argmax cosine
/// against the L2-normalized codebook → gather `codebook[idx]` → `project_out`.
/// Returns `(quantized [n,dim], indices [n])`. (Straight-through / commitment
/// are training-only and handled by the caller via `wm_core::vq`.)
pub fn vq_quantize(gpu: &Gpu, x: &[f32], w: &VqWeights, n: u32, dim: u32, code_dim: u32, n_codes: u32) -> (Vec<f32>, Vec<u32>) {
    let att = biased_attn();
    let ones = vec![1.0f32; code_dim as usize];
    let onesb = gpu.storage_init("ones", &ones);

    // project_in -> [n, cd]
    let xb = gpu.storage_init("x", x);
    let piw = gpu.storage_init("piw", &w.project_in_w);
    let pib = gpu.storage_init("pib", &w.project_in_b);
    let z = linear_bias(gpu, &xb, &piw, &pib, n, dim, code_dim);
    // transform_input = L2-normalize the projected input ONLY (g = ones). The
    // codebook is used RAW: the reference cosine codebook picks argmax of
    // (l2norm(z) · embed_raw) — the stored embed is only ~unit-norm, so
    // normalizing it too would change which code wins.
    let zn = gpu.storage((n * code_dim) as u64);
    gpu.submit(&[], &[att.step_l2norm(gpu, n, code_dim, 0.0, &z, &onesb, &zn)]);
    let cb = gpu.storage_init("cb", &w.codebook);
    // argmax (l2norm(z) · embed_raw) -> packed [idx, dot] per query
    let packed = gpu.storage((2 * n) as u64);
    gpu.submit(&[], &[gpu.step(k::VQ_ARGMAX_DOT, &[&zn, &cb, &packed], &[n, n_codes, code_dim], n)]);
    let packed_v = gpu.read(&packed, (2 * n) as usize);
    let indices: Vec<u32> = packed_v.chunks_exact(2).map(|c| c[0] as u32).collect();

    // gather codebook[idx] (raw) then project_out -> [n, dim]
    let idxb = gpu.storage(n as u64);
    gpu.write(&idxb, &indices);
    let q = gpu.storage((n * code_dim) as u64);
    gpu.submit(&[], &[gpu.step(k::EMBED, &[&idxb, &cb, &q], &[code_dim, n], n * code_dim)]);
    let pow = gpu.storage_init("pow", &w.project_out_w);
    let pob = gpu.storage_init("pob", &w.project_out_b);
    let out = linear_bias(gpu, &q, &pow, &pob, n, code_dim, dim);
    (gpu.read(&out, (n * dim) as usize), indices)
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
/// = ALiBi `[heads,t,t]`). Each of the six stages is residual. `temporal_first`
/// selects the order: `false` = "st" (encoder), `true` = "ts" (decoder). PEGs
/// are causal (temporal pad `(2,0)`). Matches `STBlock.{spatial_temporal,
/// temporal_spatial}_forward`.
#[allow(clippy::too_many_arguments)]
pub fn stblock_forward(
    gpu: &Gpu, x: &[f32], b: u32, t: u32, h: u32, w: u32, dim: u32, heads: u32, head_dim: u32,
    wts: &StBlockWeights, spatial_bias: &[f32], temporal_bias: &[f32], peg_causal: bool,
    temporal_first: bool,
) -> Vec<f32> {
    let (bu, tu, hu, wu, du) = (b as usize, t as usize, h as usize, w as usize, dim as usize);
    let hw = h * w;

    // Spatial half: rows folded as (b t)(h w), directly contiguous.
    let spatial = |xin: &[f32]| -> Vec<f32> {
        let mut xs = xin.to_vec();
        let p = peg_forward_w(gpu, &xs, &wts.spatial_peg, b, t, h, w, dim, peg_causal);
        xs = add(&xs, &p);
        let a = attn_forward(gpu, &xs, b * t, hw, dim, heads, head_dim, &wts.spatial_attn, spatial_bias, false);
        xs = add(&xs, &a);
        let f = geglu_forward(gpu, &xs, b * t * hw, dim, ff_inner(dim), &wts.spatial_ff);
        add(&xs, &f)
    };
    // Temporal half: PEG on the same video, then causal attention over t
    // (rows re-folded as (b h w) t).
    let temporal = |xin: &[f32]| -> Vec<f32> {
        let mut xs = xin.to_vec();
        let p = peg_forward_w(gpu, &xs, &wts.temporal_peg, b, t, h, w, dim, peg_causal);
        xs = add(&xs, &p);
        let mut xt = to_temporal(&xs, bu, tu, hu, wu, du);
        let a = attn_forward(gpu, &xt, b * h * w, t, dim, heads, head_dim, &wts.temporal_attn, temporal_bias, true);
        xt = add(&xt, &a);
        let f = geglu_forward(gpu, &xt, b * h * w * t, dim, ff_inner(dim), &wts.temporal_ff);
        xt = add(&xt, &f);
        from_temporal(&xt, bu, tu, hu, wu, du)
    };

    if temporal_first {
        spatial(&temporal(x))
    } else {
        temporal(&spatial(x))
    }
}

/// GEGLU inner dim used by GenieRedux: `round(dim * 4 * 2/3)`.
pub fn ff_inner(dim: u32) -> u32 {
    ((dim as f64) * 4.0 * 2.0 / 3.0) as u32
}

// ---- patch embed / unpatch (the video <-> token boundary) ----

/// `nn.Linear` with bias: `y = x @ w^T + b`. (The attention/FF linears are
/// bias-free; patch-embed, to_pixels and to_logits carry a bias.)
fn linear_bias(gpu: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, b: &DeviceBuffer, m: u32, kk: u32, n: u32) -> DeviceBuffer {
    let out = matmul(gpu, x, w, m, kk, n);
    gpu.submit(&[], &[gpu.step(k::BIAS_ADD, &[&out, b], &[m, n], m * n)]);
    out
}
fn layernorm_b(gpu: &Gpu, x: &[f32], g: &[f32], beta: &[f32], rows: u32, dim: u32) -> DeviceBuffer {
    let xb = gpu.storage_init("x", x);
    let gb = gpu.storage_init("g", g);
    let bb = gpu.storage_init("b", beta);
    layernorm(gpu, &xb, &gb, &bb, rows, dim)
}

/// Weights of a `to_patch_emb[_first_frame]`: LayerNorm(pf) → Linear(pf→dim) →
/// LayerNorm(dim), all with bias (`nn.LayerNorm`/`nn.Linear`). `pf` = the patch
/// feature count = channels·patch·patch (48 for 4×4×3).
pub struct PatchEmbedWeights {
    pub ln1_g: Vec<f32>, pub ln1_b: Vec<f32>, // [pf]
    pub lin_w: Vec<f32>, pub lin_b: Vec<f32>, // [dim,pf], [dim]
    pub ln2_g: Vec<f32>, pub ln2_b: Vec<f32>, // [dim]
}

/// Patchify a channels-first video `[b,c,t,H,W]` into `[b,t,h,w,pf]` (patch
/// `p×p`, `pf = c*p*p`, feature order `(c p1 p2)`), matching the einops
/// `b c t (h p1) (w p2) -> b t h w (c p1 p2)` (temporal_patch_size = 1).
fn patchify(video: &[f32], b: usize, c: usize, t: usize, hh: usize, ww: usize, p: usize) -> (Vec<f32>, usize, usize) {
    let (h, w) = (hh / p, ww / p);
    let pf = c * p * p;
    let mut out = vec![0.0f32; b * t * h * w * pf];
    for bb in 0..b { for tt in 0..t { for hy in 0..h { for wx in 0..w {
        for cc in 0..c { for p1 in 0..p { for p2 in 0..p {
            let src = ((((bb*c+cc)*t+tt)*hh + hy*p+p1)*ww) + wx*p+p2;
            let pidx = (cc*p + p1)*p + p2;
            let dst = ((((bb*t+tt)*h+hy)*w+wx)*pf) + pidx;
            out[dst] = video[src];
        }}}
    }}}}
    (out, h, w)
}

/// `to_patch_emb`: patchify `[b,c,t,H,W]` then LN→Linear→LN → `[b,t,h,w,dim]`.
#[allow(clippy::too_many_arguments)]
pub fn patch_embed(gpu: &Gpu, video: &[f32], w: &PatchEmbedWeights, b: u32, c: u32, t: u32, hh: u32, ww: u32, p: u32, dim: u32) -> Vec<f32> {
    let (patches, h, wd) = patchify(video, b as usize, c as usize, t as usize, hh as usize, ww as usize, p as usize);
    let pf = (c * p * p) as u32;
    let rows = b * t * h as u32 * wd as u32;
    let n1 = layernorm_b(gpu, &patches, &w.ln1_g, &w.ln1_b, rows, pf);
    let lw = gpu.storage_init("lw", &w.lin_w);
    let lb = gpu.storage_init("lb", &w.lin_b);
    let lin = linear_bias(gpu, &n1, &lw, &lb, rows, pf, dim);
    let lin_v = gpu.read(&lin, (rows * dim) as usize);
    let n2 = layernorm_b(gpu, &lin_v, &w.ln2_g, &w.ln2_b, rows, dim);
    gpu.read(&n2, (rows * dim) as usize)
}

/// Weights of a `to_pixels[_first_frame]`: `Linear(dim→pf)` (with bias).
pub struct ToPixelsWeights {
    pub lin_w: Vec<f32>, // [pf, dim]
    pub lin_b: Vec<f32>, // [pf]
}

/// `to_pixels`: Linear then unpatch `[b,t,h,w,dim]` → video `[b,c,t,H,W]`
/// (inverse layout of [`patchify`]).
#[allow(clippy::too_many_arguments)]
pub fn to_pixels(gpu: &Gpu, tokens: &[f32], w: &ToPixelsWeights, b: u32, t: u32, h: u32, wd: u32, dim: u32, c: u32, p: u32) -> Vec<f32> {
    let (bu, tu, hu, wu, cu, pu) = (b as usize, t as usize, h as usize, wd as usize, c as usize, p as usize);
    let pf = cu * pu * pu;
    let rows = b * t * h * wd;
    let xb = gpu.storage_init("x", tokens);
    let lw = gpu.storage_init("lw", &w.lin_w);
    let lb = gpu.storage_init("lb", &w.lin_b);
    let lin = linear_bias(gpu, &xb, &lw, &lb, rows, dim, pf as u32);
    let patched = gpu.read(&lin, (rows as usize) * pf);
    // unpatch to [b,c,t,H,W]
    let (hh, ww) = (hu * pu, wu * pu);
    let mut video = vec![0.0f32; bu*cu*tu*hh*ww];
    for bb in 0..bu { for tt in 0..tu { for hy in 0..hu { for wx in 0..wu {
        for cc in 0..cu { for p1 in 0..pu { for p2 in 0..pu {
            let pidx = (cc*pu + p1)*pu + p2;
            let src = ((((bb*tu+tt)*hu+hy)*wu+wx)*pf) + pidx;
            let dst = ((((bb*cu+cc)*tu+tt)*hh + hy*pu+p1)*ww) + wx*pu+p2;
            video[dst] = patched[src];
        }}}
    }}}}
    video
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

/// Split a channels-first video `[b,c,f,H,W]` into frame-range `[b,c,t,H,W]`.
fn frame_slice(video: &[f32], b: usize, c: usize, f: usize, hh: usize, ww: usize, t0: usize, t1: usize) -> Vec<f32> {
    let t = t1 - t0;
    let mut o = vec![0.0f32; b*c*t*hh*ww];
    for bb in 0..b { for cc in 0..c { for tt in 0..t { for p in 0..hh*ww {
        o[(((bb*c+cc)*t+tt)*hh*ww)+p] = video[(((bb*c+cc)*f+(t0+tt))*hh*ww)+p];
    }}}}
    o
}

/// Weights of the full ST-ViViT tokenizer (encode → VQ → decode).
pub struct TokenizerWeights {
    pub patch_first: PatchEmbedWeights,
    pub patch_rest: PatchEmbedWeights,
    pub encoder: StTransformerWeights, // 8 blocks, order "st"
    pub vq: VqWeights,
    pub decoder: StTransformerWeights, // 8 blocks, order "ts"
    pub to_pixels_first: ToPixelsWeights,
    pub to_pixels_rest: ToPixelsWeights,
    pub cpb_net: Vec<bias::CpbLayer>,  // spatial ContinuousPositionBias MLP
}

/// Full tokenizer forward: reconstruct a channels-first video `[b,c,f,H,W]`
/// through patch-embed (first frame + rest, separate weights) → encoder(8,"st")
/// → cosine VQ → decoder(8,"ts") → to_pixels. Returns `(recon [b,c,f,H,W],
/// codebook indices [b*f*h*w])`.
#[allow(clippy::too_many_arguments)]
pub fn tokenizer_forward(
    gpu: &Gpu, video: &[f32], w: &TokenizerWeights,
    b: u32, c: u32, f: u32, hh: u32, ww: u32, p: u32, dim: u32, heads: u32, head_dim: u32,
    code_dim: u32, n_codes: u32,
) -> (Vec<f32>, Vec<u32>) {
    let (bu, cu, fu, hhu, wwu, pu) = (b as usize, c as usize, f as usize, hh as usize, ww as usize, p as usize);
    let (h, wd) = ((hhu / pu) as u32, (wwu / pu) as u32);

    // patch-embed first frame + rest, concat along t -> [b,f,h,w,dim]
    let first = frame_slice(video, bu, cu, fu, hhu, wwu, 0, 1);
    let rest = frame_slice(video, bu, cu, fu, hhu, wwu, 1, fu);
    let tok_first = patch_embed(gpu, &first, &w.patch_first, b, c, 1, hh, ww, p, dim);
    let tok_rest = patch_embed(gpu, &rest, &w.patch_rest, b, c, f - 1, hh, ww, p, dim);
    let frame_toklen = (h * wd * dim) as usize;
    let mut tokens = vec![0.0f32; bu * fu * frame_toklen];
    for bb in 0..bu {
        let dst = bb * fu * frame_toklen;
        tokens[dst..dst + frame_toklen].copy_from_slice(&tok_first[bb * frame_toklen..(bb + 1) * frame_toklen]);
        let src = bb * (fu - 1) * frame_toklen;
        tokens[dst + frame_toklen..dst + fu * frame_toklen]
            .copy_from_slice(&tok_rest[src..src + (fu - 1) * frame_toklen]);
    }

    // position biases + encode (st)
    let sbias = bias::cpb_bias(&w.cpb_net, h as usize, wd as usize, heads as usize);
    let tbias = bias::alibi_bias(heads as usize, fu);
    let enc = sttransformer_forward(gpu, &tokens, b, f, h, wd, dim, heads, head_dim, &w.encoder, &sbias, &tbias, true, false);

    // VQ over all tokens
    let n = b * f * h * wd;
    let (quant, indices) = vq_quantize(gpu, &enc, &w.vq, n, dim, code_dim, n_codes);

    // decode (ts) then to_pixels first/rest
    let dec = sttransformer_forward(gpu, &quant, b, f, h, wd, dim, heads, head_dim, &w.decoder, &sbias, &tbias, true, true);
    let dfirst = {
        let mut v = vec![0.0f32; bu * frame_toklen];
        for bb in 0..bu { v[bb*frame_toklen..(bb+1)*frame_toklen].copy_from_slice(&dec[bb*fu*frame_toklen..bb*fu*frame_toklen+frame_toklen]); }
        v
    };
    let drest = {
        let mut v = vec![0.0f32; bu * (fu-1) * frame_toklen];
        for bb in 0..bu { v[bb*(fu-1)*frame_toklen..(bb+1)*(fu-1)*frame_toklen].copy_from_slice(&dec[bb*fu*frame_toklen+frame_toklen..(bb+1)*fu*frame_toklen]); }
        v
    };
    let pix_first = to_pixels(gpu, &dfirst, &w.to_pixels_first, b, 1, h, wd, dim, c, p);
    let pix_rest = to_pixels(gpu, &drest, &w.to_pixels_rest, b, f - 1, h, wd, dim, c, p);
    // concat along the frame axis -> [b,c,f,H,W]
    let mut recon = vec![0.0f32; bu*cu*fu*hhu*wwu];
    for bb in 0..bu { for cc in 0..cu {
        let dst0 = ((bb*cu+cc)*fu)*hhu*wwu;
        let sf = ((bb*cu+cc)*1)*hhu*wwu;
        recon[dst0..dst0+hhu*wwu].copy_from_slice(&pix_first[sf..sf+hhu*wwu]);
        for tt in 0..fu-1 {
            let src = ((bb*cu+cc)*(fu-1)+tt)*hhu*wwu;
            let dst = ((bb*cu+cc)*fu+(tt+1))*hhu*wwu;
            recon[dst..dst+hhu*wwu].copy_from_slice(&pix_rest[src..src+hhu*wwu]);
        }
    }}
    (recon, indices)
}

/// Run the block stack over a channels-last `[b,t,h,w,dim]` video and apply the
/// final LayerNorm (no bias). Returns `[b,t,h,w,dim]`.
#[allow(clippy::too_many_arguments)]
pub fn sttransformer_forward(
    gpu: &Gpu, x: &[f32], b: u32, t: u32, h: u32, w: u32, dim: u32, heads: u32, head_dim: u32,
    wts: &StTransformerWeights, spatial_bias: &[f32], temporal_bias: &[f32], peg_causal: bool,
    temporal_first: bool,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for blk in &wts.layers {
        cur = stblock_forward(gpu, &cur, b, t, h, w, dim, heads, head_dim, blk, spatial_bias, temporal_bias, peg_causal, temporal_first);
    }
    let rows = b * t * h * w;
    let xb = gpu.storage_init("x", &cur);
    let g = gpu.storage_init("g", &wts.norm_out_gamma);
    let beta = gpu.storage(dim as u64);
    let out = layernorm(gpu, &xb, &g, &beta, rows, dim);
    gpu.read(&out, (rows * dim) as usize)
}
