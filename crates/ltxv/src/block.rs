// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `BasicAVTransformerBlock`'s video-only path, as a recorded brain
//! kernel graph - `ltx_core.model.transformer.transformer.
//! BasicAVTransformerBlock.forward`'s `run_vx` branch (the audio branch and
//! the audio<->video cross-attention are entirely absent from this
//! milestone; see `crates/ltxv/src/lib.rs`'s module doc).
//!
//! ## Block order (pinned against source + the golden's real taps, not
//! ## assumed from the class name)
//!
//! 1. **adaLN-single self-attn modulation**: `norm_vx = rms_norm(x, eps) *
//!    (1+scale_msa) + shift_msa` - an UNWEIGHTED RMSNorm (`ada_zero_function`
//!    passes `norm_weights=None`), not the learnable QK-norm below.
//! 2. **Self-attention** (`attn1`): QKV linear (biased) -> QK-RMSNorm
//!    (learnable, full `inner_dim`, NOT per-head - `Attention.q_norm`/
//!    `k_norm` run before the head split) -> RoPE (split/rotate-half, see
//!    `crate::rope`) -> attention -> output projection (biased).
//! 3. **Gated residual + fused re-norm**: `x = x + attn1_out * gate_msa`,
//!    then `x_normed = rms_norm(x, eps)` - again UNWEIGHTED. This is the
//!    ONLY norm between self-attention and text cross-attention (`post_sa_
//!    function` returns both the updated `x` AND this norm in one call) -
//!    there is no separate `norm2` the way a standard PixArt block has one.
//! 4. **Text cross-attention with AdaLN modulation** (`cross_attention_
//!    adaln=true`): the query side is modulated by PER-TOKEN `shift_q`/
//!    `scale_q` (rows 6-7 of the 9-row table) exactly like step 1; the KEY/
//!    VALUE side is modulated by this BLOCK's own static
//!    `prompt_scale_shift_table` (`[2, dim]`, `[shift_kv, scale_kv]`) -
//!    NOT the per-token table, and NOT per-token at all
//!    (`use_prompt_adaln_single=false`, so there is no timestep MLP on this
//!    path; see `crate::config`'s doc). `attn2` gets NO RoPE (`pe=None` at
//!    this call site in the reference - only self-attention rotates).
//!    The cross-attention output is scaled by `gate_q` (row 8) before the
//!    residual add.
//! 5. **MLP sublayer**: `vx_scaled = rms_norm(x, eps) * (1+scale_mlp) +
//!    shift_mlp` (rows 3-4), `GELU(tanh)` FFN (mult=4, `ff_bias=false` for
//!    the real LTX-2.5 config), gated by `gate_mlp` (row 5).
//!
//! Every RMSNorm above that is NOT the learnable QK-norm is the plain
//! `torch.nn.functional.rms_norm(x, weight=None, eps)` form - dispatched
//! here as `rmsnorm_eps` with an all-ones weight buffer, since no kernel in
//! this repo has a "no learnable gain" RMSNorm variant and folding a
//! constant-1 weight into the existing kernel is exact, not an
//! approximation.
//!
//! ## Attention: which kernels, and why
//!
//! Both `attn1` (self, `t_dec == t_enc == T`) and `attn2` (cross, `t_dec ==
//! T`, `t_enc == context_len`) dispatch the SAME generic trio
//! (`attn_scores_cross` / `attn_softmax_cross` / `attn_apply_cross`,
//! `crates/model/src/block.rs`'s `push_cross`/Wan's non-flash path uses the
//! same trio for cross-attention) rather than a self-attention-specific
//! kernel: the trio's `q_stride`/`kv_stride`/`q_off`/`k_off`/`v_off` are
//! plain parameters, not hardcoded to a fused-QKV layout, so passing
//! separate `[T, dim]` Q/K/V buffers at `stride=dim, off=0` for BOTH
//! self- and cross-attention is exactly what the kernel already supports -
//! no new kernel, no fused-QKV packing needed at this token count. Query-
//! chunking (`model::block::chunked_bidir_fwd`'s reason to exist at Wan's
//! 30k-token scale) is not needed here: the whole `[heads, T, T]` (or
//! `[heads, T, context_len]`) score slab is at most `4*8*8*4 = 1024` bytes.
//!
//! ## Which kernel for RoPE, and why
//!
//! See `crate::rope::apply_rope_step`'s doc - `rope_neox` is refuted
//! (analytic angle, no table input); `rope2d`, dispatched once per head,
//! is the match.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;

use crate::config::LtxDitConfig;
use crate::rope::apply_rope_step;

// Kernel-table indices (order matches KERNELS below).
const K_MATMUL: usize = 0;
const K_BIAS_ADD: usize = 1;
const K_RMSNORM_EPS: usize = 2;
const K_GELU: usize = 3;
const K_MUL: usize = 4;
const K_ADD2: usize = 5;
const K_GATE_ROW: usize = 6;
const K_ATTN_SCORES: usize = 7;
const K_ATTN_SOFTMAX: usize = 8;
const K_ATTN_APPLY: usize = 9;
const K_ROPE2D: usize = 10;

/// Every kernel this block dispatches - all pre-existing, all at their
/// documented general contract (see this module's doc for why no new kernel
/// was needed anywhere in the block).
pub const KERNELS: [(&str, &str); 11] = [
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("gelu", kernels::GELU),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("gate_row", kernels::GATE_ROW),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("rope2d", kernels::ROPE2D),
];

fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("ltxv dit: missing weight {name}")).1
}

fn upload(gpu: &Gpu, w: &Tensors, name: &str) -> DeviceBuffer {
    let data = tget(w, name);
    let buf = gpu.storage(data.len() as u64);
    gpu.write_f32(&buf, data);
    buf
}

/// One `Attention` module's weights (`attn1` or `attn2`).
struct AttnWeights {
    wq: DeviceBuffer,
    bq: DeviceBuffer,
    wk: DeviceBuffer,
    bk: DeviceBuffer,
    wv: DeviceBuffer,
    bv: DeviceBuffer,
    wo: DeviceBuffer,
    bo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

impl AttnWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str) -> AttnWeights {
        AttnWeights {
            wq: upload(gpu, w, &format!("{prefix}.to_q.weight")),
            bq: upload(gpu, w, &format!("{prefix}.to_q.bias")),
            wk: upload(gpu, w, &format!("{prefix}.to_k.weight")),
            bk: upload(gpu, w, &format!("{prefix}.to_k.bias")),
            wv: upload(gpu, w, &format!("{prefix}.to_v.weight")),
            bv: upload(gpu, w, &format!("{prefix}.to_v.bias")),
            wo: upload(gpu, w, &format!("{prefix}.to_out.0.weight")),
            bo: upload(gpu, w, &format!("{prefix}.to_out.0.bias")),
            q_norm: upload(gpu, w, &format!("{prefix}.q_norm.weight")),
            k_norm: upload(gpu, w, &format!("{prefix}.k_norm.weight")),
        }
    }
}

/// The FFN's two linears - `net.0.proj` (GELUApprox's inner Linear) and
/// `net.2` (the output Linear). Both bias-free at `ff_bias=false`.
struct FfWeights {
    w1: DeviceBuffer, // [4*dim, dim]
    w2: DeviceBuffer, // [dim, 4*dim]
}

impl FfWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str) -> FfWeights {
        FfWeights { w1: upload(gpu, w, &format!("{prefix}.net.0.proj.weight")), w2: upload(gpu, w, &format!("{prefix}.net.2.weight")) }
    }
}

/// Resident (upload-once) weights of one block.
struct BlockWeights {
    attn1: AttnWeights,
    attn2: AttnWeights,
    ff: FfWeights,
    /// `[9, dim]` host copy - combined with the shared per-token adaLN
    /// output on every forward (`dit::adaln::add_table` at `rows=T`), so
    /// kept on the host rather than uploaded once.
    scale_shift_table: Vec<f32>,
    /// `[2, dim]` host copy - `[shift_kv, scale_kv]`, static per block (see
    /// this module's doc, step 4).
    prompt_scale_shift_table: Vec<f32>,
}

impl BlockWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, dim: usize) -> BlockWeights {
        let sst = tget(w, &format!("{prefix}.scale_shift_table")).to_vec();
        assert_eq!(sst.len(), 9 * dim, "{prefix}.scale_shift_table must be [9, dim]");
        let pst = tget(w, &format!("{prefix}.prompt_scale_shift_table")).to_vec();
        assert_eq!(pst.len(), 2 * dim, "{prefix}.prompt_scale_shift_table must be [2, dim]");
        BlockWeights {
            attn1: AttnWeights::upload(gpu, w, &format!("{prefix}.attn1")),
            attn2: AttnWeights::upload(gpu, w, &format!("{prefix}.attn2")),
            ff: FfWeights::upload(gpu, w, &format!("{prefix}.ff")),
            scale_shift_table: sst,
            prompt_scale_shift_table: pst,
        }
    }
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    gpu.write_f32(buf, data);
}

/// `out = x @ Wᵀ (+ b)`, `x: [m,k]`, `w: [n,k]`, `out: [m,n]`.
fn linear(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, b: Option<&DeviceBuffer>, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    s.push(gpu.step(K_MATMUL, &[x, w, out], &[m, k, n], m * n));
    if let Some(b) = b {
        s.push(gpu.step(K_BIAS_ADD, &[out, b], &[m, n], m * n));
    }
}

/// RMSNorm over the full row width (`dim`, never per-head) - `w` is either a
/// learnable gain (QK-norm) or an all-ones buffer (the "no learnable gain"
/// `ada_zero_function`/`post_sa_function` norms - see this module's doc).
fn rmsnorm(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    s.push(gpu.step(K_RMSNORM_EPS, &[x, w, out], &[dim, rows, f(eps)], rows));
}

fn mul(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_MUL, &[a, b, y], &[n], n));
}

fn add2(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_ADD2, &[a, b, y], &[n], n));
}

/// `y[r,d] = x[r,d] + g[k,d]*h[r,d]`, `k = r/rows_per_cond`. `rows_per_cond=1`
/// (`NC=rows`, one gate row per token) is what makes this kernel - built for
/// Wan's per-FORWARD gate - serve LTX's per-TOKEN gate unchanged; see
/// `dit::adaln::add_table`'s doc for the same "rows encodes the
/// (in)dependence" generalisation one level up the stack.
#[allow(clippy::too_many_arguments)]
fn gate_row(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, g: &DeviceBuffer, h: &DeviceBuffer, y: &DeviceBuffer, rows: u32, dim: u32) {
    s.push(gpu.step(K_GATE_ROW, &[x, g, h, y], &[rows, dim, 1], rows * dim));
}

/// `(1+scale)*rms_norm(x,eps) + shift`, per token - PixArt's `ada_zero_
/// function`. `one_plus_scale`/`shift` are `[rows, dim]` (already `1+scale`
/// baked in on upload - see [`LtxBlock::forward`]).
#[allow(clippy::too_many_arguments)]
fn ada_zero(gpu: &Gpu, s: &mut Vec<Step>, ones: &DeviceBuffer, x: &DeviceBuffer, one_plus_scale: &DeviceBuffer, shift: &DeviceBuffer, tmp: &DeviceBuffer, tmp2: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    rmsnorm(gpu, s, x, ones, tmp, dim, rows, eps);
    mul(gpu, s, tmp, one_plus_scale, tmp2, rows * dim);
    add2(gpu, s, tmp2, shift, out, rows * dim);
}

/// One (self- or cross-)attention call: QKV projections, QK-RMSNorm, optional
/// per-head RoPE, attention, output projection. `q_in`/`kv_in` are `[nq/nk,
/// dim]`; returns the `[nq, dim]` output-projected result.
#[allow(clippy::too_many_arguments)]
fn attention(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &AttnWeights,
    dim: u32,
    heads: u32,
    head_dim: u32,
    q_in: &DeviceBuffer,
    kv_in: &DeviceBuffer,
    nq: u32,
    nk: u32,
    rope: Option<(&[DeviceBuffer], &[DeviceBuffer])>,
    kernel_rope2d: usize,
    eps: f32,
) -> DeviceBuffer {
    let q_pre = gpu.storage((nq * dim) as u64);
    let k_pre = gpu.storage((nk * dim) as u64);
    let v = gpu.storage((nk * dim) as u64);
    linear(gpu, s, q_in, &w.wq, Some(&w.bq), &q_pre, nq, dim, dim);
    linear(gpu, s, kv_in, &w.wk, Some(&w.bk), &k_pre, nk, dim, dim);
    linear(gpu, s, kv_in, &w.wv, Some(&w.bv), &v, nk, dim, dim);

    let q = gpu.storage((nq * dim) as u64);
    let k = gpu.storage((nk * dim) as u64);
    rmsnorm(gpu, s, &q_pre, &w.q_norm, &q, dim, nq, eps);
    rmsnorm(gpu, s, &k_pre, &w.k_norm, &k, dim, nk, eps);

    if let Some((cos_bufs, sin_bufs)) = rope {
        for h in 0..heads {
            let off = h * head_dim;
            s.push(apply_rope_step(gpu, kernel_rope2d, &q, &cos_bufs[h as usize], &sin_bufs[h as usize], nq, head_dim, dim, off));
            s.push(apply_rope_step(gpu, kernel_rope2d, &k, &cos_bufs[h as usize], &sin_bufs[h as usize], nk, head_dim, dim, off));
        }
    }

    let scores = gpu.storage((heads * nq * nk) as u64);
    let probs = gpu.storage((heads * nq * nk) as u64);
    let ctx = gpu.storage((nq * dim) as u64);
    s.push(gpu.step(K_ATTN_SCORES, &[&q, &k, &scores], &[1, heads, nq, nk, head_dim, dim, dim, 0, 0], heads * nq * nk));
    s.push(gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, nq, nk], heads * nq));
    s.push(gpu.step(K_ATTN_APPLY, &[&probs, &v, &ctx], &[1, heads, nq, nk, head_dim, dim, 0, dim], heads * nq * head_dim));

    let out = gpu.storage((nq * dim) as u64);
    linear(gpu, s, &ctx, &w.wo, Some(&w.bo), &out, nq, dim, dim);
    out
}

/// A layer's per-token modulation, sliced from the combined `[T, 9, dim]`
/// table (`dit::adaln::add_table(adaln_table, block.scale_shift_table,
/// rows=T, width=9*dim)`) - row order `shift,scale,gate` for self-attn
/// (0-2), MLP (3-5), then the cross-attention-adaln path (6-8), pinned
/// against `transformer.py`'s `get_ada_values` call sites (see this
/// module's doc).
struct Mod {
    shift_msa: Vec<f32>,
    one_plus_scale_msa: Vec<f32>,
    gate_msa: Vec<f32>,
    shift_mlp: Vec<f32>,
    one_plus_scale_mlp: Vec<f32>,
    gate_mlp: Vec<f32>,
    shift_q: Vec<f32>,
    one_plus_scale_q: Vec<f32>,
    gate_q: Vec<f32>,
}

fn slice_mod(combined: &[f32], t: usize, dim: usize) -> Mod {
    // combined: [T, 9, dim] row-major.
    let row = |i: usize| -> Vec<f32> {
        let mut v = vec![0f32; t * dim];
        for ti in 0..t {
            v[ti * dim..ti * dim + dim].copy_from_slice(&combined[(ti * 9 + i) * dim..(ti * 9 + i) * dim + dim]);
        }
        v
    };
    let one_plus = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| 1.0 + x).collect() };
    Mod {
        shift_msa: row(0),
        one_plus_scale_msa: one_plus(row(1)),
        gate_msa: row(2),
        shift_mlp: row(3),
        one_plus_scale_mlp: one_plus(row(4)),
        gate_mlp: row(5),
        shift_q: row(6),
        one_plus_scale_q: one_plus(row(7)),
        gate_q: row(8),
    }
}

/// The internal taps a parity test bisects with - the golden's
/// `b0_attn1_out`/`b0_attn2_out`/`b0_ff_out` (module HOOK outputs, i.e.
/// `b0_attn2_out` is the RAW cross-attention output before the `*gate_q`
/// multiply the block applies afterward - see `transformer.py`'s
/// `apply_cross_attention_adaln`, where the hook sits on `attn2` itself,
/// inside the function that later scales its return).
pub struct BlockTaps {
    pub attn1_out: Vec<f32>,
    pub attn2_out: Vec<f32>,
    pub ff_out: Vec<f32>,
}

/// One `BasicAVTransformerBlock` (video-only), weights resident, for a fixed
/// token/context length.
pub struct LtxBlock {
    gpu: Gpu,
    cfg: LtxDitConfig,
    w: BlockWeights,
    context_len: u32,
    ones_t: DeviceBuffer,
}

impl LtxBlock {
    /// `weights`: the checkpoint's tensors, keyed by canonical name (see
    /// `crate::dit::load_tiny_weights`). `prefix`: e.g. `"transformer_blocks.0"`.
    pub fn on(gpu: Gpu, cfg: &LtxDitConfig, weights: &Tensors, prefix: &str, tokens: u32, context_len: u32) -> LtxBlock {
        cfg.assert_supported();
        let dim = cfg.inner_dim as usize;
        let w = BlockWeights::upload(&gpu, weights, prefix, dim);
        let ones_t = gpu.storage(dim as u64);
        wf(&gpu, &ones_t, &vec![1.0f32; dim]);
        let _ = tokens;
        LtxBlock { gpu, cfg: *cfg, w, context_len, ones_t }
    }

    /// One block forward.
    ///
    /// `x`: `[T, dim]` current hidden state. `adaln_table`: `[T, 9*dim]` raw
    /// per-token adaLN-single linear output (shared across every block -
    /// this block's OWN `scale_shift_table` is added in here, per
    /// `get_ada_values`). `context`: `[context_len, dim]` RAW text context
    /// (this block's `prompt_scale_shift_table` modulates it here, fresh
    /// each block). `cos_bufs`/`sin_bufs`: this head's `[T, head_dim/2]`
    /// device-resident RoPE tables (built once, shared by every block -
    /// see `crate::dit`).
    ///
    /// Also returns [`BlockTaps`] (attn1/attn2/ff internal outputs) -
    /// unconditionally, since reading back three more `[T, dim]`-scale
    /// buffers is negligible at this milestone's token counts and it keeps
    /// this the ONE forward path a parity test replays (no separate
    /// "taps-enabled" build, unlike `vae3d.rs`'s env-var toggle, which
    /// exists there because the VAE's per-block taps are large real-weight
    /// buffers this crate's much smaller DiT tensors do not need to guard).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, x: &[f32], adaln_table: &[f32], context: &[f32], cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32) -> (Vec<f32>, BlockTaps) {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let dim = cfg.inner_dim;
        let heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let eps = cfg.norm_eps;
        let ctx_len = self.context_len;
        assert_eq!(x.len(), (t * dim) as usize);
        assert_eq!(adaln_table.len(), (t * 9 * dim) as usize);
        assert_eq!(context.len(), (ctx_len * dim) as usize);

        // Combine the shared per-token adaLN table with this block's own
        // static table (dit::adaln::add_table at rows=T), then slice.
        let combined = dit::adaln::add_table(adaln_table, &self.w.scale_shift_table, t as usize, 9 * dim as usize);
        let m = slice_mod(&combined, t as usize, dim as usize);

        // The block's static [shift_kv, scale_kv] (rows 0,1 of
        // prompt_scale_shift_table), broadcast to every context row -
        // per-block, NOT per-token (use_prompt_adaln_single=false).
        let pst = &self.w.prompt_scale_shift_table;
        let (shift_kv_row, scale_kv_row) = (&pst[0..dim as usize], &pst[dim as usize..2 * dim as usize]);
        let mut shift_kv = vec![0f32; (ctx_len * dim) as usize];
        let mut one_plus_scale_kv = vec![0f32; (ctx_len * dim) as usize];
        for r in 0..ctx_len as usize {
            shift_kv[r * dim as usize..r * dim as usize + dim as usize].copy_from_slice(shift_kv_row);
            for (d, v) in scale_kv_row.iter().enumerate() {
                one_plus_scale_kv[r * dim as usize + d] = 1.0 + v;
            }
        }

        let x_buf = gpu.storage((t * dim) as u64);
        wf(gpu, &x_buf, x);
        let ctx_buf = gpu.storage((ctx_len * dim) as u64);
        wf(gpu, &ctx_buf, context);

        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(gpu, &b, v);
            b
        };
        let shift_msa = up(&m.shift_msa);
        let one_plus_scale_msa = up(&m.one_plus_scale_msa);
        let gate_msa = up(&m.gate_msa);
        let shift_mlp = up(&m.shift_mlp);
        let one_plus_scale_mlp = up(&m.one_plus_scale_mlp);
        let gate_mlp = up(&m.gate_mlp);
        let shift_q = up(&m.shift_q);
        let one_plus_scale_q = up(&m.one_plus_scale_q);
        let gate_q = up(&m.gate_q);
        let shift_kv_buf = up(&shift_kv);
        let one_plus_scale_kv_buf = up(&one_plus_scale_kv);

        let mut s: Vec<Step> = Vec::new();
        let td = t * dim;

        // --- self-attention ------------------------------------------------
        let tmp1 = gpu.storage(td as u64);
        let tmp2 = gpu.storage(td as u64);
        let norm_vx = gpu.storage(td as u64);
        ada_zero(gpu, &mut s, &self.ones_t, &x_buf, &one_plus_scale_msa, &shift_msa, &tmp1, &tmp2, &norm_vx, dim, t, eps);
        let attn1_out = attention(gpu, &mut s, &self.w.attn1, dim, heads, head_dim, &norm_vx, &norm_vx, t, t, Some((cos_bufs, sin_bufs)), K_ROPE2D, eps);
        let x_fma = gpu.storage(td as u64);
        gate_row(gpu, &mut s, &x_buf, &gate_msa, &attn1_out, &x_fma, t, dim);

        // Fused re-norm feeding straight into text cross-attention - no
        // separate norm2 (see this module's doc, step 3).
        let x_normed = gpu.storage(td as u64);
        rmsnorm(gpu, &mut s, &x_fma, &self.ones_t, &x_normed, dim, t, eps);

        // --- text cross-attention with adaLN modulation --------------------
        let attn_input_tmp1 = gpu.storage(td as u64);
        let attn_input = gpu.storage(td as u64);
        mul(gpu, &mut s, &x_normed, &one_plus_scale_q, &attn_input_tmp1, td);
        add2(gpu, &mut s, &attn_input_tmp1, &shift_q, &attn_input, td);

        let ctxd = ctx_len * dim;
        let enc_tmp1 = gpu.storage(ctxd as u64);
        let enc_hidden = gpu.storage(ctxd as u64);
        mul(gpu, &mut s, &ctx_buf, &one_plus_scale_kv_buf, &enc_tmp1, ctxd);
        add2(gpu, &mut s, &enc_tmp1, &shift_kv_buf, &enc_hidden, ctxd);

        let ca_raw = attention(gpu, &mut s, &self.w.attn2, dim, heads, head_dim, &attn_input, &enc_hidden, t, ctx_len, None, K_ROPE2D, eps);
        let ca_gated = gpu.storage(td as u64);
        mul(gpu, &mut s, &ca_raw, &gate_q, &ca_gated, td);
        let x2 = gpu.storage(td as u64);
        add2(gpu, &mut s, &x_fma, &ca_gated, &x2, td);

        // --- MLP -------------------------------------------------------------
        let mlp_tmp1 = gpu.storage(td as u64);
        let mlp_tmp2 = gpu.storage(td as u64);
        let vx_scaled = gpu.storage(td as u64);
        ada_zero(gpu, &mut s, &self.ones_t, &x2, &one_plus_scale_mlp, &shift_mlp, &mlp_tmp1, &mlp_tmp2, &vx_scaled, dim, t, eps);
        let ff_dim = dim * 4;
        let h_pre = gpu.storage((t * ff_dim) as u64);
        linear(gpu, &mut s, &vx_scaled, &self.w.ff.w1, None, &h_pre, t, dim, ff_dim);
        let h_act = gpu.storage((t * ff_dim) as u64);
        s.push(gpu.step(K_GELU, &[&h_pre, &h_act], &[t * ff_dim], t * ff_dim));
        let ff_out = gpu.storage(td as u64);
        linear(gpu, &mut s, &h_act, &self.w.ff.w2, None, &ff_out, t, ff_dim, dim);
        let x3 = gpu.storage(td as u64);
        gate_row(gpu, &mut s, &x2, &gate_mlp, &ff_out, &x3, t, dim);

        gpu.submit(&[], &s);
        let out = gpu.read(&x3, td as usize);
        let taps = BlockTaps { attn1_out: gpu.read(&attn1_out, td as usize), attn2_out: gpu.read(&ca_raw, td as usize), ff_out: gpu.read(&ff_out, td as usize) };
        (out, taps)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

/// Open a device for the DiT's kernel table. `None` takes brain's default.
pub fn open_device(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}
