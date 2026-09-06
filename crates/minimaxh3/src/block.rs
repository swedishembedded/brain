// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's two block kinds as a recorded-and-run device kernel graph,
//! `crate::vocoder`'s eager Ctx/kernels-array dispatch style: every op below
//! immediately builds, submits and returns a fresh [`DeviceBuffer`] (no
//! deferred `Vec<Step>` graph, matching `vocoder::conv1d_same` etc.).
//!
//! Two kinds, per `transformer_minimax_h3.py`:
//!
//! - [`refiner_block_forward`]: `MiniMaxH3TokenRefinerBlock` - plain pre-norm
//!   self-attention + SwiGLU, no AdaLN, no RoPE.
//! - [`block_forward`]: `MiniMaxH3TransformerBlock` - the same pre-norm
//!   self-attention + SwiGLU sandwich, but with RoPE'd attention and every
//!   sub-layer's pre-norm/post-norm modulated by AdaLN-Zero parameters
//!   selected PER ROW of the packed sequence (`adaln_indices`) - never a
//!   single shared modulation vector, so the usual "fold token-independent
//!   modulation into the norm weight" shortcut does not apply here: the
//!   modulation genuinely varies per row of the packed sequence (per
//!   `(timestep, modality)`), not once per forward.
//!
//! ## AdaLN table layout - a deliberate deviation from the reference's own
//!
//! The reference's `adaln_proj.linear` is ONE fused `Linear(time_embed_dim,
//! 6*hidden_size*3)`; its output reshapes to `[num_timesteps*3, 6*hidden]`
//! with row order `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]` (timestep-major,
//! modality-minor), addressed by `adaln_indices = timestep_indices*3 +
//! token_tags`.
//!
//! This port instead builds SIX per-parameter tables (`shift_msa`,
//! `scale_msa`, `gate_msa`, `shift_mlp`, `scale_mlp`, `gate_mlp`), each
//! `[3*num_timesteps, hidden]` with row order MODALITY-major,
//! TIMESTEP-minor (`row = modality*num_timesteps + timestep_index`), and
//! addresses them with `adaln_indices = token_tags*num_timesteps +
//! timestep_indices` (`crate::model::adaln_indices`). Both encode the exact
//! same information (six real-valued modulation parameters per `(timestep,
//! modality)` pair); this port's row order is chosen because it is a
//! CONTIGUOUS host slice of the checkpoint's own weight layout, not because
//! it matches the reference's own row order. The reference's fused output
//! feature axis is modality-major
//! (`[mod0's 6*hidden | mod1's 6*hidden | mod2's 6*hidden]`, confirmed by its
//! own `.view(-1, 6*hidden)` reshape), so each `(modality, param)` pair's
//! `[hidden, time_embed_dim]` weight slice is already one contiguous row
//! range of `adaln_proj.linear.weight`. Building the reference's own
//! timestep-major table would need an interleaving SCATTER kernel this
//! workspace has no existing primitive for; this port's own row order needs
//! none - it splits the fused weight into contiguous host slices at forward
//! time instead (the same idea as splitting a fused weight at checkpoint
//! import time, just applied per-forward here since the table's only input,
//! `temb`, is itself rebuilt every forward from the timestep).
//! [`crate::model::adaln_indices`] is this port's OWN convention, internally
//! consistent end to end - nothing downstream ever reads the reference's row
//! order directly, so this changes no observable output.
//!
//! Swedish Embedded AB implements this diffusion transformer block port for
//! its clients. If your team needs expertise in porting large transformer
//! architectures to new inference stacks, you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};
use model::hostmath::linear_rows;

use crate::config::{H3TransformerConfig, MODALITY_NUM};

const K_MATMUL: usize = 0;
const K_BIAS_ADD: usize = 1;
const K_RMSNORM_EPS: usize = 2;
const K_ROPE2D_PARTIAL: usize = 3;
const K_ATTN_SCORES_QK: usize = 4;
const K_ATTN_SOFTMAX_BIDIR: usize = 5;
const K_ATTN_APPLY_FULL: usize = 6;
const K_SILU_MUL: usize = 7;
const K_EMBED: usize = 8;
const K_GATE_ROW: usize = 9;
const K_MUL: usize = 10;
const K_ADD2: usize = 11;
const K_ROW_SCATTER: usize = 12;
/// The 128x128 register-tiled GEMM `crate::block::linear` picks via
/// `model::block::pick_gemm` once a shape clears the measured tile-fill
/// crossover (`select::GEMM_TILE_MIN_ROWS`/`_COLS`) - the same kernel ~15
/// other model crates (`qwen3`, `wan`, `clip`, `t5encoder`, `vae`, ...)
/// already register alongside plain `matmul` for exactly this purpose. On
/// CPU this and `matmul` are the SAME dispatched AVX2 fast path by kernel
/// name (`backend_cpu::FastIdx` routes both to `fast_ops::matmul_abt`) - the
/// choice only changes behavior on a real GPU, where `matmul` is a naive
/// one-thread-per-output kernel with no operand reuse and (for this crate's
/// wide, `col = idx % n`-indexed weight rows) a fully uncoalesced access
/// pattern, while `matmul_reg3` stages both operands through workgroup
/// memory.
const K_MATMUL_REG3: usize = 13;
/// Packs the three separate `[seq_len, inner]` q/k/v projections into the one
/// `[seq_len, 3*inner]` slab every kernel in the flash family reads through
/// `qkv_stride`/`q_off`/`k_off`/`v_off`. Only [`attention_flash`] dispatches
/// it - the materialized trio reads q/k/v as three bindings and needs no slab.
const K_PACK_QKV: usize = 14;
/// The four interchangeable bidirectional flash-attention rungs, registered as
/// a complete set so `model::block::flash_bidir_variant` can walk the whole
/// ladder from this crate's queried `DeviceCaps` (see [`FLASH_IDS`]).
const K_FLASH_ATTN_BIDIR: usize = 15;
const K_FLASH_ATTN_BIDIR_SPLIT: usize = 16;
const K_FLASH_ATTN_BIDIR_REG: usize = 17;
const K_FLASH_ATTN_BIDIR_REG2: usize = 18;

pub const KERNELS: [(&str, &str); 19] = [
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("attn_scores_qk", kernels::ATTN_SCORES_QK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("silu_mul", kernels::SILU_MUL),
    ("embed", kernels::EMBED),
    ("gate_row", kernels::GATE_ROW),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("row_scatter", kernels::ROW_SCATTER),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("pack_qkv", kernels::PACK_QKV),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
];

/// This crate's rung set for [`model::block::flash_bidir_variant`] - all four
/// registered, so the selector picks the best one this device's queried caps
/// allow rather than this crate pinning a rung by hand.
const FLASH_IDS: model::block::FlashIds = model::block::FlashIds {
    bidir: K_FLASH_ATTN_BIDIR,
    split: Some(K_FLASH_ATTN_BIDIR_SPLIT),
    reg: Some(K_FLASH_ATTN_BIDIR_REG),
    reg2: Some(K_FLASH_ATTN_BIDIR_REG2),
};

/// One open device + the shared kernel table - every op below dispatches
/// through this, matching `crate::vocoder::Ctx`'s own role.
pub struct Ctx {
    pub gpu: Gpu,
}

impl Ctx {
    pub fn new(device: Option<&str>) -> Ctx {
        Ctx { gpu: Gpu::open(device, &KERNELS) }
    }

    pub fn upload(&self, data: &[f32]) -> DeviceBuffer {
        self.gpu.storage_init("minimaxh3", data)
    }

    pub fn upload_u32(&self, data: &[u32]) -> DeviceBuffer {
        let b = self.gpu.storage(data.len() as u64);
        self.gpu.write(&b, data);
        b
    }
}

// ---------------- eager device ops ----------------

/// `y = x @ w^T (+ bias)`. `w` is `[n, k]` row-major (PyTorch `nn.Linear`
/// convention); `bias` is `[n]` or `None` for the attention/FFN linears,
/// which the reference builds with `bias=False` throughout.
///
/// Kernel choice goes through `model::block::pick_gemm` (the same seam
/// `qwen3`/`wan`/`clip`/`t5encoder`/`vae` and a dozen other model crates
/// already use), not a hardcoded `K_MATMUL` - every one of this crate's
/// linears is a `[m,k]x[n,k]^T` GEMM at real transformer widths (`n` in the
/// thousands for q/k/v/fc1/fc2), squarely the "large enough to fill a
/// 128x128 tile" regime `pick_gemm` was measured against. `K_MATMUL_REG3`
/// is a no-op on CPU (backend-cpu's native fast path treats it and `matmul`
/// identically) and the fix for a real GPU, where plain `matmul` measured
/// ~20x slower than the CPU path on this crate's own shapes precisely
/// because it has neither operand reuse nor coalesced access (adjacent GPU
/// threads read weight rows `k` floats apart).
pub(crate) fn linear(cx: &Ctx, x: &DeviceBuffer, w: &DeviceBuffer, bias: Option<&DeviceBuffer>, m: u32, k: u32, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((m * n) as u64);
    let (mm, threads) = model::block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false);
    cx.gpu.submit(&[], &[cx.gpu.step(mm, &[x, w, &y], &[m, k, n], threads)]);
    if let Some(b) = bias {
        cx.gpu.submit(&[], &[cx.gpu.step(K_BIAS_ADD, &[&y, b], &[m, n], m * n)]);
    }
    y
}

/// RMSNorm, one row per token: `d` channels, `rows` tokens.
pub(crate) fn rmsnorm(cx: &Ctx, x: &DeviceBuffer, w: &DeviceBuffer, rows: u32, d: u32, eps: f32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_RMSNORM_EPS, &[x, w, &y], &[d, rows, gpu_core::f(eps)], rows)]);
    y
}

/// In-place partial RoPE on `buf` (`[rows, heads*head_dim]`, contiguous
/// per-head layout), against precomputed `[rows, half]` `cos`/`sin` tables
/// (`crate::rope::build_tables`). Rotates the leading `2*half` channels of
/// every head; the remaining `head_dim - 2*half` pass through unchanged.
#[allow(clippy::too_many_arguments)]
fn rope_partial(cx: &Ctx, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32, head_dim: u32, half: u32) {
    let total = rows * heads * half;
    let row_stride = heads * head_dim;
    cx.gpu.submit(&[], &[cx.gpu.step(K_ROPE2D_PARTIAL, &[buf, cos, sin], &[rows, heads, half, row_stride, 0, rows, gpu_core::f(1.0), head_dim], total)]);
}

/// Full (bidirectional, unmasked) self-attention from three SEPARATE
/// `[seq_len, heads*head_dim]` q/k/v buffers - `MiniMaxH3AttnProcessor`'s own
/// shape (no causal mask, no cross-attention anywhere in the model). Returns
/// `[seq_len, heads*head_dim]`.
///
/// Both arms below compute the SAME function - `softmax(q·kᵀ/√head_dim)·v`,
/// unmasked, in the same `[seq_len, heads*head_dim]` row-major output layout -
/// and differ only in whether the `[heads, seq_len, seq_len]` scores and probs
/// matrices are ever written to device memory. They are picked by
/// [`attn_mode`], never by shape.
fn attention(cx: &Ctx, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    match attn_mode(&cx.gpu, head_dim) {
        AttnMode::Flash => attention_flash(cx, q, k, v, seq_len, heads, head_dim),
        AttnMode::Materialized => attention_trio(cx, q, k, v, seq_len, heads, head_dim),
    }
}

/// The self-attention path for this device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnMode {
    /// One fused `flash_attn_bidir*` dispatch, online softmax, nothing
    /// `O(seq_len²)` ever allocated.
    Flash,
    /// The `attn_scores_qk -> attn_softmax_bidir -> attn_apply_full` trio,
    /// which materializes both `[heads, seq_len, seq_len]` matrices.
    Materialized,
}

/// Which attention path this device takes.
///
/// Flash is the default wherever the device can run it, because the trio's two
/// materialized `[heads, seq_len, seq_len]` slabs are what make this model OOM
/// long before the weights do: H3 attends over ONE packed sequence holding
/// every video, audio and text token of the whole clip at once, so `seq_len`
/// grows with the square of the canvas edge and the slabs grow with its fourth
/// power. At 56 heads the pair costs `2·56·seq_len²·4` bytes - already tens of
/// gigabytes at the token counts a 256x256 canvas produces, against a 24 GB
/// card and a per-binding ceiling far below that. Flash's peak attention
/// memory is `O(seq_len·head_dim)` instead.
///
/// The device check is [`model::block::flash_gate`] - the shared outer gate
/// every flash-family caller in this workspace goes through, whose
/// `workgroup_reductions` bit is false on the Cranelift CPU JIT (these kernels
/// need two or three top-level barriers where that JIT splits at one). So the
/// CPU path keeps the trio, which `backend_cpu::FastIdx` routes to native
/// dispatched fast paths by kernel NAME, and which therefore stays the
/// reference definition of the math this crate's tiny-config tests check.
///
/// This crate's own extra condition is `head_dim <= 128`, the family's hard
/// limit (`flash_bidir_step` asserts it): checked here so an out-of-range
/// config falls back rather than panicking. `H3TransformerConfig`'s real
/// `attention_head_dim` is 128 - exactly at the limit, not under it, so any
/// future variant with a wider head silently takes the trio and must be
/// re-measured rather than assumed to fit.
///
/// `BRAIN_MINIMAXH3_ATTN=flash|trio` forces either arm. Without it, an A/B on
/// a device that can run flash compares the fused path against itself and
/// reports a meaningless parity - which looks like evidence and is not; it is
/// also how the memory measurement gets its "before" number without checking
/// out an older commit.
pub fn attn_mode(gpu: &Gpu, head_dim: u32) -> AttnMode {
    match std::env::var("BRAIN_MINIMAXH3_ATTN").ok().as_deref() {
        Some("flash") => return AttnMode::Flash,
        Some("trio") => return AttnMode::Materialized,
        _ => {}
    }
    if model::block::flash_gate(&gpu.caps(), head_dim <= 128) {
        AttnMode::Flash
    } else {
        AttnMode::Materialized
    }
}

/// The materialized `scores -> softmax -> apply` trio: two
/// `[heads, seq_len, seq_len]` device buffers, three dispatches. The reference
/// definition of this crate's attention, and the only arm the CPU JIT can run.
///
/// `heads * seq_len * seq_len` is computed in u64 and bounded, because in u32
/// it WRAPS - at the real 56 heads, past `seq_len` 8757. A wrapped count is
/// not a clean failure: it sizes the scores buffer and the dispatch's thread
/// count from the same wrong number, so a shape that wraps to something small
/// allocates happily and computes garbage with nothing anywhere saying so. It
/// is only visible at all today because `seq_len` 9900 happens to wrap to
/// 4.77 GB, which still trips the device's own maximum-buffer-size check.
/// Same defect, same reasoning and same fix as the video VAE's untiled
/// attention (see `video_vae`'s own scores sizing); this arm just reaches it
/// at a lower `seq_len` because it runs 56 heads rather than 32.
///
/// The bound is a refusal rather than a widening because there is nothing to
/// widen INTO: a dispatch takes a u32 thread count, so a score matrix with
/// more than `u32::MAX` elements cannot be dispatched at all in this form.
/// [`attention_flash`] has no such ceiling - it never builds the matrix - so
/// on any device that can run it this limit is unreachable.
fn attention_trio(cx: &Ctx, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    let inner = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_scores = heads as u64 * seq_len as u64 * seq_len as u64;
    assert!(
        n_scores <= u32::MAX as u64,
        "minimaxh3 attention: materialized scores need {n_scores} elements at seq_len {seq_len} x {heads} heads, \
         past the u32 dispatch ceiling. This shape needs the fused flash path (a device with workgroup reductions, \
         and BRAIN_MINIMAXH3_ATTN not pinned to \"trio\")."
    );
    let scores = cx.gpu.storage(n_scores);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SCORES_QK, &[q, k, &scores], &[1, heads, seq_len, head_dim, inner, 0, gpu_core::f(scale)], n_scores as u32)]);
    let probs = cx.gpu.storage(n_scores);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SOFTMAX_BIDIR, &[&scores, &probs], &[1, heads, seq_len], heads * seq_len)]);
    let out = cx.gpu.storage((seq_len * inner) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_APPLY_FULL, &[&probs, v, &out], &[1, heads, seq_len, head_dim, inner, inner], heads * seq_len * head_dim)]);
    out
}

/// One fused flash-attention dispatch over the whole packed sequence.
///
/// The family reads q, k and v as three REGIONS of a single row-major slab
/// rather than three bindings, so the projections are packed into
/// `[seq_len, 3*inner]` first (`pack_qkv`, one dispatch, one pass over
/// `3·seq_len·inner` elements). That slab is the entire extra allocation this
/// path makes, and it is linear in `seq_len` where the trio's pair of matrices
/// is quadratic: at 56 heads the crossover is at `seq_len = 3·inner/heads =
/// 3·head_dim = 384` tokens, i.e. everything past a trivially small sequence.
///
/// The shape contract `flash_bidir_step` documents is met exactly by that
/// slab: `stride = 3*inner`, `q_off/k_off/v_off = 0/inner/2*inner`,
/// `d_model = inner = heads*head_dim`, `bsz = 1` (H3 packs the whole clip into
/// ONE sequence, so there is never a second sample to batch). The kernel
/// applies `1/sqrt(head_dim)` itself and masks nothing, which is what this
/// model wants - the scale matches [`attention_trio`]'s explicit one and the
/// attention is bidirectional in both arms.
///
/// RoPE has already been applied to `q`/`k` in place by the caller, so packing
/// AFTER it (rather than rotating inside the slab, as `minimaxmusic3` does)
/// keeps the rotation dispatches bit-identical to the trio path's.
fn attention_flash(cx: &Ctx, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    let inner = heads * head_dim;
    let qkv = cx.gpu.storage((seq_len * 3 * inner) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_PACK_QKV, &[q, k, v, &qkv], &[seq_len, inner], seq_len * 3 * inner)]);
    let out = cx.gpu.storage((seq_len * inner) as u64);
    cx.gpu.submit(&[], &[model::block::flash_bidir_step(&cx.gpu, FLASH_IDS, 1, heads, seq_len, head_dim, inner, &qkv, &out)]);
    out
}

/// `y = silu(gate) * value` - the checkpoint's fused SwiGLU
/// (`SwiGLU.forward`: `proj(x).chunk(2,-1) -> (value, gate); value *
/// silu(gate)`). `gate`/`value` are the two halves of the fused `fc1`
/// projection, split at IMPORT time into separate contiguous buffers (see
/// [`BlockWeights`]'s own doc), so this is a plain elementwise call, not a
/// strided/interleaved one.
fn swiglu(cx: &Ctx, value: &DeviceBuffer, gate: &DeviceBuffer, total: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(total as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_SILU_MUL, &[gate, value, &y], &[total], total)]);
    y
}

/// Row gather: `y[r, :] = table[idx[r], :]`.
pub(crate) fn gather_rows(cx: &Ctx, idx: &DeviceBuffer, table: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_EMBED, &[idx, table, &y], &[d, rows], rows * d)]);
    y
}

/// `y = a * b`, fresh buffer.
fn mul(cx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(n as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_MUL, &[a, b, &y], &[n], n)]);
    y
}

/// `y = a + b`, fresh buffer.
pub(crate) fn add2(cx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(n as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// `y = normed * (1 + scale) + shift`, fresh buffer - the AdaLN-Zero
/// pre-norm modulation, `scale`/`shift` already gathered to one row per
/// token (see [`block_forward`]; also used directly by `crate::model` for
/// `norm_out`'s per-timestep, non-gated modulation).
pub(crate) fn modulate(cx: &Ctx, normed: &DeviceBuffer, scale: &DeviceBuffer, shift: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let scaled = mul(cx, normed, scale, n);
    let plus_x = add2(cx, &scaled, normed, n);
    add2(cx, &plus_x, shift, n)
}

/// `y[r,:] = x[r,:] + gate[r,:] * h[r,:]` - the AdaLN-Zero gated residual
/// merge, `gate` already gathered to one row per token.
fn gated_residual(cx: &Ctx, x: &DeviceBuffer, gate: &DeviceBuffer, h: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_GATE_ROW, &[x, gate, h, &y], &[rows, d, 1], rows * d)]);
    y
}

/// Scatter `src`'s `n_idx` rows into `out` at `idx`'s positions - `out` must
/// already be zeroed for any row `idx` never names (row-major `[n_idx, d]`
/// `src`, `[n_rows_out, d]` `out`).
pub fn row_scatter(cx: &Ctx, idx: &DeviceBuffer, src: &DeviceBuffer, out: &DeviceBuffer, n_idx: u32, d: u32, n_rows_out: u32) {
    cx.gpu.submit(&[], &[cx.gpu.step(K_ROW_SCATTER, &[idx, src, out], &[n_idx, d, n_rows_out], n_idx * d)]);
}

// ---------------- weights ----------------

/// One `MiniMaxH3TransformerBlock`'s weights. `fc1_value`/`fc1_gate` are
/// split from the reference's fused `SwiGLU.proj.weight` (`[2*ffn_dim,
/// hidden]`, the checkpoint's own `mlp.fc1.weight`) - a HOST-side row-slice
/// of the loaded flat tensor at import time, never an offset view read at
/// forward time (`crate::model`'s loader does this split; see its own doc).
/// `wq`/`wk`/`wv`
/// are loaded directly from the reference's own separate `to_q`/`to_k`/
/// `to_v` tensors (this port's golden dump does not call the reference's
/// optional `fuse_projections()`) - splitting the real checkpoint's own
/// fused `qkv_proj` tensor the same way is deferred to whichever later phase
/// wires up real-checkpoint import; the forward math is identical either
/// way. `adaln_w`/`adaln_b` stay HOST-side (never uploaded as-is): they are
/// read fresh every forward by [`block_forward`] to build this block's small
/// per-timestep modulation tables (see this module's own doc on the
/// row-order deviation).
pub struct BlockWeights {
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub norm_q: DeviceBuffer,
    pub norm_k: DeviceBuffer,
    pub norm1: DeviceBuffer,
    pub norm2: DeviceBuffer,
    pub fc1_value: DeviceBuffer,
    pub fc1_gate: DeviceBuffer,
    pub fc2: DeviceBuffer,
    /// `adaln_proj.linear.weight`, `[6*hidden*MODALITY_NUM, time_embed_dim]`
    /// row-major, untouched (host).
    pub adaln_w: Vec<f32>,
    /// `adaln_proj.linear.bias`, `[6*hidden*MODALITY_NUM]` (host).
    pub adaln_b: Vec<f32>,
}

/// One `MiniMaxH3TokenRefinerBlock`'s weights - the same attention/FFN
/// shape as [`BlockWeights`] with no AdaLN and no RoPE.
pub struct RefinerBlockWeights {
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub norm_q: DeviceBuffer,
    pub norm_k: DeviceBuffer,
    pub norm1: DeviceBuffer,
    pub norm2: DeviceBuffer,
    pub fc1_value: DeviceBuffer,
    pub fc1_gate: DeviceBuffer,
    pub fc2: DeviceBuffer,
}

// ---------------- streaming per-block loading ----------------
//
// One place backing BOTH `H3Transformer::load`'s eager whole-model build
// (every block resident, for a caller that reuses one instance across many
// forwards - the resident-serving path) and a caller that processes blocks
// ONE AT A TIME, dropping each block's weights before loading the next
// (`model::tests::dit_matches_the_real_reference_numerically_layer_by_layer`'s
// own reason to exist: a real-weight VALIDATION run that only checks three
// tap points has no business holding all `num_layers` blocks' weights
// resident just to reach them - a single-block validation pass needs at
// most one block's weights (~2.6GB at these real dimensions) at a time, not
// the whole ~132GB fp32 model).

/// Fetch-upload-drop one f32 tensor as a device buffer. `advise_drop`s the
/// source's pages behind it immediately - this tensor's data is fully
/// consumed once `ctx.upload` returns.
///
/// Deliberately `Ctx::upload`/`storage_init`, not `paramstore::upload::
/// Uploader`: a real-hardware measurement (`crates/gpu-core/tests/
/// bench_upload_cadence.rs`) found routing this crate's loads through
/// `Uploader` is a regression on the CPU backend (a "device buffer" there
/// already IS host memory, so `storage_init` is one direct copy and
/// `Uploader`'s chunking only adds per-chunk overhead with nothing to avoid)
/// and within measurement noise on a real P40 (the forked wgpu's
/// `create_buffer_init` no longer carries the VRAM-doubling defect
/// `Uploader`'s own module doc was written against - see `crates/gpu-core/
/// tests/vram_overhead.rs`). An earlier version of this function used
/// `Uploader` based on a since-retracted ~12-minute-vs-CPU GPU measurement
/// that turned out to be dominated by unrelated system contention, not this
/// function - including a separate probe bug that briefly pointed the same
/// way for a different reason.
pub fn load_dev(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, name: &str) -> DeviceBuffer {
    let mut buf: Option<DeviceBuffer> = None;
    let found = tensors.with_tensor(name, &mut |data| buf = Some(ctx.upload(data)));
    assert!(found, "minimaxh3 model: missing tensor {name:?}");
    tensors.advise_drop(name);
    buf.unwrap()
}

/// [`load_dev`]'s host-side twin, for the small per-block tensors
/// (`adaln_proj`'s weight/bias) [`block_forward`] reads fresh every call
/// rather than uploading once.
pub fn load_host(tensors: &dyn checkpoint::TensorSource, name: &str) -> Vec<f32> {
    let mut out: Option<Vec<f32>> = None;
    let found = tensors.with_tensor(name, &mut |data| out = Some(data.to_vec()));
    assert!(found, "minimaxh3 model: missing tensor {name:?}");
    tensors.advise_drop(name);
    out.unwrap()
}

/// Split the fused SwiGLU projection's two output-feature halves into
/// separate contiguous buffers. The split itself must happen INSIDE the
/// `with_tensor` callback (the borrowed slice does not outlive it); the
/// row-count check that would read the tensor's own 2D shape is a numel
/// check instead - a streaming `TensorSource` has no shape to hand back,
/// and the total element count is exactly what the split below depends on.
pub fn load_fc1(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, prefix: &str, hidden: u32, ffn: u32) -> (DeviceBuffer, DeviceBuffer) {
    let name = format!("{prefix}.ff.net.0.proj.weight");
    let half = (ffn * hidden) as usize;
    let mut out: Option<(DeviceBuffer, DeviceBuffer)> = None;
    let found = tensors.with_tensor(&name, &mut |data| {
        assert_eq!(data.len(), 2 * half, "{name}: expected {} elements, got {}", 2 * half, data.len());
        out = Some((ctx.upload(&data[..half]), ctx.upload(&data[half..2 * half])));
    });
    assert!(found, "minimaxh3 model: missing tensor {name:?}");
    tensors.advise_drop(&name);
    out.unwrap()
}

/// The 6 attention-projection tensors shared by both block kinds' `.attn`
/// sub-module naming (`to_q`/`to_k`/`to_v`/`norm_q`/`norm_k`/`to_out.0`).
pub fn load_attn(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, prefix: &str) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    (
        load_dev(tensors, ctx, &format!("{prefix}.to_q.weight")),
        load_dev(tensors, ctx, &format!("{prefix}.to_k.weight")),
        load_dev(tensors, ctx, &format!("{prefix}.to_v.weight")),
        load_dev(tensors, ctx, &format!("{prefix}.norm_q.weight")),
        load_dev(tensors, ctx, &format!("{prefix}.norm_k.weight")),
        load_dev(tensors, ctx, &format!("{prefix}.to_out.0.weight")),
    )
}

/// Load `transformer_blocks.{index}`'s weights - a single [`BlockWeights`],
/// nothing else. A caller processing blocks one at a time drops the
/// returned value (freeing its device buffers) before loading the next
/// index.
pub fn load_block(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, index: usize, hidden: u32, ffn: u32) -> BlockWeights {
    let p = format!("transformer_blocks.{index}");
    let (wq, wk, wv, norm_q, norm_k, wo) = load_attn(tensors, ctx, &format!("{p}.attn"));
    let (fc1_value, fc1_gate) = load_fc1(tensors, ctx, &p, hidden, ffn);
    BlockWeights {
        wq,
        wk,
        wv,
        wo,
        norm_q,
        norm_k,
        norm1: load_dev(tensors, ctx, &format!("{p}.norm1.weight")),
        norm2: load_dev(tensors, ctx, &format!("{p}.norm2.weight")),
        fc1_value,
        fc1_gate,
        fc2: load_dev(tensors, ctx, &format!("{p}.ff.net.2.weight")),
        adaln_w: load_host(tensors, &format!("{p}.adaln_proj.linear.weight")),
        adaln_b: load_host(tensors, &format!("{p}.adaln_proj.linear.bias")),
    }
}

/// [`load_block`]'s `token_refiner.refiner_blocks.{index}` analogue.
pub fn load_refiner_block(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, index: usize, hidden: u32, ffn: u32) -> RefinerBlockWeights {
    let p = format!("token_refiner.refiner_blocks.{index}");
    let (wq, wk, wv, norm_q, norm_k, wo) = load_attn(tensors, ctx, &format!("{p}.attn"));
    let (fc1_value, fc1_gate) = load_fc1(tensors, ctx, &p, hidden, ffn);
    RefinerBlockWeights {
        wq,
        wk,
        wv,
        wo,
        norm_q,
        norm_k,
        norm1: load_dev(tensors, ctx, &format!("{p}.norm1.weight")),
        norm2: load_dev(tensors, ctx, &format!("{p}.norm2.weight")),
        fc1_value,
        fc1_gate,
        fc2: load_dev(tensors, ctx, &format!("{p}.ff.net.2.weight")),
    }
}

// ---------------- forwards ----------------

/// `MiniMaxH3TokenRefinerBlock.forward`: plain pre-norm attention (no RoPE)
/// + plain pre-norm SwiGLU FFN, both residual. `x`: `[seq_len, hidden]`.
pub fn refiner_block_forward(cx: &Ctx, w: &RefinerBlockWeights, x: &DeviceBuffer, cfg: &H3TransformerConfig, seq_len: u32) -> DeviceBuffer {
    let (hidden, heads, hd, inner, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.inner_dim(), cfg.ffn_dim);

    let n1 = rmsnorm(cx, x, &w.norm1, seq_len, hidden, cfg.norm_eps);
    let q = linear(cx, &n1, &w.wq, None, seq_len, hidden, inner);
    let k = linear(cx, &n1, &w.wk, None, seq_len, hidden, inner);
    let v = linear(cx, &n1, &w.wv, None, seq_len, hidden, inner);
    let qn = rmsnorm(cx, &q, &w.norm_q, seq_len * heads, hd, cfg.qk_norm_eps);
    let kn = rmsnorm(cx, &k, &w.norm_k, seq_len * heads, hd, cfg.qk_norm_eps);
    let attn = attention(cx, &qn, &kn, &v, seq_len, heads, hd);
    let attn_out = linear(cx, &attn, &w.wo, None, seq_len, inner, hidden);
    let x1 = add2(cx, x, &attn_out, seq_len * hidden);

    let n2 = rmsnorm(cx, &x1, &w.norm2, seq_len, hidden, cfg.norm_eps);
    let value = linear(cx, &n2, &w.fc1_value, None, seq_len, hidden, ffn);
    let gate = linear(cx, &n2, &w.fc1_gate, None, seq_len, hidden, ffn);
    let act = swiglu(cx, &value, &gate, seq_len * ffn);
    let ff_out = linear(cx, &act, &w.fc2, None, seq_len, ffn, hidden);
    add2(cx, &x1, &ff_out, seq_len * hidden)
}

/// Build this block's six per-(modality, timestep) modulation tables, each
/// row-major `[MODALITY_NUM*num_timesteps, hidden]` in this port's own
/// modality-major row order (see this module's own doc). `temb_silu` is
/// `silu(temb)`, `[num_timesteps, time_embed_dim]` - the reference applies
/// `silu` before every AdaLN projection, never once up front.
fn adaln_tables(w: &BlockWeights, temb_silu: &[f32], num_timesteps: usize, hidden: usize, time_embed_dim: usize) -> [Vec<f32>; 6] {
    let mut tables: [Vec<f32>; 6] = Default::default();
    for t in tables.iter_mut() {
        *t = vec![0f32; MODALITY_NUM as usize * num_timesteps * hidden];
    }
    for modality in 0..MODALITY_NUM as usize {
        for (param, table) in tables.iter_mut().enumerate() {
            let feat_row0 = (modality * 6 + param) * hidden;
            let w_slice = &w.adaln_w[feat_row0 * time_embed_dim..(feat_row0 + hidden) * time_embed_dim];
            let b_slice = &w.adaln_b[feat_row0..feat_row0 + hidden];
            let out = linear_rows(temb_silu, w_slice, num_timesteps, time_embed_dim, hidden);
            for t_idx in 0..num_timesteps {
                let dst = &mut table[(modality * num_timesteps + t_idx) * hidden..(modality * num_timesteps + t_idx + 1) * hidden];
                for c in 0..hidden {
                    dst[c] = out[t_idx * hidden + c] + b_slice[c];
                }
            }
        }
    }
    tables
}

/// `MiniMaxH3TransformerBlock.forward`: RoPE'd, AdaLN-Zero modulated
/// pre-norm attention + pre-norm SwiGLU FFN, both residual-gated per row.
/// `x`: `[seq_len, hidden]`; `adaln_indices`: `[seq_len]` u32 device buffer
/// (`crate::model::adaln_indices` - THIS port's row convention, not the
/// reference's, see this module's own doc); `cos`/`sin`: `[seq_len, half]`
/// (`crate::rope::build_tables`); `temb_silu`/`num_timesteps`: this
/// forward's shared timestep conditioning, `silu`'d once by the caller
/// (every block reads the identical `temb`, only the projection differs).
///
/// Returns `(output, attn_out)` - `attn_out` is the attention sub-block's own
/// output (post `to_out`, pre gated residual), exposed as a parity tap
/// (`crate::model`'s numeric parity test reads it for block 0) rather than
/// only the block's final output, so a bug localizes to whichever half of
/// the block introduced it.
#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    cx: &Ctx,
    w: &BlockWeights,
    x: &DeviceBuffer,
    adaln_indices: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    temb_silu: &[f32],
    num_timesteps: usize,
    cfg: &H3TransformerConfig,
    seq_len: u32,
) -> (DeviceBuffer, DeviceBuffer) {
    let (hidden, heads, hd, inner, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.inner_dim(), cfg.ffn_dim);
    let half = 3 * cfg.rope_freq_dim;

    let tables = adaln_tables(w, temb_silu, num_timesteps, hidden as usize, cfg.time_embed_dim as usize);
    let table_dev: Vec<DeviceBuffer> = tables.iter().map(|t| cx.upload(t)).collect();
    let rows = MODALITY_NUM * num_timesteps as u32;
    let g = |i: usize| gather_rows(cx, adaln_indices, &table_dev[i], seq_len, hidden);
    let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (g(0), g(1), g(2), g(3), g(4), g(5));
    debug_assert!(rows > 0, "at least one (modality, timestep) row must exist");

    // attention sub-block
    let n1 = rmsnorm(cx, x, &w.norm1, seq_len, hidden, cfg.norm_eps);
    let mod1 = modulate(cx, &n1, &scale_msa, &shift_msa, seq_len * hidden);
    let q = linear(cx, &mod1, &w.wq, None, seq_len, hidden, inner);
    let k = linear(cx, &mod1, &w.wk, None, seq_len, hidden, inner);
    let v = linear(cx, &mod1, &w.wv, None, seq_len, hidden, inner);
    let qn = rmsnorm(cx, &q, &w.norm_q, seq_len * heads, hd, cfg.qk_norm_eps);
    let kn = rmsnorm(cx, &k, &w.norm_k, seq_len * heads, hd, cfg.qk_norm_eps);
    rope_partial(cx, &qn, cos, sin, seq_len, heads, hd, half);
    rope_partial(cx, &kn, cos, sin, seq_len, heads, hd, half);
    let attn = attention(cx, &qn, &kn, &v, seq_len, heads, hd);
    let attn_out = linear(cx, &attn, &w.wo, None, seq_len, inner, hidden);
    let h1 = gated_residual(cx, x, &gate_msa, &attn_out, seq_len, hidden);

    // FFN sub-block
    let n2 = rmsnorm(cx, &h1, &w.norm2, seq_len, hidden, cfg.norm_eps);
    let mod2 = modulate(cx, &n2, &scale_mlp, &shift_mlp, seq_len * hidden);
    let value = linear(cx, &mod2, &w.fc1_value, None, seq_len, hidden, ffn);
    let gate = linear(cx, &mod2, &w.fc1_gate, None, seq_len, hidden, ffn);
    let act = swiglu(cx, &value, &gate, seq_len * ffn);
    let ff_out = linear(cx, &act, &w.fc2, None, seq_len, ffn, hidden);
    let out = gated_residual(cx, &h1, &gate_mlp, &ff_out, seq_len, hidden);
    (out, attn_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cosine similarity and max absolute difference between two same-length
    /// vectors - the pair of numbers this workspace's numeric parity gates are
    /// always stated in.
    fn cos_and_maxabs(a: &[f32], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            maxabs = maxabs.max((x - y).abs());
        }
        (dot / (na.sqrt() * nb.sqrt()), maxabs)
    }

    /// Deterministic pseudo-random fill - a SplitMix64 draw's top 24 bits
    /// mapped into `[-1, 1)`, so both arms see byte-identical inputs without
    /// pulling an RNG dependency into this module.
    ///
    /// The RANGE is load-bearing, not cosmetic. `q·k` runs over `head_dim`
    /// terms and is then scaled by only `1/sqrt(head_dim)`, so inputs an order
    /// of magnitude wide push the logits far enough apart that `softmax`
    /// saturates to a one-hot pick of a single `v` row - at which point BOTH
    /// arms return that row verbatim, agree bit-for-bit, and prove nothing
    /// about the online softmax. An earlier version of this generator divided
    /// by 2048 instead of 2^23 and produced values up to 8191; it reported
    /// cosine 1.0000000000 and max_abs exactly 0 at every sequence length,
    /// which looked like an unusually clean pass and was in fact a vacuous
    /// one. [`assert_mixing`] below is the standing guard against that.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn std_dev(v: &[f32]) -> f64 {
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Fail unless the attention actually AVERAGED over many keys.
    ///
    /// Attention output is a convex combination of `v` rows, so a softmax that
    /// spreads its mass over `n` keys shrinks the spread of the result by
    /// roughly `sqrt(n)`, while one that has saturated to a single key
    /// reproduces a `v` row unchanged and leaves the spread equal to `v`'s.
    /// Comparing two arms on a saturated softmax compares two copies of the
    /// same lookup and would pass no matter what the online softmax did with
    /// its running max and rescale - the exact code this test exists to check.
    /// Half of `v`'s own standard deviation is a deliberately loose threshold:
    /// it is nowhere near the `1/sqrt(37)` a near-uniform mix over even the
    /// shortest sequence here would give, so it fires only on genuine
    /// saturation and never on an unlucky draw.
    fn assert_mixing(out: &[f32], v: &[f32], seq_len: u32) {
        let (so, sv) = (std_dev(out), std_dev(v));
        assert!(
            so < 0.5 * sv,
            "seq_len {seq_len}: softmax did not mix - output std {so:.4} vs v std {sv:.4}. \
             The logits have saturated to a one-hot pick, so both attention arms are \
             returning the same v row verbatim and this comparison proves nothing."
        );
    }

    /// A score matrix too large to address in u32 must be REFUSED, not
    /// silently wrapped.
    ///
    /// At the real 56 heads, `heads * seq_len * seq_len` exceeds `u32::MAX`
    /// past `seq_len` 8757 - well inside the range a 512x512 canvas reaches
    /// (~9900 packed rows). Computed in u32 it wraps, and the wrapped value
    /// then sizes BOTH the scores buffer and the dispatch's thread count, so
    /// the failure mode is a plausible-looking allocation full of garbage
    /// rather than an error. This is the same defect, at a lower `seq_len`
    /// because of the head count, that the video VAE's untiled attention
    /// already had to bound.
    ///
    /// Runs on the CPU device with deliberately undersized q/k/v: the bound is
    /// checked before any buffer is read, which is the whole point - a shape
    /// this large must be rejected before it allocates anything.
    #[test]
    #[should_panic(expected = "past the u32 dispatch ceiling")]
    fn a_score_matrix_too_large_to_address_is_refused_rather_than_wrapped() {
        let cx = Ctx::new(Some("cpu"));
        let (q, k, v) = (cx.upload(&[0.0]), cx.upload(&[0.0]), cx.upload(&[0.0]));
        attention_trio(&cx, &q, &k, &v, 9900, 56, 128);
    }

    /// The fused flash path must compute the SAME attention as the
    /// materialized trio, not merely something close: the same
    /// `1/sqrt(head_dim)` scale, the same unmasked bidirectional coverage, the
    /// same `[seq_len, heads*head_dim]` output layout. Only the accumulation
    /// ORDER differs - the online softmax rescales as it streams where the
    /// trio normalizes once - so the two must agree to fp32 rounding, and
    /// nothing beyond rounding is being traded away for the memory win.
    ///
    /// Runs at the REAL `head_dim` of 128 - the flash family's hard ceiling,
    /// which `H3TransformerConfig::real()` sits exactly on, not under - and at
    /// sequence lengths deliberately coprime to the kernels' tiles: none is a
    /// multiple of the `BR` query tile (64, or 128 on the `reg2` rung) or of
    /// the `BC = 16` key tile, so the partial-tile guards and the zero-filled
    /// tail rows are exercised rather than skipped. 37 is also shorter than a
    /// single query tile, the degenerate case where one workgroup covers the
    /// whole sequence.
    ///
    /// GPU-only by construction: the flash family needs workgroup barriers the
    /// Cranelift CPU JIT cannot split a kernel body at, so on the default
    /// `cpu` device both arms would BE the trio and the comparison would be
    /// vacuous. Point `BRAIN_MINIMAXH3_TEST_DEVICE` at a real device to run
    /// it; it skips rather than passing silently otherwise.
    #[test]
    fn flash_attention_matches_the_materialized_trio_numerically() {
        let Ok(device) = std::env::var("BRAIN_MINIMAXH3_TEST_DEVICE") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_TEST_DEVICE not set - the flash family is GPU-only, so there is nothing to compare against on the CPU JIT");
            return;
        };
        let cx = Ctx::new(Some(&device));
        if attn_mode(&cx.gpu, 128) != AttnMode::Flash {
            brain_testutil::skip(&format!("device {device:?} cannot run the flash-attention family - both arms would be the trio"));
            return;
        }

        let (heads, head_dim) = (4u32, 128u32);
        let inner = heads * head_dim;
        for &seq_len in &[37u32, 100, 200] {
            let n = (seq_len * inner) as usize;
            let v_host = fill(n, 3);
            let (q, k, v) = (cx.upload(&fill(n, 1)), cx.upload(&fill(n, 2)), cx.upload(&v_host));
            let want = cx.gpu.read(&attention_trio(&cx, &q, &k, &v, seq_len, heads, head_dim), n);
            let got = cx.gpu.read(&attention_flash(&cx, &q, &k, &v, seq_len, heads, head_dim), n);
            assert_mixing(&want, &v_host, seq_len);
            let (cos, maxabs) = cos_and_maxabs(&want, &got);
            println!("seq_len {seq_len}: cosine {cos:.10}, max_abs {maxabs:.3e}");
            assert!(cos > 0.999999, "seq_len {seq_len}: flash vs trio cosine {cos:.10} (max_abs {maxabs:.3e})");
            assert!(maxabs < 1e-4, "seq_len {seq_len}: flash vs trio max_abs {maxabs:.3e} (cosine {cos:.10})");
        }
    }
}
