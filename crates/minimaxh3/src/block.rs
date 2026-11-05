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

use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{Ops, Weight};

use crate::config::{H3TransformerConfig, MODALITY_NUM};

const K_BIAS_ADD: usize = 0;
const K_RMSNORM_ROWS: usize = 1;
const K_ROPE2D_PARTIAL: usize = 2;
const K_ATTN_SCORES_QK: usize = 3;
const K_ATTN_SOFTMAX_BIDIR: usize = 4;
const K_ATTN_APPLY_FULL: usize = 5;
const K_SILU_MUL: usize = 6;
const K_EMBED: usize = 7;
const K_GATE_ROWS: usize = 8;
const K_MODULATE_ROWS: usize = 9;
const K_ADD2: usize = 10;
const K_ROW_SCATTER: usize = 11;
/// Packs the three separate `[seq_len, inner]` q/k/v projections into the one
/// `[seq_len, 3*inner]` slab every kernel in the flash family reads through
/// `qkv_stride`/`q_off`/`k_off`/`v_off`. Only [`attention_flash`] dispatches
/// it - the materialized trio reads q/k/v as three bindings and needs no slab.
const K_PACK_QKV: usize = 12;
/// The four interchangeable bidirectional flash-attention rungs, registered as
/// a complete set so `model::block::flash_bidir_variant` can walk the whole
/// ladder from this crate's queried `DeviceCaps` (see [`FLASH_IDS`]).
const K_FLASH_ATTN_BIDIR: usize = 13;
const K_FLASH_ATTN_BIDIR_SPLIT: usize = 14;
const K_FLASH_ATTN_BIDIR_REG: usize = 15;
const K_FLASH_ATTN_BIDIR_REG2: usize = 16;

pub const KERNELS: [(&str, &str); 17] = [
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("attn_scores_qk", kernels::ATTN_SCORES_QK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("silu_mul", kernels::SILU_MUL),
    ("embed", kernels::EMBED),
    ("gate_rows", kernels::GATE_ROWS),
    ("modulate_rows", kernels::MODULATE_ROWS),
    ("add2", kernels::ADD2),
    ("row_scatter", kernels::ROW_SCATTER),
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

/// Every kernel a [`Ctx`]'s device registers: this crate's own [`KERNELS`]
/// at their fixed indices, then the `model::ops` façade set its linears
/// dispatch through (a name already registered keeps its source - both lists
/// take them from the same `kernels` catalogue).
pub fn pipelines() -> Vec<(&'static str, &'static str)> {
    let mut v = KERNELS.to_vec();
    for &entry in model::ops::kernel_list() {
        if !v.iter().any(|(name, _)| *name == entry.0) {
            v.push(entry);
        }
    }
    v
}

/// One open device + the shared kernel table - every op below dispatches
/// through this, matching `crate::vocoder::Ctx`'s own role.
pub struct Ctx {
    pub gpu: Gpu,
    /// The linears' dispatch: `model::ops` owns the kernel choice for each
    /// weight's tier, on this same device.
    pub ops: Ops,
    /// Reused device buffers for [`load_block_streaming`]'s uniform
    /// per-block shapes - see that function's doc for why this exists.
    /// `None` until the streaming block loop's first block; every model
    /// crate's other `Ctx` use (eager resident load, video VAE, vocoder)
    /// never touches this field.
    block_slots: std::cell::RefCell<Option<BlockWeights>>,
    /// The self-attention arm, when pinned: a memory or speed A/B sets it.
    /// `None` takes [`attn_mode`]'s pick for each call's head width.
    pub attn: Option<AttnMode>,
}

impl Ctx {
    pub fn new(device: Option<&str>) -> Ctx {
        let gpu = Gpu::open(device, &pipelines());
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("minimaxh3: {e}"));
        // The quantized activations are allocated on this shared handle, once
        // per linear group of every block; holding its freed blocks lets each
        // block take the previous one's (see `Gpu::hold_freed_blocks`).
        ops.gpu().hold_freed_blocks(true);
        Ctx { gpu, ops, block_slots: std::cell::RefCell::new(None), attn: None }
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

/// `y = x @ w^T (+ bias)` for `m` rows of `x`. `w` is `[n, k]` (PyTorch
/// `nn.Linear` convention) at whatever tier it was loaded at; `bias` is `[n]`
/// or `None` for the attention/FFN linears, which the reference builds with
/// `bias=False` throughout.
pub(crate) fn linear(cx: &Ctx, x: &DeviceBuffer, w: &Weight, bias: Option<&DeviceBuffer>, m: u32) -> DeviceBuffer {
    let [y] = linears(cx, x, [w], m);
    if let Some(b) = bias {
        cx.gpu.submit(&[], &[cx.gpu.step(K_BIAS_ADD, &[&y, b], &[m, w.n()], m * w.n())]);
    }
    y
}

/// Every linear in `ws` over the same `[m, k]` input - q/k/v over one
/// modulated row block, the SwiGLU halves over another. `model::ops` picks
/// each product's kernel from its weight's tier and the shape; the input is
/// quantized once for all of them when any weight is quantized, and read as
/// f32 when none is.
pub(crate) fn linears<const N: usize>(cx: &Ctx, x: &DeviceBuffer, ws: [&Weight; N], m: u32) -> [DeviceBuffer; N] {
    let k = ws[0].k();
    let mut steps = Vec::new();
    let act = if ws.iter().all(|w| w.reads_f32_activations()) { cx.ops.act_f32(x, 0, m, k) } else { cx.ops.act(&mut steps, x, 0, m, k) };
    let ys = ws.map(|w| {
        assert_eq!(w.k(), k, "minimaxh3 linears: every weight must read the same input width");
        let y = cx.gpu.storage((m * w.n()) as u64);
        cx.ops.matmul(&mut steps, w, &act, &y, 0);
        y
    });
    cx.gpu.submit(&[], &steps);
    ys
}

/// RMSNorm, one row per token: `d` channels, `rows` tokens. Dispatches the
/// shared `rmsnorm_rows` (one 64-thread workgroup per row) rather than the
/// per-thread `rmsnorm`: this crate's call sites are both narrow-row regimes
/// (`rows = seq_len` at `d = hidden`, and `rows = seq_len * heads` at
/// `d = head_dim`), exactly where the per-element variant's uncoalesced reads
/// measured worst (`rmsnorm_rows.wgsl`'s own sweep note).
pub(crate) fn rmsnorm(cx: &Ctx, x: &DeviceBuffer, w: &DeviceBuffer, rows: u32, d: u32, eps: f32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.dispatch(K_RMSNORM_ROWS, &[x, w, &y], &[d, rows, gpu_core::f(eps)], gpu_core::Dispatch::Workgroups(rows))]);
    y
}

/// In-place partial RoPE on `buf` (`[rows, heads*head_dim]`, contiguous
/// per-head layout), against precomputed `[rows, half]` `cos`/`sin` tables
/// (`crate::rope::build_tables`). Rotates the leading `2*half` channels of
/// every head; the remaining `head_dim - 2*half` pass through unchanged.
///
/// Dispatches through `model::block::rope2d_partial_fwd` (the shared builder
/// `qwen3vl`/`minimaxmusic3`/`wan` route their `rope2d_partial` through) rather
/// than hand-building the kernel's eight params here - this table's contiguous
/// per-axis layout differs from `qwen3vl::mrope`'s interleaved stride-3 map and
/// from flux's interleaved-pair convention (`crate::rope`'s own doc), but the
/// consuming KERNEL and its param contract are the shared one.
#[allow(clippy::too_many_arguments)]
fn rope_partial(cx: &Ctx, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32, head_dim: u32, half: u32) {
    cx.gpu.submit(&[], &[model::block::rope2d_partial_fwd(&cx.gpu, K_ROPE2D_PARTIAL, buf, cos, sin, rows, heads, half, heads * head_dim, 0, head_dim)]);
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
pub(crate) fn attention(gpu: &Gpu, ids: &AttnKernels, mode: AttnMode, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    match mode {
        AttnMode::Flash => attention_flash(gpu, ids, q, k, v, seq_len, heads, head_dim),
        AttnMode::Materialized => attention_trio(gpu, ids, q, k, v, seq_len, heads, head_dim),
    }
}

/// Where a device's kernel table holds the attention kernels - each module
/// that attends (the DiT's blocks, the video VAE's decoder) registers them at
/// its own indices and runs the one [`attention`].
pub(crate) struct AttnKernels {
    pub scores_qk: usize,
    pub softmax_bidir: usize,
    pub apply_full: usize,
    pub pack_qkv: usize,
    pub flash: model::block::FlashIds,
}

/// The DiT blocks' own [`AttnKernels`], in [`KERNELS`].
const DIT_ATTN: AttnKernels = AttnKernels { scores_qk: K_ATTN_SCORES_QK, softmax_bidir: K_ATTN_SOFTMAX_BIDIR, apply_full: K_ATTN_APPLY_FULL, pack_qkv: K_PACK_QKV, flash: FLASH_IDS };

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
/// An A/B pins [`Ctx::attn`] instead: on a device that can run flash,
/// comparing this pick against itself would report a meaningless parity.
pub fn attn_mode(gpu: &Gpu, head_dim: u32) -> AttnMode {
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
fn attention_trio(gpu: &Gpu, ids: &AttnKernels, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    let inner = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_scores = heads as u64 * seq_len as u64 * seq_len as u64;
    assert!(
        n_scores <= u32::MAX as u64,
        "minimaxh3 attention: materialized scores need {n_scores} elements at seq_len {seq_len} x {heads} heads, \
         past the u32 dispatch ceiling. This shape needs the fused flash path (a device with workgroup reductions, \
         and the attention arm not pinned to the materialized trio)."
    );
    let scores = gpu.storage(n_scores);
    gpu.submit(&[], &[gpu.step(ids.scores_qk, &[q, k, &scores], &[1, heads, seq_len, head_dim, inner, 0, gpu_core::f(scale)], n_scores as u32)]);
    let probs = gpu.storage(n_scores);
    gpu.submit(&[], &[gpu.step(ids.softmax_bidir, &[&scores, &probs], &[1, heads, seq_len], heads * seq_len)]);
    let out = gpu.storage((seq_len * inner) as u64);
    gpu.submit(&[], &[gpu.step(ids.apply_full, &[&probs, v, &out], &[1, heads, seq_len, head_dim, inner, inner], heads * seq_len * head_dim)]);
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
fn attention_flash(gpu: &Gpu, ids: &AttnKernels, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    let inner = heads * head_dim;
    let qkv = gpu.storage((seq_len * 3 * inner) as u64);
    gpu.submit(&[], &[gpu.step(ids.pack_qkv, &[q, k, v, &qkv], &[seq_len, inner], seq_len * 3 * inner)]);
    let out = gpu.storage((seq_len * inner) as u64);
    gpu.submit(&[], &[model::block::flash_bidir_step(gpu, ids.flash, 1, heads, seq_len, head_dim, inner, &qkv, &out)]);
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

/// `y = a + b`, fresh buffer.
pub(crate) fn add2(cx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(n as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// `y[r,:] = normed[r,:] * (1 + scale[idx[r],:]) + shift[idx[r],:]`, fresh
/// buffer - the AdaLN-Zero pre-norm modulation, read straight from the
/// distinct-row tables (see [`block_forward`]; also used by `crate::model`
/// for `norm_out`'s per-timestep, non-gated modulation). The per-token
/// scale/shift are never materialized as `[rows, d]` buffers.
pub(crate) fn modulate(cx: &Ctx, normed: &DeviceBuffer, idx: &DeviceBuffer, scale: &DeviceBuffer, shift: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_MODULATE_ROWS, &[normed, scale, shift, idx, &y], &[rows, d], rows * d)]);
    y
}

/// `y[r,:] = x[r,:] + gate[idx[r],:] * h[r,:]` - the AdaLN-Zero gated
/// residual merge, the gate read from its distinct-row table.
fn gated_residual(cx: &Ctx, x: &DeviceBuffer, idx: &DeviceBuffer, gate: &DeviceBuffer, h: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_GATE_ROWS, &[x, gate, idx, h, &y], &[rows, d], rows * d)]);
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
/// fused `qkv_proj` tensor the same way is deferred until real-checkpoint
/// import is wired up; the forward math is identical either way.
/// `adaln_proj` is not here at all: its output for a timestep is a constant
/// row, cached by [`crate::adaln::AdalnCache`] and handed to
/// [`block_forward`] per forward.
#[derive(Clone)]
pub struct BlockWeights {
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub norm_q: DeviceBuffer,
    pub norm_k: DeviceBuffer,
    pub norm1: DeviceBuffer,
    pub norm2: DeviceBuffer,
    pub fc1_value: Weight,
    pub fc1_gate: Weight,
    pub fc2: Weight,
}

/// One `MiniMaxH3TokenRefinerBlock`'s weights - the same attention/FFN
/// shape as [`BlockWeights`] with no AdaLN and no RoPE.
#[derive(Clone)]
pub struct RefinerBlockWeights {
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub norm_q: DeviceBuffer,
    pub norm_k: DeviceBuffer,
    pub norm1: DeviceBuffer,
    pub norm2: DeviceBuffer,
    pub fc1_value: Weight,
    pub fc1_gate: Weight,
    pub fc2: Weight,
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

/// One `[n, k]` linear at `tier`, read in bounded chunks
/// (`model::ops::Weight::upload_from`); the device's own capability has the
/// last word on the tier (`Dtype::promote`).
pub fn load_linear(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, name: &str, n: u32, k: u32, tier: Dtype) -> Weight {
    let w = Weight::upload_from(&ctx.ops, tensors, name, n as usize, k as usize, tier).unwrap_or_else(|e| panic!("minimaxh3 model: {e}"));
    tensors.advise_drop(name);
    w
}

/// The fused SwiGLU projection's two output-feature halves (value rows
/// first, then gate) as two linears, each read straight out of its row range
/// of the one checkpoint tensor by a declarative slice plan - never the
/// whole fused tensor at once.
pub fn load_fc1(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, prefix: &str, hidden: u32, ffn: u32, tier: Dtype) -> (Weight, Weight) {
    let name = format!("{prefix}.ff.net.0.proj.weight");
    let half = (ffn * hidden) as usize;
    let plan = [("value", 0), ("gate", half)].into_iter().map(|(part, start)| (part.to_string(), checkpoint::remap::Fetch::Slice { name: name.clone(), start, len: half })).collect();
    let halves = checkpoint::remap::RemapSource::new(tensors, plan);
    let value = Weight::upload_from(&ctx.ops, &halves, "value", ffn as usize, hidden as usize, tier).unwrap_or_else(|e| panic!("minimaxh3 model: {name}: {e}"));
    let gate = Weight::upload_from(&ctx.ops, &halves, "gate", ffn as usize, hidden as usize, tier).unwrap_or_else(|e| panic!("minimaxh3 model: {name}: {e}"));
    tensors.advise_drop(&name);
    (value, gate)
}

/// The attention projections and q/k norms shared by both block kinds'
/// `.attn` sub-module naming (`to_q`/`to_k`/`to_v`/`norm_q`/`norm_k`/`to_out.0`).
pub struct AttnWeights {
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub norm_q: DeviceBuffer,
    pub norm_k: DeviceBuffer,
    pub wo: Weight,
}

pub fn load_attn(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, prefix: &str, hidden: u32, inner: u32, tier: Dtype) -> AttnWeights {
    AttnWeights {
        wq: load_linear(tensors, ctx, &format!("{prefix}.to_q.weight"), inner, hidden, tier),
        wk: load_linear(tensors, ctx, &format!("{prefix}.to_k.weight"), inner, hidden, tier),
        wv: load_linear(tensors, ctx, &format!("{prefix}.to_v.weight"), inner, hidden, tier),
        norm_q: load_dev(tensors, ctx, &format!("{prefix}.norm_q.weight")),
        norm_k: load_dev(tensors, ctx, &format!("{prefix}.norm_k.weight")),
        wo: load_linear(tensors, ctx, &format!("{prefix}.to_out.0.weight"), hidden, inner, tier),
    }
}

/// Load `transformer_blocks.{index}`'s weights - a single [`BlockWeights`],
/// its linears at `tier`, nothing else. A caller processing blocks one at a
/// time drops the returned value (freeing its device buffers) before loading
/// the next index.
pub fn load_block(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, index: usize, cfg: &H3TransformerConfig, tier: Dtype) -> BlockWeights {
    let p = format!("transformer_blocks.{index}");
    let (hidden, inner, ffn) = (cfg.hidden_size, cfg.inner_dim(), cfg.ffn_dim);
    let attn = load_attn(tensors, ctx, &format!("{p}.attn"), hidden, inner, tier);
    let (fc1_value, fc1_gate) = load_fc1(tensors, ctx, &p, hidden, ffn, tier);
    BlockWeights {
        wq: attn.wq,
        wk: attn.wk,
        wv: attn.wv,
        wo: attn.wo,
        norm_q: attn.norm_q,
        norm_k: attn.norm_k,
        norm1: load_dev(tensors, ctx, &format!("{p}.norm1.weight")),
        norm2: load_dev(tensors, ctx, &format!("{p}.norm2.weight")),
        fc1_value,
        fc1_gate,
        fc2: load_linear(tensors, ctx, &format!("{p}.ff.net.2.weight"), hidden, ffn, tier),
    }
}

/// [`load_dev`]'s in-place twin: overwrite an already-allocated buffer's
/// bytes instead of allocating a new one. Chunked (matching `ltxv`'s own
/// `write_f32_chunked` convention, `1<<20` words) rather than one
/// `write_f32` call for a multi-hundred-MB tensor - see `Gpu::write_at`'s
/// own doc for why an unchunked call leaves a same-size staging
/// allocation permanently resident on this engine's non-ReBAR GPUs.
fn write_dev(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, name: &str, buf: &DeviceBuffer) {
    let mut wrote = false;
    let found = tensors.with_tensor(name, &mut |data| {
        ctx.gpu.write_f32_chunked(buf, data, 1 << 20);
        wrote = true;
    });
    assert!(found, "minimaxh3 model: missing tensor {name:?}");
    debug_assert!(wrote);
    tensors.advise_drop(name);
}

/// The dense buffer a streamed slot's linear is rewritten through.
fn slot(w: &Weight) -> &DeviceBuffer {
    w.dense_buffer().expect("a streamed block's linears are dense f32 slots")
}

/// [`load_block_streaming`]'s in-place rewrite of one block's device tensors.
fn write_block(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, p: &str, hidden: u32, ffn: u32, s: &BlockWeights) {
    for (leaf, w) in [("attn.to_q", &s.wq), ("attn.to_k", &s.wk), ("attn.to_v", &s.wv), ("attn.to_out.0", &s.wo), ("ff.net.2", &s.fc2)] {
        write_dev(tensors, ctx, &format!("{p}.{leaf}.weight"), slot(w));
    }
    for (leaf, buf) in [("attn.norm_q", &s.norm_q), ("attn.norm_k", &s.norm_k), ("norm1", &s.norm1), ("norm2", &s.norm2)] {
        write_dev(tensors, ctx, &format!("{p}.{leaf}.weight"), buf);
    }
    let name = format!("{p}.ff.net.0.proj.weight");
    let half = (ffn * hidden) as usize;
    let found = tensors.with_tensor(&name, &mut |data| {
        assert_eq!(data.len(), 2 * half, "{name}: expected {} elements, got {}", 2 * half, data.len());
        ctx.gpu.write_f32_chunked(slot(&s.fc1_value), &data[..half], 1 << 20);
        ctx.gpu.write_f32_chunked(slot(&s.fc1_gate), &data[half..], 1 << 20);
    });
    assert!(found, "minimaxh3 model: missing tensor {name:?}");
    tensors.advise_drop(&name);
}

/// [`load_block`]'s reused-buffer twin for the streaming denoise loop: the
/// first call allocates the block's device tensors once, at fp32 (sized by
/// the config, uniform across every block AND every denoise step); every
/// later call overwrites those SAME buffers in place instead of allocating
/// fresh ones. Safe to call back-to-back for different block indices (and
/// across denoise steps, since the caller keeps the same `Ctx` alive for the
/// whole generation - see `H3Transformer::forward_streaming`'s own doc)
/// because the streaming loop's own `cx.gpu.poll_wait()` after every block
/// already guarantees the previous block's GPU work has fully completed -
/// and so stopped reading these buffers - before the next block's weights
/// land.
///
/// Exists because a real P40 measurement showed fresh per-block allocations
/// (50 blocks x 16 steps x 11 buffers of large create/destroy cycles on one
/// long-lived wgpu device) eventually fragment the Vulkan suballocator's
/// free list badly enough to fail an allocation outright, well below the
/// card's actual VRAM ceiling (`nvidia-smi`'s reserved-byte counter stayed
/// bounded the whole time - the fragmentation is invisible to it). This cuts
/// that to 11 allocations for the whole generation. Not used by
/// [`load_block`]'s resident callers, which need every block's buffers
/// alive at once.
pub fn load_block_streaming(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, index: usize, cfg: &H3TransformerConfig) -> BlockWeights {
    let p = format!("transformer_blocks.{index}");
    let mut slots = ctx.block_slots.borrow_mut();
    match slots.as_ref() {
        None => *slots = Some(load_block(tensors, ctx, index, cfg, Dtype::F32)),
        Some(s) => write_block(tensors, ctx, &p, cfg.hidden_size, cfg.ffn_dim, s),
    }
    let s = slots.as_ref().expect("filled above");
    BlockWeights {
        wq: s.wq.clone(),
        wk: s.wk.clone(),
        wv: s.wv.clone(),
        wo: s.wo.clone(),
        norm_q: s.norm_q.clone(),
        norm_k: s.norm_k.clone(),
        norm1: s.norm1.clone(),
        norm2: s.norm2.clone(),
        fc1_value: s.fc1_value.clone(),
        fc1_gate: s.fc1_gate.clone(),
        fc2: s.fc2.clone(),
    }
}

/// [`load_block`]'s `token_refiner.refiner_blocks.{index}` analogue.
pub fn load_refiner_block(tensors: &dyn checkpoint::TensorSource, ctx: &Ctx, index: usize, cfg: &H3TransformerConfig, tier: Dtype) -> RefinerBlockWeights {
    let p = format!("token_refiner.refiner_blocks.{index}");
    let (hidden, inner, ffn) = (cfg.hidden_size, cfg.inner_dim(), cfg.ffn_dim);
    let attn = load_attn(tensors, ctx, &format!("{p}.attn"), hidden, inner, tier);
    let (fc1_value, fc1_gate) = load_fc1(tensors, ctx, &p, hidden, ffn, tier);
    RefinerBlockWeights {
        wq: attn.wq,
        wk: attn.wk,
        wv: attn.wv,
        wo: attn.wo,
        norm_q: attn.norm_q,
        norm_k: attn.norm_k,
        norm1: load_dev(tensors, ctx, &format!("{p}.norm1.weight")),
        norm2: load_dev(tensors, ctx, &format!("{p}.norm2.weight")),
        fc1_value,
        fc1_gate,
        fc2: load_linear(tensors, ctx, &format!("{p}.ff.net.2.weight"), hidden, ffn, tier),
    }
}

// ---------------- forwards ----------------

/// `MiniMaxH3TokenRefinerBlock.forward`: plain pre-norm attention (no RoPE)
/// + plain pre-norm SwiGLU FFN, both residual. `x`: `[seq_len, hidden]`.
pub fn refiner_block_forward(cx: &Ctx, w: &RefinerBlockWeights, x: &DeviceBuffer, cfg: &H3TransformerConfig, seq_len: u32) -> DeviceBuffer {
    let (hidden, heads, hd, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.ffn_dim);

    let n1 = rmsnorm(cx, x, &w.norm1, seq_len, hidden, cfg.norm_eps);
    let [q, k, v] = linears(cx, &n1, [&w.wq, &w.wk, &w.wv], seq_len);
    let qn = rmsnorm(cx, &q, &w.norm_q, seq_len * heads, hd, cfg.qk_norm_eps);
    let kn = rmsnorm(cx, &k, &w.norm_k, seq_len * heads, hd, cfg.qk_norm_eps);
    let mode = cx.attn.unwrap_or_else(|| attn_mode(&cx.gpu, hd));
    let attn = attention(&cx.gpu, &DIT_ATTN, mode, &qn, &kn, &v, seq_len, heads, hd);
    let attn_out = linear(cx, &attn, &w.wo, None, seq_len);
    let x1 = add2(cx, x, &attn_out, seq_len * hidden);

    let n2 = rmsnorm(cx, &x1, &w.norm2, seq_len, hidden, cfg.norm_eps);
    let [value, gate] = linears(cx, &n2, [&w.fc1_value, &w.fc1_gate], seq_len);
    let act = swiglu(cx, &value, &gate, seq_len * ffn);
    let ff_out = linear(cx, &act, &w.fc2, None, seq_len);
    add2(cx, &x1, &ff_out, seq_len * hidden)
}

/// The six per-forward AdaLN-Zero tables `[MODALITY_NUM*num_timesteps,
/// hidden]` (shift/scale/gate for attention, then for the FFN), addressed by
/// [`crate::model::adaln_indices`] (modality-major, timestep-minor), each a
/// contiguous slice of a timestep's cached `adaln_proj` row
/// ([`crate::adaln`]).
fn adaln_tables(rows: &[&[f32]], hidden: usize) -> [Vec<f32>; 6] {
    let num_timesteps = rows.len();
    let mut tables: [Vec<f32>; 6] = Default::default();
    for t in tables.iter_mut() {
        *t = vec![0f32; MODALITY_NUM as usize * num_timesteps * hidden];
    }
    for modality in 0..MODALITY_NUM as usize {
        for (param, table) in tables.iter_mut().enumerate() {
            let feat0 = (modality * 6 + param) * hidden;
            for (t_idx, row) in rows.iter().enumerate() {
                let dst = (modality * num_timesteps + t_idx) * hidden;
                table[dst..dst + hidden].copy_from_slice(&row[feat0..feat0 + hidden]);
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
/// (`crate::rope::build_tables`); `modulation`: this block's cached
/// `adaln_proj` row for each of the forward's distinct timesteps, in
/// `timestep` order ([`crate::adaln::AdalnPlan::block_rows`]).
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
    modulation: &[&[f32]],
    cfg: &H3TransformerConfig,
    seq_len: u32,
) -> (DeviceBuffer, DeviceBuffer) {
    let (hidden, heads, hd, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.ffn_dim);
    let half = 3 * cfg.rope_freq_dim;

    assert!(!modulation.is_empty(), "block_forward: at least one timestep's modulation is required");
    let tables = adaln_tables(modulation, hidden as usize);
    let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = tables.map(|t| cx.upload(&t));

    // attention sub-block
    let n1 = rmsnorm(cx, x, &w.norm1, seq_len, hidden, cfg.norm_eps);
    let mod1 = modulate(cx, &n1, adaln_indices, &scale_msa, &shift_msa, seq_len, hidden);
    let [q, k, v] = linears(cx, &mod1, [&w.wq, &w.wk, &w.wv], seq_len);
    let qn = rmsnorm(cx, &q, &w.norm_q, seq_len * heads, hd, cfg.qk_norm_eps);
    let kn = rmsnorm(cx, &k, &w.norm_k, seq_len * heads, hd, cfg.qk_norm_eps);
    rope_partial(cx, &qn, cos, sin, seq_len, heads, hd, half);
    rope_partial(cx, &kn, cos, sin, seq_len, heads, hd, half);
    let mode = cx.attn.unwrap_or_else(|| attn_mode(&cx.gpu, hd));
    let attn = attention(&cx.gpu, &DIT_ATTN, mode, &qn, &kn, &v, seq_len, heads, hd);
    let attn_out = linear(cx, &attn, &w.wo, None, seq_len);
    let h1 = gated_residual(cx, x, adaln_indices, &gate_msa, &attn_out, seq_len, hidden);

    // FFN sub-block
    let n2 = rmsnorm(cx, &h1, &w.norm2, seq_len, hidden, cfg.norm_eps);
    let mod2 = modulate(cx, &n2, adaln_indices, &scale_mlp, &shift_mlp, seq_len, hidden);
    let [value, gate] = linears(cx, &mod2, [&w.fc1_value, &w.fc1_gate], seq_len);
    let act = swiglu(cx, &value, &gate, seq_len * ffn);
    let ff_out = linear(cx, &act, &w.fc2, None, seq_len);
    let out = gated_residual(cx, &h1, adaln_indices, &gate_mlp, &ff_out, seq_len, hidden);
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
        attention_trio(&cx.gpu, &DIT_ATTN, &q, &k, &v, 9900, 56, 128);
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
    /// Cranelift CPU JIT cannot split a kernel body at, so there both arms
    /// would BE the trio and the comparison would be vacuous. It runs on the ambient device and skips, rather than passing
    /// silently, where that device cannot run flash.
    #[test]
    fn flash_attention_matches_the_materialized_trio_numerically() {
        let cx = Ctx::new(None);
        if attn_mode(&cx.gpu, 128) != AttnMode::Flash {
            brain_testutil::skip("the ambient device cannot run the flash-attention family - both arms would be the trio");
            return;
        }

        let (heads, head_dim) = (4u32, 128u32);
        let inner = heads * head_dim;
        for &seq_len in &[37u32, 100, 200] {
            let n = (seq_len * inner) as usize;
            let v_host = fill(n, 3);
            let (q, k, v) = (cx.upload(&fill(n, 1)), cx.upload(&fill(n, 2)), cx.upload(&v_host));
            let want = cx.gpu.read(&attention_trio(&cx.gpu, &DIT_ATTN, &q, &k, &v, seq_len, heads, head_dim), n);
            let got = cx.gpu.read(&attention_flash(&cx.gpu, &DIT_ATTN, &q, &k, &v, seq_len, heads, head_dim), n);
            assert_mixing(&want, &v_host, seq_len);
            let (cos, maxabs) = cos_and_maxabs(&want, &got);
            println!("seq_len {seq_len}: cosine {cos:.10}, max_abs {maxabs:.3e}");
            assert!(cos > 0.999999, "seq_len {seq_len}: flash vs trio cosine {cos:.10} (max_abs {maxabs:.3e})");
            assert!(maxabs < 1e-4, "seq_len {seq_len}: flash vs trio max_abs {maxabs:.3e} (cosine {cos:.10})");
        }
    }
}
