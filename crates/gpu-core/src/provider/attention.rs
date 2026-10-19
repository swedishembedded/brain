// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Causal grouped-query attention for training, computed in query blocks around
//! the vendor library's tensor-core GEMMs.
//!
//! Swedish Embedded AB implements long-sequence attention training for its
//! clients. If your team needs expertise in training transformers at sequence
//! lengths where the attention matrix dominates the step, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The scalar flash-attention trio runs at a few TFLOP/s; this path runs the
//! two matrix products of attention (`Q K^T` and `P V`) and the three of its
//! backward (`dP`, `dQ`, `dK`/`dV`) as bf16 tensor-core GEMMs with fp32
//! accumulation, with small element-wise kernels between them. A block of `R`
//! query rows of one KV group's `g` query heads is a strided batch of `g`
//! products against the group's shared keys, so the score matrix that exists at
//! any time is `g x R x keys` (bounded by [`SCORE_BLOCK_BYTES`]), not
//! `heads x T x T`. Causal block skipping: a block only reads the keys up to its
//! own last row. The backward recomputes the scores from the saved row
//! log-sum-exp, as FlashAttention does (arXiv 2205.14135, 2307.08691); it
//! differs in materialising a score block between the GEMMs instead of keeping
//! it in registers.
//!
//! Layouts are the training mixer's: `q` `[n, heads * head_dim]`, `k`/`v`
//! `[n, kv_heads * head_dim]`, row `t` of head `h` at `t * stride + h * head_dim`.

use backend_api::{BindKind, LibraryDtype, LibraryGemm};

use super::library::{GemmPath, LibraryGemmProvider, Operand};
use crate::{DeviceBuffer, Gpu, Step};

/// Upper bound on the fp32 score block (and, in the backward, on the score
/// gradient block) one query block materialises.
pub const SCORE_BLOCK_BYTES: u64 = 192 << 20;

/// The attention problem: `b` sequences of `t` tokens, grouped-query heads.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    /// Sequences (the library path takes one).
    pub b: u32,
    /// Tokens per sequence.
    pub t: u32,
    /// Query heads.
    pub n_heads: u32,
    /// Key/value heads.
    pub n_kv_heads: u32,
    /// Head width.
    pub head_dim: u32,
    /// Query rows per block; `None` takes the most the score bound allows. Only a
    /// test that wants several blocks at a small length sets it (a multiple of 8).
    pub block_rows: Option<u32>,
}

const SOFTMAX_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite, BindKind::StorageReadWrite];
const DSCORE_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];
const ROWDOT_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// A row-major strided-batch product: `C = alpha op(A) op(B) + beta C` per batch
/// entry, with `op(A)` `m x k`, `op(B)` `k x n` and row strides `lda`/`ldb`/`ldc`.
struct Product {
    m: u32,
    n: u32,
    k: u32,
    trans_a: bool,
    trans_b: bool,
    lda: u32,
    ldb: u32,
    ldc: u32,
    batch: u32,
    stride: (i64, i64, i64),
    beta: f32,
}

impl LibraryGemmProvider {
    /// Whether the library attention path takes this problem: an active
    /// tensor-core path, one sequence, widths and lengths in whole 8-element
    /// groups (16-byte rows), a head count that divides into KV groups, and
    /// at least one query block that fits [`SCORE_BLOCK_BYTES`].
    pub fn attention_supported(&self, shape: &AttnShape) -> bool {
        self.is_active()
            && shape.b == 1
            && shape.n_kv_heads > 0
            && shape.n_heads.is_multiple_of(shape.n_kv_heads)
            && shape.t.is_multiple_of(8)
            && shape.head_dim.is_multiple_of(8)
            && shape.t > 0
            && Self::block_rows(shape) >= 8
    }

    /// Query rows per block: the most that keep the score block under the bound.
    fn block_rows(shape: &AttnShape) -> u32 {
        if let Some(rows) = shape.block_rows {
            return rows;
        }
        let g = (shape.n_heads / shape.n_kv_heads.max(1)) as u64;
        let rows = SCORE_BLOCK_BYTES / (g * shape.t as u64 * 4).max(1);
        ((rows.min(shape.t as u64) / 8) * 8) as u32
    }

    /// Run `p` over `a`, `b`, `c` (row-major) as one library step.
    fn batched(&self, gpu: &Gpu, steps: &mut Vec<Step>, a: &Operand, b: &Operand, c: &Operand, p: &Product) -> bool {
        // Row-major `C = op(A) op(B)` is column-major `C^T = op(B)^T op(A)^T`.
        let (sa, sb, sc) = p.stride;
        let gemm = LibraryGemm {
            trans_a: p.trans_b,
            trans_b: p.trans_a,
            m: p.n,
            n: p.m,
            k: p.k,
            lda: p.ldb,
            ldb: p.lda,
            ldc: p.ldc,
            a: LibraryDtype::Bf16,
            b: LibraryDtype::Bf16,
            c: LibraryDtype::F32,
            alpha: 1.0,
            beta: p.beta,
            batch: p.batch,
            stride_a: sb,
            stride_b: sa,
            stride_c: sc,
            tf32: false,
        };
        match gpu.step_library_gemm(&gemm, &[b.buf, a.buf, c.buf], &[b.range, a.range, c.range]) {
            Some(step) => {
                steps.push(step);
                true
            }
            None => false,
        }
    }

    /// Round `count` fp32 values of `src` to a new bf16 buffer.
    fn cast(&self, gpu: &Gpu, steps: &mut Vec<Step>, src: &DeviceBuffer, count: u64) -> Option<DeviceBuffer> {
        let kernel = kernels_cuda::get("cast_f32_bf16")?;
        let id = gpu.native_kernel(kernel, &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite])?;
        let out = gpu.storage(count.div_ceil(2));
        steps.push(gpu.step_native_sliced(id, &[src, &out], &[(0, count), (0, 0)], &[u32::try_from(count).ok()?], kernel.blocks_for(1, u32::try_from(count).ok()?))?);
        Some(out)
    }

    fn native(&self, gpu: &Gpu, name: &str, bindings: &'static [BindKind], bufs: &[&DeviceBuffer], params: &[u32], blocks: u32) -> Option<Step> {
        let kernel = kernels_cuda::get(name)?;
        let id = gpu.native_kernel(kernel, bindings)?;
        gpu.step_native_sliced(id, bufs, &vec![(0, 0); bufs.len()], params, blocks)
    }

    /// The forward: `ctx = softmax(causal(scale q k^T)) v` per head, and the row
    /// log-sum-exp `lse` (`[n_heads, t]`, in scaled units) the backward needs.
    /// `ctx` is overwritten. `None` (nothing to submit) when the problem is not
    /// supported or a kernel was declined; the caller runs its own attention.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fwd(&self, gpu: &Gpu, shape: &AttnShape, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, ctx: &DeviceBuffer, lse: &DeviceBuffer) -> Option<Vec<Step>> {
        if !self.attention_supported(shape) || self.path() != &GemmPath::Bf16TensorCore {
            return None;
        }
        let (t, nh, nkv, hd) = (shape.t, shape.n_heads, shape.n_kv_heads, shape.head_dim);
        let (qd, kvd, g) = (nh * hd, nkv * hd, nh / nkv);
        let scale = 1.0 / (hd as f32).sqrt();
        let rb = Self::block_rows(shape);
        let mut steps = Vec::new();
        let qb = self.cast(gpu, &mut steps, q, t as u64 * qd as u64)?;
        let kb = self.cast(gpu, &mut steps, k, t as u64 * kvd as u64)?;
        let vb = self.cast(gpu, &mut steps, v, t as u64 * kvd as u64)?;
        let s = gpu.storage(g as u64 * rb as u64 * t as u64);
        let p = gpu.storage(g as u64 * rb as u64 * t as u64 / 2);
        let off = |elems: u64, bf16: bool| (if bf16 { elems / 2 } else { elems }, 0u64);
        for r0 in (0..t).step_by(rb as usize) {
            let ri = rb.min(t - r0);
            let lk = r0 + ri;
            for kvh in 0..nkv {
                let h0 = kvh * g;
                let scores = Product { m: ri, n: lk, k: hd, trans_a: false, trans_b: true, lda: qd, ldb: kvd, ldc: lk, batch: g, stride: (hd as i64, 0, (ri * lk) as i64), beta: 0.0 };
                let q_blk = Operand { buf: &qb, range: off(r0 as u64 * qd as u64 + h0 as u64 * hd as u64, true) };
                let k_head = Operand { buf: &kb, range: off(kvh as u64 * hd as u64, true) };
                if !self.batched(gpu, &mut steps, &q_blk, &k_head, &Operand { buf: &s, range: (0, 0) }, &scores) {
                    return None;
                }
                steps.push(self.native(gpu, "attn_softmax_fwd", SOFTMAX_BINDINGS, &[&s, &p, lse], &[g, ri, lk, r0, h0, t, scale.to_bits()], g * ri)?);
                let mix = Product { m: ri, n: hd, k: lk, trans_a: false, trans_b: false, lda: lk, ldb: kvd, ldc: qd, batch: g, stride: ((ri * lk) as i64, 0, hd as i64), beta: 0.0 };
                let v_head = Operand { buf: &vb, range: off(kvh as u64 * hd as u64, true) };
                let out = Operand { buf: ctx, range: off(r0 as u64 * qd as u64 + h0 as u64 * hd as u64, false) };
                if !self.batched(gpu, &mut steps, &Operand { buf: &p, range: (0, 0) }, &v_head, &out, &mix) {
                    return None;
                }
            }
        }
        Some(steps)
    }

    /// The backward: `d_q`, `d_k`, `d_v` from `d_ctx`, the forward's inputs,
    /// output and `lse`. All three outputs must be zero on entry (`d_k`/`d_v`
    /// accumulate over query blocks); `d_q` is overwritten block by block.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_bwd(
        &self,
        gpu: &Gpu,
        shape: &AttnShape,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        ctx: &DeviceBuffer,
        lse: &DeviceBuffer,
        d_ctx: &DeviceBuffer,
        d_q: &DeviceBuffer,
        d_k: &DeviceBuffer,
        d_v: &DeviceBuffer,
    ) -> Option<Vec<Step>> {
        if !self.attention_supported(shape) || self.path() != &GemmPath::Bf16TensorCore {
            return None;
        }
        let (t, nh, nkv, hd) = (shape.t, shape.n_heads, shape.n_kv_heads, shape.head_dim);
        let (qd, kvd, g) = (nh * hd, nkv * hd, nh / nkv);
        let scale = 1.0 / (hd as f32).sqrt();
        let rb = Self::block_rows(shape);
        let mut steps = Vec::new();
        let qb = self.cast(gpu, &mut steps, q, t as u64 * qd as u64)?;
        let kb = self.cast(gpu, &mut steps, k, t as u64 * kvd as u64)?;
        let vb = self.cast(gpu, &mut steps, v, t as u64 * kvd as u64)?;
        let dob = self.cast(gpu, &mut steps, d_ctx, t as u64 * qd as u64)?;
        // D[t, h] = dO . O over the head's row.
        let dsum = gpu.storage(t as u64 * nh as u64);
        steps.push(self.native(gpu, "attn_rowdot", ROWDOT_BINDINGS, &[d_ctx, ctx, &dsum], &[t * nh, hd], t * nh)?);
        let blk = g as u64 * rb as u64 * t as u64;
        let (s, dp) = (gpu.storage(blk), gpu.storage(blk));
        let (p, ds) = (gpu.storage(blk / 2), gpu.storage(blk / 2));
        let half = |elems: u64| (elems / 2, 0u64);
        let whole = |buf| Operand { buf, range: (0, 0) };
        for r0 in (0..t).step_by(rb as usize) {
            let ri = rb.min(t - r0);
            let lk = r0 + ri;
            for kvh in 0..nkv {
                let h0 = kvh * g;
                let q_off = r0 as u64 * qd as u64 + h0 as u64 * hd as u64;
                let k_head = Operand { buf: &kb, range: half(kvh as u64 * hd as u64) };
                let v_head = Operand { buf: &vb, range: half(kvh as u64 * hd as u64) };
                // S = Q K^T and dP = dO V^T share one shape: `ri x lk` scores per head.
                let scores = Product { m: ri, n: lk, k: hd, trans_a: false, trans_b: true, lda: qd, ldb: kvd, ldc: lk, batch: g, stride: (hd as i64, 0, (ri * lk) as i64), beta: 0.0 };
                if !self.batched(gpu, &mut steps, &Operand { buf: &qb, range: half(q_off) }, &k_head, &whole(&s), &scores) {
                    return None;
                }
                if !self.batched(gpu, &mut steps, &Operand { buf: &dob, range: half(q_off) }, &v_head, &whole(&dp), &scores) {
                    return None;
                }
                steps.push(self.native(gpu, "attn_dscore", DSCORE_BINDINGS, &[&s, &dp, lse, &dsum, &p, &ds], &[g, ri, lk, r0, h0, t, scale.to_bits(), nh], g * ri)?);
                // dQ = dS K, one product per head of the group, written once.
                let dq = Product { m: ri, n: hd, k: lk, trans_a: false, trans_b: false, lda: lk, ldb: kvd, ldc: qd, batch: g, stride: ((ri * lk) as i64, 0, hd as i64), beta: 0.0 };
                let dq_out = Operand { buf: d_q, range: (q_off, 0) };
                if !self.batched(gpu, &mut steps, &whole(&ds), &k_head, &dq_out, &dq) {
                    return None;
                }
                // dK += dS^T Q and dV += P^T dO, accumulated over the group's heads.
                for j in 0..g {
                    let head_off = r0 as u64 * qd as u64 + (h0 + j) as u64 * hd as u64;
                    let rows = (j * ri * lk) as u64 / 2;
                    let acc = Product { m: lk, n: hd, k: ri, trans_a: true, trans_b: false, lda: lk, ldb: qd, ldc: kvd, batch: 1, stride: (0, 0, 0), beta: 1.0 };
                    let q_head = Operand { buf: &qb, range: half(head_off) };
                    let do_head = Operand { buf: &dob, range: half(head_off) };
                    let dk_out = Operand { buf: d_k, range: (kvh as u64 * hd as u64, 0) };
                    let dv_out = Operand { buf: d_v, range: (kvh as u64 * hd as u64, 0) };
                    if !self.batched(gpu, &mut steps, &Operand { buf: &ds, range: (rows, 0) }, &q_head, &dk_out, &acc)
                        || !self.batched(gpu, &mut steps, &Operand { buf: &p, range: (rows, 0) }, &do_head, &dv_out, &acc)
                    {
                        return None;
                    }
                }
            }
        }
        Some(steps)
    }
}
