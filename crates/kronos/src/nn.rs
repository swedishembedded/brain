// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared neural-net plumbing for both Kronos nets: the kernel pipeline, per-op
//! device helpers, and the `TransformerBlock` forward (pre-norm RMSNorm →
//! causal-scaled MHA with NeoX RoPE → residual → RMSNorm → SwiGLU → residual)
//! reused by the tokenizer's encoder/decoder blocks AND the AR decoder's blocks.
//!
//! Immediate per-op submits (as in `chronos2::model`) sidestep dynamic-length
//! buffer-lifetime issues; correctness of each kernel is covered by the
//! isolation tests, so this layer only wires them together.

use gpu_core::{f, DeviceBuffer, Gpu};
use std::collections::HashMap;

// Kernel pipeline indices (order must match PIPELINES).
pub const MATMUL: usize = 0;
pub const BIAS_ADD: usize = 1;
pub const RMSNORM: usize = 2;
pub const ROPE_NEOX: usize = 3;
pub const ATTN_SCORES_QK: usize = 4;
pub const ATTN_SOFTMAX_FULL: usize = 5;
pub const ATTN_APPLY_FULL: usize = 6;
pub const SILU_GATE: usize = 7;
pub const ADD: usize = 8;
pub const BSQ_QUANTIZE: usize = 9;
pub const MATMUL_TILED: usize = 10;

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_neox", kernels::ROPE_NEOX),
    ("attn_scores_qk", kernels::ATTN_SCORES_QK),
    ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("silu_gate", kernels::SILU_GATE),
    ("add", kernels::ADD),
    ("bsq_quantize", kernels::BSQ_QUANTIZE),
    ("matmul_tiled", kernels::MATMUL_TILED),
];

/// Workgroups for the tiled GEMM (32×32 output tile) → invocation count.
#[inline]
fn tiled_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(32) * n.div_ceil(32) * 64) as u32
}

/// A per-op executor over a GPU + a named weight map. Cheap to construct per
/// forward; holds no state beyond the borrows.
pub struct Ops<'a> {
    pub gpu: &'a Gpu,
    pub w: &'a HashMap<String, DeviceBuffer>,
    pub rope_theta: f32,
}

impl<'a> Ops<'a> {
    pub fn wt(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("kronos: weight {name} not loaded"))
    }

    /// `out = x @ W^T`, x[m,k] W[n,k] out[m,n].
    pub fn mm(&self, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) {
        // Tiled GEMM for large-m matmuls; naive for small-m (its 32-row tiles
        // would be mostly padding). See the note in chronos2::model::mm.
        let (kind, threads) =
            if m >= 64 { (MATMUL_TILED, tiled_threads(m, n)) } else { (MATMUL, (m * n) as u32) };
        let s = self.gpu.step(kind, &[x, self.wt(wname), out], &[m as u32, k as u32, n as u32], threads);
        self.gpu.submit(&[], &[s]);
    }
    pub fn bias(&self, out: &DeviceBuffer, bname: &str, m: usize, n: usize) {
        let s = self.gpu.step(BIAS_ADD, &[out, self.wt(bname)], &[m as u32, n as u32], (m * n) as u32);
        self.gpu.submit(&[], &[s]);
    }
    /// Linear with bias: `out = x@W^T + b` into a fresh buffer.
    pub fn linear(&self, x: &DeviceBuffer, wname: &str, bname: &str, m: usize, k: usize, n: usize) -> DeviceBuffer {
        let out = self.gpu.storage((m * n) as u64);
        self.mm(x, wname, &out, m, k, n);
        self.bias(&out, bname, m, n);
        out
    }
    pub fn rms(&self, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, d: usize, rows: usize) {
        let s = self.gpu.step(RMSNORM, &[x, self.wt(wname), out], &[d as u32, rows as u32], rows as u32);
        self.gpu.submit(&[], &[s]);
    }
    pub fn add(&self, src: &DeviceBuffer, dst: &DeviceBuffer, total: usize) {
        let s = self.gpu.step(ADD, &[src, dst], &[total as u32], total as u32);
        self.gpu.submit(&[], &[s]);
    }
    pub fn rope(&self, buf: &DeviceBuffer, s: usize, heads: usize, hd: usize) {
        let row_stride = heads * hd;
        let st = self.gpu.step(
            ROPE_NEOX,
            &[buf],
            &[s as u32, heads as u32, hd as u32, row_stride as u32, 0, f(self.rope_theta)],
            (s * heads * (hd / 2)) as u32,
        );
        self.gpu.submit(&[], &[st]);
    }
    pub fn silu_gate(&self, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, total: usize) {
        let s = self.gpu.step(SILU_GATE, &[a, b, out], &[total as u32], total as u32);
        self.gpu.submit(&[], &[s]);
    }

    /// One `TransformerBlock`, in place on `x` [S, d]. Self-attention is causal +
    /// scaled with NeoX RoPE; FFN is SwiGLU. `prefix` is e.g. `encoder.0` /
    /// `transformer.3`.
    pub fn transformer_block(&self, prefix: &str, x: &DeviceBuffer, s: usize, d: usize, ff: usize, heads: usize) {
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();

        // --- self attention (causal, scaled) ---
        let xn = self.gpu.storage((s * d) as u64);
        self.rms(x, &format!("{prefix}.norm1.weight"), &xn, d, s);
        let q = self.linear(&xn, &format!("{prefix}.self_attn.q_proj.weight"), &format!("{prefix}.self_attn.q_proj.bias"), s, d, d);
        let k = self.linear(&xn, &format!("{prefix}.self_attn.k_proj.weight"), &format!("{prefix}.self_attn.k_proj.bias"), s, d, d);
        let v = self.linear(&xn, &format!("{prefix}.self_attn.v_proj.weight"), &format!("{prefix}.self_attn.v_proj.bias"), s, d, d);
        self.rope(&q, s, heads, hd);
        self.rope(&k, s, heads, hd);
        let scores = self.gpu.storage((heads * s * s) as u64);
        let sc = self.gpu.step(
            ATTN_SCORES_QK,
            &[&q, &k, &scores],
            &[1, heads as u32, s as u32, hd as u32, d as u32, 1, f(scale)],
            (heads * s * s) as u32,
        );
        self.gpu.submit(&[], &[sc]);
        let probs = self.gpu.storage((heads * s * s) as u64);
        let sm = self.gpu.step(ATTN_SOFTMAX_FULL, &[&scores, &probs], &[1, heads as u32, s as u32], (heads * s) as u32);
        self.gpu.submit(&[], &[sm]);
        let ctx = self.gpu.storage((s * d) as u64);
        let ap = self.gpu.step(
            ATTN_APPLY_FULL,
            &[&probs, &v, &ctx],
            &[1, heads as u32, s as u32, hd as u32, d as u32, d as u32],
            (heads * s * hd) as u32,
        );
        self.gpu.submit(&[], &[ap]);
        let o = self.linear(&ctx, &format!("{prefix}.self_attn.out_proj.weight"), &format!("{prefix}.self_attn.out_proj.bias"), s, d, d);
        self.add(&o, x, s * d);

        // --- SwiGLU FFN (no bias) ---
        let xn2 = self.gpu.storage((s * d) as u64);
        self.rms(x, &format!("{prefix}.norm2.weight"), &xn2, d, s);
        let a = self.gpu.storage((s * ff) as u64);
        self.mm(&xn2, &format!("{prefix}.ffn.w1.weight"), &a, s, d, ff);
        let b = self.gpu.storage((s * ff) as u64);
        self.mm(&xn2, &format!("{prefix}.ffn.w3.weight"), &b, s, d, ff);
        let g = self.gpu.storage((s * ff) as u64);
        self.silu_gate(&a, &b, &g, s * ff);
        let ffo = self.gpu.storage((s * d) as u64);
        self.mm(&g, &format!("{prefix}.ffn.w2.weight"), &ffo, s, ff, d);
        self.add(&ffo, x, s * d);
    }
}

/// Load a name→values weight map into device buffers, validated against
/// `param_list` (name present + numel).
pub fn load_weights(
    gpu: &Gpu,
    param_list: &[(String, Vec<usize>)],
    weights: &HashMap<String, Vec<f32>>,
) -> Result<HashMap<String, DeviceBuffer>, String> {
    let mut w = HashMap::new();
    for (name, shape) in param_list {
        let numel: usize = shape.iter().product();
        let data = weights.get(name).ok_or_else(|| format!("kronos: missing weight {name}"))?;
        if data.len() != numel {
            return Err(format!("kronos: {name} has {} elems, expected {numel}", data.len()));
        }
        w.insert(name.clone(), gpu.storage_init(name, data));
    }
    Ok(w)
}
