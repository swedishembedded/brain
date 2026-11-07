// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The kernel table and the small recorded steps every Ripple graph is built
//! from, shared by the prefill layer ([`crate::layer`]) and the serving engine
//! ([`crate::serve`]).
//!
//! Swedish Embedded AB implements solutions for porting hybrid local/global
//! attention decoders to portable GPU and CPU backends for its clients. If
//! your team needs expertise in kernel graphs for gated attention and clipped
//! feed-forward layers then you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

pub(crate) const K_MATMUL: usize = 0;
pub(crate) const K_RMSNORM: usize = 1;
pub(crate) const K_ROPE2D: usize = 2;
pub(crate) const K_GQA_SCORES_WIN: usize = 3;
pub(crate) const K_ATTN_SOFTMAX: usize = 4;
pub(crate) const K_GQA_APPLY: usize = 5;
pub(crate) const K_SIGMOID: usize = 6;
pub(crate) const K_MUL: usize = 7;
pub(crate) const K_ADD2: usize = 8;
pub(crate) const K_SILU_MUL: usize = 9;
pub(crate) const K_CLAMP_RANGE: usize = 10;
pub(crate) const K_MIX: usize = 11;
pub(crate) const K_ROPE_PAGED: usize = 12;
pub(crate) const K_KV_APPEND: usize = 13;
pub(crate) const K_SCORES: usize = 14;
pub(crate) const K_SOFTMAX: usize = 15;
pub(crate) const K_APPLY: usize = 16;
pub(crate) const K_COPY_WITHIN: usize = 17;

/// The kernels the prefill forward dispatches, in the order of the `K_*` indices.
pub const KERNELS: [(&str, &str); 12] = [
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope2d", kernels::ROPE2D),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("sigmoid", kernels::SIGMOID),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("silu_mul", kernels::SILU_MUL),
    ("clamp_range", kernels::CLAMP_RANGE),
    // `y = a[row] * x + b[row] * f` with per-row coefficients: the value
    // residual's lerp, `x0 + s * (xn - x0) = (1 - s) * x0 + s * xn`.
    ("edm_mix", kernels::EDM_MIX),
];

/// The kernels the serving engine adds to [`KERNELS`]: per-row RoPE positions,
/// paged KV append, and the paged decode attention triad (scores, softmax,
/// apply), all of which take a per-row block table and key count.
pub(crate) const SERVE_EXTRA: [(&str, &str); 6] = [
    ("rope_paged", kernels::ROPE_PAGED),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("copy_within", kernels::COPY_WITHIN),
];

/// `out[m, n] = x[m, k] @ w[n, k]^T`.
pub(crate) fn linear(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, m: u32, k: u32, n: u32) -> DeviceBuffer {
    let out = gpu.storage(u64::from(m) * u64::from(n));
    s.push(gpu.step(K_MATMUL, &[x, w, &out], &[m, k, n], m * n));
    out
}

pub(crate) fn rmsnorm(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &DeviceBuffer, dim: u32, rows: u32, eps: f32) -> DeviceBuffer {
    let out = gpu.storage(u64::from(dim) * u64::from(rows));
    s.push(gpu.step(K_RMSNORM, &[x, gain, &out], &[dim, rows, f(eps)], rows));
    out
}

/// `out = a (op) b` over `n` elements for an elementwise two-input kernel
/// (`mul`, `add2`, `silu_mul`).
pub(crate) fn binary(gpu: &Gpu, s: &mut Vec<Step>, kernel: usize, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let out = gpu.storage(u64::from(n));
    s.push(gpu.step(kernel, &[a, b, &out], &[n], n));
    out
}

pub(crate) fn clamp(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, lo: f32, hi: f32, n: u32) -> DeviceBuffer {
    let out = gpu.storage(u64::from(n));
    s.push(gpu.step(K_CLAMP_RANGE, &[x, &out], &[n, f(lo), f(hi)], n));
    out
}

pub(crate) fn sigmoid(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let out = gpu.storage(u64::from(n));
    s.push(gpu.step(K_SIGMOID, &[x, &out], &[n], n));
    out
}

/// The value residual's input, `(1 - mix) * x0 + mix * xn` over `[rows, hidden]`.
pub(crate) fn value_mix(gpu: &Gpu, s: &mut Vec<Step>, x0: &DeviceBuffer, xn: &DeviceBuffer, mix: f32, rows: u32, hidden: u32) -> DeviceBuffer {
    // The same coefficients on every row.
    let coeffs: Vec<f32> = (0..rows).flat_map(|_| [1.0 - mix, mix]).collect();
    let ab = gpu.storage(coeffs.len() as u64);
    gpu.write_f32(&ab, &coeffs);
    let out = gpu.storage(u64::from(rows) * u64::from(hidden));
    s.push(gpu.step(K_MIX, &[x0, xn, &ab, &out], &[rows * hidden, hidden], rows * hidden));
    out
}

/// The feed-forward sub-block after the attention residual: `x1 + post_norm(
/// down(silu(min(gate(m), c)) * clamp(up(m), -c, c)))` with `m = pre_norm(x1)`.
pub(crate) struct FeedForward<'a> {
    pub pre_ln: &'a DeviceBuffer,
    pub post_ln: &'a DeviceBuffer,
    pub gate: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub down: &'a DeviceBuffer,
    pub hidden: u32,
    pub inter: u32,
    pub eps: f32,
    pub clip: Option<f32>,
}

impl FeedForward<'_> {
    pub(crate) fn record(&self, gpu: &Gpu, s: &mut Vec<Step>, x1: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let m = rmsnorm(gpu, s, x1, self.pre_ln, self.hidden, rows, self.eps);
        let mut gate = linear(gpu, s, &m, self.gate, rows, self.hidden, self.inter);
        let mut up = linear(gpu, s, &m, self.up, rows, self.hidden, self.inter);
        if let Some(limit) = self.clip {
            gate = clamp(gpu, s, &gate, -f32::MAX, limit, rows * self.inter);
            up = clamp(gpu, s, &up, -limit, limit, rows * self.inter);
        }
        let act = binary(gpu, s, K_SILU_MUL, &gate, &up, rows * self.inter);
        let mlp = linear(gpu, s, &act, self.down, rows, self.inter, self.hidden);
        let mlp_normed = rmsnorm(gpu, s, &mlp, self.post_ln, self.hidden, rows, self.eps);
        binary(gpu, s, K_ADD2, x1, &mlp_normed, rows * self.hidden)
    }
}
