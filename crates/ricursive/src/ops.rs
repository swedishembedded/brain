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
use model::ops::{Ops, Weight};

use crate::config::RicursiveConfig;

pub(crate) const K_RMSNORM: usize = 0;
pub(crate) const K_ROPE2D: usize = 1;
pub(crate) const K_GQA_SCORES_WIN: usize = 2;
pub(crate) const K_ATTN_SOFTMAX: usize = 3;
pub(crate) const K_GQA_APPLY: usize = 4;
pub(crate) const K_SIGMOID: usize = 5;
pub(crate) const K_MUL: usize = 6;
pub(crate) const K_ADD2: usize = 7;
pub(crate) const K_SILU_MUL: usize = 8;
pub(crate) const K_CLAMP_RANGE: usize = 9;
pub(crate) const K_MIX: usize = 10;
pub(crate) const K_ROPE_PAGED: usize = 11;
pub(crate) const K_SOFTMAX: usize = 12;
pub(crate) const K_COPY_WITHIN: usize = 13;

/// The `(head_dim, group)` the fused decode attention is specialised for: the model's own.
pub(crate) fn flash_shape(cfg: &RicursiveConfig) -> (u32, u32) {
    (cfg.head_dim as u32, cfg.group() as u32)
}

/// The kernels Ripple's own graph dispatches by index, in the order of the `K_*` indices. The products, the
/// KV append and the paged attention scores and apply are not here: they go through `model::ops::Ops` and
/// `model::kv_tier::KvKernels`, which resolve their kernels by name from [`pipelines`].
const OWN_KERNELS: [(&str, &str); 14] = [
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
    // Per-row RoPE positions, for the sliding layers' ring.
    ("rope_paged", kernels::ROPE_PAGED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("copy_within", kernels::COPY_WITHIN),
];

/// Every pipeline a Ripple device registers: its own kernels at the `K_*` indices, then the shared façade's set
/// (every weight tier's products and tile decode: `model::ops::training_kernel_list`), then the KV tiers'
/// kernels (`model::kv_tier::kernel_list`), then `extra` (the sparse-attention kernels). A name that two of
/// them share is registered once.
pub fn pipelines(cfg: &RicursiveConfig, extra: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    fn add(table: &mut Vec<(&'static str, &'static str)>, list: &[(&'static str, &'static str)]) {
        for &entry in list {
            if !table.iter().any(|(name, _)| *name == entry.0) {
                table.push(entry);
            }
        }
    }
    let mut table: Vec<(&'static str, &'static str)> = OWN_KERNELS.to_vec();
    add(&mut table, model::ops::training_kernel_list());
    // The fused decode attention is specialised for this model's own head shape (80 query heads over 8 kv heads of
    // 128 is a group of 10), at every KV tier; the paged triad stays the CPU JIT's path and the reference.
    let fused = model::kv_tier::kernel_list_for(&table, &[flash_shape(cfg)]);
    add(&mut table, &fused);
    add(&mut table, extra);
    table
}

/// The activation operand every product of `weights` reads from the rows of `x` (`[rows, k]`): the f32 rows when
/// every weight reads them as they are (f32, bf16, fp8, NVFP4 ...), the quantized rows when one of them is an
/// int8 tier, made once and shared by all the products that read `x`.
fn activation(ops: &Ops, s: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, k: u32, weights: &[&Weight]) -> model::ops::Act {
    if weights.iter().all(|w| w.reads_f32_activations()) {
        ops.act_f32(x, 0, rows, k)
    } else {
        ops.act(s, x, 0, rows, k)
    }
}

/// `out[rows, w.n] = x[rows, w.k] @ w^T` for a weight of any tier, through the shared `Ops` façade.
pub(crate) fn linear(ops: &Ops, s: &mut Vec<Step>, x: &DeviceBuffer, w: &Weight, rows: u32) -> DeviceBuffer {
    linears(ops, s, x, &[w], rows).remove(0)
}

/// [`linear`] for several weights that read the same rows of `x` (q, k and v; gate and up): the activation is
/// prepared once.
pub(crate) fn linears(ops: &Ops, s: &mut Vec<Step>, x: &DeviceBuffer, weights: &[&Weight], rows: u32) -> Vec<DeviceBuffer> {
    let act = activation(ops, s, x, rows, weights[0].k(), weights);
    weights
        .iter()
        .map(|w| {
            assert_eq!(w.k(), weights[0].k(), "ricursive: weights that share an activation must share its width");
            let out = ops.gpu().storage(u64::from(rows) * u64::from(w.n()));
            ops.matmul(s, w, &act, &out, 0);
            out
        })
        .collect()
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
    pub gate: &'a Weight,
    pub up: &'a Weight,
    pub down: &'a Weight,
    pub hidden: u32,
    pub inter: u32,
    pub eps: f32,
    pub clip: Option<f32>,
}

impl FeedForward<'_> {
    pub(crate) fn record(&self, ops: &Ops, s: &mut Vec<Step>, x1: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let gpu = ops.gpu();
        let m = rmsnorm(gpu, s, x1, self.pre_ln, self.hidden, rows, self.eps);
        let mut gate_up = linears(ops, s, &m, &[self.gate, self.up], rows);
        let (mut up, mut gate) = (gate_up.pop().expect("up"), gate_up.pop().expect("gate"));
        if let Some(limit) = self.clip {
            gate = clamp(gpu, s, &gate, -f32::MAX, limit, rows * self.inter);
            up = clamp(gpu, s, &up, -limit, limit, rows * self.inter);
        }
        let act = binary(gpu, s, K_SILU_MUL, &gate, &up, rows * self.inter);
        let mlp = linear(ops, s, &act, self.down, rows);
        let mlp_normed = rmsnorm(gpu, s, &mlp, self.post_ln, self.hidden, rows, self.eps);
        binary(gpu, s, K_ADD2, x1, &mlp_normed, rows * self.hidden)
    }
}
