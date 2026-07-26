// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `ZImageTransformerBlock` forward as a recorded brain kernel graph.
//!
//! Mirrors diffusers `ZImageTransformerBlock.forward` (single-stream, global
//! modulation): a double-RMSNorm sandwich per sub-block with adaLN scale on the
//! pre-norm and adaLN gate on the post-norm, QK-normalized attention with
//! multi-axis interleaved RoPE, and a SwiGLU MLP. adaLN scale/gate are folded
//! into the four RMSNorm weights on the host (see crate docs), so the device
//! graph is: rmsnorm → q/k/v → qk-norm → RoPE → pack → bidir attention →
//! out-proj → post-norm → residual → rmsnorm → SwiGLU → post-norm → residual.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Z-Image RMSNorm epsilon (diffusers `norm_eps` / attention `eps`).
const EPS: f32 = 1e-5;

// Kernel-table indices (order matches KERNELS).
const K_RMSNORM: usize = 0;
const K_MATMUL: usize = 1;
const K_ROPE: usize = 2;
const K_PACK: usize = 3;
const K_SCORES: usize = 4;
const K_SOFTMAX: usize = 5;
const K_APPLY: usize = 6;
const K_SILU_MUL: usize = 7;
const K_ADD2: usize = 8;

const KERNELS: [(&str, &str); 9] = [
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul", kernels::MATMUL),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
];

/// Host tensors by name → `(shape, row-major f32 data)`.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Shape parameters of one Z-Image block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDims {
    pub dim: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    /// adaLN conditioning width = `min(dim, 256)`.
    pub cdim: u32,
    /// SwiGLU hidden width = `dim*8/3`.
    pub hidden: u32,
}

impl BlockDims {
    pub fn new(dim: u32, n_heads: u32) -> BlockDims {
        BlockDims {
            dim,
            n_heads,
            head_dim: dim / n_heads,
            cdim: dim.min(256),
            hidden: dim * 8 / 3,
        }
    }
}

/// A single-block forward graph with weights resident, for a fixed token count.
pub struct ZImageBlock {
    gpu: Gpu,
    d: BlockDims,
    t: u32,
    modulation: bool,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    // Folded (per-forward) RMSNorm weight buffers.
    w_an1: DeviceBuffer,
    w_an2: DeviceBuffer,
    w_fn1: DeviceBuffer,
    w_fn2: DeviceBuffer,
    out: DeviceBuffer,
    // Host copies of the raw norm weights + adaLN projection (for folding).
    raw_an1: Vec<f32>,
    raw_an2: Vec<f32>,
    raw_fn1: Vec<f32>,
    raw_fn2: Vec<f32>,
    adaln_w: Vec<f32>,
    adaln_b: Vec<f32>,
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

impl ZImageBlock {
    /// Build the forward graph for `prefix.*` weights and `t` tokens. When
    /// `modulation` is false (context refiner) the norm weights are used raw and
    /// `c` is ignored.
    pub fn new(
        tensors: &Tensors,
        prefix: &str,
        d: BlockDims,
        t: u32,
        modulation: bool,
        device: Option<&str>,
    ) -> ZImageBlock {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let get = |n: &str| -> Vec<f32> {
            tensors.get(&format!("{prefix}.{n}")).unwrap_or_else(|| panic!("zimage: missing {prefix}.{n}")).1.clone()
        };
        let dev = |n: &str| gpu.storage_init(&format!("{prefix}.{n}"), &get(n));

        let (dim, nh, hd, hidden) = (d.dim, d.n_heads, d.head_dim, d.hidden);
        let half = hd / 2;
        let td = (t * dim) as u64;

        // Static weights.
        let wq = dev("attention.to_q.weight");
        let wk = dev("attention.to_k.weight");
        let wv = dev("attention.to_v.weight");
        let wo = dev("attention.to_out.0.weight");
        let nq = dev("attention.norm_q.weight");
        let nk = dev("attention.norm_k.weight");
        let w1 = dev("feed_forward.w1.weight");
        let w2 = dev("feed_forward.w2.weight");
        let w3 = dev("feed_forward.w3.weight");

        // Per-forward folded norm weights (uploaded in forward()).
        let w_an1 = gpu.storage(dim as u64);
        let w_an2 = gpu.storage(dim as u64);
        let w_fn1 = gpu.storage(dim as u64);
        let w_fn2 = gpu.storage(dim as u64);

        // Inputs.
        let x_in = gpu.storage(td);
        let cos = gpu.storage((t * half) as u64);
        let sin = gpu.storage((t * half) as u64);

        // Intermediates (kept alive by the recorded steps).
        let a = |n: u64| gpu.storage(n);
        let n1 = a(td);
        let q = a(td);
        let k = a(td);
        let v = a(td);
        let qn = a(td);
        let kn = a(td);
        let qr = a(td);
        let kr = a(td);
        let qkv = a((t * 3 * dim) as u64);
        let scores = a((nh * t * t) as u64);
        let probs = a((nh * t * t) as u64);
        let ctx = a(td);
        let attn_out = a(td);
        let n2 = a(td);
        let x1 = a(td);
        let f1 = a(td);
        let g = a((t * hidden) as u64);
        let u = a((t * hidden) as u64);
        let hsw = a((t * hidden) as u64);
        let ff = a(td);
        let f2 = a(td);
        let out = gpu.storage(td);

        let mm = |x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, m: u32, kk: u32, n: u32| {
            gpu.step(K_MATMUL, &[x, w, o], &[m, kk, n], m * n)
        };
        let mut s: Vec<Step> = Vec::new();
        // 1. attention sub-block
        s.push(gpu.step(K_RMSNORM, &[&x_in, &w_an1, &n1], &[dim, t, f(EPS)], t));
        s.push(mm(&n1, &wq, &q, t, dim, dim));
        s.push(mm(&n1, &wk, &k, t, dim, dim));
        s.push(mm(&n1, &wv, &v, t, dim, dim));
        s.push(gpu.step(K_RMSNORM, &[&q, &nq, &qn], &[hd, t * nh, f(EPS)], t * nh));
        s.push(gpu.step(K_RMSNORM, &[&k, &nk, &kn], &[hd, t * nh, f(EPS)], t * nh));
        s.push(gpu.step(K_ROPE, &[&qn, &cos, &sin, &qr], &[t, nh, hd, half], t * nh * half));
        s.push(gpu.step(K_ROPE, &[&kn, &cos, &sin, &kr], &[t, nh, hd, half], t * nh * half));
        s.push(gpu.step(K_PACK, &[&qr, &kr, &v, &qkv], &[t, dim], t * 3 * dim));
        s.push(gpu.step(K_SCORES, &[&qkv, &scores], &[1, nh, t, hd, 3 * dim, 0, dim], nh * t * t));
        s.push(gpu.step(K_SOFTMAX, &[&scores, &probs], &[1, nh, t], nh * t));
        s.push(gpu.step(K_APPLY, &[&probs, &qkv, &ctx], &[1, nh, t, hd, 3 * dim, 2 * dim, dim], nh * t * hd));
        s.push(mm(&ctx, &wo, &attn_out, t, dim, dim));
        s.push(gpu.step(K_RMSNORM, &[&attn_out, &w_an2, &n2], &[dim, t, f(EPS)], t));
        s.push(gpu.step(K_ADD2, &[&x_in, &n2, &x1], &[t * dim], t * dim));
        // 2. MLP sub-block
        s.push(gpu.step(K_RMSNORM, &[&x1, &w_fn1, &f1], &[dim, t, f(EPS)], t));
        s.push(mm(&f1, &w1, &g, t, dim, hidden));
        s.push(mm(&f1, &w3, &u, t, dim, hidden));
        s.push(gpu.step(K_SILU_MUL, &[&g, &u, &hsw], &[t * hidden], t * hidden));
        s.push(mm(&hsw, &w2, &ff, t, hidden, dim));
        s.push(gpu.step(K_RMSNORM, &[&ff, &w_fn2, &f2], &[dim, t, f(EPS)], t));
        s.push(gpu.step(K_ADD2, &[&x1, &f2, &out], &[t * dim], t * dim));

        ZImageBlock {
            gpu,
            d,
            t,
            modulation,
            steps: s,
            x_in,
            cos,
            sin,
            w_an1,
            w_an2,
            w_fn1,
            w_fn2,
            out,
            raw_an1: get("attention_norm1.weight"),
            raw_an2: get("attention_norm2.weight"),
            raw_fn1: get("ffn_norm1.weight"),
            raw_fn2: get("ffn_norm2.weight"),
            adaln_w: if modulation { get("adaLN_modulation.0.weight") } else { Vec::new() },
            adaln_b: if modulation { get("adaLN_modulation.0.bias") } else { Vec::new() },
        }
    }

    /// Forward one block. `x`: `[t·dim]`; `c`: `[cdim]` adaLN conditioning
    /// (ignored when `modulation=false`); `cos`/`sin`: `[t·head_dim/2]` RoPE
    /// tables. Returns `[t·dim]`.
    pub fn forward(&self, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let dim = self.d.dim as usize;
        // Fold adaLN scale/gate into the four norm weights (or use raw).
        let (an1, an2, fn1, fn2) = if self.modulation {
            let cdim = self.d.cdim as usize;
            // mod = adaLN_w · c + adaLN_b  → [4·dim]
            let mut m = vec![0f32; 4 * dim];
            for (i, mi) in m.iter_mut().enumerate() {
                let mut acc = self.adaln_b[i];
                let row = &self.adaln_w[i * cdim..i * cdim + cdim];
                for (wj, &cj) in row.iter().zip(c) {
                    acc += wj * cj;
                }
                *mi = acc;
            }
            let scale_msa = &m[0..dim];
            let gate_msa = &m[dim..2 * dim];
            let scale_mlp = &m[2 * dim..3 * dim];
            let gate_mlp = &m[3 * dim..4 * dim];
            let fold_scale = |raw: &[f32], s: &[f32]| -> Vec<f32> {
                raw.iter().zip(s).map(|(&w, &sc)| w * (1.0 + sc)).collect()
            };
            let fold_gate = |raw: &[f32], gt: &[f32]| -> Vec<f32> {
                raw.iter().zip(gt).map(|(&w, &g)| w * g.tanh()).collect()
            };
            (
                fold_scale(&self.raw_an1, scale_msa),
                fold_gate(&self.raw_an2, gate_msa),
                fold_scale(&self.raw_fn1, scale_mlp),
                fold_gate(&self.raw_fn2, gate_mlp),
            )
        } else {
            (self.raw_an1.clone(), self.raw_an2.clone(), self.raw_fn1.clone(), self.raw_fn2.clone())
        };
        wf(&self.gpu, &self.w_an1, &an1);
        wf(&self.gpu, &self.w_an2, &an2);
        wf(&self.gpu, &self.w_fn1, &fn1);
        wf(&self.gpu, &self.w_fn2, &fn2);
        wf(&self.gpu, &self.x_in, x);
        wf(&self.gpu, &self.cos, cos);
        wf(&self.gpu, &self.sin, sin);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (self.t * self.d.dim) as usize)
    }
}
