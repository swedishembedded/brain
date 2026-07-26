// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Z-Image single-stream transformer block as a recorded brain kernel graph.
//!
//! Mirrors diffusers `ZImageTransformerBlock.forward` (single-stream, global
//! modulation): a double-RMSNorm sandwich per sub-block with adaLN scale on the
//! pre-norm and adaLN gate on the post-norm, QK-normalized attention with
//! multi-axis interleaved RoPE, and a SwiGLU MLP. adaLN scale/gate are folded
//! into the four RMSNorm weights on the host (`rmsnorm(x,w)·scale =
//! rmsnorm(x,w·scale)`, `gate·rmsnorm(y,w)=rmsnorm(y,w·gate)`), so no scale/gate
//! kernels are needed.
//!
//! The step-builder ([`build_block_steps`]) and resident weights ([`BlockWeights`])
//! are shared: [`ZImageBlock`] owns a device and runs one block (the parity
//! reference), while the device-resident chain (`crate::dev`) uploads every
//! block's weights once and records the whole stack into one graph.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Z-Image RMSNorm epsilon (diffusers `norm_eps` / attention `eps`).
pub(crate) const EPS: f32 = 1e-5;

// Kernel-table indices (order matches KERNELS).
pub(crate) const K_RMSNORM: usize = 0;
pub(crate) const K_MATMUL: usize = 1;
pub(crate) const K_ROPE: usize = 2;
pub(crate) const K_PACK: usize = 3;
pub(crate) const K_SCORES: usize = 4;
pub(crate) const K_SOFTMAX: usize = 5;
pub(crate) const K_APPLY: usize = 6;
pub(crate) const K_SILU_MUL: usize = 7;
pub(crate) const K_ADD2: usize = 8;
pub(crate) const K_MATMUL_REG2: usize = 9;

pub(crate) const KERNELS: [(&str, &str); 10] = [
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul", kernels::MATMUL),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    // GPU-only fast GEMM (software-pipelined register tiling). The CPU JIT can't
    // compile its barrier, so CPU uses the naive `matmul` (native AVX2 path).
    ("matmul_reg2", kernels::MATMUL_REG2),
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
        BlockDims { dim, n_heads, head_dim: dim / n_heads, cdim: dim.min(256), hidden: dim * 8 / 3 }
    }
}

pub(crate) fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

/// Resident (upload-once) static weights of one block.
pub(crate) struct BlockWeights {
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub nq: DeviceBuffer,
    pub nk: DeviceBuffer,
    pub w1: DeviceBuffer,
    pub w2: DeviceBuffer,
    pub w3: DeviceBuffer,
}

impl BlockWeights {
    pub fn upload(gpu: &Gpu, t: &Tensors, prefix: &str) -> BlockWeights {
        let dev = |n: &str| {
            let key = format!("{prefix}.{n}");
            gpu.storage_init(&key, &t.get(&key).unwrap_or_else(|| panic!("zimage: missing {key}")).1)
        };
        BlockWeights {
            wq: dev("attention.to_q.weight"),
            wk: dev("attention.to_k.weight"),
            wv: dev("attention.to_v.weight"),
            wo: dev("attention.to_out.0.weight"),
            nq: dev("attention.norm_q.weight"),
            nk: dev("attention.norm_k.weight"),
            w1: dev("feed_forward.w1.weight"),
            w2: dev("feed_forward.w2.weight"),
            w3: dev("feed_forward.w3.weight"),
        }
    }
}

/// The four per-forward folded-norm buffers (rewritten each forward from the
/// timestep conditioning; see [`fold_adaln`]).
pub(crate) struct NormBufs {
    pub an1: DeviceBuffer,
    pub an2: DeviceBuffer,
    pub fn1: DeviceBuffer,
    pub fn2: DeviceBuffer,
    // Host copies of the raw norm weights + adaLN projection (for folding).
    pub raw_an1: Vec<f32>,
    pub raw_an2: Vec<f32>,
    pub raw_fn1: Vec<f32>,
    pub raw_fn2: Vec<f32>,
    pub adaln_w: Vec<f32>,
    pub adaln_b: Vec<f32>,
    pub modulation: bool,
}

impl NormBufs {
    pub fn new(gpu: &Gpu, t: &Tensors, prefix: &str, dim: u32, modulation: bool) -> NormBufs {
        let get = |n: &str| t.get(&format!("{prefix}.{n}")).unwrap_or_else(|| panic!("zimage: missing {prefix}.{n}")).1.clone();
        NormBufs {
            an1: gpu.storage(dim as u64),
            an2: gpu.storage(dim as u64),
            fn1: gpu.storage(dim as u64),
            fn2: gpu.storage(dim as u64),
            raw_an1: get("attention_norm1.weight"),
            raw_an2: get("attention_norm2.weight"),
            raw_fn1: get("ffn_norm1.weight"),
            raw_fn2: get("ffn_norm2.weight"),
            adaln_w: if modulation { get("adaLN_modulation.0.weight") } else { Vec::new() },
            adaln_b: if modulation { get("adaLN_modulation.0.bias") } else { Vec::new() },
            modulation,
        }
    }

    /// Fold the timestep conditioning `c` into the four norm weights and upload.
    pub fn upload_folded(&self, gpu: &Gpu, c: &[f32], dim: usize, cdim: usize) {
        let (an1, an2, fn1, fn2) = fold_adaln(self, c, dim, cdim);
        wf(gpu, &self.an1, &an1);
        wf(gpu, &self.an2, &an2);
        wf(gpu, &self.fn1, &fn1);
        wf(gpu, &self.fn2, &fn2);
    }
}

/// adaLN fold: `mod = adaLN_w·c + adaLN_b` → `(scale_msa, gate_msa, scale_mlp,
/// gate_mlp)`; norms become `raw·(1+scale)` / `raw·tanh(gate)`. When
/// `modulation=false` the raw norm weights pass through unchanged.
pub(crate) fn fold_adaln(nb: &NormBufs, c: &[f32], dim: usize, cdim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if !nb.modulation {
        return (nb.raw_an1.clone(), nb.raw_an2.clone(), nb.raw_fn1.clone(), nb.raw_fn2.clone());
    }
    let mut m = vec![0f32; 4 * dim];
    for (i, mi) in m.iter_mut().enumerate() {
        let mut acc = nb.adaln_b[i];
        for (wj, &cj) in nb.adaln_w[i * cdim..i * cdim + cdim].iter().zip(c) {
            acc += wj * cj;
        }
        *mi = acc;
    }
    let fold_scale = |raw: &[f32], s: &[f32]| -> Vec<f32> { raw.iter().zip(s).map(|(&w, &sc)| w * (1.0 + sc)).collect() };
    let fold_gate = |raw: &[f32], g: &[f32]| -> Vec<f32> { raw.iter().zip(g).map(|(&w, &g)| w * g.tanh()).collect() };
    (
        fold_scale(&nb.raw_an1, &m[0..dim]),
        fold_gate(&nb.raw_an2, &m[dim..2 * dim]),
        fold_scale(&nb.raw_fn1, &m[2 * dim..3 * dim]),
        fold_gate(&nb.raw_fn2, &m[3 * dim..4 * dim]),
    )
}

/// Append one block's forward steps to `s`, reading `x_in` and the shared
/// `cos`/`sin` RoPE tables, and return the fresh output buffer (for chaining).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_steps(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &BlockWeights,
    nb: &NormBufs,
    x_in: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    d: BlockDims,
    t: u32,
    reg2: bool,
) -> DeviceBuffer {
    let (dim, nh, hd, hidden) = (d.dim, d.n_heads, d.head_dim, d.hidden);
    let half = hd / 2;
    let td = (t * dim) as u64;
    let a = |n: u64| gpu.storage(n);
    let (n1, q, k, v, qn, kn, qr, kr) = (a(td), a(td), a(td), a(td), a(td), a(td), a(td), a(td));
    let qkv = a((t * 3 * dim) as u64);
    let (scores, probs) = (a((nh * t * t) as u64), a((nh * t * t) as u64));
    let (ctx, attn_out, n2, x1, f1) = (a(td), a(td), a(td), a(td), a(td));
    let (g, u, hsw, ff, f2, out) = (a((t * hidden) as u64), a((t * hidden) as u64), a((t * hidden) as u64), a(td), a(td), a(td));

    // GPU: register-tiled matmul_reg2 (128×128 tile, 256 threads). CPU: naive
    // matmul (native AVX2 fast path; the JIT can't compile reg2's barrier).
    let mm = |x: &DeviceBuffer, wt: &DeviceBuffer, o: &DeviceBuffer, m: u32, kk: u32, n: u32| {
        if reg2 {
            gpu.step(K_MATMUL_REG2, &[x, wt, o], &[m, kk, n], m.div_ceil(128) * n.div_ceil(128) * 256)
        } else {
            gpu.step(K_MATMUL, &[x, wt, o], &[m, kk, n], m * n)
        }
    };
    // attention
    s.push(gpu.step(K_RMSNORM, &[x_in, &nb.an1, &n1], &[dim, t, f(EPS)], t));
    s.push(mm(&n1, &w.wq, &q, t, dim, dim));
    s.push(mm(&n1, &w.wk, &k, t, dim, dim));
    s.push(mm(&n1, &w.wv, &v, t, dim, dim));
    s.push(gpu.step(K_RMSNORM, &[&q, &w.nq, &qn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_RMSNORM, &[&k, &w.nk, &kn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_ROPE, &[&qn, cos, sin, &qr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_ROPE, &[&kn, cos, sin, &kr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_PACK, &[&qr, &kr, &v, &qkv], &[t, dim], t * 3 * dim));
    s.push(gpu.step(K_SCORES, &[&qkv, &scores], &[1, nh, t, hd, 3 * dim, 0, dim], nh * t * t));
    s.push(gpu.step(K_SOFTMAX, &[&scores, &probs], &[1, nh, t], nh * t));
    s.push(gpu.step(K_APPLY, &[&probs, &qkv, &ctx], &[1, nh, t, hd, 3 * dim, 2 * dim, dim], nh * t * hd));
    s.push(mm(&ctx, &w.wo, &attn_out, t, dim, dim));
    s.push(gpu.step(K_RMSNORM, &[&attn_out, &nb.an2, &n2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[x_in, &n2, &x1], &[t * dim], t * dim));
    // MLP
    s.push(gpu.step(K_RMSNORM, &[&x1, &nb.fn1, &f1], &[dim, t, f(EPS)], t));
    s.push(mm(&f1, &w.w1, &g, t, dim, hidden));
    s.push(mm(&f1, &w.w3, &u, t, dim, hidden));
    s.push(gpu.step(K_SILU_MUL, &[&g, &u, &hsw], &[t * hidden], t * hidden));
    s.push(mm(&hsw, &w.w2, &ff, t, hidden, dim));
    s.push(gpu.step(K_RMSNORM, &[&ff, &nb.fn2, &f2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[&x1, &f2, &out], &[t * dim], t * dim));
    out
}

/// A single-block forward graph with weights resident, for a fixed token count.
/// This is the parity reference; the device-resident chain lives in `crate::dev`.
pub struct ZImageBlock {
    gpu: Gpu,
    d: BlockDims,
    t: u32,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    nb: NormBufs,
    out: DeviceBuffer,
}

impl ZImageBlock {
    pub fn new(tensors: &Tensors, prefix: &str, d: BlockDims, t: u32, modulation: bool, device: Option<&str>) -> ZImageBlock {
        let reg2 = device != Some("cpu");
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let w = BlockWeights::upload(&gpu, tensors, prefix);
        let nb = NormBufs::new(&gpu, tensors, prefix, d.dim, modulation);
        let half = d.head_dim / 2;
        let x_in = gpu.storage((t * d.dim) as u64);
        let cos = gpu.storage((t * half) as u64);
        let sin = gpu.storage((t * half) as u64);
        let mut steps = Vec::new();
        let out = build_block_steps(&gpu, &mut steps, &w, &nb, &x_in, &cos, &sin, d, t, reg2);
        ZImageBlock { gpu, d, t, steps, x_in, cos, sin, nb, out }
    }

    /// Forward one block. `x`: `[t·dim]`; `c`: `[cdim]` adaLN conditioning
    /// (ignored when `modulation=false`); `cos`/`sin`: `[t·head_dim/2]`.
    pub fn forward(&self, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        self.nb.upload_folded(&self.gpu, c, self.d.dim as usize, self.d.cdim as usize);
        wf(&self.gpu, &self.x_in, x);
        wf(&self.gpu, &self.cos, cos);
        wf(&self.gpu, &self.sin, sin);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (self.t * self.d.dim) as usize)
    }
}
