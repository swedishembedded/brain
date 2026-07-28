// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream text decoder pieces. Built up incrementally; today: the sparse-MoE
//! FFN (GeGLU-shift experts + top-k router), which mirrors `crates/moe`'s dense-
//! over-all-experts FFN but swaps SwiGLU for Moondream's GeGLU-with-+1-shift
//! (`geglu_shift`) and a single fc1 split into its `h`/`g` halves (`w_h`/`w_g`).

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Decoder kernel pipeline (indices used below).
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("matmul", kernels::MATMUL),                     // 0
        ("router_gate", kernels::ROUTER_GATE),           // 1
        ("geglu_shift", kernels::GEGLU_SHIFT),           // 2
        ("scale_add", kernels::SCALE_ADD),               // 3
        ("layernorm", kernels::LAYERNORM),               // 4
        ("gelu", kernels::GELU),                         // 5 (tanh gelu_approx)
        ("bias_add", kernels::BIAS_ADD),                 // 6
        ("add2", kernels::ADD2),                         // 7
        ("rope_partial", kernels::ROPE_PARTIAL),         // 8
        ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR), // 9
        ("attn_prefix_mask", kernels::ATTN_PREFIX_MASK), // 10
        ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR), // 11
        ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR), // 12
    ]
}

const LN_EPS: f32 = 1e-5;

/// One Moondream decoder block — the PARALLEL attn+MLP form: a single shared
/// LayerNorm feeds BOTH the attention and the FFN, and `x = x + l_attn + l_mlp`
/// (a 3-way residual). The attention is full MHA with **partial RoPE** and the
/// **prefix-LM mask** (image prefix bidirectional, else causal); the FFN here is
/// the dense variant (layers 0..3). (The tau temperature and the MoE FFN variant
/// for layers 4..23 are added on top; backward + gradcheck are a follow-up.)
/// Weight keys: `ln.weight`/`ln.bias`, `attn.qkv.weight` `[3d, d]`,
/// `attn.proj.weight` `[d,d]`/`attn.proj.bias`, `mlp.fc1.weight` `[ff,d]`/`.bias`,
/// `mlp.fc2.weight` `[d,ff]`/`.bias`.
pub struct MoondreamBlock<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    d: u32,
    n_heads: u32,
    head_dim: u32,
    ff: u32,
    t: u32,
    prefix: u32,
    rot_dim: u32,
    theta: f32,
    // scratch
    l_in: DeviceBuffer,
    qkv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    l_attn: DeviceBuffer,
    h: DeviceBuffer,
    h2: DeviceBuffer,
    l_mlp: DeviceBuffer,
    mid: DeviceBuffer,
    out: DeviceBuffer,
    /// `Some` for the MoE layers (4..23); `None` uses the dense FFN.
    moe: Option<MoeFfn<'g>>,
}

impl<'g> MoondreamBlock<'g> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, n_heads: u32, head_dim: u32, ff: u32, prefix: u32, rot_dim: u32, theta: f32) -> MoondreamBlock<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        let slab = (n_heads * t * t) as u64;
        MoondreamBlock {
            gpu,
            w,
            d,
            n_heads,
            head_dim,
            ff,
            t,
            prefix,
            rot_dim,
            theta,
            l_in: gpu.storage((t * d) as u64),
            qkv: gpu.storage((t * 3 * d) as u64),
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            ctx: gpu.storage((t * d) as u64),
            l_attn: gpu.storage((t * d) as u64),
            h: gpu.storage((t * ff) as u64),
            h2: gpu.storage((t * ff) as u64),
            l_mlp: gpu.storage((t * d) as u64),
            mid: gpu.storage((t * d) as u64),
            out: gpu.storage((t * d) as u64),
            moe: None,
        }
    }
    /// Attach an MoE FFN (replaces the dense FFN branch) for a deep layer.
    pub fn with_moe(mut self, moe: MoeFfn<'g>) -> Self {
        self.moe = Some(moe);
        self
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("block weight missing: {n}"))
    }
    pub fn forward(&self, x: &DeviceBuffer) -> &DeviceBuffer {
        let g = self.gpu;
        let (t, d, nh, hd, ff) = (self.t, self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let mut s: Vec<Step> = Vec::new();

        // Shared LayerNorm (with bias).
        s.push(g.step(4, &[x, self.wb("ln.weight"), self.wb("ln.bias"), &self.l_in], &[d, t, f(LN_EPS)], t));
        // --- attention branch ---
        // fused qkv = l_in · wqkv^T  ([t, 3d])
        s.push(g.step(0, &[&self.l_in, self.wb("attn.qkv.weight"), &self.qkv], &[t, d, stride3], t * stride3));
        // partial RoPE on q (off 0) and k (off d)
        let half = self.rot_dim / 2;
        s.push(g.step(8, &[&self.qkv], &[t, nh, hd, stride3, 0, t, f(self.theta), self.rot_dim], t * nh * half));
        s.push(g.step(8, &[&self.qkv], &[t, nh, hd, stride3, d, t, f(self.theta), self.rot_dim], t * nh * half));
        // bidir scores → prefix-LM mask → bidir softmax → bidir apply
        s.push(g.step(9, &[&self.qkv, &self.scores], &[1, nh, t, hd, stride3, 0, d], nh * t * t));
        s.push(g.step(10, &[&self.scores], &[1, nh, t, self.prefix], nh * t * t));
        s.push(g.step(11, &[&self.scores, &self.probs], &[1, nh, t], nh * t));
        s.push(g.step(12, &[&self.probs, &self.qkv, &self.ctx], &[1, nh, t, hd, stride3, 2 * d, d], nh * t * hd));
        // proj + bias → l_attn. Submit phase 1 (LN + attention) so l_in/l_attn are
        // ready before the FFN (MoE submits internally).
        s.push(g.step(0, &[&self.ctx, self.wb("attn.proj.weight"), &self.l_attn], &[t, d, d], t * d));
        s.push(g.step(6, &[&self.l_attn, self.wb("attn.proj.bias")], &[t, d], t * d));
        g.submit(&[], &s);

        // --- FFN branch on the SAME l_in: MoE (layers 4..23) or dense. ---
        let l_mlp: &DeviceBuffer = if let Some(moe) = &self.moe {
            moe.forward(&self.l_in)
        } else {
            g.submit(
                &[],
                &[
                    g.step(0, &[&self.l_in, self.wb("mlp.fc1.weight"), &self.h], &[t, d, ff], t * ff),
                    g.step(6, &[&self.h, self.wb("mlp.fc1.bias")], &[t, ff], t * ff),
                    g.step(5, &[&self.h, &self.h2], &[t * ff], t * ff), // tanh GELU
                    g.step(0, &[&self.h2, self.wb("mlp.fc2.weight"), &self.l_mlp], &[t, ff, d], t * d),
                    g.step(6, &[&self.l_mlp, self.wb("mlp.fc2.bias")], &[t, d], t * d),
                ],
            );
            &self.l_mlp
        };
        // --- 3-way residual: out = x + l_attn + l_mlp ---
        g.submit(
            &[],
            &[
                g.step(7, &[x, &self.l_attn, &self.mid], &[t * d], t * d),
                g.step(7, &[&self.mid, l_mlp, &self.out], &[t * d], t * d),
            ],
        );
        &self.out
    }
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }
}

/// Sparse-MoE FFN: `router → for each expert (w_h, w_g → geglu_shift → w_down) →
/// gate-weighted accumulate`. Returns the mixed output `[t, d]` (no residual — the
/// parallel block owns the 3-way residual). Weight keys: `router.weight` `[e, d]`,
/// and per expert `experts.{e}.{w_h,w_g}.weight` `[inner, d]`, `w_down.weight`
/// `[d, inner]`.
pub struct MoeFfn<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    e: u32,
    top_k: u32,
    d: u32,
    inner: u32,
    // scratch
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    h: DeviceBuffer,
    g: DeviceBuffer,
    act: DeviceBuffer,
    eout: DeviceBuffer,
    acc: DeviceBuffer,
    t: u32,
}

impl<'g> MoeFfn<'g> {
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, inner: u32, e: u32, top_k: u32) -> MoeFfn<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        MoeFfn {
            gpu,
            w,
            e,
            top_k,
            d,
            inner,
            logits: gpu.storage((t * e) as u64),
            gate: gpu.storage((t * e) as u64),
            h: gpu.storage((t * inner) as u64),
            g: gpu.storage((t * inner) as u64),
            act: gpu.storage((t * inner) as u64),
            eout: gpu.storage((t * d) as u64),
            acc: gpu.storage((t * d) as u64),
            t,
        }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("moe weight missing: {n}"))
    }
    pub fn forward(&self, xn: &DeviceBuffer) -> &DeviceBuffer {
        let (t, d, inner, e) = (self.t, self.d, self.inner, self.e);
        let mut s: Vec<Step> = Vec::new();
        // Router: logits = xn·router.weight^T, then top-k softmax gate.
        s.push(self.gpu.step(0, &[xn, self.wb("router.weight"), &self.logits], &[t, d, e], t * e));
        s.push(self.gpu.step(1, &[&self.logits, &self.gate], &[t, e, self.top_k], t));
        for ei in 0..e {
            let ep = |leaf: &str| self.wb(&format!("experts.{ei}.{leaf}"));
            s.push(self.gpu.step(0, &[xn, ep("w_h.weight"), &self.h], &[t, d, inner], t * inner));
            s.push(self.gpu.step(0, &[xn, ep("w_g.weight"), &self.g], &[t, d, inner], t * inner));
            s.push(self.gpu.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner)); // gelu(h)·(g+1)
            s.push(self.gpu.step(0, &[&self.act, ep("w_down.weight"), &self.eout], &[t, inner, d], t * d));
            let acc = if ei == 0 { 0u32 } else { 1u32 };
            s.push(self.gpu.step(3, &[&self.gate, &self.eout, &self.acc], &[t, d, e, ei, acc], t * d));
        }
        self.gpu.submit(&[], &s);
        &self.acc
    }
    /// Number of output elements (`t·d`).
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    #[test]
    fn parallel_block_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let mut rng = Rng::new(6);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("ln.weight".into(), vec![1.0; d as usize]);
        w.insert("ln.bias".into(), r(d as usize));
        w.insert("attn.qkv.weight".into(), r((3 * d * d) as usize));
        w.insert("attn.proj.weight".into(), r((d * d) as usize));
        w.insert("attn.proj.bias".into(), r(d as usize));
        w.insert("mlp.fc1.weight".into(), r((ff * d) as usize));
        w.insert("mlp.fc1.bias".into(), r(ff as usize));
        w.insert("mlp.fc2.weight".into(), r((d * ff) as usize));
        w.insert("mlp.fc2.bias".into(), r(d as usize));
        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let x = gpu.storage_init("x", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(blk.forward(&x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn parallel_block_with_moe_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let (inner, e, top_k) = (4u32, 3u32, 2u32);
        let mut rng = Rng::new(8);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut bw = HashMap::new();
        bw.insert("ln.weight".into(), vec![1.0; d as usize]);
        bw.insert("ln.bias".into(), r(d as usize));
        bw.insert("attn.qkv.weight".into(), r((3 * d * d) as usize));
        bw.insert("attn.proj.weight".into(), r((d * d) as usize));
        bw.insert("attn.proj.bias".into(), r(d as usize));
        // dense fc weights present but unused when MoE is attached.
        bw.insert("mlp.fc1.weight".into(), r((ff * d) as usize));
        bw.insert("mlp.fc1.bias".into(), r(ff as usize));
        bw.insert("mlp.fc2.weight".into(), r((d * ff) as usize));
        bw.insert("mlp.fc2.bias".into(), r(d as usize));
        let mut mw = HashMap::new();
        mw.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            mw.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            mw.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            mw.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let moe = MoeFfn::new(&gpu, &mw, t, d, inner, e, top_k);
        let blk = MoondreamBlock::new(&gpu, &bw, t, d, nh, hd, ff, prefix, rot, 1.5e6).with_moe(moe);
        let x = gpu.storage_init("x", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(blk.forward(&x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn moe_ffn_geglu_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 8u32, 4u32, 3u32, 2u32);
        let mut rng = Rng::new(5);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            w.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let ffn = MoeFfn::new(&gpu, &w, t, d, inner, e, top_k);
        let xn = gpu.storage_init("xn", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(ffn.forward(&xn), ffn.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-9));
    }
}
