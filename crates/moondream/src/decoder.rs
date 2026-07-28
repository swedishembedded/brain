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
        ("embed", kernels::EMBED),                       // 13
        ("splice", kernels::SPLICE),                     // 14
        ("ce_value", kernels::CE_VALUE_MASKED),          // 15
        ("gelu_erf", kernels::GELU_ERF),                 // 16 (tau tok_feat: erf GELU)
        ("tau_scale", kernels::TAU_SCALE),               // 17
    ]
}

/// Masked cross-entropy ignore index (matches the loaders' `-1 i32` as `u32`).
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Logistic sigmoid (host, for the tau position term).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The Moondream text decoder: token embedding → splice the projected image tokens
/// into the prefix rows → a stack of [`MoondreamBlock`]s (dense 0..3, MoE 4..23) →
/// post-LN → lm_head → masked cross-entropy. Image tokens occupy rows
/// `[1, 1+n_img)` (after the bos), a positional prefix (no placeholder token).
pub struct MoondreamDecoder<'g> {
    gpu: &'g Gpu,
    blocks: Vec<MoondreamBlock<'g>>,
    w: HashMap<String, DeviceBuffer>,
    d: u32,
    vocab: u32,
    t: u32,
    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    res: DeviceBuffer,
    normed: DeviceBuffer,
    logits: DeviceBuffer,
    ce: DeviceBuffer,
    n_img: u32,
}

impl<'g> MoondreamDecoder<'g> {
    /// Build from per-layer prefixed weights (`blocks.{l}.…`) plus `tok.weight`
    /// `[vocab,d]`, `post_ln.weight`/`.bias`, `lm_head.weight` `[vocab,d]`/`.bias`.
    /// `moe_layers` marks which layers use the MoE FFN (their MoE weights are under
    /// `blocks.{l}.moe.…`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, blocks: Vec<MoondreamBlock<'g>>, t: u32, d: u32, vocab: u32, n_img: u32) -> MoondreamDecoder<'g> {
        let w = weights
            .iter()
            .filter(|(k, _)| !k.starts_with("blocks."))
            .map(|(k, v)| (k.clone(), gpu.storage_init(k, v)))
            .collect();
        MoondreamDecoder {
            gpu,
            blocks,
            w,
            d,
            vocab,
            t,
            tokens: gpu.storage(t as u64),
            targets: gpu.storage(t as u64),
            res: gpu.storage((t * d) as u64),
            normed: gpu.storage((t * d) as u64),
            logits: gpu.storage((t * vocab) as u64),
            ce: gpu.storage(t as u64),
            n_img,
        }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("decoder weight missing: {n}"))
    }
    /// Forward → mean masked cross-entropy. `tokens`/`targets` length `t`
    /// (targets IGNORE at the image + non-supervised rows); `image_embeds` is the
    /// `[n_img, d]` connector output spliced at rows `[1, 1+n_img)`.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], image_embeds: &[f32]) -> f32 {
        let g = self.gpu;
        let (t, d, v) = (self.t, self.d, self.vocab);
        g.write(&self.tokens, tokens);
        g.write(&self.targets, targets);
        let img = g.storage_init("md.img", image_embeds);
        // embed → res, then splice image tokens at rows [1, 1+n_img) (base = 1·d).
        g.submit(
            &[],
            &[
                g.step(13, &[&self.tokens, self.wb("tok.weight"), &self.res], &[d, t], t * d),
                g.step(14, &[&img, &self.res], &[self.n_img * d, d], self.n_img * d),
            ],
        );
        // Block stack (each returns its own output buffer).
        let mut cur: &DeviceBuffer = &self.res;
        for b in &self.blocks {
            cur = b.forward(cur);
        }
        // post-LN → lm_head → CE.
        g.submit(
            &[],
            &[
                g.step(4, &[cur, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &self.normed], &[d, t, f(LN_EPS)], t),
                g.step(0, &[&self.normed, self.wb("lm_head.weight"), &self.logits], &[t, d, v], t * v),
                g.step(6, &[&self.logits, self.wb("lm_head.bias")], &[t, v], t * v),
                g.step(15, &[&self.logits, &self.targets, &self.ce], &[t, v, IGNORE], t),
            ],
        );
        let ce = g.read(&self.ce, t as usize);
        let count = targets.iter().filter(|&&x| x != IGNORE).count().max(1) as f32;
        ce.iter().sum::<f32>() / count
    }
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
    /// True when `attn.tau.*` weights are present (per-head attention temperature).
    tau: bool,
    // scratch
    l_in: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv2: DeviceBuffer,   // tau-scaled qkv (q,v scaled; k passthrough)
    tok_feat: DeviceBuffer, // gelu_erf(qkv) [t, 3d]
    tqr: DeviceBuffer,    // tok_feat·wqᵀ [t, nh]
    tvr: DeviceBuffer,    // tok_feat·wvᵀ [t, nh]
    s3: DeviceBuffer,     // [3·nh, t] per-(head,token) scale (q | k=1 | v)
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
        let w: HashMap<String, DeviceBuffer> = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        let slab = (n_heads * t * t) as u64;
        let tau = w.contains_key("attn.tau.wq");
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
            tau,
            l_in: gpu.storage((t * d) as u64),
            qkv: gpu.storage((t * 3 * d) as u64),
            qkv2: gpu.storage((t * 3 * d) as u64),
            tok_feat: gpu.storage((t * 3 * d) as u64),
            tqr: gpu.storage((t * n_heads) as u64),
            tvr: gpu.storage((t * n_heads) as u64),
            s3: gpu.storage((3 * n_heads * t) as u64),
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
        // Per-head attention temperature (tau): scale q and v (NOT k) by a per-
        // (head,token) scalar computed from the raw qkv, BEFORE RoPE. tok_feat is
        // erf-GELU over the full 3d projection; tok_q/tok_v = tanh(tok_feat·w{q,v}ᵀ);
        // tau_pos = 0.5+sigmoid(alpha·ln(pos+1)) folds on host (positions = row).
        // Scalar scaling commutes with the RoPE rotation, so applying it here (pre-
        // RoPE, matching the reference) is equivalent up to that rotation. The tiny
        // tanh+tau_pos assembly into the [3·nh, t] scale (q | k=1 | v) folds on host.
        let qkv = if self.tau {
            s.push(g.step(16, &[&self.qkv, &self.tok_feat], &[t * stride3], t * stride3));
            s.push(g.step(0, &[&self.tok_feat, self.wb("attn.tau.wq"), &self.tqr], &[t, stride3, nh], t * nh));
            s.push(g.step(0, &[&self.tok_feat, self.wb("attn.tau.wv"), &self.tvr], &[t, stride3, nh], t * nh));
            g.submit(&[], &s);
            s = Vec::new();
            let tqr = g.read(&self.tqr, (t * nh) as usize);
            let tvr = g.read(&self.tvr, (t * nh) as usize);
            let alpha = g.read(self.wb("attn.tau.alpha"), nh as usize);
            let mut s3 = vec![1.0f32; (3 * nh * t) as usize];
            for h in 0..nh as usize {
                for row in 0..t as usize {
                    let tau_pos = 0.5 + sigmoid(alpha[h] * ((row + 1) as f32).ln());
                    s3[h * t as usize + row] = tqr[row * nh as usize + h].tanh() + tau_pos;
                    s3[(2 * nh as usize + h) * t as usize + row] = tvr[row * nh as usize + h].tanh() + tau_pos;
                }
            }
            let packed: Vec<u32> = s3.iter().map(|&x| f(x)).collect();
            g.write(&self.s3, &packed);
            // Treat qkv as [t, 3·nh, hd]: scale q-heads by s_q, k-heads by 1, v by s_v.
            s.push(g.step(17, &[&self.qkv, &self.s3, &self.qkv2], &[t, 3 * nh, hd], t * stride3));
            &self.qkv2
        } else {
            &self.qkv
        };
        // partial RoPE on q (off 0) and k (off d)
        let half = self.rot_dim / 2;
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, 0, t, f(self.theta), self.rot_dim], t * nh * half));
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, d, t, f(self.theta), self.rot_dim], t * nh * half));
        // bidir scores → prefix-LM mask → bidir softmax → bidir apply
        s.push(g.step(9, &[qkv, &self.scores], &[1, nh, t, hd, stride3, 0, d], nh * t * t));
        s.push(g.step(10, &[&self.scores], &[1, nh, t, self.prefix], nh * t * t));
        s.push(g.step(11, &[&self.scores, &self.probs], &[1, nh, t], nh * t));
        s.push(g.step(12, &[&self.probs, qkv, &self.ctx], &[1, nh, t, hd, stride3, 2 * d, d], nh * t * hd));
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
    fn tau_temperature_changes_block_output() {
        // A block with attn.tau.* present applies per-head temperature to q,v and
        // must differ from the same weights without tau.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let mut rng = Rng::new(11);
        let base = block_weights(d, ff, &mut rng);
        let x: Vec<f32> = (0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect();

        let blk = MoondreamBlock::new(&gpu, &base, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb = gpu.storage_init("x", &x);
        assert!(!blk.tau);
        let plain = gpu.read(blk.forward(&xb), blk.numel());

        let mut tw = base.clone();
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        tw.insert("attn.tau.wq".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.wv".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.alpha".into(), r(nh as usize));
        let tblk = MoondreamBlock::new(&gpu, &tw, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb2 = gpu.storage_init("x2", &x);
        assert!(tblk.tau);
        let tau = gpu.read(tblk.forward(&xb2), tblk.numel());

        assert!(tau.iter().all(|v| v.is_finite()));
        let diff: f32 = plain.iter().zip(&tau).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "tau must change the block output, Σ|Δ|={diff}");
    }

    fn block_weights(d: u32, ff: u32, rng: &mut Rng) -> HashMap<String, Vec<f32>> {
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
        w
    }

    #[test]
    fn full_decoder_forward_is_finite() {
        // t=8 stream: bos + 4 image + 3 text (prefix=5). 2 dense layers.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, vocab, n_img, prefix, rot) = (8u32, 16u32, 2u32, 8u32, 32u32, 23u32, 4u32, 5u32, 4u32);
        let mut rng = Rng::new(9);
        let bw0 = block_weights(d, ff, &mut rng);
        let bw1 = block_weights(d, ff, &mut rng);
        let blocks = vec![
            MoondreamBlock::new(&gpu, &bw0, t, d, nh, hd, ff, prefix, rot, 1.5e6),
            MoondreamBlock::new(&gpu, &bw1, t, d, nh, hd, ff, prefix, rot, 1.5e6),
        ];
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut dw = HashMap::new();
        dw.insert("tok.weight".into(), r((vocab * d) as usize));
        dw.insert("post_ln.weight".into(), vec![1.0; d as usize]);
        dw.insert("post_ln.bias".into(), r(d as usize));
        dw.insert("lm_head.weight".into(), r((vocab * d) as usize));
        dw.insert("lm_head.bias".into(), r(vocab as usize));
        let dec = MoondreamDecoder::new(&gpu, &dw, blocks, t, d, vocab, n_img);

        let tokens = vec![0u32, 5, 5, 5, 5, 7, 9, 11]; // bos + image + text
        let mut targets = vec![5u32, 0, 0, 0, 0, 9, 11, 13];
        for tg in targets.iter_mut().take(5).skip(1) {
            *tg = IGNORE;
        }
        let img: Vec<f32> = r((n_img * d) as usize);
        let loss = dec.forward(&tokens, &targets, &img);
        assert!(loss.is_finite() && loss > 0.0, "moondream decoder loss must be finite+positive, got {loss}");
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
