// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! KV-cached autoregressive rollout for the Kronos decoder — the fast path.
//!
//! The un-cached rollout re-runs `decode_s1` over the full growing window every
//! step: 8 causal-attention layers over `T` tokens, `O(T²)` per step. This module
//! prefills the context **once**, caching each layer's RoPE'd K/V (and the
//! `decode_s1` context output used by `decode_s2`), then advances one token at a
//! time: every projection is an `m=1` matvec and attention is a single query over
//! the cached keys. `O(T²)` prefill + `O(T)` per step instead of `O(T²)` per step.
//!
//! Correctness rests on two facts: (1) **RoPE is shift-invariant** — `q_i·k_j`
//! depends only on `i−j` — so keys cached at their absolute positions stay valid
//! as the window slides (we simply window the attention range to the last
//! `max_context`, matching the reference's re-indexing); (2) **V carries no
//! RoPE**, so the cached attention output is numerically identical.
//!
//! This path is pure host math (`f32`), reproducing the WGSL kernels exactly:
//! RMSNorm eps `1e-6`, NeoX half-split RoPE (`θ=10000`), scaled dot-product
//! attention (`1/√head_dim`). It is validated against the un-cached decoder to
//! cosine `>0.999` (`tests/kvcache_parity.rs`).

/// Host copies of the decoder's weights + embedding tables, plus the derived
/// dims, for the cached rollout. Built once per forecast.
pub struct HostW {
    pub d: usize,
    pub ff: usize,
    pub nl: usize,
    pub heads: usize,
    pub hd: usize,
    pub s1v: usize,
    pub s2v: usize,
    pub dep_heads: usize,
    pub dep_hd: usize,
    pub max_ctx: usize,
    pub sd: f32,
    // per-layer [d]/[d,d]/[ff,d]/[d,ff]
    pub norm1: Vec<Vec<f32>>,
    pub qw: Vec<Vec<f32>>,
    pub qb: Vec<Vec<f32>>,
    pub kw: Vec<Vec<f32>>,
    pub kb: Vec<Vec<f32>>,
    pub vw: Vec<Vec<f32>>,
    pub vb: Vec<Vec<f32>>,
    pub ow: Vec<Vec<f32>>,
    pub ob: Vec<Vec<f32>>,
    pub norm2: Vec<Vec<f32>>,
    pub w1: Vec<Vec<f32>>,
    pub w3: Vec<Vec<f32>>,
    pub w2: Vec<Vec<f32>>,
    pub normf: Vec<f32>,
    pub ps1w: Vec<f32>,
    pub ps1b: Vec<f32>,
    pub ps2w: Vec<f32>,
    pub ps2b: Vec<f32>,
    pub fusw: Vec<f32>,
    pub fusb: Vec<f32>,
    // dependency (cross-attn) layer
    pub dqw: Vec<f32>,
    pub dqb: Vec<f32>,
    pub dkw: Vec<f32>,
    pub dkb: Vec<f32>,
    pub dvw: Vec<f32>,
    pub dvb: Vec<f32>,
    pub dow: Vec<f32>,
    pub dob: Vec<f32>,
    pub dnorm: Vec<f32>,
    // embedding tables (host)
    pub emb_s1: Vec<f32>,
    pub emb_s2: Vec<f32>,
    pub cal: Vec<Vec<f32>>, // 5 calendar tables in minute,hour,weekday,day,month order
}

const THETA: f32 = 10000.0;
const EPS: f32 = 1e-6;

/// Growing per-layer K/V + context caches for one rollout. `Clone` so a prefilled
/// context can be **forked** into N sampling branches (shared-prefill sampling).
#[derive(Default, Clone)]
pub struct Cache {
    pub k: Vec<Vec<f32>>, // [nl][len*d] RoPE'd keys
    pub v: Vec<Vec<f32>>, // [nl][len*d]
    pub ctx: Vec<f32>,    // [len*d] decode_s1 final-norm output per position
    /// Dependency (cross-attention, S2) K/V, RoPE'd at each ctx position's
    /// absolute index and grown incrementally alongside `ctx`. Lets
    /// `dep_step_cached` project each position's dep K/V **once** instead of
    /// re-projecting the whole window every decode step. Valid across window
    /// slides by the same RoPE shift-invariance as the main K cache.
    pub dep_k: Vec<f32>,
    pub dep_v: Vec<f32>,
}

impl HostW {
    pub fn new_cache(&self) -> Cache {
        Cache {
            k: vec![Vec::new(); self.nl],
            v: vec![Vec::new(); self.nl],
            ctx: Vec::new(),
            dep_k: Vec::new(),
            dep_v: Vec::new(),
        }
    }

    /// Process one token (id `s1`,`s2`; calendar `stamp` = 5 indices; absolute
    /// position `pos`), appending its K/V to `cache` and its ctx to `cache.ctx`.
    /// Returns the token's `s1` logits `[s1v]`.
    pub fn step_token(&self, s1: u32, s2: u32, stamp: &[u32], pos: usize, cache: &mut Cache) -> Vec<f32> {
        let d = self.d;
        // hierarchical embedding: [emb_s1*√d | emb_s2*√d] -> fusion_proj -> [d]
        let mut cat = vec![0f32; 2 * d];
        let (o1, o2) = (s1 as usize * d, s2 as usize * d);
        for i in 0..d {
            cat[i] = self.emb_s1[o1 + i] * self.sd;
            cat[d + i] = self.emb_s2[o2 + i] * self.sd;
        }
        let mut x = linear(&cat, &self.fusw, &self.fusb, d, 2 * d);
        // + summed calendar embeddings (5 tables)
        if !stamp.is_empty() {
            for (ci, tbl) in self.cal.iter().enumerate() {
                let base = stamp[ci] as usize * d;
                for i in 0..d {
                    x[i] += tbl[base + i];
                }
            }
        }

        for l in 0..self.nl {
            let xn = hostmath::rmsnorm(&x, &self.norm1[l], EPS);
            let mut q = linear(&xn, &self.qw[l], &self.qb[l], d, d);
            let mut k = linear(&xn, &self.kw[l], &self.kb[l], d, d);
            let vv = linear(&xn, &self.vw[l], &self.vb[l], d, d);
            hostmath::rope_neox_row(&mut q, self.heads, self.hd, pos, THETA);
            hostmath::rope_neox_row(&mut k, self.heads, self.hd, pos, THETA);
            cache.k[l].extend_from_slice(&k);
            cache.v[l].extend_from_slice(&vv);
            let len = cache.k[l].len() / d;
            let w0 = len.saturating_sub(self.max_ctx);
            let scale = 1.0 / (self.hd as f32).sqrt();
            let attn = attend(&q, &cache.k[l], &cache.v[l], w0, len, self.heads, self.hd, d, scale);
            let o = linear(&attn, &self.ow[l], &self.ob[l], d, d);
            for i in 0..d {
                x[i] += o[i];
            }
            // SwiGLU FFN (no bias)
            let xn2 = hostmath::rmsnorm(&x, &self.norm2[l], EPS);
            let a = hostmath::matvec(&self.w1[l], &xn2, self.ff, d);
            let b = hostmath::matvec(&self.w3[l], &xn2, self.ff, d);
            let g: Vec<f32> = (0..self.ff).map(|i| hostmath::silu(a[i]) * b[i]).collect();
            let ffo = hostmath::matvec(&self.w2[l], &g, d, self.ff);
            for i in 0..d {
                x[i] += ffo[i];
            }
        }
        let ctx = hostmath::rmsnorm(&x, &self.normf, EPS);
        cache.ctx.extend_from_slice(&ctx);
        linear(&ctx, &self.ps1w, &self.ps1b, self.s1v, d)
    }

    /// `decode_s2` for the last position: cross-attend `sibling(sampled_s1)`
    /// (RAW `emb_s1`, no `√d`) over the cached context, return `s2` logits `[s2v]`.
    pub fn dep_step(&self, sampled_s1: u32, ctx: &[f32]) -> Vec<f32> {
        let d = self.d;
        let len = ctx.len() / d;
        let w0 = len.saturating_sub(self.max_ctx);
        let pos_last = len - 1;
        let heads = self.dep_heads;
        let hd = self.dep_hd;
        let scale = 1.0 / (hd as f32).sqrt();

        let sib = &self.emb_s1[sampled_s1 as usize * d..sampled_s1 as usize * d + d];
        let mut q = linear(sib, &self.dqw, &self.dqb, d, d);
        hostmath::rope_neox_row(&mut q, heads, hd, pos_last, THETA);
        // k/v from the cached context (non-causal cross-attention over w0..len)
        let win = len - w0;
        let mut kbuf = vec![0f32; win * d];
        let mut vbuf = vec![0f32; win * d];
        for (wi, j) in (w0..len).enumerate() {
            let cj = &ctx[j * d..j * d + d];
            let mut kj = linear(cj, &self.dkw, &self.dkb, d, d);
            hostmath::rope_neox_row(&mut kj, heads, hd, j, THETA);
            let vj = linear(cj, &self.dvw, &self.dvb, d, d);
            kbuf[wi * d..wi * d + d].copy_from_slice(&kj);
            vbuf[wi * d..wi * d + d].copy_from_slice(&vj);
        }
        let attn = attend(&q, &kbuf, &vbuf, 0, win, heads, hd, d, scale);
        let o = linear(&attn, &self.dow, &self.dob, d, d);
        // norm(context[last] + attn_out)
        let mut sum = vec![0f32; d];
        for i in 0..d {
            sum[i] = ctx[pos_last * d + i] + o[i];
        }
        let normed = hostmath::rmsnorm(&sum, &self.dnorm, EPS);
        linear(&normed, &self.ps2w, &self.ps2b, self.s2v, d)
    }

    /// KV-cached [`dep_step`]: project each **new** ctx position's dependency K/V
    /// once (RoPE'd at its absolute index) into `cache.dep_k/dep_v`, then do a
    /// single windowed cross-attention. Turns the per-decode-step cost from
    /// `O(window)` projections (the whole context, re-projected every step) into
    /// `O(1)` new projections + attention. Numerically identical to [`dep_step`]
    /// (parity gate: `tests/kvcache_parity.rs`).
    /// Extend the dep K/V cache to cover every `ctx` position (RoPE'd at its
    /// absolute index). Idempotent; call once after prefill to share the dep K/V
    /// across sampling forks, or per decode step for the single new position.
    pub fn ensure_dep_kv(&self, cache: &mut Cache) {
        let d = self.d;
        let (heads, hd) = (self.dep_heads, self.dep_hd);
        let len = cache.ctx.len() / d;
        let have = cache.dep_k.len() / d;
        for j in have..len {
            let cj = &cache.ctx[j * d..j * d + d];
            let mut kj = linear(cj, &self.dkw, &self.dkb, d, d);
            hostmath::rope_neox_row(&mut kj, heads, hd, j, THETA);
            let vj = linear(cj, &self.dvw, &self.dvb, d, d);
            cache.dep_k.extend_from_slice(&kj);
            cache.dep_v.extend_from_slice(&vj);
        }
    }

    pub fn dep_step_cached(&self, sampled_s1: u32, cache: &mut Cache) -> Vec<f32> {
        let d = self.d;
        let (heads, hd) = (self.dep_heads, self.dep_hd);
        self.ensure_dep_kv(cache);
        let len = cache.ctx.len() / d;
        let w0 = len.saturating_sub(self.max_ctx);
        let pos_last = len - 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let sib = &self.emb_s1[sampled_s1 as usize * d..sampled_s1 as usize * d + d];
        let mut q = linear(sib, &self.dqw, &self.dqb, d, d);
        hostmath::rope_neox_row(&mut q, heads, hd, pos_last, THETA);
        let win = len - w0;
        let attn = attend(&q, &cache.dep_k[w0 * d..len * d], &cache.dep_v[w0 * d..len * d], 0, win, heads, hd, d, scale);
        let o = linear(&attn, &self.dow, &self.dob, d, d);
        let mut sum = vec![0f32; d];
        for i in 0..d {
            sum[i] = cache.ctx[pos_last * d + i] + o[i];
        }
        let normed = hostmath::rmsnorm(&sum, &self.dnorm, EPS);
        linear(&normed, &self.ps2w, &self.ps2b, self.s2v, d)
    }
}

// -- host kernels (exact reproductions of the WGSL) --------------------------


// Host-parallel loops go through the CPU scheduler's primitives — rayon lives
// only in backend-cpu, so `--device cpuN` pool policy governs every loop.
use backend_cpu::par;

// Elementwise/normalisation math comes from `model::hostmath` — the single
// implementation, checked against the WGSL kernels. Called directly: a local
// alias is how six crates ended up with six copies of rmsnorm.
use model::hostmath;


/// `y[o] = sum_i x[i]*W[o*inp+i] + b[o]` (PyTorch `nn.Linear`, `W` is `[out,inp]`).
/// The projection is the AVX2+FMA `matmul_abt` kernel (`C[out,1] = W[out,inp] ·
/// x[1,inp]ᵀ`): rayon over output rows AND an 8-wide vectorised dot per row — the
/// host KV rollout is a stack of these, so this is the CPU hot path.
fn linear(x: &[f32], w: &[f32], b: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut y = vec![0f32; out];
    backend_cpu::fast_ops::matmul_abt(w, &x[..inp], &mut y, out, inp, 1);
    for (yo, &bo) in y.iter_mut().zip(b) {
        *yo += bo;
    }
    y
}

/// Single-query multi-head attention: `q` `[heads*hd]` over keys/values
/// `[len*d]` (windowed to `[w0,len)`), `scale`d, softmax over the window. Returns
/// the context `[heads*hd]`. Causality is implicit: the query is the newest
/// position and every windowed key is at or before it.
#[allow(clippy::too_many_arguments)]
fn attend(q: &[f32], kc: &[f32], vc: &[f32], w0: usize, len: usize, heads: usize, hd: usize, d: usize, scale: f32) -> Vec<f32> {
    // Each head is independent — compute its [hd] context in parallel, then
    // concatenate in head order (`off = h*hd`) to recover [heads*hd].
    let head = |h: usize| -> Vec<f32> {
        let off = h * hd;
        let win = len - w0;
        let mut sc = vec![0f32; win];
        let mut mx = f32::NEG_INFINITY;
        for (wi, j) in (w0..len).enumerate() {
            let kb = j * d + off;
            let mut dot = 0f32;
            for dd in 0..hd {
                dot += q[off + dd] * kc[kb + dd];
            }
            dot *= scale;
            sc[wi] = dot;
            mx = mx.max(dot);
        }
        let mut sum = 0f32;
        for s in sc.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let mut o = vec![0f32; hd];
        for dd in 0..hd {
            let mut acc = 0f32;
            for (wi, j) in (w0..len).enumerate() {
                acc += sc[wi] / sum * vc[j * d + off + dd];
            }
            o[dd] = acc;
        }
        o
    };
    par::flat_map_f32(heads, head)
}
