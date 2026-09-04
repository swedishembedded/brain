// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS Talker **KV-cached incremental decode** — the real-time lever.
//!
//! The shared GPU decoder ([`crate::gen::TalkerGen`], [`crate::mtp::MtpModel`])
//! has no key/value cache: each autoregressive frame re-runs the *whole* growing
//! context through every layer, so generating `T` frames costs `O(T²)` decoder
//! work. Because the attention is strictly causal, the hidden state of a past
//! position never changes once produced, and a new position only needs (a) its
//! own projections / MLP and (b) attention of its single query against the cached
//! keys/values of all earlier positions. Caching K/V therefore turns each step
//! into `O(1)` projection/MLP work plus `O(t)` attention — `O(T)` amortised per
//! frame instead of `O(T²)`.
//!
//! This module is **purely additive**: it is a self-contained CPU reference of
//! the exact same arithmetic the WGSL engine runs (RMSNorm eps 1e-6, `y = x·Wᵀ`,
//! half-split RoPE base θ, GQA score scale `1/√head_dim`, causal max-subtracted
//! softmax, `SiLU(x)=x·σ(x)`). It does **not** touch the gradient-checked
//! `qwen3::Qwen` / `TalkerGen` forward graphs, which remain the parity reference.
//! [`CpuTalker::forward_full`] reproduces the full-recompute hidden states (used
//! to prove the cache is exact and that the CPU math matches the GPU engine), and
//! [`CpuTalker::step`] is the incremental cached path used by generation.

use std::collections::HashMap;

use backend_cpu::par;

use crate::config::TalkerConfig;
use crate::talker::TextProjection;

// Elementwise/normalisation math comes from `model::hostmath` — the single
// implementation, checked against the WGSL kernels. Called directly rather than
// re-exported: a local alias is how six crates ended up with six copies.
use model::hostmath;


/// Per-layer decoder weights (row-major, brain convention `W:[out,in]`).
pub(crate) struct LayerW {
    ln1: Vec<f32>,    // [d]
    wq: Vec<f32>,     // [hq, d]
    wk: Vec<f32>,     // [hkv, d]
    wv: Vec<f32>,     // [hkv, d]
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>, // [head_dim]
    wo: Vec<f32>,     // [d, hq]
    ln2: Vec<f32>,    // [d]
    gate: Vec<f32>,   // [ff, d]
    up: Vec<f32>,     // [ff, d]
    down: Vec<f32>,   // [d, ff]
}

/// Per-layer running key/value cache for the incremental path.
#[derive(Default, Clone)]
pub(crate) struct Kv {
    k: Vec<f32>, // [t, hkv] post-QK-norm, post-RoPE keys
    v: Vec<f32>, // [t, hkv] values
}

/// The decoder block dimensions shared by [`decoder_layer_step`] /
/// [`decoder_forward_full`] (a Qwen3 GQA block, identical for the Talker and the
/// MTP code-predictor — only the values differ).
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub d: usize,     // d_model
    pub hd: usize,    // head_dim
    pub nh: usize,    // n query heads
    pub nkv: usize,   // n kv heads
    pub ff: usize,    // d_ff
    pub theta: f32,   // rope base
}

/// A CPU-resident, KV-cached Talker decoder. Holds the frozen decoder weights in
/// host memory and (for the cached path) a per-layer growing K/V cache plus the
/// next position index.
pub struct CpuTalker {
    pub cfg: TalkerConfig,
    layers: Vec<LayerW>,
    norm: Vec<f32>, // final RMSNorm gain [d]
    // incremental-decode state
    cache: Vec<Kv>,
    pos: usize,
    // CPU codec/text front-end (populated by `load`; empty for decoder-only test
    // construction). Mirrors `crate::gen::TalkerGen`'s tables.
    pub text: Option<TextProjection>,
    codec_embedding: Vec<f32>, // [vocab, d] (= tok.weight)
    codec_head: Vec<f32>,      // [vocab, d] (= lm_head.weight)
}

pub(crate) const EPS: f32 = 1e-6; // matches the WGSL rmsnorm kernel (hardcoded)

/// `Σ row[k]·x[k]` — the inner dot of `model::hostmath::matvec` and the per-step
/// attention. Uses AVX2+FMA when the CPU supports it (8 f32 lanes/iter), else a
/// scalar fallback.
/// The 8-lane partial sums reorder the reduction, so results differ from the
/// strictly-sequential scalar sum only in the last ~1 ULP — well inside the
/// engine parity tolerances, and the KV-cache exactness tests use one impl for
/// both cached and uncached paths so they stay bit-identical.
#[inline]
pub(crate) fn dot(row: &[f32], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime AVX2+FMA feature check above.
            return unsafe { dot_avx2(row, x) };
        }
    }
    dot_scalar(row, x)
}

#[inline]
fn dot_scalar(row: &[f32], x: &[f32]) -> f32 {
    let n = row.len().min(x.len());
    let mut acc = 0.0f32;
    for k in 0..n {
        acc += row[k] * x[k];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(row: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len().min(x.len());
    // Four independent accumulators hide FMA latency (32 f32/iter).
    let (mut a0, mut a1, mut a2, mut a3) =
        (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    let rp = row.as_ptr();
    let xp = x.as_ptr();
    let mut k = 0usize;
    while k + 32 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(rp.add(k)), _mm256_loadu_ps(xp.add(k)), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(rp.add(k + 8)), _mm256_loadu_ps(xp.add(k + 8)), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(rp.add(k + 16)), _mm256_loadu_ps(xp.add(k + 16)), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(rp.add(k + 24)), _mm256_loadu_ps(xp.add(k + 24)), a3);
        k += 32;
    }
    while k + 8 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(rp.add(k)), _mm256_loadu_ps(xp.add(k)), a0);
        k += 8;
    }
    let sum8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum8);
    let mut s = tmp.iter().sum::<f32>();
    while k < n {
        s += *row.get_unchecked(k) * *x.get_unchecked(k);
        k += 1;
    }
    s
}

/// In-place per-head RMSNorm (QK-norm) over `head_dim` of a `[n_heads*head_dim]`
/// row, gain `w:[head_dim]`.
fn qk_norm(buf: &mut [f32], w: &[f32], n_heads: usize, hd: usize) {
    for h in 0..n_heads {
        let seg = &mut buf[h * hd..h * hd + hd];
        let mut ss = 0.0f32;
        for &v in seg.iter() {
            ss += v * v;
        }
        let inv = 1.0f32 / (ss / hd as f32 + EPS).sqrt();
        for (c, v) in seg.iter_mut().enumerate() {
            *v = w[c] * *v * inv;
        }
    }
}

/// Build the per-layer weight list (`blocks.{l}.*`) from a name->tensor accessor.
/// Shared by the Talker ([`CpuTalker`]) and the MTP ([`crate::gen_kv_mtp::CpuMtp`])
/// since both are the same Qwen3 decoder block with the same leaf names.
pub(crate) fn load_layers(n_layers: u32, take: &dyn Fn(&str) -> Vec<f32>) -> Vec<LayerW> {
    let mut layers = Vec::with_capacity(n_layers as usize);
    for l in 0..n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        layers.push(LayerW {
            ln1: take(&p("ln1.weight")),
            wq: take(&p("attn.wq.weight")),
            wk: take(&p("attn.wk.weight")),
            wv: take(&p("attn.wv.weight")),
            q_norm: take(&p("attn.q_norm.weight")),
            k_norm: take(&p("attn.k_norm.weight")),
            wo: take(&p("attn.wo.weight")),
            ln2: take(&p("ln2.weight")),
            gate: take(&p("mlp.gate.weight")),
            up: take(&p("mlp.up.weight")),
            down: take(&p("mlp.down.weight")),
        });
    }
    layers
}

/// Run **one** new position `x_in:[d]` through a single decoder layer, appending
/// its key/value to `kv` and attending causally over `[0..=pos]`. Returns the
/// layer output `[d]`. The KV-cached incremental core shared by the Talker and
/// the MTP code-predictor (same Qwen3 GQA block arithmetic).
pub(crate) fn decoder_layer_step(
    lw: &LayerW,
    kv: &mut Kv,
    dm: Dims,
    x_in: &[f32],
    pos: usize,
) -> Vec<f32> {
    let Dims { d, hd, nh, nkv, ff, theta } = dm;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let group = nh / nkv;

    // --- attention ---
    let h1 = hostmath::rmsnorm(x_in, &lw.ln1, EPS);
    let mut q = hostmath::matvec(&lw.wq, &h1, hq, d);
    let mut k = hostmath::matvec(&lw.wk, &h1, hkv, d);
    let vv = hostmath::matvec(&lw.wv, &h1, hkv, d);
    qk_norm(&mut q, &lw.q_norm, nh, hd);
    qk_norm(&mut k, &lw.k_norm, nkv, hd);
    hostmath::rope_neox_row(&mut q, nh, hd, pos, theta);
    hostmath::rope_neox_row(&mut k, nkv, hd, pos, theta);

    // append to cache, then attend over all cached positions.
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&vv);
    let kc = &kv.k;
    let vc = &kv.v;
    let t = pos + 1; // cached length
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut ctx = vec![0.0f32; hq];
    let mut scores = vec![0.0f32; t];
    for h in 0..nh {
        let hkv_head = h / group;
        let qh = &q[h * hd..h * hd + hd];
        let mut mx = f32::NEG_INFINITY;
        for (j, sj) in scores.iter_mut().enumerate() {
            let kbase = j * hkv + hkv_head * hd;
            let s = dot(qh, &kc[kbase..kbase + hd]) * scale;
            *sj = s;
            if s > mx {
                mx = s;
            }
        }
        let mut sum = 0.0f32;
        for sj in scores.iter_mut() {
            *sj = (*sj - mx).exp();
            sum += *sj;
        }
        let inv = 1.0f32 / sum;
        let cbase = h * hd;
        for (j, &sj) in scores.iter().enumerate() {
            let p = sj * inv;
            let vbase = j * hkv + hkv_head * hd;
            for dd in 0..hd {
                ctx[cbase + dd] += p * vc[vbase + dd];
            }
        }
    }
    let attn = hostmath::matvec(&lw.wo, &ctx, d, hq);
    let xmid: Vec<f32> = x_in.iter().zip(&attn).map(|(a, b)| a + b).collect();

    // --- SwiGLU MLP ---
    let h2 = hostmath::rmsnorm(&xmid, &lw.ln2, EPS);
    let gate = hostmath::matvec(&lw.gate, &h2, ff, d);
    let up = hostmath::matvec(&lw.up, &h2, ff, d);
    let hmlp: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| hostmath::silu(g) * u).collect();
    let mlp_out = hostmath::matvec(&lw.down, &hmlp, d, ff);
    xmid.iter().zip(&mlp_out).map(|(a, b)| a + b).collect()
}

/// **Full recompute** of all positions from `inputs_embeds:[n, d]` (no cache);
/// returns every position's final-norm hidden state `[n, d]`. The uncached
/// `O(T²)` reference that the cached path is proven equal to. Shared by the
/// Talker and the MTP code-predictor.
pub(crate) fn decoder_forward_full(
    layers: &[LayerW],
    norm: &[f32],
    dm: Dims,
    inputs_embeds: &[f32],
) -> Vec<f32> {
    let Dims { d, hd, nh, nkv, ff, theta } = dm;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let group = nh / nkv;
    let n = inputs_embeds.len() / d;
    assert_eq!(inputs_embeds.len(), n * d);
    let scale = 1.0f32 / (hd as f32).sqrt();

    let mut x = inputs_embeds.to_vec();
    for lw in layers {
        let mut q = vec![0.0f32; n * hq];
        let mut k = vec![0.0f32; n * hkv];
        let mut v = vec![0.0f32; n * hkv];
        for i in 0..n {
            let xi = &x[i * d..i * d + d];
            let h1 = hostmath::rmsnorm(xi, &lw.ln1, EPS);
            let mut qi = hostmath::matvec(&lw.wq, &h1, hq, d);
            let mut ki = hostmath::matvec(&lw.wk, &h1, hkv, d);
            let vi = hostmath::matvec(&lw.wv, &h1, hkv, d);
            qk_norm(&mut qi, &lw.q_norm, nh, hd);
            qk_norm(&mut ki, &lw.k_norm, nkv, hd);
            hostmath::rope_neox_row(&mut qi, nh, hd, i, theta);
            hostmath::rope_neox_row(&mut ki, nkv, hd, i, theta);
            q[i * hq..i * hq + hq].copy_from_slice(&qi);
            k[i * hkv..i * hkv + hkv].copy_from_slice(&ki);
            v[i * hkv..i * hkv + hkv].copy_from_slice(&vi);
        }
        for i in 0..n {
            let mut ctx = vec![0.0f32; hq];
            let mut scores = vec![0.0f32; i + 1];
            for h in 0..nh {
                let hkv_head = h / group;
                let qh = &q[i * hq + h * hd..i * hq + h * hd + hd];
                let mut mx = f32::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate() {
                    let kbase = j * hkv + hkv_head * hd;
                    let mut s = 0.0f32;
                    for dd in 0..hd {
                        s += qh[dd] * k[kbase + dd];
                    }
                    s *= scale;
                    *sj = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0f32;
                for sj in scores.iter_mut() {
                    *sj = (*sj - mx).exp();
                    sum += *sj;
                }
                let inv = 1.0f32 / sum;
                for (j, &sj) in scores.iter().enumerate() {
                    let p = sj * inv;
                    let vbase = j * hkv + hkv_head * hd;
                    for dd in 0..hd {
                        ctx[h * hd + dd] += p * v[vbase + dd];
                    }
                }
            }
            let attn = hostmath::matvec(&lw.wo, &ctx, d, hq);
            let xi = &x[i * d..i * d + d];
            let xmid: Vec<f32> = xi.iter().zip(&attn).map(|(a, b)| a + b).collect();
            let h2 = hostmath::rmsnorm(&xmid, &lw.ln2, EPS);
            let gate = hostmath::matvec(&lw.gate, &h2, ff, d);
            let up = hostmath::matvec(&lw.up, &h2, ff, d);
            let hmlp: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| hostmath::silu(g) * u).collect();
            let mlp_out = hostmath::matvec(&lw.down, &hmlp, d, ff);
            let row = &mut x[i * d..i * d + d];
            for (r, (a, b)) in row.iter_mut().zip(xmid.iter().zip(&mlp_out)) {
                *r = a + b;
            }
        }
    }
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let h = hostmath::rmsnorm(&x[i * d..i * d + d], norm, EPS);
        out[i * d..i * d + d].copy_from_slice(&h);
    }
    out
}

impl CpuTalker {
    /// Dimensions of the Talker decoder block.
    pub(crate) fn dims(&self) -> Dims {
        let c = &self.cfg;
        Dims {
            d: c.d_model as usize,
            hd: c.head_dim as usize,
            nh: c.n_heads as usize,
            nkv: c.n_kv_heads as usize,
            ff: c.d_ff as usize,
            theta: c.rope_theta,
        }
    }

    /// Build from a decoder weight map (`blocks.{l}.*` + `norm.weight`). Mirrors
    /// the names [`crate::gen::TalkerGen`] / [`qwen3::Qwen`] use, so the same
    /// frozen weights can drive either path.
    pub fn from_decoder_map(cfg: TalkerConfig, w: &HashMap<String, Vec<f32>>) -> CpuTalker {
        let take = |n: &str| {
            w.get(n)
                .unwrap_or_else(|| panic!("CpuTalker: missing weight {n}"))
                .clone()
        };
        let layers = load_layers(cfg.n_layers, &take);
        let norm = take("norm.weight");
        let n = cfg.n_layers as usize;
        CpuTalker {
            cfg,
            layers,
            norm,
            cache: vec![Kv::default(); n],
            pos: 0,
            text: None,
            codec_embedding: Vec::new(),
            codec_head: Vec::new(),
        }
    }

    /// Load the decoder weights from a brain Talker checkpoint (the container
    /// written by [`crate::import::import_talker`], also consumed by
    /// [`crate::gen::TalkerGen::load`]).
    pub fn load(path: &str) -> CpuTalker {
        let c = checkpoint::load(path);
        let qcfg = qwen3::QwenConfig::from_json(&c.header["config"]);
        let cfg = TalkerConfig::from_qwen(&qcfg);
        let mut map = HashMap::new();
        for l in 0..cfg.n_layers {
            for leaf in [
                "ln1.weight",
                "attn.wq.weight",
                "attn.wk.weight",
                "attn.wv.weight",
                "attn.q_norm.weight",
                "attn.k_norm.weight",
                "attn.wo.weight",
                "ln2.weight",
                "mlp.gate.weight",
                "mlp.up.weight",
                "mlp.down.weight",
            ] {
                let n = format!("blocks.{l}.{leaf}");
                let t = c
                    .find(&n, "")
                    .cloned()
                    .unwrap_or_else(|| panic!("CpuTalker::load missing {n}"));
                map.insert(n, t);
            }
        }
        map.insert(
            "norm.weight".to_string(),
            c.find("norm.weight", "")
                .cloned()
                .expect("missing norm.weight"),
        );
        let mut t = CpuTalker::from_decoder_map(cfg, &map);

        // CPU codec/text front-end (same tensors `TalkerGen::load` reads).
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("CpuTalker::load missing {name}"))
        };
        t.codec_embedding = take("tok.weight");
        t.codec_head = take("lm_head.weight");
        let fc1_w = take("text_projection.fc1.weight");
        let fc1_b = take("text_projection.fc1.bias");
        let fc2_w = take("text_projection.fc2.weight");
        let fc2_b = take("text_projection.fc2.bias");
        let text_embedding = c.find("text_embedding.weight", "").cloned();
        let inter = fc1_b.len();
        let in_dim = fc1_w.len() / inter;
        let out = fc2_b.len();
        let text_vocab = text_embedding.as_ref().map(|e| e.len() / in_dim).unwrap_or(0);
        t.cfg.text_hidden_size = in_dim as u32;
        if text_vocab > 0 {
            t.cfg.text_vocab_size = text_vocab as u32;
        }
        t.text = Some(TextProjection {
            text_embedding,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            in_dim,
            inter,
            out,
            text_vocab,
        });
        t
    }

    pub fn d(&self) -> usize {
        self.cfg.d_model as usize
    }

    /// Talker codebook-0 embedding row for `id` (`[d_model]`). Mirrors
    /// [`crate::gen::TalkerGen::codec_embed`].
    pub fn codec_embed(&self, id: u32) -> &[f32] {
        let d = self.d();
        let s = id as usize * d;
        &self.codec_embedding[s..s + d]
    }

    /// Codebook-0 logits (`[vocab]`) for a single final-norm hidden row. Mirrors
    /// [`crate::gen::TalkerGen::codec_head_logits`].
    pub fn codec_head_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        assert_eq!(hidden_row.len(), d);
        let mut out = vec![0.0f32; v];
        par::each_mut(&mut out, |o, dst| {
            let wrow = &self.codec_head[o * d..o * d + d];
            *dst = (0..d).map(|k| wrow[k] * hidden_row[k]).sum();
        });
        out
    }

    /// Number of positions consumed by the incremental cache so far (the next
    /// [`CpuTalker::step`] decodes absolute position `pos`).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The text projection front-end read by [`CpuTalker::load`], or `None`
    /// for a decoder-only test construction that never loaded one.
    pub fn text_projection(&self) -> Option<&crate::talker::TextProjection> {
        self.text.as_ref()
    }

    /// Reset the incremental K/V cache (start a fresh sequence).
    pub fn reset(&mut self) {
        for kv in &mut self.cache {
            kv.k.clear();
            kv.v.clear();
        }
        self.pos = 0;
    }

    /// Incremental cached decode of **one** position. `embed:[d]` is the input
    /// embedding of the new position; returns its final-norm hidden state `[d]`.
    /// The position index advances automatically (call [`CpuTalker::reset`] to
    /// start a new sequence).
    pub fn step(&mut self, embed: &[f32]) -> Vec<f32> {
        let d = self.d();
        assert_eq!(embed.len(), d, "embed must be [d_model]");
        let pos = self.pos;
        let dims = self.dims();
        let mut x = embed.to_vec();
        for l in 0..self.layers.len() {
            x = decoder_layer_step(&self.layers[l], &mut self.cache[l], dims, &x, pos);
        }
        self.pos += 1;
        hostmath::rmsnorm(&x, &self.norm, EPS)
    }

    /// **Full recompute** of all positions from `inputs_embeds:[n, d]` (no cache);
    /// returns every position's final-norm hidden state `[n, d]`. This is the
    /// reference the cached path is proven equal to, and mirrors
    /// [`crate::gen::TalkerGen::forward`]. Uses an independent (non-mutating) pass
    /// so it can serve as the `O(T²)` uncached baseline.
    pub fn forward_full(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        decoder_forward_full(&self.layers, &self.norm, self.dims(), inputs_embeds)
    }
}

/// Prompt assembly against the CPU Talker's own host tables, so a purely
/// host-side pipeline (`crate::engine`, `crate::batch`) can build its prompts
/// without also constructing the `gpu_core`-backed [`crate::gen::TalkerGen`]
/// just to read three embedding tables. Same tensors, same tables - `load`
/// reads exactly what `TalkerGen::load` does.
impl crate::prompt::TalkerHost for CpuTalker {
    fn d(&self) -> usize {
        self.cfg.d_model as usize
    }
    fn text(&self) -> &TextProjection {
        self.text.as_ref().expect("CpuTalker has no text_projection: built decoder-only, not loaded from a checkpoint")
    }
    fn codec_embed(&self, id: u32) -> &[f32] {
        CpuTalker::codec_embed(self, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TalkerModel;
    use std::collections::HashMap;
    use std::time::Instant;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// The SIMD `dot` must agree with the strictly-scalar reference within a few
    /// ULPs across odd lengths (covers the 32/8-wide bodies and the scalar tail).
    #[test]
    fn dot_simd_matches_scalar() {
        use data::rng::Rng;
        let mut rng = Rng::new(1234);
        for &n in &[0usize, 1, 7, 8, 31, 32, 33, 127, 128, 129, 1024, 3072] {
            let a: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let simd = dot(&a, &b);
            let scalar = dot_scalar(&a, &b);
            let tol = 1e-4 * (n as f32).sqrt().max(1.0);
            assert!(
                (simd - scalar).abs() <= tol,
                "n={n}: simd={simd} scalar={scalar} diff={}",
                (simd - scalar).abs()
            );
        }
    }

    /// Decoder param names (mirrors gen.rs/mtp.rs `decoder_param_list`).
    fn decoder_names(cfg: &TalkerConfig) -> Vec<String> {
        let mut v = Vec::new();
        for l in 0..cfg.n_layers {
            for leaf in [
                "ln1.weight",
                "attn.wq.weight",
                "attn.wk.weight",
                "attn.wv.weight",
                "attn.q_norm.weight",
                "attn.k_norm.weight",
                "attn.wo.weight",
                "ln2.weight",
                "mlp.gate.weight",
                "mlp.up.weight",
                "mlp.down.weight",
            ] {
                v.push(format!("blocks.{l}.{leaf}"));
            }
        }
        v.push("norm.weight".to_string());
        v
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The KV-cached incremental decode must reproduce the full-recompute hidden
    /// states (exactness of the cache), and both must match the GPU engine's
    /// `TalkerModel::logits_all` (faithfulness of the CPU math to the kernels).
    #[test]
    fn kv_cache_matches_full_recompute_and_engine() {
        if gpu_disabled() {
            return;
        }
        let cfg = TalkerConfig::tiny(); // d16 L2 GQA 4/2 hd8 ff32 vocab23 untied
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let seq = 7usize;
        let model = TalkerModel::new_trainable(cfg.clone(), 1, seq as u32, 7);

        // pull decoder weights out of the engine model into a CPU map.
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for n in decoder_names(&cfg) {
            map.insert(n.clone(), model.read_weight(&n));
        }
        let mut cpu = CpuTalker::from_decoder_map(cfg.clone(), &map);

        // reference: GPU engine logits over a token sequence (the untied head).
        let ids: Vec<u32> = (0..seq).map(|i| ((i * 5 + 1) % v) as u32).collect();
        let reference = model.logits_all(&ids); // [seq*v]

        // embed ids via the codec table (tok.weight) to drive the CPU decoder.
        let tok = model.read_weight("tok.weight"); // [vocab, d]
        let head = model.read_weight("lm_head.weight"); // [vocab, d]
        let mut embeds = vec![0.0f32; seq * d];
        for (i, &id) in ids.iter().enumerate() {
            embeds[i * d..(i + 1) * d]
                .copy_from_slice(&tok[id as usize * d..(id as usize + 1) * d]);
        }

        // CPU full-recompute hidden -> logits, vs engine reference.
        let hidden_full = cpu.forward_full(&embeds);
        let logits_of = |hidden: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; seq * v];
            for i in 0..seq {
                let h = &hidden[i * d..i * d + d];
                for o in 0..v {
                    let wrow = &head[o * d..o * d + d];
                    out[i * v + o] = (0..d).map(|k| wrow[k] * h[k]).sum();
                }
            }
            out
        };
        let logits_full = logits_of(&hidden_full);
        let engine_err = maxabs(&logits_full, &reference);

        // CPU cached incremental hidden, fed one position at a time.
        cpu.reset();
        let mut hidden_cached = vec![0.0f32; seq * d];
        for i in 0..seq {
            let h = cpu.step(&embeds[i * d..(i + 1) * d]);
            hidden_cached[i * d..(i + 1) * d].copy_from_slice(&h);
        }
        let cache_err = maxabs(&hidden_cached, &hidden_full);
        let logits_cached = logits_of(&hidden_cached);
        let cached_vs_ref = maxabs(&logits_cached, &reference);

        eprintln!(
            "KV-cache: cached-vs-fullrecompute max-abs={cache_err:.3e}, \
             fullrecompute-vs-engine max-abs={engine_err:.3e}, \
             cached-vs-engine logits max-abs={cached_vs_ref:.3e}"
        );
        // The cache is algebraically identical to full recompute -> ~fp noise.
        assert!(cache_err < 1e-4, "KV-cache not exact vs recompute: {cache_err}");
        // CPU math faithfully matches the WGSL engine (differs only by f32 sum
        // order across kernels).
        assert!(engine_err < 1e-2, "CPU decoder diverges from engine: {engine_err}");
        assert!(cached_vs_ref < 1e-2, "cached logits diverge from engine: {cached_vs_ref}");
    }

    /// Measure the cached vs uncached (full-recompute-per-frame) cost over a
    /// ~20-frame generation. Uses a modestly-sized decoder so the O(T²)→O(T)
    /// difference is visible while staying light on a loaded machine.
    #[test]
    fn kv_cache_speedup_20_frames() {
        // pure CPU arithmetic — no GPU needed.
        let cfg = TalkerConfig {
            n_layers: 3,
            d_model: 192,
            head_dim: 32,
            n_heads: 6,
            n_kv_heads: 3,
            d_ff: 384,
            vocab: 64,
            ..TalkerConfig::tiny()
        };
        let d = cfg.d_model as usize;
        let frames = 20usize;

        // deterministic pseudo-random weights.
        use data::rng::Rng;
        let mut rng = Rng::new(42);
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for n in decoder_names(&cfg) {
            let numel = {
                let hd = cfg.head_dim as usize;
                let dd = cfg.d_model as usize;
                let ff = cfg.d_ff as usize;
                let hq = cfg.n_heads as usize * hd;
                let hkv = cfg.n_kv_heads as usize * hd;
                if n.ends_with("q_norm.weight") || n.ends_with("k_norm.weight") {
                    hd
                } else if n.ends_with("ln1.weight")
                    || n.ends_with("ln2.weight")
                    || n == "norm.weight"
                {
                    dd
                } else if n.ends_with("attn.wq.weight") {
                    hq * dd
                } else if n.ends_with("attn.wk.weight") || n.ends_with("attn.wv.weight") {
                    hkv * dd
                } else if n.ends_with("attn.wo.weight") {
                    dd * hq
                } else if n.ends_with("mlp.down.weight") {
                    dd * ff
                } else {
                    ff * dd
                }
            };
            let w = if n.contains("norm") || n.ends_with("ln1.weight") || n.ends_with("ln2.weight") {
                vec![1.0f32; numel]
            } else {
                (0..numel).map(|_| rng.next_gaussian() as f32 * 0.02).collect()
            };
            map.insert(n, w);
        }
        let mut cpu = CpuTalker::from_decoder_map(cfg.clone(), &map);

        let mut embeds: Vec<f32> = (0..frames * d)
            .map(|_| rng.next_gaussian() as f32 * 0.1)
            .collect();

        // Uncached: re-run the whole growing context each frame (current path).
        let t0 = Instant::now();
        let mut sink = 0.0f32;
        for f in 1..=frames {
            let h = cpu.forward_full(&embeds[..f * d]);
            sink += h[h.len() - 1];
        }
        let uncached = t0.elapsed();

        // Cached: one incremental step per frame.
        cpu.reset();
        let t1 = Instant::now();
        for f in 0..frames {
            let h = cpu.step(&embeds[f * d..(f + 1) * d]);
            sink += h[0];
        }
        let cached = t1.elapsed();
        // keep `sink`/`embeds` from being optimised away.
        embeds[0] = sink * 0.0;

        let unc_ms = uncached.as_secs_f64() * 1e3;
        let cac_ms = cached.as_secs_f64() * 1e3;
        eprintln!(
            "KV-cache speedup over {frames} frames (d={d}, L={}): \
             uncached {unc_ms:.2} ms ({:.3} ms/frame) vs cached {cac_ms:.2} ms \
             ({:.3} ms/frame) -> {:.1}x",
            cfg.n_layers,
            unc_ms / frames as f64,
            cac_ms / frames as f64,
            unc_ms / cac_ms.max(1e-9),
        );
        assert!(cached < uncached, "cached must be faster than full recompute");
    }
}
