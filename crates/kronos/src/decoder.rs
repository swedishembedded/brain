// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The autoregressive decoder forward: `decode_s1` (tokens+calendar → s1
//! logits + context) and `decode_s2` (context + sampled s1 → s2 logits via the
//! dependency layer). Reuses the shared causal `transformer_block`; the new
//! pieces are the hierarchical embedding (s1/s2 fused, `×√d` scale), the summed
//! calendar embeddings, and the non-causal scaled cross-attention.
//!
//! Embedding tables (emb_s1/s2 + the five calendar tables) are kept on the host
//! and gathered per step — cheap next to the transformer, and it keeps the token
//! ids (host `u32`s) out of the device path.

use crate::config::KronosConfig;
use crate::nn::{self, Ops, ATTN_APPLY_FULL, ATTN_SCORES_QK, ATTN_SOFTMAX_FULL};
use gpu_core::{f, DeviceBuffer, Gpu};
use std::collections::HashMap;
use std::sync::OnceLock;

const CAL: [(&str, usize); 5] =
    [("minute", 60), ("hour", 24), ("weekday", 7), ("day", 32), ("month", 13)];

pub struct KronosDecoder {
    gpu: Gpu,
    cfg: KronosConfig,
    w: HashMap<String, DeviceBuffer>,
    // host copies of the gathered tables
    emb_s1: Vec<f32>,
    emb_s2: Vec<f32>,
    cal: Vec<Vec<f32>>, // one table per CAL entry, in CAL order
    // lazily-built, reused host weight set for the KV-cached rollout (reading all
    // ~24M weights off the device is expensive; do it once, not per sample).
    host_w: OnceLock<crate::kvcache::HostW>,
}

impl KronosDecoder {
    pub fn from_weights(
        cfg: KronosConfig,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosDecoder, String> {
        Self::from_weights_on(Gpu::new(nn::PIPELINES), cfg, weights)
    }

    /// Build on an existing device handle — see `KronosTokenizer::from_weights_on`.
    pub fn from_weights_on(
        gpu: Gpu,
        cfg: KronosConfig,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosDecoder, String> {
        let w = nn::load_weights(&gpu, &cfg.param_list(), weights)?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.get(name).cloned().ok_or_else(|| format!("kronos: missing {name}"))
        };
        let suffix = if cfg.learn_te { "weight" } else { "emb.weight" };
        let cal: Result<Vec<Vec<f32>>, String> =
            CAL.iter().map(|(n, _)| get(&format!("time_emb.{n}_embed.{suffix}"))).collect();
        Ok(KronosDecoder {
            emb_s1: get("embedding.emb_s1.weight")?,
            emb_s2: get("embedding.emb_s2.weight")?,
            cal: cal?,
            gpu,
            cfg,
            w,
            host_w: OnceLock::new(),
        })
    }

    pub fn config(&self) -> &KronosConfig {
        &self.cfg
    }

    fn ops(&self) -> Ops<'_> {
        Ops { gpu: &self.gpu, w: &self.w, rope_theta: 10000.0 }
    }

    /// Gather `rows[i]` of a `[vocab, d]` table into a `[T, d]` host buffer,
    /// scaled by `scale`.
    fn gather(table: &[f32], rows: &[u32], d: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows.len() * d];
        for (i, &r) in rows.iter().enumerate() {
            let src = (r as usize) * d;
            for c in 0..d {
                out[i * d + c] = table[src + c] * scale;
            }
        }
        out
    }

    /// `decode_s1`: fuse the (s1,s2) embeddings (+calendar), run the causal
    /// transformer, return `(s1_logits [T, s1_vocab], context [T, d])`.
    /// `stamp` is `[T, 5]` calendar indices (minute,hour,weekday,day,month).
    /// Host-assemble the decoder input embedding `x` `[T, d]` from the token
    /// streams: hierarchical `[emb_s1·√d | emb_s2·√d] → fusion_proj` plus the
    /// summed calendar embeddings (skipped for an empty `stamp`, matching the
    /// reference `decode_s1(..., stamp=None)`). Returns the on-device buffer.
    fn embed_x(&self, s1: &[u32], s2: &[u32], stamp: &[u32]) -> DeviceBuffer {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let t = s1.len();
        let ops = self.ops();
        let sqrt_d = (d as f32).sqrt();

        let e1 = Self::gather(&self.emb_s1, s1, d, sqrt_d);
        let e2 = Self::gather(&self.emb_s2, s2, d, sqrt_d);
        let mut cat = vec![0.0f32; t * 2 * d];
        for i in 0..t {
            cat[i * 2 * d..i * 2 * d + d].copy_from_slice(&e1[i * d..i * d + d]);
            cat[i * 2 * d + d..i * 2 * d + 2 * d].copy_from_slice(&e2[i * d..i * d + d]);
        }
        let catd = self.gpu.storage_init("cat", &cat);
        let x = ops.linear(&catd, "embedding.fusion_proj.weight", "embedding.fusion_proj.bias", t, 2 * d, d);
        if !stamp.is_empty() {
            let mut te = vec![0.0f32; t * d];
            for (ci, (_, _)) in CAL.iter().enumerate() {
                let idx: Vec<u32> = (0..t).map(|i| stamp[i * 5 + ci]).collect();
                let g = Self::gather(&self.cal[ci], &idx, d, 1.0);
                for j in 0..t * d {
                    te[j] += g[j];
                }
            }
            let ted = self.gpu.storage_init("te", &te);
            ops.add(&ted, &x, t * d);
        }
        x
    }

    /// The host token-embedding `x` `[T, d]` read back to the host — the exact
    /// input the ONNX/NPU `decode_s1` graph consumes (feed it to
    /// [`core_forward_s1`](Self::core_forward_s1) or the NPU s1 session).
    pub fn embed_tokens(&self, s1: &[u32], s2: &[u32], stamp: &[u32]) -> Vec<f32> {
        let x = self.embed_x(s1, s2, stamp);
        self.gpu.read(&x, s1.len() * self.cfg.d_model)
    }

    /// The RAW `emb_s1` sibling embedding `sib` `[T, d]` (no `√d` scale) that
    /// `decode_s2` cross-attends — the `sib` input the NPU s2 graph consumes.
    pub fn sib_embed(&self, s1: &[u32]) -> Vec<f32> {
        Self::gather(&self.emb_s1, s1, self.cfg.d_model, 1.0)
    }

    pub fn decode_s1(&self, s1: &[u32], s2: &[u32], stamp: &[u32]) -> (Vec<f32>, DeviceBuffer) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let t = s1.len();
        let ops = self.ops();

        let x = self.embed_x(s1, s2, stamp);

        // causal transformer blocks + final norm
        for i in 0..cfg.n_layers {
            ops.transformer_block(&format!("transformer.{i}"), &x, t, d, cfg.ff_dim, cfg.n_heads);
        }
        let ctx = self.gpu.storage((t * d) as u64);
        ops.rms(&x, "norm.weight", &ctx, d, t);

        // proj_s1
        let logits = ops.linear(&ctx, "head.proj_s1.weight", "head.proj_s1.bias", t, d, cfg.s1_vocab());
        (self.gpu.read(&logits, t * cfg.s1_vocab()), ctx)
    }

    /// Run the decoder core (transformer stack → final norm → `proj_s1`) on a
    /// host-assembled embedding `x_emb` `[T, d]` — the fusion_proj output with
    /// calendar embeddings already summed in. Returns `(s1_logits [T, s1_vocab],
    /// ctx [T, d])`. This is exactly the compute the ONNX/NPU decoder graph
    /// performs (`kronos_decoder_topology`), so it is the parity reference: same
    /// `x` in, same `s1_logits`+`ctx` out. `decode_s1` builds `x` on-device and
    /// keeps `ctx` there; this variant takes `x` from the host and reads both
    /// back, matching the graph's I/O contract.
    pub fn core_forward_s1(&self, x_emb: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        assert_eq!(x_emb.len(), t * d, "x_emb must be [T, d]");
        let ops = self.ops();
        let x = self.gpu.storage_init("xemb", x_emb);
        for i in 0..cfg.n_layers {
            ops.transformer_block(&format!("transformer.{i}"), &x, t, d, cfg.ff_dim, cfg.n_heads);
        }
        let ctx = self.gpu.storage((t * d) as u64);
        ops.rms(&x, "norm.weight", &ctx, d, t);
        let logits = ops.linear(&ctx, "head.proj_s1.weight", "head.proj_s1.bias", t, d, cfg.s1_vocab());
        (self.gpu.read(&logits, t * cfg.s1_vocab()), self.gpu.read(&ctx, t * d))
    }

    /// `decode_s2`: cross-attend the sampled-s1 sibling embedding against the
    /// context, return `s2_logits [T, s2_vocab]`.
    pub fn decode_s2(&self, context: &DeviceBuffer, sampled_s1: &[u32]) -> Vec<f32> {
        let t = sampled_s1.len();
        // sibling embedding (RAW emb_s1, no √d scale)
        let sib = Self::gather(&self.emb_s1, sampled_s1, self.cfg.d_model, 1.0);
        let sibd = self.gpu.storage_init("sib", &sib);
        self.dep_forward(context, &sibd, t)
    }

    /// Run the dependency-layer cross-attention on host-provided `ctx` `[T,d]`
    /// and sibling embedding `sib_emb` `[T,d]` (RAW `emb_s1` rows). Returns
    /// `s2_logits [T, s2_vocab]`. This is exactly the `decode_s2` ONNX/NPU graph
    /// input contract (`kronos_decoder_topology::build_kronos_dep_graph`), so it
    /// is that graph's parity reference.
    pub fn core_forward_s2(&self, ctx: &[f32], sib_emb: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        assert_eq!(ctx.len(), t * d, "ctx must be [T, d]");
        assert_eq!(sib_emb.len(), t * d, "sib_emb must be [T, d]");
        let ctxd = self.gpu.storage_init("ctx_in", ctx);
        let sibd = self.gpu.storage_init("sib_in", sib_emb);
        self.dep_forward(&ctxd, &sibd, t)
    }

    /// Shared dependency-layer body: non-causal scaled cross-attention (q from
    /// `sibd`, k/v from `context`) → `norm(context + attn)` → `proj_s2`.
    ///
    /// **No RoPE.** The reference's cross-attention builds its rotary table from
    /// the QUERY length, and `decode_s2` is only ever called with a single
    /// sampled `s1` (query length 1), so the table is position 0 - `cos=1`,
    /// `sin=0` - and the rotation is the identity for the keys too (the length-1
    /// table broadcasts over the whole key window). The dependency layer is
    /// therefore position-agnostic at inference, and rotating q at the last
    /// position against keys at their own positions is a different operator: it
    /// changed the sampled `s2` token from the second rollout step onward.
    fn dep_forward(&self, context: &DeviceBuffer, sibd: &DeviceBuffer, t: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let heads = cfg.dep_n_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let ops = self.ops();

        // cross-attention: q from sibling, k/v from context (non-causal, scaled,
        // no rotary - see the note above)
        let q = ops.linear(sibd, "dep_layer.cross_attn.q_proj.weight", "dep_layer.cross_attn.q_proj.bias", t, d, d);
        let k = ops.linear(context, "dep_layer.cross_attn.k_proj.weight", "dep_layer.cross_attn.k_proj.bias", t, d, d);
        let v = ops.linear(context, "dep_layer.cross_attn.v_proj.weight", "dep_layer.cross_attn.v_proj.bias", t, d, d);
        let scores = self.gpu.storage((heads * t * t) as u64);
        let sc = self.gpu.step(
            ATTN_SCORES_QK,
            &[&q, &k, &scores],
            &[1, heads as u32, t as u32, hd as u32, d as u32, 0, f(scale)], // causal=0
            (heads * t * t) as u32,
        );
        self.gpu.submit(&[], &[sc]);
        let probs = self.gpu.storage((heads * t * t) as u64);
        let sm = self.gpu.step(ATTN_SOFTMAX_FULL, &[&scores, &probs], &[1, heads as u32, t as u32], (heads * t) as u32);
        self.gpu.submit(&[], &[sm]);
        let ctxo = self.gpu.storage((t * d) as u64);
        let ap = self.gpu.step(
            ATTN_APPLY_FULL,
            &[&probs, &v, &ctxo],
            &[1, heads as u32, t as u32, hd as u32, d as u32, d as u32],
            (heads * t * hd) as u32,
        );
        self.gpu.submit(&[], &[ap]);
        let o = ops.linear(&ctxo, "dep_layer.cross_attn.out_proj.weight", "dep_layer.cross_attn.out_proj.bias", t, d, d);

        // norm(context + attn_out)
        let sum = self.gpu.storage((t * d) as u64);
        // sum = context; then += o ; then rmsnorm
        let ccopy = self.gpu.read(context, t * d);
        let sumd = self.gpu.storage_init("sum", &ccopy);
        ops.add(&o, &sumd, t * d);
        let _ = sum;
        let normed = self.gpu.storage((t * d) as u64);
        ops.rms(&sumd, "dep_layer.norm.weight", &normed, d, t);

        // proj_s2
        let logits = ops.linear(&normed, "head.proj_s2.weight", "head.proj_s2.bias", t, d, cfg.s2_vocab());
        self.gpu.read(&logits, t * cfg.s2_vocab())
    }

    /// The host weight set for the KV-cached rollout, built once and cached. All
    /// samples of a forecast (and successive forecasts on this decoder) reuse it,
    /// so the ~24M-weight device→host read happens a single time.
    pub fn host_weights(&self) -> &crate::kvcache::HostW {
        self.host_w.get_or_init(|| self.build_host_weights())
    }

    /// Read the decoder's weights + embedding tables to the host and assemble a
    /// [`crate::kvcache::HostW`] for the KV-cached rollout (the fast path).
    fn build_host_weights(&self) -> crate::kvcache::HostW {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let ff = cfg.ff_dim;
        let rd = |name: &str, numel: usize| {
            self.gpu.read(self.w.get(name).unwrap_or_else(|| panic!("kronos: weight {name}")), numel)
        };
        let per = |suffix: &str, numel: usize| -> Vec<Vec<f32>> {
            (0..cfg.n_layers).map(|l| rd(&format!("transformer.{l}.{suffix}"), numel)).collect()
        };
        let proj = |p: &str| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            (per(&format!("self_attn.{p}.weight"), d * d), per(&format!("self_attn.{p}.bias"), d))
        };
        let (qw, qb) = proj("q_proj");
        let (kw, kb) = proj("k_proj");
        let (vw, vb) = proj("v_proj");
        let (ow, ob) = proj("out_proj");
        let dep = |p: &str, numel: usize| rd(&format!("dep_layer.cross_attn.{p}"), numel);
        crate::kvcache::HostW {
            d,
            ff,
            nl: cfg.n_layers,
            heads: cfg.n_heads,
            hd: d / cfg.n_heads,
            s1v: cfg.s1_vocab(),
            s2v: cfg.s2_vocab(),
            dep_heads: cfg.dep_n_heads,
            dep_hd: d / cfg.dep_n_heads,
            max_ctx: cfg.max_context,
            sd: (d as f32).sqrt(),
            norm1: per("norm1.weight", d),
            qw,
            qb,
            kw,
            kb,
            vw,
            vb,
            ow,
            ob,
            norm2: per("norm2.weight", d),
            w1: per("ffn.w1.weight", ff * d),
            w3: per("ffn.w3.weight", ff * d),
            w2: per("ffn.w2.weight", d * ff),
            normf: rd("norm.weight", d),
            ps1w: rd("head.proj_s1.weight", cfg.s1_vocab() * d),
            ps1b: rd("head.proj_s1.bias", cfg.s1_vocab()),
            ps2w: rd("head.proj_s2.weight", cfg.s2_vocab() * d),
            ps2b: rd("head.proj_s2.bias", cfg.s2_vocab()),
            fusw: rd("embedding.fusion_proj.weight", d * 2 * d),
            fusb: rd("embedding.fusion_proj.bias", d),
            dqw: dep("q_proj.weight", d * d),
            dqb: dep("q_proj.bias", d),
            dkw: dep("k_proj.weight", d * d),
            dkb: dep("k_proj.bias", d),
            dvw: dep("v_proj.weight", d * d),
            dvb: dep("v_proj.bias", d),
            dow: dep("out_proj.weight", d * d),
            dob: dep("out_proj.bias", d),
            dnorm: rd("dep_layer.norm.weight", d),
            emb_s1: self.emb_s1.clone(),
            emb_s2: self.emb_s2.clone(),
            cal: self.cal.clone(),
        }
    }

    // ---- GPU/CPU-portable incremental KV-cache decode -----------------------
    //
    // The device twin of [`crate::kvcache::HostW::step_token`]: one new `(s1,s2)`
    // token is embedded, run through the causal decoder with its RoPE'd K/V
    // appended to a persistent per-layer cache, and attended by a SINGLE query
    // over the cached keys (sliding window `[w0, t)` where `w0 =
    // t.saturating_sub(max_context)`). Every stage is expressed in the shared
    // WGSL op set (matmul/bias/rmsnorm/silu_gate/add + the decode kernels
    // rope_at/kv_append/attn_decode_*), so it runs on whatever backend `Gpu`
    // selected — real GPU or the wgsl-cpu JIT (`BRAIN_DEVICE=cpu`). `O(T)` per
    // token vs `decode_s1`'s `O(T²)` full recompute.

    /// Allocate a fresh GPU KV cache for a rollout of up to `cap` tokens.
    pub fn new_gpu_cache(&self, cap: usize) -> GpuKvCache {
        let d = self.cfg.d_model;
        let heads = self.cfg.n_heads;
        let k = (0..self.cfg.n_layers).map(|_| self.gpu.storage((cap * d) as u64)).collect();
        let v = (0..self.cfg.n_layers).map(|_| self.gpu.storage((cap * d) as u64)).collect();
        GpuKvCache {
            k,
            v,
            ctx: self.gpu.storage((cap * d) as u64),
            scores: self.gpu.storage((heads * cap) as u64),
            probs: self.gpu.storage((heads * cap) as u64),
            cap,
            pos: 0,
        }
    }

    /// RoPE-at-absolute-position over `n_rows` contiguous rows of `buf`
    /// (`[n_rows, heads*hd]`), row `r` rotated at position `pos_base + r`
    /// (NeoX half-split, θ=`rope_theta`). Generalises `rope_neox` (which is the
    /// `pos_base=0` case) so the dep-layer window can rope keys at absolute
    /// positions `w0..len`.
    fn rope_at_rows(&self, buf: &DeviceBuffer, pos_base: usize, n_rows: usize, heads: usize, hd: usize) {
        let half = hd / 2;
        let st = self.gpu.step(
            crate::nn::ROPE_AT,
            &[buf],
            &[n_rows as u32, heads as u32, hd as u32, (heads * hd) as u32, 0, pos_base as u32, f(self.ops().rope_theta)],
            (n_rows * heads * half) as u32,
        );
        self.gpu.submit(&[], &[st]);
    }

    /// Append `src` `[width]` into cache row `row` (`dst[row*width..] = src`).
    fn kv_append(&self, src: &DeviceBuffer, dst: &DeviceBuffer, width: usize, row: usize) {
        let st = self.gpu.step(crate::nn::KV_APPEND, &[src, dst], &[width as u32, row as u32], width as u32);
        self.gpu.submit(&[], &[st]);
    }

    /// Single-query windowed attention over a cached K/V: `q` `[heads*hd]` vs
    /// `kcache`/`vcache` (`[*, heads*hd]`, rows `0..t` valid), window `[w0, t)`,
    /// max-subtracted softmax, `scale`d. Writes the context `[heads*hd]` into
    /// `out`. Uses `scores`/`probs` (row-stride `cap`) as scratch.
    #[allow(clippy::too_many_arguments)]
    fn decode_attend(&self, q: &DeviceBuffer, kcache: &DeviceBuffer, vcache: &DeviceBuffer, out: &DeviceBuffer, scores: &DeviceBuffer, probs: &DeviceBuffer, heads: usize, hd: usize, t: usize, w0: usize, cap: usize, scale: f32) {
        let g = &self.gpu;
        let kv_stride = heads * hd; // MHA: n_kv_heads = n_heads, group = 1
        let sc = if w0 == 0 {
            g.step(crate::nn::ATTN_DECODE_SCORES, &[q, kcache, scores], &[heads as u32, 1, hd as u32, t as u32, cap as u32, kv_stride as u32, f(scale)], (heads * t) as u32)
        } else {
            g.step(crate::nn::ATTN_DECODE_SCORES_WIN, &[q, kcache, scores], &[heads as u32, 1, hd as u32, t as u32, cap as u32, kv_stride as u32, w0 as u32, f(scale)], (heads * t) as u32)
        };
        g.submit(&[], &[sc]);
        let sm = g.step(crate::nn::DECODE_SOFTMAX, &[scores, probs], &[heads as u32, t as u32, cap as u32], heads as u32);
        g.submit(&[], &[sm]);
        let ap = g.step(crate::nn::ATTN_DECODE_APPLY, &[probs, vcache, out], &[heads as u32, 1, hd as u32, t as u32, cap as u32, kv_stride as u32], (heads * hd) as u32);
        g.submit(&[], &[ap]);
    }

    /// **Incremental S1 decode** of one `(s1,s2)` token (`stamp` = 5 calendar
    /// indices, or empty) at the current cache position. Mirrors
    /// [`crate::kvcache::HostW::step_token`]: appends this token's K/V to the
    /// per-layer cache, stashes its final-norm context row into `cache.ctx` (for
    /// [`Self::kv_dep_step`]), and returns the token's `s1` logits `[s1_vocab]`.
    pub fn kv_step(&self, cache: &mut GpuKvCache, s1: u32, s2: u32, stamp: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let ff = cfg.ff_dim;
        let heads = cfg.n_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = cache.pos;
        let cap = cache.cap;
        assert!(pos < cap, "kv_step: pos {pos} exceeds cache cap {cap}");
        let t = pos + 1;
        let w0 = t.saturating_sub(cfg.max_context);
        let ops = self.ops();
        let g = &self.gpu;

        // hierarchical embedding [1, d] (fusion_proj + calendar), on device.
        let x = self.embed_x(&[s1], &[s2], stamp);

        for l in 0..cfg.n_layers {
            let pfx = format!("transformer.{l}");
            // --- self-attention (RMSNorm -> qkv+bias -> RoPE@pos -> append -> attend) ---
            let xn = g.storage(d as u64);
            ops.rms(&x, &format!("{pfx}.norm1.weight"), &xn, d, 1);
            let q = ops.linear(&xn, &format!("{pfx}.self_attn.q_proj.weight"), &format!("{pfx}.self_attn.q_proj.bias"), 1, d, d);
            let k = ops.linear(&xn, &format!("{pfx}.self_attn.k_proj.weight"), &format!("{pfx}.self_attn.k_proj.bias"), 1, d, d);
            let v = ops.linear(&xn, &format!("{pfx}.self_attn.v_proj.weight"), &format!("{pfx}.self_attn.v_proj.bias"), 1, d, d);
            self.rope_at_rows(&q, pos, 1, heads, hd);
            self.rope_at_rows(&k, pos, 1, heads, hd);
            self.kv_append(&k, &cache.k[l], d, pos);
            self.kv_append(&v, &cache.v[l], d, pos);
            let ctxb = g.storage(d as u64);
            self.decode_attend(&q, &cache.k[l], &cache.v[l], &ctxb, &cache.scores, &cache.probs, heads, hd, t, w0, cap, scale);
            let o = ops.linear(&ctxb, &format!("{pfx}.self_attn.out_proj.weight"), &format!("{pfx}.self_attn.out_proj.bias"), 1, d, d);
            ops.add(&o, &x, d); // x += attn_out

            // --- SwiGLU FFN (no bias) ---
            let xn2 = g.storage(d as u64);
            ops.rms(&x, &format!("{pfx}.norm2.weight"), &xn2, d, 1);
            let a = g.storage(ff as u64);
            ops.mm(&xn2, &format!("{pfx}.ffn.w1.weight"), &a, 1, d, ff);
            let b = g.storage(ff as u64);
            ops.mm(&xn2, &format!("{pfx}.ffn.w3.weight"), &b, 1, d, ff);
            let gg = g.storage(ff as u64);
            ops.silu_gate(&a, &b, &gg, ff);
            let ffo = g.storage(d as u64);
            ops.mm(&gg, &format!("{pfx}.ffn.w2.weight"), &ffo, 1, ff, d);
            ops.add(&ffo, &x, d); // x += ffn_out
        }

        // final norm -> context row (stashed for the dep stage) -> proj_s1
        let ctx = g.storage(d as u64);
        ops.rms(&x, "norm.weight", &ctx, d, 1);
        self.kv_append(&ctx, &cache.ctx, d, pos);
        let logits = ops.linear(&ctx, "head.proj_s1.weight", "head.proj_s1.bias", 1, d, cfg.s1_vocab());
        cache.pos += 1;
        g.read(&logits, cfg.s1_vocab())
    }

    /// **Incremental S2 decode** (dependency stage) for the just-decoded token:
    /// cross-attend `sibling(sampled_s1)` (RAW `emb_s1`, no √d) over the cached
    /// S1 context window, return `s2` logits `[s2_vocab]`. Mirrors
    /// [`crate::kvcache::HostW::dep_step`]. Call after [`Self::kv_step`].
    pub fn kv_dep_step(&self, cache: &GpuKvCache, sampled_s1: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let heads = cfg.dep_n_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let len = cache.pos;
        assert!(len > 0, "kv_dep_step: empty context");
        let w0 = len.saturating_sub(cfg.max_context);
        let pos_last = len - 1;
        let win = len - w0;
        let ops = self.ops();
        let g = &self.gpu;

        // q from the sibling embedding, RoPE'd at the last absolute position.
        let sib = Self::gather(&self.emb_s1, &[sampled_s1], d, 1.0);
        let sibd = g.storage_init("dep_sib", &sib);
        let q = ops.linear(&sibd, "dep_layer.cross_attn.q_proj.weight", "dep_layer.cross_attn.q_proj.bias", 1, d, d);
        self.rope_at_rows(&q, pos_last, 1, heads, hd);

        // k/v from the cached context window rows [w0, len); k RoPE'd at absolute
        // positions w0..len. (Read the whole cache prefix, slice the window.)
        let ctx_all = g.read(&cache.ctx, len * d);
        let ctx_win = &ctx_all[w0 * d..len * d];
        let ctxwd = g.storage_init("dep_ctx", ctx_win);
        let k = ops.linear(&ctxwd, "dep_layer.cross_attn.k_proj.weight", "dep_layer.cross_attn.k_proj.bias", win, d, d);
        let v = ops.linear(&ctxwd, "dep_layer.cross_attn.v_proj.weight", "dep_layer.cross_attn.v_proj.bias", win, d, d);
        self.rope_at_rows(&k, w0, win, heads, hd);

        // single-query non-causal cross-attention over the window (window already
        // sliced -> local w0 = 0, t = win, cap = win).
        let scores = g.storage((heads * win) as u64);
        let probs = g.storage((heads * win) as u64);
        let ctxo = g.storage(d as u64);
        self.decode_attend(&q, &k, &v, &ctxo, &scores, &probs, heads, hd, win, 0, win, scale);
        let o = ops.linear(&ctxo, "dep_layer.cross_attn.out_proj.weight", "dep_layer.cross_attn.out_proj.bias", 1, d, d);

        // norm(context[last] + attn_out) -> proj_s2
        let sumd = g.storage_init("dep_sum", &ctx_win[(win - 1) * d..win * d]);
        ops.add(&o, &sumd, d);
        let normed = g.storage(d as u64);
        ops.rms(&sumd, "dep_layer.norm.weight", &normed, d, 1);
        let logits = ops.linear(&normed, "head.proj_s2.weight", "head.proj_s2.bias", 1, d, cfg.s2_vocab());
        g.read(&logits, cfg.s2_vocab())
    }
}

/// Persistent per-layer K/V + context caches for a GPU/CPU incremental rollout,
/// produced by [`KronosDecoder::new_gpu_cache`]. `pos` is the number of tokens
/// already decoded (the next [`KronosDecoder::kv_step`] writes cache row `pos`).
pub struct GpuKvCache {
    k: Vec<DeviceBuffer>, // per layer, [cap, d] RoPE'd keys
    v: Vec<DeviceBuffer>, // per layer, [cap, d] values
    ctx: DeviceBuffer,    // [cap, d] final-norm S1 context per position
    scores: DeviceBuffer, // [n_heads, cap] scratch
    probs: DeviceBuffer,  // [n_heads, cap] scratch
    cap: usize,
    pos: usize,
}

impl GpuKvCache {
    /// Tokens decoded so far (the next `kv_step` targets cache row `pos`).
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// Reset to an empty sequence (next `kv_step` is position 0). Buffers are
    /// overwritten in place, so no reallocation is needed.
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn zero_decoder() -> KronosDecoder {
        let cfg = KronosConfig::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        KronosDecoder::from_weights_on(gpu_core::testgpu::dev(crate::nn::PIPELINES), cfg, &weights).unwrap()
    }

    #[test]
    fn decode_s1_then_s2_run_end_to_end() {
        if skip() {
            return;
        }
        let dec = zero_decoder();
        let cfg = dec.config().clone();
        let t = 6;
        let s1: Vec<u32> = (0..t).map(|i| (i as u32) % cfg.s1_vocab() as u32).collect();
        let s2: Vec<u32> = (0..t).map(|i| (i as u32 * 3) % cfg.s2_vocab() as u32).collect();
        let stamp: Vec<u32> = vec![0; t * 5]; // all-zero calendar
        let (s1_logits, ctx) = dec.decode_s1(&s1, &s2, &stamp);
        assert_eq!(s1_logits.len(), t * cfg.s1_vocab());
        // zero weights -> logits all 0
        assert!(s1_logits.iter().all(|&v| v.abs() < 1e-4), "s1 logits zero");
        let s2_logits = dec.decode_s2(&ctx, &s1);
        assert_eq!(s2_logits.len(), t * cfg.s2_vocab());
        assert!(s2_logits.iter().all(|&v| v.abs() < 1e-4), "s2 logits zero");
    }

    // -- KV-step parity ------------------------------------------------------

    /// Tiny SplitMix64 -> Box-Muller gaussians, so the test needs no dev-dep.
    struct Rng(u64);
    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f32(&mut self) -> f32 {
            ((self.u64() >> 40) as f32) / (1u64 << 24) as f32
        }
        fn gauss(&mut self) -> f32 {
            let u1 = (self.f32()).max(1e-7);
            let u2 = self.f32();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
    }

    fn random_decoder(cfg: &KronosConfig, seed: u64) -> KronosDecoder {
        let mut rng = Rng(seed);
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, shape) in cfg.param_list() {
            let n: usize = shape.iter().product();
            // RMSNorm weights ~1 (weight-only norm); everything else small gaussian.
            let v = if name.ends_with("norm.weight") || name.contains(".norm1.") || name.contains(".norm2.") {
                (0..n).map(|_| 1.0 + rng.gauss() * 0.02).collect()
            } else {
                (0..n).map(|_| rng.gauss() * 0.08).collect()
            };
            map.insert(name, v);
        }
        KronosDecoder::from_weights_on(gpu_core::testgpu::dev(crate::nn::PIPELINES), cfg.clone(), &map).unwrap()
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The GPU/CPU-portable incremental `kv_step` (S1) + `kv_dep_step` (S2) must
    /// reproduce BOTH the hand-rolled host oracle (`kvcache::HostW::step_token` /
    /// `dep_step`) AND the `O(T²)` full recompute (`decode_s1`), per token, to
    /// maxabs < 3e-3. Runs on whatever backend `Gpu` picked (GPU or
    /// `BRAIN_DEVICE=cpu`) — one op set, any device. Sequence < `max_context`, so
    /// the sliding window is inactive (w0 = 0); the windowed path is exercised by
    /// `kv_step_matches_reference_windowed`.
    #[test]
    fn kv_step_matches_reference() {
        if skip() {
            return;
        }
        let cfg = KronosConfig::tiny(); // d16 L2 heads4 hd4 ff32 s1v/s2v=16 dep_heads2 max_ctx64
        let seq = 6usize;
        assert!(seq < cfg.max_context, "keep seq < max_context so w0=0");
        let dec = random_decoder(&cfg, 0xC0FFEE);
        let (s1v, s2v) = (cfg.s1_vocab(), cfg.s2_vocab());

        let mut rng = Rng(7);
        let s1: Vec<u32> = (0..seq).map(|_| (rng.u64() % s1v as u64) as u32).collect();
        let s2: Vec<u32> = (0..seq).map(|_| (rng.u64() % s2v as u64) as u32).collect();
        // Non-trivial calendar stamps (minute,hour,weekday,day,month) within table sizes.
        let sizes = [60u64, 24, 7, 32, 13];
        let stamp: Vec<u32> = (0..seq * 5).map(|j| (rng.u64() % sizes[j % 5]) as u32).collect();

        // Host oracle rollout (the exact CPU KV reference).
        let hw = dec.host_weights();
        let mut hc = hw.new_cache();

        // GPU/CPU incremental rollout.
        let mut cache = dec.new_gpu_cache(seq);

        let (mut worst_oracle_s1, mut worst_full_s1, mut worst_s2) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..seq {
            let row = &stamp[i * 5..i * 5 + 5];
            let oracle_s1 = hw.step_token(s1[i], s2[i], row, i, &mut hc);
            let gpu_s1 = dec.kv_step(&mut cache, s1[i], s2[i], row);
            assert_eq!(gpu_s1.len(), s1v);
            worst_oracle_s1 = worst_oracle_s1.max(maxabs(&gpu_s1, &oracle_s1));

            // full recompute of the prefix; compare the last row's s1 logits.
            let (full, _ctx) = dec.decode_s1(&s1[..=i], &s2[..=i], &stamp[..(i + 1) * 5]);
            worst_full_s1 = worst_full_s1.max(maxabs(&gpu_s1, &full[i * s1v..(i + 1) * s1v]));

            // S2 dependency stage for this position (arbitrary but shared sampled s1).
            let sampled = s1[i];
            let oracle_s2 = hw.dep_step(sampled, &hc.ctx);
            let gpu_s2 = dec.kv_dep_step(&cache, sampled);
            assert_eq!(gpu_s2.len(), s2v);
            worst_s2 = worst_s2.max(maxabs(&gpu_s2, &oracle_s2));
        }
        println!("kv_step S1 vs host-oracle maxabs = {worst_oracle_s1:.3e}");
        println!("kv_step S1 vs full-recompute maxabs = {worst_full_s1:.3e}");
        println!("kv_dep_step S2 vs host-oracle maxabs = {worst_s2:.3e}");
        assert!(worst_oracle_s1 < 3e-3, "S1 vs oracle maxabs = {worst_oracle_s1}");
        assert!(worst_full_s1 < 3e-3, "S1 vs full recompute maxabs = {worst_full_s1}");
        assert!(worst_s2 < 3e-3, "S2 vs oracle maxabs = {worst_s2}");
    }

    /// Sliding-window path: a sequence LONGER than `max_context` forces
    /// `w0 = t - max_context > 0`, exercising `attn_decode_scores_win`. Validated
    /// against the host oracle, whose `step_token`/`dep_step` window identically.
    #[test]
    fn kv_step_matches_reference_windowed() {
        if skip() {
            return;
        }
        let mut cfg = KronosConfig::tiny();
        cfg.max_context = 4; // tiny window so a short seq still slides
        let seq = 9usize;
        assert!(seq > cfg.max_context, "seq must exceed max_context to slide the window");
        let dec = random_decoder(&cfg, 0xBEEF);
        let (s1v, s2v) = (cfg.s1_vocab(), cfg.s2_vocab());

        let mut rng = Rng(11);
        let s1: Vec<u32> = (0..seq).map(|_| (rng.u64() % s1v as u64) as u32).collect();
        let s2: Vec<u32> = (0..seq).map(|_| (rng.u64() % s2v as u64) as u32).collect();
        let sizes = [60u64, 24, 7, 32, 13];
        let stamp: Vec<u32> = (0..seq * 5).map(|j| (rng.u64() % sizes[j % 5]) as u32).collect();

        let hw = dec.host_weights();
        let mut hc = hw.new_cache();
        let mut cache = dec.new_gpu_cache(seq);
        let (mut worst_s1, mut worst_s2) = (0.0f32, 0.0f32);
        for i in 0..seq {
            let row = &stamp[i * 5..i * 5 + 5];
            let oracle_s1 = hw.step_token(s1[i], s2[i], row, i, &mut hc);
            let gpu_s1 = dec.kv_step(&mut cache, s1[i], s2[i], row);
            worst_s1 = worst_s1.max(maxabs(&gpu_s1, &oracle_s1));
            let sampled = s1[i];
            let oracle_s2 = hw.dep_step(sampled, &hc.ctx);
            let gpu_s2 = dec.kv_dep_step(&cache, sampled);
            worst_s2 = worst_s2.max(maxabs(&gpu_s2, &oracle_s2));
        }
        println!("windowed kv_step S1 vs oracle maxabs = {worst_s1:.3e}");
        println!("windowed kv_dep_step S2 vs oracle maxabs = {worst_s2:.3e}");
        assert!(worst_s1 < 3e-3, "windowed S1 vs oracle maxabs = {worst_s1}");
        assert!(worst_s2 < 3e-3, "windowed S2 vs oracle maxabs = {worst_s2}");
    }
}
