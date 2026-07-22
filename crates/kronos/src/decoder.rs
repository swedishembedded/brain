// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The autoregressive decoder forward: `decode_s1` (tokens+calendar → s1 logits
//! + context) and `decode_s2` (context + sampled s1 → s2 logits via the
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
use std::cell::OnceCell;
use std::collections::HashMap;

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
    host_w: OnceCell<crate::kvcache::HostW>,
}

impl KronosDecoder {
    pub fn from_weights(
        cfg: KronosConfig,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosDecoder, String> {
        let gpu = Gpu::new(nn::PIPELINES);
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
            host_w: OnceCell::new(),
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
    fn dep_forward(&self, context: &DeviceBuffer, sibd: &DeviceBuffer, t: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let heads = cfg.dep_n_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let ops = self.ops();

        // cross-attention: q from sibling, k/v from context (non-causal, scaled)
        let q = ops.linear(&sibd, "dep_layer.cross_attn.q_proj.weight", "dep_layer.cross_attn.q_proj.bias", t, d, d);
        let k = ops.linear(context, "dep_layer.cross_attn.k_proj.weight", "dep_layer.cross_attn.k_proj.bias", t, d, d);
        let v = ops.linear(context, "dep_layer.cross_attn.v_proj.weight", "dep_layer.cross_attn.v_proj.bias", t, d, d);
        ops.rope(&q, t, heads, hd);
        ops.rope(&k, t, heads, hd);
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
        if self.host_w.get().is_none() {
            let hw = self.build_host_weights();
            let _ = self.host_w.set(hw);
        }
        self.host_w.get().expect("host_w initialized")
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
        KronosDecoder::from_weights(cfg, &weights).unwrap()
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
}
