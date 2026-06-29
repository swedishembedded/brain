// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **MTP code-predictor**, KV-cached on the CPU — the per-frame speed
//! lever for the cached generation path.
//!
//! The engine [`crate::mtp::MtpModel`] re-runs the *whole* 5-layer decoder over
//! the growing `[hidden, cb0, cb1..k]` sequence once per residual codebook (15
//! GPU round-trips per audio frame), which dominates cached generation. This is
//! an **exact scalar+rayon mirror** of `MtpModel`'s residual generation (same
//! RMSNorm eps, half-split RoPE base θ, GQA `1/√head_dim`, SwiGLU/SiLU, the
//! per-codebook input-embedding gather and output heads), but with a per-layer
//! key/value cache so the 15 residual steps inside one frame are **incremental**
//! (`O(1)` projections + `O(t)` attention each) instead of 15 full re-forwards.
//!
//! The decoder-block arithmetic is shared with the Talker via
//! [`crate::gen_kv`]'s `decoder_layer_step` / `decoder_forward_full` (the MTP and
//! the Talker are the same Qwen3 GQA block). Only the front-end differs: the
//! input sequence is `[talker_hidden, cb0_embed, cb1_embed, …]` and each position
//! `i≥1` has its own output head `lm_head[i-1]`.

use std::collections::HashMap;

use crate::config::MtpConfig;
use crate::gen_kv::{
    decoder_forward_full, decoder_layer_step, load_layers, matvec, rmsnorm, Dims, Kv, LayerW,
};

/// A CPU-resident, KV-cached MTP code-predictor. Holds the frozen 5-layer decoder
/// weights, the residual codebook input-embedding tables, and the per-position
/// output heads, plus a per-layer growing K/V cache for the incremental path.
pub struct CpuMtp {
    pub cfg: MtpConfig,
    layers: Vec<LayerW>,
    norm: Vec<f32>,
    // incremental-decode state (one frame's 16-length sequence)
    cache: Vec<Kv>,
    pos: usize,
    // CPU front-end: residual input-embedding tables + output heads.
    codec_embedding: Vec<Vec<f32>>, // [n_residual][vocab*d]
    lm_head: Vec<Vec<f32>>,         // [n_residual][vocab*d]
}

impl CpuMtp {
    /// Decoder-block dimensions (shared GQA block with the Talker).
    fn dims(&self) -> Dims {
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

    fn d(&self) -> usize {
        self.cfg.d_model as usize
    }

    /// Build from in-memory weight parts: the decoder map (`blocks.{l}.*` +
    /// `norm.weight`), the residual input-embedding tables, and the output heads.
    /// Mirrors [`crate::mtp::MtpModel::build`]'s name/role layout.
    pub fn from_parts(
        cfg: MtpConfig,
        decoder: HashMap<String, Vec<f32>>,
        codec_embedding: Vec<Vec<f32>>,
        lm_head: Vec<Vec<f32>>,
    ) -> CpuMtp {
        let take = |n: &str| {
            decoder
                .get(n)
                .unwrap_or_else(|| panic!("CpuMtp::from_parts missing weight {n}"))
                .clone()
        };
        let layers = load_layers(cfg.n_layers, &take);
        let norm = take("norm.weight");
        let n = cfg.n_layers as usize;
        CpuMtp {
            cfg,
            layers,
            norm,
            cache: vec![Kv::default(); n],
            pos: 0,
            codec_embedding,
            lm_head,
        }
    }

    /// Load an inference-only MTP from a brain checkpoint written by
    /// [`crate::import::import_mtp`] — the same container
    /// [`crate::mtp::MtpModel::load_inference`] reads.
    pub fn load(path: &str) -> CpuMtp {
        let c = checkpoint::load(path);
        let cfg = MtpConfig::from_brain_json(&c.header["config"]);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("CpuMtp::load missing {name}"))
        };
        let layers = load_layers(cfg.n_layers, &take);
        let norm = take("norm.weight");
        let nres = cfg.n_residual() as usize;
        let codec_embedding = (0..nres)
            .map(|i| take(&format!("codec_embedding.{i}.weight")))
            .collect();
        let lm_head = (0..nres)
            .map(|i| take(&format!("lm_head.{i}.weight")))
            .collect();
        let n = cfg.n_layers as usize;
        CpuMtp {
            cfg,
            layers,
            norm,
            cache: vec![Kv::default(); n],
            pos: 0,
            codec_embedding,
            lm_head,
        }
    }

    /// Reset the per-frame K/V cache (start the next frame's 16-length sequence).
    fn reset(&mut self) {
        let n = self.layers.len();
        self.cache = vec![Kv::default(); n];
        self.pos = 0;
    }

    /// Incremental cached decode of **one** position; returns its final-norm
    /// hidden state `[d]`. Advances the position index.
    fn step(&mut self, embed: &[f32]) -> Vec<f32> {
        let dims = self.dims();
        let pos = self.pos;
        let mut x = embed.to_vec();
        for l in 0..self.layers.len() {
            x = decoder_layer_step(&self.layers[l], &mut self.cache[l], dims, &x, pos);
        }
        self.pos += 1;
        rmsnorm(&x, &self.norm)
    }

    /// `lm_head[idx]` logits (`[vocab]`) for a final-norm hidden row.
    fn head_logits(&self, idx: usize, hidden: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        matvec(&self.lm_head[idx], hidden, v, d)
    }

    /// Argmax over a logits row (ties to the lowest index), matching
    /// [`crate::mtp::MtpModel::generate_residuals`].
    fn argmax(row: &[f32]) -> usize {
        let mut best = 0usize;
        for j in 1..row.len() {
            if row[j] > row[best] {
                best = j;
            }
        }
        best
    }

    /// Per-frame residual codebook generation (greedy), **KV-cached**: bit-exact
    /// mirror of [`crate::mtp::MtpModel::generate_residuals`] but with one
    /// incremental decoder step per residual codebook instead of a full
    /// re-forward. Given the Talker final hidden state (`talker_hidden`, `[d]`)
    /// and the Talker codebook-0 embedding (`cb0_embed`, `[d]`), returns
    /// `(codes, residual_embed_sum)`:
    ///   * `codes` — the 15 residual codebook ids (codebooks 1..15),
    ///   * `residual_embed_sum` — `Σ_{i=1}^{15} codec_embedding[i-1][code_i]`.
    pub fn generate_residuals(
        &mut self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let d = self.d();
        let nres = (self.cfg.num_code_groups - 1) as usize; // 15
        assert_eq!(talker_hidden.len(), d);
        assert_eq!(cb0_embed.len(), d);

        self.reset();
        // pos 0: the Talker hidden (no head reads it).
        let _h0 = self.step(talker_hidden);

        let mut codes = vec![0u32; nres];
        let mut res_sum = vec![0.0f32; d];
        // pos k (1..=nres): input is codebook (k-1)'s embedding (pos 1 = cb0).
        let mut input = cb0_embed.to_vec();
        for k in 1..=nres {
            let hidden = self.step(&input); // final-norm hidden at position k
            let logits = self.head_logits(k - 1, &hidden);
            let best = Self::argmax(&logits);
            codes[k - 1] = best as u32;
            // codec_embedding[k-1] embeds codebook k.
            let r = &self.codec_embedding[k - 1][best * d..(best + 1) * d];
            for j in 0..d {
                res_sum[j] += r[j];
            }
            if k < nres {
                input = r.to_vec(); // next position's input embedding
            }
        }
        (codes, res_sum)
    }

    /// Residual codebook logits for an assembled `[num_code_groups, d]` input
    /// sequence, via the **uncached full recompute** — the exact mirror of
    /// [`crate::mtp::MtpModel::logits`] (used for parity testing). Returns
    /// `[(num_code_groups-1)*vocab]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn logits_full(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        let t = self.cfg.num_code_groups as usize;
        assert_eq!(inputs_embeds.len(), t * d);
        let hidden = decoder_forward_full(&self.layers, &self.norm, self.dims(), inputs_embeds);
        let mut out = vec![0.0f32; (t - 1) * v];
        for i in 1..t {
            let h = &hidden[i * d..(i + 1) * d];
            out[(i - 1) * v..i * v].copy_from_slice(&self.head_logits(i - 1, h));
        }
        out
    }

    /// Uncached autoregressive residual generation (full re-forward per step) —
    /// the `O(T²)` reference the cached [`Self::generate_residuals`] is proven
    /// equal to. Mirrors the engine's per-step re-forward, on the CPU.
    #[cfg(test)]
    fn generate_residuals_uncached(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        let t = self.cfg.num_code_groups as usize;
        let nres = t - 1;
        let mut emb = vec![0.0f32; t * d];
        emb[0..d].copy_from_slice(talker_hidden);
        emb[d..2 * d].copy_from_slice(cb0_embed);
        let mut codes = vec![0u32; nres];
        let mut res_sum = vec![0.0f32; d];
        for k in 1..=nres {
            let logits = self.logits_full(&emb);
            let row = &logits[(k - 1) * v..k * v];
            let best = Self::argmax(row);
            codes[k - 1] = best as u32;
            let r = self.codec_embedding[k - 1][best * d..(best + 1) * d].to_vec();
            for j in 0..d {
                res_sum[j] += r[j];
            }
            if k < nres {
                emb[(k + 1) * d..(k + 2) * d].copy_from_slice(&r);
            }
        }
        (codes, res_sum)
    }
}

impl crate::prompt::MtpHost for CpuMtp {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        let d = self.d();
        let s = code as usize * d;
        &self.codec_embedding[residual_idx][s..s + d]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::MtpModel;
    use data::rng::Rng;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Build random MTP weight parts (decoder map + residual embeddings + heads).
    fn synth_weights(
        cfg: &MtpConfig,
        seed: u64,
    ) -> (HashMap<String, Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = Rng::new(seed);
        let mut normal = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
        };
        let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());
        let mut decoder = HashMap::new();
        for (n, numel) in MtpModel::decoder_param_list(cfg) {
            let v = if n.ends_with("norm.weight")
                || n.ends_with("ln1.weight")
                || n.ends_with("ln2.weight")
            {
                vec![1.0; numel]
            } else if n.ends_with("attn.wo.weight") || n.ends_with("mlp.down.weight") {
                normal(numel, proj_std)
            } else {
                normal(numel, 0.02)
            };
            decoder.insert(n, v);
        }
        let nres = cfg.n_residual() as usize;
        let d = cfg.d_model as usize;
        let vv = cfg.vocab as usize;
        let codec_embedding = (0..nres).map(|_| normal(vv * d, 0.02)).collect();
        let lm_head = (0..nres).map(|_| normal(vv * d, 0.02)).collect();
        (decoder, codec_embedding, lm_head)
    }

    /// The KV-cached CpuMtp must (a) reproduce its own uncached full-recompute
    /// generation (cache exactness, pure CPU) and (b) match the WGSL engine
    /// `MtpModel::generate_residuals` on a fixed `(talker_hidden, cb0_embed)`.
    #[test]
    fn cpu_mtp_matches_engine() {
        let cfg = MtpConfig::tiny(); // num_code_groups=4 -> 3 residual codebooks
        let d = cfg.d_model as usize;
        let (decoder, codec_embedding, lm_head) = synth_weights(&cfg, 7);
        let mut cpu = CpuMtp::from_parts(
            cfg.clone(),
            decoder.clone(),
            codec_embedding.clone(),
            lm_head.clone(),
        );

        let mut rng = Rng::new(99);
        let talker_hidden: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
        let cb0_embed: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32 * 0.5).collect();

        // (a) cache exactness — cached vs uncached full-recompute, CPU only.
        let (codes_cached, res_cached) = cpu.generate_residuals(&talker_hidden, &cb0_embed);
        let (codes_unc, res_unc) = cpu.generate_residuals_uncached(&talker_hidden, &cb0_embed);
        let cache_err = maxabs(&res_cached, &res_unc);
        eprintln!("CpuMtp KV-cache: codes_eq={} res-sum max-abs={cache_err:.3e}", codes_cached == codes_unc);
        assert_eq!(codes_cached, codes_unc, "KV cache changed the residual codes");
        assert!(cache_err < 1e-4, "cache res_sum not exact vs recompute: {cache_err}");

        // (b) engine faithfulness (needs the GPU/CPU engine backend).
        if gpu_disabled() {
            return;
        }
        let gpu = MtpModel::build(cfg.clone(), decoder, codec_embedding, lm_head);

        // logits parity on a fixed assembled sequence.
        let nres_in = cfg.num_code_groups as usize - 2;
        let residual_codes: Vec<u32> =
            (0..nres_in).map(|i| (i as u32 * 3 + 1) % cfg.vocab).collect();
        let emb = gpu.assemble(&talker_hidden, &cb0_embed, &residual_codes);
        let logits_gpu = gpu.logits(&emb);
        let logits_cpu = cpu.logits_full(&emb);
        let logit_err = maxabs(&logits_cpu, &logits_gpu);

        // end-to-end residual generation parity.
        let (codes_gpu, res_gpu) = gpu.generate_residuals(&talker_hidden, &cb0_embed);
        let res_err = maxabs(&res_cached, &res_gpu);
        eprintln!(
            "CpuMtp parity: logits-vs-engine max-abs={logit_err:.3e}, \
             res-sum-vs-engine max-abs={res_err:.3e}, codes_eq={}",
            codes_cached == codes_gpu
        );
        assert!(logit_err < 1e-2, "CpuMtp logits diverge from engine: {logit_err}");
        assert_eq!(codes_cached, codes_gpu, "CpuMtp codes differ from engine");
        assert!(res_err < 1e-3, "CpuMtp res_sum diverges from engine: {res_err}");
    }
}
