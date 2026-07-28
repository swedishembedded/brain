// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The BSQ tokenizer forward: `encode` (OHLCV bars → hierarchical `(s1, s2)`
//! tokens) and `decode` (tokens → reconstructed bars). Causal transformer
//! encoder/decoder around the parameter-free BSQ bottleneck.

use crate::config::KronosTokenizerConfig;
use crate::nn::{self, Ops, BSQ_QUANTIZE};
use crate::preprocess;
use gpu_core::{f, DeviceBuffer, Gpu};
use std::collections::HashMap;

/// A loaded Kronos BSQ tokenizer.
pub struct KronosTokenizer {
    gpu: Gpu,
    cfg: KronosTokenizerConfig,
    w: HashMap<String, DeviceBuffer>,
}

impl KronosTokenizer {
    pub fn from_weights(
        cfg: KronosTokenizerConfig,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosTokenizer, String> {
        Self::from_weights_on(Gpu::new(nn::PIPELINES), cfg, weights)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads. One device per
    /// component is both slow (a full device init each) and hazardous — many
    /// concurrent devices on one card deadlocked the test suite.
    pub fn from_weights_on(
        gpu: Gpu,
        cfg: KronosTokenizerConfig,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosTokenizer, String> {
        let w = nn::load_weights(&gpu, &cfg.param_list(), weights)?;
        Ok(KronosTokenizer { gpu, cfg, w })
    }

    pub fn config(&self) -> &KronosTokenizerConfig {
        &self.cfg
    }

    fn ops(&self) -> Ops<'_> {
        Ops { gpu: &self.gpu, w: &self.w, rope_theta: 10000.0 }
    }

    /// Encode normalized OHLCV(+amount) bars `[T, d_in]` into `(s1, s2)` token
    /// streams, each length `T`.
    pub fn encode(&self, bars: &[f32], t: usize) -> (Vec<u32>, Vec<u32>) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let k = cfg.codebook_dim();
        let ops = self.ops();

        let x = self.gpu.storage_init("bars", bars); // [T, d_in]
        // embed: d_in -> d_model
        let z = ops.linear(&x, "embed.weight", "embed.bias", t, cfg.d_in, d);
        // causal encoder blocks
        for i in 0..cfg.enc_blocks() {
            ops.transformer_block(&format!("encoder.{i}"), &z, t, d, cfg.ff_dim, cfg.n_heads);
        }
        // quant_embed: d_model -> k, then BSQ sign-quantize
        let q = ops.linear(&z, "quant_embed.weight", "quant_embed.bias", t, d, k);
        let inv_sqrt_k = 1.0 / (k as f32).sqrt();
        let bsq = self.gpu.step(BSQ_QUANTIZE, &[&q], &[(t * k) as u32, f(inv_sqrt_k)], (t * k) as u32);
        self.gpu.submit(&[], &[bsq]);
        let zq = self.gpu.read(&q, t * k);
        preprocess::quantized_to_indices(&zq, t, cfg.s1_bits, cfg.s2_bits)
    }

    /// Decode `(s1, s2)` token streams back into reconstructed bars `[T, d_in]`.
    pub fn decode(&self, s1: &[u32], s2: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let k = cfg.codebook_dim();
        let t = s1.len();
        let ops = self.ops();

        // tokens -> bipolar [T, k]
        let bip = preprocess::indices_to_bipolar(s1, s2, cfg.s1_bits, cfg.s2_bits);
        let bipd = self.gpu.storage_init("bip", &bip);
        // post_quant_embed: k -> d_model
        let z = ops.linear(&bipd, "post_quant_embed.weight", "post_quant_embed.bias", t, k, d);
        // causal decoder blocks
        for i in 0..cfg.dec_blocks() {
            ops.transformer_block(&format!("decoder.{i}"), &z, t, d, cfg.ff_dim, cfg.n_heads);
        }
        // head: d_model -> d_in
        let out = ops.linear(&z, "head.weight", "head.bias", t, d, cfg.d_in);
        self.gpu.read(&out, t * cfg.d_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn zero_tokenizer() -> KronosTokenizer {
        let cfg = KronosTokenizerConfig::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        KronosTokenizer::from_weights_on(gpu_core::testgpu::dev(crate::nn::PIPELINES), cfg, &weights).unwrap()
    }

    #[test]
    fn encode_decode_run_end_to_end_with_zero_weights() {
        if skip() {
            return;
        }
        // With zero weights: embed->0, encoder(0)->0, quant_embed->0, and
        // bsq_quantize(0) = sign(0)= -1 (z>0 is false) -> all bits 0 -> s1=s2=0.
        // decode(0,0): bipolar all -1/√k, post_quant_embed(zero W)->0, blocks->0,
        // head->0. So tokens are all 0 and the reconstruction is all 0 — the
        // whole pipeline runs and shapes are correct.
        let tok = zero_tokenizer();
        let t = 8;
        let feat = tok.config().d_in;
        let bars: Vec<f32> = (0..t * feat).map(|i| (i as f32) * 0.01).collect();
        let (s1, s2) = tok.encode(&bars, t);
        assert_eq!(s1.len(), t);
        assert_eq!(s2.len(), t);
        assert!(s1.iter().all(|&x| x == 0) && s2.iter().all(|&x| x == 0), "zero-weight tokens are 0");
        let recon = tok.decode(&s1, &s2);
        assert_eq!(recon.len(), t * feat);
        assert!(recon.iter().all(|&v| v.abs() < 1e-4), "zero-weight reconstruction is 0");
    }

    #[test]
    fn encode_is_deterministic() {
        if skip() {
            return;
        }
        let tok = zero_tokenizer();
        let t = 5;
        let feat = tok.config().d_in;
        let bars: Vec<f32> = (0..t * feat).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
        let a = tok.encode(&bars, t);
        let b = tok.encode(&bars, t);
        assert_eq!(a, b);
    }
}
