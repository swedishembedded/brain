// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `KronosModel` — the tokenizer + decoder wired into an end-to-end forecast:
//! normalize → encode context bars → autoregressive token rollout (dual head:
//! sample s1, then s2 conditioned on it) → decode tokens → denormalize.
//!
//! Sampling is host-side. The default is **argmax** (deterministic — the right
//! choice for a parity harness, which compares logits); temperature / top-k /
//! top-p with a seeded RNG is available via [`GenOpts`] and replicates the
//! reference's quirk that a set `top_k` short-circuits `top_p`.

use crate::config::{KronosConfig, KronosTokenizerConfig};
use crate::decoder::KronosDecoder;
use crate::preprocess;
use crate::tokenizer::KronosTokenizer;
use gpu_core::Gpu;
use std::collections::HashMap;

/// Sampling options for the rollout.
#[derive(Clone, Debug)]
pub struct GenOpts {
    /// Softmax temperature (ignored when `argmax`).
    pub temperature: f32,
    /// Top-k filter (0 = off). A non-zero `top_k` short-circuits `top_p`
    /// (reference quirk).
    pub top_k: usize,
    /// Nucleus (top-p) filter (1.0 = off).
    pub top_p: f32,
    /// Deterministic argmax instead of sampling.
    pub argmax: bool,
    /// RNG seed for sampling.
    pub seed: u64,
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts { temperature: 1.0, top_k: 0, top_p: 1.0, argmax: true, seed: 0 }
    }
}

/// A complete Kronos model: the BSQ tokenizer + the AR decoder.
pub struct KronosModel {
    tokenizer: KronosTokenizer,
    decoder: KronosDecoder,
}

impl KronosModel {
    pub fn new(tokenizer: KronosTokenizer, decoder: KronosDecoder) -> KronosModel {
        KronosModel { tokenizer, decoder }
    }

    /// Build both nets from separate weight maps + configs.
    pub fn from_weights(
        tok_cfg: KronosTokenizerConfig,
        tok_w: &HashMap<String, Vec<f32>>,
        dec_cfg: KronosConfig,
        dec_w: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosModel, String> {
        // Tokenizer and decoder run the same kernel set, so they share ONE
        // device: two handles, one device on the card.
        Self::from_weights_on(Gpu::new(crate::nn::PIPELINES), tok_cfg, tok_w, dec_cfg, dec_w)
    }

    /// Build both nets on an existing device handle — one device per process,
    /// however many models it loads.
    pub fn from_weights_on(
        gpu: Gpu,
        tok_cfg: KronosTokenizerConfig,
        tok_w: &HashMap<String, Vec<f32>>,
        dec_cfg: KronosConfig,
        dec_w: &HashMap<String, Vec<f32>>,
    ) -> Result<KronosModel, String> {
        // share_or_new: on a backend without a shareable device (native Vulkan
        // today) this builds fresh instead of panicking.
        let tok_gpu = gpu.share_or_new(crate::nn::PIPELINES);
        Ok(KronosModel {
            tokenizer: KronosTokenizer::from_weights_on(tok_gpu, tok_cfg, tok_w)?,
            decoder: KronosDecoder::from_weights_on(gpu, dec_cfg, dec_w)?,
        })
    }

    pub fn feat(&self) -> usize {
        self.tokenizer.config().d_in
    }
    pub fn max_context(&self) -> usize {
        self.decoder.config().max_context
    }
    /// The decoder config (for building a trainable twin).
    pub fn decoder_config(&self) -> &KronosConfig {
        self.decoder.config()
    }
    /// The decoder (host embedding + s1/s2 cores) — used by the NPU AR-loop driver.
    pub fn decoder(&self) -> &KronosDecoder {
        &self.decoder
    }
    /// The frozen BSQ tokenizer — decode generated `(s1, s2)` tails back to bars.
    pub fn tokenizer(&self) -> &KronosTokenizer {
        &self.tokenizer
    }
    /// Frozen-tokenizer path: per-feature-normalize `bars` `[t, feat]` (past-only,
    /// clip ±5, the inference contract) then BSQ-encode → `(s1, s2)` token streams
    /// `[t]`. Used to build fine-tuning batches without touching the tokenizer.
    pub fn tokenize(&self, bars: &[f32], t: usize) -> (Vec<u32>, Vec<u32>) {
        let (_norm, x) = preprocess::normalize(bars, t, self.feat(), 5.0);
        self.tokenizer.encode(&x, t)
    }

    /// Forecast `pred_len` future bars from `bars` `[T, feat]` (row-major
    /// OHLCV(+amount)). `ctx_stamp` is `[T, 5]` and `fut_stamp` is
    /// `[pred_len, 5]` calendar indices. Returns `[pred_len, feat]`.
    pub fn forecast(
        &self,
        bars: &[f32],
        ctx_stamp: &[u32],
        fut_stamp: &[u32],
        pred_len: usize,
        opts: &GenOpts,
    ) -> Vec<f32> {
        let feat = self.feat();
        let t = bars.len() / feat;
        let max_ctx = self.max_context();

        // 1. normalize the context (per feature, clip ±5)
        let (norm, x) = preprocess::normalize(bars, t, feat, 5.0);

        // 2. encode context bars -> token streams
        let (mut s1, mut s2) = self.tokenizer.encode(&x, t);

        // full calendar stream = ctx ++ future
        let mut stamp = ctx_stamp.to_vec();
        stamp.extend_from_slice(fut_stamp);

        // 3. autoregressive rollout
        let mut rng = SplitMix64::new(opts.seed);
        for step in 0..pred_len {
            let len = s1.len();
            let w0 = len.saturating_sub(max_ctx);
            let s1w = &s1[w0..];
            let s2w = &s2[w0..];
            // calendar for this window: bars w0..len map into `stamp`
            let stw: Vec<u32> = (w0..len).flat_map(|i| stamp[i * 5..i * 5 + 5].to_vec()).collect();

            let (s1_logits, ctx) = self.decoder.decode_s1(s1w, s2w, &stw);
            let ww = s1w.len();
            let vs1 = self.decoder.config().s1_vocab();
            let last_s1 = &s1_logits[(ww - 1) * vs1..ww * vs1];
            let samp_s1 = sample(last_s1, opts, &mut rng);

            // s2 conditioned on the just-sampled s1 (at the last position)
            let mut s1_cond: Vec<u32> = s1w.to_vec();
            *s1_cond.last_mut().unwrap() = samp_s1;
            let s2_logits = self.decoder.decode_s2(&ctx, &s1_cond);
            let vs2 = self.decoder.config().s2_vocab();
            let last_s2 = &s2_logits[(ww - 1) * vs2..ww * vs2];
            let samp_s2 = sample(last_s2, opts, &mut rng);

            s1.push(samp_s1);
            s2.push(samp_s2);
            let _ = step;
        }

        // 4. decode the generated tail + denormalize
        let gen1 = &s1[t..];
        let gen2 = &s2[t..];
        let recon = self.tokenizer.decode(gen1, gen2); // [pred_len, feat] (normalized)
        preprocess::denormalize(&norm, &recon, pred_len, feat)
    }

    /// KV-cached forecast — the fast path. Identical result to [`forecast`] (up to
    /// float noise) but prefills the context once and advances one token at a time
    /// over a per-layer K/V cache, turning the `O(T²)`-per-step rollout into an
    /// `O(T²)` prefill + `O(T)`-per-step tail. See [`crate::kvcache`].
    pub fn forecast_cached(
        &self,
        bars: &[f32],
        ctx_stamp: &[u32],
        fut_stamp: &[u32],
        pred_len: usize,
        opts: &GenOpts,
    ) -> Vec<f32> {
        let feat = self.feat();
        let t = bars.len() / feat;
        let (norm, x) = preprocess::normalize(bars, t, feat, 5.0);
        let (mut s1, mut s2) = self.tokenizer.encode(&x, t);
        let mut stamp = ctx_stamp.to_vec();
        stamp.extend_from_slice(fut_stamp);

        let hw = self.decoder.host_weights();
        let mut cache = hw.new_cache();
        let mut rng = SplitMix64::new(opts.seed);

        // prefill the context (absolute positions 0..t); attention self-windows to
        // max_context, so this matches the reference's initial window.
        let mut last_s1_logits = Vec::new();
        for p in 0..t {
            last_s1_logits = hw.step_token(s1[p], s2[p], &stamp[p * 5..p * 5 + 5], p, &mut cache);
        }

        for step in 0..pred_len {
            let samp_s1 = sample(&last_s1_logits, opts, &mut rng);
            let s2_logits = hw.dep_step(samp_s1, &cache.ctx);
            let samp_s2 = sample(&s2_logits, opts, &mut rng);
            s1.push(samp_s1);
            s2.push(samp_s2);
            let ppos = t + step;
            last_s1_logits = hw.step_token(samp_s1, samp_s2, &stamp[ppos * 5..ppos * 5 + 5], ppos, &mut cache);
        }

        let recon = self.tokenizer.decode(&s1[t..], &s2[t..]);
        preprocess::denormalize(&norm, &recon, pred_len, feat)
    }

    /// Forecast with the transformer **cores injected** — the seam the NPU path
    /// uses. Same host pipeline as [`forecast`] (normalize → encode context →
    /// AR rollout with per-window calendar → decode → denormalize), but the
    /// post-embedding s1/s2 cores are supplied by the caller and the attention
    /// window is held at a **fixed `win`** (the compiled graph's `T`) rather than
    /// growing to `max_context`. That fixed-shape requirement means the result
    /// tracks but does not bit-match [`forecast`] once the rollout slides the
    /// window. `s1_core(x[win*D]) -> (ctx[win*D], s1_logits[win*s1_vocab])`;
    /// `s2_core(ctx[win*D], sib[win*D]) -> s2_logits[win*s2_vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forecast_with_cores<F1, F2>(
        &self,
        bars: &[f32],
        ctx_stamp: &[u32],
        fut_stamp: &[u32],
        pred_len: usize,
        opts: &GenOpts,
        win: usize,
        mut s1_core: F1,
        mut s2_core: F2,
    ) -> Vec<f32>
    where
        F1: FnMut(&[f32]) -> (Vec<f32>, Vec<f32>),
        F2: FnMut(&[f32], &[f32]) -> Vec<f32>,
    {
        let feat = self.feat();
        let t = bars.len() / feat;
        let (norm, x) = preprocess::normalize(bars, t, feat, 5.0);
        let (mut s1, mut s2) = self.tokenizer.encode(&x, t);
        let mut stamp = ctx_stamp.to_vec();
        stamp.extend_from_slice(fut_stamp);
        let dec = &self.decoder;
        let (vs1, vs2) = (dec.config().s1_vocab(), dec.config().s2_vocab());
        let mut rng = SplitMix64::new(opts.seed);
        for _ in 0..pred_len {
            let len = s1.len();
            let w0 = len.saturating_sub(win);
            let (s1w, s2w) = (&s1[w0..], &s2[w0..]);
            let ww = s1w.len();
            let stw: Vec<u32> = (w0..len).flat_map(|i| stamp[i * 5..i * 5 + 5].to_vec()).collect();
            let x_emb = dec.embed_tokens(s1w, s2w, &stw);
            let (ctx, s1_logits) = s1_core(&x_emb);
            let samp_s1 = sample(&s1_logits[(ww - 1) * vs1..ww * vs1], opts, &mut rng);
            let mut s1_cond = s1w.to_vec();
            *s1_cond.last_mut().unwrap() = samp_s1;
            let sib = dec.sib_embed(&s1_cond);
            let s2_logits = s2_core(&ctx, &sib);
            let samp_s2 = sample(&s2_logits[(ww - 1) * vs2..ww * vs2], opts, &mut rng);
            s1.push(samp_s1);
            s2.push(samp_s2);
        }
        let recon = self.tokenizer.decode(&s1[t..], &s2[t..]);
        preprocess::denormalize(&norm, &recon, pred_len, feat)
    }
}

/// Sample a token id from logits per [`GenOpts`].
fn sample(logits: &[f32], opts: &GenOpts, rng: &mut SplitMix64) -> u32 {
    if opts.argmax {
        return argmax(logits);
    }
    // temperature
    let temp = opts.temperature.max(1e-6);
    let mut lg: Vec<f32> = logits.iter().map(|&v| v / temp).collect();
    // top-k short-circuits top-p (reference quirk)
    if opts.top_k > 0 {
        keep_top_k(&mut lg, opts.top_k);
    } else if opts.top_p < 1.0 {
        keep_top_p(&mut lg, opts.top_p);
    }
    // softmax + multinomial
    let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = lg.iter().map(|&v| if v.is_finite() { (v - mx).exp() } else { 0.0 }).collect();
    let sum: f32 = exps.iter().sum();
    let mut u = rng.next_f32() * sum;
    for (i, &e) in exps.iter().enumerate() {
        u -= e;
        if u <= 0.0 {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

fn argmax(x: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn keep_top_k(lg: &mut [f32], k: usize) {
    if k >= lg.len() {
        return;
    }
    let mut sorted: Vec<f32> = lg.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let thresh = sorted[k - 1];
    for v in lg.iter_mut() {
        if *v < thresh {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn keep_top_p(lg: &mut [f32], p: f32) {
    // softmax to probs, sort desc, keep the smallest prefix with cumsum > p.
    let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = lg.iter().map(|&v| (v - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut idx: Vec<usize> = (0..lg.len()).collect();
    idx.sort_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut keep = vec![false; lg.len()];
    for &i in &idx {
        keep[i] = true;
        cum += exps[i] / sum;
        if cum > p {
            break;
        }
    }
    for i in 0..lg.len() {
        if !keep[i] {
            lg[i] = f32::NEG_INFINITY;
        }
    }
}

/// SplitMix64 for reproducible sampling.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn zero_model() -> KronosModel {
        let tc = KronosTokenizerConfig::tiny();
        let dc = KronosConfig::tiny();
        let tw: HashMap<String, Vec<f32>> =
            tc.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        let dw: HashMap<String, Vec<f32>> =
            dc.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        KronosModel::from_weights_on(gpu_core::testgpu::dev(crate::nn::PIPELINES), tc, &tw, dc, &dw).unwrap()
    }

    #[test]
    fn forecast_runs_end_to_end_and_is_deterministic() {
        if skip() {
            return;
        }
        let m = zero_model();
        let feat = m.feat();
        let t = 20;
        let bars: Vec<f32> = (0..t * feat).map(|i| (i as f32).sin()).collect();
        let ctx_stamp = vec![0u32; t * 5];
        let pred_len = 5;
        let fut_stamp = vec![0u32; pred_len * 5];
        let out = m.forecast(&bars, &ctx_stamp, &fut_stamp, pred_len, &GenOpts::default());
        assert_eq!(out.len(), pred_len * feat);
        assert!(out.iter().all(|v| v.is_finite()));
        // argmax rollout is deterministic
        let out2 = m.forecast(&bars, &ctx_stamp, &fut_stamp, pred_len, &GenOpts::default());
        assert_eq!(out, out2);
    }

    #[test]
    fn cached_forecast_runs_and_is_deterministic() {
        if skip() {
            return;
        }
        let m = zero_model();
        let feat = m.feat();
        let t = 20;
        let bars: Vec<f32> = (0..t * feat).map(|i| (i as f32).sin()).collect();
        let ctx_stamp = vec![0u32; t * 5];
        let pred_len = 5;
        let fut_stamp = vec![0u32; pred_len * 5];
        let out = m.forecast_cached(&bars, &ctx_stamp, &fut_stamp, pred_len, &GenOpts::default());
        assert_eq!(out.len(), pred_len * feat);
        assert!(out.iter().all(|v| v.is_finite()));
        let out2 = m.forecast_cached(&bars, &ctx_stamp, &fut_stamp, pred_len, &GenOpts::default());
        assert_eq!(out, out2);
    }

    #[test]
    fn top_k_short_circuits_top_p() {
        // with top_k set AND top_p set, only top_k is applied (reference quirk).
        let mut lg = vec![1.0, 2.0, 3.0, 4.0];
        let mut rng = SplitMix64::new(0);
        let opts = GenOpts { argmax: false, top_k: 1, top_p: 0.1, temperature: 1.0, seed: 0 };
        // top_k=1 keeps only index 3 -> always sampled
        for _ in 0..8 {
            let mut l = lg.clone();
            assert_eq!(sample(&mut l, &opts, &mut rng), 3);
        }
        let _ = &mut lg;
    }
}
