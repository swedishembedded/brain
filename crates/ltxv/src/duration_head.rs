// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's real-weight **duration-prediction head** - a small regression
//! head predicting shot duration (in seconds) from the video/audio
//! embeddings-Connector's pooled/aggregated hidden states. Ported from
//! `ltx_core.duration_head.{duration_head,model_configurator}`
//! (`scratchpad/reference/ltxv/packages/ltx-core/src/ltx_core/duration_head/`),
//! real weights (`ltx-2.5-duration-head-bf16.safetensors`, 15 tensors,
//! ~4MB), real parity (`crates/ltxv/tests/duration_head_parity.rs`).
//!
//! # Eager host math, not a device graph
//!
//! Every tensor here is tiny - `pooler_hidden_dim=256`, a handful of tokens,
//! `num_queries=1` - the same "not worth a lazy graph" call
//! `crates/ltxv/src/audio_vae.rs`'s module doc already made for its own
//! small per-call tensors. [`DurationHead::forward`] is plain `f32` host
//! arithmetic; there is no `Gpu`/`Builder` anywhere in this module.
//!
//! # Op sequence (`DurationHead.forward`, `AttentionPooler.forward`)
//!
//! ```text
//! video_tok  = video_input_proj(video_tokens) + video_modality_emb   # per token
//! audio_tok  = audio_input_proj(audio_tokens) + audio_modality_emb   # per token
//! tokens     = cat([video_tok, audio_tok], dim=token)
//! pooled     = MultiheadAttention(query_tokens, tokens, tokens)      # see below
//! hidden     = gelu_tanh(mlp_hidden(pooled.flatten()))
//! duration   = exp(mlp_out(hidden))                                  # seconds
//! ```
//! Either modality may be absent (`forward` requires at least one); `tokens`
//! is then just the present modality's own projected+tagged sequence.
//!
//! # `nn.MultiheadAttention`'s exact decomposition - pinned against source,
//! # not the PyTorch docstring alone
//!
//! `AttentionPooler.cross_attn` is `torch.nn.MultiheadAttention(embed_dim=256,
//! num_heads=4, batch_first=True)`, an opaque fused op on the reference side.
//! `tools/goldens/ltxv_duration_head_dump_reference.py` computes the SAME
//! pooled output a second, independent way (`manual_mha`: unpack
//! `in_proj_weight`/`in_proj_bias` into `[Wq;Wk;Wv]`/`[bq;bk;bv]` in THAT
//! row order, split into `num_heads` heads of `hidden_dim/num_heads` each,
//! scaled dot-product attention with `scale = 1/sqrt(head_dim)` and softmax
//! over the KEY axis, then `out_proj`) and asserts the two agree at cosine
//! `>= 1 - 1e-6` - that manual decomposition IS this module's
//! [`DurationHead::forward`], transcribed to Rust unchanged. `query_tokens`
//! (`[num_queries, hidden_dim]`) is the attention's Q input, expanded over
//! the batch; every position of `tokens` (the K/V input) is attendable -
//! there is no mask (see the reference module's own docstring: the upstream
//! Connector always substitutes learnable registers for padded positions, so
//! by the time tokens reach this head every position is already valid - this
//! port never receives padding to mask in the first place, since the
//! synthetic golden's inputs are always fully "real" tokens).
//!
//! `pooled.reshape(B,-1)` (`num_queries` folded into the feature axis before
//! `mlp_hidden`) generalises the real config's `num_queries=1` case, where it
//! is a no-op reshape - implemented in general for `num_queries` rather than
//! hardcoding 1, since [`DurationHeadConfig::tensor_manifest`] already
//! derives `mlp_hidden.weight`'s input width as `hidden_dim * num_queries`.
//!
//! `gelu_tanh` is the `approximate="tanh"` GELU
//! (`0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))`), not the exact erf form - `F.gelu(...,
//! approximate="tanh")` in `duration_head.py`.

use vae::blocks::Tensors;

/// Real LTX-2.5 `DurationHead` config - `video_cross_attention_dim`/
/// `audio_cross_attention_dim` mirror the main transformer's own
/// `cross_attention_dim`/`audio_cross_attention_dim` (4096/2048); the
/// pooler's own hyperparameters (`pooler_hidden_dim`, `num_queries`,
/// `num_pooler_heads`, `mlp_hidden`) come from a `duration_head` sub-dict
/// that is EMPTY in the real checkpoint's metadata, so every one of them is
/// `DurationHeadConfigurator.from_metadata`'s own JAX-matching default -
/// confirmed against the real header (`in_proj_weight` is `[768,256]` =
/// `3*256`, `query_tokens` is `[1,256]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DurationHeadConfig {
    pub video_dim: u32,
    pub audio_dim: u32,
    pub hidden: u32,
    pub num_queries: u32,
    pub num_heads: u32,
    pub mlp_hidden: u32,
}

impl Default for DurationHeadConfig {
    fn default() -> Self {
        Self::ltx25()
    }
}

impl DurationHeadConfig {
    pub fn ltx25() -> DurationHeadConfig {
        DurationHeadConfig { video_dim: 4096, audio_dim: 2048, hidden: 256, num_queries: 1, num_heads: 4, mlp_hidden: 256 }
    }

    /// Every tensor this model reads, in the checkpoint's `duration_head.`
    /// prefix STRIPPED name space (see [`crate::import::import_duration_head`])
    /// - 15 tensors, cross-checked against the real header.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (h, vd, ad) = (self.hidden as usize, self.video_dim as usize, self.audio_dim as usize);
        let (nq, mh) = (self.num_queries as usize, self.mlp_hidden as usize);
        vec![
            ("video_input_proj.weight".into(), vec![h, vd]),
            ("video_input_proj.bias".into(), vec![h]),
            ("video_modality_emb".into(), vec![h]),
            ("audio_input_proj.weight".into(), vec![h, ad]),
            ("audio_input_proj.bias".into(), vec![h]),
            ("audio_modality_emb".into(), vec![h]),
            ("attention_pooler.query_tokens".into(), vec![nq, h]),
            ("attention_pooler.cross_attn.in_proj_weight".into(), vec![3 * h, h]),
            ("attention_pooler.cross_attn.in_proj_bias".into(), vec![3 * h]),
            ("attention_pooler.cross_attn.out_proj.weight".into(), vec![h, h]),
            ("attention_pooler.cross_attn.out_proj.bias".into(), vec![h]),
            ("mlp_hidden.weight".into(), vec![mh, h * nq]),
            ("mlp_hidden.bias".into(), vec![mh]),
            ("mlp_out.weight".into(), vec![1, mh]),
            ("mlp_out.bias".into(), vec![1]),
        ]
    }
}

/// `y[i,:] = x[i,:] @ w^T + b`, `x: [rows, in_dim]`, `w: [out_dim, in_dim]`
/// (torch's own `nn.Linear.weight` layout), `b: [out_dim]`.
fn linear(x: &[f32], rows: usize, in_dim: usize, out_dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(b.len(), out_dim);
    let mut y = vec![0.0f32; rows * out_dim];
    for i in 0..rows {
        for o in 0..out_dim {
            let mut acc = b[o];
            let wrow = &w[o * in_dim..(o + 1) * in_dim];
            let xrow = &x[i * in_dim..(i + 1) * in_dim];
            for k in 0..in_dim {
                acc += xrow[k] * wrow[k];
            }
            y[i * out_dim + o] = acc;
        }
    }
    y
}

/// `F.gelu(x, approximate="tanh")`.
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// The duration head, over an already-imported [`Tensors`] map (see
/// [`crate::import::import_duration_head`]).
pub struct DurationHead<'a> {
    cfg: DurationHeadConfig,
    t: &'a Tensors,
}

impl<'a> DurationHead<'a> {
    pub fn new(cfg: DurationHeadConfig, t: &'a Tensors) -> DurationHead<'a> {
        DurationHead { cfg, t }
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.t.get(name).unwrap_or_else(|| panic!("ltxv duration head: missing tensor {name}")).1
    }

    /// Project + tag one modality's `[tokens, dim]` sequence with its input
    /// projection and modality embedding - `video_input_proj(x) +
    /// video_modality_emb` (or the audio twin).
    fn project_modality(&self, prefix: &str, tokens: &[f32], n: usize, dim: u32) -> Vec<f32> {
        let h = self.cfg.hidden as usize;
        let w = self.get(&format!("{prefix}_input_proj.weight"));
        let b = self.get(&format!("{prefix}_input_proj.bias"));
        let mut y = linear(tokens, n, dim as usize, h, w, b);
        let emb = self.get(&format!("{prefix}_modality_emb"));
        for row in y.chunks_mut(h) {
            for (v, e) in row.iter_mut().zip(emb) {
                *v += e;
            }
        }
        y
    }

    /// `AttentionPooler.forward`: `num_queries` learnable queries cross-attend
    /// `tokens` (`[n_tokens, hidden]`), `num_heads` heads. Returns
    /// `[num_queries, hidden]`. See this module's doc for the exact
    /// decomposition (pinned against the golden's own independent
    /// `manual_mha`).
    fn attention_pool(&self, tokens: &[f32], n_tokens: usize) -> Vec<f32> {
        let h = self.cfg.hidden as usize;
        let nq = self.cfg.num_queries as usize;
        let heads = self.cfg.num_heads as usize;
        let hd = h / heads;
        assert_eq!(hd * heads, h, "hidden {h} not divisible by num_heads {heads}");

        let query = self.get("attention_pooler.query_tokens");
        let in_w = self.get("attention_pooler.cross_attn.in_proj_weight");
        let in_b = self.get("attention_pooler.cross_attn.in_proj_bias");
        let out_w = self.get("attention_pooler.cross_attn.out_proj.weight");
        let out_b = self.get("attention_pooler.cross_attn.out_proj.bias");

        // `in_proj_weight`/`in_proj_bias` pack `[Wq;Wk;Wv]`/`[bq;bk;bv]` in
        // THAT row order (torch's `nn.MultiheadAttention` convention, pinned
        // by the golden's `manual_mha` self-validation).
        let (wq, wk, wv) = (&in_w[0..h * h], &in_w[h * h..2 * h * h], &in_w[2 * h * h..3 * h * h]);
        let (bq, bk, bv) = (&in_b[0..h], &in_b[h..2 * h], &in_b[2 * h..3 * h]);

        let q = linear(query, nq, h, h, wq, bq); // [nq, h]
        let k = linear(tokens, n_tokens, h, h, wk, bk); // [n_tokens, h]
        let v = linear(tokens, n_tokens, h, h, wv, bv); // [n_tokens, h]

        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; nq * h];
        for head in 0..heads {
            let off = head * hd;
            for qi in 0..nq {
                let mut scores = vec![0.0f32; n_tokens];
                for (ti, s) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc += q[qi * h + off + d] * k[ti * h + off + d];
                    }
                    *s = acc * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for (ti, s) in scores.iter().enumerate() {
                        acc += s * v[ti * h + off + d];
                    }
                    ctx[qi * h + off + d] = acc;
                }
            }
        }
        linear(&ctx, nq, h, h, out_w, out_b)
    }

    /// Predict duration in seconds from either or both modalities' pooled
    /// token sequences (`[tokens, dim]`, row-major). At least one must be
    /// present. Returns a single scalar (this port only ever handles one
    /// instance at a time, matching every other `crate::*` forward's `B=1`
    /// convention).
    pub fn forward(&self, video_tokens: Option<&[f32]>, audio_tokens: Option<&[f32]>) -> f32 {
        self.forward_taps(video_tokens, audio_tokens).duration
    }

    /// [`DurationHead::forward`], with every intermediate stage kept - for
    /// parity testing against `tools/goldens/ltxv_duration_head_dump_reference.py`'s
    /// own per-stage taps (`video_proj`/`audio_proj`/`tokens`/`pooled`/
    /// `hidden`/`log_duration`/`duration`), mirroring the tap-readback shape
    /// every device-graph module in this crate (`LtxVaeEncoder::read_stage`,
    /// [`crate::upsampler::LatentUpsampler::read_tap`]) exposes - this module
    /// has no device graph, so the taps are just the plain `Vec<f32>`s host
    /// math already produced, not a device readback.
    pub fn forward_taps(&self, video_tokens: Option<&[f32]>, audio_tokens: Option<&[f32]>) -> DurationHeadTaps {
        assert!(video_tokens.is_some() || audio_tokens.is_some(), "duration head needs at least one modality");
        let h = self.cfg.hidden as usize;

        let video_proj = video_tokens.map(|v| self.project_modality("video", v, v.len() / self.cfg.video_dim as usize, self.cfg.video_dim));
        let audio_proj = audio_tokens.map(|a| self.project_modality("audio", a, a.len() / self.cfg.audio_dim as usize, self.cfg.audio_dim));

        let mut tokens: Vec<f32> = Vec::new();
        let mut n_tokens = 0usize;
        if let Some(v) = &video_proj {
            tokens.extend_from_slice(v);
            n_tokens += v.len() / h;
        }
        if let Some(a) = &audio_proj {
            tokens.extend_from_slice(a);
            n_tokens += a.len() / h;
        }

        let pooled = self.attention_pool(&tokens, n_tokens); // [num_queries, h]

        let nq = self.cfg.num_queries as usize;
        let mh = self.cfg.mlp_hidden as usize;
        let mlp_w = self.get("mlp_hidden.weight");
        let mlp_b = self.get("mlp_hidden.bias");
        let hidden: Vec<f32> = linear(&pooled, 1, h * nq, mh, mlp_w, mlp_b).into_iter().map(gelu_tanh).collect();

        let out_w = self.get("mlp_out.weight");
        let out_b = self.get("mlp_out.bias");
        let log_duration = linear(&hidden, 1, mh, 1, out_w, out_b)[0];

        DurationHeadTaps { video_proj, audio_proj, tokens, pooled, hidden, log_duration, duration: log_duration.exp() }
    }
}

/// Every intermediate of one [`DurationHead::forward_taps`] call - see that
/// method's doc.
pub struct DurationHeadTaps {
    pub video_proj: Option<Vec<f32>>,
    pub audio_proj: Option<Vec<f32>>,
    pub tokens: Vec<f32>,
    pub pooled: Vec<f32>,
    pub hidden: Vec<f32>,
    pub log_duration: f32,
    pub duration: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        let m = DurationHeadConfig::ltx25().tensor_manifest();
        assert_eq!(m.len(), 15, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("attention_pooler.cross_attn.in_proj_weight"), vec![768, 256]);
        assert_eq!(get("attention_pooler.query_tokens"), vec![1, 256]);
        assert_eq!(get("video_input_proj.weight"), vec![256, 4096]);
        assert_eq!(get("audio_input_proj.weight"), vec![256, 2048]);
        assert_eq!(get("mlp_hidden.weight"), vec![256, 256]);
        assert_eq!(get("mlp_out.weight"), vec![1, 256]);
    }

    #[test]
    fn gelu_tanh_matches_known_points() {
        assert!((gelu_tanh(0.0)).abs() < 1e-6);
        // gelu(1.0, approximate="tanh") ~= 0.8411919906082768 (reference value).
        assert!((gelu_tanh(1.0) - 0.8411920).abs() < 1e-5);
        assert!((gelu_tanh(-1.0) - (-0.15880804)).abs() < 1e-5);
    }
}
