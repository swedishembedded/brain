// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shapes for the new half of DeepSeek-OCR-2's vision tower: the Qwen2-shaped
//! GQA resampler that replaces v1's CLIP-L/14 tower, plus the SAM config it
//! sits on top of ([`sam1::SamViTConfig`], reused unmodified - only its
//! `compress_out` differs from v1's preset, since there is no CLIP width to
//! bridge to any more).
//!
//! Flat parameter names here are an independent restatement of
//! `crates/gguf/src/deepseekocr2_vision.rs`'s classifier output, the way
//! `sam1::SamViTConfig::param_list` already independently restates the SAM
//! half of that same file - neither crate depends on the other for this list,
//! and a real-checkpoint import (M6) is what proves the two still agree.

use sam1::SamViTConfig;

/// The 24-layer Qwen2-shaped GQA tower's own shape. Not `qwen3::QwenConfig`:
/// this tower is a bare resampler (no token embedding, no LM head, a mask
/// `qwen3::Qwen` cannot express), so a purpose-built struct is more honest
/// than borrowing a config type shaped for a causal decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2EncoderConfig {
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub ffn_hidden: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Rows in the query bank paired with a local (768x768) tile.
    pub n_query_local: u32,
    /// Rows in the query bank paired with the global (1024x1024) view.
    pub n_query_global: u32,
}

impl Qwen2EncoderConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim()
    }
    /// GQA's repeat factor: how many query heads share one KV head.
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }

    /// The real DeepSeek-OCR-2 shape, pinned from the mmproj's own KV and
    /// tensor shapes rather than `config.json`'s stale `downsample_channels`.
    pub fn deepseek_ocr2() -> Qwen2EncoderConfig {
        Qwen2EncoderConfig {
            d_model: 896,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            ffn_hidden: 4864,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            n_query_local: 144,
            n_query_global: 256,
        }
    }

    fn check(&self) {
        assert!(self.n_heads > 0 && self.d_model.is_multiple_of(self.n_heads), "d_model must be a whole number of heads");
        assert!(self.n_kv_heads > 0 && self.n_heads.is_multiple_of(self.n_kv_heads), "n_heads must be a whole multiple of n_kv_heads");
        // `encoder.rs` dispatches `model::block::rmsnorm_fwd`, whose kernel
        // hardcodes `model::block::RMSNORM_EPS` (1e-6) rather than reading a
        // runtime value - the same fixed-eps family `qwen3`/`deepseek2` use.
        // The real checkpoint's own KV states an eps that ROUNDS to 1e-6 but
        // is not bit-identical to it (9.999999974752427e-07); this assertion
        // makes that assumption explicit and catches a future checkpoint
        // whose eps genuinely differs, rather than silently ignoring it.
        assert!(
            (self.rms_eps - model::block::RMSNORM_EPS).abs() < 1e-4,
            "rms_eps {} is too far from the fixed kernel constant {} for rmsnorm_fwd to be a faithful dispatch",
            self.rms_eps,
            model::block::RMSNORM_EPS
        );
    }

    /// One layer's flat parameter names, in `vision.encoder.blocks.{l}.*`.
    fn layer_params(&self, l: u32) -> Vec<(String, usize)> {
        let (d, kv, ff) = (self.d_model as usize, self.kv_dim() as usize, self.ffn_hidden as usize);
        let b = |leaf: &str| format!("vision.encoder.blocks.{l}.{leaf}");
        vec![
            (b("norm1.weight"), d),
            (b("norm2.weight"), d),
            (b("attn.q.weight"), d * d),
            (b("attn.q.bias"), d),
            (b("attn.k.weight"), kv * d),
            (b("attn.k.bias"), kv),
            (b("attn.v.weight"), kv * d),
            (b("attn.v.bias"), kv),
            (b("attn.out.weight"), d * d),
            (b("mlp.gate.weight"), ff * d),
            (b("mlp.up.weight"), ff * d),
            (b("mlp.down.weight"), d * ff),
        ]
    }

    /// The tower's own parameters: every layer, the final norm and the two
    /// learned query banks. Does NOT include the projector or the view
    /// separator - those belong to [`DeepseekOcr2VisionConfig`], since they
    /// are not properties of the encoder stack itself.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        self.check();
        let mut out = Vec::new();
        for l in 0..self.n_layers {
            out.extend(self.layer_params(l));
        }
        out.push(("vision.encoder.norm.weight".to_string(), self.d_model as usize));
        out.push(("vision.query_local.weight".to_string(), self.n_query_local as usize * self.d_model as usize));
        out.push(("vision.query_global.weight".to_string(), self.n_query_global as usize * self.d_model as usize));
        out
    }
}

/// The whole `DeepEncoder V2` tower: SAM's grid, the resampler above, and the
/// single linear projector into the decoder's width.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcr2VisionConfig {
    pub sam: SamViTConfig,
    pub encoder: Qwen2EncoderConfig,
    /// The language model's own hidden width - the projector's output and
    /// the view separator's width.
    pub decoder_hidden: u32,
}

impl DeepseekOcr2VisionConfig {
    pub fn check(&self) {
        self.encoder.check();
        assert_eq!(self.sam.compress_out, self.encoder.d_model, "SAM's compressor must emit exactly the encoder's own width - there is no bridge in this tower");
    }

    /// The projector's and separator's flat parameter names - kept apart from
    /// [`Qwen2EncoderConfig::param_list`] since they are properties of the
    /// whole tower, not the repeated block stack.
    pub fn projector_param_list(&self) -> Vec<(String, usize)> {
        let (pin, pout) = (self.encoder.d_model as usize, self.decoder_hidden as usize);
        vec![
            ("vision.projector.fc.weight".to_string(), pout * pin),
            ("vision.projector.fc.bias".to_string(), pout),
            ("vision.view_separator".to_string(), pout),
        ]
    }
}
