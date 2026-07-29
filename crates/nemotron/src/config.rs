// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron 3.5 ASR Streaming (0.6B) config. FastConformer encoder + RNN-T
//! transducer + language-prompt conditioning. Values mirror the HF
//! `nvidia/nemotron-3.5-asr-streaming-0.6b/config.json`.

#[derive(Clone, Copy, Debug)]
pub struct NemotronConfig {
    // --- encoder (FastConformer) ---
    pub num_mel_bins: u32,       // 128
    pub hidden: u32,             // 1024
    pub n_layers: u32,           // 24
    pub n_heads: u32,            // 8 (head_dim 128)
    pub intermediate: u32,       // 4096
    pub conv_kernel: u32,        // 9  (Conformer depthwise conv)
    pub subsampling_factor: u32, // 8  (3 stride-2 stages)
    pub subsampling_channels: u32, // 256
    pub subsampling_kernel: u32, // 3
    pub subsampling_stride: u32, // 2
    pub sliding_window: u32,     // 57 (left ctx = 56)
    pub default_lookahead: u32,  // 3
    pub ln_eps: f32,             // 1e-5 (torch LayerNorm default)
    // --- RNN-T decoder ---
    pub decoder_hidden: u32,     // 640
    pub num_decoder_layers: u32, // 2 (LSTM)
    pub vocab: u32,              // 13088
    pub blank_token_id: u32,     // 13087
    pub max_symbols_per_step: u32, // 10
    // --- language prompt ---
    pub num_prompts: u32,          // 128
    pub prompt_intermediate: u32,  // 2048
    pub default_prompt_id: u32,    // 101 (auto)
}

impl NemotronConfig {
    pub fn nemotron_3_5_asr_0_6b() -> NemotronConfig {
        NemotronConfig {
            num_mel_bins: 128,
            hidden: 1024,
            n_layers: 24,
            n_heads: 8,
            intermediate: 4096,
            conv_kernel: 9,
            subsampling_factor: 8,
            subsampling_channels: 256,
            subsampling_kernel: 3,
            subsampling_stride: 2,
            sliding_window: 57,
            default_lookahead: 3,
            ln_eps: 1e-5,
            decoder_hidden: 640,
            num_decoder_layers: 2,
            vocab: 13088,
            blank_token_id: 13087,
            max_symbols_per_step: 10,
            num_prompts: 128,
            prompt_intermediate: 2048,
            default_prompt_id: 101,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden / self.n_heads
    }

    /// Number of stride-2 subsampling stages (`log2(factor)`).
    pub fn subsampling_stages(&self) -> u32 {
        self.subsampling_factor.trailing_zeros()
    }

    /// Output freq bins after the subsampling stack (causal pad `(k-1, s-1)` each stage).
    pub fn subsampling_out_freq(&self) -> u32 {
        let (k, s) = (self.subsampling_kernel, self.subsampling_stride);
        let mut w = self.num_mel_bins;
        for _ in 0..self.subsampling_stages() {
            w = (w + (k - 1) + (s - 1) - k) / s + 1;
        }
        w
    }

    /// Flattened subsampling output width = `channels * out_freq` (encoder linear input).
    pub fn subsampling_out_hidden(&self) -> u32 {
        self.subsampling_channels * self.subsampling_out_freq()
    }
}
