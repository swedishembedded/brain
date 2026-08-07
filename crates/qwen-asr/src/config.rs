// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR configuration. The audio encoder is a Whisper/Qwen-omni-style
//! conv-stem + windowed transformer; the text decoder is a stock Qwen3-1.7B
//! (reused from `crates/qwen`). Values mirror `Qwen/Qwen3-ASR-1.7B-hf/config.json`.

/// Audio encoder (`qwen3_asr_encoder`).
#[derive(Clone, Copy, Debug)]
pub struct AudioEncoderConfig {
    pub num_mel_bins: u32,      // 128
    pub d_model: u32,           // 1024
    pub n_heads: u32,           // 16 (head_dim 64)
    pub ffn_dim: u32,           // 4096
    pub n_layers: u32,          // 24
    pub downsample_hidden: u32, // 480  (conv channels)
    pub output_dim: u32,        // 2048 (projector out = decoder hidden)
    pub n_window: u32,          // 50   (chunk = 2*n_window = 100 mel frames)
    pub n_window_infer: u32,    // 800  (attention window, in mel frames)
    pub max_pos: u32,           // 13   (sinusoidal table length)
    pub eps: f32,               // LayerNorm eps (torch default 1e-5)
}

impl AudioEncoderConfig {
    pub fn qwen3_asr() -> AudioEncoderConfig {
        AudioEncoderConfig {
            num_mel_bins: 128,
            d_model: 1024,
            n_heads: 16,
            ffn_dim: 4096,
            n_layers: 24,
            downsample_hidden: 480,
            output_dim: 2048,
            n_window: 50,
            n_window_infer: 800,
            max_pos: 13,
            eps: 1e-5,
        }
    }

    /// Qwen3-Omni's Thinker audio tower (`thinker_config.audio_config`) — the
    /// SAME Whisper/Qwen-omni-style conv-stem + windowed-transformer shape
    /// [`qwen3_asr`] already models, just wider/deeper: `num_mel_bins`,
    /// `downsample_hidden`, `output_dim`, `n_window`, `n_window_infer` and
    /// `max_pos` are identical between the two models (confirmed against the
    /// released `Qwen/Qwen3-Omni-30B-A3B-Instruct` config — see
    /// `docs/models/omni/status.md` "Facts"); only `d_model`/`n_heads`/
    /// `ffn_dim`/`n_layers` differ. `crates/omni` reuses [`crate::encoder::AudioEncoder`]
    /// unchanged with this preset — no new encoder code, only a config.
    pub fn qwen3_omni() -> AudioEncoderConfig {
        AudioEncoderConfig {
            num_mel_bins: 128,
            d_model: 1280,
            n_heads: 20,
            ffn_dim: 5120,
            n_layers: 32,
            downsample_hidden: 480,
            output_dim: 2048,
            n_window: 50,
            n_window_infer: 800,
            max_pos: 13,
            eps: 1e-5,
        }
    }

    /// Mel frames per conv chunk (`2 * n_window`).
    pub fn chunk_len(&self) -> u32 {
        2 * self.n_window
    }

    /// Post-conv token count for a chunk of `frames` valid mel frames:
    /// three (k=3, s=2, p=1) convolutions → `((frames-1)/2+1 -1)/2+1 -1)/2+1`.
    pub fn post_cnn_len(&self, frames: u32) -> u32 {
        if frames == 0 {
            return 0;
        }
        let mut l = frames;
        for _ in 0..3 {
            l = (l - 1) / 2 + 1;
        }
        l
    }

    /// Post-conv frequency bins: mel bins through three (k=3,s=2,p=1) convs.
    pub fn post_cnn_freq(&self) -> u32 {
        let mut f = self.num_mel_bins;
        for _ in 0..3 {
            f = (f - 1) / 2 + 1;
        }
        f
    }

    /// Flattened conv-out input width: `downsample_hidden * post_cnn_freq`.
    pub fn conv_out_in(&self) -> u32 {
        self.downsample_hidden * self.post_cnn_freq()
    }

    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
}

/// Full Qwen3-ASR config: audio encoder + (Qwen3) text decoder handle.
#[derive(Clone, Debug)]
pub struct QwenAsrConfig {
    pub audio: AudioEncoderConfig,
    pub audio_token_id: u32,     // 151676
    pub timestamp_token_id: u32, // 151705
    /// The text decoder is a stock Qwen3-1.7B; see `qwen::config::QwenConfig::qwen3_1_7b`.
    pub text: qwen::config::QwenConfig,
}

impl QwenAsrConfig {
    pub fn qwen3_asr_1_7b() -> QwenAsrConfig {
        QwenAsrConfig {
            audio: AudioEncoderConfig::qwen3_asr(),
            audio_token_id: 151676,
            timestamp_token_id: 151705,
            text: qwen::config::QwenConfig::qwen3_1_7b(),
        }
    }
}
