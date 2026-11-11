// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ECAPA-TDNN speaker-encoder configuration (Qwen3-TTS `speaker_encoder`).
//! Defaults from `configuration_qwen3_tts.py`; `enc_dim` from the checkpoint's
//! `speaker_encoder_config` (1024 for the 0.6B/1.7B base models).

#[derive(Clone, Debug)]
pub struct SpeakerConfig {
    pub mel_dim: u32,              // 128 (input log-mel features)
    pub enc_dim: u32,             // 1024 (output speaker embedding)
    pub enc_channels: Vec<u32>,   // [512,512,512,512,1536]
    pub enc_kernel_sizes: Vec<u32>, // [5,3,3,3,1]
    pub enc_dilations: Vec<u32>,  // [1,2,3,4,1]
    pub enc_attention_channels: u32, // 128
    pub enc_res2net_scale: u32,   // 8
    pub enc_se_channels: u32,     // 128
    pub sample_rate: u32,         // 24000
}

impl Default for SpeakerConfig {
    fn default() -> SpeakerConfig {
        SpeakerConfig {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
            sample_rate: 24000,
        }
    }
}

impl SpeakerConfig {
    /// The largest reflect pad, in frames, of the encoder's "same" layers:
    /// `dilation * (kernel - 1) / 2`.
    pub fn max_pad_frames(&self) -> usize {
        self.enc_kernel_sizes.iter().zip(&self.enc_dilations).map(|(&k, &d)| (d * (k - 1) / 2) as usize).max().unwrap_or(0)
    }

    /// The fewest 24 kHz samples the encoder can read: a reflect pad needs
    /// more frames than it pads, and the front end gives
    /// `1 + (n - HOP) / HOP` frames for `n >= HOP`.
    pub fn min_samples(&self) -> usize {
        (self.max_pad_frames() + 1) * crate::mel::HOP
    }

    /// Parse from the top-level 0.6B/1.7B `config.json` (`speaker_encoder_config`
    /// only carries `enc_dim`/`sample_rate`; the rest are ECAPA defaults).
    pub fn from_json(root: &serde_json::Value) -> SpeakerConfig {
        let s = &root["speaker_encoder_config"];
        let mut c = SpeakerConfig::default();
        if let Some(d) = s["enc_dim"].as_u64() {
            c.enc_dim = d as u32;
        }
        if let Some(sr) = s["sample_rate"].as_u64() {
            c.sample_rate = sr as u32;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `min_samples` is the shortest clip whose log-mel has more frames than
    /// the widest reflect pad, and one hop less is too short.
    #[test]
    fn min_samples_is_the_shortest_clip_every_layer_can_pad() {
        let cfg = SpeakerConfig::default();
        let pad = cfg.max_pad_frames();
        assert_eq!(pad, 4);
        let frames = |n: usize| crate::mel::log_mel(&vec![0.1f32; n]).1;
        assert!(frames(cfg.min_samples()) > pad);
        assert!(frames(cfg.min_samples() - crate::mel::HOP) <= pad);
    }
}
