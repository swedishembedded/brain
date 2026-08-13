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
