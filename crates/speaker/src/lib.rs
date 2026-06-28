// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ECAPA-TDNN speaker encoder for Qwen3-TTS voice cloning.
//!
//! Front-end log-mel (`audio::mel`) -> initial TDNN block -> 3× SE-Res2Net
//! blocks (Res2Net dilated convs + Squeeze-Excitation) -> multi-layer feature
//! aggregation -> attentive statistics pooling -> 1×1 conv -> speaker embedding.
//! All convs are 1D (`audio::conv`); imports the official `speaker_encoder.*`
//! weights from the 0.6B/1.7B checkpoints.

pub mod config;

pub use config::SpeakerConfig;
