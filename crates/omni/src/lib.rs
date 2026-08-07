// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni-30B-A3B-Instruct: an omni-modal model — text, audio and
//! vision/video in, text and speech out.
//!
//! Three components chained end to end (see `docs/models/omni/readme.md` for
//! the full architecture and `docs/models/omni/status.md` for the build
//! ledger): the **Thinker** (a Qwen3-MoE decoder with audio/vision towers
//! spliced in), the **Talker** (a Qwen3-MoE decoder + MTP code predictor,
//! consuming the Thinker's hidden state), and **Code2Wav** (RVQ decode to a
//! 24 kHz waveform). Built on `model::moe`'s sparse top-k core, reusing
//! `qwen-asr`'s audio encoder shape, `qwenvl`'s vision tower + M-RoPE,
//! `tts`'s MTP code predictor, and `codec`'s SEANet vocoder.

pub mod config;
pub mod import;
