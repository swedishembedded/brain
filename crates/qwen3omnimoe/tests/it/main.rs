// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod audio_parity;
mod caps_conformance;
mod code2wav_parity;
mod code_predictor_parity;
mod generate_e2e;
mod int8_thinker_executor;
mod int8_thinker_multi_gpu;
mod int8_thinker_periodic_reclaim;
mod int8_thinker_placement;
mod lm_head_i8_parity;
mod real_chat_template_multimodal_splice;
mod real_config;
mod real_import_coverage;
mod speak_e2e;
mod talker_decode;
mod talker_int8_parity;
mod talker_kv_cache;
mod talker_layer_parity;
mod thinker_decode;
mod thinker_int8_parity;
mod thinker_layer_parity;
mod video_generate_e2e;
mod vision_parity;
