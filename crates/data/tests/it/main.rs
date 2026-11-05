// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod chat_dataset_generic;
mod chat_final_turn_mask;
mod chat_image_parts;
mod chat_packed_answers;
mod chat_rollout_records;
mod chat_teacher_context;
mod chat_sample_encode;
mod chat_template_cross_check;
mod clip_tokenizer_parity;
mod deepseek_chat_render;
mod deepseek_tokenizer_parity;
mod imageset_captions_yaml;
mod imageset_pairs;
mod llama_bpe_parity;
mod t5xxl_precompiled_parity;
mod umt5_tokenizer_parity;
mod wordpiece_parity;
