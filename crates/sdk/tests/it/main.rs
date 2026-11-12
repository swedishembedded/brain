// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `auto_pipeline`, `depth_pipeline`, `detection_pipeline`, `embedding_pipeline`, `forecast_pipeline`, `ground_pipeline`, `image_pipeline`, `music_pipeline`, `restore_pipeline`, `segment_pipeline`, `text_pipeline`, `text_pipeline_variants`, `transcribe_pipeline`, `tts_pipeline`, `upscale_pipeline`, `video_pipeline`, `vlm_pipeline` keep binaries of their own.

#[path = "../chat_fixture/mod.rs"]
mod chat_fixture;
#[path = "../common/mod.rs"]
mod common;

mod audio_clip;
mod chat_finetune;
mod chat_finetune_policy_gradient;
mod chat_finetune_policy_gradient_archs;
mod chat_finetune_qwen35;
mod chat_finetune_qwen35moe;
mod chat_finetune_ricursive;
mod text_ricursive;
mod toy_training;
mod chat_full_finetune;
mod chat_pipeline;
mod continual_reader;
mod creature;
mod decision_eval;
mod decision_minibatch;
mod decision_persistence;
mod decision_pipeline;
mod decision_promote;
mod decision_promote_laya;
mod decision_train;
mod decision_train_laya;
mod default_seeds_decision;
mod default_seeds_text;
mod document_study;
mod emit_held_out_records;
mod ensemble;
mod improve;
mod model_optimizer;
mod model_pruner;
mod preference_finetune;
mod score_long_record_real;
mod score_tokens_qwen35;
mod speech_ingress;
mod timeline;
mod tts_policy;
mod tts_profile;
mod tts_resident;
