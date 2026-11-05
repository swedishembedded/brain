// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `anchor_mid_two_stage_real`, `anchor_two_stage_real`, `block_pipeline`, `scratch_pool`, `vae_parity`, `vae_tiling` keep binaries of their own.

mod anchor_real;
mod anchor_with_mid_real;
mod anchors;
mod audio_parity;
mod audio_seam;
mod av_block_grad;
mod av_concept_learning;
mod av_dit_parity;
mod av_lora_reload;
mod av_lora_train;
mod av_overfit;
mod av_recompute;
mod av_shard_2gpu_real;
mod av_shard_parity;
mod block_grad;
mod block_weight_cache;
mod bwe_parity;
mod cfg_parallel;
mod clip_stability_real;
mod connector_real_parity;
mod device_bytes_real;
mod device_residency;
mod dit_parity;
mod duration_head_parity;
mod gguf_quant_real;
mod host_forward_parity;
mod int8_compute;
mod int8_storage;
mod longform;
mod longform_mid_real;
mod lora_train;
mod maskcond_parity;
mod motion_real;
mod na_decoder_parity;
mod overfit;
mod prompt_adherence_real;
mod recompute;
mod refcond_parity;
mod shard_parity;
mod streamed_vs_eager_real;
mod upsampler_parity;
mod upscale;
