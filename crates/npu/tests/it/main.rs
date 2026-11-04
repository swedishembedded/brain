// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `npu_live` keeps a binary of its own.

mod codec_onnx;
mod deepseekocr2_onnx;
mod depth_npu_render;
mod depth_onnx;
mod export;
mod face_onnx;
mod glm_onnx;
mod group_norm_onnx;
mod kronos_kvcache;
mod lfm_onnx;
mod mirror_onnx;
mod nemotron_encoder;
mod nemotron_subsampling;
mod npu_runtime;
mod npugraph;
mod ov_link;
mod parity;
mod qwen35moe_onnx;
mod qwen_asr_encoder;
mod qwen_onnx;
mod qwenvl_onnx;
mod roofline;
mod talker_onnx;
mod upscale_onnx;
mod vae_onnx;
mod wm_onnx;
