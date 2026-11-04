// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `decode_plans`, `gguf_conformance`, `gguf_mixed_step`, `gguf_score`, `hf_fp8_conformance`, `hf_fp8_embedded_prompt`, `hf_fp8_speculative`, `hf_fp8_vision_real`, `vision_train_real` keep binaries of their own.

#[path = "../common/mod.rs"]
mod common;

mod adapter_bank;
mod adapter_sample;
mod adapter_variants;
mod arena_drain;
mod bf16_pack;
mod chunk_round_reclaim;
mod chunk_tape_decode_parity;
mod chunked_prefill;
mod convergence;
mod convert_base;
mod cuda_leaks;
mod decode_fusion;
mod decode_steady_state;
mod decode_step;
mod dflash2_real;
mod dora_packed_bases;
mod edgerock_config;
mod edgerock_decode;
mod edgerock_golden_parity;
mod edgerock_hf_fixture;
mod edgerock_real_weights;
mod edgerock_serve;
mod edgerock_training;
mod finetune_streamed_guard;
mod fp8_source;
mod full_ft_precision;
mod fused_gate_up;
mod fused_qkv;
mod fused_qkvz;
mod gdn_intermediate_trace;
mod gdn_layer_boundary_trace;
mod gdn_state_pool_real;
mod generate_streaming;
mod generate_streaming_mtp;
mod gguf_i8_divergence_diagnostic;
mod gguf_i8_vs_fp32_real;
mod gguf_kv_tier_real;
mod gguf_reference_parity_real;
mod gguf_resident_real;
mod gguf_resident_spec_real;
mod gguf_resident_verify_cost_real;
mod gguf_vs_fp8_permutation_search;
mod gguf_vs_fp8_weights_real;
mod golden_parity;
mod import_mtp_real_weight;
mod int8_real_weight_sanity;
mod kv_tier;
mod kv_tier_bench;
mod layer_streamed_full_ft;
mod lora_adapter_file;
mod lora_batch;
mod lora_bf16_base;
mod lora_bf16_residuals;
mod lora_cuda_matches_cpu;
mod lora_decode;
mod lora_flash;
mod lora_fp8_base;
mod lora_freezes_base;
mod lora_gdn_segments;
mod lora_gemm_precision;
mod lora_grad_norm;
mod lora_head;
mod lora_nf4_base;
mod lora_real_dims_cuda_matches_cpu;
mod lora_recompute;
mod lora_resume_bf16_base;
mod lora_resume_gate;
mod lora_roundtrip;
mod lora_seq_len;
mod lora_serve_fold;
mod mixed_precision;
mod model_i8_smoke;
mod model_q4_smoke;
mod model_smoke;
mod mtp_convergence;
mod objectives;
mod param_list_matches_golden;
mod prefill_embeds;
mod prefill_fusion;
mod prefill_round_census;
mod real_weight_streaming;
mod resident_checkpoint;
mod rmsnorm_variant_agreement;
mod sample_generate;
mod score_readout;
mod serve;
mod serve_conformance;
mod shard_parity;
mod spec_decode;
mod stream_profile;
mod stream_train_equivalence;
mod stream_train_packed;
mod stream_train_real;
mod streaming_forward;
mod train_memory;
mod train_tier_trajectory;
mod unified_pool_real;
mod vision_parity;
mod vision_train;
mod vision_train_streamed;
mod vl;
mod yarn_rope_scaling;
