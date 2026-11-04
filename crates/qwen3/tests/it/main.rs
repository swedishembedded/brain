// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `bench_train_p40` keeps a binary of its own.

mod adamw_hyperparameters;
mod adapter_variants;
mod batched_prefill;
mod bf16_dx_reg;
mod chunked_head;
mod deepseek_chat_format;
mod deepseek_gguf_parity;
mod deepseek_parity;
mod deepseek_refs_resolve;
mod dp_parity;
mod encoder_parity;
mod engine_adapter_bank;
mod engine_half_weights;
mod engine_prefill_mixed;
mod engine_rollout;
mod engine_variants;
mod external_head;
mod fim_request;
mod finetune_full_hf;
mod flops;
mod flux2_text_parity;
mod frozen_build;
mod full_ft_precision;
mod full_ft_resume;
mod gguf_export_roundtrip;
mod gguf_serving_real;
mod gguf_vs_safetensors_real;
mod head_tiled_backward;
mod hf_export_roundtrip;
mod hf_fixture;
mod hf_shard_source_real;
mod integration_qwen3;
mod kv_offload_real;
mod layer_streamed_full_ft;
mod lora_adapter_file;
mod lora_bf16_base;
mod lora_gemm_precision;
mod lora_hf_base;
mod lora_int8_base;
mod lora_int8_base_real;
mod lora_layout;
mod lora_learning_gate;
mod lora_packed_bases;
mod lora_pipeline;
mod lora_pipeline_overlap;
mod lora_resume_gate;
mod lora_roundtrip;
mod lora_runtime_adapter;
mod lora_saved_boundaries;
mod lora_serve_fold;
mod lora_snapshots;
mod lsa_engine;
mod lsa_native;
mod mem_budget_inference;
mod mixed_precision;
mod mm_pipeline;
mod mm_splice_regions;
mod no_kernel_names;
mod parity;
mod peft_export;
mod real_full_ft_precision;
mod real_resident_full_ft;
mod resident_full_ft;
mod rope_scaling_model;
mod serve_conformance;
mod shard_microbatch;
mod shard_parity;
mod slice;
mod step_timing;
