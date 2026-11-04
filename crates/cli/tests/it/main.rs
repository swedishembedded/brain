// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod caps_color;
mod device_routing;
mod doctor_cli;
mod flops_coverage;
mod flops_diffusion_cli;
mod flops_help;
mod flops_stage_model;
mod forecast_finetune_timesfm3;
mod lpips_cli;
mod model_ids;
mod models_list;
mod oos_skill_eval;
mod perf_longctx;
mod profile_collectives_cli;
mod qwen_eval;
mod qwen_lora_finetune;
mod roofline;
mod seed_flag;
mod serve_cli;
mod serve_connection;
mod version;
mod yolov8_eval;
mod zipdepth_smoke;
