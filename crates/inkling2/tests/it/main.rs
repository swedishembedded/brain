// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary. A test here must not change process-wide state (the current
//! directory, environment variables): the others run beside it.

mod activation_tier;
mod declared_tier;
mod image_text;
mod kv_precision;
mod layout;
mod real_lora;
mod real_parity;
mod serve;
mod serve_adapters;
mod serve_images;
mod tiny_parity;
mod training;
mod vision;
mod vision_train;
