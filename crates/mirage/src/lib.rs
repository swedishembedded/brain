// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage: a 51M-parameter visual perception model.
//!
//! This crate is the foundation layer: everything needed to read a released
//! checkpoint safely before any math runs.
//!
//! - [`config`]: the typed configuration.
//! - [`spec`]: the canonical tensor manifest derived from it.
//! - [`import`]: the strict safetensors reader and writer.
//! - [`cact`]: the deployment-container reader and its quantisation decoder.
//! - [`cactmap`]: the mapping of the container's positional entries back to
//!   canonical tensors.
//! - [`tokenizer`]: the SentencePiece-BPE text front end.
//! - [`decoder`]: the decoder with incremental state and greedy generation.
//! - [`describe`], [`form`] and [`model`]: structured results and the end-to-end pipeline.
//! - [`caps`], [`resolver`] and [`memory_cache`]: the served actions, the model-store rules and the decoder-memory reuse.
//! - [`vision`]: the vision forward pass (encoder, heads, reader, memory).
//! - [`train`]: the generic-precision training graph (hand-written, finite-difference-checked backward primitives).
//!
//! Swedish Embedded AB implements solutions for deploying compact vision
//! models on constrained devices for its clients. If your team needs expertise
//! in on-device perception then you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod cact;
pub mod caps;
pub mod cactmap;
pub mod config;
pub mod decoder;
pub mod device;
pub mod describe;
pub mod form;
pub mod import;
pub mod memory_cache;
pub mod model;
pub mod resolver;
pub mod spec;
pub mod tokenizer;
pub mod train;
pub mod vision;
