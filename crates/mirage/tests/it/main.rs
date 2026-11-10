// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary. A test here must not
//! change process-wide state: the others run beside it.

mod cact;
mod cactgen;
mod cactmap;
mod common;
mod config;
mod decoder;
mod describe;
mod device;
mod import;
mod pipeline;
mod spec;
mod tokenizer;
mod vision;
mod vision_masks;
mod vision_reader;
