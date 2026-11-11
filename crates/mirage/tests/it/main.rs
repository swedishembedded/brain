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
mod device_model;
mod device_cact;
mod device_cact_accept;
mod device_coop;
mod device_decoder;
mod device_reader;
mod device_train;
mod device_trainer;
mod device_vision;
mod fd;
mod import;
mod pipeline;
mod serving;
mod spec;
mod tokenizer;
mod train_fit;
mod train_graph;
mod train_real;
mod train_ops;
mod vision;
mod vision_masks;
mod vision_reader;
