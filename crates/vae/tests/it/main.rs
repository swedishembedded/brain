// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `blocks3d_norm`, `native_conv_route` keep binaries of their own.

#[path = "../zeros/mod.rs"]
mod zeros;

mod batch_forward;
mod decode_parity;
mod encode_parity;
mod flux2_parity;
mod footprint;
mod recompute;
mod reverse_memory;
mod sdxl_decode_parity;
mod tiled_parity;
