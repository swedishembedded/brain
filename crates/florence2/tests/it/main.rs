// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod davit_channel_block_parity;
mod davit_full_forward_parity;
mod davit_image_project_parity;
mod davit_patch_embed_parity;
mod davit_spatial_block_parity;
mod ground_smoke;
mod import_end_to_end_parity;
mod recompute;
mod text_lm_parity;
mod train_overfit;
