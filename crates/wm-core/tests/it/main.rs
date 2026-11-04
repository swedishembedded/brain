// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod attn_bias;
mod attn_bwd;
mod attn_helper;
mod conv_bwd;
mod dwconv3d;
mod film;
mod gn;
mod gn_fd;
mod gn_multigroup;
mod l2norm_scale;
mod worldmodel_trait;
