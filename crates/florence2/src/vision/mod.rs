// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DaViT vision encoder. [`config`] is the verified stage/block schedule;
//! the block-level forward (patch embed, per-block dual depthwise-conv
//! residual, window attention, channel attention, MLP) composes from
//! brain's existing ViT-block primitives (`model::vit::WindowPlan`,
//! `model::block::chunked_bidir_fwd`) and is the next piece to land.

pub mod config;

pub use config::{DavitConfig, DavitPatchSpec, DavitStageSpec};
