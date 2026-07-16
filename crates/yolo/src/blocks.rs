// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The conv blocks now live in [`vision::blocks`] so they can be shared with
//! brain's other vision models instead of forked.
//!
//! Re-exported here permanently: `head.rs`, `model.rs` and the P2 block tests all
//! import `crate::blocks::*`, and `brain-npu` documents against `yolo::blocks::Conv`.

pub use vision::blocks::*;
