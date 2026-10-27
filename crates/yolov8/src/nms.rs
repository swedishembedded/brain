// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Non-Maximum Suppression, **re-exported** from `imaging::nms`.
//!
//! The implementation (and its strict-`>` suppression rule) moved there so a
//! second detector can use it without depending on this crate's GPU training
//! code; this module keeps the names every YOLOv8 call site already uses.

pub use imaging::nms::{nms, nms_agnostic, Detection};
