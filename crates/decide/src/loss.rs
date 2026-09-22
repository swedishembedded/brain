// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision objective, and the softmax it is built on.
//!
//! Re-exported from `brain-rlcd`, which owns this math now: it is pure
//! `&[f32]` host arithmetic with no dependency on this crate's model, and
//! `brain-modernbert` (the Laya decision backbone) needs the same objective
//! without depending on `brain-decide` to get it. See
//! [`rlcd::scoring`](../../rlcd/scoring/index.html)'s module doc for the full
//! reasoning and the math this crate used to own directly.
//!
//! Every name below is unchanged from before the move, so no caller here or
//! in the SDK needed to change.

pub use rlcd::scoring::{decision_loss, decision_loss_soft, softmax, LossConfig};
