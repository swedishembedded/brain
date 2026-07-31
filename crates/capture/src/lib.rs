// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Webcam capture for the depth demo.
//!
//! Two layers, split so the hard, subtle parts are testable with ZERO hardware:
//!   * [`slot`] — a single-slot latest-frame buffer between the capture thread and
//!     the render loop. Pure.
//!   * [`v4l2`] (behind the `linux` cfg) — the ioctl FFI + device open/stream. The
//!     only part that needs a camera; its ABI constants are pinned by tests.
//!
//! The pixel-format conversion this crate used to own (`convert::yuyv_to_rgb`)
//! moved to `imaging::color::yuyv_to_rgb`, the workspace's one home for colour
//! conversion; this crate is V4L2 and nothing else.
//!
//! The capture thread blocks in `DQBUF` ~33ms and always overwrites the slot
//! (producer wins), so the render loop takes the latest frame and never rebuilds a
//! backlog — the same latest-state discipline the SDL keystroke path uses.

pub mod slot;

#[cfg(target_os = "linux")]
pub mod v4l2;

pub use slot::{Frame, FrameSlot, SlotStats};

#[cfg(target_os = "linux")]
pub use v4l2::Device;
