// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A single-slot, latest-frame buffer between the capture thread and the render
//! loop. Capacity ONE, the producer always wins.
//!
//! NOT a channel, and the reasoning is load-bearing (plan R-notes):
//!   * `mpsc` is unbounded — a 3s render stall rebuilds a backlog (921KB/frame) and
//!     3s of unrepayable latency.
//!   * `sync_channel(1)` blocks the PRODUCER, so a slow consumer pushes the capture
//!     thread late back into DQBUF, the driver's ring fills, and V4L2 drops frames
//!     for us with no counter and no control.
//!   * A one-slot buffer where the producer overwrites keeps latency at one frame,
//!     counts what it drops, and never blocks either side. The render loop's Hsm
//!     queue then has a high-water mark of 2 and never needs bounding.

use std::sync::{Arc, Mutex};

/// What the slot has seen: how many frames were pushed, and how many were dropped
/// because the consumer had not taken the previous one yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotStats {
    pub pushed: u64,
    pub dropped: u64,
    pub taken: u64,
}

/// One RGB frame plus its dimensions.
#[derive(Clone, Debug)]
pub struct Frame {
    pub rgb: Vec<u8>,
    pub w: u32,
    pub h: u32,
    /// Monotonic frame index from the producer, for the HUD / drop accounting.
    pub seq: u64,
}

struct Inner {
    slot: Option<Frame>,
    stats: SlotStats,
}

/// A cloneable handle to the shared slot. The capture thread holds one, the render
/// loop the other; both refer to the same `Arc<Mutex<..>>`.
#[derive(Clone)]
pub struct FrameSlot {
    inner: Arc<Mutex<Inner>>,
}

impl Default for FrameSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSlot {
    pub fn new() -> FrameSlot {
        FrameSlot { inner: Arc::new(Mutex::new(Inner { slot: None, stats: SlotStats::default() })) }
    }

    /// Producer side: install the latest frame, overwriting any un-taken one. Never
    /// blocks. An overwrite increments `dropped` — that count IS the frame-drop
    /// signal the HUD shows.
    pub fn push(&self, frame: Frame) {
        let mut g = self.inner.lock().unwrap();
        g.stats.pushed += 1;
        if g.slot.is_some() {
            g.stats.dropped += 1;
        }
        g.slot = Some(frame);
    }

    /// Consumer side: take the latest frame if there is one, leaving the slot empty.
    /// Never blocks; returns `None` when no new frame has arrived since the last take
    /// (a tick with no frame posts nothing and the machine does not move).
    pub fn take(&self) -> Option<Frame> {
        let mut g = self.inner.lock().unwrap();
        let f = g.slot.take();
        if f.is_some() {
            g.stats.taken += 1;
        }
        f
    }

    pub fn stats(&self) -> SlotStats {
        self.inner.lock().unwrap().stats
    }
}
