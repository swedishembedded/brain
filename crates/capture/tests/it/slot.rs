// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-logic tests for the slot — the part that needs no camera.
//!
//! The YUYV conversion tests moved with the function to
//! `crates/imaging/tests/it/color.rs`.
use capture::slot::Frame;
use capture::{FrameSlot, Stamp, StampSource};

fn frame(seq: u64) -> Frame {
    Frame { rgb: vec![seq as u8; 12], w: 2, h: 2, seq, stamp: Stamp { ns: 1_000 + seq, source: StampSource::FrameEnd } }
}

/// The slot keeps only the LATEST frame and counts overwrites as drops — the whole
/// point of a one-slot buffer over a channel.
#[test]
fn slot_keeps_latest_and_counts_drops() {
    let s = FrameSlot::new();
    assert!(s.take().is_none(), "empty slot yields nothing");
    s.push(frame(1));
    s.push(frame(2)); // overwrites 1 -> one drop
    s.push(frame(3)); // overwrites 2 -> two drops
    let got = s.take().expect("a frame");
    assert_eq!(got.seq, 3, "take must return the LATEST, not the oldest");
    assert!(s.take().is_none(), "slot is empty after a take");
    let st = s.stats();
    assert_eq!((st.pushed, st.dropped, st.taken), (3, 2, 1));
}

/// Producer never blocks and consumer never blocks — a slow consumer just misses
/// intermediate frames, it does not stall the producer or rebuild a backlog.
#[test]
fn slot_is_lossy_not_blocking() {
    let s = FrameSlot::new();
    for i in 0..100 {
        s.push(frame(i));
    }
    // Only the last survives; 99 were dropped.
    assert_eq!(s.take().unwrap().seq, 99);
    assert_eq!(s.stats().dropped, 99);
}

/// The slot is Send + Sync so a capture thread and the render loop can share it.
#[test]
fn slot_is_shareable_across_threads() {
    let s = FrameSlot::new();
    let p = s.clone();
    let t = std::thread::spawn(move || {
        for i in 0..10 {
            p.push(frame(i));
        }
    });
    t.join().unwrap();
    assert!(s.take().is_some());
    assert_eq!(s.stats().pushed, 10);
}

/// The frame a consumer takes carries the capture time of THAT frame, not of
/// one it replaced: age accounting reads the stamp of what it is processing.
#[test]
fn the_taken_frame_carries_its_own_capture_stamp() {
    let s = FrameSlot::new();
    s.push(frame(4));
    s.push(frame(5));
    let got = s.take().unwrap();
    assert_eq!((got.seq, got.stamp.ns), (5, 1_005));
}
