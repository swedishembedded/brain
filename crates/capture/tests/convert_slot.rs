// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-logic tests for the parts that need no camera: YUYV conversion + the slot.
use capture::slot::Frame;
use capture::{yuyv_to_rgb, FrameSlot};

/// A grey YUYV pixel (Y=128, U=V=128) must decode to mid-grey, and pure luma with
/// neutral chroma must stay achromatic (R==G==B) — the property that catches a
/// swapped U/V or a wrong coefficient.
#[test]
fn neutral_chroma_is_greyscale() {
    // Two pixels: Y0=64, Y1=192, U=V=128 (neutral).
    let yuyv = [64u8, 128, 192, 128];
    let rgb = yuyv_to_rgb(&yuyv, 2, 1);
    assert_eq!(rgb.len(), 6);
    // Each pixel achromatic and equal to its luma.
    assert!(rgb[0] == rgb[1] && rgb[1] == rgb[2], "px0 not grey: {:?}", &rgb[0..3]);
    assert!(rgb[3] == rgb[4] && rgb[4] == rgb[5], "px1 not grey: {:?}", &rgb[3..6]);
    assert!((rgb[0] as i32 - 64).abs() <= 1, "px0 luma off: {}", rgb[0]);
    assert!((rgb[3] as i32 - 192).abs() <= 1, "px1 luma off: {}", rgb[3]);
}

/// Positive V (red chroma) must push red up and blue down relative to grey; positive
/// U must push blue up. Pins the coefficient signs (a swapped R/B channel is the
/// classic YUYV bug).
#[test]
fn chroma_pushes_the_right_channels() {
    // Y=128, U=128 (neutral), V=200 (red).
    let redish = yuyv_to_rgb(&[128, 128, 128, 200], 2, 1);
    assert!(redish[0] > redish[2], "positive V must make red > blue, got {:?}", &redish[0..3]);
    // Y=128, U=200 (blue), V=128 (neutral).
    let blueish = yuyv_to_rgb(&[128, 200, 128, 128], 2, 1);
    assert!(blueish[2] > blueish[0], "positive U must make blue > red, got {:?}", &blueish[0..3]);
}

#[test]
#[should_panic(expected = "even")]
fn odd_width_panics() {
    yuyv_to_rgb(&[0; 6], 3, 1);
}

fn frame(seq: u64) -> Frame {
    Frame { rgb: vec![seq as u8; 12], w: 2, h: 2, seq }
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
