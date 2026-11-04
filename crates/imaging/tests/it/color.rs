// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YUYV -> RGB colour conversion, moved here with the function from
//! `crates/capture`. These are the tests that pin the BT.601 coefficient signs;
//! they caught nothing new in the move, which is the point of moving them with it.
use imaging::color::yuyv_to_rgb;

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

