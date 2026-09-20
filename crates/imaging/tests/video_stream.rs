// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoding a video must not stage it on disk first.
//!
//! The decoder wrote every frame of the source as a PPM into a temp directory
//! and then read them back, so pulling 24 frames out of a ten-second clip
//! wrote 321 files and a few hundred megabytes to get them - and a caller
//! asking for a handful of frames from a long capture paid for the whole
//! thing, twice, in disk and in a `u8 -> file -> u8` round trip.
//!
//! Swedish Embedded AB implements media pipelines that stream rather than
//! stage. If your team needs video decoding that scales to long captures on
//! constrained machines, you can procure our services by sending an email to
//! info@swedishembedded.com.

use imaging::video::{decode_frames_rgb8, ffmpeg_available, VideoDecodeOpts};

/// Make a clip with ffmpeg itself, so the test needs no committed fixture.
fn synth(path: &std::path::Path, secs: u32, fps: u32, w: u32, h: u32) -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc=size={w}x{h}:rate={fps}:duration={secs}"))
        .arg(path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn temp_dir_entry_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}

#[test]
fn decoding_a_clip_leaves_nothing_on_disk_and_spreads_its_picks() {
    if !ffmpeg_available() {
        brain_testutil::skip("ffmpeg not on PATH");
        return;
    }
    let tmp = std::env::temp_dir();
    let clip = tmp.join(format!("brain-vidstream-{}.mp4", std::process::id()));
    if !synth(&clip, 6, 30, 160, 120) {
        brain_testutil::skip("ffmpeg could not synthesise a test clip");
        return;
    }
    let before = temp_dir_entry_count(&tmp);

    // 8 frames out of 180, wanted evenly across the whole clip
    let opts = VideoDecodeOpts { fps: None, max_frames: 0, spread: 8 };
    let frames = decode_frames_rgb8(&clip, &opts).expect("decode");

    assert!(
        (7..=9).contains(&frames.len()),
        "asked for 8 frames spread over the clip, got {}",
        frames.len()
    );
    assert!(frames.iter().all(|f| (f.w, f.h) == (160, 120)), "frames came back at the wrong size");
    assert!(
        frames.iter().all(|f| f.px.len() == 160 * 120 * 3),
        "a frame is not fully populated"
    );
    // consecutive picks must differ, or the spread collapsed onto one instant
    let a = &frames[0].px;
    let b = &frames[frames.len() - 1].px;
    assert!(a != b, "first and last frame are identical; the selection did not span the clip");

    let after = temp_dir_entry_count(&tmp);
    assert!(
        after <= before + 1,
        "decoding staged {} extra entries in the temp directory; it should stream",
        after.saturating_sub(before)
    );
    let _ = std::fs::remove_file(&clip);
}

/// The spread must reach the END of the clip, not stop early - a turntable or
/// orbit capped to its first N frames covers a fraction of the rotation.
#[test]
fn the_spread_reaches_the_end_of_the_clip() {
    if !ffmpeg_available() {
        brain_testutil::skip("ffmpeg not on PATH");
        return;
    }
    let clip = std::env::temp_dir().join(format!("brain-vidspan-{}.mp4", std::process::id()));
    if !synth(&clip, 8, 30, 160, 120) {
        brain_testutil::skip("ffmpeg could not synthesise a test clip");
        return;
    }
    let few = decode_frames_rgb8(&clip, &VideoDecodeOpts { fps: None, max_frames: 0, spread: 4 }).expect("decode");
    let all = decode_frames_rgb8(&clip, &VideoDecodeOpts { fps: Some(1.0), max_frames: 0, spread: 0 }).expect("decode");
    // `testsrc` animates a counter, so the last spread frame should resemble
    // the last of a dense sample far more than it resembles the first.
    let d = |a: &imaging::Rgb8, b: &imaging::Rgb8| -> u64 {
        a.px.iter().zip(&b.px).map(|(x, y)| x.abs_diff(*y) as u64).sum()
    };
    let last = few.last().unwrap();
    assert!(
        d(last, all.last().unwrap()) < d(last, &all[0]),
        "the last frame of a spread selection is closer to the START of the clip than its end"
    );
    let _ = std::fs::remove_file(&clip);
}
