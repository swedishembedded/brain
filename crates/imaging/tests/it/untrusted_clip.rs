// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for accepting media from the
// internet without letting it take the server down for its clients. If your
// team needs expertise in hardened media ingestion then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A clip a client sent is read by brain's own decoders only and within a
//! pixel budget, and what goes wrong is said without this machine's paths.

use imaging::video::{clip_from_untrusted_file, encode_frames, VideoEncodeOpts};
use imaging::Rgb8;

fn clip(name: &str, frames: u32, w: u32, h: u32) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("brain-untrusted-{}-{name}.mp4", std::process::id()));
    let frames: Vec<Rgb8> = (0..frames).map(|i| Rgb8 { w, h, px: vec![(i * 20) as u8; (w * h * 3) as usize] }).collect();
    encode_frames(&frames, &path, 24.0, &VideoEncodeOpts::default()).expect("encode a test clip");
    path
}

#[test]
fn a_clip_within_its_budget_reads_and_one_over_it_is_refused() {
    let path = clip("budget", 6, 64, 48);
    let blob = clip_from_untrusted_file(&path, 100, 6 * 64 * 48).expect("exactly at the budget");
    assert_eq!(blob.meta["frames"], 6);
    let refusal = clip_from_untrusted_file(&path, 100, 5 * 64 * 48).unwrap_err();
    assert!(refusal.contains("pixels"), "{refusal}");
    let refusal = clip_from_untrusted_file(&path, 5, u64::MAX).unwrap_err();
    assert!(refusal.contains("5 frames"), "{refusal}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn bytes_brain_does_not_decode_itself_are_refused_without_naming_the_file() {
    // A playlist is what an external demuxer would follow off the machine.
    let path = std::env::temp_dir().join(format!("brain-untrusted-{}-playlist.mp4", std::process::id()));
    std::fs::write(&path, b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://192.0.2.1/x.ts\n").unwrap();
    let refusal = clip_from_untrusted_file(&path, 100, u64::MAX).unwrap_err();
    assert!(!refusal.contains(path.to_str().unwrap()) && !refusal.contains("/tmp"), "a refusal names no path: {refusal}");
    assert!(refusal.contains("AV1") || refusal.contains("malformed"), "{refusal}");
    std::fs::remove_file(&path).ok();
}

/// A clip only ffmpeg can read (H.264) is read by the operator's
/// `clip_from_file`, and refused when a client sent it.
#[test]
fn a_clip_only_ffmpeg_reads_never_reaches_ffmpeg_from_a_request() {
    let path = std::env::temp_dir().join(format!("brain-untrusted-{}-h264.mp4", std::process::id()));
    let made = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=0.25", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&path)
        .status();
    match made {
        Ok(status) if status.success() => {}
        _ => return brain_testutil::skip("ffmpeg with libx264 makes the H.264 clip and is not runnable here"),
    }
    imaging::video::clip_from_file(&path, 100).expect("an operator's own file is read through ffmpeg");
    let refusal = clip_from_untrusted_file(&path, 100, u64::MAX).unwrap_err();
    assert!(refusal.contains("not AV1"), "{refusal}");
    std::fs::remove_file(&path).ok();
}
