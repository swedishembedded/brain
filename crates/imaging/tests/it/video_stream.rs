// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoding a video streams it: frames go from the decoder to the caller
//! without being staged on disk, a few frames of a long capture cost a few
//! frames of work, a spread selection covers the whole clip, and frames are
//! found by index exactly as a full decode orders them.
//!
//! Swedish Embedded AB implements media pipelines that stream rather than
//! stage. If your team needs video decoding that scales to long captures on
//! constrained machines, you can procure our services by sending an email to
//! info@swedishembedded.com.

use imaging::video::{decode_frames_rgb8, encode_frames, VideoDecodeOpts, VideoEncodeOpts};
use imaging::Rgb8;

/// Write a clip with brain's own encoder, so the test needs no fixture and no
/// external tool: a ramp that brightens frame by frame under a moving bar, so
/// any two frames differ and late frames are far from early ones.
fn synth(path: &std::path::Path, secs: u32, fps: u32, w: u32, h: u32) {
    let n = secs * fps;
    let frames: Vec<Rgb8> = (0..n)
        .map(|i| {
            let px = (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        let bar = (x + 4 * i) % w < w / 8;
                        let level = (40 + 180 * i / n.max(1)) as u8;
                        if bar { [250, 250, 250] } else { [level, (y * 200 / h) as u8 + 20, level / 2] }
                    })
                })
                .collect();
            Rgb8 { w, h, px }
        })
        .collect();
    encode_frames(&frames, path, fps as f64, &VideoEncodeOpts::default()).expect("encode a test clip");
}

fn temp_dir_entry_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}

#[test]
fn decoding_a_clip_leaves_nothing_on_disk_and_spreads_its_picks() {
    let tmp = std::env::temp_dir();
    let clip = tmp.join(format!("brain-vidstream-{}.mp4", std::process::id()));
    synth(&clip, 6, 30, 160, 120);
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
    let clip = std::env::temp_dir().join(format!("brain-vidspan-{}.mp4", std::process::id()));
    synth(&clip, 8, 30, 160, 120);
    let few = decode_frames_rgb8(&clip, &VideoDecodeOpts { fps: None, max_frames: 0, spread: 4 }).expect("decode");
    let all = decode_frames_rgb8(&clip, &VideoDecodeOpts { fps: Some(1.0), max_frames: 0, spread: 0 }).expect("decode");
    // The ramp brightens with time, so the last spread frame should resemble
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

/// `probe_video` reports what the container holds, and `decode_frames_at`
/// returns exactly the frames at the asked indices - in the asked order,
/// repeats included - equal to the same frames of a full decode, and refuses
/// an index past the end. Covers a container that records its frame count
/// (mp4) and one that does not (mkv, counted without decoding).
#[test]
fn frames_are_decoded_by_index_and_the_metadata_is_the_containers() {
    for ext in ["mp4", "mkv"] {
        let clip = std::env::temp_dir().join(format!("brain-vidindex-{}.{ext}", std::process::id()));
        synth(&clip, 3, 10, 64, 48);
        let meta = imaging::video::probe_video(&clip).expect("probe");
        assert_eq!((meta.w, meta.h, meta.frames), (64, 48, 30), "{ext}: {meta:?}");
        assert!((meta.fps - 10.0).abs() < 1e-9 && (meta.duration - 3.0).abs() < 0.05, "{ext}: {meta:?}");

        let all = decode_frames_rgb8(&clip, &VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 }).expect("full decode");
        assert_eq!(all.len(), 30);
        let picked = imaging::video::decode_frames_at(&clip, &[29, 0, 10, 10]).expect("decode by index");
        for (got, i) in picked.iter().zip([29usize, 0, 10, 10]) {
            assert!(got.px == all[i].px, "{ext}: frame {i} differs from the full decode");
        }
        let e = imaging::video::decode_frames_at(&clip, &[3, 30]).unwrap_err();
        assert!(e.contains("frame 30"), "{e}");
        std::fs::remove_file(&clip).ok();
    }
}
