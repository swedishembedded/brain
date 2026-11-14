// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What brain writes, brain reads back: every container, frame for frame, at
//! the frame rate and size it was written with, within the fidelity of the
//! default quality; with sound, the file still reads; another container or
//! codec is `Unsupported` (so a caller can hand it to another decoder). When
//! an `ffprobe` is on PATH, it must agree: an AV1 stream, and an Opus one
//! when a sound track was written.

use std::path::{Path, PathBuf};
use std::process::Command;

use videocodec::{decode, decode_from, probe, Audio, EncodeSettings, Error, VideoWriter};

const FPS: f64 = 24.0;
const FRAMES: usize = 30;
/// Close coding, so a decode that converts colour or places planes wrongly
/// (which costs 10 dB and more) cannot hide under the coding loss.
const SETTINGS: EncodeSettings = EncodeSettings { quantizer: 30, speed: 9, key_frame_interval: Some(12), tiles: 0 };
/// Against what a lossless codec would return: the frame through 4:2:0 and
/// back (the pattern's saturated hard edges cost 4:2:0 itself 31-36 dB, so
/// the frame as written would measure the colour format, not the codec).
/// Inter frames of a few blocks measure 34-40 dB here at any parity of size.
const MIN_PSNR: f64 = 34.0;

fn through_420(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    use videocodec::color::{rgb_to_yuv420, yuv_to_rgb, Colour, Matrix, Planes, Yuv420};
    let yuv = rgb_to_yuv420(rgb, w, h);
    let (cw, _) = Yuv420::chroma_size(w, h);
    let widen = |v: &[u8]| v.iter().map(|&x| x as u16).collect::<Vec<u16>>();
    let (y, u, v) = (widen(&yuv.y), widen(&yuv.u), widen(&yuv.v));
    let planes = Planes { w, h, bits: 8, y: (&y, w), uv: Some(((&u, cw), (&v, cw))), chroma_shift: (1, 1) };
    yuv_to_rgb(&planes, Colour { matrix: Matrix::BT709, full_range: false })
}

/// A moving gradient with a bright square: smooth regions, an edge, motion.
fn frame(i: usize, w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let inside = (x + 3 * i) % w < w / 4 && y > h / 4 && y < h / 2;
            if inside {
                px.extend([240, 230, 60]);
            } else {
                px.extend([(x * 200 / w) as u8 + 20, (y * 180 / h) as u8 + 30, ((x + y + 2 * i) % 160) as u8 + 40]);
            }
        }
    }
    px
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn stereo(seconds: f64, rate: u32) -> Audio {
    let n = (seconds * rate as f64) as usize;
    let tone = |hz: f64| (0..n).map(|i| (0.3 * (2.0 * std::f64::consts::PI * hz * i as f64 / rate as f64).sin()) as f32).collect();
    Audio { channels: vec![tone(440.0), tone(660.0)], sample_rate: rate }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-videocodec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn write(path: &Path, w: usize, h: usize, audio: Option<&Audio>) -> Vec<Vec<u8>> {
    let frames: Vec<Vec<u8>> = (0..FRAMES).map(|i| frame(i, w, h)).collect();
    let mut writer = VideoWriter::create(path, w as u32, h as u32, FPS, SETTINGS).expect("a writer");
    for f in &frames {
        writer.push(f).expect("a frame");
    }
    writer.finish(audio).expect("the file");
    frames
}

fn read_back(path: &Path, frames: &[Vec<u8>], w: usize, h: usize) {
    let info = probe(path).expect("probe");
    assert_eq!((info.w, info.h, info.frames, info.codec.as_str()), (w as u32, h as u32, Some(FRAMES as u64), "av1"), "{}", path.display());
    assert!((info.fps.expect("a frame rate") - FPS).abs() < 1e-6, "{info:?}");
    assert!((info.duration.expect("a duration") - FRAMES as f64 / FPS).abs() < 0.05, "{info:?}");

    let mut got = Vec::new();
    decode(path, &mut |f| {
        got.push(f);
        true
    })
    .expect("decode");
    assert_eq!(got.len(), FRAMES, "{}", path.display());
    for (i, f) in got.iter().enumerate() {
        assert_eq!((f.index, f.w, f.h), (i as u64, w as u32, h as u32));
        assert!((f.time - i as f64 / FPS).abs() < 2e-3, "frame {i} at {} s", f.time);
        let q = psnr(&f.rgb, &through_420(&frames[i], w, h));
        assert!(q > MIN_PSNR, "{}: frame {i} at {q:.1} dB", path.display());
    }
}

#[test]
fn every_container_reads_back_what_was_written() {
    for name in ["clip.webm", "clip.mkv", "clip.mp4"] {
        let path = scratch(name);
        let frames = write(&path, 96, 64, None);
        read_back(&path, &frames, 96, 64);
    }
}

#[test]
fn odd_sizes_keep_their_size() {
    let path = scratch("odd.mp4");
    let frames = write(&path, 65, 33, None);
    read_back(&path, &frames, 65, 33);
}

#[test]
fn a_sound_track_rides_along_in_both_containers() {
    let audio = stereo(FRAMES as f64 / FPS, 16_000);
    for name in ["sound.webm", "sound.mp4"] {
        let path = scratch(name);
        let frames = write(&path, 64, 48, Some(&audio));
        read_back(&path, &frames, 64, 48);
        ffprobe_agrees(&path, true);
    }
}

#[test]
fn stopping_early_decodes_no_further() {
    let path = scratch("early.webm");
    write(&path, 64, 48, None);
    let mut seen = 0;
    decode(&path, &mut |_| {
        seen += 1;
        seen < 5
    })
    .expect("decode");
    assert_eq!(seen, 5);
}

/// A read that starts at frame `first` decodes from the key frame at or before it, not from the start: its frames
/// are bit for bit the frames a whole decode gives from `first` on, and the decoder handles at most one key-frame
/// interval of units before `first` (SETTINGS forces a key frame at least every 12 frames).
#[test]
fn a_late_start_decodes_from_the_preceding_key_frame() {
    for name in ["seek.webm", "seek.mp4"] {
        let path = scratch(name);
        write(&path, 64, 48, None);
        let mut all = Vec::new();
        decode(&path, &mut |f| {
            all.push(f.rgb);
            true
        })
        .expect("decode");
        for first in [0u64, 5, 12, 13, 25, FRAMES as u64 - 1] {
            let mut got = Vec::new();
            let done = decode_from(&path, first, &mut |f| {
                got.push((f.index, f.rgb));
                true
            })
            .expect("decode_from");
            let want: Vec<(u64, Vec<u8>)> = all.iter().enumerate().skip(first as usize).map(|(i, px)| (i as u64, px.clone())).collect();
            assert!(got == want, "{name}: frames from {first} differ from the whole decode's");
            let extra = done.units_decoded - (FRAMES as u64 - first);
            assert!(extra < SETTINGS.key_frame_interval.expect("the fixture forces key frames"), "{name}: from {first} the decoder handled {extra} units before it");
        }
    }
}

#[test]
fn other_files_are_unsupported_not_malformed() {
    let path = scratch("not-video.png");
    std::fs::write(&path, b"\x89PNG\r\n\x1a\n0123456789abcdef").unwrap();
    assert!(matches!(probe(&path), Err(Error::Unsupported(_))), "{:?}", probe(&path));
    let err = VideoWriter::create(&scratch("x.avi"), 8, 8, FPS, EncodeSettings::default()).err().expect("an .avi is refused");
    assert!(err.to_string().contains(".webm"), "{err}");
}

/// When an `ffprobe` is installed, the files are what any player expects.
fn ffprobe_agrees(path: &Path, with_sound: bool) {
    let Ok(out) = Command::new("ffprobe").args(["-v", "error", "-show_entries", "stream=codec_name,codec_type", "-of", "csv=p=0"]).arg(path).output() else {
        brain_testutil::skip_unavailable("ffprobe not installed");
        return;
    };
    assert!(out.status.success(), "ffprobe rejects {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    let streams = String::from_utf8_lossy(&out.stdout);
    assert!(streams.contains("av1,video"), "{}: {streams}", path.display());
    assert_eq!(streams.contains("opus,audio"), with_sound, "{}: {streams}", path.display());
}

/// **Quality does not pulse.** Under rav1e's fixed quantizer a key frame is
/// coded about a quarter finer than the frames between, and those settle a
/// few dB lower within a few dozen frames, so a key frame forced into the
/// middle of a shot reads as the picture sharpening all at once - measured
/// on a real 313-frame generated clip, +3.9 dB at each of frames 120 and 240
/// against at most 0.84 dB between any two frames without them. The default
/// settings force none after the first frame: a shot longer than the old
/// 120-frame interval, with no cut in it, is one key frame.
#[test]
fn default_settings_force_no_key_frame_into_a_shot() {
    let (w, h, n) = (96usize, 64usize, 200usize);
    let mut enc = videocodec::av1::Encoder::new(w, h, (24, 1), EncodeSettings::default()).expect("an encoder");
    let mut packets = Vec::new();
    for i in 0..n {
        packets.extend(enc.push(&frame(i, w, h)).expect("a frame"));
    }
    packets.extend(enc.finish().expect("the last units"));
    assert_eq!(packets.len(), n);
    let keys: Vec<u64> = packets.iter().filter(|p| p.key).map(|p| p.frame).collect();
    assert_eq!(keys, vec![0], "key frames inside one shot");

    // A caller who asks for an interval gets it.
    let mut enc = videocodec::av1::Encoder::new(w, h, (24, 1), EncodeSettings { key_frame_interval: Some(60), ..EncodeSettings::default() }).expect("an encoder");
    let mut packets = Vec::new();
    for i in 0..n {
        packets.extend(enc.push(&frame(i, w, h)).expect("a frame"));
    }
    packets.extend(enc.finish().expect("the last units"));
    assert!(packets.iter().filter(|p| p.key).count() >= n / 60, "an asked-for interval is honoured");
}
