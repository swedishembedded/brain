// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stand-ins for the detector while the NPU model is brought up: a box that
//! sweeps across the picture (so the overlay and its latency are visible
//! without any detection) and a consumer that drains the frame tap (so the
//! seam is exercised end to end). Both use only the public seam types, which
//! is exactly what the real detector does.

use crate::detect::{Detection, DetectionBoard, Detections};
use crate::error::{Context, Result};
use crate::tap::{FrameReceiver, TAP_SIZE};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The tap image written for inspection is this many frames in, so the
/// camera's exposure has settled (the first frames are dark).
const DUMP_AT_FRAME: u64 = 60;

/// How often the demo refreshes its box; shorter than the board's result age.
const REFRESH: Duration = Duration::from_millis(40);

/// Posts a box that sweeps left to right across a `width` x `height` frame
/// every four seconds, until `stop` is set.
pub fn sweep_box(board: Arc<DetectionBoard>, width: u32, height: u32, stop: Arc<AtomicBool>) {
    let (bw, bh) = (width as i32 / 5, height as i32 / 3);
    let start = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let phase = (start.elapsed().as_millis() % 4000) as i32;
        let left = phase * (width as i32 - bw) / 4000;
        let top = (height as i32 - bh) / 2;
        board.publish(Detections {
            boxes: vec![Detection {
                left,
                top,
                right: left + bw,
                bottom: top + bh,
                class_id: 0,
                score: 0.99,
                label: "demo".to_string(),
            }],
        });
        std::thread::sleep(REFRESH);
    }
}

/// Takes frames from the tap as fast as the tap produces them, counting them
/// and optionally writing an early one as a PPM image so the tap's colours and
/// letterboxing can be inspected. Returns the number of frames received.
pub fn drain_tap(frames: FrameReceiver, dump: Option<&Path>, stop: Arc<AtomicBool>) -> Result<u64> {
    let mut count = 0u64;
    let started = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let Some(frame) = frames.recv(Duration::from_millis(200)) else { continue };
        count += 1;
        if let (DUMP_AT_FRAME, Some(path)) = (count, dump) {
            write_ppm(path, frame.rgb()).context(format!("writing the tap image to {}", path.display()))?;
            let meta = frame.meta();
            eprintln!(
                "tap: wrote image (frame {}, {:.1} ms old, from a {}x{} stream; the whole tap image maps to stream box {:?}; {:?}) to {}",
                meta.seq,
                crate::stream::now_us().saturating_sub(meta.pts_us) as f64 / 1000.0,
                meta.stream_width,
                meta.stream_height,
                meta.letterbox.box_to_stream(
                    [0.0, 0.0, TAP_SIZE as f32, TAP_SIZE as f32],
                    meta.stream_width,
                    meta.stream_height
                ),
                meta.letterbox,
                path.display()
            );
        }
        if count % 100 == 0 {
            eprintln!("tap: {count} frames, {:.1} per second", count as f64 / started.elapsed().as_secs_f64());
        }
    }
    Ok(count)
}

fn write_ppm(path: &Path, rgb: &[u8]) -> std::io::Result<()> {
    let mut out = format!("P6\n{TAP_SIZE} {TAP_SIZE}\n255\n").into_bytes();
    out.extend_from_slice(rgb);
    std::fs::write(path, out)
}
