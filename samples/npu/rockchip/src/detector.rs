// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The detector thread: tap frames in, brain's `DetectionPipeline` on the NPU,
//! boxes out to the board the video thread draws from.
//!
//! The tap hands over the freshest letterboxed frame whenever this thread
//! asks, so detection runs as fast as the NPU allows and never queues: a frame
//! that arrives while an inference is running is simply not taken. Boxes come
//! back in tap coordinates and are mapped to the stream through the frame's
//! own letterbox.
//!
//! Swedish Embedded AB implements real-time edge AI pipelines for embedded
//! Linux boards for its clients. If your team needs expertise in running
//! detection next to a live video stream then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::detect::{Detection, DetectionBoard, Detections};
use crate::error::{Error, Result};
use crate::tap::{FrameReceiver, TAP_SIZE};

/// How long one wait for a tap frame lasts before the stop flag is checked.
const FRAME_WAIT: Duration = Duration::from_millis(200);
/// How often throughput and latency are reported.
const REPORT_EVERY: Duration = Duration::from_secs(5);

/// What the detector did over its run.
#[derive(Debug, Default)]
pub struct DetectorStats {
    pub frames: u64,
    pub detections: u64,
}

/// Run detection on tap frames until `stop` is set. With `dump`, the first
/// frame the detector sees is also written there, to inspect the tap.
pub fn run(pipeline: brain::DetectionPipeline, options: brain::DetectOptions, frames: FrameReceiver, board: Arc<DetectionBoard>, mut dump: Option<PathBuf>, stop: Arc<AtomicBool>) -> Result<DetectorStats> {
    let mut stats = DetectorStats::default();
    let mut window = Window::default();
    let mut last_seq = None;
    while !stop.load(Ordering::Acquire) {
        let Some(frame) = frames.recv(FRAME_WAIT) else { continue };
        let started = Instant::now();
        let meta = *frame.meta();
        let image = brain::Image::from_rgb8(TAP_SIZE, TAP_SIZE, frame.rgb().to_vec()).map_err(|e| Error::new(format!("tap frame: {e}")))?;
        // Release the tap buffer before the inference, so the next frame can be
        // converted while this one is detected.
        drop(frame);
        if let Some(path) = dump.take() {
            image.save(&path).map_err(|e| Error::new(format!("writing the tap image to {}: {e}", path.display())))?;
            eprintln!("detector: wrote its first input ({TAP_SIZE}x{TAP_SIZE}, {:?}) to {}", meta.letterbox, path.display());
        }
        let found = pipeline.detect_with(&image, options).map_err(|e| Error::new(format!("detection: {e}")))?;
        let boxes: Vec<Detection> = found
            .iter()
            .map(|d| {
                let [left, top, right, bottom] = meta.letterbox.box_to_stream([d.x1, d.y1, d.x2, d.y2], meta.stream_width, meta.stream_height);
                let label = pipeline.class_name(d.class).map_or_else(|| format!("class {}", d.class), str::to_string);
                Detection { left, top, right, bottom, class_id: d.class, score: d.confidence, label }
            })
            .collect();
        stats.frames += 1;
        stats.detections += boxes.len() as u64;
        board.publish(Detections { boxes });
        window.frames += 1;
        window.busy += started.elapsed();
        // How old the camera frame is when its boxes reach the overlay, and
        // how many camera frames went by between two inferences.
        window.age_us += crate::stream::now_us().saturating_sub(meta.pts_us);
        window.skipped += last_seq.map_or(0, |s: u64| meta.seq.saturating_sub(s + 1));
        last_seq = Some(meta.seq);
        if window.start.elapsed() >= REPORT_EVERY {
            window.report();
            window = Window::default();
        }
    }
    Ok(stats)
}

/// Throughput and latency over one reporting window.
struct Window {
    start: Instant,
    frames: u32,
    busy: Duration,
    age_us: u64,
    skipped: u64,
}

impl Default for Window {
    fn default() -> Self {
        Window { start: Instant::now(), frames: 0, busy: Duration::ZERO, age_us: 0, skipped: 0 }
    }
}

impl Window {
    fn report(&self) {
        let n = f64::from(self.frames.max(1));
        eprintln!(
            "detector: {:.1} inferences/s, {:.1} ms per inference, boxes {:.1} ms after capture, {:.1} camera frames between inferences",
            f64::from(self.frames) / self.start.elapsed().as_secs_f64(),
            self.busy.as_secs_f64() * 1e3 / n,
            self.age_us as f64 / 1e3 / n,
            self.skipped as f64 / n
        );
    }
}
