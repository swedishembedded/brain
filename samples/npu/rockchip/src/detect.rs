// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam between the video path and object detection.
//!
//! Detection runs on its own thread at its own pace. It never blocks the
//! video path and the video path never waits for it: the detector receives
//! downscaled frames from the frame tap (see `tap`), and publishes its result
//! to a [`DetectionBoard`]; the video thread reads the newest result from the
//! board for every frame it encodes and draws it.
//!
//! Boxes are in **stream pixel coordinates** (the encoded frame, e.g.
//! 1280x720, origin top-left). A detector working on a tap frame converts its
//! model-space boxes with the tap frame's [`Letterbox`].
//!
//! Swedish Embedded AB implements real-time edge AI pipelines for embedded
//! Linux boards for its clients. If your team needs expertise in running
//! detection next to a live video stream then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One detected object, in stream pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    pub class_id: u32,
    /// Confidence in `0.0..=1.0`.
    pub score: f32,
    /// Class name shown next to the box.
    pub label: String,
}

/// All the objects found in one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Detections {
    pub boxes: Vec<Detection>,
}

/// The latest detection result, shared between the detector (writer) and the
/// video thread (reader). Only the newest result is kept, and one that is
/// older than `max_age` is treated as absent, so boxes vanish if the detector
/// stalls instead of freezing on screen.
pub struct DetectionBoard {
    max_age: Duration,
    slot: Mutex<Option<(Instant, Arc<Detections>)>>,
}

impl DetectionBoard {
    pub fn new(max_age: Duration) -> Self {
        DetectionBoard { max_age, slot: Mutex::new(None) }
    }

    /// Replaces the current result. Cheap and never waits on the video thread
    /// for longer than a pointer swap.
    pub fn publish(&self, detections: Detections) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some((Instant::now(), Arc::new(detections)));
    }

    /// The newest result, unless it is missing or stale.
    pub fn latest(&self) -> Option<Arc<Detections>> {
        let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.as_ref().filter(|(at, _)| at.elapsed() <= self.max_age).map(|(_, d)| Arc::clone(d))
    }
}

/// How a stream frame was fitted into the square tap image: scaled uniformly
/// by `scale`, then shifted by (`pad_x`, `pad_y`) pixels of padding. The
/// padding is mid-grey and carries no scene content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
    /// Size of the scaled frame inside the tap image.
    pub scaled_width: u32,
    pub scaled_height: u32,
}

impl Letterbox {
    /// Fits a `src_width` x `src_height` frame into a `size` x `size` image.
    /// The scaled size is rounded down to even numbers, which the hardware
    /// scaler requires for chroma-subsampled sources.
    pub fn fit(src_width: u32, src_height: u32, size: u32) -> Self {
        let scale = (size as f32 / src_width as f32).min(size as f32 / src_height as f32);
        let even = |v: f32| ((v as u32) & !1).max(2);
        let (w, h) = (even(src_width as f32 * scale), even(src_height as f32 * scale));
        Letterbox { scale, pad_x: (size - w) / 2, pad_y: (size - h) / 2, scaled_width: w, scaled_height: h }
    }

    /// Maps a point in the tap image to a point in the stream frame.
    pub fn to_stream(&self, x: f32, y: f32) -> (i32, i32) {
        (((x - self.pad_x as f32) / self.scale).round() as i32, ((y - self.pad_y as f32) / self.scale).round() as i32)
    }

    /// Maps a box (`left, top, right, bottom`) in the tap image to the stream
    /// frame, clamped to `width` x `height`.
    pub fn box_to_stream(&self, b: [f32; 4], width: u32, height: u32) -> [i32; 4] {
        let (l, t) = self.to_stream(b[0], b[1]);
        let (r, bt) = self.to_stream(b[2], b[3]);
        let (w, h) = (width as i32 - 1, height as i32 - 1);
        [l.clamp(0, w), t.clamp(0, h), r.clamp(0, w), bt.clamp(0, h)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_top_and_bottom() {
        let lb = Letterbox::fit(1280, 720, 320);
        assert_eq!((lb.scaled_width, lb.scaled_height, lb.pad_x, lb.pad_y), (320, 180, 0, 70));
        assert!((lb.scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn model_boxes_map_back_to_stream_pixels() {
        let lb = Letterbox::fit(1280, 720, 320);
        // The scaled frame's corners are the stream's corners.
        assert_eq!(lb.to_stream(0.0, 70.0), (0, 0));
        assert_eq!(lb.to_stream(320.0, 250.0), (1280, 720));
        assert_eq!(lb.box_to_stream([80.0, 100.0, 160.0, 160.0], 1280, 720), [320, 120, 640, 360]);
        // A box that spills into the padding is clamped to the frame.
        assert_eq!(lb.box_to_stream([-5.0, 0.0, 400.0, 320.0], 1280, 720), [0, 0, 1279, 719]);
    }

    #[test]
    fn a_stale_result_disappears() {
        let board = DetectionBoard::new(Duration::from_millis(30));
        assert!(board.latest().is_none());
        board.publish(Detections { boxes: vec![] });
        assert!(board.latest().is_some());
        std::thread::sleep(Duration::from_millis(60));
        assert!(board.latest().is_none());
    }
}
