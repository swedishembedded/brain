// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The detect-track-recover loop for one face.
//!
//! The detector runs only when no face is tracked. While one is, each
//! frame's landmark crop comes from the previous frame's landmarks (their
//! extent, grown by the crop margins), which keeps the face centred and
//! scaled in the crop as it moves, and costs one small network per frame
//! instead of two. When the tracked crop no longer yields a confident face,
//! the same frame is re-detected before the face is declared lost, so a fast
//! move costs one extra detector pass rather than a dropped frame.
//!
//! One face, the most confident: a tracker that switched between viewers
//! would move the rendered eye from one person to another.

use std::path::Path;

use gpu_core::Gpu;

use crate::config::{NetConfig, DETECTOR};
use crate::decode::{self, FaceBox, Landmarks};
use crate::model::Net;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Detector peaks below this are not faces.
    pub detection_threshold: f32,
    /// A landmark result whose mean confidence is below this is not a face.
    pub landmark_threshold: f32,
}

impl Default for TrackerConfig {
    /// The reference tracker's defaults for these networks.
    fn default() -> TrackerConfig {
        TrackerConfig { detection_threshold: decode::DETECTION_THRESHOLD, landmark_threshold: 0.6 }
    }
}

/// What one frame produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Tracked {
    pub landmarks: Landmarks,
    /// The crop the landmarks were read from.
    pub crop: [i32; 4],
    /// Whether the detector ran this frame.
    pub detected: bool,
}

pub struct Tracker {
    detector: Net,
    landmarks: Net,
    config: TrackerConfig,
    tracked: Option<FaceBox>,
}

impl Tracker {
    /// Load the detector and the landmark network `landmarks` (one of
    /// [`crate::config::LANDMARKS_FAST`], [`crate::config::LANDMARKS_BEST`])
    /// from the directory holding the released `.onnx` files, on `device`.
    pub fn load(device: &Gpu, dir: &Path, landmarks: NetConfig, config: TrackerConfig) -> Result<Tracker, String> {
        Ok(Tracker { detector: Net::load(device, dir, DETECTOR)?, landmarks: Net::load(device, dir, landmarks)?, config, tracked: None })
    }

    /// Forget the tracked face; the next frame detects.
    pub fn reset(&mut self) {
        self.tracked = None;
    }

    /// The most confident face in an RGB frame, if any.
    pub fn detect(&self, rgb: &[u8], w: u32, h: u32) -> Option<FaceBox> {
        self.detector.forward(&decode::detector_input(rgb, w, h));
        let out = self.detector.read_output();
        let pooled = self.detector.read_pooled().expect("the detector has a peak pool");
        decode::strongest_face(&out, &pooled, w, h, self.config.detection_threshold)
    }

    /// The landmarks inside `b`'s crop, if they are a confident face.
    pub fn landmarks_in(&self, rgb: &[u8], w: u32, h: u32, b: &FaceBox) -> Option<(Landmarks, [i32; 4])> {
        let crop = decode::crop_for(b, w, h)?;
        self.landmarks.forward(&decode::crop_input(rgb, w, crop));
        let lm = decode::landmarks(&self.landmarks.read_output(), crop);
        (lm.score() >= self.config.landmark_threshold).then_some((lm, crop))
    }

    /// Track through one RGB frame (`w * h * 3` bytes, row-major).
    pub fn track(&mut self, rgb: &[u8], w: u32, h: u32) -> Option<Tracked> {
        assert_eq!(rgb.len(), (w * h * 3) as usize, "track: a {w}x{h} RGB frame is {} bytes", w * h * 3);
        if let Some(b) = self.tracked {
            if let Some((landmarks, crop)) = self.landmarks_in(rgb, w, h, &b) {
                self.tracked = Some(landmarks.extent());
                return Some(Tracked { landmarks, crop, detected: false });
            }
        }
        self.tracked = None;
        let b = self.detect(rgb, w, h)?;
        let (landmarks, crop) = self.landmarks_in(rgb, w, h, &b)?;
        self.tracked = Some(landmarks.extent());
        Some(Tracked { landmarks, crop, detected: true })
    }
}
