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
use crate::infer::Infer;
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
    detector: Box<dyn Infer>,
    landmarks: Box<dyn Infer>,
    config: TrackerConfig,
    tracked: Option<FaceBox>,
}

impl Tracker {
    /// Load the detector and the landmark network `landmarks` (one of
    /// [`crate::config::LANDMARKS_FAST`], [`crate::config::LANDMARKS_BEST`])
    /// from the directory holding the released `.onnx` files, on `device`.
    pub fn load(device: &Gpu, dir: &Path, landmarks: NetConfig, config: TrackerConfig) -> Result<Tracker, String> {
        Ok(Tracker::new(Box::new(Net::load(device, dir, DETECTOR)?), Box::new(Net::load(device, dir, landmarks)?), config))
    }

    /// Track with any two executors - brain's kernels, OpenVINO, or one of
    /// each. `detector` must run [`DETECTOR`].
    pub fn new(detector: Box<dyn Infer>, landmarks: Box<dyn Infer>, config: TrackerConfig) -> Tracker {
        assert!(detector.config().peak_pool, "Tracker::new: the first network must be the detector");
        Tracker { detector, landmarks, config, tracked: None }
    }

    /// Where the two networks run.
    pub fn devices(&self) -> (String, String) {
        (self.detector.device(), self.landmarks.device())
    }

    /// Forget the tracked face; the next frame detects.
    pub fn reset(&mut self) {
        self.tracked = None;
    }

    /// The most confident face in an RGB frame, if any.
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<FaceBox>, String> {
        let o = self.detector.infer(&decode::detector_input(rgb, w, h))?;
        let pooled = o.pooled.ok_or("the detector returned no peak pool")?;
        Ok(decode::strongest_face(&o.map, &pooled, w, h, self.config.detection_threshold))
    }

    /// The landmarks inside `b`'s crop, if they are a confident face.
    pub fn landmarks_in(&mut self, rgb: &[u8], w: u32, h: u32, b: &FaceBox) -> Result<Option<(Landmarks, [i32; 4])>, String> {
        let Some(crop) = decode::crop_for(b, w, h) else { return Ok(None) };
        let o = self.landmarks.infer(&decode::crop_input(rgb, w, crop))?;
        let lm = decode::landmarks(&o.map, crop);
        Ok((lm.score() >= self.config.landmark_threshold).then_some((lm, crop)))
    }

    /// Track through one RGB frame (`w * h * 3` bytes, row-major). `Ok(None)`
    /// is "no face"; an error is an executor failure.
    pub fn track(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<Tracked>, String> {
        assert_eq!(rgb.len(), (w * h * 3) as usize, "track: a {w}x{h} RGB frame is {} bytes", w * h * 3);
        if let Some(b) = self.tracked {
            if let Some((landmarks, crop)) = self.landmarks_in(rgb, w, h, &b)? {
                self.tracked = Some(landmarks.extent());
                return Ok(Some(Tracked { landmarks, crop, detected: false }));
            }
        }
        self.tracked = None;
        let Some(b) = self.detect(rgb, w, h)? else { return Ok(None) };
        let Some((landmarks, crop)) = self.landmarks_in(rgb, w, h, &b)? else { return Ok(None) };
        self.tracked = Some(landmarks.extent());
        Ok(Some(Tracked { landmarks, crop, detected: true }))
    }
}
