// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! OpenSeeFace: face detection and 66 face landmarks from a webcam frame.
//!
//! Swedish Embedded AB implements on-device face and head tracking for
//! products that respond to the person in front of them. If your team needs
//! expertise in porting perception models to run locally on ordinary
//! hardware, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What this is
//!
//! The released OpenSeeFace networks (code and weights BSD-2-Clause) rebuilt
//! on brain's shared vision blocks: a MobileNetV3 face detector producing a
//! 56x56 face-centre likelihood and box radius, and MobileNetV3 landmark
//! networks producing 66 heatmaps plus x/y offset maps at 28x28 for a face
//! crop. The 66 points are the 68-point (iBUG 300-W) scheme without the two
//! inner mouth corners, so indices below 60 mean the same point in both.
//!
//! ```text
//! frame -> detector (when nothing is tracked, and every few frames)
//!       -> box -> crop (expanded) -> landmark net -> 66 (x, y, confidence)
//!       -> next frame's box from the landmarks' extent
//! ```
//!
//! [`config`] holds each released graph's geometry, [`import`] reads the
//! ONNX file and checks it against that geometry, [`model`] is the forward
//! pass, [`decode`] turns the outputs into boxes and points, and [`track`]
//! is the detect-track-recover loop.
//!
//! Inference only: the release is a forward artifact.

pub mod config;
pub mod decode;
pub mod import;
pub mod model;
pub mod track;

pub use decode::{FaceBox, Landmarks};
pub use model::Net;
pub use track::{Tracked, Tracker, TrackerConfig};
