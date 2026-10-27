// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running object detectors on small
// edge devices for its clients. If your team needs expertise in deploying
// detection networks to NPUs and microcontroller-class SoCs then you can procure
// our services by sending an email to info@swedishembedded.com.

//! [`Yolov5Config`]: everything that fixes the shape of a YOLOv5 network and the
//! meaning of its outputs, carried in a checkpoint's `brain.config` header.
//!
//! The topology is not stored: it follows from the published YOLOv5 v6
//! description (a CSP backbone, an SPP-fast block, a PAN neck and a three-scale
//! anchor head) scaled by `depth_multiple` / `width_multiple`
//! ([`crate::graph::Graph::build`]). What the config does carry is what a graph
//! cannot derive: the class count, the activation the checkpoint was trained
//! with, and the anchors the head's outputs are expressed against.
//!
//! The header is validated at the boundary, field by field and strictly: an
//! unknown key, a missing key, or a value outside what the graph builder
//! supports is an [`Error::Config`] naming the field - never a default.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;

/// The value of the `arch` header field. It makes a checkpoint of another
/// architecture fail to classify as this one even if its other keys happen to
/// parse.
pub const ARCH: &str = "yolov5";

/// Outputs per anchor beyond the class scores: `x, y, w, h, objectness`.
pub const BOX_FIELDS: u32 = 5;

/// Anchors per detection scale.
pub const ANCHORS_PER_SCALE: usize = 3;

/// The detection scales (P3, P4, P5), by stride. Fixed by the architecture.
pub const STRIDES: [u32; 3] = [8, 16, 32];

/// The activation after every convolution of the backbone and neck.
///
/// Stock YOLOv5 is trained with SiLU; Rockchip's NPU export (the first
/// checkpoint brain imports) retrains with ReLU because the NPU has no SiLU.
/// The two share every tensor name and shape, so this is a switch, not a second
/// architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Silu,
}

/// The shape-determining configuration of one YOLOv5 network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Yolov5Config {
    /// Always [`ARCH`].
    pub arch: String,
    /// Number of classes.
    pub nc: u32,
    /// Scales the number of bottlenecks per C3 block (`n = max(round(n * depth), 1)`).
    pub depth_multiple: f32,
    /// Scales every channel count, rounded up to a multiple of 8.
    pub width_multiple: f32,
    pub activation: Activation,
    /// Anchor `(w, h)` pairs in input pixels, flattened per scale:
    /// `anchors[scale] = [w0, h0, w1, h1, w2, h2]`, scales ordered P3, P4, P5.
    pub anchors: [[f32; 6]; 3],
    /// The input side (a multiple of 32) the checkpoint was exported at. A
    /// default for [`crate::Yolov5::load`]: the network is fully convolutional
    /// and runs at any multiple of 32.
    pub input: u32,
}

/// The COCO anchors every released YOLOv5 checkpoint uses, in pixels, P3 first.
pub const COCO_ANCHORS: [[f32; 6]; 3] = [
    [10.0, 13.0, 16.0, 30.0, 33.0, 23.0],
    [30.0, 61.0, 62.0, 45.0, 59.0, 119.0],
    [116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
];

impl Yolov5Config {
    /// YOLOv5s (`depth 0.33`, `width 0.50`) on the 80 COCO classes with the
    /// COCO anchors, at the 640 input the model zoo exports.
    pub fn yolov5s(activation: Activation) -> Yolov5Config {
        Yolov5Config {
            arch: ARCH.to_string(),
            nc: 80,
            depth_multiple: 0.33,
            width_multiple: 0.50,
            activation,
            anchors: COCO_ANCHORS,
            input: 640,
        }
    }

    /// Channels of one detection head output: `anchors * (5 + nc)`.
    pub fn head_channels(&self) -> u32 {
        ANCHORS_PER_SCALE as u32 * (BOX_FIELDS + self.nc)
    }

    /// Parse and validate a `brain.config` header value.
    pub fn from_json(v: &Value) -> Result<Yolov5Config, Error> {
        let cfg: Yolov5Config = serde_json::from_value(v.clone()).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// The header value [`Yolov5Config::from_json`] reads back.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("a Yolov5Config always serializes")
    }

    fn validate(&self) -> Result<(), Error> {
        let bad = |field: &str, why: String| Err(Error::Config(format!("{field}: {why}")));
        if self.arch != ARCH {
            return bad("arch", format!("expected {ARCH:?}, got {:?}", self.arch));
        }
        if self.nc == 0 || self.nc > 4096 {
            return bad("nc", format!("must be in 1..=4096, got {}", self.nc));
        }
        for (field, v) in [("depth_multiple", self.depth_multiple), ("width_multiple", self.width_multiple)] {
            if !(v.is_finite() && v > 0.0 && v <= 8.0) {
                return bad(field, format!("must be a finite number in (0, 8], got {v}"));
            }
        }
        if self.anchors.iter().flatten().any(|a| !(a.is_finite() && *a > 0.0)) {
            return bad("anchors", "every anchor side must be a finite positive number".to_string());
        }
        let max_stride = STRIDES[2];
        if self.input == 0 || !self.input.is_multiple_of(max_stride) {
            return bad("input", format!("must be a positive multiple of {max_stride}, got {}", self.input));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_is_lossless() {
        let cfg = Yolov5Config::yolov5s(Activation::Relu);
        assert_eq!(Yolov5Config::from_json(&cfg.to_json()).unwrap(), cfg);
    }

    #[test]
    fn a_header_is_rejected_by_field_not_defaulted() {
        let good = Yolov5Config::yolov5s(Activation::Silu).to_json();

        let mut unknown = good.clone();
        unknown["strides"] = serde_json::json!([8, 16, 32]);
        assert!(matches!(Yolov5Config::from_json(&unknown), Err(Error::Config(m)) if m.contains("strides")));

        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove("anchors");
        assert!(matches!(Yolov5Config::from_json(&missing), Err(Error::Config(m)) if m.contains("anchors")));

        let mut foreign = good.clone();
        foreign["arch"] = serde_json::json!("yolov8");
        assert!(matches!(Yolov5Config::from_json(&foreign), Err(Error::Config(m)) if m.starts_with("arch")));

        let mut ragged = good.clone();
        ragged["input"] = serde_json::json!(330);
        assert!(matches!(Yolov5Config::from_json(&ragged), Err(Error::Config(m)) if m.starts_with("input")));

        // An empty object (another model's header) must not "succeed with defaults".
        assert!(Yolov5Config::from_json(&serde_json::json!({})).is_err());
    }
}
