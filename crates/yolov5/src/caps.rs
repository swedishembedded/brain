// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv5's capability manifest: the same one-action `detect` surface as every
//! detector, declared statically (no weights needed). It is what
//! `brain::DetectionPipeline::capabilities` reflects for a YOLOv5 checkpoint.
//! A serving `Provider` over it is not part of this crate yet.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

/// The model id this manifest is registered under.
pub const MODEL: &str = "brain/yolov5";

pub fn manifest() -> Manifest {
    let detect = ActionSpec::new("detect", "detect objects in an image (letterbox -> forward -> anchor decode -> NMS)")
        .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format YOLOv5 checkpoint (.safetensors)").required())
        .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold (objectness x class score)").default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").default(json!(0.45)))
        .input(BlobSpec::new("image", Media::Image, "the image to run detection on").required())
        .output(BlobSpec::new("detections", Media::Text, "JSON array of {bbox:[x1,y1,x2,y2], conf, class} in image coords"));
    Manifest::new(MODEL, "YOLOv5 anchor-based object detector - single-image detection.", vec![detect])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declares_the_one_detect_action_with_the_shared_defaults() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        assert_eq!(m.actions[0].name, "detect");
    }
}
