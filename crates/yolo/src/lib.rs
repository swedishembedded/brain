// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8-style anchor-free object detector (P0 skeleton).
//!
//! - [`config`] — [`YoloConfig`] + variant presets, implementing
//!   [`model::ModelConfig`].
//!
//! Backbone/neck/head forward+backward, DFL box decode, and the loss come in
//! later phases (P2/P3); this phase wires up only the config so the crate
//! compiles and slots into the workspace.

pub mod assign;
pub mod blocks;
pub mod boxmath;
pub mod config;
pub mod head;
pub mod infer;
pub mod init;
pub mod loss;
pub mod model;
pub mod net;
pub mod nms;

pub use boxmath::Letterbox;
pub use config::YoloConfig;
pub use init::init_model as init_weights;
pub use model::{GtBox, LossMode, Yolo};
pub use net::{Ctx, Shape};
pub use nms::{nms, nms_agnostic, Detection};

#[cfg(test)]
mod tests {
    use super::*;
    use ::model::ModelConfig;

    #[test]
    fn tiny_config_json_roundtrip() {
        let cfg = YoloConfig::tiny(2);
        let back = YoloConfig::from_json(&cfg.to_json());
        assert_eq!(back.input, cfg.input);
        assert_eq!(back.nc, 2);
        assert_eq!(back.reg_max, cfg.reg_max);
        assert_eq!(back.channels, cfg.channels);
        assert_eq!(back.strides, cfg.strides);
        // ModelConfig seam: vocab carries nc, block_size carries input.
        assert_eq!(back.vocab(), 2);
        assert_eq!(back.block_size(), cfg.input);
    }

    #[test]
    fn yolov8n_has_80_classes() {
        assert_eq!(YoloConfig::yolov8n().nc, 80);
    }
}
