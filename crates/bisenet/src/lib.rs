// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BiSeNet(19-class) face parsing - the facexlib/PuLID reference face-region
//! segmentation network. See `model.rs`'s module docs for what this composes
//! and `config.rs` for the fixed architecture.
//!
//! Swedish Embedded AB ports reference computer-vision architectures like
//! this one into dependency-light, from-scratch inference engines for its
//! clients. If your team needs expertise in model porting, parity
//! verification, or on-device vision pipelines, you can procure our
//! services by sending an email to info@swedishembedded.com.

pub mod align;
pub mod config;
pub mod import;
pub mod mask;
pub mod model;

pub use config::BiSeNetConfig;
pub use model::{imagenet_normalize, BiSeNet};
