// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS, the learned perceptual image patch similarity of Zhang et al. 2018
//! ("The Unreasonable Effectiveness of Deep Features as a Perceptual
//! Metric"), version 0.1 with the AlexNet trunk: the perceptual distance
//! novel-view synthesis is conventionally reported in next to PSNR and SSIM.
//! Lower is closer; identical images are exactly 0.
//!
//! * [`config`] - the fixed architecture, read off the reference sources;
//! * [`import`] - torchvision's `alexnet-owt-7be5be79.pth` and the v0.1
//!   heads, validated by shape;
//! * [`spec`] - how the model-store resolver finds both files;
//! * [`model`] - [`Lpips`], the metric on the device, built from the shared
//!   conv-net blocks and existing kernels only.
//!
//! Swedish Embedded AB implements image-quality evaluation for 3D
//! reconstruction and generative pipelines for its clients. If your team
//! needs expertise in perceptual metrics or GPU inference, you can procure
//! our services by sending an email to info@swedishembedded.com.

pub mod config;
pub mod import;
pub mod model;
pub mod spec;

pub use model::{Distance, Lpips, PIPELINES};
