// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR ("Scaling Up to Excellence", CVPR 2024) - photo-realistic blind image
//! restoration, driven by a **frozen SDXL 1.0 base UNet** plus a 1.24B control
//! trunk (`GLVControl`, a hand-written copy of SDXL's encoder + middle block)
//! and 12 adaptor modules (10 `ZeroSFT` + 2 `ZeroCrossAttn`).
//!
//! This crate started as an id-reservation placeholder, fixing the crate
//! directory name, the package name, the `brain supir <verb>` CLI word, and
//! this page's own filename in one place before anything else was written.
//!
//! ## Status
//!
//! The seams (`vae::blocks::skipfuse::SkipFuse`, `Op::Mix`,
//! `diffusion::restore`) and this crate's forward - [`config`] (the tensor
//! manifest), [`import`] (two-way checkpoint coverage), [`trunk`]
//! (`GLVControl`), [`adaptors`] (the 12 `ZeroSFT`/`ZeroCrossAttn` modules)
//! and [`model`] (trunk + adaptors + the frozen UNet, one graph) - are
//! implemented and weight-free-gated. `pipeline.rs` (the full restoration
//! loop: dual encode, dual-CLIP conditioning, `RestoreEDMSampler`, colour
//! fix) and real-checkpoint parity are open.
//!
//! ## Licence
//!
//! The SUPIR weights are released under the SUPIR Software License Agreement
//! (copyright 2024 SupPixel Pty Ltd): **non-commercial only**, and the
//! definition of commercial use expressly includes SaaS deployment and using
//! outputs as ML training data. There is no official HuggingFace repo. This
//! crate ships no `default_ref`/auto-fetch for that reason - a user points
//! brain at weights they obtained themselves, at their own licensing risk.

pub mod adaptors;
pub mod config;
pub mod import;
pub mod init;
pub mod finetune;
pub mod int8;
pub mod lora;
pub mod model;
pub mod train;
pub mod trunk;
