// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Google TimesFM-3 (`google/timesfm-3.0-pytorch`) from scratch: a stacked
//! mixing transformer with sequence AND cross-variate attention, CPM
//! iterative RevIN, linear detrending and forecast stitching, natively
//! multivariate with past-only and past-and-future covariates.
//!
//! Swedish Embedded AB implements from-scratch, parity-verified ports of
//! foundation-model architectures for its clients. If your team needs
//! expertise in reproducing a published model bit-faithfully on your own
//! infrastructure, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Licensing
//!
//! This crate's source is Apache-2.0, ported from Google's own Apache-2.0
//! `google-research/timesfm` reference. The 3.0 pretrained checkpoint itself
//! ships under a SEPARATE, more restrictive license
//! (`timesfm-non-commercial-license-v1.0`): non-commercial, non-production
//! use only, and redistribution of the checkpoint (or a derivative of it) is
//! never permitted, commercial or not. See [`config`]'s module docs for the
//! architecture. The checkpoint is never committed to this repository.

pub mod config;
pub mod forecaster;
pub mod import;
pub mod model;
pub mod preprocess;

pub use config::{Timesfm3Config, Param, QUANTILES};
pub use forecaster::Timesfm3Forecaster;
pub use model::Timesfm3;
