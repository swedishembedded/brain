// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training dataset — re-export of the hoisted loader.
//!
//! The captioned-image folder loader moved to [`data::imageset`] so Z-Image and
//! FLUX.2 share ONE implementation (the workspace one-implementation rule);
//! this module keeps zimage's public API (`zimage::dataset::{Sample,
//! load_dir}`) unchanged. See `data::imageset` for the caption formats and
//! pre-processing contract.

pub use data::imageset::{load_dir, Sample};
