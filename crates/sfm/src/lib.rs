// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion: camera poses, one shared calibration and a sparse
//! point cloud, recovered from nothing but a set of overlapping photographs.

pub mod ba;
pub mod camera;
pub mod incremental;
pub mod linalg;
pub mod matching;
pub mod pnp;
pub mod sift;
pub mod twoview;
