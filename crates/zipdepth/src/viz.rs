// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Re-export of `imaging::viz` - colormaps, robust bounds, the side-by-side
//! compositor and the bitmap-font HUD text used to live here. They moved to
//! `crates/imaging` because none of it is depth-specific (a colormap over
//! `[f32]` is exactly what a SAM 2 mask or any other single-channel model
//! output needs too); this module is now the "was here, still works" seam so
//! `zipdepth::viz::{Colormap, colorize, ...}` keeps resolving unchanged.

pub use imaging::viz::*;
