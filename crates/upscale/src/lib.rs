// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN (`RRDBNet`) super-resolution — the imaging pipeline's
//! detail/upscale tail.
//!
//! **What this is.** The generator half of Real-ESRGAN: a residual-in-residual
//! dense block trunk that maps `[3,H,W]` to `[3,scale*H,scale*W]`. The
//! discriminator is a training-time component and is not part of the released
//! inference checkpoint, so it is not ported here.
//!
//! **Kernels: none new.** The whole net is conv + LeakyReLU + channel concat +
//! nearest-2x upsample. `conv`/`upsample`/`add` come from
//! [`vae::blocks::Builder`] — the same shared conv-block builder `AutoencoderKL`
//! and the VQGAN family are built from — and only `leaky_relu`, `concat2` and
//! `scale_add` are appended. That is the second time in this workstream a port
//! turned out to need zero kernels (`crates/controlnet` was the first), which is
//! the payoff for the shared builder rather than a coincidence.
//!
//! **Shape is derived, not hardcoded.** Real-ESRGAN ships `x4plus`,
//! `x4plus_anime_6B` and `x2plus` over one architecture, differing only in
//! numbers recoverable from the tensor shapes. [`config::RrdbConfig::from_tensors`]
//! reads them, and a round-trip test pins that `param_list` and `from_tensors`
//! are inverses, so "derived" is checkable rather than a claim in a comment.
//!
//! ```ignore
//! let t = upscale::import::load("RealESRGAN_x4plus.pth")?;
//! let cfg = upscale::config::RrdbConfig::from_tensors(&shapes)?;
//! let net = upscale::model::Rrdb::new(gpu, cfg, &t, h, w, false);
//! let hr = net.run(&chw);            // [3, 4h, 4w], clamped to [0,1]
//! ```

pub mod caps;
pub mod config;
pub mod import;
pub mod model;

pub use config::RrdbConfig;
pub use model::{Rrdb, KERNELS};
