// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan2.1 / Wan2.2 video diffusion transformer.
//!
//! Wan generates video by denoising a **3D latent volume** - `(frames, height,
//! width)` at a (4, 8, 8) VAE stride - with a diffusion transformer, under
//! flow matching. Its shape differs from brain's existing image generators in
//! three ways that decide most of the port:
//!
//! 1. **The attention topology is SDXL's, not MMDiT's.** Every block runs
//!    self-attention over the flattened latent volume and then a *separate*
//!    cross-attention into the text encoding. FLUX.2 and Z-Image instead
//!    concatenate text into one joint sequence. So the piece to reuse is
//!    `sdxlunet`'s cross-attention wiring, not `flux2`'s.
//! 2. **Position encoding is three-axis RoPE.** `head_dim` splits as
//!    `[d - 2*(d/3), d/3, d/3]` across (frame, height, width) - which is
//!    `dit::rope::RopeConfig` with `n_axes = 3`, already generic.
//! 3. **The sequence is long.** 81 frames at 480p is 32,760 tokens after the
//!    (1, 2, 2) patch; at 720p (14B only) it is 75,600. A dense score matrix
//!    is 4.3 GB per head at 480p and 22.9 GB at 720p, so chunked or flash
//!    attention is a correctness prerequisite here, not an optimization.
//!
//! Every number in [`config`] is transcribed from upstream
//! (`Wan-Video/Wan2.1`, `wan/configs/*.py` and `generate.py`) rather than from
//! the HF `config.json`, because the sampling defaults that matter - shift,
//! steps, guidance - live in the CLI's argument defaults, not in any config
//! file the checkpoint ships.

pub mod block;
pub mod caps;
pub mod config;
pub mod dev;
pub mod devgrad;
pub mod finetune;
pub mod grad;
pub mod import;
pub mod lora;
pub mod model;
pub mod modelgrad;
pub mod pipeline;
pub mod rope;
pub mod train;
pub mod vae3d;

pub use block::{AttnMode, WanBlock};
pub use devgrad::BlockDev;
pub use train::DeviceTrainer;
pub use config::{Task, WanConfig};
pub use dev::WanDitDev;
pub use model::WanDit;
pub use pipeline::{generate, generate_hot, GenOpts, HotDit, Paths, Solver, Timings, Video};
pub use vae3d::{WanVaeConfig, WanVaeDecoder, WanVaeEncoder};
