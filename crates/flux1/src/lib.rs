// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1 (dev / Kontext / schnell) in brain.
//!
//! Black Forest Labs' 12 B MMDiT rectified-flow transformer: 19 double-stream
//! img/txt blocks with joint attention, then 38 single-stream parallel blocks,
//! over the 16-channel `AutoencoderKL` latent, conditioned on a T5-XXL sequence
//! plus a CLIP-L pooled vector. `FLUX.1-Kontext-dev` is the same network
//! trained for **instructed editing**: reference images are VAE-encoded and
//! appended as extra tokens whose position ids carry an axis-0 offset, and the
//! prediction is truncated to the noise span.
//!
//! This is a **derivation of [`flux2`](../flux2/index.html)**, not a fresh
//! port: the two share the block skeleton, the joint attention, the multi-axis
//! interleaved RoPE (`dit::rope`), the int8 path (`model::int8`) and the whole
//! kernel set. [`model`] documents each of the four architectural differences
//! and how it is expressed. **No kernel and no shared block was added for
//! FLUX.1.**
//!
//! Canonical tensor names are the BFL reference names (`double_blocks.N.…`,
//! `single_blocks.N.…`); the diffusers `transformer/` layout is remapped onto
//! them at import ([`import::import_diffusers`]) with two-way coverage.
//!
//! Scope today: **forward parity only**. The backward/gradcheck path, the
//! sampling pipeline (schedule + VAE + text encoders) and the serving contract
//! are follow-ups.

pub mod caps;
pub mod config;
pub mod import;
pub mod inject;
pub mod model;
pub mod pipeline;

pub use config::Flux1Config;
pub use import::{import_bfl, import_diffusers, truncate_to_depth, Tensors};
pub use inject::{BlockInject, InjectSite};
pub use model::{position_ids, Flux1Model, Precision, Trace, KERNELS};
