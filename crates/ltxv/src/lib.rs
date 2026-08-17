// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5: a two-stream (video + audio) diffusion transformer.
//!
//! **This crate currently implements only the causal 3D video VAE - encoder
//! and conv decoder.** The convolution-free "diffusion decoder"
//! (`DiffusionVideoDecoder`, 3D neighborhood attention), the audio stack, and
//! the DiT itself are not implemented yet - see [`vae3d`]'s module doc for
//! the full architecture write-up and the conventions that differ from
//! `crates/wan`'s causal VAE.

pub mod import;
pub mod patchify;
pub mod vae3d;

pub use vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder};
