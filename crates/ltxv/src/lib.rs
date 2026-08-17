// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5: a two-stream (video + audio) diffusion transformer.
//!
//! This crate implements the causal 3D video VAE (encoder + conv decoder,
//! see [`vae3d`]'s module doc), the **video-ONLY stream of the DiT**
//! ([`config`]/[`rope`]/[`block`]/[`dit`]): `BasicAVTransformerBlock`'s video
//! path (self-attention, text cross-attention with AdaLN modulation, the
//! FFN) and the full tiny-model forward around it (`patchify_proj`, the
//! per-token adaLN-single timestep conditioning, the output stage), and (as
//! of this milestone, M4) the **end-to-end `t2v` pipeline + serving
//! contract** ([`pipeline`]/[`caps`]): real scheduler math, a real CFG
//! denoise loop, the real VAE decode, and the capability/CLI/residency
//! plumbing every model in this repo ships - composed as a SMOKE TEST (real
//! VAE + tiny random-weight DiT + a stub text context, no real 22B DiT or
//! Gemma-4 text encoder yet, see [`pipeline`]'s module doc for exactly what
//! that means and why). The audio stream, the audio<->video cross-attention,
//! the real text encoder / embeddings-connector, and the convolution-free
//! "diffusion decoder" (`DiffusionVideoDecoder`, 3D neighborhood attention)
//! are NOT implemented - later milestones on this port's own roadmap ledger.
//!
//! Only the tiny-config op sequence is proven for the DiT itself
//! (`config::LtxDitConfig::tiny`, `crates/ltxv/tests/dit_parity.rs`) -
//! real-22B-checkpoint weight import is a separate, later milestone
//! (`dit::load_tiny_weights` is a simple name-keyed loader for the golden's
//! OWN tiny weights only, not a real-checkpoint importer;
//! `dit::random_tiny_weights` is what [`pipeline::generate`] actually uses).

pub mod block;
pub mod caps;
pub mod config;
pub mod dit;
pub mod import;
pub mod patchify;
pub mod pipeline;
pub mod rope;
pub mod vae3d;

pub use config::LtxDitConfig;
pub use dit::{load_tiny_weights, DitTaps, LtxDit};
pub use pipeline::{GenOpts, Paths};
pub use vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder};
