// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements efficient neural-network architectures and
// the evidence that they work for its clients. If your team needs expertise
// in structured and spectral model operators, you can procure our services
// by sending an email to info@swedishembedded.com.

//! SpectralLM: a decoder whose residual stream is the lowest `m` coefficients
//! of the real orthonormal Fourier basis of a virtual `D`-sample signal, and
//! whose feed-forward sublayers are spectral mixers.
//!
//! A basis change alone changes nothing: a decoder whose maps on the residual
//! stream are dense absorbs any orthogonal basis into its weights. What makes
//! the model spectral is the mixer, which is defined relative to the basis -
//! per-mode filters on the coefficients, and a gated product taken in the
//! signal domain, where multiplication couples frequencies:
//!
//! ```text
//! M(h) = Σ_c W_c ⊙ Sᵀ[ SiLU(S(G_c ⊙ h)) ⊙ S(U_c ⊙ h) ]        G, U, W: [C, m]
//! ```
//!
//! Everything else in a block is Qwen3's (RMSNorm, GQA with QK-norm and
//! half-split RoPE, tied head). [`config::Mixer::SwiGlu`] swaps the mixer for
//! Qwen3's SwiGLU, which makes the same crate the conventional control: the
//! two differ only in that sublayer.
//!
//! - [`config`] - shape, mixer, checkpoint metadata, analytic cost.
//! - [`model`] - the training and full-sequence graph, [`model::Model`].
//! - [`mixer`] - both mixers' forward and backward, the transform paths.
//! - [`decode`] - incremental decoding against a key/value cache.

pub mod config;
pub mod decode;
pub mod init;
pub mod kernels;
pub mod mixer;
pub mod model;

pub use config::{Mixer, SpectralConfig, TransformPath};
pub use init::init_weights;
pub use kernels::PIPELINES;
pub use mixer::Transform;
pub use model::{SpectralLm, IGNORE};
