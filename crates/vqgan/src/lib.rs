// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! VQ encoder / decoder / codebook lookup — the VQGAN core shared by CodeFormer
//! and other VQ-latent models.
//!
//! `image → encoder → nearest-codebook assignment → generator → image`, imported
//! 1:1 from `basicsr/archs/vqgan_arch.py` and parity-gated stage by stage
//! against the step-1 goldens (`testdata/restore/vqgan/`).
//!
//! **Nothing here is a new block or a new kernel.** The convolutional graph is
//! [`vae::blocks::Builder`] — the same conv / GroupNorm / SiLU / residual /
//! nearest-upsample / single-head-attention implementation `AutoencoderKL`
//! runs — selected with [`vae::blocks::BlockNames::vqgan`] for the reference's
//! leaf names (`conv_out` shortcut, `q/k/v/proj_out` attention over `norm`).
//! The codebook search is the existing `vq_argmin` kernel dispatched through
//! [`wm_core::vq::Vq`]; the code lookup is the existing `embed` gather.
//!
//! What this crate owns is the **schedule** ([`config::VqganConfig`], a flat
//! `nn.ModuleList` whose indices the checkpoint names positionally), the
//! two-way-validated [`import`], and the graph wiring in [`model`].
//!
//! Scope: the forward graph ([`model`]) plus the **training** graph ([`train`]) —
//! an SSA forward, a hand-written reverse over `vae::blocks::grad`, and the VQ
//! straight-through estimator, gated by `gradcheck::check_vqgan`. The
//! CodeFormer transformer / controllable feature transformation / fidelity dial
//! are `crates/restore`.
//!
//! The serving contract is met by [`caps`] (the `encode`/`decode`
//! `capability::Provider`), `crates/cli/src/resident_restore.rs` (the residency
//! adapter, `BRAIN_VQGAN_WEIGHTS`) and `examples/restore/` — see
//! `docs/serving-contract.md`.

pub mod caps;
pub mod config;
pub mod import;
pub mod model;
pub mod train;

pub use config::{Block, VqganConfig};
pub use import::Import;
pub use model::{Codebook, Reconstruction, Vqgan, KERNELS};
pub use train::{VqganTrainer, TRAIN_KERNELS, TRAIN_PIPELINES};
