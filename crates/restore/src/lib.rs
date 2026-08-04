// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer blind face restoration — the parts `crates/vqgan` does not cover.
//!
//! `crates/vqgan` ships the VQ autoencoder CodeFormer is built on (encoder,
//! 1024×256 codebook, generator), forward-parity-gated at cosine 1.000000000
//! with zero code-index disagreements. **None of it is re-implemented here.**
//! What this crate adds is exactly what turns that autoencoder into a
//! degradation-robust face restorer:
//!
//! 1. the **code-prediction Transformer** — nine pre-LN `TransformerSALayer`s
//!    over the flattened encoder output with a learned position embedding,
//!    whose 1024-way head *predicts* the codebook indices instead of looking up
//!    the nearest neighbour. Nearest-neighbour lookup follows the degradation;
//!    a global-attention prediction does not, which is the paper's whole claim;
//! 2. the **controllable feature transformation** (`Fuse_sft_block`) that
//!    injects encoder features back into the generator at four resolutions;
//! 3. the **identity-fidelity dial `w`** that scales that injection —
//!    `w = 0` maximum quality, `w = 1` maximum fidelity to the input.
//!
//! Face detection and 5-point alignment for a full in-the-wild pipeline come
//! from `crates/facenet` (SCRFD + Umeyama similarity alignment, already
//! parity-gated); this crate takes an aligned 512×512 face and gives one back.
//!
//! Scope today: the **forward** ([`model`]), parity-gated per stage against
//! `tools/codeformer_restore_dump_reference.py` goldens at several `w`
//! including both endpoints, plus the **stage-II training graph** ([`train`]) —
//! the code-prediction Transformer under the code-token cross-entropy, with the
//! VQ autoencoder frozen exactly as the reference's stage II freezes it. That
//! backward is gated by `gradcheck::check_codeformer` on both the P40 and
//! `backend-cpu`. Still follow-ups: training the CFT / the dial `w`, and
//! composing this reverse with `vqgan::train`'s so a gradient reaches the
//! encoder end to end.
//!
//! The serving contract is met by [`caps`] (the `restore_face`
//! `capability::Provider`), `crates/cli/src/resident_restore.rs` (the residency
//! adapter, `BRAIN_RESTORE_WEIGHTS`) and `examples/restore/` — see
//! `docs/serving-contract.md`.

pub mod caps;
pub mod config;
pub mod import;
pub mod model;
pub mod train;

pub use config::{CodeFormerConfig, FuseTap, FUSE_TAPS};
pub use import::Import;
pub use model::{CodeFormer, Restoration, KERNELS};
pub use train::{CodeTransformerTrainer, TRAIN_PIPELINES};
