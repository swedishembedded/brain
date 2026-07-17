// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Monocular depth models for brain.
//!
//! One crate, two architectures behind a shared output contract — dense
//! **relative inverse depth**, `[N,1,H,W]` at the input's own resolution — rather
//! than one crate per architecture. ZipDepth (pure conv) and Depth Anything 3
//! (ViT + DPT) have nothing in common internally, but they share the `DepthModel`
//! seam, the loss, the eval metrics, the dataset, the preprocessing, the CLI and
//! the NPU quantization path. That is most of the code; the model internals are
//! one module each.
//!
//! brain has no precedent for a crate per architecture — `crates/npu` already
//! holds five `*_topology.rs` under one roof. `crates/glm` and `crates/qwen` are
//! separate because they are different model FAMILIES (different tokenizers,
//! decode loops, runtimes), not two backbones behind one output.
//!
//! Split `crates/da3` off only if DA3 grows the any-view machinery (CameraEnc/Dec,
//! the depth-ray target); the arch enum is the seam that keeps that split cheap.

pub mod blocks;
pub mod config;
pub mod init;
pub mod net;
pub mod fuse;
pub mod import;
pub mod predict;
pub mod viz;
pub mod model;

pub use config::{pick_groups, GlobalMode, ZipConfig};
pub use fuse::{fuse_qarep, Branch};
pub use init::init_model as init_weights;
pub use import::{load as load_checkpoint, load_into};
pub use model::ZipDepth;
pub use predict::Predictor;
