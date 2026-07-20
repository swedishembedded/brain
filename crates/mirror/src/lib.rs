// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 (HY-World 2.0) multi-view 3D reconstruction in brain.
//!
//! A 1.26B fp32 feed-forward model: per-frame DINOv2 ViT-L/14 patch encoding,
//! a 24-level alternating frame/global attention trunk with normalized 2D
//! RoPE, and DPT-style dense heads (depth / points / normals / Gaussians)
//! plus an iterative camera head. Weights import 1:1 from the reference
//! `model.safetensors`; outputs feed the `splat` rasterizer.
//!
//! Populated phase by phase (P0: config/param_list/import).

pub mod cam;
pub mod config;
pub mod dpt;
pub mod gaussians;
pub mod import;
pub mod model;
pub mod preprocess;
pub mod rope2d;
