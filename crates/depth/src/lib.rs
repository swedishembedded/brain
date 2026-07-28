// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Monocular depth for brain: **ZipDepth** (6.1M pure conv,
//! RepVGG-reparameterizable), with a dense **relative inverse depth** output
//! contract — `[N,1,H,W]` at the input's own resolution. Model, checkpoint
//! import, reference-exact predictor, training loop, demo rendering
//! (colormaps/stereo/effects) and the INT8 calibration report all live here;
//! the NPU export path is `npu::depth_topology`. (Depth Anything 3 was once
//! planned as a second architecture behind the same contract; it is DROPPED.)

pub mod blocks;
pub mod caps;
pub mod config;
pub mod init;
pub mod net;
pub mod fuse;
pub mod import;
pub mod predict;
pub mod quant;
pub mod effects;
pub mod stereo;
pub mod train;
pub mod viz;
pub mod model;

pub use config::{pick_groups, GlobalMode, ZipConfig};
pub use fuse::{fuse_qarep, Branch};
pub use init::init_model as init_weights;
pub use import::{cfg_for_checkpoint, load as load_checkpoint, load_into, tensor_names};
pub use model::ZipDepth;
pub use predict::Predictor;
pub use quant::{collect_activation_stats, ActStatsCollector, LayerReport};
pub use effects::{depth_blur, fog};
pub use stereo::{autostereogram, autostereogram_textured, stereo_pair, StereoOpts};
