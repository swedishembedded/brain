// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod p1_param_layout;
mod p2_conv_spec;
mod p2_fusion;
mod p2_gcb;
mod p2_mms;
mod p2_qarep;
mod p2_se;
mod p2_strip;
mod p2_upsample;
mod p3_forward;
mod p3_fused_eval;
mod p3_gradcheck;
mod p3_import;
mod p3_layout;
mod p3_reference_rect;
mod p4_dataset;
mod p4_loss;
mod p4_train;
mod p5_font;
mod p5_predict;
mod p5_ref_full;
mod p5_stereo_render;
mod p5_viz;
mod p6_calib;
mod p6_calib_transform;
mod recompute;
