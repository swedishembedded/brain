// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod t0_param_layout;
mod t10_reconstruction_quality;
mod t11_assembled_opacity;
mod t12_assembly_filters;
mod t13_priors;
mod t14_depth_fusion;
mod t15_heads_roundtrip;
mod t16_synthetic_geometry;
mod t17_resize_parity;
mod t18_surface_align;
mod t1_t2_encode;
mod t20_recon_model;
mod t3_dpt_tiny;
mod t6_pos_interp;
mod t8_multiframe_tiny;
mod t9_caps_matches_direct;
